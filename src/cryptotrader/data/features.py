# src/cryptotrader/data/features.py
"""Micro-structure feature engineering for 1-minute intraday models.

The model derives its entire edge from price/volume action (no news). The feature
set below is chosen for tree-based learners (LightGBM/XGBoost), which handle
collinear, differently-scaled, NaN-bearing inputs gracefully — so we deliberately
expose *raw, interpretable* signals rather than a heavily orthogonalised set.

Feature families
----------------
Trend / momentum   : multi-horizon log-momentum, momentum acceleration, RSI.
Mean reversion     : price z-score, VWAP deviation (+ its z-score).
Volatility         : Wilder ATR (abs + %), realized vol, candle range.
Order-flow proxy   : close-location-value, signed-volume imbalance z-score.
Volume regime      : volume z-score and volume ratio (spike detection).
Candle shape       : body %, upper/lower wick %, up/down streak.

Look-ahead safety
-----------------
Every transform uses *backward-looking* rolling windows only, so feature row ``t``
depends solely on bars ``<= t``. The incremental :meth:`update` path re-runs the
exact same vectorised code over a bounded buffer, guaranteeing backtest/live
parity (no risk of a hand-written online formula drifting from the batch one).
"""

from __future__ import annotations

from collections import deque
from typing import Final

import numpy as np
import pandas as pd

from cryptotrader.core.interfaces import FeatureCalculator
from cryptotrader.core.types import Bar

_EPS: Final = 1e-12


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range using Wilder's smoothing (RMA)."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder's RMA == EWM with alpha = 1/period.
    return true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Classic Wilder RSI in ``[0, 100]``."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + _EPS)
    return 100.0 - 100.0 / (1.0 + rs)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - mean) / std over a backward window."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / (std + _EPS)


class MicrostructureFeatureEngine(FeatureCalculator):
    """Computes the intraday micro-structure feature matrix.

    Parameters
    ----------
    atr_period:
        Lookback for Wilder ATR and RSI.
    vwap_window:
        Rolling window (in bars) for the VWAP anchor.
    momentum_windows:
        Horizons (in bars) for multi-scale log-momentum.
    volume_spike_window:
        Window for volume z-score / ratio (order-flow anomaly detection).
    zscore_window:
        Window for price and VWAP-deviation z-scores.
    """

    def __init__(
        self,
        atr_period: int = 14,
        vwap_window: int = 60,
        momentum_windows: list[int] | None = None,
        volume_spike_window: int = 30,
        zscore_window: int = 60,
    ) -> None:
        self.atr_period = atr_period
        self.vwap_window = vwap_window
        self.momentum_windows = sorted(momentum_windows or [3, 5, 15])
        self.volume_spike_window = volume_spike_window
        self.zscore_window = zscore_window

        # Longest window any feature needs -> warmup / buffer size.
        self._warmup = max(
            atr_period,
            vwap_window,
            volume_spike_window,
            zscore_window,
            max(self.momentum_windows),
        ) + 2

        self._names = self._build_feature_names()
        # Live buffer: keep enough history for one correct incremental row.
        # Recursive features (Wilder ATR/RSI) depend on full history; a generous
        # buffer makes the live EWM converge to the batch value (decays ~e^-k).
        self._buffer: deque[Bar] = deque(maxlen=max(self._warmup * 5, 600))
        # Cache of the most recent incremental feature row (live mode), so callers
        # (e.g. the live engine) can read derived values like ATR without paying
        # for a second full transform over the buffer.
        self._last_features: pd.Series | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def feature_names(self) -> list[str]:
        return list(self._names)

    @property
    def last_features(self) -> pd.Series | None:
        """Most recent feature row produced by :meth:`update` (or ``None``)."""
        return self._last_features

    @property
    def warmup(self) -> int:
        """Number of leading bars whose features are (partially) undefined."""
        return self._warmup

    def transform(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Vectorised feature matrix aligned to ``ohlcv.index``.

        Leading rows lacking sufficient history contain NaNs; callers should drop
        them for training and treat NaN rows as "stay flat" in the backtest.
        """
        o, h, l, c, v = (ohlcv[x] for x in ("open", "high", "low", "close", "volume"))
        feats: dict[str, pd.Series] = {}

        log_ret = np.log(c / c.shift(1))
        feats["ret_1"] = log_ret

        # --- Trend / momentum --------------------------------------------------
        for w in self.momentum_windows:
            feats[f"mom_{w}"] = np.log(c / c.shift(w))
        shortest = self.momentum_windows[0]
        feats["mom_accel"] = feats[f"mom_{shortest}"] - feats[f"mom_{shortest}"].shift(shortest)
        feats["rsi"] = _rsi(c, self.atr_period)

        # --- Volatility --------------------------------------------------------
        atr = _wilder_atr(h, l, c, self.atr_period)
        feats["atr"] = atr
        feats["atr_pct"] = atr / (c + _EPS)
        feats["realized_vol"] = log_ret.rolling(
            self.momentum_windows[-1], min_periods=self.momentum_windows[-1]
        ).std(ddof=0)
        feats["range_pct"] = (h - l) / (c + _EPS)

        # --- Mean reversion: VWAP + price z-score ------------------------------
        typical = (h + l + c) / 3.0
        pv = (typical * v).rolling(self.vwap_window, min_periods=self.vwap_window).sum()
        vol_sum = v.rolling(self.vwap_window, min_periods=self.vwap_window).sum()
        vwap = pv / (vol_sum + _EPS)
        vwap_dev = c / (vwap + _EPS) - 1.0
        feats["vwap_dev"] = vwap_dev
        feats["vwap_dev_z"] = _zscore(vwap_dev, self.zscore_window)
        feats["price_z"] = _zscore(c, self.zscore_window)

        # --- Order-flow proxy (no L2 book on 1m candles) -----------------------
        # Close-location-value in [-1, +1]: where in the bar range did we close?
        clv = ((c - l) - (h - c)) / ((h - l) + _EPS)
        feats["clv"] = clv
        # Signed-volume imbalance proxy, normalised over the spike window.
        signed_volume = clv * v
        feats["ofi_z"] = _zscore(signed_volume, self.volume_spike_window)

        # --- Volume regime / spike detection -----------------------------------
        feats["vol_z"] = _zscore(v, self.volume_spike_window)
        vol_ma = v.rolling(self.volume_spike_window, min_periods=self.volume_spike_window).mean()
        feats["vol_ratio"] = v / (vol_ma + _EPS)

        # --- Candle shape ------------------------------------------------------
        rng = (h - l) + _EPS
        feats["body_pct"] = (c - o) / rng
        feats["upper_wick"] = (h - np.maximum(o, c)) / rng
        feats["lower_wick"] = (np.minimum(o, c) - l) / rng
        up = (c > o).astype(float)
        feats["up_streak"] = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
        feats["up_streak"] = feats["up_streak"] * np.where(c >= o, 1.0, -1.0)

        frame = pd.DataFrame(feats, index=ohlcv.index)
        return frame[self._names]

    def update(self, bar: Bar) -> pd.Series | None:
        """Incremental path for live trading.

        Appends ``bar`` to a bounded buffer, recomputes features over that buffer
        via :meth:`transform`, and returns the latest feature row, or ``None`` if
        not enough history has accumulated yet.
        """
        self._buffer.append(bar)
        if len(self._buffer) < self._warmup:
            return None
        df = pd.DataFrame(
            [
                {
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                }
                for b in self._buffer
            ],
            index=pd.DatetimeIndex([b.timestamp for b in self._buffer]),
        )
        row = self.transform(df).iloc[-1]
        self._last_features = row
        return row if not row.isna().any() else None

    def reset(self) -> None:
        """Clear the live buffer (e.g. on reconnect)."""
        self._buffer.clear()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_feature_names(self) -> list[str]:
        names = ["ret_1"]
        names += [f"mom_{w}" for w in self.momentum_windows]
        names += [
            "mom_accel",
            "rsi",
            "atr",
            "atr_pct",
            "realized_vol",
            "range_pct",
            "vwap_dev",
            "vwap_dev_z",
            "price_z",
            "clv",
            "ofi_z",
            "vol_z",
            "vol_ratio",
            "body_pct",
            "upper_wick",
            "lower_wick",
            "up_streak",
        ]
        return names
