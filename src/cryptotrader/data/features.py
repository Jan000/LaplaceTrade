# src/cryptotrader/data/features.py
"""Micro-structure + technical feature engineering for intraday ML models.

Every indicator window is a constructor argument (fed from ``FeatureConfig``), so
the whole feature set is tunable from config/env without touching code. The set
is grounded in the ML-trading literature for tree learners:

* M. Lopez de Prado, "Advances in Financial Machine Learning" (Wiley, 2018),
  Ch. 5 — fractionally differentiated features (stationarity with memory).
* GBDT feature-importance studies on BTC favour RSI(14/30), MACD, multi-horizon
  momentum and Stochastic %K/%D (e.g. arXiv:2410.06935, arXiv:2501.07580).

Look-ahead safety: every transform uses backward-looking windows only, and the
incremental :meth:`update` re-runs the same vectorised code over a bounded
buffer, guaranteeing backtest/live parity.
"""

from __future__ import annotations

from collections import deque
from typing import Final

import numpy as np
import pandas as pd

from cryptotrader.core.interfaces import FeatureCalculator
from cryptotrader.core.types import Bar

_EPS: Final = 1e-12


def _rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    return _rma(_true_range(high, low, close), period)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    rs = _rma(gain, period) / (_rma(loss, period) + _EPS)
    return 100.0 - 100.0 / (1.0 + rs)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / (std + _EPS)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0.0), 0.0)
    minus_dm = down.where((down > up) & (down > 0.0), 0.0)
    atr = _rma(_true_range(high, low, close), period)
    plus_di = 100.0 * _rma(plus_dm, period) / (atr + _EPS)
    minus_di = 100.0 * _rma(minus_dm, period) / (atr + _EPS)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + _EPS)
    adx = _rma(dx, period)
    di_diff = (plus_di - minus_di) / (plus_di + minus_di + _EPS)
    return adx, di_diff


def _frac_weights(d: float, size: int) -> np.ndarray:
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)


def _fracdiff(logp: pd.Series, d: float, window: int) -> pd.Series:
    """Fixed-width fractional differentiation (Lopez de Prado, ch. 5), vectorised."""
    w = _frac_weights(d, window)
    arr = logp.to_numpy(dtype=float)
    n = arr.shape[0]
    conv = np.convolve(arr, w)[:n]
    conv[: window - 1] = np.nan
    return pd.Series(conv, index=logp.index)


class MicrostructureFeatureEngine(FeatureCalculator):
    """Computes the intraday feature matrix; every window is configurable."""

    def __init__(
        self,
        atr_period: int = 14,
        vwap_window: int = 60,
        momentum_windows: list[int] | None = None,
        extra_momentum: int = 30,
        volume_spike_window: int = 30,
        zscore_window: int = 60,
        rsi_fast: int = 7,
        rsi_slow: int = 30,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        stoch_k: int = 14,
        stoch_d: int = 3,
        bollinger_window: int = 20,
        bollinger_std: float = 2.0,
        adx_period: int = 14,
        donchian: int = 20,
        parkinson: int = 20,
        obv_z: int = 30,
        amihud: int = 20,
        fracdiff_d: float = 0.4,
        fracdiff_window: int = 60,
        use_taker_flow: bool = False,
        use_funding: bool = False,
        use_open_interest: bool = False,
        use_cross_asset: bool = False,
        cross_symbol: str = "ETH/USDT",
        cross_corr_window: int = 60,
    ) -> None:
        self.atr_period = atr_period
        self.vwap_window = vwap_window
        self.momentum_windows = sorted(momentum_windows or [3, 5, 15])
        self.extra_momentum = extra_momentum
        self.volume_spike_window = volume_spike_window
        self.zscore_window = zscore_window
        self.rsi_fast = rsi_fast
        self.rsi_slow = rsi_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.bollinger_window = bollinger_window
        self.bollinger_std = bollinger_std
        self.adx_period = adx_period
        self.donchian = donchian
        self.parkinson = parkinson
        self.obv_z = obv_z
        self.amihud = amihud
        self.fracdiff_d = fracdiff_d
        self.fracdiff_window = fracdiff_window
        self.use_taker_flow = use_taker_flow
        self.use_funding = use_funding
        self.use_open_interest = use_open_interest
        self.use_cross_asset = use_cross_asset
        self.cross_symbol = cross_symbol
        self.cross_corr_window = cross_corr_window

        self._warmup = max(
            atr_period, vwap_window, volume_spike_window, zscore_window,
            max(self.momentum_windows), extra_momentum, macd_slow, rsi_slow,
            bollinger_window, adx_period, donchian, parkinson, obv_z, amihud,
            fracdiff_window, cross_corr_window,
        ) + 2

        self._names = self._build_feature_names()
        self._buffer: deque[Bar] = deque(maxlen=max(self._warmup * 5, 700))
        self._last_features: pd.Series | None = None

    @property
    def feature_names(self) -> list[str]:
        return list(self._names)

    @property
    def last_features(self) -> pd.Series | None:
        return self._last_features

    @property
    def warmup(self) -> int:
        return self._warmup

    def transform(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        o, h, l, c, v = (ohlcv[x] for x in ("open", "high", "low", "close", "volume"))
        feats: dict[str, pd.Series] = {}

        log_ret = np.log(c / c.shift(1))
        feats["ret_1"] = log_ret

        for w in self.momentum_windows:
            feats[f"mom_{w}"] = np.log(c / c.shift(w))
        feats[f"mom_{self.extra_momentum}"] = np.log(c / c.shift(self.extra_momentum))
        shortest = self.momentum_windows[0]
        feats["mom_accel"] = feats[f"mom_{shortest}"] - feats[f"mom_{shortest}"].shift(shortest)
        feats["rsi"] = _rsi(c, self.atr_period)
        feats["rsi_fast"] = _rsi(c, self.rsi_fast)
        feats["rsi_slow"] = _rsi(c, self.rsi_slow)

        ema_fast = c.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = c.ewm(span=self.macd_slow, adjust=False).mean()
        macd = (ema_fast - ema_slow) / (c + _EPS)
        macd_signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        feats["macd"] = macd
        feats["macd_signal"] = macd_signal
        feats["macd_hist"] = macd - macd_signal

        adx, di_diff = _adx(h, l, c, self.adx_period)
        feats["adx"] = adx
        feats["di_diff"] = di_diff

        atr = _wilder_atr(h, l, c, self.atr_period)
        feats["atr"] = atr
        feats["atr_pct"] = atr / (c + _EPS)
        longest = self.momentum_windows[-1]
        feats["realized_vol"] = log_ret.rolling(longest, min_periods=longest).std(ddof=0)
        feats["range_pct"] = (h - l) / (c + _EPS)
        hl = np.log((h + _EPS) / (l + _EPS)) ** 2
        feats["parkinson_vol"] = np.sqrt(
            hl.rolling(self.parkinson, min_periods=self.parkinson).mean() / (4.0 * np.log(2.0))
        )

        typical = (h + l + c) / 3.0
        pv = (typical * v).rolling(self.vwap_window, min_periods=self.vwap_window).sum()
        vol_sum = v.rolling(self.vwap_window, min_periods=self.vwap_window).sum()
        vwap = pv / (vol_sum + _EPS)
        vwap_dev = c / (vwap + _EPS) - 1.0
        feats["vwap_dev"] = vwap_dev
        feats["vwap_dev_z"] = _zscore(vwap_dev, self.zscore_window)
        feats["price_z"] = _zscore(c, self.zscore_window)

        bmean = c.rolling(self.bollinger_window, min_periods=self.bollinger_window).mean()
        bstd = c.rolling(self.bollinger_window, min_periods=self.bollinger_window).std(ddof=0)
        upper = bmean + self.bollinger_std * bstd
        lower = bmean - self.bollinger_std * bstd
        feats["boll_pctb"] = (c - lower) / ((upper - lower) + _EPS)
        feats["boll_bw"] = (upper - lower) / (bmean + _EPS)

        ll = l.rolling(self.stoch_k, min_periods=self.stoch_k).min()
        hh = h.rolling(self.stoch_k, min_periods=self.stoch_k).max()
        stoch_k = 100.0 * (c - ll) / ((hh - ll) + _EPS)
        feats["stoch_k"] = stoch_k
        feats["stoch_d"] = stoch_k.rolling(self.stoch_d, min_periods=self.stoch_d).mean()

        dlow = l.rolling(self.donchian, min_periods=self.donchian).min()
        dhigh = h.rolling(self.donchian, min_periods=self.donchian).max()
        feats["donchian_pos"] = (c - dlow) / ((dhigh - dlow) + _EPS)

        feats["fracdiff"] = _fracdiff(np.log(c + _EPS), self.fracdiff_d, self.fracdiff_window)

        clv = ((c - l) - (h - c)) / ((h - l) + _EPS)
        feats["clv"] = clv
        feats["ofi_z"] = _zscore(clv * v, self.volume_spike_window)
        obv = (np.sign(c.diff().fillna(0.0)) * v).cumsum()
        feats["obv_z"] = _zscore(obv.diff(), self.obv_z)
        amihud = log_ret.abs() / ((v * c) + _EPS)
        feats["amihud"] = amihud.rolling(self.amihud, min_periods=self.amihud).mean() * 1e6

        feats["vol_z"] = _zscore(v, self.volume_spike_window)
        vol_ma = v.rolling(self.volume_spike_window, min_periods=self.volume_spike_window).mean()
        feats["vol_ratio"] = v / (vol_ma + _EPS)

        rng = (h - l) + _EPS
        feats["body_pct"] = (c - o) / rng
        feats["upper_wick"] = (h - np.maximum(o, c)) / rng
        feats["lower_wick"] = (np.minimum(o, c) - l) / rng
        up = (c > o).astype(float)
        streak = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
        feats["up_streak"] = streak * np.where(c >= o, 1.0, -1.0)

        idx = ohlcv.index
        if isinstance(idx, pd.DatetimeIndex):
            hour = idx.hour + idx.minute / 60.0
            feats["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24.0), index=idx)
            feats["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24.0), index=idx)
        else:
            feats["hour_sin"] = pd.Series(0.0, index=idx)
            feats["hour_cos"] = pd.Series(0.0, index=idx)

        # --- Optional extended data sources (computed only when enabled; the
        # source columns are zero-filled if absent so feature_names stay stable).
        cols = ohlcv.columns
        zero = pd.Series(0.0, index=ohlcv.index)
        if self.use_taker_flow:
            if "taker_buy_base" in cols:
                tb = ohlcv["taker_buy_base"]
                ratio = tb / (v + _EPS)
                feats["taker_buy_ratio"] = ratio
                feats["taker_flow_z"] = _zscore(2.0 * ratio - 1.0, self.volume_spike_window)
            else:
                feats["taker_buy_ratio"] = zero
                feats["taker_flow_z"] = zero
            if "num_trades" in cols:
                nt = ohlcv["num_trades"]
                feats["trade_intensity_z"] = _zscore(nt, self.volume_spike_window)
                feats["avg_trade_size_z"] = _zscore(v / (nt + _EPS), self.volume_spike_window)
            else:
                feats["trade_intensity_z"] = zero
                feats["avg_trade_size_z"] = zero
        if self.use_funding:
            fr = ohlcv["funding_rate"] if "funding_rate" in cols else zero
            feats["funding_rate"] = fr
            feats["funding_z"] = _zscore(fr, self.zscore_window) if "funding_rate" in cols else zero
        if self.use_open_interest:
            if "open_interest" in cols:
                oi = ohlcv["open_interest"]
                oic = np.log((oi + _EPS) / (oi.shift(1) + _EPS))
                feats["oi_change"] = oic
                feats["oi_z"] = _zscore(oic, self.zscore_window)
            else:
                feats["oi_change"] = zero
                feats["oi_z"] = zero
        if self.use_cross_asset:
            if "cross_close" in cols:
                cc = ohlcv["cross_close"]
                cr = np.log(cc / cc.shift(1))
                feats["cross_ret"] = cr
                feats["cross_corr"] = log_ret.rolling(
                    self.cross_corr_window, min_periods=self.cross_corr_window
                ).corr(cr)
                longest_w = self.momentum_windows[-1]
                feats["rel_strength"] = feats[f"mom_{longest_w}"] - np.log(cc / cc.shift(longest_w))
            else:
                feats["cross_ret"] = zero
                feats["cross_corr"] = zero
                feats["rel_strength"] = zero

        for _ext in ("taker_buy_ratio", "taker_flow_z", "trade_intensity_z",
                     "avg_trade_size_z", "funding_rate", "funding_z", "oi_change",
                     "oi_z", "cross_ret", "cross_corr", "rel_strength"):
            if _ext in feats:
                feats[_ext] = feats[_ext].fillna(0.0)

        frame = pd.DataFrame(feats, index=ohlcv.index)
        # atr is kept as a trailing helper column (labels + ATR sizing); it is
        # NOT a model feature (raw atr is price-scaled / non-stationary).
        return frame[self._names + ["atr"]]

    def update(self, bar: Bar) -> pd.Series | None:
        self._buffer.append(bar)
        if len(self._buffer) < self._warmup:
            return None
        df = pd.DataFrame(
            [{"open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
             for b in self._buffer],
            index=pd.DatetimeIndex([b.timestamp for b in self._buffer]),
        )
        row = self.transform(df).iloc[-1]
        self._last_features = row
        return row if not row.isna().any() else None

    def reset(self) -> None:
        self._buffer.clear()
        self._last_features = None

    def _build_feature_names(self) -> list[str]:
        names = ["ret_1"]
        names += [f"mom_{w}" for w in self.momentum_windows]
        names += [
            f"mom_{self.extra_momentum}", "mom_accel", "rsi", "rsi_fast", "rsi_slow",
            "macd", "macd_signal", "macd_hist", "adx", "di_diff",
            "atr_pct", "realized_vol", "range_pct", "parkinson_vol",
            "vwap_dev", "vwap_dev_z", "price_z", "boll_pctb", "boll_bw",
            "stoch_k", "stoch_d", "donchian_pos", "fracdiff",
            "clv", "ofi_z", "obv_z", "amihud", "vol_z", "vol_ratio",
            "body_pct", "upper_wick", "lower_wick", "up_streak",
            "hour_sin", "hour_cos",
        ]
        if self.use_taker_flow:
            names += ["taker_buy_ratio", "taker_flow_z", "trade_intensity_z", "avg_trade_size_z"]
        if self.use_funding:
            names += ["funding_rate", "funding_z"]
        if self.use_open_interest:
            names += ["oi_change", "oi_z"]
        if self.use_cross_asset:
            names += ["cross_ret", "cross_corr", "rel_strength"]
        return names
