# src/cryptotrader/integrations/mt5/bridge.py
"""Stateless trading-decision service for the MetaTrader 5 bridge.

Given the bars + current position an MT5 Expert Advisor sends, this returns the
SAME decision the live engine would make for that symbol — by reusing the exact
feature engine, model, strategy thresholds/filters and ATR risk gate. It is
deliberately stateless: the EA reports its own position each call, and the broker
holds the native stop-loss / take-profit, so no per-symbol portfolio is kept here.

What is reused (so "what you backtest is what MT5 trades" stays true):

* :class:`MicrostructureFeatureEngine` — identical features (recorded crypto
  micro-structure is force-disabled; MT5 brokers have no L2 order book).
* the per-symbol model via :func:`resolve_model` / :func:`load_predictor`.
* :class:`MLStrategy` thresholds + trend/vol regime filters.
* :class:`ATRRiskManager` — the EV/cost entry gate and the ATR stop/TP distances.

The EA does the broker-specific part (lot sizing from ``risk_fraction``, order
placement, native SL/TP), because only it knows the instrument's contract specs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from cryptotrader.config import RunMode, Settings
from cryptotrader.core.events import MarketEvent
from cryptotrader.core.types import Bar, Side
from cryptotrader.data.features import MicrostructureFeatureEngine
from cryptotrader.ml.model import MomentumBaselinePredictor
from cryptotrader.risk.manager import ATRRiskManager
from cryptotrader.strategy.ml_strategy import MLStrategy

logger = logging.getLogger(__name__)

_QUOTES = ("USDT", "USDC", "USD", "EUR", "BUSD")
_MIN_BARS = 30


# --------------------------------------------------------------------------- #
# Symbol mapping
# --------------------------------------------------------------------------- #
def _heuristic_crypto(mt5_symbol: str) -> str | None:
    """Best-effort MT5 -> internal symbol for obvious crypto pairs (BTCUSDT -> BTC/USDT).

    A bare fiat quote (BTCUSD) is normalised to the USDT model space, since models are
    trained on …/USDT. Returns None when nothing sensible can be inferred — non-crypto
    instruments must be mapped explicitly in ``mt5.symbol_map``.
    """
    s = "".join(ch for ch in mt5_symbol.upper() if ch.isalnum())
    for q in _QUOTES:
        if s.endswith(q) and len(s) > len(q):
            base = s[: -len(q)]
            quote = "USDT" if q in ("USD", "BUSD") else q
            return f"{base}/{quote}"
    return None


def map_symbol(settings: Settings, mt5_symbol: str) -> str | None:
    """Map a broker symbol to the internal model symbol.

    Explicit ``mt5.symbol_map`` (case-insensitive) wins; otherwise a crypto heuristic
    is tried. Returns None if unmappable (the decision is then a no-op)."""
    raw = (mt5_symbol or "").strip()
    if not raw:
        return None
    mapping = settings.mt5.symbol_map or {}
    if raw in mapping:
        return mapping[raw]
    low = raw.lower()
    for k, v in mapping.items():
        if k.lower() == low:
            return v
    return _heuristic_crypto(raw)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_BAR_KEYS = {
    "t": ("t", "time", "timestamp", "ts"),
    "open": ("o", "open"),
    "high": ("h", "high"),
    "low": ("l", "low"),
    "close": ("c", "close"),
    "volume": ("v", "volume", "vol", "tick_volume"),
}


def _first(d: dict, names: tuple[str, ...], default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def bars_to_df(bars: list[dict]) -> pd.DataFrame:
    """Build a UTC-indexed OHLCV frame from the EA's bar list (flexible key names)."""
    rows = []
    for b in bars:
        t = _first(b, _BAR_KEYS["t"])
        if t is None:
            continue
        rows.append({
            "timestamp": t,
            "open": float(_first(b, _BAR_KEYS["open"], 0.0)),
            "high": float(_first(b, _BAR_KEYS["high"], 0.0)),
            "low": float(_first(b, _BAR_KEYS["low"], 0.0)),
            "close": float(_first(b, _BAR_KEYS["close"], 0.0)),
            "volume": float(_first(b, _BAR_KEYS["volume"], 0.0)),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Accept ISO strings or epoch seconds/milliseconds.
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        unit = "ms" if float(ts.iloc[-1]) > 1e11 else "s"
        df["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(ts, utc=True)
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    return df[~df.index.duplicated(keep="last")]


def _pos_side(position: dict | None) -> Side:
    if not position:
        return Side.FLAT
    raw = str(position.get("side", "")).lower()
    vol = float(position.get("volume", position.get("quantity", 0)) or 0)
    if raw in ("long", "buy", "1") or (raw == "" and vol > 0):
        return Side.LONG
    if raw in ("short", "sell", "-1") or (raw == "" and vol < 0):
        return Side.SHORT
    return Side.FLAT


def _load_predictor(settings: Settings, model_symbol: str):
    """Resolve the per-symbol predictor. Honours ``mt5.require_model``.

    Returns ``(predictor_or_None, info)``. ``predictor`` is None when a matching model
    is required but missing — the caller then returns a no-trade decision."""
    from cryptotrader.ml.registry import resolve_model

    path, meta = resolve_model(settings)
    tf = settings.exchange.timeframe
    matches = bool(path is not None and meta
                   and meta.get("symbol") == model_symbol and meta.get("timeframe") == tf)
    info = {"exists": path is not None, "matches": matches,
            "model_symbol": (meta or {}).get("symbol") if meta else None,
            "model_timeframe": (meta or {}).get("timeframe") if meta else None}
    if settings.mt5.require_model and not matches:
        info["reason"] = "no matching model (require_model)"
        return None, info
    if path is not None:
        from cryptotrader.ml.meta import load_predictor

        return load_predictor(path), info
    info["baseline"] = True
    return MomentumBaselinePredictor(), info


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #
def decide(
    settings: Settings,
    *,
    symbol: str,
    timeframe: str | None = None,
    bars: list[dict] | None = None,
    position: dict | None = None,
    account: dict | None = None,
) -> dict:
    """Return a trading decision for one MT5 symbol.

    Actions: ``open_long`` / ``open_short`` (flat + gated signal), ``close``
    (opposite signal or time-exit), ``hold`` (keep the current position), ``none``
    (stay flat). The EA reconciles its broker position to match.
    """
    now = datetime.now(tz=timezone.utc).isoformat()
    model_symbol = map_symbol(settings, symbol)
    base = {"symbol": symbol, "model_symbol": model_symbol, "action": "none",
            "direction": "flat", "confidence": 0.0, "reason": "", "ts": now,
            "poll_seconds": settings.mt5.poll_seconds}
    if model_symbol is None:
        return {**base, "reason": "unmapped_symbol"}
    bars = bars or []
    if len(bars) < _MIN_BARS:
        return {**base, "reason": "insufficient_bars"}

    sub = settings.model_copy(deep=True)
    sub.exchange.symbol = model_symbol
    sub.exchange.timeframe = timeframe or settings.mt5.default_timeframe or sub.exchange.timeframe
    sub.features.use_recorded = False        # recorder obs are crypto-exchange specific
    sub.mode = RunMode.LIVE
    base["timeframe"] = sub.exchange.timeframe

    df = bars_to_df(bars)
    if df.empty:
        return {**base, "reason": "no_valid_bars"}

    fe = MicrostructureFeatureEngine(**sub.features.model_dump())
    feats = fe.transform(df).dropna()
    if feats.empty:
        return {**base, "reason": "insufficient_bars",
                "need_bars": int(fe.warmup + sub.barriers.horizon + 5)}

    row = feats.iloc[-1]
    atr = float(row.get("atr", 0.0))
    trend = float(row.get("trend_sig", 0.0))
    vol_pct = float(row.get("vol_pct", 0.5))
    last = df.iloc[-1]
    last_close = float(last["close"])
    bar = Bar(df.index[-1].to_pydatetime(), float(last["open"]), float(last["high"]),
              float(last["low"]), last_close, float(last["volume"]))

    predictor, model_info = _load_predictor(sub, model_symbol)
    out = {**base, "model": model_info, "entry_ref": round(last_close, 8),
           "atr": round(atr, 8), "bar_time": df.index[-1].isoformat(),
           "risk_fraction": float(sub.risk.risk_per_trade),
           "max_hold_bars": int(sub.barriers.horizon or 0),
           "stop_distance": round(sub.barriers.sl_mult * atr, 8),
           "tp_distance": round(sub.barriers.tp_mult * atr, 8)}
    if predictor is None:
        return {**out, "reason": "no_model"}

    pred = predictor.predict(row)
    out["primary_direction"] = pred.direction.name.lower()
    out["confidence"] = round(float(pred.confidence), 4)

    strat = MLStrategy(predictor, sub.strategy, model_symbol)
    signal = strat._to_signal(MarketEvent(bar), pred, trend, vol_pct)

    cur = _pos_side(position)
    bars_held = int((position or {}).get("bars_held") or 0)
    horizon = int(sub.barriers.horizon or 0)

    # Time-exit (vertical barrier): close a managed position older than the horizon.
    # The broker's native stop/TP handle the price barriers, so we only add the clock.
    if cur is not Side.FLAT and horizon and bars_held >= horizon:
        return {**out, "action": "close", "direction": "flat", "reason": "time_exit"}

    if signal is None:
        # No fresh entry signal: keep any open position (its SL/TP/time manage it).
        return {**out, "action": "hold" if cur is not Side.FLAT else "none",
                "direction": cur.name.lower() if cur is not Side.FLAT else "flat",
                "reason": "no_signal"}

    side = signal.side
    if cur is side:
        return {**out, "action": "hold", "direction": side.name.lower(), "reason": "in_position"}
    if cur is not Side.FLAT:
        return {**out, "action": "close", "direction": "flat", "reason": "reverse_signal",
                "next_direction": side.name.lower()}

    # Flat + signal -> run the SAME EV/cost gate + ATR sizing as the live engine.
    risk = ATRRiskManager(sub.risk, sub.barriers, sub.execution)
    equity = float((account or {}).get("equity") or sub.risk.account_equity)
    order = risk.size_order(signal, bar, atr, equity, has_open_position=False)
    if order is None:
        return {**out, "action": "none", "direction": "flat", "reason": "ev_gate_rejected"}

    dir_sign = 1.0 if side is Side.LONG else -1.0
    return {**out,
            "action": "open_long" if side is Side.LONG else "open_short",
            "direction": side.name.lower(),
            "confidence": round(float(signal.confidence), 4),
            "stop_distance": round(order.stop_distance, 8),
            "tp_distance": round(order.tp_distance, 8),
            "stop_loss": round(last_close - dir_sign * order.stop_distance, 8),
            "take_profit": round(last_close + dir_sign * order.tp_distance, 8),
            "reason": "enter"}
