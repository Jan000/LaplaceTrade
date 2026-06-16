# src/cryptotrader/integrations/mt5/store.py
"""Persist the bars an MT5 Expert Advisor sends, so they become a training set.

MT5 broker instruments have no ccxt history, so the only way to train a model on
them is to record what the EA streams. Each (internal symbol, timeframe) pair is
appended to a parquet file under ``data/mt5/``; rows are de-duplicated by timestamp
and capped at ``mt5.capture_max_rows``. ``scripts/train_model.py --mt5`` reads it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cryptotrader.config import Settings
from cryptotrader.ml.registry import safe_symbol

logger = logging.getLogger(__name__)


def mt5_dir(settings: Settings) -> Path:
    """Directory holding captured MT5 OHLCV (next to the SQLite DB, on the data volume)."""
    return Path(settings.persistence.db_path).parent / "mt5"


def capture_path(settings: Settings, symbol: str, timeframe: str) -> Path:
    return mt5_dir(settings) / f"{safe_symbol(symbol)}_{timeframe}.parquet"


def append_bars(settings: Settings, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
    """Merge ``df`` (UTC-indexed OHLCV) into the symbol/timeframe parquet. Returns total rows.

    Best-effort: never raises into the request path — a capture failure must not block a
    trading decision."""
    if df is None or df.empty:
        return 0
    path = capture_path(settings, symbol, timeframe)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            merged = pd.concat([existing, df[["open", "high", "low", "close", "volume"]]])
        else:
            merged = df[["open", "high", "low", "close", "volume"]].copy()
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        cap = int(settings.mt5.capture_max_rows or 0)
        if cap > 0 and len(merged) > cap:
            merged = merged.iloc[-cap:]
        merged.to_parquet(path)
        return len(merged)
    except Exception:  # pragma: no cover - defensive
        logger.warning("MT5 bar capture failed for %s %s", symbol, timeframe, exc_info=True)
        return 0


def load_ohlcv(settings: Settings, symbol: str, timeframe: str) -> pd.DataFrame:
    """Load captured OHLCV for a symbol/timeframe (empty frame if none recorded yet)."""
    path = capture_path(settings, symbol, timeframe)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path).sort_index()
    except Exception:  # pragma: no cover
        logger.warning("Could not read captured MT5 data at %s", path, exc_info=True)
        return pd.DataFrame()


def capture_counts(settings: Settings) -> dict[str, int]:
    """Row count per captured parquet file (for the /api/mt5/status dashboard)."""
    d = mt5_dir(settings)
    out: dict[str, int] = {}
    if not d.exists():
        return out
    for p in sorted(d.glob("*.parquet")):
        try:
            out[p.stem] = len(pd.read_parquet(p, columns=["close"]))
        except Exception:  # pragma: no cover
            out[p.stem] = -1
    return out
