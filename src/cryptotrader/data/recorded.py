# src/cryptotrader/data/recorded.py
"""Use the live recorder's `observations` as model features.

The recorder collects microstructure signals (order-book imbalance, microprice, depth,
taker flow, spread) that free history can't provide. This module loads them from the DB
and merges them onto the OHLCV bars LEAK-FREE: each bar gets the LAST observation sampled
within its interval (known at the bar's close, which is when the model would act), so no
future information leaks. Bars with no observation (all of history before recording began)
get NaN -> the feature engine zero-fills them.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# source column -> (typed db column | metrics-json key)
_FIELDS = {
    "obs_imbalance": ("ob_imbalance", None),
    "obs_imb5": (None, "imbalance_top5"),
    "obs_micro": (None, "microprice_dev_bps"),
    "obs_depthimb": (None, "depth_imbalance_1pct"),
    "obs_taker": (None, "taker_buy_ratio"),
    "obs_spread": ("spread_bps", None),
}
OBS_COLUMNS = tuple(_FIELDS)


def load_recorded(db_path, symbol: str) -> pd.DataFrame:
    """All recorded observations for ``symbol`` as a timestamp-indexed obs_* DataFrame."""
    p = Path(db_path)
    if not p.exists():
        return pd.DataFrame()
    try:
        con = sqlite3.connect(str(p))
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT timestamp, ob_imbalance, spread_bps, metrics FROM observations "
            "WHERE symbol = ? ORDER BY timestamp", (symbol,)).fetchall()
        con.close()
    except Exception:
        logger.warning("could not read observations for %s", symbol, exc_info=True)
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    recs = []
    for r in rows:
        m = {}
        if r["metrics"]:
            try:
                m = json.loads(r["metrics"])
            except Exception:
                m = {}
        rec = {"timestamp": r["timestamp"]}
        for col, (typed, mkey) in _FIELDS.items():
            v = r[typed] if typed else m.get(mkey)
            rec[col] = float(v) if isinstance(v, (int, float)) else None
        recs.append(rec)
    df = pd.DataFrame(recs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("timestamp").sort_index()


def merge_recorded(ohlcv: pd.DataFrame, db_path, symbol: str, timeframe: str) -> pd.DataFrame:
    """Attach obs_* columns to ``ohlcv`` (last observation per bar; leak-free). No-op if
    there's no recorded data yet — the columns are simply absent and zero-filled later."""
    rec = load_recorded(db_path, symbol)
    if rec.empty or not isinstance(ohlcv.index, pd.DatetimeIndex):
        return ohlcv
    try:
        # last obs within each bar interval [open, open+tf), labelled by the bar's open
        binned = rec.resample(timeframe, label="left", closed="left").last()
        aligned = binned.reindex(ohlcv.index, method="ffill")
        out = ohlcv.copy()
        for c in OBS_COLUMNS:
            if c in aligned:
                out[c] = aligned[c]
        return out
    except Exception:
        logger.warning("recorded merge failed for %s", symbol, exc_info=True)
        return ohlcv


def latest_obs_dict(rows: list[dict]) -> dict:
    """Build an obs_* dict from the newest observation row (for live inference)."""
    if not rows:
        return {}
    r = rows[0]
    m = {}
    if r.get("metrics"):
        try:
            m = json.loads(r["metrics"])
        except Exception:
            m = {}
    out = {}
    for col, (typed, mkey) in _FIELDS.items():
        v = r.get(typed) if typed else m.get(mkey)
        if isinstance(v, (int, float)):
            out[col] = float(v)
    return out
