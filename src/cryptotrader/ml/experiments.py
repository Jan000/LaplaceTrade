# src/cryptotrader/ml/experiments.py
"""Append-only experiment log: which settings produced which result.

Every training / walk-forward / holdout run appends one JSON line capturing a curated
snapshot of the settings that affect the outcome plus the run's metrics. This makes the
(window-sensitive) tuning reproducible and auditable — you can see exactly which knob
moved which number, instead of relying on memory. Stored in the git-ignored ``models/``
build dir, so it persists across runs without polluting the repo.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from cryptotrader.ml import registry

logger = logging.getLogger(__name__)


def _experiments_file() -> Path:
    return registry.MODELS_DIR / "experiments.jsonl"


def settings_snapshot(settings) -> dict:
    """Curated, flat snapshot of the settings that actually change results."""
    f, m, b, s, r, d = (settings.features, settings.model, settings.barriers,
                        settings.strategy, settings.risk, settings.data)
    return {
        "timeframe": settings.exchange.timeframe,
        "history_days": d.history_days,
        "train_symbols": list(d.train_symbols),
        # feature modules
        "use_funding": f.use_funding, "use_breadth": f.use_breadth,
        "use_cross_asset": f.use_cross_asset, "use_htf": f.use_htf,
        "use_taker_flow": f.use_taker_flow, "use_open_interest": f.use_open_interest,
        "use_fear_greed": f.use_fear_greed,
        # model
        "ensemble_size": m.ensemble_size, "n_estimators": m.n_estimators,
        "learning_rate": m.learning_rate, "num_leaves": m.num_leaves,
        "min_child_samples": m.min_child_samples, "reg_lambda": m.reg_lambda,
        "use_calibration": m.use_calibration, "use_meta_labeling": m.use_meta_labeling,
        "drop_features": list(m.drop_features),
        # labels / barriers
        "label_method": b.label_method, "tp_mult": b.tp_mult, "sl_mult": b.sl_mult,
        "horizon": b.horizon,
        # decision layer
        "long_threshold": s.long_threshold, "trend_filter": s.trend_filter,
        "vol_gate": s.vol_gate, "use_ev_filter": r.use_ev_filter,
        "min_expected_value": r.min_expected_value,
    }


def log_experiment(kind: str, symbol: str, settings, result: dict) -> None:
    """Append one experiment record. Never raises (logging must not break a run)."""
    try:
        path = _experiments_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "symbol": symbol,
            "config": settings_snapshot(settings),
            "result": result,
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    except Exception:  # pragma: no cover - never let tracking break training
        logger.exception("Could not append experiment record")


def read_experiments(limit: int = 500) -> list[dict]:
    """Most recent ``limit`` experiment records (newest first)."""
    path = _experiments_file()
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:  # pragma: no cover
        return []
    return rows[-limit:][::-1]
