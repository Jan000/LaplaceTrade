# src/cryptotrader/ml/registry.py
"""Per-symbol model registry: file paths + training metadata.

Models are stored per symbol so several coins can have their own model side by side
(``models/model_BTCUSDT.pkl``). A JSON sidecar records what the model was trained for
(symbol, timeframe, pooled symbols, when), which powers the dashboard's symbol
guardrail — refusing to place REAL orders with a model trained for a different
symbol/timeframe.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def safe_symbol(symbol: str) -> str:
    """Filesystem-safe form of a symbol (BTC/USDT -> BTCUSDT)."""
    return symbol.replace("/", "").replace(":", "")


def model_path_for(symbol: str, base: Path | str | None = None) -> Path:
    return Path(base or MODELS_DIR) / f"model_{safe_symbol(symbol)}.pkl"


def holdout_path_for(symbol: str, base: Path | str | None = None) -> Path:
    return Path(base or MODELS_DIR) / f"holdout_{safe_symbol(symbol)}.parquet"


def meta_path_for(model_path: str | Path) -> Path:
    p = Path(model_path)
    return p.with_name(p.name + ".meta.json")  # model_BTCUSDT.pkl.meta.json


def validation_path_for(kind: str, symbol: str, base: Path | str | None = None) -> Path:
    """Path for a persisted validation result, e.g. models/walkforward_BTCUSDT.json."""
    return Path(base or MODELS_DIR) / f"{kind}_{safe_symbol(symbol)}.json"


def write_validation(kind: str, symbol: str, result: dict) -> None:
    p = validation_path_for(kind, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**result, "symbol": symbol, "kind": kind,
               "saved_at": datetime.now(tz=timezone.utc).isoformat()}
    # default= coerces stray numpy scalars (np.bool_, np.float64, …) to native types so a
    # value like a numpy bool can never break the write (it has no plain-bool subclassing).
    p.write_text(
        json.dumps(payload, indent=2,
                   default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )


def read_validation(kind: str, symbol: str) -> dict | None:
    p = validation_path_for(kind, symbol)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover
        return None


def write_meta(model_path: str | Path, meta: dict) -> None:
    payload = {**meta, "saved_at": datetime.now(tz=timezone.utc).isoformat()}
    meta_path_for(model_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_meta(model_path: str | Path) -> dict | None:
    p = meta_path_for(model_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - corrupt sidecar
        logger.warning("Unreadable model meta at %s", p)
        return None


def list_models(base: Path | str | None = None) -> list[dict]:
    """All trained per-symbol models on disk, with their metadata."""
    root = Path(base or MODELS_DIR)
    out: list[dict] = []
    if not root.exists():
        return out
    for p in sorted(root.glob("model_*.pkl")):
        meta = read_meta(p) or {}
        out.append({"path": str(p), "symbol": meta.get("symbol"), "meta": meta})
    return out


def resolve_model(settings) -> tuple[Path | None, dict | None]:
    """Choose the model to load for ``settings.exchange.symbol``.

    Precedence: the per-symbol file ``models/model_<SYMBOL>.pkl`` → an explicit
    ``strategy.model_path`` (legacy single-model override) → none (use the baseline).
    Returns ``(path_or_None, meta_or_None)``.
    """
    per = model_path_for(settings.exchange.symbol)
    if per.exists():
        return per, read_meta(per)
    mp = settings.strategy.model_path
    if mp and Path(mp).exists():
        return Path(mp), read_meta(mp)
    return None, None
