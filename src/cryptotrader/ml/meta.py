# src/cryptotrader/ml/meta.py
"""Meta-labeling (Lopez de Prado, "Advances in Financial ML", ch. 3).

A *primary* model predicts the trade DIRECTION (the existing 3-class LightGBM).
A *secondary* (meta) model then predicts whether acting on that signal will WIN
— a binary bet/no-bet filter that improves precision without touching the
primary's recall.

Crucial detail: the meta-model is trained on the primary's **out-of-fold**
predictions (primary fit on the first half, predicted on the second half), so the
meta-labels reflect realistic primary errors rather than in-sample overfitting.
The primary that ships is then refit on all training data.

The wrapper implements the Predictor protocol and returns the primary's direction
with ``confidence = P(win)`` from the meta-model — so the strategy's long/short
threshold simply becomes the minimum win-probability required to trade.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cryptotrader.core.types import Prediction, Side
from cryptotrader.ml.model import LightGBMPredictor

logger = logging.getLogger(__name__)

_META_EXTRA = ["primary_dir", "primary_conf"]

_DEFAULT_META_PARAMS = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


class MetaLabeledPredictor:
    """Primary direction model + binary meta 'should I act?' model."""

    def __init__(self, primary: LightGBMPredictor, meta_model, feature_names: list[str]) -> None:
        self._primary = primary
        self._meta = meta_model
        self._feature_names = list(feature_names)

    # ------------------------------------------------------------------ #
    # Predictor protocol
    # ------------------------------------------------------------------ #
    def predict_batch(self, features: pd.DataFrame) -> list[Prediction]:
        primary = self._primary.predict_batch(features)
        pdir = np.array([int(p.direction) for p in primary], dtype=float)
        pconf = np.array([p.confidence for p in primary], dtype=float)

        meta_X = features[self._feature_names].copy()
        meta_X["primary_dir"] = pdir
        meta_X["primary_conf"] = pconf
        pwin = self._meta.predict_proba(meta_X[self._feature_names + _META_EXTRA])[:, 1]

        out: list[Prediction] = []
        for p, w in zip(primary, pwin):
            # Direction from the primary; conviction = meta P(win). FLAT primary
            # stays FLAT (and zero conviction) so it is never traded.
            conf = 0.0 if p.direction is Side.FLAT else float(w)
            out.append(Prediction(p.direction, conf, (1.0 - float(w), float(w))))
        return out

    def predict(self, features: pd.Series) -> Prediction:
        return self.predict_batch(features.to_frame().T)[0]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(
            {"primary_model": self._primary._model,
             "primary_features": self._primary._feature_names,
             "meta": self._meta,
             "features": self._feature_names},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MetaLabeledPredictor":
        import joblib

        blob = joblib.load(path)
        primary = LightGBMPredictor()
        primary._model = blob["primary_model"]
        primary._feature_names = blob["primary_features"]
        return cls(primary, blob["meta"], blob["features"])


def train_meta_labeled(
    features: pd.DataFrame,
    labels: pd.Series,
    lgbm_params: dict,
    eval_fraction: float = 0.2,
    embargo: int = 15,
    meta_params: dict | None = None,
    sample_weight: pd.Series | None = None,
) -> tuple[MetaLabeledPredictor, dict[str, float]]:
    """Train primary + meta with out-of-fold meta-labels.

    Parameters
    ----------
    features, labels:
        Aligned feature matrix and triple-barrier labels (-1/0/+1).
    lgbm_params:
        Primary model hyperparameters (from MLConfig.to_lgbm_params()).
    embargo:
        Bars skipped between the primary-fit half and the meta half, so the
        forward-looking label horizon of the first half can't leak into the
        second. Set to the label horizon.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lightgbm is required for meta-labeling.") from exc

    data = features.join(labels).dropna()
    X = data[features.columns]
    y = data[labels.name].astype(int)
    n = len(X)
    if n < 200:
        raise RuntimeError("Not enough rows to train a meta model.")

    half = n // 2
    # 1) Primary on the first half, OOF predictions on the (embargoed) second half.
    primary_oof = LightGBMPredictor(lgbm_params)
    primary_oof.train(X.iloc[:half], y.iloc[:half], eval_fraction=eval_fraction,
                      sample_weight=sample_weight)

    b_start = min(half + embargo, n - 1)
    Xb, yb = X.iloc[b_start:], y.iloc[b_start:]
    preds = primary_oof.predict_batch(Xb)
    pdir = np.array([int(p.direction) for p in preds])
    pconf = np.array([p.confidence for p in preds])
    fired = pdir != 0
    if fired.sum() < 50:
        raise RuntimeError("Primary fired on too few bars to train a meta model.")

    meta_X = Xb[fired].copy()
    meta_X["primary_dir"] = pdir[fired].astype(float)
    meta_X["primary_conf"] = pconf[fired]
    # Meta label: 1 if the directional bet hit its take-profit barrier.
    meta_y = (np.sign(yb.to_numpy()[fired]) == pdir[fired]).astype(int)

    from cryptotrader.ml.model import _NON_FEATURE_COLS

    feat_cols = [c for c in features.columns if c not in _NON_FEATURE_COLS]
    cols = feat_cols + _META_EXTRA
    meta = lgb.LGBMClassifier(**(meta_params or _DEFAULT_META_PARAMS))
    meta.fit(meta_X[cols], meta_y)
    meta_acc = float((meta.predict(meta_X[cols]) == meta_y).mean())
    base_rate = float(meta_y.mean())

    # 2) Ship a primary refit on ALL training data, wrapped with the meta-model.
    final_primary = LightGBMPredictor(lgbm_params)
    final_primary.train(X, y, eval_fraction=eval_fraction, sample_weight=sample_weight)
    logger.info(
        "Meta-labeling: meta train acc=%.3f vs base win-rate=%.3f on %d fired bars",
        meta_acc, base_rate, int(fired.sum()),
    )
    return MetaLabeledPredictor(final_primary, meta, feat_cols), {
        "meta_train_accuracy": meta_acc,
        "primary_win_base_rate": base_rate,
        "n_meta_samples": float(fired.sum()),
    }


def load_predictor(path: str | Path):
    """Load either a plain LightGBM model or a meta-labeled one (auto-detect)."""
    import joblib

    blob = joblib.load(path)
    if "meta" in blob:
        return MetaLabeledPredictor.load(path)
    if "ensemble" in blob:
        from cryptotrader.ml.model import EnsemblePredictor

        return EnsemblePredictor.load(path)
    return LightGBMPredictor().load(path)
