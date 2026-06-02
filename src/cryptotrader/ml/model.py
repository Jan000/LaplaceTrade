# src/cryptotrader/ml/model.py
"""ML engine: triple-barrier labeling + swappable predictors.

Two predictors implement the :class:`~cryptotrader.core.interfaces.Predictor`
contract:

* :class:`LightGBMPredictor` — the production path. LightGBM is preferred over
  XGBoost here for its leaf-wise growth and very low single-row inference latency
  on wide tabular feature sets, which matters in the live hot loop.
* :class:`MomentumBaselinePredictor` — a dependency-free rule-based predictor used
  for smoke tests and as a benchmark the ML model must beat.

Labels come from the **triple-barrier method**: for each bar we look forward a
fixed horizon and label +1/-1/0 depending on whether an ATR-scaled take-profit or
stop-loss barrier is touched first. Labels are *intentionally* forward-looking —
that is correct for supervised targets and never leaks into the backtest, which
consumes only the model's predictions on (backward-looking) features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cryptotrader.core.types import Prediction, Side

logger = logging.getLogger(__name__)

# Map the 3 model classes <-> trading direction.
_CLASS_TO_SIDE = {0: Side.SHORT, 1: Side.FLAT, 2: Side.LONG}
_SIDE_TO_CLASS = {Side.SHORT: 0, Side.FLAT: 1, Side.LONG: 2}


def make_triple_barrier_labels(
    ohlcv: pd.DataFrame,
    atr: pd.Series,
    horizon: int = 15,
    tp_mult: float = 1.5,
    sl_mult: float = 1.5,
) -> pd.Series:
    """Label each bar via the triple-barrier method.

    Parameters
    ----------
    ohlcv:
        OHLCV frame.
    atr:
        ATR series aligned to ``ohlcv`` (sets barrier widths per bar).
    horizon:
        Vertical barrier: max number of forward bars to wait.
    tp_mult, sl_mult:
        Take-profit / stop-loss barrier widths in ATR units.

    Returns
    -------
    pandas.Series of int in {-1, 0, +1}, aligned to ``ohlcv.index``. The last
    ``horizon`` rows are 0 (insufficient lookahead) and should be dropped before
    training.
    """
    close = ohlcv["close"].to_numpy()
    high = ohlcv["high"].to_numpy()
    low = ohlcv["low"].to_numpy()
    atr_arr = atr.to_numpy()
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n - horizon):
        a = atr_arr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        entry = close[i]
        upper = entry + tp_mult * a
        lower = entry - sl_mult * a
        label = 0
        for j in range(i + 1, i + 1 + horizon):
            if high[j] >= upper:
                label = 1
                break
            if low[j] <= lower:
                label = -1
                break
        labels[i] = label

    return pd.Series(labels, index=ohlcv.index, name="label")


class MomentumBaselinePredictor:
    """Dependency-free rule-based predictor (benchmark / smoke test).

    Blends short-horizon momentum (normalised by realized volatility) with a mild
    mean-reversion tilt from the VWAP-deviation z-score. Confidence is a logistic
    squash of the combined score into ``[0.5, 1]``.
    """

    def __init__(self, momentum_col: str = "mom_5", k: float = 8.0) -> None:
        self.momentum_col = momentum_col
        self.k = k

    def _score(self, feats: pd.DataFrame) -> np.ndarray:
        mom = feats[self.momentum_col].to_numpy()
        vol = feats["realized_vol"].to_numpy()
        vwap_z = feats.get("vwap_dev_z", pd.Series(0.0, index=feats.index)).to_numpy()
        norm_mom = mom / (vol + 1e-9)
        # Trend-following on momentum, fade extreme VWAP stretch.
        return norm_mom - 0.15 * vwap_z

    def predict_batch(self, features: pd.DataFrame) -> list[Prediction]:
        score = self._score(features)
        proba_up = 1.0 / (1.0 + np.exp(-self.k * score))  # logistic
        out: list[Prediction] = []
        for p in proba_up:
            if p >= 0.5:
                out.append(Prediction(Side.LONG, float(p), (1 - p, p)))
            else:
                out.append(Prediction(Side.SHORT, float(1 - p), (1 - p, p)))
        return out

    def predict(self, features: pd.Series) -> Prediction:
        return self.predict_batch(features.to_frame().T)[0]


class LightGBMPredictor:
    """LightGBM 3-class predictor implementing the Predictor protocol."""

    def __init__(self, params: dict | None = None) -> None:
        self._params = params or {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": 400,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "verbosity": -1,
        }
        self._model = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------ #
    # Training / persistence
    # ------------------------------------------------------------------ #
    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        eval_fraction: float = 0.2,
    ) -> dict[str, float]:
        """Fit the model on aligned features/labels with a time-ordered split."""
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("lightgbm is required to train LightGBMPredictor.") from exc

        data = features.join(labels).dropna()
        X = data[features.columns]
        y = data[labels.name].map(lambda v: _SIDE_TO_CLASS[Side(int(np.sign(v)))])
        self._feature_names = list(features.columns)

        split = int(len(X) * (1.0 - eval_fraction))
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        self._model = lgb.LGBMClassifier(**self._params)
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)] if len(X_val) else None,
            eval_metric="multi_logloss",
        )
        acc = float((self._model.predict(X_val) == y_val).mean()) if len(X_val) else 0.0
        logger.info("LightGBM trained on %d rows, val accuracy=%.3f", len(X_train), acc)
        return {"val_accuracy": acc, "n_train": float(len(X_train))}

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump({"model": self._model, "features": self._feature_names}, path)

    def load(self, path: str | Path) -> "LightGBMPredictor":
        import joblib

        blob = joblib.load(path)
        self._model = blob["model"]
        self._feature_names = blob["features"]
        return self

    # ------------------------------------------------------------------ #
    # Inference (Predictor protocol)
    # ------------------------------------------------------------------ #
    def predict_batch(self, features: pd.DataFrame) -> list[Prediction]:
        if self._model is None:
            raise RuntimeError("Model is not trained/loaded.")
        cols = self._feature_names or list(features.columns)
        proba = self._model.predict_proba(features[cols])
        out: list[Prediction] = []
        for row in proba:
            cls = int(np.argmax(row))
            out.append(Prediction(_CLASS_TO_SIDE[cls], float(row[cls]), tuple(row)))
        return out

    def predict(self, features: pd.Series) -> Prediction:
        return self.predict_batch(features.to_frame().T)[0]
