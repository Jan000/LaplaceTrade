# src/cryptotrader/ml/model.py
"""ML engine: triple-barrier labeling, sample weights, and swappable predictors."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cryptotrader.core.types import Prediction, Side

logger = logging.getLogger(__name__)

_CLASS_TO_SIDE = {0: Side.SHORT, 1: Side.FLAT, 2: Side.LONG}
_SIDE_TO_CLASS = {Side.SHORT: 0, Side.FLAT: 1, Side.LONG: 2}

# Helper columns produced by the feature engine that are NOT model features:
#  * "atr"       — used for labels + ATR position sizing (raw atr is non-stationary).
#  * "trend_sig" — used only by the strategy's regime filter.
_NON_FEATURE_COLS = {"atr", "trend_sig", "vol_pct"}


def make_triple_barrier_labels(
    ohlcv: pd.DataFrame,
    atr: pd.Series,
    horizon: int = 15,
    tp_mult: float = 1.5,
    sl_mult: float = 1.5,
    return_events: bool = False,
):
    """Label each bar via the triple-barrier method.

    Returns a Series of int in {-1, 0, +1}. With ``return_events=True`` also
    returns a Series ``t1`` giving the *end position* of each label's window (the
    bar where a barrier was touched, else the vertical/time barrier) — needed for
    sample-uniqueness weighting.
    """
    close = ohlcv["close"].to_numpy()
    high = ohlcv["high"].to_numpy()
    low = ohlcv["low"].to_numpy()
    atr_arr = atr.to_numpy()
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    t1 = np.arange(n, dtype=np.int64)

    for i in range(n - horizon):
        a = atr_arr[i]
        end = i + horizon
        if not np.isfinite(a) or a <= 0:
            t1[i] = end
            continue
        entry = close[i]
        upper = entry + tp_mult * a
        lower = entry - sl_mult * a
        label = 0
        for j in range(i + 1, i + 1 + horizon):
            if high[j] >= upper:
                label, end = 1, j
                break
            if low[j] <= lower:
                label, end = -1, j
                break
        labels[i] = label
        t1[i] = end

    # Tail rows have no full forward window: clamp their span to the last bar.
    for i in range(max(0, n - horizon), n):
        t1[i] = n - 1

    labels_s = pd.Series(labels, index=ohlcv.index, name="label")
    if return_events:
        return labels_s, pd.Series(t1, index=ohlcv.index, name="t1")
    return labels_s


def make_trend_scanning_labels(
    ohlcv: pd.DataFrame,
    min_window: int = 5,
    max_window: int = 20,
    t_threshold: float = 0.0,
    return_events: bool = False,
):
    """Trend-scanning labels (Lopez de Prado, *ML for Asset Managers*, ch. 5).

    For each bar, fit a linear trend on log-price over every forward window length in
    ``[min_window, max_window]``, keep the window whose slope has the largest |t-stat|,
    and label the bar by the **sign of that most-significant trend** (0 if |t| <
    ``t_threshold``). This targets the dominant statistically-significant move ahead,
    rather than which fixed ATR barrier is touched first. ``t1`` (the chosen window end)
    drives the same uniqueness weighting as the triple-barrier labels.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    logp = np.log(ohlcv["close"].to_numpy(dtype=float))
    n = len(logp)
    best_abs_t = np.zeros(n)
    best_slope = np.zeros(n)
    best_len = np.full(n, 0, dtype=np.int64)
    max_window = min(max_window, n - 1)
    for L in range(max(3, min_window), max_window + 1):
        x = np.arange(L, dtype=float)
        xm = x.mean()
        xc = x - xm
        sxx = float((xc * xc).sum())
        win = sliding_window_view(logp, L)                 # (n-L+1, L)
        ym = win.mean(axis=1)
        slope = (win @ xc) / sxx                            # sum(xc*y)/Sxx  (sum xc = 0)
        sst = ((win - ym[:, None]) ** 2).sum(axis=1)
        sse = np.maximum(sst - slope * slope * sxx, 1e-12)
        t = slope / np.sqrt(sse / (L - 2) / sxx)
        at = np.abs(t)
        m = at > best_abs_t[: len(at)]                      # starts that improve at this L
        idx = np.where(m)[0]
        best_abs_t[idx] = at[idx]
        best_slope[idx] = slope[idx]
        best_len[idx] = L

    labels = np.sign(best_slope).astype(np.int8)
    if t_threshold > 0:
        labels[best_abs_t < t_threshold] = 0
    pos = np.arange(n, dtype=np.int64)
    t1 = np.minimum(pos + np.where(best_len > 0, best_len, 1), n - 1)
    labels_s = pd.Series(labels, index=ohlcv.index, name="label")
    if return_events:
        return labels_s, pd.Series(t1, index=ohlcv.index, name="t1")
    return labels_s


def make_labels(ohlcv: pd.DataFrame, atr: pd.Series, barriers, return_events: bool = False):
    """Dispatch to the configured labelling method (triple-barrier or trend-scanning)."""
    if getattr(barriers, "label_method", "triple_barrier") == "trend_scan":
        return make_trend_scanning_labels(
            ohlcv, barriers.ts_min_window, barriers.ts_max_window,
            barriers.ts_t_threshold, return_events=return_events,
        )
    ltp, lsl = barriers.label_barriers
    return make_triple_barrier_labels(
        ohlcv, atr, horizon=barriers.horizon, tp_mult=ltp, sl_mult=lsl,
        return_events=return_events,
    )


def make_sample_weights(t1: pd.Series) -> pd.Series:
    """Average-uniqueness sample weights for overlapping labels (Lopez de Prado, ch. 4).

    Each label ``i`` spans bar positions ``[i, t1[i]]``. Bars covered by many
    concurrent labels contribute less unique information, so a label's weight is
    the mean of ``1/concurrency`` over its span. Weights are normalised to mean 1.
    """
    end = t1.to_numpy().astype(np.int64)
    n = len(end)
    if n == 0:
        return pd.Series([], index=t1.index, name="weight", dtype=float)
    max_pos = int(end.max()) + 1
    conc = np.zeros(max_pos + 1, dtype=float)
    for i in range(n):
        conc[i : end[i] + 1] += 1.0
    w = np.empty(n, dtype=float)
    for i in range(n):
        seg = conc[i : end[i] + 1]
        w[i] = float(np.mean(1.0 / np.maximum(seg, 1.0))) if seg.size else 1.0
    mean_w = w.mean()
    if mean_w > 0:
        w = w / mean_w
    return pd.Series(w, index=t1.index, name="weight")


class MomentumBaselinePredictor:
    """Dependency-free rule-based predictor (benchmark / smoke test)."""

    def __init__(self, momentum_col: str = "mom_5", k: float = 8.0) -> None:
        self.momentum_col = momentum_col
        self.k = k

    def _score(self, feats: pd.DataFrame) -> np.ndarray:
        mom = feats[self.momentum_col].to_numpy()
        vol = feats["realized_vol"].to_numpy()
        vwap_z = feats.get("vwap_dev_z", pd.Series(0.0, index=feats.index)).to_numpy()
        norm_mom = mom / (vol + 1e-9)
        return norm_mom - 0.15 * vwap_z

    def predict_batch(self, features: pd.DataFrame) -> list[Prediction]:
        score = self._score(features)
        proba_up = 1.0 / (1.0 + np.exp(-self.k * score))
        out: list[Prediction] = []
        for p in proba_up:
            if p >= 0.5:
                out.append(Prediction(Side.LONG, float(p), (1 - p, p)))
            else:
                out.append(Prediction(Side.SHORT, float(1 - p), (1 - p, p)))
        return out

    def predict(self, features: pd.Series) -> Prediction:
        return self.predict_batch(features.to_frame().T)[0]


def _apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale a probability matrix. T>1 softens, T<1 sharpens; argmax unchanged."""
    if not temperature or temperature == 1.0:
        return proba
    logp = np.log(np.clip(proba, 1e-12, 1.0)) / float(temperature)
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """Pick the temperature minimising multiclass NLL on a held-out set (grid, no scipy)."""
    if len(y) == 0:
        return 1.0
    idx = np.arange(len(y))
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.5, 3.0, 51):
        p = _apply_temperature(proba, t)
        nll = -float(np.mean(np.log(p[idx, y] + 1e-12)))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


class EnsemblePredictor:
    """Averages the class probabilities of several LightGBM members.

    Each member is the same model trained with a different random seed (and bagging),
    so they make different errors. Averaging their probabilities cancels seed/sampling
    variance — the dominant noise source when the training set is only a few thousand
    rows — without changing the systematic signal. Implements the Predictor protocol.

    ``temperature`` (default 1.0 = off) applies post-hoc temperature scaling to the
    averaged probabilities so the reported confidence is calibrated — making the
    strategy's entry thresholds and EV gate meaningful. It is fit on held-out data.
    """

    def __init__(self, members: list["LightGBMPredictor"], temperature: float = 1.0) -> None:
        if not members:
            raise ValueError("EnsemblePredictor needs at least one member.")
        self._members = members
        self.temperature = float(temperature)

    def proba_matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Averaged raw class probabilities of the members (before temperature)."""
        proba = None
        for m in self._members:
            p = m.predict_proba_matrix(features)
            proba = p if proba is None else proba + p
        return proba / len(self._members)

    def predict_batch(self, features: pd.DataFrame) -> list[Prediction]:
        proba = _apply_temperature(self.proba_matrix(features), self.temperature)
        out: list[Prediction] = []
        for row in proba:
            cls = int(np.argmax(row))
            out.append(Prediction(_CLASS_TO_SIDE[cls], float(row[cls]), tuple(row)))
        return out

    def predict(self, features: pd.Series) -> Prediction:
        return self.predict_batch(features.to_frame().T)[0]

    def save(self, path: str | Path) -> None:
        import joblib

        joblib.dump(
            {"ensemble": [(m._model, m._feature_names) for m in self._members],
             "temperature": self.temperature}, path
        )

    @classmethod
    def load(cls, path: str | Path) -> "EnsemblePredictor":
        import joblib

        blob = joblib.load(path)
        members = []
        for model, names in blob["ensemble"]:
            m = LightGBMPredictor()
            m._model, m._feature_names = model, names
            members.append(m)
        return cls(members, temperature=float(blob.get("temperature", 1.0)))


class LightGBMPredictor:
    """LightGBM 3-class predictor implementing the Predictor protocol."""

    def __init__(self, params: dict | None = None, drop_features: list[str] | None = None) -> None:
        self._drop = set(drop_features or [])
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

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        eval_fraction: float = 0.2,
        sample_weight: pd.Series | None = None,
    ) -> dict[str, float]:
        """Fit on aligned features/labels (time-ordered split, optional weights)."""
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("lightgbm is required to train LightGBMPredictor.") from exc

        cols = [c for c in features.columns if c not in _NON_FEATURE_COLS and c not in self._drop]
        data = features[cols].join(labels).dropna()
        X = data[cols]
        y = data[labels.name].map(lambda v: _SIDE_TO_CLASS[Side(int(np.sign(v)))])
        self._feature_names = list(cols)

        split = int(len(X) * (1.0 - eval_fraction))
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        sw_train = None
        if sample_weight is not None:
            sw = sample_weight.reindex(data.index)
            sw_train = sw.iloc[:split].to_numpy()

        self._model = lgb.LGBMClassifier(**self._params)
        self._model.fit(
            X_train,
            y_train,
            sample_weight=sw_train,
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

    def predict_proba_matrix(self, features: pd.DataFrame) -> np.ndarray:
        """Raw (n, 3) class-probability matrix on the model's own feature columns."""
        if self._model is None:
            raise RuntimeError("Model is not trained/loaded.")
        cols = self._feature_names or [
            c for c in features.columns if c not in _NON_FEATURE_COLS and c not in self._drop
        ]
        return self._model.predict_proba(features[cols])

    def predict_batch(self, features: pd.DataFrame) -> list[Prediction]:
        proba = self.predict_proba_matrix(features)
        out: list[Prediction] = []
        for row in proba:
            cls = int(np.argmax(row))
            out.append(Prediction(_CLASS_TO_SIDE[cls], float(row[cls]), tuple(row)))
        return out

    def predict(self, features: pd.Series) -> Prediction:
        return self.predict_batch(features.to_frame().T)[0]


def build_ensemble(settings, feats, labels, weights, cal_feats=None, cal_labels=None):
    """Train the seed-ensemble (shared by train_model.py and walkforward.py).

    Returns ``(predictor, metrics)``. When ``model.use_calibration`` is on and a
    calibration slice is supplied, the ensemble's probabilities are temperature-scaled
    (fit on that held-out slice). Returns an :class:`EnsemblePredictor` whenever
    calibration is applied (so the temperature is carried), otherwise a single
    :class:`LightGBMPredictor` for ``ensemble_size == 1``.
    """
    n = max(1, settings.model.ensemble_size)
    members: list[LightGBMPredictor] = []
    metrics: dict[str, float] = {"val_accuracy": 0.0}
    for k in range(n):
        params = settings.model.to_lgbm_params()
        params["random_state"] = settings.model.random_state + k
        m = LightGBMPredictor(params, drop_features=settings.model.drop_features)
        metrics = m.train(feats, labels, eval_fraction=settings.model.eval_fraction,
                          sample_weight=weights)
        members.append(m)

    use_cal = getattr(settings.model, "use_calibration", False) and cal_feats is not None
    if not use_cal:
        return (members[0] if n == 1 else EnsemblePredictor(members)), metrics

    pred = EnsemblePredictor(members)        # wrap (even a single member) so T is carried
    try:
        names = members[0]._feature_names
        df = cal_feats[names].join(cal_labels.rename("y")).dropna()
        if len(df) >= 30:
            proba = pred.proba_matrix(df)
            y = df["y"].map(lambda v: _SIDE_TO_CLASS[Side(int(np.sign(v)))]).to_numpy()
            pred.temperature = fit_temperature(proba, y)
            logger.info("Calibration: temperature T=%.3f fit on %d held-out rows",
                        pred.temperature, len(df))
        else:
            logger.warning("Calibration skipped: only %d clean held-out rows.", len(df))
    except Exception:  # pragma: no cover - calibration must never break training
        logger.exception("Temperature calibration failed; using uncalibrated ensemble.")
    return pred, metrics
