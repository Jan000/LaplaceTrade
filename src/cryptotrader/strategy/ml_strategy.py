# src/cryptotrader/strategy/ml_strategy.py
"""MLStrategy — turns model predictions into directional signals.

This is the single seam between the ML engine and the rest of the system. It owns
a :class:`Predictor` and a :class:`FeatureCalculator` and converts probabilistic
predictions into :class:`SignalEvent` objects via confidence thresholds. It does
*no* sizing or execution.

Backtest vs live
----------------
* **Backtest** — the engine calls :meth:`prepare` once with the full feature
  matrix; the strategy runs a single batch inference and then serves O(1) lookups
  per bar (fast, and free of look-ahead because features are backward-looking).
* **Live** — :meth:`prepare` is never called; each :meth:`on_market` incrementally
  updates the feature engine with the new bar and runs single-row inference.
"""

from __future__ import annotations

import pandas as pd

from cryptotrader.config import StrategyConfig
from cryptotrader.core.events import MarketEvent, SignalEvent
from cryptotrader.core.interfaces import FeatureCalculator, Predictor, Strategy
from cryptotrader.core.types import Prediction, Side


class MLStrategy(Strategy):
    """Threshold-based strategy over a :class:`Predictor`."""

    def __init__(
        self,
        predictor: Predictor,
        config: StrategyConfig,
        symbol: str,
        feature_engine: FeatureCalculator | None = None,
    ) -> None:
        self._predictor = predictor
        self._config = config
        self._symbol = symbol
        self._feature_engine = feature_engine
        # Populated by prepare() in backtest mode; empty in live mode.
        self._predictions: dict[pd.Timestamp, Prediction] = {}
        self._replay = False

    # ------------------------------------------------------------------ #
    # Backtest precompute
    # ------------------------------------------------------------------ #
    def prepare(self, features: pd.DataFrame) -> None:
        """Batch-predict over the full (warmup-trimmed) feature matrix."""
        valid = features.dropna()
        if valid.empty:
            self._replay = True
            return
        preds = self._predictor.predict_batch(valid)
        self._predictions = dict(zip(valid.index, preds))
        self._replay = True

    # ------------------------------------------------------------------ #
    # Per-bar decision
    # ------------------------------------------------------------------ #
    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        prediction = self._lookup_prediction(event)
        if prediction is None:
            return None
        return self._to_signal(event, prediction)

    def _lookup_prediction(self, event: MarketEvent) -> Prediction | None:
        if self._replay:
            return self._predictions.get(pd.Timestamp(event.bar.timestamp))
        if self._feature_engine is None:
            raise RuntimeError("Live mode requires a feature_engine.")
        row = self._feature_engine.update(event.bar)
        return self._predictor.predict(row) if row is not None else None

    def _to_signal(self, event: MarketEvent, pred: Prediction) -> SignalEvent | None:
        bar = event.bar
        if pred.direction is Side.LONG and pred.confidence >= self._config.long_threshold:
            return SignalEvent(self._symbol, bar.timestamp, Side.LONG, pred.confidence)
        if (
            pred.direction is Side.SHORT
            and self._config.allow_short
            and pred.confidence >= self._config.short_threshold
        ):
            return SignalEvent(self._symbol, bar.timestamp, Side.SHORT, pred.confidence)
        return None
