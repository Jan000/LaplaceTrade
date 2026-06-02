# src/cryptotrader/core/interfaces.py
"""Abstract interfaces that define the seams between modules.

These contracts are the heart of the architecture. By programming against them,
the backtester never imports a concrete exchange client, and the live engine
never imports the simulator. The single most important contract for this MVP is
:class:`Predictor` — it is the *only* thing the strategy knows about the ML
engine, which keeps model internals (LightGBM, XGBoost, an ensemble, ...) fully
swappable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

import pandas as pd

from cryptotrader.core.events import FillEvent, MarketEvent, OrderEvent, SignalEvent
from cryptotrader.core.types import Bar, Prediction


class DataHandler(ABC):
    """Produces :class:`MarketEvent` objects, historical or live.

    The backtester and live engine consume this identically; the only difference
    is whether bars come from a cached DataFrame or a websocket.
    """

    @abstractmethod
    def stream(self) -> AsyncIterator[MarketEvent]:
        """Yield market events in strict chronological order."""
        raise NotImplementedError


class FeatureCalculator(ABC):
    """Turns raw OHLCV into the feature matrix the model consumes.

    Two access patterns are supported to serve both run modes:

    * :meth:`transform` — fully vectorised, used once over historical data for
      training and for the backtest (backward-looking windows only -> no leak).
    * :meth:`update` — incremental, O(1)-amortised per new bar, used live.
    """

    @abstractmethod
    def transform(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute the full feature matrix for a historical OHLCV frame."""
        raise NotImplementedError

    @abstractmethod
    def update(self, bar: Bar) -> pd.Series | None:
        """Push one new closed bar and return its feature row (or ``None``)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Stable, ordered list of feature column names."""
        raise NotImplementedError


@runtime_checkable
class Predictor(Protocol):
    """The ML engine contract — the strategy's *only* view of the model.

    Implementations may wrap LightGBM, XGBoost, a calibrated ensemble, or even a
    rule-based baseline. They must be cheap to call (single-row inference in the
    hot path) and must never look into the future.
    """

    def predict(self, features: pd.Series) -> Prediction:
        """Return a :class:`Prediction` for a single feature row."""
        ...

    def predict_batch(self, features: pd.DataFrame) -> list[Prediction]:
        """Vectorised inference over many rows (used to speed up backtests)."""
        ...


class Strategy(ABC):
    """Translates market events into directional :class:`SignalEvent` objects.

    A strategy owns a :class:`FeatureCalculator` and a :class:`Predictor`; it
    contains *no* execution or sizing logic.
    """

    def prepare(self, features: pd.DataFrame) -> None:
        """Optional backtest hook: precompute over the full feature matrix.

        The backtester calls this once before the event loop so that strategies
        can run *batch* inference (far faster than per-bar calls). Live strategies
        ignore it and compute incrementally in :meth:`on_market`. Default: no-op.
        """
        return None

    @abstractmethod
    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        """React to a closed bar; return a signal or ``None`` to stay put."""
        raise NotImplementedError


class RiskManager(ABC):
    """Sizes positions and computes protective levels.

    Converts a conviction-only :class:`SignalEvent` into a concrete
    :class:`OrderEvent`, or ``None`` if the trade is rejected (e.g. exposure
    limits, insufficient equity).
    """

    @abstractmethod
    def size_order(
        self,
        signal: SignalEvent,
        last_bar: Bar,
        atr: float,
        equity: float,
        has_open_position: bool,
    ) -> OrderEvent | None:
        """Return a sized order for ``signal`` or ``None`` to skip it."""
        raise NotImplementedError


class ExecutionHandler(ABC):
    """Executes orders and emits realistic :class:`FillEvent` objects.

    Implemented twice: a simulator (slippage + fees, used in backtests) and a
    ccxt-backed live handler. Both honour the same contract.
    """

    @abstractmethod
    def execute(self, order: OrderEvent, reference_bar: Bar) -> FillEvent:
        """Execute ``order`` against ``reference_bar`` and return the fill."""
        raise NotImplementedError
