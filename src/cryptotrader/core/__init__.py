# src/cryptotrader/core/__init__.py
"""Core domain primitives: value types, the event hierarchy, and interfaces."""

from cryptotrader.core.events import (
    Event,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from cryptotrader.core.interfaces import (
    DataHandler,
    ExecutionHandler,
    FeatureCalculator,
    Predictor,
    RiskManager,
    Strategy,
)
from cryptotrader.core.types import (
    Bar,
    OrderType,
    Position,
    Prediction,
    Side,
    Trade,
)

__all__ = [
    "Event",
    "EventType",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "DataHandler",
    "Strategy",
    "FeatureCalculator",
    "Predictor",
    "RiskManager",
    "ExecutionHandler",
    "Bar",
    "Side",
    "OrderType",
    "Position",
    "Trade",
    "Prediction",
]
