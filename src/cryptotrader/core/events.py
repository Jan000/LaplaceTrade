# src/cryptotrader/core/events.py
"""The event hierarchy that drives the whole system.

Both the backtester and the live engine are *event-driven*: a single queue is fed
with events that flow strictly in one direction

    MarketEvent -> SignalEvent -> OrderEvent -> FillEvent

This decoupling is what lets us reuse the exact same Strategy / Risk / Portfolio
code in backtest and live mode — only the producers of ``MarketEvent`` and the
consumers of ``OrderEvent`` differ.

Each event carries its :class:`EventType` as a ``ClassVar`` (a constant per
subclass) rather than an instance field, so the slotted dataclasses stay minimal
and fast to construct in the hot loop.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from cryptotrader.core.types import Bar, OrderType, Side


class EventType(enum.Enum):
    MARKET = "market"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"


class Event:
    """Base marker for all events; ``type`` is set per subclass."""

    __slots__ = ()
    type: ClassVar[EventType]


@dataclass(slots=True)
class MarketEvent(Event):
    """Emitted whenever a new candle has *closed* and is safe to act on.

    Carrying the closed :class:`Bar` (rather than a tick) is what keeps the
    backtest free of look-ahead bias: a strategy may only ever see fully closed
    candles up to and including ``bar.timestamp``.
    """

    type: ClassVar[EventType] = EventType.MARKET
    bar: Bar


@dataclass(slots=True)
class SignalEvent(Event):
    """A directional trading intent produced by the strategy/ML engine.

    The strategy expresses *intent and conviction*; it does not size the trade or
    place orders — that is the job of the risk manager and execution handler.
    """

    type: ClassVar[EventType] = EventType.SIGNAL
    symbol: str
    timestamp: datetime
    side: Side
    confidence: float


@dataclass(slots=True)
class OrderEvent(Event):
    """A concrete, risk-sized order ready to be sent to an execution handler.

    Protective levels are expressed as *distances* in price units rather than
    absolute levels, so the portfolio can anchor them to the actual fill price
    (which differs from the signal-bar close by slippage and the next-bar gap).
    """

    type: ClassVar[EventType] = EventType.ORDER
    symbol: str
    timestamp: datetime
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    stop_distance: float = 0.0  # hard-stop distance from fill (atr_stop_mult * ATR)
    trail_distance: float = 0.0  # trailing take-profit distance (atr_trail_mult * ATR)
    is_exit: bool = False


@dataclass(slots=True)
class FillEvent(Event):
    """Confirmation that an order executed, including realistic costs."""

    type: ClassVar[EventType] = EventType.FILL
    symbol: str
    timestamp: datetime
    side: Side
    quantity: float
    fill_price: float
    fee: float
    slippage: float
    is_exit: bool
