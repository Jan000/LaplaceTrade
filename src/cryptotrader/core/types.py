# src/cryptotrader/core/types.py
"""Immutable / lightweight value types shared across the whole system.

These objects are deliberately tiny and ``slots``-based: in the hot backtest loop
millions of them may be created, so we avoid per-instance ``__dict__`` overhead.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone


class Side(enum.IntEnum):
    """Directional bias of a signal or position.

    Encoded as integers so they can be multiplied directly with price deltas
    when computing PnL (``pnl = side * (exit - entry)``).
    """

    FLAT = 0
    LONG = 1
    SHORT = -1


class OrderType(enum.Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(slots=True, frozen=True)
class Bar:
    """A single closed OHLCV candle.

    Attributes
    ----------
    timestamp:
        Close time of the candle (timezone-aware, UTC).
    open, high, low, close:
        Standard OHLC prices.
    volume:
        Base-asset volume traded during the candle.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_ccxt(cls, row: list[float]) -> "Bar":
        """Build a :class:`Bar` from a ccxt OHLCV row ``[ms, o, h, l, c, v]``."""
        ts = datetime.fromtimestamp(row[0] / 1000.0, tz=timezone.utc)
        return cls(ts, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]))


@dataclass(slots=True)
class Prediction:
    """Output of the ML engine for a single timestep.

    Attributes
    ----------
    direction:
        Discrete directional call (-1 short, 0 flat, +1 long).
    confidence:
        Calibrated probability / score in ``[0, 1]`` backing ``direction``.
    raw:
        Optional raw model output (e.g. class probabilities) for diagnostics.
    """

    direction: Side
    confidence: float
    raw: tuple[float, ...] = ()


@dataclass(slots=True)
class Position:
    """A currently open position. Mutated in place by the portfolio."""

    symbol: str
    side: Side
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    # Trailing take-profit distance in price units (atr_trail_mult * ATR@entry).
    trail_distance: float = 0.0
    # High-water / low-water mark of price while the position is open,
    # used both for trailing stops and for the Max Efficiency Ratio.
    mfe_price: float = 0.0  # most favourable excursion price seen so far
    mae_price: float = 0.0  # most adverse excursion price seen so far

    def unrealized_pnl(self, price: float) -> float:
        """PnL in quote currency if the position were closed at ``price``."""
        return int(self.side) * (price - self.entry_price) * self.quantity


@dataclass(slots=True)
class Trade:
    """A fully closed round-trip trade, plus post-trade analytics."""

    symbol: str
    side: Side
    quantity: float
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    fees: float
    gross_pnl: float
    net_pnl: float
    # Best price reached between entry and exit (favourable direction).
    best_price: float
    exit_reason: str = "signal"
    efficiency_ratio: float = field(default=0.0)

    @property
    def return_pct(self) -> float:
        """Net return on notional at entry."""
        notional = self.entry_price * self.quantity
        return self.net_pnl / notional if notional else 0.0
