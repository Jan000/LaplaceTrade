# src/cryptotrader/core/types.py
"""Immutable / lightweight value types shared across the whole system."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone


class Side(enum.IntEnum):
    """Directional bias of a signal or position."""

    FLAT = 0
    LONG = 1
    SHORT = -1


class OrderType(enum.Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(slots=True, frozen=True)
class Bar:
    """A single closed OHLCV candle."""

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
    """Output of the ML engine for a single timestep."""

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
    trail_distance: float = 0.0
    max_hold_bars: int = 0
    bars_held: int = 0
    mfe_price: float = 0.0
    mae_price: float = 0.0

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
    best_price: float
    exit_reason: str = "signal"
    efficiency_ratio: float = field(default=0.0)

    @property
    def return_pct(self) -> float:
        """Net return on notional at entry."""
        notional = self.entry_price * self.quantity
        return self.net_pnl / notional if notional else 0.0
