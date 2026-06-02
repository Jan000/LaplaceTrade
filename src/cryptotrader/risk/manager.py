# src/cryptotrader/risk/manager.py
"""ATR-based risk manager: volatility-scaled sizing and protective levels.

Sizing rule (fixed-fractional, volatility-normalised):

    risk_amount   = equity * risk_per_trade           # quote ccy at risk
    stop_distance = atr_stop_mult * ATR               # per-unit risk
    quantity      = risk_amount / stop_distance        # units of base asset

Because ``stop_distance`` scales with ATR, the *monetary* risk per trade is held
roughly constant across volatility regimes — small size in turbulent markets,
larger size in calm ones. The hard stop is a fixed ATR multiple from entry; the
trailing take-profit distance (consumed by the portfolio) is a separate ATR
multiple.
"""

from __future__ import annotations

from cryptotrader.config import RiskConfig
from cryptotrader.core.events import OrderEvent, SignalEvent
from cryptotrader.core.interfaces import RiskManager
from cryptotrader.core.types import Bar, OrderType, Side


class ATRRiskManager(RiskManager):
    """Volatility-scaled fixed-fractional position sizing."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def size_order(
        self,
        signal: SignalEvent,
        last_bar: Bar,
        atr: float,
        equity: float,
        has_open_position: bool,
    ) -> OrderEvent | None:
        """Convert a conviction signal into a sized entry order, or reject it."""
        if has_open_position:
            return None  # MVP: single concurrent position
        if signal.side is Side.FLAT or atr <= 0.0 or equity <= 0.0:
            return None

        stop_distance = self.config.atr_stop_mult * atr
        if stop_distance <= 0.0:
            return None

        risk_amount = equity * self.config.risk_per_trade
        quantity = risk_amount / stop_distance
        if quantity <= 0.0:
            return None

        trail_distance = self.config.atr_trail_mult * atr
        return OrderEvent(
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            side=signal.side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            stop_distance=stop_distance,
            trail_distance=trail_distance,
            is_exit=False,
        )
