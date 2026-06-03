# src/cryptotrader/risk/manager.py
"""ATR-based risk manager with explicit cost control.

Sizing (fixed-fractional, volatility-normalised):

    risk_amount   = equity * risk_per_trade
    stop_distance = sl_mult * ATR
    quantity      = risk_amount / stop_distance      # then capped by leverage

Two cost-control gates address the dominant failure mode of 1m/15m intraday
trading — fees and slippage scale with *notional*, which the ATR-scaled size can
blow up in calm regimes until each round trip costs as much as it risks:

* **Leverage cap** — notional is capped at ``max_leverage * equity`` so per-trade
  cost stays proportional to capital instead of exploding when ATR is small.
* **Cost-aware edge filter** — a trade is only taken if its take-profit target
  (``tp_mult * ATR``) clears the estimated round-trip cost by a safety factor,
  so low-volatility bars whose expected edge can't beat costs are skipped.
"""

from __future__ import annotations

from cryptotrader.config import BarrierConfig, ExecutionConfig, RiskConfig
from cryptotrader.core.events import OrderEvent, SignalEvent
from cryptotrader.core.interfaces import RiskManager
from cryptotrader.core.types import Bar, OrderType, Side


class ATRRiskManager(RiskManager):
    """Volatility-scaled sizing with leverage cap and cost-aware filtering."""

    def __init__(
        self,
        config: RiskConfig,
        barriers: BarrierConfig | None = None,
        execution: ExecutionConfig | None = None,
    ) -> None:
        self.config = config
        self.barriers = barriers or BarrierConfig()
        self.execution = execution

    def _round_trip_cost_per_unit(self, price: float) -> float:
        """Estimated entry+exit fees + slippage per unit of base asset."""
        if self.execution is None:
            return 0.0
        fee = 2.0 * self.execution.taker_fee
        slip = 2.0 * (self.execution.slippage_bps / 10_000.0)
        return price * (fee + slip)

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

        price = last_bar.close
        stop_distance = self.barriers.sl_mult * atr
        tp_distance = self.barriers.tp_mult * atr
        if stop_distance <= 0.0:
            return None

        # Cost-aware edge filter: skip trades whose target can't clear costs.
        if self.config.min_edge_cost_ratio > 0.0:
            min_edge = self.config.min_edge_cost_ratio * self._round_trip_cost_per_unit(price)
            if tp_distance < min_edge:
                return None

        risk_amount = equity * self.config.risk_per_trade
        quantity = risk_amount / stop_distance

        # Leverage cap: notional must not exceed max_leverage * equity.
        if self.config.max_leverage > 0.0 and price > 0.0:
            max_qty = self.config.max_leverage * equity / price
            quantity = min(quantity, max_qty)
        if quantity <= 0.0:
            return None

        return OrderEvent(
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            side=signal.side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            stop_distance=stop_distance,
            tp_distance=tp_distance,
            max_hold_bars=self.barriers.horizon,
            is_exit=False,
        )
