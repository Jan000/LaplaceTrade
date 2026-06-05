# src/cryptotrader/risk/manager.py
"""ATR-based risk manager with an expected-value entry gate.

Sizing (fixed-fractional, volatility-normalised):

    risk_amount   = equity * risk_per_trade
    stop_distance = sl_mult * ATR
    quantity      = risk_amount / stop_distance      # then capped by leverage

Entry gate
----------
With meta-labeling the signal carries ``confidence = P(win)``. The principled,
cost-aware filter is then **expected value**:

    EV = P(win) * tp_distance - (1 - P(win)) * sl_distance - round_trip_cost

Trade only when ``EV > min_expected_value``. Unlike the older cost-ratio filter,
EV does not loosen when fees fall (it folds the cost in directly), so it self-
calibrates to the execution cost. The legacy ``min_edge_cost_ratio`` filter
remains available when ``use_ev_filter`` is off.
"""

from __future__ import annotations

from cryptotrader.config import BarrierConfig, ExecutionConfig, RiskConfig
from cryptotrader.core.events import OrderEvent, SignalEvent
from cryptotrader.core.interfaces import RiskManager
from cryptotrader.core.types import Bar, OrderType, Side


class ATRRiskManager(RiskManager):
    """Volatility-scaled sizing with leverage cap and an EV (or cost-ratio) gate."""

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

        cost = self._round_trip_cost_per_unit(price)
        if self.config.use_ev_filter:
            # P(win) from the (meta) predictor's confidence.
            p_win = max(0.0, min(1.0, signal.confidence))
            ev = p_win * tp_distance - (1.0 - p_win) * stop_distance - cost
            if ev <= self.config.min_expected_value:
                return None
        elif self.config.min_edge_cost_ratio > 0.0:
            if tp_distance < self.config.min_edge_cost_ratio * cost:
                return None

        risk_amount = equity * self.config.risk_per_trade
        quantity = risk_amount / stop_distance
        if self.config.max_leverage > 0.0 and price > 0.0:
            quantity = min(quantity, self.config.max_leverage * equity / price)
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
