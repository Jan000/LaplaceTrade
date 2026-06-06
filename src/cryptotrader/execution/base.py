# src/cryptotrader/execution/base.py
"""Shared execution cost model.

Both the backtest simulator and the live *paper* handler must apply *identical*
slippage and fee mechanics, otherwise paper results would silently diverge from
backtest expectations. Centralising the cost math here guarantees that parity.
"""

from __future__ import annotations

from cryptotrader.config import ExecutionConfig
from cryptotrader.core.events import FillEvent, OrderEvent
from cryptotrader.core.types import OrderType


def apply_costs(
    order: OrderEvent,
    base_price: float,
    config: ExecutionConfig,
) -> FillEvent:
    """Build a FillEvent for ``order`` at ``base_price`` with realistic costs.

    * MARKET (taker): bps slippage *against* the trader (buys fill higher, sells
      lower) plus the taker fee on notional.
    * LIMIT (maker): the order rests at a price we chose, so there is no slippage;
      it fills exactly at ``base_price`` (the posted limit) and pays the maker fee.
    """
    if order.order_type is OrderType.LIMIT:
        filled = base_price
        slippage_cost = 0.0
        fee = filled * order.quantity * config.maker_fee
        return FillEvent(
            symbol=order.symbol,
            timestamp=order.timestamp,
            side=order.side,
            quantity=order.quantity,
            fill_price=filled,
            fee=fee,
            slippage=slippage_cost,
            is_exit=order.is_exit,
        )

    slip = config.slippage_bps / 10_000.0
    direction = int(order.side)  # +1 buy, -1 sell
    filled = base_price * (1.0 + direction * slip)
    slippage_cost = abs(filled - base_price) * order.quantity
    fee = filled * order.quantity * config.taker_fee
    return FillEvent(
        symbol=order.symbol,
        timestamp=order.timestamp,
        side=order.side,
        quantity=order.quantity,
        fill_price=filled,
        fee=fee,
        slippage=slippage_cost,
        is_exit=order.is_exit,
    )
