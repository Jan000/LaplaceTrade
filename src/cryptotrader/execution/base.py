# src/cryptotrader/execution/base.py
"""Shared execution cost model.

Both the backtest simulator and the live *paper* handler must apply *identical*
slippage and fee mechanics, otherwise paper results would silently diverge from
backtest expectations. Centralising the cost math here guarantees that parity.
"""

from __future__ import annotations

from cryptotrader.config import ExecutionConfig
from cryptotrader.core.events import FillEvent, OrderEvent


def apply_costs(
    order: OrderEvent,
    base_price: float,
    config: ExecutionConfig,
) -> FillEvent:
    """Apply bps slippage + taker fee to ``base_price`` and build a FillEvent.

    Slippage always works against the trader: buys (``side > 0``) fill higher,
    sells (``side < 0``) fill lower. The fee is the taker fee on filled notional.
    """
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
