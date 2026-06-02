# src/cryptotrader/execution/paper.py
"""Paper (dry-run) execution handler for live mode.

Runs against the *real* live data stream but never sends an order to the exchange:
fills are simulated instantly at the most recent price using the exact same cost
model as the backtester (:func:`cryptotrader.execution.base.apply_costs`). This is
the safe default for the MVP — it exercises the entire live code path (data feed,
strategy, risk, portfolio, persistence, dashboard) with zero capital at risk.

The interface is ``async`` so it is drop-in interchangeable with the real
:class:`~cryptotrader.execution.live.CCXTExecutionHandler`, whose order placement
is genuinely I/O-bound.
"""

from __future__ import annotations

import logging

from cryptotrader.config import ExecutionConfig
from cryptotrader.core.events import FillEvent, OrderEvent
from cryptotrader.core.types import Bar
from cryptotrader.execution.base import apply_costs

logger = logging.getLogger(__name__)


class PaperExecutionHandler:
    """Simulated fills against live prices (no exchange order is placed)."""

    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config

    async def execute(
        self,
        order: OrderEvent,
        reference_bar: Bar,
        fill_price: float | None = None,
    ) -> FillEvent:
        """Fill ``order`` instantly at ``fill_price`` or the reference close.

        In live mode there is no "next bar open" to fill against, so a market
        order fills at the latest closed price; stop/trailing exits fill at the
        explicit ``fill_price`` (the stop level) the engine supplies.
        """
        base_price = fill_price if fill_price is not None else reference_bar.close
        fill = apply_costs(order, base_price, self.config)
        logger.info(
            "[PAPER] %s %s %.6f @ %.2f (fee %.4f)",
            "EXIT" if order.is_exit else "ENTRY",
            order.side.name,
            fill.quantity,
            fill.fill_price,
            fill.fee,
        )
        return fill
