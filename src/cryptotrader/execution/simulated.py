# src/cryptotrader/execution/simulated.py
"""Simulated execution handler with realistic costs.

Models the two dominant intraday frictions:

* **Slippage** — a fixed basis-point haircut applied *against* the trade
  direction (you buy a touch higher, sell a touch lower).
* **Fees** — exchange taker fee on notional.

Fills default to the *open of the reference (next) bar* to avoid look-ahead. For
stop/trailing exits the engine supplies an explicit ``fill_price`` (the stop
level), which is the conservative assumption that the stop fills exactly at its
trigger plus slippage.
"""

from __future__ import annotations

from cryptotrader.config import ExecutionConfig
from cryptotrader.core.events import FillEvent, OrderEvent
from cryptotrader.core.interfaces import ExecutionHandler
from cryptotrader.core.types import Bar
from cryptotrader.execution.base import apply_costs


class SimulatedExecutionHandler(ExecutionHandler):
    """Backtest execution with bps slippage and taker fees."""

    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config

    def execute(
        self,
        order: OrderEvent,
        reference_bar: Bar,
        fill_price: float | None = None,
    ) -> FillEvent:
        """Execute ``order`` and return the resulting :class:`FillEvent`.

        Parameters
        ----------
        order:
            The order to fill.
        reference_bar:
            Bar whose ``open`` is used as the base price for market fills.
        fill_price:
            Optional explicit base price (used for stop/trailing exits).
        """
        base_price = fill_price if fill_price is not None else reference_bar.open
        return apply_costs(order, base_price, self.config)
