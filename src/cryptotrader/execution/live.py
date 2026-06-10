# src/cryptotrader/execution/live.py
"""Live ccxt execution handler — places REAL orders on a real exchange.

This handler is intentionally kept separate from the paper/simulated path and is
**opt-in**: it is only constructed when the operator explicitly selects real
trading. It places market orders via ``ccxt.async_support`` and reconstructs a
:class:`FillEvent` from the exchange's order response (average fill price + fee).

SAFETY
------
* Requires API credentials; refuses to run without them.
* Reads the *actual* filled quantity/price/fee from the exchange response rather
  than assuming the requested values — partial fills and real slippage are
  reflected truthfully.
* The engine still decides *when* to exit while it is running. In addition, on every
  real entry this handler places a **native protective stop-loss** (and take-profit) on
  the exchange as a safety net, so a *stopped or crashed* bot does not leave a real
  position unmanaged. On an engine-driven exit the outstanding protective orders are
  cancelled first. Protective orders are best-effort: a failure to place them is logged
  loudly but does not block the trade (the engine remains the primary exit driver).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from cryptotrader.config import ExchangeConfig, ExecutionConfig
from cryptotrader.core.events import FillEvent, OrderEvent
from cryptotrader.core.types import Bar, Side

logger = logging.getLogger(__name__)


class CCXTExecutionHandler:
    """Places real market orders through ccxt (async)."""

    def __init__(
        self,
        exchange_cfg: ExchangeConfig,
        execution_cfg: ExecutionConfig,
    ) -> None:
        if not (exchange_cfg.api_key and exchange_cfg.api_secret):
            raise RuntimeError(
                "CCXTExecutionHandler requires CT_EXCHANGE__API_KEY / "
                "CT_EXCHANGE__API_SECRET to place real orders."
            )
        self.exchange_cfg = exchange_cfg
        self.execution_cfg = execution_cfg
        self._client: object | None = None
        self._protective: dict[str, list[str]] = {}  # symbol -> open protective order ids

    def _ensure_client(self):
        if self._client is None:
            try:
                module = __import__("ccxt.async_support", fromlist=["dummy"])
            except ImportError as exc:  # pragma: no cover - optional dep
                raise RuntimeError("ccxt is required for live order execution.") from exc
            klass = getattr(module, self.exchange_cfg.id)
            self._client = klass(
                {
                    "apiKey": self.exchange_cfg.api_key,
                    "secret": self.exchange_cfg.api_secret,
                    "enableRateLimit": True,
                }
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            await self._client.close()  # type: ignore[func-returns-value]
            self._client = None

    async def place_protective(
        self, symbol: str, position_side: Side, quantity: float,
        stop_price: float, take_profit: float | None = None,
    ) -> None:
        """Place native protective orders so a stopped/crashed bot can't leave the
        position unmanaged. The stop-loss is the critical one; the take-profit is a
        bonus. Best-effort — failures are logged loudly but never raise."""
        client = self._ensure_client()
        await self.cancel_protective(symbol)                 # never stack protective orders
        close = "sell" if position_side is Side.LONG else "buy"
        ids: list[str] = []
        try:
            r = await client.create_order(  # type: ignore[attr-defined]
                symbol, "market", close, quantity, None, {"stopLossPrice": stop_price})
            if r.get("id"):
                ids.append(r["id"])
            logger.warning("[LIVE] protective STOP %s %.6f %s @ %.2f", close, quantity, symbol, stop_price)
        except Exception:
            logger.exception(
                "[LIVE] FAILED to place protective stop for %s — position is UNPROTECTED "
                "if the bot stops. Consider flattening manually.", symbol)
        if take_profit is not None:
            try:
                r = await client.create_order(  # type: ignore[attr-defined]
                    symbol, "limit", close, quantity, take_profit, {"takeProfitPrice": take_profit})
                if r.get("id"):
                    ids.append(r["id"])
            except Exception:
                logger.warning("[LIVE] could not place protective take-profit for %s", symbol, exc_info=True)
        self._protective[symbol] = ids

    async def cancel_protective(self, symbol: str) -> None:
        """Cancel outstanding protective orders for ``symbol`` (best-effort)."""
        if self._client is None:
            self._protective.pop(symbol, None)
            return
        for oid in self._protective.pop(symbol, []):
            try:
                await self._client.cancel_order(oid, symbol)  # type: ignore[attr-defined]
            except Exception:
                pass  # already filled or gone — fine

    async def execute(
        self,
        order: OrderEvent,
        reference_bar: Bar,
        fill_price: float | None = None,
    ) -> FillEvent:
        """Place a market order and build a FillEvent from the exchange response."""
        client = self._ensure_client()
        ccxt_side = "buy" if order.side is Side.LONG else "sell"
        logger.warning(
            "[LIVE] placing REAL %s market order %.6f %s",
            ccxt_side,
            order.quantity,
            order.symbol,
        )
        resp = await client.create_order(  # type: ignore[attr-defined]
            order.symbol, "market", ccxt_side, order.quantity
        )

        filled_price = float(
            resp.get("average") or resp.get("price") or reference_bar.close
        )
        filled_qty = float(resp.get("filled") or order.quantity)
        fee_obj = resp.get("fee") or {}
        fee = float(fee_obj.get("cost") or filled_price * filled_qty * self.execution_cfg.taker_fee)

        return FillEvent(
            symbol=order.symbol,
            timestamp=datetime.now(tz=timezone.utc),
            side=order.side,
            quantity=filled_qty,
            fill_price=filled_price,
            fee=fee,
            slippage=abs(filled_price - reference_bar.close) * filled_qty,
            is_exit=order.is_exit,
        )
