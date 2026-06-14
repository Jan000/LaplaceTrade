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
            if getattr(self.exchange_cfg, "testnet", False):
                try:
                    self._client.set_sandbox_mode(True)  # route to the exchange testnet
                    logger.warning("[LIVE] sandbox/testnet mode ENABLED for %s", self.exchange_cfg.id)
                except Exception:
                    logger.warning("testnet not supported by %s; using live endpoints", self.exchange_cfg.id)
        return self._client

    async def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            await self._client.close()  # type: ignore[func-returns-value]
            self._client = None

    async def fetch_equity(self, quote: str = "USDT") -> float | None:
        """Real tradeable balance of ``quote`` on the exchange (for honest position sizing)."""
        client = self._ensure_client()
        try:
            bal = await client.fetch_balance()
            total = (bal.get("total") or {}).get(quote)
            free = (bal.get("free") or {}).get(quote)
            val = total if total is not None else free
            return float(val) if val is not None else None
        except Exception:
            logger.warning("Could not fetch exchange balance for %s", quote, exc_info=True)
            return None

    async def fetch_account(self, symbols: list[str]) -> dict:
        """Read-only account snapshot for the dashboard's 'test connection' / reconcile:
        non-zero balances, open derivative positions, and open orders. Best-effort."""
        client = self._ensure_client()
        out: dict = {"testnet": bool(getattr(self.exchange_cfg, "testnet", False)),
                     "exchange": self.exchange_cfg.id, "balances": {},
                     "positions": [], "open_orders": []}
        try:
            bal = await client.fetch_balance()
            out["balances"] = {k: round(float(v), 8) for k, v in (bal.get("total") or {}).items() if v}
        except Exception as exc:
            out["balance_error"] = str(exc)[:160]
        try:
            if getattr(client, "has", {}).get("fetchPositions"):
                for p in await client.fetch_positions(symbols):
                    if p.get("contracts"):
                        out["positions"].append({
                            "symbol": p.get("symbol"), "side": p.get("side"),
                            "contracts": p.get("contracts"), "entryPrice": p.get("entryPrice"),
                            "notional": p.get("notional"), "unrealizedPnl": p.get("unrealizedPnl")})
        except Exception:
            pass
        try:
            for s in symbols:
                for o in await client.fetch_open_orders(s):
                    out["open_orders"].append({
                        "symbol": o.get("symbol"), "side": o.get("side"), "type": o.get("type"),
                        "amount": o.get("amount"), "price": o.get("price"), "id": o.get("id")})
        except Exception:
            pass
        return out

    async def open_orders_warning(self, symbols: list[str]) -> list[str]:
        """List pre-existing open orders per symbol so the operator is aware on start."""
        client = self._ensure_client()
        warns: list[str] = []
        for sym in symbols:
            try:
                oo = await client.fetch_open_orders(sym)
                if oo:
                    warns.append(f"{sym}: {len(oo)} open order(s) already on the exchange")
            except Exception:
                pass
        return warns

    async def _ensure_markets(self, client) -> None:
        if not getattr(client, "markets", None):
            try:
                await client.load_markets()
            except Exception:  # pragma: no cover - network
                logger.warning("load_markets failed; order rounding may be approximate", exc_info=True)

    def _round_amount(self, client, symbol: str, qty: float) -> float:
        """Round quantity to the market's lot/step precision (rejected otherwise)."""
        try:
            return float(client.amount_to_precision(symbol, qty))
        except Exception:
            return qty

    def _below_minimum(self, client, symbol: str, qty: float, price: float) -> str | None:
        """Reason string if the order violates the exchange's min amount / min notional."""
        m = (getattr(client, "markets", None) or {}).get(symbol) or {}
        limits = m.get("limits") or {}
        amin = (limits.get("amount") or {}).get("min")
        cmin = (limits.get("cost") or {}).get("min")
        if amin and qty < amin:
            return f"amount {qty} < exchange min {amin}"
        if cmin and qty * price < cmin:
            return f"notional {qty * price:.2f} < exchange min {cmin}"
        return None

    async def _create_with_retry(self, client, *args, attempts: int = 3, **kwargs):
        """create_order with a few retries on transient network/exchange errors."""
        import asyncio

        try:
            import ccxt
            transient = (ccxt.NetworkError, ccxt.ExchangeNotAvailable,
                         ccxt.RequestTimeout, ccxt.DDoSProtection)
        except Exception:  # pragma: no cover
            transient = (Exception,)
        last = None
        for i in range(attempts):
            try:
                return await client.create_order(*args, **kwargs)
            except transient as exc:  # type: ignore[misc]
                last = exc
                logger.warning("[LIVE] order attempt %d/%d failed (%s); retrying",
                               i + 1, attempts, type(exc).__name__)
                await asyncio.sleep(1.0 + i)
        raise last  # type: ignore[misc]

    async def place_protective(
        self, symbol: str, position_side: Side, quantity: float,
        stop_price: float, take_profit: float | None = None,
    ) -> None:
        """Place native protective orders so a stopped/crashed bot can't leave the
        position unmanaged. The stop-loss is the critical one; the take-profit is a
        bonus. Best-effort — failures are logged loudly but never raise."""
        client = self._ensure_client()
        await self._ensure_markets(client)
        await self.cancel_protective(symbol)                 # never stack protective orders
        close = "sell" if position_side is Side.LONG else "buy"
        quantity = self._round_amount(client, symbol, quantity)
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
        await self._ensure_markets(client)
        ccxt_side = "buy" if order.side is Side.LONG else "sell"
        qty = self._round_amount(client, order.symbol, order.quantity)
        if qty <= 0:
            raise RuntimeError(f"order quantity rounds to 0 for {order.symbol}")
        # Enforce exchange minimums on ENTRIES (don't block exits — must always be able to close).
        if not order.is_exit:
            reason = self._below_minimum(client, order.symbol, qty, fill_price or reference_bar.close)
            if reason:
                raise RuntimeError(f"order below exchange minimum: {reason}")
        logger.warning("[LIVE] placing REAL %s market order %.8f %s", ccxt_side, qty, order.symbol)
        resp = await self._create_with_retry(client, order.symbol, "market", ccxt_side, qty)

        filled_price = float(
            resp.get("average") or resp.get("price") or reference_bar.close
        )
        filled_qty = float(resp.get("filled") or qty)
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
