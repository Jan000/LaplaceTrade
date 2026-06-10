# tests/test_live_execution.py
"""CCXTExecutionHandler protective-order logic, validated against a fake ccxt client."""

from __future__ import annotations

import pytest

from cryptotrader.config import ExchangeConfig, ExecutionConfig
from cryptotrader.core.types import Side
from cryptotrader.execution.live import CCXTExecutionHandler


class FakeClient:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.cancelled: list[tuple] = []
        self._n = 0

    async def create_order(self, symbol, type_, side, qty, price=None, params=None):
        self._n += 1
        self.created.append({"symbol": symbol, "type": type_, "side": side,
                             "qty": qty, "price": price, "params": params or {}})
        return {"id": f"o{self._n}", "average": price or 100.0, "filled": qty, "fee": {"cost": 0.1}}

    async def cancel_order(self, oid, symbol):
        self.cancelled.append((oid, symbol))


def _handler() -> tuple[CCXTExecutionHandler, FakeClient]:
    h = CCXTExecutionHandler(
        ExchangeConfig(id="binance", symbol="BTC/USDT", api_key="k", api_secret="s"),
        ExecutionConfig(),
    )
    fake = FakeClient()
    h._client = fake  # inject (so _ensure_client returns it; no network)
    return h, fake


@pytest.mark.asyncio
async def test_place_and_cancel_protective() -> None:
    h, fake = _handler()
    await h.place_protective("BTC/USDT", Side.LONG, 0.5, stop_price=90.0, take_profit=120.0)

    # A long is protected by SELL stop-loss + SELL take-profit.
    assert [c["side"] for c in fake.created] == ["sell", "sell"]
    assert fake.created[0]["params"].get("stopLossPrice") == 90.0
    assert fake.created[1]["params"].get("takeProfitPrice") == 120.0
    assert h._protective["BTC/USDT"] == ["o1", "o2"]

    # Cancelling clears the tracked ids and cancels both on the exchange.
    await h.cancel_protective("BTC/USDT")
    assert {c[0] for c in fake.cancelled} == {"o1", "o2"}
    assert "BTC/USDT" not in h._protective


@pytest.mark.asyncio
async def test_protective_short_closes_with_buy_and_replaces() -> None:
    h, fake = _handler()
    await h.place_protective("ETH/USDT", Side.SHORT, 1.0, stop_price=110.0)
    assert fake.created[0]["side"] == "buy"            # closing a short = buy
    # Re-placing cancels the previous protective order first (never stack).
    await h.place_protective("ETH/USDT", Side.SHORT, 1.0, stop_price=111.0)
    assert ("o1", "ETH/USDT") in fake.cancelled
