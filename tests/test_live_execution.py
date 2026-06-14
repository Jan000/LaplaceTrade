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
        self.markets = {"BTC/USDT": {"limits": {"amount": {"min": 0.0001},
                                                "cost": {"min": 10.0}}},
                        "ETH/USDT": {"limits": {"amount": {"min": 0.001}, "cost": {"min": 10.0}}}}

    async def load_markets(self):
        return self.markets

    def amount_to_precision(self, symbol, qty):
        return round(float(qty), 3)               # 3-decimal lot step

    async def create_order(self, symbol, type_, side, qty, price=None, params=None):
        self._n += 1
        self.created.append({"symbol": symbol, "type": type_, "side": side,
                             "qty": qty, "price": price, "params": params or {}})
        return {"id": f"o{self._n}", "average": price or 100.0, "filled": qty, "fee": {"cost": 0.1}}

    async def cancel_order(self, oid, symbol):
        self.cancelled.append((oid, symbol))

    has = {"fetchPositions": True}

    async def fetch_balance(self):
        return {"total": {"USDT": 1234.5, "BTC": 0.01}, "free": {"USDT": 1200.0}}

    async def fetch_positions(self, symbols):
        return [{"symbol": "BTC/USDT", "side": "long", "contracts": 0.01,
                 "entryPrice": 60000.0, "notional": 600.0, "unrealizedPnl": 5.0}]

    async def fetch_open_orders(self, symbol):
        return [{"symbol": symbol, "side": "sell", "type": "limit",
                 "amount": 0.01, "price": 65000.0, "id": "x1"}]


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


@pytest.mark.asyncio
async def test_execute_rounds_amount_and_enforces_minimum() -> None:
    from cryptotrader.core.events import OrderEvent
    from cryptotrader.core.types import Bar, OrderType
    from datetime import datetime, timezone

    h, fake = _handler()
    bar = Bar(datetime.now(tz=timezone.utc), 100.0, 101.0, 99.0, 100.0, 5.0)

    # Quantity rounded to the 3-decimal lot step before sending.
    o = OrderEvent("BTC/USDT", bar.timestamp, Side.LONG, 0.123456, OrderType.MARKET, is_exit=False)
    await h.execute(o, bar, fill_price=100.0)
    assert fake.created[-1]["qty"] == 0.123

    # An entry below the exchange min notional (0.001 * 100 = $0.1 < $10) is rejected.
    small = OrderEvent("BTC/USDT", bar.timestamp, Side.LONG, 0.001, OrderType.MARKET, is_exit=False)
    with pytest.raises(RuntimeError):
        await h.execute(small, bar, fill_price=100.0)


@pytest.mark.asyncio
async def test_create_with_retry(monkeypatch) -> None:
    import asyncio
    import ccxt

    async def _no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    h, _ = _handler()

    calls = {"n": 0}

    class Flaky:
        async def create_order(self, *a, **k):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ccxt.NetworkError("transient")
            return {"id": "ok"}

    r = await h._create_with_retry(Flaky(), "BTC/USDT", "market", "buy", 1.0)
    assert r["id"] == "ok" and calls["n"] == 3


@pytest.mark.asyncio
async def test_fetch_account() -> None:
    h, _ = _handler()
    acct = await h.fetch_account(["BTC/USDT"])
    assert acct["balances"]["USDT"] == 1234.5
    assert acct["positions"][0]["symbol"] == "BTC/USDT" and acct["positions"][0]["entryPrice"] == 60000.0
    assert acct["open_orders"][0]["id"] == "x1"
