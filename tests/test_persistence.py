# tests/test_persistence.py
"""Tests for the async SQLite persistence layer."""

from __future__ import annotations

from datetime import datetime, timezone

from cryptotrader.core.types import Side, Trade
from cryptotrader.persistence import TradeStore


def _trade() -> Trade:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return Trade(
        symbol="BTC/USDT", side=Side.LONG, quantity=0.5,
        entry_time=now, entry_price=30_000.0,
        exit_time=now, exit_price=30_300.0,
        fees=1.2, gross_pnl=150.0, net_pnl=148.8,
        best_price=30_450.0, exit_reason="trailing_stop", efficiency_ratio=0.66,
    )


async def test_roundtrip_trade_and_equity(tmp_path) -> None:
    async with TradeStore(tmp_path / "t.sqlite") as store:
        run_id = await store.start_run(
            mode="paper", symbol="BTC/USDT", exchange="binance", initial_equity=10_000.0,
        )
        await store.record_trade(run_id, _trade())
        await store.record_equity(run_id, datetime.now(tz=timezone.utc), 10_148.8)
        await store.record_features(run_id, datetime.now(tz=timezone.utc), {"atr": 12.3})

        assert await store.latest_run_id() == run_id
        trades = await store.get_trades(run_id)
        assert len(trades) == 1
        assert trades[0]["exit_reason"] == "trailing_stop"
        assert abs(trades[0]["efficiency_ratio"] - 0.66) < 1e-9
        curve = await store.get_equity_curve(run_id)
        assert len(curve) == 1 and curve[0]["equity"] == 10_148.8


async def test_latest_run_none_on_empty(tmp_path) -> None:
    async with TradeStore(tmp_path / "empty.sqlite") as store:
        assert await store.latest_run_id() is None
        assert await store.get_trades(0) == []
