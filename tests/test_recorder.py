# tests/test_recorder.py
"""Live recorder: order-book metric maths + observation persistence round-trip."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from cryptotrader.config import Settings
from cryptotrader.data.recorder import _order_book_metrics, _trade_flow_metrics
from cryptotrader.persistence import TradeStore


def test_order_book_metrics() -> None:
    ob = {"bids": [[100.0, 3.0], [99.0, 1.0]], "asks": [[101.0, 1.0], [102.0, 1.0]]}
    m = _order_book_metrics(ob)
    assert m["mid_price"] == 100.5
    assert abs(m["spread_bps"] - (1.0 / 100.5 * 1e4)) < 1e-6
    # bid_vol 4 vs ask_vol 2 -> imbalance (4-2)/6 = +0.333
    assert abs(m["ob_imbalance"] - (2.0 / 6.0)) < 1e-9
    assert m["best_bid"] == 100.0 and m["best_ask"] == 101.0
    assert m["depth_imbalance_1pct"] is not None       # both sides within ±1% of 100.5
    assert _order_book_metrics({"bids": [], "asks": []}) == {}


async def test_record_and_count_observations(tmp_path) -> None:
    async with TradeStore(tmp_path / "obs.sqlite") as store:
        now = datetime.now(tz=timezone.utc)
        await store.record_observation(now, "BTC/USDT", mid_price=50000.0,
                                       ob_imbalance=0.2, cb_premium=0.0004, spread_bps=1.5)
        await store.record_observation(now, "BTC/USDT", mid_price=50010.0, ob_imbalance=-0.1)
        await store.record_observation(now, "ETH/USDT", funding_rate=1e-5)
        assert await store.observation_count() == {"BTC/USDT": 2, "ETH/USDT": 1}
        rows = await store.get_observations("BTC/USDT")
        assert len(rows) == 2 and rows[0]["mid_price"] in (50000.0, 50010.0)
        assert any(r["cb_premium"] == 0.0004 for r in rows)
        # detailed payload kept verbatim in the metrics JSON column
        import json
        detailed = [json.loads(r["metrics"]) for r in rows if r["metrics"]]
        assert any("ob_imbalance" in d for d in detailed)


def test_trade_flow_metrics() -> None:
    trades = [{"side": "buy", "amount": 3.0}, {"side": "sell", "amount": 1.0}]
    m = _trade_flow_metrics(trades)
    assert m["taker_buy_ratio"] == 0.75 and m["trade_count"] == 2 and m["avg_trade_size"] == 2.0
    assert _trade_flow_metrics([]) == {}


async def test_recorder_controller_lifecycle(monkeypatch) -> None:
    """RecorderController starts/stops a background task (stubbed recorder, no network)."""
    from cryptotrader.api.recorder_control import RecorderController

    class StubRecorder:
        def __init__(self, *a, **k):
            self._stop = False

        def stop(self):
            self._stop = True

        async def run(self):
            while not self._stop:
                await asyncio.sleep(0.01)

    monkeypatch.setattr("cryptotrader.data.recorder.MarketRecorder", StubRecorder)
    c = RecorderController(Settings())
    assert not c.is_running
    await c.start(["BTC/USDT", "ETH/USDT"], interval=1.0)
    assert c.is_running and c.status()["symbols"] == ["BTC/USDT", "ETH/USDT"]
    await c.stop()
    assert not c.is_running and c.status()["running"] is False
