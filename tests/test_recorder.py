# tests/test_recorder.py
"""Live recorder: order-book metric maths + observation persistence round-trip."""

from __future__ import annotations

from datetime import datetime, timezone

from cryptotrader.data.recorder import _order_book_metrics
from cryptotrader.persistence import TradeStore


def test_order_book_metrics() -> None:
    ob = {"bids": [[100.0, 3.0], [99.0, 1.0]], "asks": [[101.0, 1.0], [102.0, 1.0]]}
    m = _order_book_metrics(ob, levels=20)
    assert m["mid_price"] == 100.5
    assert abs(m["spread_bps"] - (1.0 / 100.5 * 1e4)) < 1e-6
    # bid_vol 4 vs ask_vol 2 -> imbalance (4-2)/6 = +0.333
    assert abs(m["ob_imbalance"] - (2.0 / 6.0)) < 1e-9
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
        assert rows[-1]["cb_premium"] == 0.0004 or rows[0]["cb_premium"] == 0.0004
