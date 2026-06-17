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


def test_recorder_loop_survives_write_errors(monkeypatch) -> None:
    """A failing record_observation (e.g. DB locked) must NOT propagate out of the run loop —
    the recorder records the error and keeps cycling instead of dying silently."""
    import cryptotrader.data.recorder as drec

    r = drec.MarketRecorder(Settings(), ["BTC/USDT"], interval=0.01)

    async def fake_sample(_sym):
        return {"mid_price": 100.0}

    class BadStore:
        async def record_observation(self, *a, **k):
            raise RuntimeError("database is locked")

        async def close(self):
            pass

    class FakeTS:
        def __init__(self, *a, **k):
            pass

        async def connect(self):
            return BadStore()

    monkeypatch.setattr(r, "sample_symbol", fake_sample)
    monkeypatch.setattr(drec, "TradeStore", FakeTS)
    monkeypatch.setattr(r, "aclose", lambda: asyncio.sleep(0))

    async def main():
        task = asyncio.create_task(r.run())
        for _ in range(300):
            await asyncio.sleep(0)
            if r.cycles >= 1:
                break
        assert r.cycles >= 1                       # kept cycling despite the write error
        assert r.last_error and "locked" in r.last_error
        r.stop()
        await asyncio.wait_for(task, timeout=2)

    asyncio.run(main())


def test_supervisor_keeps_recorder_alive_after_crash(monkeypatch) -> None:
    """If the recorder run() raises, the controller keeps the supervised task alive and
    restarts it — the old code let one crash kill recording for good (shown 'inactive')."""
    import cryptotrader.api.recorder_control as rc
    import cryptotrader.data.recorder as drec

    calls = {"n": 0}

    class FakeRec:
        def __init__(self, *a, **k):
            self._stop = False

        def stop(self):
            self._stop = True

        async def run(self):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("boom")          # first run crashes
            while not self._stop:                   # then stay alive until stopped
                await asyncio.sleep(0.001)

    monkeypatch.setattr(drec, "MarketRecorder", FakeRec)

    async def main():
        real_sleep = asyncio.sleep
        monkeypatch.setattr(rc.asyncio, "sleep", lambda _d: real_sleep(0))  # instant backoff
        c = rc.RecorderController(Settings())
        await c.start(["BTC/USDT"], 1)
        for _ in range(500):
            await real_sleep(0)
            if calls["n"] >= 2:
                break
        assert c.is_running and calls["n"] >= 2     # survived the crash and restarted
        await c.stop()
        assert not c.is_running

    asyncio.run(main())


async def test_merge_recorded_is_leak_free(tmp_path) -> None:
    """Each bar gets the LAST observation within its interval (known at bar close)."""
    import pandas as pd
    from datetime import datetime, timedelta, timezone

    from cryptotrader.data.recorded import merge_recorded

    db = tmp_path / "rec.sqlite"
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    async with TradeStore(db) as st:
        await st.record_observation(base + timedelta(minutes=10), "BTC/USDT",
                                    ob_imbalance=0.1, spread_bps=2.0, taker_buy_ratio=0.6)
        await st.record_observation(base + timedelta(minutes=200), "BTC/USDT",   # later, same 4h bar
                                    ob_imbalance=0.3, spread_bps=2.5, imbalance_top5=0.25)
        await st.record_observation(base + timedelta(hours=4, minutes=5), "BTC/USDT",  # next bar
                                    ob_imbalance=-0.2, microprice_dev_bps=-1.0)
    idx = pd.date_range(base, periods=3, freq="4h", tz="UTC")
    ohlcv = pd.DataFrame({c: [1.0, 1.0, 1.0] for c in ("open", "high", "low", "close", "volume")},
                         index=idx)
    out = merge_recorded(ohlcv, db, "BTC/USDT", "4h")
    assert "obs_imbalance" in out.columns
    assert abs(out["obs_imbalance"].iloc[0] - 0.3) < 1e-9     # last obs in bar 0
    assert abs(out["obs_imbalance"].iloc[1] - (-0.2)) < 1e-9  # obs in bar 1


def test_recorded_features_present_and_external() -> None:
    from cryptotrader.data.features import MicrostructureFeatureEngine
    from cryptotrader.data.ingestion import make_synthetic_ohlcv

    fe = MicrostructureFeatureEngine(use_recorded=True)
    assert "rec_ob_imb" in fe.feature_names and "rec_taker" in fe.feature_names
    fe.set_external({"obs_imbalance": 0.5, "obs_taker": 0.7})   # live injection
    feats = fe.transform(make_synthetic_ohlcv(n=200, seed=1))
    assert abs(feats["rec_ob_imb"].iloc[-1] - 0.5) < 1e-9      # last row = current obs
    assert abs(feats["rec_taker"].iloc[-1] - 0.7) < 1e-9
