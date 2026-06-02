# tests/test_live_engine.py
"""Tests for the async live engine and paper execution handler.

The engine is driven by a ReplayDataHandler (no network), so this also exercises
the full live wiring: data feed -> features -> strategy -> risk -> paper fills ->
portfolio -> persistence -> shared state.

These tests drive the async code via ``asyncio.run`` inside synchronous test
functions, keeping them independent of any ambient event-loop policy.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from cryptotrader.config import Settings
from cryptotrader.core.events import OrderEvent
from cryptotrader.core.types import Bar, Side
from cryptotrader.data.features import MicrostructureFeatureEngine
from cryptotrader.data.ingestion import ReplayDataHandler, make_synthetic_ohlcv
from cryptotrader.execution.paper import PaperExecutionHandler
from cryptotrader.live.engine import LiveTradingEngine
from cryptotrader.live.state import EngineState, StateBroadcaster
from cryptotrader.ml.model import MomentumBaselinePredictor
from cryptotrader.persistence import TradeStore
from cryptotrader.risk.manager import ATRRiskManager
from cryptotrader.strategy.ml_strategy import MLStrategy


def _strategy(settings, fe):
    return MLStrategy(
        MomentumBaselinePredictor(), settings.strategy, settings.exchange.symbol,
        feature_engine=fe,
    )


def test_paper_execution_costs_hurt():
    async def run():
        settings = Settings()
        handler = PaperExecutionHandler(settings.execution)
        bar = Bar(datetime(2024, 1, 1, tzinfo=timezone.utc), 100, 101, 99, 100, 10)
        buy = OrderEvent("BTC/USDT", bar.timestamp, Side.LONG, 1.0)
        fill = await handler.execute(buy, bar)
        assert fill.fill_price > 100.0
        assert fill.fee > 0.0

    asyncio.run(run())


def test_live_engine_runs_and_persists(tmp_path):
    async def run():
        settings = Settings()
        ohlcv = make_synthetic_ohlcv(n=160, seed=11)
        fe = MicrostructureFeatureEngine()
        state = EngineState()
        broadcaster = StateBroadcaster()
        q = broadcaster.subscribe(maxsize=10_000)

        store = await TradeStore(tmp_path / "live.sqlite").connect()
        try:
            engine = LiveTradingEngine(
                data_handler=ReplayDataHandler(ohlcv, delay=0.0),
                feature_engine=fe,
                strategy=_strategy(settings, fe),
                risk_manager=ATRRiskManager(settings.risk),
                execution_handler=PaperExecutionHandler(settings.execution),
                settings=settings,
                state=state,
                broadcaster=broadcaster,
                store=store,
            )
            await engine.run()
            run_id = await store.latest_run_id()
            trades = await store.get_trades(run_id)
        finally:
            await store.close()

        received = []
        while not q.empty():
            received.append(q.get_nowait())

        assert state.status == "stopped"
        assert state.n_trades > 0
        assert len(trades) == state.n_trades
        assert len(received) > 0
        for t in trades:
            assert t["efficiency_ratio"] <= 1.0 + 1e-9

    asyncio.run(run())


def test_engine_stop_is_graceful():
    async def run():
        settings = Settings()
        ohlcv = make_synthetic_ohlcv(n=120, seed=5)
        fe = MicrostructureFeatureEngine()
        engine = LiveTradingEngine(
            data_handler=ReplayDataHandler(ohlcv, delay=0.0),
            feature_engine=fe,
            strategy=_strategy(settings, fe),
            risk_manager=ATRRiskManager(settings.risk),
            execution_handler=PaperExecutionHandler(settings.execution),
            settings=settings,
            state=EngineState(),
            broadcaster=None,
            store=None,
        )
        engine.stop()
        await engine.run()

    asyncio.run(run())
