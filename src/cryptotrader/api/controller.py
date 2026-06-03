# src/cryptotrader/api/controller.py
"""Engine lifecycle controller behind the dashboard."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from cryptotrader.config import RunMode, Settings
from cryptotrader.data.features import MicrostructureFeatureEngine
from cryptotrader.data.ingestion import (
    LiveDataHandler,
    MarketDataFeed,
    ReplayDataHandler,
    make_synthetic_ohlcv,
)
from cryptotrader.execution.paper import PaperExecutionHandler
from cryptotrader.live.engine import LiveTradingEngine
from cryptotrader.live.state import EngineState, StateBroadcaster
from cryptotrader.ml.model import MomentumBaselinePredictor
from cryptotrader.persistence import TradeStore
from cryptotrader.risk.manager import ATRRiskManager
from cryptotrader.strategy.ml_strategy import MLStrategy

logger = logging.getLogger(__name__)


class EngineController:
    """Starts/stops the live engine and exposes shared state to the API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = EngineState(symbol=settings.exchange.symbol)
        self.broadcaster = StateBroadcaster()
        self._engine: LiveTradingEngine | None = None
        self._task: asyncio.Task | None = None
        self._store: TradeStore | None = None
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, mode: str = "simulation", real_orders: bool = False) -> None:
        """Start the engine in ``simulation`` or ``live`` mode (idempotent)."""
        async with self._lock:
            if self.is_running:
                return

            feature_engine = self._build_feature_engine()
            predictor = self._build_predictor()
            strategy = MLStrategy(
                predictor, self.settings.strategy, self.settings.exchange.symbol,
                feature_engine=feature_engine,
            )
            risk = ATRRiskManager(self.settings.risk, self.settings.barriers, self.settings.execution)

            if mode == "live":
                feed = MarketDataFeed(
                    exchange_id=self.settings.exchange.id,
                    symbol=self.settings.exchange.symbol,
                    timeframe=self.settings.exchange.timeframe,
                    api_key=self.settings.exchange.api_key,
                    api_secret=self.settings.exchange.api_secret,
                )
                data_handler = LiveDataHandler(feed)
                execution = self._build_execution(real_orders)
                self.settings.mode = RunMode.LIVE
            else:
                ohlcv = self._simulation_ohlcv()
                data_handler = ReplayDataHandler(ohlcv, delay=0.01)
                execution = PaperExecutionHandler(self.settings.execution)
                self.settings.mode = RunMode.BACKTEST

            # Persistence is best-effort: if the DB can't be opened (e.g. a
            # read-only or network filesystem that rejects SQLite WAL), the
            # engine still runs and the dashboard still works — we simply don't
            # log to disk. This keeps "press Start" from ever silently failing.
            self._store = None
            try:
                self._store = await TradeStore(
                    self.settings.persistence.db_path
                ).connect()
            except Exception:
                logger.exception(
                    "TradeStore unavailable (%s); running without persistence.",
                    self.settings.persistence.db_path,
                )

            self.state = EngineState(symbol=self.settings.exchange.symbol)
            self.state.mode = mode
            self._engine = LiveTradingEngine(
                data_handler=data_handler,
                feature_engine=feature_engine,
                strategy=strategy,
                risk_manager=risk,
                execution_handler=execution,
                settings=self.settings,
                state=self.state,
                broadcaster=self.broadcaster,
                store=self._store,
            )
            self._task = asyncio.create_task(self._run_guarded())
            logger.info("Engine started in %s mode", mode)

    async def _run_guarded(self) -> None:
        try:
            assert self._engine is not None
            await self._engine.run()
        finally:
            if self._store is not None:
                await self._store.close()
                self._store = None

    async def stop(self) -> None:
        """Request a graceful stop and await termination."""
        async with self._lock:
            if self._engine is not None:
                self._engine.stop()
            if self._task is not None:
                try:
                    await asyncio.wait_for(self._task, timeout=10.0)
                except asyncio.TimeoutError:  # pragma: no cover
                    self._task.cancel()
                self._task = None
            self.state.status = "stopped"

    def _build_feature_engine(self) -> MicrostructureFeatureEngine:
        return MicrostructureFeatureEngine(**self.settings.features.model_dump())

    def _build_execution(self, real_orders: bool):
        if real_orders:
            from cryptotrader.execution.live import CCXTExecutionHandler

            return CCXTExecutionHandler(self.settings.exchange, self.settings.execution)
        return PaperExecutionHandler(self.settings.execution)

    def _build_predictor(self):
        """Load the trained LightGBM model if configured, else the baseline."""
        path = self.settings.strategy.model_path
        if path is not None and Path(path).exists():
            from cryptotrader.ml.model import LightGBMPredictor

            logger.info("Loading trained model from %s", path)
            return LightGBMPredictor().load(path)
        logger.info("No trained model configured; using momentum baseline.")
        return MomentumBaselinePredictor()

    def _simulation_ohlcv(self):
        """Real held-out data for replay if configured, else synthetic."""
        replay = self.settings.data.replay_file
        if replay is not None and Path(replay).exists():
            import pandas as pd

            logger.info("Replaying real data from %s", replay)
            return pd.read_parquet(replay)
        return make_synthetic_ohlcv(n=6000, seed=7)
