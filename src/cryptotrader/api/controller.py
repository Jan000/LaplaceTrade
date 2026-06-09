# src/cryptotrader/api/controller.py
"""Engine lifecycle controller behind the dashboard.

Runs one :class:`LiveTradingEngine` per traded symbol concurrently (``data.trade_symbols``,
or just ``exchange.symbol``). Account equity is split equally across symbols, each symbol
uses its own per-symbol model, and the per-engine states are merged into a single
aggregate snapshot for the dashboard (totals + a per-symbol breakdown).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptotrader.config import RunMode, Settings
from cryptotrader.data.features import MicrostructureFeatureEngine
from cryptotrader.data.ingestion import (
    HistoricalDataHandler,
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

_MAX_CURVE = 1000


class EngineController:
    """Starts/stops one engine per traded symbol and exposes an aggregate snapshot."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = EngineState(symbol=settings.exchange.symbol)  # idle fallback
        self.broadcaster = StateBroadcaster()
        self._engines: list[LiveTradingEngine] = []
        self._states: dict[str, EngineState] = {}
        self._tasks: list[asyncio.Task] = []
        self._stores: list[TradeStore] = []
        self._combined_curve: list[dict] = []
        self._mode = "simulation"
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return any(not t.done() for t in self._tasks)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def trade_symbols(self) -> list[str]:
        syms = list(dict.fromkeys(self.settings.data.trade_symbols or []))
        return syms or [self.settings.exchange.symbol]

    async def start(self, mode: str = "simulation", real_orders: bool = False,
                    sim_days: int | None = None) -> None:
        """Start an engine per traded symbol (idempotent).

        ``sim_days`` (simulation only): replay the last N days of REAL data instead of the
        held-out slice — a quick accelerated test of the model on recent market history.
        """
        async with self._lock:
            if self.is_running:
                return
            symbols = self.trade_symbols()

            # Real orders: keep only symbols with a verified matching model; refuse if none.
            if mode == "live" and real_orders:
                ok, bad = [], []
                for sym in symbols:
                    reason = self._real_order_block_reason(sym)
                    (bad if reason else ok).append((sym, reason))
                if not ok:
                    raise RuntimeError(
                        "Refusing REAL orders — no symbol has a matching trained model. "
                        + "; ".join(f"{s}: {r}" for s, r in bad)
                    )
                if bad:
                    logger.warning("Skipping real-order symbols without a model: %s",
                                   ", ".join(s for s, _ in bad))
                symbols = [s for s, _ in ok]

            equity_each = self.settings.risk.account_equity / max(1, len(symbols))
            environment = ("live" if real_orders else "paper") if mode == "live" else "simulation"
            self._mode = mode
            self._engines, self._states, self._tasks, self._stores = [], {}, [], []
            self._combined_curve = []

            for sym in symbols:
                await self._start_one(sym, mode, real_orders, equity_each, environment, sim_days)

            logger.info("Started %d engine(s) in %s mode: %s",
                        len(self._engines), mode, ", ".join(symbols))
            self._publish_aggregate()

    async def _start_one(self, symbol, mode, real_orders, equity_each, environment,
                         sim_days=None) -> None:
        sub = self.settings.model_copy(deep=True)
        sub.exchange.symbol = symbol
        sub.risk.account_equity = equity_each
        sub.mode = RunMode.LIVE if mode == "live" else RunMode.BACKTEST
        if sim_days:                       # quick "test on the last N days of real data"
            sub.data.sim_days = sim_days
            sub.data.replay_file = None
            sub.data.sim_source = "recent"  # forces recent real window, skipping the holdout slice

        feature_engine = MicrostructureFeatureEngine(**sub.features.model_dump())
        predictor = self._build_predictor(sub)
        strategy = MLStrategy(predictor, sub.strategy, symbol, feature_engine=feature_engine)
        risk = ATRRiskManager(sub.risk, sub.barriers, sub.execution)

        if mode == "live":
            feed = MarketDataFeed(
                exchange_id=sub.exchange.id, symbol=symbol, timeframe=sub.exchange.timeframe,
                api_key=sub.exchange.api_key, api_secret=sub.exchange.api_secret,
            )
            data_handler = LiveDataHandler(feed)
            execution = self._build_execution(sub, real_orders)
            warmup_bars = await self._fetch_warmup(sub, feature_engine)
        else:
            ohlcv = await self._simulation_ohlcv(sub)
            data_handler = ReplayDataHandler(ohlcv, delay=0.01)
            execution = PaperExecutionHandler(sub.execution)
            warmup_bars = []

        store = None
        try:
            store = await TradeStore(sub.persistence.db_path).connect()
        except Exception:
            logger.exception("TradeStore unavailable for %s; running without persistence.", symbol)

        state = EngineState(symbol=symbol)
        state.mode = mode
        state.environment = environment
        self._states[symbol] = state
        if store is not None:
            self._stores.append(store)

        engine = LiveTradingEngine(
            data_handler=data_handler, feature_engine=feature_engine, strategy=strategy,
            risk_manager=risk, execution_handler=execution, settings=sub, state=state,
            store=store, warmup_bars=warmup_bars, on_update=self._publish_aggregate,
        )
        self._engines.append(engine)
        self._tasks.append(asyncio.create_task(self._run_guarded(engine)))

    async def _run_guarded(self, engine: LiveTradingEngine) -> None:
        try:
            await engine.run()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Engine crashed")
        finally:
            self._publish_aggregate()

    async def stop(self) -> None:
        """Request a graceful stop of all engines and await termination."""
        async with self._lock:
            for e in self._engines:
                e.stop()
            for t in self._tasks:
                try:
                    await asyncio.wait_for(asyncio.shield(t), timeout=12.0)
                except (asyncio.TimeoutError, Exception):  # pragma: no cover
                    t.cancel()
            for store in self._stores:
                try:
                    await store.close()
                except Exception:  # pragma: no cover
                    pass
            self._tasks, self._engines, self._stores = [], [], []
            for st in self._states.values():
                st.status = "stopped"
            self.state.status = "stopped"
            self._publish_aggregate()

    # ------------------------------------------------------------------ #
    # Aggregate snapshot
    # ------------------------------------------------------------------ #
    def _publish_aggregate(self) -> None:
        self.broadcaster.publish(self.snapshot())

    def snapshot(self) -> dict:
        states = list(self._states.values())
        if not states:
            return self.state.snapshot()
        total_init = sum(s.initial_equity for s in states) or 1.0
        total_eq = sum(s.equity for s in states)
        n_trades = sum(s.n_trades for s in states)
        wins = sum(round(s.win_rate * s.n_trades) for s in states)
        effs_w = sum(s.avg_efficiency_ratio * s.n_trades for s in states)
        statuses = {s.status for s in states}
        status = ("running" if "running" in statuses else
                  "error" if "error" in statuses else
                  "stopped" if "stopped" in statuses else "idle")
        if status == "running" and total_eq:
            now = datetime.now(tz=timezone.utc).isoformat()
            if not self._combined_curve or self._combined_curve[-1]["t"] != now:
                self._combined_curve.append({"t": now, "equity": round(total_eq, 2)})
                if len(self._combined_curve) > _MAX_CURVE:
                    self._combined_curve = self._combined_curve[-_MAX_CURVE:]

        trades = []
        for s in states:
            for t in s.recent_trades:
                trades.append({**t, "symbol": s.symbol})
        trades.sort(key=lambda t: t.get("exit_time", ""), reverse=True)

        per_symbol = [{
            "symbol": s.symbol, "status": s.status, "equity": round(s.equity, 2),
            "total_return_pct": round((s.equity / s.initial_equity - 1) * 100, 4)
                                if s.initial_equity else 0.0,
            "last_price": s.last_price, "n_trades": s.n_trades,
            "win_rate": s.win_rate, "unrealized_pnl": round(s.unrealized_pnl, 4),
            "avg_efficiency_ratio": s.avg_efficiency_ratio, "position": s.position,
        } for s in states]

        single = states[0] if len(states) == 1 else None
        return {
            "mode": states[0].mode, "environment": states[0].environment,
            "symbol": states[0].symbol if single else f"{len(states)} symbols",
            "status": status,
            "initial_equity": round(total_init, 2), "equity": round(total_eq, 2),
            "realized_equity": round(sum(s.realized_equity for s in states), 2),
            "unrealized_pnl": round(sum(s.unrealized_pnl for s in states), 4),
            "total_return_pct": round((total_eq / total_init - 1) * 100, 4),
            "last_price": single.last_price if single else 0.0,
            "n_trades": n_trades,
            "win_rate": round(wins / n_trades, 4) if n_trades else 0.0,
            "avg_efficiency_ratio": round(effs_w / n_trades, 4) if n_trades else 0.0,
            "position": single.position if single else None,
            "recent_trades": trades[:25],
            "equity_curve": (single.equity_curve if single else self._combined_curve),
            "symbols": per_symbol,
        }

    # ------------------------------------------------------------------ #
    # Builders / helpers (per-symbol settings)
    # ------------------------------------------------------------------ #
    def _build_execution(self, settings: Settings, real_orders: bool):
        if real_orders:
            from cryptotrader.execution.live import CCXTExecutionHandler

            return CCXTExecutionHandler(settings.exchange, settings.execution)
        return PaperExecutionHandler(settings.execution)

    def _build_predictor(self, settings: Settings):
        from cryptotrader.ml.registry import resolve_model

        path, _meta = resolve_model(settings)
        if path is not None:
            from cryptotrader.ml.meta import load_predictor

            logger.info("Loading model %s for %s", path, settings.exchange.symbol)
            return load_predictor(path)
        logger.info("No trained model for %s; using momentum baseline.", settings.exchange.symbol)
        return MomentumBaselinePredictor()

    def _real_order_block_reason(self, symbol: str) -> str | None:
        """Return a reason string if real orders for ``symbol`` must be refused, else None."""
        from cryptotrader.ml.registry import resolve_model

        sub = self.settings.model_copy(deep=True)
        sub.exchange.symbol = symbol
        path, meta = resolve_model(sub)
        tf = sub.exchange.timeframe
        if path is None:
            return "no trained model"
        if not meta:
            return "model has no metadata"
        if meta.get("symbol") != symbol:
            return f"model is for {meta.get('symbol')}"
        if meta.get("timeframe") != tf:
            return f"model timeframe {meta.get('timeframe')} != {tf}"
        return None

    async def _fetch_warmup(self, settings: Settings, feature_engine) -> list:
        """Recent closed candles to prime the feature engine (best-effort)."""
        need = getattr(feature_engine, "warmup", 120) + 10
        feed = MarketDataFeed(
            exchange_id=settings.exchange.id, symbol=settings.exchange.symbol,
            timeframe=settings.exchange.timeframe, cache_dir=None,
            api_key=settings.exchange.api_key, api_secret=settings.exchange.api_secret,
        )
        try:
            start = datetime.now(tz=timezone.utc) - timedelta(milliseconds=feed._tf_ms * (need + 5))
            hist = await feed.fetch_history(start, use_cache=False)
            if hist.empty:
                return []
            bars = HistoricalDataHandler(hist).bars
            if len(bars) > 1:
                bars = bars[:-1]  # drop the possibly-forming last candle
            return bars[-need:]
        except Exception:
            logger.exception("Live warmup history unavailable for %s; starting cold.",
                             settings.exchange.symbol)
            return []
        finally:
            await feed.close()

    async def _simulation_ohlcv(self, settings: Settings):
        """Accelerated-replay OHLCV: replay_file → per-symbol held-out → recent real → synthetic."""
        import pandas as pd

        from cryptotrader.ml.registry import holdout_path_for

        # "recent": test the model on the last N days of real data. Fetch N days PLUS the
        # feature warm-up lead-in, so warm-up bars (which never trade — features still NaN)
        # are consumed first and all trades fall inside the requested window.
        if settings.data.sim_source == "recent":
            import math

            from cryptotrader.data.ingestion import _TIMEFRAME_MS

            tf_ms = _TIMEFRAME_MS.get(settings.exchange.timeframe, 4 * 3_600_000)
            warmup = MicrostructureFeatureEngine(**settings.features.model_dump()).warmup
            warmup_days = math.ceil(warmup * tf_ms / 86_400_000) + 2
            feed = MarketDataFeed(
                exchange_id=settings.exchange.id, symbol=settings.exchange.symbol,
                timeframe=settings.exchange.timeframe, cache_dir=settings.data.cache_dir,
            )
            try:
                start = datetime.now(tz=timezone.utc) - timedelta(
                    days=settings.data.sim_days + warmup_days)
                df = await feed.fetch_history(start)
                logger.info("Simulation: replaying last %d days of %s (+%d warm-up days, %d bars)",
                            settings.data.sim_days, settings.exchange.symbol, warmup_days, len(df))
                if not df.empty:
                    return df
            except Exception:
                logger.exception("Simulation recent-window fetch failed for %s; using synthetic.",
                                 settings.exchange.symbol)
            finally:
                await feed.close()
            return make_synthetic_ohlcv(n=6000, seed=7)

        replay = settings.data.replay_file
        if replay is not None and Path(replay).exists():
            return pd.read_parquet(replay)
        per_symbol = holdout_path_for(settings.exchange.symbol)
        if per_symbol.exists():
            logger.info("Simulation: replaying held-out slice %s", per_symbol)
            return pd.read_parquet(per_symbol)
        if settings.data.sim_source != "synthetic":
            feed = MarketDataFeed(
                exchange_id=settings.exchange.id, symbol=settings.exchange.symbol,
                timeframe=settings.exchange.timeframe, cache_dir=settings.data.cache_dir,
            )
            try:
                start = datetime.now(tz=timezone.utc) - timedelta(days=settings.data.sim_days)
                df = await feed.fetch_history(start)
                if not df.empty:
                    return df
            except Exception:
                logger.exception("Simulation real-data fetch failed for %s; using synthetic.",
                                 settings.exchange.symbol)
            finally:
                await feed.close()
        return make_synthetic_ohlcv(n=6000, seed=7)
