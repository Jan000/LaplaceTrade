# src/cryptotrader/live/engine.py
"""Asynchronous live trading engine.

Mirrors the backtester's decision logic on a *streaming* basis. It consumes the
same :class:`MarketEvent` interface (so the live data feed and the historical
replayer are interchangeable) and reuses the exact same
:class:`~cryptotrader.backtest.portfolio.Portfolio`, :class:`Strategy` and
:class:`RiskManager` — only the data source and the (async) execution handler
differ. That shared core is what makes "what you backtest is what you trade"
literally true.

Per closed bar
--------------
1. Manage any open position: stop / trailing-stop exit against the bar's range,
   else extend the favourable-excursion watermark.
2. Ask the strategy for a signal (incremental feature update under the hood).
3. Flat + signal -> sized entry; in-position + opposite signal -> exit.
4. Mark equity, persist (trade / equity), update shared state and broadcast.

The execution handler is awaited, so this same engine drives both the paper
handler (instant simulated fills) and the real ccxt handler (network I/O).
"""

from __future__ import annotations

import logging
from typing import Protocol

from cryptotrader.backtest.portfolio import Portfolio
from cryptotrader.config import Settings
from cryptotrader.core.events import FillEvent, MarketEvent, OrderEvent
from cryptotrader.core.interfaces import DataHandler, FeatureCalculator, RiskManager, Strategy
from cryptotrader.core.types import Bar, OrderType, Side
from cryptotrader.live.state import EngineState, StateBroadcaster
from cryptotrader.persistence import TradeStore

logger = logging.getLogger(__name__)

_MAX_CURVE_POINTS = 1000


class AsyncExecutionHandler(Protocol):
    """Async execution contract shared by the paper and ccxt live handlers."""

    async def execute(
        self, order: OrderEvent, reference_bar: Bar, fill_price: float | None = None
    ) -> FillEvent: ...


def _opposite(side: Side) -> Side:
    return Side.SHORT if side is Side.LONG else Side.LONG


class LiveTradingEngine:
    """Streaming counterpart of :class:`EventDrivenBacktester`."""

    def __init__(
        self,
        data_handler: DataHandler,
        feature_engine: FeatureCalculator,
        strategy: Strategy,
        risk_manager: RiskManager,
        execution_handler: AsyncExecutionHandler,
        settings: Settings,
        state: EngineState,
        broadcaster: StateBroadcaster | None = None,
        store: TradeStore | None = None,
        warmup_bars: list[Bar] | None = None,
        on_update=None,
    ) -> None:
        # When set, called on every state change instead of the broadcaster — lets a
        # controller aggregate several concurrent engines into one dashboard snapshot.
        self._on_update = on_update
        self._data = data_handler
        self._features = feature_engine
        # Recent history used to prime the feature engine so the strategy can predict
        # on the first live candle instead of after `warmup`×timeframe of dead air.
        self._warmup_bars = warmup_bars or []
        self._last_ts = None
        self._strategy = strategy
        self._risk = risk_manager
        self._exec = execution_handler
        self._settings = settings
        self._symbol = settings.exchange.symbol
        self._state = state
        self._broadcaster = broadcaster
        self._store = store
        self._run_id: int | None = None
        self._portfolio = Portfolio(settings.risk.account_equity, self._symbol)
        self._latest_atr: float = 0.0
        self._stop = False
        self._cooldown_bars = settings.risk.cooldown_bars
        self._cooldown_remaining = 0
        self._trades_seen = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        """Consume the live stream until cancelled or :meth:`stop` is called."""
        await self._init_state()
        self._state.status = "running"
        self._prime()
        self._publish()  # show price/equity immediately, before the first live candle
        try:
            # Act on the most recent CLOSED candle right away (the current signal),
            # then follow the live stream — de-duplicating by timestamp so the stream
            # re-emitting that same bar doesn't double-trade it.
            if self._warmup_bars and not self._stop:
                await self._on_bar(MarketEvent(self._warmup_bars[-1]))
                self._last_ts = self._warmup_bars[-1].timestamp
            async for event in self._data.stream():
                if self._stop:
                    break
                if self._last_ts is not None and event.bar.timestamp <= self._last_ts:
                    continue
                self._last_ts = event.bar.timestamp
                await self._on_bar(event)
        except Exception:  # pragma: no cover - defensive in live loop
            self._state.status = "error"
            logger.exception("Live engine crashed")
            raise
        finally:
            if self._state.status != "error":
                self._state.status = "stopped"
            self._publish()

    def stop(self) -> None:
        """Request a graceful stop after the current bar."""
        self._stop = True

    # ------------------------------------------------------------------ #
    # Per-bar loop
    # ------------------------------------------------------------------ #
    async def _init_state(self) -> None:
        self._state.mode = self._settings.mode.value
        self._state.symbol = self._symbol
        self._state.initial_equity = self._portfolio.initial_equity
        self._state.equity = self._portfolio.initial_equity
        self._state.realized_equity = self._portfolio.initial_equity
        if self._store is not None:
            self._run_id = await self._store.start_run(
                mode=self._settings.mode.value,
                symbol=self._symbol,
                exchange=self._settings.exchange.id,
                initial_equity=self._portfolio.initial_equity,
                config=self._settings.model_dump(mode="json"),
                environment=self._state.environment,
            )

    def _prime(self) -> None:
        """Warm the feature engine from recent history — no trading, just buffer fill.

        Feeds all but the most recent closed bar into the feature engine so that the
        immediate decision on the latest bar (and the first live bar) has a full
        backward window. Also seeds the displayed price/equity so the dashboard shows
        activity the instant the engine starts.
        """
        if not self._warmup_bars:
            return
        for bar in self._warmup_bars[:-1]:
            self._features.update(bar)
        last = self._warmup_bars[-1]
        self._latest_atr = self._current_atr()
        self._state.last_price = last.close
        self._state.equity = self._portfolio.equity(last.close)
        logger.info("Primed feature engine with %d warmup bars", len(self._warmup_bars))

    async def _on_bar(self, event: MarketEvent) -> None:
        bar = event.bar

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # (1) Manage an open position against this bar's range.
        if self._portfolio.has_position:
            await self._manage_position(bar)

        # A trade just closed -> start the post-trade cooldown before re-entry.
        if len(self._portfolio.trades) > self._trades_seen:
            self._trades_seen = len(self._portfolio.trades)
            self._cooldown_remaining = self._cooldown_bars

        # (2) Strategy signal (incremental feature update happens inside).
        signal = self._strategy.on_market(event)
        self._latest_atr = self._current_atr()

        # (3) Act on the signal.
        if signal is not None and signal.side is not Side.FLAT:
            if not self._portfolio.has_position and self._cooldown_remaining == 0:
                await self._try_enter(signal, bar)
            elif self._portfolio.has_position and signal.side is not self._portfolio.position_side:
                await self._exit(bar, reason="signal", fill_price=bar.close)

        # (4) Mark, persist, broadcast.
        self._portfolio.mark(bar.timestamp, bar.close)
        await self._sync_state(bar)

    async def _manage_position(self, bar: Bar) -> None:
        """Apply the triple barriers (stop-loss, take-profit, time) to this bar."""
        pos = self._portfolio.position
        assert pos is not None
        pos.bars_held += 1
        long = pos.side is Side.LONG

        stop_hit = (long and bar.low <= pos.stop_loss) or (not long and bar.high >= pos.stop_loss)
        if stop_hit:
            await self._exit(bar, reason="stop_loss", fill_price=pos.stop_loss)
            return

        tp_hit = (long and bar.high >= pos.take_profit) or (not long and bar.low <= pos.take_profit)
        if tp_hit:
            await self._exit(bar, reason="take_profit", fill_price=pos.take_profit)
            return

        if pos.max_hold_bars and pos.bars_held >= pos.max_hold_bars:
            await self._exit(bar, reason="time_exit", fill_price=bar.close)
            return

        self._portfolio.update_excursion(bar.high, bar.low)

    async def _try_enter(self, signal, bar: Bar) -> None:
        equity = self._portfolio.equity(bar.close)
        order = self._risk.size_order(
            signal, bar, self._latest_atr or self._current_atr(), equity,
            has_open_position=False,
        )
        if order is None:
            return
        fill = await self._exec.execute(order, bar, fill_price=bar.close)
        self._portfolio.open_position(
            fill, order.stop_distance, order.tp_distance, order.max_hold_bars
        )
        # Native exchange safety net: if the handler supports it (real ccxt orders),
        # place a protective stop-loss / take-profit so a stopped bot can't leave the
        # position unmanaged. Best-effort — never let it break the trading loop.
        pos = self._portfolio.position
        if pos is not None and hasattr(self._exec, "place_protective"):
            try:
                await self._exec.place_protective(
                    self._symbol, pos.side, pos.quantity, pos.stop_loss, pos.take_profit)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to place protective orders for %s", self._symbol)

    async def _exit(self, bar: Bar, reason: str, fill_price: float) -> None:
        pos = self._portfolio.position
        assert pos is not None
        # Cancel the native protective orders before the engine-driven market exit, so
        # the exchange's safety-net stop/TP can't also fire (double exit).
        if hasattr(self._exec, "cancel_protective"):
            try:
                await self._exec.cancel_protective(self._symbol)
            except Exception:  # pragma: no cover - defensive
                logger.warning("Could not cancel protective orders for %s", self._symbol)
        exit_order = OrderEvent(
            symbol=self._symbol,
            timestamp=bar.timestamp,
            side=_opposite(pos.side),
            quantity=pos.quantity,
            order_type=OrderType.MARKET,
            is_exit=True,
        )
        fill = await self._exec.execute(exit_order, bar, fill_price=fill_price)
        trade = self._portfolio.close_position(fill, exit_reason=reason)
        if self._store is not None and self._run_id is not None:
            await self._store.record_trade(self._run_id, trade)

    # ------------------------------------------------------------------ #
    # State sync
    # ------------------------------------------------------------------ #
    def _current_atr(self) -> float:
        """Latest ATR from the feature row the strategy just computed (no recompute).

        The strategy's live path already ran one incremental ``update`` per bar and
        cached the resulting row on the feature engine; we read ATR from there to
        avoid a second full transform over the buffer in the hot loop.
        """
        last = getattr(self._features, "last_features", None)
        if last is None or "atr" not in last:
            return self._latest_atr
        atr = float(last["atr"])
        return atr if atr == atr else self._latest_atr  # NaN guard

    async def _sync_state(self, bar: Bar) -> None:
        p = self._portfolio
        trades = p.trades
        wins = sum(1 for t in trades if t.net_pnl > 0)
        effs = [t.efficiency_ratio for t in trades]

        s = self._state
        s.last_price = bar.close
        s.equity = p.equity(bar.close)
        s.realized_equity = p.realized_equity
        s.unrealized_pnl = p.position.unrealized_pnl(bar.close) if p.position else 0.0
        s.n_trades = len(trades)
        s.win_rate = wins / len(trades) if trades else 0.0
        s.avg_efficiency_ratio = sum(effs) / len(effs) if effs else 0.0
        s.position = self._position_dict(bar)
        s.recent_trades = [self._trade_dict(t) for t in trades[-25:]][::-1]
        s.equity_curve = self._downsampled_curve()
        s.touch()

        if self._store is not None and self._run_id is not None:
            await self._store.record_equity(self._run_id, bar.timestamp, s.equity)
        self._publish()

    def _position_dict(self, bar: Bar) -> dict | None:
        pos = self._portfolio.position
        if pos is None:
            return None
        return {
            "side": pos.side.name,
            "quantity": round(pos.quantity, 6),
            "entry_price": round(pos.entry_price, 2),
            "stop": round(self._portfolio.trailing_stop(), 2),
            "unrealized_pnl": round(pos.unrealized_pnl(bar.close), 4),
        }

    @staticmethod
    def _trade_dict(t) -> dict:
        return {
            "side": t.side.name,
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat(),
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2),
            "net_pnl": round(t.net_pnl, 4),
            "exit_reason": t.exit_reason,
            "efficiency_ratio": round(t.efficiency_ratio, 4),
        }

    def _downsampled_curve(self) -> list[dict]:
        curve = self._portfolio.equity_curve
        if not curve:
            return []
        step = max(1, len(curve) // _MAX_CURVE_POINTS)
        return [
            {"t": ts.isoformat(), "equity": round(eq, 2)}
            for ts, eq in curve[::step]
        ]

    def _publish(self) -> None:
        if self._on_update is not None:
            self._on_update()
        elif self._broadcaster is not None:
            self._broadcaster.publish(self._state.snapshot())
