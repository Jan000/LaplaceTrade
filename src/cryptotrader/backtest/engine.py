# src/cryptotrader/backtest/engine.py
"""The event-driven backtester — the "time machine".

Design guarantees
-----------------
1. **No look-ahead.** Feature row ``t`` is built from bars ``<= t`` (backward
   windows only). A signal generated on the close of bar ``t`` is *executed at the
   open of bar ``t+1``*. Stop / trailing exits on bar ``t`` use levels that were
   fixed on bars ``< t`` (the favourable excursion is updated only *after* the
   stop check), so the same bar can never both raise the stop and dodge it.
2. **Realistic costs.** Every fill passes through the execution handler, which
   applies basis-point slippage and taker fees.
3. **Max Efficiency Ratio.** The portfolio tracks each trade's best favourable
   price; the report aggregates how much of that best-case move was captured.

Per-bar event sequence
----------------------
    open  -> (a) execute any order pending from the previous bar
          -> (b) manage open position: stop/trailing exit, then update excursion
    close -> (c) mark-to-market equity
          -> (d) ask strategy for a signal; size it; queue for next bar
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from cryptotrader.backtest.metrics import PerformanceReport, compute_metrics
from cryptotrader.backtest.portfolio import Portfolio
from cryptotrader.config import Settings
from cryptotrader.core.events import OrderEvent
from cryptotrader.core.interfaces import (
    ExecutionHandler,
    FeatureCalculator,
    RiskManager,
    Strategy,
)
from cryptotrader.core.types import Bar, OrderType, Side, Trade
from cryptotrader.data.ingestion import HistoricalDataHandler

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BacktestResult:
    """Everything produced by a backtest run."""

    report: PerformanceReport
    trades: list[Trade]
    equity_curve: pd.DataFrame  # index=timestamp, column="equity"
    features: pd.DataFrame


def _opposite(side: Side) -> Side:
    return Side.SHORT if side is Side.LONG else Side.LONG


class EventDrivenBacktester:
    """Chronological, look-ahead-free simulator wiring all modules together."""

    def __init__(
        self,
        ohlcv: pd.DataFrame,
        feature_engine: FeatureCalculator,
        strategy: Strategy,
        risk_manager: RiskManager,
        execution_handler: ExecutionHandler,
        settings: Settings,
    ) -> None:
        if not ohlcv.index.is_monotonic_increasing:
            ohlcv = ohlcv.sort_index()
        self._ohlcv = ohlcv
        self._feature_engine = feature_engine
        self._strategy = strategy
        self._risk = risk_manager
        self._exec = execution_handler
        self._settings = settings
        self._symbol = settings.exchange.symbol
        self._portfolio = Portfolio(settings.risk.account_equity, self._symbol)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> BacktestResult:
        """Execute the full backtest and return aggregated results."""
        features = self._feature_engine.transform(self._ohlcv)
        # Let the strategy precompute batch predictions for speed (backtest only).
        self._strategy.prepare(features)

        atr = features["atr"].to_numpy()
        bars = HistoricalDataHandler(self._ohlcv).bars
        pending: OrderEvent | None = None

        for i, bar in enumerate(bars):
            pending = self._step(i, bar, atr[i], pending)

        self._force_close_at_end(bars[-1] if bars else None)

        report = compute_metrics(
            self._portfolio.trades,
            self._portfolio.equity_curve,
            self._portfolio.initial_equity,
        )
        equity_df = pd.DataFrame(
            self._portfolio.equity_curve, columns=["timestamp", "equity"]
        ).set_index("timestamp")
        logger.info(
            "Backtest done: %d trades, final equity %.2f, avg efficiency %.2f%%",
            report.n_trades,
            report.final_equity,
            report.avg_efficiency_ratio * 100.0,
        )
        return BacktestResult(report, self._portfolio.trades, equity_df, features)

    # ------------------------------------------------------------------ #
    # One bar of the event loop
    # ------------------------------------------------------------------ #
    def _step(
        self, index: int, bar: Bar, atr_now: float, pending: OrderEvent | None
    ) -> OrderEvent | None:
        # (a) Execute the order queued on the previous bar, at this bar's open.
        if pending is not None:
            self._execute_pending(pending, bar)

        # (b) Manage an open position against this bar's range.
        if self._portfolio.has_position:
            self._manage_position(bar)

        # (c) Mark-to-market on the close.
        self._portfolio.mark(bar.timestamp, bar.close)

        # (d) Generate the next order from the strategy's signal.
        return self._generate_order(bar, atr_now)

    def _execute_pending(self, order: OrderEvent, bar: Bar) -> None:
        """Fill a queued order at ``bar.open`` and update the portfolio."""
        fill = self._exec.execute(order, bar)
        if order.is_exit:
            if self._portfolio.has_position:
                self._portfolio.close_position(fill, exit_reason="signal")
        else:
            self._portfolio.open_position(fill, order.stop_distance, order.trail_distance)

    def _manage_position(self, bar: Bar) -> None:
        """Apply stop / trailing-stop logic, then update the favourable excursion.

        The stop level is evaluated *before* this bar's extremes are folded into
        the MFE, so a single bar cannot both ratchet the trail and escape it.
        """
        pos = self._portfolio.position
        assert pos is not None
        stop = self._portfolio.trailing_stop()

        hit = (pos.side is Side.LONG and bar.low <= stop) or (
            pos.side is Side.SHORT and bar.high >= stop
        )
        if hit:
            reason = "trailing_stop" if stop != pos.stop_loss else "stop_loss"
            exit_order = OrderEvent(
                symbol=self._symbol,
                timestamp=bar.timestamp,
                side=_opposite(pos.side),
                quantity=pos.quantity,
                order_type=OrderType.MARKET,
                is_exit=True,
            )
            fill = self._exec.execute(exit_order, bar, fill_price=stop)
            self._portfolio.close_position(fill, exit_reason=reason)
            return

        # Not stopped: extend the excursion for use on subsequent bars.
        self._portfolio.update_excursion(bar.high, bar.low)

    def _generate_order(self, bar: Bar, atr_now: float) -> OrderEvent | None:
        """Translate a strategy signal into a queued order for the next bar."""
        from cryptotrader.core.events import MarketEvent  # local import avoids cycle

        signal = self._strategy.on_market(MarketEvent(bar))
        if signal is None or signal.side is Side.FLAT:
            return None

        if self._portfolio.has_position:
            # Opposite signal -> schedule a signal-based exit (reversals take the
            # subsequent bar to re-enter; conservative for an MVP).
            if signal.side is not self._portfolio.position_side:
                return OrderEvent(
                    symbol=self._symbol,
                    timestamp=bar.timestamp,
                    side=_opposite(self._portfolio.position_side),
                    quantity=self._portfolio.position.quantity,  # type: ignore[union-attr]
                    order_type=OrderType.MARKET,
                    is_exit=True,
                )
            return None  # same direction -> hold

        # Flat: size a fresh entry.
        equity = self._portfolio.equity(bar.close)
        return self._risk.size_order(
            signal, bar, atr_now, equity, has_open_position=False
        )

    def _force_close_at_end(self, last_bar: Bar | None) -> None:
        """Liquidate any residual position at the final close for clean accounting."""
        if last_bar is None or not self._portfolio.has_position:
            return
        pos = self._portfolio.position
        assert pos is not None
        exit_order = OrderEvent(
            symbol=self._symbol,
            timestamp=last_bar.timestamp,
            side=_opposite(pos.side),
            quantity=pos.quantity,
            order_type=OrderType.MARKET,
            is_exit=True,
        )
        fill = self._exec.execute(exit_order, last_bar, fill_price=last_bar.close)
        self._portfolio.close_position(fill, exit_reason="end_of_data")
