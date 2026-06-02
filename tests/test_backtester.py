# tests/test_backtester.py
"""End-to-end tests for the event-driven backtester."""

from __future__ import annotations

from cryptotrader.backtest.engine import EventDrivenBacktester
from cryptotrader.config import Settings
from cryptotrader.data.features import MicrostructureFeatureEngine
from cryptotrader.data.ingestion import make_synthetic_ohlcv
from cryptotrader.execution.simulated import SimulatedExecutionHandler
from cryptotrader.ml.model import MomentumBaselinePredictor
from cryptotrader.risk.manager import ATRRiskManager
from cryptotrader.strategy.ml_strategy import MLStrategy


def _run(n: int = 4000) -> object:
    settings = Settings()  # defaults; no YAML needed
    ohlcv = make_synthetic_ohlcv(n=n, seed=11)
    fe = MicrostructureFeatureEngine()
    strategy = MLStrategy(MomentumBaselinePredictor(), settings.strategy, settings.exchange.symbol)
    backtester = EventDrivenBacktester(
        ohlcv=ohlcv,
        feature_engine=fe,
        strategy=strategy,
        risk_manager=ATRRiskManager(settings.risk),
        execution_handler=SimulatedExecutionHandler(settings.execution),
        settings=settings,
    )
    return backtester.run()


def test_backtest_runs_and_trades() -> None:
    result = _run()
    assert result.report.n_trades > 0
    assert len(result.equity_curve) > 0
    # No position should remain open after the forced end-of-data liquidation.
    assert all(t.exit_reason for t in result.trades)


def test_efficiency_ratio_bounds() -> None:
    """Efficiency must never exceed 1 (cannot capture more than the best move)."""
    result = _run()
    for trade in result.trades:
        assert trade.efficiency_ratio <= 1.0 + 1e-9


def test_equity_curve_monotonic_timestamps() -> None:
    result = _run()
    ts = result.equity_curve.index
    assert ts.is_monotonic_increasing
