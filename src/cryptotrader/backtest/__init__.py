# src/cryptotrader/backtest/__init__.py
"""Event-driven backtesting: portfolio, performance metrics, and the engine."""

from cryptotrader.backtest.engine import BacktestResult, EventDrivenBacktester
from cryptotrader.backtest.metrics import PerformanceReport, compute_metrics
from cryptotrader.backtest.portfolio import Portfolio

__all__ = [
    "EventDrivenBacktester",
    "BacktestResult",
    "Portfolio",
    "PerformanceReport",
    "compute_metrics",
]
