# src/cryptotrader/execution/__init__.py
"""Order execution handlers (simulated for backtests, ccxt for live)."""

from cryptotrader.execution.simulated import SimulatedExecutionHandler

__all__ = ["SimulatedExecutionHandler"]
