# src/cryptotrader/strategy/__init__.py
"""Trading strategies: the bridge between the ML engine and the event loop."""

from cryptotrader.strategy.ml_strategy import MLStrategy

__all__ = ["MLStrategy"]
