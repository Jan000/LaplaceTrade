# src/cryptotrader/persistence/__init__.py
"""Async SQLite persistence: trade logging, equity snapshots, feature storage."""

from cryptotrader.persistence.database import TradeStore

__all__ = ["TradeStore"]
