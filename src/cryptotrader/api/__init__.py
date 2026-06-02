# src/cryptotrader/api/__init__.py
"""FastAPI control center: REST + WebSocket backend and a single-file dashboard."""

from cryptotrader.api.server import create_app

__all__ = ["create_app"]
