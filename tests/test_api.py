# tests/test_api.py
"""Smoke tests for the FastAPI control center via the in-process TestClient."""

from __future__ import annotations

from fastapi.testclient import TestClient

from cryptotrader.api.server import create_app
from cryptotrader.config import Settings


def test_dashboard_and_state(tmp_path) -> None:
    settings = Settings()
    settings.persistence.db_path = tmp_path / "api.sqlite"
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        state = client.get("/api/state").json()
        assert state["status"] in {"idle", "stopped"}
        assert client.get("/api/trades").json() == []
        assert client.get("/api/equity").json() == []
