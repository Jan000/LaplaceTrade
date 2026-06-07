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

        model = client.get("/api/model").json()
        assert "active" in model and "exists" in model
        assert model["active"] in {"trained model", "momentum baseline"}

        stats = client.get("/api/stats").json()
        assert stats["n_trades"] == 0 and "profit_factor" in stats and "by_side" in stats
        assert client.get("/api/stats?run_id=all").json()["n_trades"] == 0
        assert client.get("/api/trades?run_id=all").json() == []


def test_real_orders_guardrail(tmp_path, monkeypatch) -> None:
    """REAL orders must be refused when no matching model exists for the symbol."""
    import cryptotrader.ml.registry as registry

    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")  # empty -> no model
    settings = Settings()
    settings.strategy.model_path = None
    settings.persistence.db_path = tmp_path / "g.sqlite"
    app = create_app(settings)
    with TestClient(app) as client:
        m = client.get("/api/model").json()
        assert m["exists"] is False and m["real_ok"] is False
        r = client.post("/api/start", json={"mode": "live", "real_orders": True})
        assert r.status_code == 400
        assert "no trained model" in r.json().get("error", "").lower()


def test_runs_keys_and_jobs(tmp_path, monkeypatch) -> None:
    import cryptotrader.config as cfg

    monkeypatch.setattr(cfg, "SECRETS_FILE", str(tmp_path / "secrets.yaml"))
    settings = Settings()
    settings.persistence.db_path = tmp_path / "api.sqlite"
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/api/runs").json() == []

        keys = client.get("/api/keys").json()
        assert keys["has_key"] is False and keys["has_secret"] is False

        r = client.post("/api/keys", json={"api_key": "abc", "api_secret": "xyz"})
        assert r.json()["status"] == "saved"
        keys2 = client.get("/api/keys").json()
        assert keys2["has_key"] is True and keys2["has_secret"] is True
        # secrets must never be echoed back, anywhere
        assert "abc" not in str(keys2) and "xyz" not in str(keys2)
        assert client.get("/api/config").json()["exchange"]["api_key"] in (None, "")

        client.delete("/api/keys")
        assert client.get("/api/keys").json()["has_key"] is False

        status = client.get("/api/job/status?kind=walkforward").json()
        assert status["kind"] == "walkforward" and "running" in status
        assert client.post("/api/job", json={"kind": "bogus"}).status_code == 400
