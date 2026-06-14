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

        jobs = client.get("/api/jobs").json()
        assert jobs["jobs"] == [] and jobs["any_running"] is False
        assert client.post("/api/job", json={"kind": "bogus"}).status_code == 400


def test_experiments_log_and_endpoint(tmp_path, monkeypatch) -> None:
    """Experiment records round-trip and /api/experiments serves them (newest first)."""
    import cryptotrader.ml.registry as registry
    from cryptotrader.ml.experiments import log_experiment, read_experiments

    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    s = Settings()
    log_experiment("walkforward", "BTC/USDT", s, {"compounded_return_pct": 11.1, "verdict": "ROBUST"})
    log_experiment("train", "ETH/USDT", s, {"val_accuracy": 0.51})
    rows = read_experiments()
    assert len(rows) == 2 and rows[0]["symbol"] == "ETH/USDT"            # newest first
    assert "use_calibration" in rows[0]["config"] and "label_method" in rows[0]["config"]

    settings = Settings()
    settings.persistence.db_path = tmp_path / "x.sqlite"
    app = create_app(settings)
    with TestClient(app) as client:
        served = client.get("/api/experiments").json()
        assert isinstance(served, list) and served[0]["kind"] == "train"


def test_observation_series_endpoint(tmp_path) -> None:
    """/api/observations/series returns the ASC time series + per-metric summary stats."""
    import asyncio
    from datetime import datetime, timezone

    from cryptotrader.persistence import TradeStore

    settings = Settings()
    settings.persistence.db_path = tmp_path / "obs.sqlite"

    async def seed() -> None:
        async with TradeStore(settings.persistence.db_path) as st:
            for i in range(3):
                await st.record_observation(
                    datetime.now(tz=timezone.utc), "BTC/USDT",
                    mid_price=100.0 + i, ob_imbalance=0.1 * i, microprice_dev_bps=float(i))

    asyncio.run(seed())
    with TestClient(create_app(settings)) as client:
        d = client.get("/api/observations/series?symbol=BTC/USDT").json()
        assert d["n"] == 3
        assert "mid_price" in d["fields"] and "microprice_dev_bps" in d["fields"]
        assert d["series"][0]["mid_price"] == 100.0          # ascending by time
        assert d["stats"]["mid_price"]["max"] == 102.0 and d["stats"]["mid_price"]["n"] == 3


def test_clear_runs_endpoint(tmp_path) -> None:
    """/api/runs/clear wipes the selected environment (no keys reload to keep the db_path)."""
    settings = Settings()
    settings.persistence.db_path = tmp_path / "clear.sqlite"
    app = create_app(settings)
    with TestClient(app) as client:
        r = client.post("/api/runs/clear", json={"environment": "simulation"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "cleared" and body["environment"] == "simulation"
        assert body["runs_deleted"] == 0  # empty isolated DB


def test_controller_aggregate_snapshot() -> None:
    """The controller merges per-symbol engine states into one aggregate snapshot."""
    from cryptotrader.api.controller import EngineController
    from cryptotrader.live.state import EngineState

    c = EngineController(Settings())
    a = EngineState(symbol="BTC/USDT")
    a.initial_equity, a.equity, a.n_trades, a.win_rate, a.status = 5000, 5500, 4, 0.5, "running"
    b = EngineState(symbol="ETH/USDT")
    b.initial_equity, b.equity, b.n_trades, b.win_rate, b.status = 5000, 4800, 2, 0.0, "running"
    c._states = {"BTC/USDT": a, "ETH/USDT": b}

    snap = c.snapshot()
    assert snap["equity"] == 10300 and snap["initial_equity"] == 10000
    assert snap["n_trades"] == 6 and snap["status"] == "running"
    assert snap["position"] is None  # multi-symbol -> per-symbol breakdown instead
    assert {s["symbol"] for s in snap["symbols"]} == {"BTC/USDT", "ETH/USDT"}


def test_circuit_breaker_detects_limits() -> None:
    from datetime import datetime, timezone

    from cryptotrader.api.controller import EngineController
    from cryptotrader.live.state import EngineState

    s = Settings()
    s.risk.max_daily_loss_pct = 0.05
    s.risk.max_drawdown_pct = 0.10
    c = EngineController(s)
    st = EngineState(symbol="BTC/USDT")
    st.initial_equity, st.equity = 10_000, 9_400          # −6% on the day
    c._states = {"BTC/USDT": st}
    c._day = datetime.now(tz=timezone.utc).date().isoformat()
    c._day_start_equity, c._peak_equity, c._halted = 10_000, 10_000, False
    assert "daily loss" in (c._risk_breach() or "")        # 6% ≥ 5%

    st.equity = 9_600                                       # −4% day, −4% DD: within limits
    c._day_start_equity, c._peak_equity = 10_000, 10_000
    assert c._risk_breach() is None

    st.equity = 9_800                                       # DD from an 11k peak = 10.9% ≥ 10%
    c._day_start_equity, c._peak_equity = 9_800, 11_000
    assert "drawdown" in (c._risk_breach() or "")

    c._halted = True                                       # halted -> no further breach
    assert c._risk_breach() is None


async def test_flatten_all_and_resume(monkeypatch) -> None:
    from cryptotrader.api.controller import EngineController

    calls = []

    class FakeEngine:
        async def flatten(self, reason):
            calls.append(("flatten", reason))

        def resume(self):
            calls.append(("resume", None))

    c = EngineController(Settings())
    c._engines = [FakeEngine(), FakeEngine()]
    r = await c.flatten_all("unit test")
    assert c._halted and r["halted"] and sum(1 for x in calls if x[0] == "flatten") == 2
    c.resume()
    assert not c._halted and any(x[0] == "resume" for x in calls)


def test_config_redacts_secrets() -> None:
    from cryptotrader.api.management import _redact

    cfg = {"exchange": {"api_key": "k", "api_secret": "s"},
           "notify": {"telegram_bot_token": "tok", "webhook_url": "https://hook"}}
    r = _redact(cfg)
    assert r["exchange"]["api_key"] is None and r["exchange"]["api_secret"] is None
    assert r["notify"]["telegram_bot_token"] is None      # secret hidden
    assert r["notify"]["webhook_url"] == "https://hook"   # non-secret kept


def test_notify_level_gating() -> None:
    from cryptotrader.ops.notify import _enabled

    cfg = Settings().notify  # min_level default "warning"
    assert _enabled(cfg, "warning") and _enabled(cfg, "critical")
    assert not _enabled(cfg, "info")


def test_jobmanager_concurrent_logic(tmp_path, monkeypatch) -> None:
    """JobManager tracks several jobs at once, keyed by kind:symbol (no real subprocess)."""
    import cryptotrader.api.management as mgmt

    monkeypatch.setattr(mgmt, "LOG_DIR", tmp_path / "logs")
    jm = mgmt.JobManager()

    class FakeProc:
        def __init__(self, rc): self.returncode = rc

    jm._jobs["train:BTC/USDT"] = {"kind": "train", "symbol": "BTC/USDT",
                                  "proc": FakeProc(None), "status": "running"}
    jm._jobs["walkforward:ETH/USDT"] = {"kind": "walkforward", "symbol": "ETH/USDT",
                                        "proc": FakeProc(0), "status": "done"}
    assert mgmt.JobManager.key("train", "BTC/USDT") == "train:BTC/USDT"
    assert mgmt.JobManager.key("holdout", None) == "holdout:config"
    assert jm.any_running is True
    by_key = {j["key"]: j for j in jm.list()}
    assert by_key["train:BTC/USDT"]["running"] is True
    assert by_key["walkforward:ETH/USDT"]["running"] is False
    assert by_key["walkforward:ETH/USDT"]["status"] == "done"


def test_symbols_table(tmp_path, monkeypatch) -> None:
    import cryptotrader.ml.registry as registry

    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")  # no models on disk
    settings = Settings()
    settings.persistence.db_path = tmp_path / "sym.sqlite"
    app = create_app(settings)
    with TestClient(app) as client:
        d = client.get("/api/symbols").json()
        assert d["configured"] == settings.exchange.symbol
        syms = {r["symbol"]: r for r in d["symbols"]}
        assert settings.exchange.symbol in syms
        active = syms[settings.exchange.symbol]
        assert active["active"] is True and active["has_model"] is False
        assert active["n_trades"] == 0
