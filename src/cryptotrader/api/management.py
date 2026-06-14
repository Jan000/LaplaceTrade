# src/cryptotrader/api/management.py
"""Config, secrets, background-job and run-history APIs for the dashboard.

Lets the web UI:
* read/persist the entire Settings tree to ``config/config.yaml`` (secrets redacted),
* manage exchange API keys in a git-ignored ``config/secrets.yaml`` (never echoed back),
* launch background jobs — training, walk-forward and holdout validation — and tail
  their logs,
* browse past runs persisted in SQLite.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml
from fastapi.responses import JSONResponse

from cryptotrader import config as ct_config
from cryptotrader.config import Settings

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/config.yaml")
LOG_DIR = Path("data")
_SECRET_KEYS = ("api_key", "api_secret")


def _deep_merge(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _redact(cfg: dict) -> dict:
    """Strip API secrets from a config dict before sending it to the browser."""
    ex = cfg.get("exchange")
    if isinstance(ex, dict):
        for k in _SECRET_KEYS:
            if k in ex:
                ex[k] = None
    nt = cfg.get("notify")
    if isinstance(nt, dict) and nt.get("telegram_bot_token"):
        nt["telegram_bot_token"] = None      # never echo the bot token
    db = cfg.get("dashboard")
    if isinstance(db, dict) and db.get("auth_password"):
        db["auth_password"] = None           # never echo the dashboard password
    return cfg


def _safe(symbol: str | None) -> str:
    return (symbol or "config").replace("/", "").replace(":", "")


class JobManager:
    """Runs background jobs (training / walk-forward / holdout) CONCURRENTLY.

    Each job is ``scripts/<x>.py`` as a subprocess, keyed by ``kind:symbol`` with its own
    log file, so several symbols can train/validate at once and each log is viewable
    independently. Re-starting an identical job that is still running is rejected.
    """

    SCRIPTS = {
        "train": "scripts/train_model.py",
        "walkforward": "scripts/walkforward.py",
        "holdout": "scripts/holdout.py",
    }

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}

    @staticmethod
    def key(kind: str, symbol: str | None) -> str:
        return f"{kind}:{symbol or 'config'}"

    def _log_path(self, kind: str, symbol: str | None) -> Path:
        return LOG_DIR / f"job_{kind}_{_safe(symbol)}.log"

    async def start(self, kind: str, symbol: str | None = None,
                    extra_args: list[str] | None = None) -> tuple[str | None, str]:
        if kind not in self.SCRIPTS:
            return None, "unknown_kind"
        key = self.key(kind, symbol)
        existing = self._jobs.get(key)
        if existing and existing["proc"].returncode is None:
            return key, "busy"  # this exact job is already running
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        args = (["--symbol", symbol] if symbol else []) + list(extra_args or [])
        logf = open(self._log_path(kind, symbol), "wb")
        # -u + PYTHONUNBUFFERED so the script's stdout/stderr stream to the log file live
        # (block-buffered stdout otherwise only appears when the process exits — the
        # cause of a multi-minute job showing "(no output yet)" the whole time).
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", self.SCRIPTS[kind], *args,
            stdout=logf, stderr=asyncio.subprocess.STDOUT, env=env,
        )
        self._jobs[key] = {"kind": kind, "symbol": symbol, "proc": proc, "status": "running"}
        asyncio.create_task(self._wait(key))
        return key, "started"

    async def _wait(self, key: str) -> None:
        job = self._jobs[key]
        rc = await job["proc"].wait()
        job["status"] = "done" if rc == 0 else "failed"

    def list(self) -> list[dict]:
        out = []
        for key, j in sorted(self._jobs.items()):
            running = j["proc"].returncode is None
            out.append({"key": key, "kind": j["kind"], "symbol": j["symbol"],
                        "status": "running" if running else j["status"], "running": running})
        return out

    @property
    def any_running(self) -> bool:
        return any(j["proc"].returncode is None for j in self._jobs.values())

    def log_tail(self, key: str, n: int = 8000) -> str:
        j = self._jobs.get(key)
        if not j:
            return ""
        try:
            return self._log_path(j["kind"], j["symbol"]).read_text(errors="replace")[-n:]
        except Exception:
            return ""


def register_management_routes(app, controller) -> None:
    """Attach config / secrets / jobs / runs routes to ``app``."""
    jobs = JobManager()
    app.state.jobs = jobs

    # ---------------------------------------------------------------- config
    @app.get("/api/config")
    async def get_config() -> JSONResponse:
        # Fresh from YAML + env so the UI always shows the effective config, secrets redacted.
        return JSONResponse(_redact(Settings.load().model_dump(mode="json")))

    @app.post("/api/config")
    async def set_config(updates: dict) -> JSONResponse:
        try:
            # Never let API secrets be written into the (git-tracked) config.yaml.
            if isinstance(updates.get("exchange"), dict):
                for k in _SECRET_KEYS:
                    updates["exchange"].pop(k, None)
            raw: dict = {}
            if CONFIG_PATH.exists():
                raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
            _deep_merge(raw, updates)
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(
                yaml.safe_dump(raw, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            _reload_settings(app, controller)
            return JSONResponse({"status": "saved"})
        except Exception as exc:
            logger.exception("Saving config failed")
            return JSONResponse({"status": "error", "error": str(exc)}, status_code=400)

    # ---------------------------------------------------------------- secrets / keys
    @app.get("/api/keys")
    async def get_keys() -> JSONResponse:
        s = app.state.settings
        return JSONResponse({
            "exchange_id": s.exchange.id,
            "has_key": bool(s.exchange.api_key),
            "has_secret": bool(s.exchange.api_secret),
        })

    @app.post("/api/keys")
    async def set_keys(body: dict) -> JSONResponse:
        """Persist API key/secret to the git-ignored secrets file (values never echoed)."""
        try:
            key = (body.get("api_key") or "").strip()
            secret = (body.get("api_secret") or "").strip()
            if not key or not secret:
                return JSONResponse(
                    {"status": "error", "error": "api_key and api_secret are required"},
                    status_code=400,
                )
            path = Path(ct_config.SECRETS_FILE)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump({"exchange": {"api_key": key, "api_secret": secret}},
                               sort_keys=False),
                encoding="utf-8",
            )
            try:  # best-effort hardening on POSIX
                path.chmod(0o600)
            except Exception:  # pragma: no cover - Windows / unsupported FS
                pass
            _reload_settings(app, controller)
            return JSONResponse({"status": "saved"})
        except Exception as exc:
            logger.exception("Saving keys failed")
            return JSONResponse({"status": "error", "error": str(exc)}, status_code=400)

    @app.delete("/api/keys")
    async def delete_keys() -> JSONResponse:
        try:
            Path(ct_config.SECRETS_FILE).unlink(missing_ok=True)
            _reload_settings(app, controller)
            return JSONResponse({"status": "cleared"})
        except Exception as exc:
            return JSONResponse({"status": "error", "error": str(exc)}, status_code=400)

    # ---------------------------------------------------------------- jobs
    @app.post("/api/job")
    async def start_job(body: dict) -> JSONResponse:
        kind = body.get("kind", "")
        symbol = body.get("symbol") or None
        args = body.get("args") or []
        if not isinstance(args, list):
            args = []
        key, result = await jobs.start(kind, symbol, [str(a) for a in args])
        code = 200 if result in {"started", "busy"} else 400
        return JSONResponse({"status": result, "key": key, "kind": kind, "symbol": symbol},
                            status_code=code)

    @app.get("/api/jobs")
    async def list_jobs() -> JSONResponse:
        return JSONResponse({"jobs": jobs.list(), "any_running": jobs.any_running})

    @app.get("/api/job/log")
    async def job_log(key: str) -> JSONResponse:
        running = any(j["key"] == key and j["running"] for j in jobs.list())
        status = next((j["status"] for j in jobs.list() if j["key"] == key), "unknown")
        return JSONResponse({"key": key, "status": status, "running": running,
                             "log": jobs.log_tail(key)})

    # ---------------------------------------------------------------- runs
    @app.get("/api/runs")
    async def list_runs(limit: int = 100) -> JSONResponse:
        from cryptotrader.persistence import TradeStore

        try:
            store = await TradeStore(app.state.settings.persistence.db_path).connect()
        except Exception:
            return JSONResponse([])
        try:
            return JSONResponse(await store.list_runs(limit))
        finally:
            await store.close()

    @app.post("/api/runs/clear")
    async def clear_runs(body: dict) -> JSONResponse:
        """Delete persisted runs (+ trades/equity) for an environment, or all."""
        from cryptotrader.persistence import TradeStore

        if controller.is_running:
            return JSONResponse(
                {"status": "error", "error": "Stop the engine before clearing runs."},
                status_code=400)
        env = body.get("environment")
        if env == "all":
            env = None
        try:
            store = await TradeStore(app.state.settings.persistence.db_path).connect()
        except Exception:
            return JSONResponse({"status": "error", "error": "database unavailable"},
                                status_code=400)
        try:
            n = await store.clear_runs(env)
            return JSONResponse({"status": "cleared", "runs_deleted": n,
                                 "environment": env or "all"})
        finally:
            await store.close()


def _reload_settings(app, controller) -> None:
    new = Settings.load()
    app.state.settings = new
    controller.settings = new
