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
    return cfg


class JobManager:
    """Runs one background job at a time (training / walk-forward / holdout).

    Each job is ``scripts/<x>.py`` as a subprocess; stdout+stderr stream to a
    per-kind log file so each dashboard panel can show its own last output even
    after a different job has run.
    """

    SCRIPTS = {
        "train": "scripts/train_model.py",
        "walkforward": "scripts/walkforward.py",
        "holdout": "scripts/holdout.py",
    }

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._kind: str | None = None
        self.status: dict[str, str] = {k: "idle" for k in self.SCRIPTS}

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def running_kind(self) -> str | None:
        return self._kind if self.running else None

    def _log_path(self, kind: str) -> Path:
        return LOG_DIR / f"job_{kind}.log"

    async def start(self, kind: str, extra_args: list[str] | None = None) -> str:
        if kind not in self.SCRIPTS:
            return "unknown_kind"
        if self.running:
            return "busy"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logf = open(self._log_path(kind), "wb")
        self._kind = kind
        self.status[kind] = "running"
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, self.SCRIPTS[kind], *(extra_args or []),
            stdout=logf, stderr=asyncio.subprocess.STDOUT,
        )
        asyncio.create_task(self._wait(kind))
        return "started"

    async def _wait(self, kind: str) -> None:
        assert self._proc is not None
        rc = await self._proc.wait()
        self.status[kind] = "done" if rc == 0 else "failed"
        self._proc = None
        self._kind = None

    def log_tail(self, kind: str, n: int = 8000) -> str:
        try:
            return self._log_path(kind).read_text(errors="replace")[-n:]
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
        args = body.get("args") or []
        if not isinstance(args, list):
            args = []
        result = await jobs.start(kind, [str(a) for a in args])
        code = 200 if result in {"started", "busy"} else 400
        return JSONResponse({"status": result, "kind": kind}, status_code=code)

    @app.get("/api/job/status")
    async def job_status(kind: str = "train") -> JSONResponse:
        return JSONResponse({
            "kind": kind,
            "status": jobs.status.get(kind, "idle"),
            "running": jobs.running and jobs.running_kind == kind,
            "running_kind": jobs.running_kind,
            "log": jobs.log_tail(kind),
        })

    # Back-compat aliases for the original training endpoints.
    @app.post("/api/train")
    async def train() -> JSONResponse:
        result = await jobs.start("train")
        return JSONResponse({"status": "already_running" if result == "busy" else result})

    @app.get("/api/train/status")
    async def train_status() -> JSONResponse:
        return JSONResponse({
            "status": jobs.status.get("train", "idle"),
            "running": jobs.running and jobs.running_kind == "train",
            "log": jobs.log_tail("train"),
        })

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


def _reload_settings(app, controller) -> None:
    new = Settings.load()
    app.state.settings = new
    controller.settings = new
