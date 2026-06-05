# src/cryptotrader/api/management.py
"""Config-management + training-trigger API for the dashboard.

Lets the web UI read and persist the *entire* Settings tree (every training and
live parameter, incl. data-source toggles) to config/config.yaml, and launch a
training run as a background subprocess that picks up that config.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import yaml
from fastapi.responses import JSONResponse

from cryptotrader.config import Settings

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("config/config.yaml")
TRAIN_LOG = Path("data/train.log")


def _deep_merge(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


class TrainingManager:
    """Runs scripts/train_model.py as a background subprocess (one at a time)."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self.status = "idle"

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self, extra_args: list[str] | None = None) -> None:
        if self.running:
            return
        TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
        logf = open(TRAIN_LOG, "wb")
        self.status = "running"
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, "scripts/train_model.py", *(extra_args or []),
            stdout=logf, stderr=asyncio.subprocess.STDOUT,
        )
        asyncio.create_task(self._wait())

    async def _wait(self) -> None:
        assert self._proc is not None
        await self._proc.wait()
        self.status = "done" if self._proc.returncode == 0 else "failed"

    def log_tail(self, n: int = 6000) -> str:
        try:
            return TRAIN_LOG.read_text(errors="replace")[-n:]
        except Exception:
            return ""


def register_management_routes(app, controller) -> None:
    """Attach /api/config (GET/POST) and /api/train(/status) to ``app``."""
    trainer = TrainingManager()
    app.state.trainer = trainer

    @app.get("/api/config")
    async def get_config() -> JSONResponse:
        # Fresh from YAML + env so the UI always shows the effective config.
        return JSONResponse(Settings.load().model_dump(mode="json"))

    @app.post("/api/config")
    async def set_config(updates: dict) -> JSONResponse:
        try:
            raw: dict = {}
            if CONFIG_PATH.exists():
                raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
            _deep_merge(raw, updates)
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(
                yaml.safe_dump(raw, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            new = Settings.load()
            app.state.settings = new
            controller.settings = new
            return JSONResponse({"status": "saved"})
        except Exception as exc:
            logger.exception("Saving config failed")
            return JSONResponse({"status": "error", "error": str(exc)}, status_code=400)

    @app.post("/api/train")
    async def train() -> JSONResponse:
        if trainer.running:
            return JSONResponse({"status": "already_running"})
        await trainer.start()
        return JSONResponse({"status": "started"})

    @app.get("/api/train/status")
    async def train_status() -> JSONResponse:
        return JSONResponse(
            {"status": trainer.status, "running": trainer.running, "log": trainer.log_tail()}
        )
