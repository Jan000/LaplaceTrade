# src/cryptotrader/api/recorder_control.py
"""Dashboard-controllable wrapper around the live MarketRecorder.

Owns the recorder as a background asyncio task so the operator can start/stop it and see
its status from the UI, without running a separate CLI process.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from cryptotrader.config import Settings

logger = logging.getLogger(__name__)


class RecorderController:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._recorder = None
        self._task: asyncio.Task | None = None
        self._symbols: list[str] = []
        self._interval: float = 120.0
        self._started_at: str | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, symbols: list[str], interval: float = 120.0) -> None:
        if self.is_running:
            return
        from cryptotrader.data.recorder import MarketRecorder

        self._symbols = list(symbols)
        self._interval = float(interval)
        self._recorder = MarketRecorder(self.settings, self._symbols, interval=self._interval)
        self._task = asyncio.create_task(self._run_guarded())
        self._started_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        logger.info("Recorder started for %d symbols (every %.0fs)", len(self._symbols), self._interval)

    async def _run_guarded(self) -> None:
        try:
            await self._recorder.run()
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            logger.exception("Recorder crashed")

    async def stop(self) -> None:
        if self._recorder is not None:
            self._recorder.stop()
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=15.0)
            except (asyncio.TimeoutError, Exception):  # pragma: no cover
                self._task.cancel()
        self._task = None
        self._recorder = None
        logger.info("Recorder stopped")

    def status(self) -> dict:
        return {
            "running": self.is_running,
            "symbols": self._symbols,
            "interval": self._interval,
            "started_at": self._started_at if self.is_running else None,
        }
