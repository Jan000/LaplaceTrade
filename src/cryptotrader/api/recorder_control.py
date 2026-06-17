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
        self._want_running = False     # operator intent — drives the supervisor's restart loop

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, symbols: list[str], interval: float = 120.0) -> None:
        if self.is_running:
            return
        self._symbols = list(symbols)
        self._interval = float(interval)
        self._want_running = True
        self._started_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        # The recorder is built INSIDE the supervised task: a construction or run-loop error
        # is caught and retried with backoff instead of 500-ing the start call or dying for good.
        self._task = asyncio.create_task(self._supervise())
        logger.info("Recorder started for %d symbols (every %.0fs)", len(self._symbols), self._interval)

    async def _supervise(self) -> None:
        """Keep a MarketRecorder running while the operator wants it — restart on any crash."""
        from cryptotrader.data.recorder import MarketRecorder

        backoff = 5.0
        while self._want_running:
            try:
                self._recorder = MarketRecorder(self.settings, self._symbols, interval=self._interval)
                await self._recorder.run()          # returns only when recorder.stop() is called
                backoff = 5.0
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Recorder crashed; restarting in %.0fs", backoff)
            if not self._want_running:
                break
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff * 2, 120.0)       # exponential backoff, capped

    async def stop(self) -> None:
        self._want_running = False
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
        st = {
            "running": self.is_running,
            "symbols": self._symbols,
            "interval": self._interval,
            "started_at": self._started_at if self.is_running else None,
        }
        stats = getattr(self._recorder, "stats", None)
        if callable(stats):
            st.update(stats())                       # writes / cycles / last_write / last_error
        return st
