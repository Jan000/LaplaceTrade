# src/cryptotrader/api/scheduler.py
"""Background scheduler for unattended operation.

* **Auto-retrain** — every ``data.retrain_interval_days`` it launches train + walk-forward
  jobs for the traded symbols (via the JobManager) so the deployed model stays fresh on a
  rolling window. The last-run time is persisted so the cadence survives restarts.
* **Daily summary** — once per UTC day, alerts the running account's equity/PnL.

Never raises into the loop; all actions are best-effort.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_CHECK_SECONDS = 1800.0   # how often the loop wakes to evaluate due work


class Scheduler:
    def __init__(self, settings, jobs, controller) -> None:
        self.settings = settings
        self.jobs = jobs
        self.controller = controller
        self._task: asyncio.Task | None = None
        self._stop = False
        self._last_summary_day: str | None = None
        self._last_retrain: datetime | None = self._load()

    # --- persistence of the last retrain time -------------------------------
    def _state_path(self):
        from cryptotrader.ml import registry
        return registry.MODELS_DIR / "scheduler_state.json"

    def _load(self) -> datetime | None:
        try:
            p = self._state_path()
            if p.exists():
                ts = json.loads(p.read_text(encoding="utf-8")).get("last_retrain")
                return datetime.fromisoformat(ts) if ts else None
        except Exception:  # pragma: no cover
            pass
        return None

    def _save(self) -> None:
        try:
            p = self._state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({
                "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            }), encoding="utf-8")
        except Exception:  # pragma: no cover
            logger.warning("could not persist scheduler state", exc_info=True)

    # --- lifecycle ----------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.is_running:
            return
        self._stop = False
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop = True
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        logger.info("Scheduler started")
        try:
            while not self._stop:
                try:
                    await self._tick()
                except Exception:  # pragma: no cover - never die
                    logger.exception("scheduler tick failed")
                slept = 0.0
                while slept < _CHECK_SECONDS and not self._stop:
                    await asyncio.sleep(min(2.0, _CHECK_SECONDS - slept))
                    slept += 2.0
        except asyncio.CancelledError:  # pragma: no cover
            pass

    # --- work ---------------------------------------------------------------
    def _trade_symbols(self) -> list[str]:
        return list(self.settings.data.trade_symbols or []) or [self.settings.exchange.symbol]

    async def _tick(self) -> None:
        now = datetime.now(tz=timezone.utc)
        days = float(self.settings.data.retrain_interval_days or 0)
        if days > 0 and not self.jobs.any_running:
            due = (self._last_retrain is None
                   or (now - self._last_retrain).total_seconds() >= days * 86400)
            if due:
                await self.retrain_now()
        if self.settings.notify.daily_summary:
            day = now.date().isoformat()
            if self._last_summary_day != day:
                self._last_summary_day = day
                await self._daily_summary()

    async def retrain_now(self) -> dict:
        """Launch train + walk-forward jobs for every traded symbol (best-effort)."""
        started: list[str] = []
        for sym in self._trade_symbols():
            for kind in ("train", "walkforward"):
                try:
                    await self.jobs.start(kind, sym)
                    started.append(f"{kind}:{sym}")
                except Exception:  # pragma: no cover
                    logger.warning("could not start %s for %s", kind, sym, exc_info=True)
        self._last_retrain = datetime.now(tz=timezone.utc)
        self._save()
        logger.warning("Auto-retrain launched: %s", ", ".join(started) or "nothing")
        try:
            from cryptotrader.ops.notify import notify
            await notify(self.settings, f"🔁 Auto-retrain started ({', '.join(started)}). "
                                        "Restart the engine to deploy the new model.", level="warning")
        except Exception:  # pragma: no cover
            pass
        return {"started": started, "last_retrain": self._last_retrain.isoformat()}

    async def _daily_summary(self) -> None:
        snap = self.controller.snapshot()
        if snap.get("status") != "running":
            return
        try:
            from cryptotrader.ops.notify import notify
            await notify(self.settings,
                         f"📊 Daily summary — equity {snap.get('equity')} "
                         f"({snap.get('total_return_pct', 0):+.2f}%), "
                         f"{snap.get('n_trades', 0)} trades, "
                         f"win {snap.get('win_rate', 0) * 100:.0f}%"
                         f"{' · HALTED' if snap.get('halted') else ''}", level="warning")
        except Exception:  # pragma: no cover
            pass

    def status(self) -> dict:
        days = float(self.settings.data.retrain_interval_days or 0)
        nxt = None
        if days > 0 and self._last_retrain is not None:
            from datetime import timedelta
            nxt = (self._last_retrain + timedelta(days=days)).isoformat()
        return {
            "running": self.is_running,
            "retrain_interval_days": days,
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "next_retrain": nxt,
            "daily_summary": bool(self.settings.notify.daily_summary),
        }
