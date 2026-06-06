# src/cryptotrader/live/state.py
"""Shared runtime state and a pub/sub broadcaster for the dashboard.

The live engine owns a single :class:`EngineState` and mutates it on every bar.
The FastAPI layer reads ``snapshot()`` for REST polling and subscribes to a
:class:`StateBroadcaster` to push real-time updates over WebSockets. Decoupling
the engine from the transport keeps the engine testable in isolation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class EngineState:
    """Mutable snapshot of everything the dashboard needs to render."""

    mode: str = "paper"
    environment: str = "simulation"  # simulation | paper | live (real money)
    symbol: str = ""
    status: str = "idle"  # idle | running | stopped | error
    initial_equity: float = 0.0
    equity: float = 0.0
    realized_equity: float = 0.0
    unrealized_pnl: float = 0.0
    last_price: float = 0.0
    last_update: str = ""
    n_trades: int = 0
    win_rate: float = 0.0
    avg_efficiency_ratio: float = 0.0
    # Open-position summary (None when flat).
    position: dict[str, Any] | None = None
    # Rolling tail of recent closed trades (newest first), for the table.
    recent_trades: list[dict[str, Any]] = field(default_factory=list)
    # Down-sampled equity curve points {t, equity} for the chart.
    equity_curve: list[dict[str, Any]] = field(default_factory=list)

    def touch(self) -> None:
        self.last_update = datetime.now(tz=timezone.utc).isoformat()

    @property
    def total_return_pct(self) -> float:
        if self.initial_equity <= 0:
            return 0.0
        return (self.equity / self.initial_equity - 1.0) * 100.0

    def snapshot(self) -> dict[str, Any]:
        """Plain-dict view for JSON serialisation."""
        return {
            "mode": self.mode,
            "environment": self.environment,
            "symbol": self.symbol,
            "status": self.status,
            "initial_equity": round(self.initial_equity, 2),
            "equity": round(self.equity, 2),
            "realized_equity": round(self.realized_equity, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "total_return_pct": round(self.total_return_pct, 4),
            "last_price": round(self.last_price, 2),
            "last_update": self.last_update,
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "avg_efficiency_ratio": round(self.avg_efficiency_ratio, 4),
            "position": self.position,
            "recent_trades": self.recent_trades,
            "equity_curve": self.equity_curve,
        }


class StateBroadcaster:
    """Fan-out of state snapshots to any number of async subscribers.

    Each subscriber gets its own bounded queue; slow consumers drop the oldest
    update rather than back-pressuring the engine's hot loop.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def subscribe(self, maxsize: int = 16) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def publish(self, payload: dict[str, Any]) -> None:
        """Non-blocking push to every subscriber (drop-oldest on overflow)."""
        for q in self._subscribers:
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover
                    pass
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:  # pragma: no cover
                pass
