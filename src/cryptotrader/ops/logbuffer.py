# src/cryptotrader/ops/logbuffer.py
"""In-memory ring buffer of recent log records, surfaced in the dashboard.

Lets an operator diagnose a running 24/7 instance (errors, halts, order activity) without
shell access to the server. Capped, so it never grows unbounded.
"""

from __future__ import annotations

import collections
import logging

_BUFFER: collections.deque[str] = collections.deque(maxlen=1000)


class _RingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _BUFFER.append(self.format(record))
        except Exception:  # pragma: no cover - logging must never raise
            pass


def install_log_capture(level: int = logging.INFO) -> None:
    """Attach the ring-buffer handler to the root logger (idempotent)."""
    root = logging.getLogger()
    if any(isinstance(h, _RingHandler) for h in root.handlers):
        return
    h = _RingHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    h.setLevel(level)
    root.addHandler(h)
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)


def recent_logs(n: int = 300) -> list[str]:
    return list(_BUFFER)[-n:]
