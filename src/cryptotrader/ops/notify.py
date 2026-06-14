# src/cryptotrader/ops/notify.py
"""Best-effort out-of-band alerts for unattended 24/7 operation.

Sends a short message to a generic webhook (Slack/Discord/…) and/or Telegram. Never
raises — alerting must not break trading. Gated by ``notify.min_level``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_LEVELS = {"info": 0, "warning": 1, "critical": 2}


def _enabled(cfg, level: str) -> bool:
    return _LEVELS.get(level, 0) >= _LEVELS.get(cfg.min_level, 1)


async def notify(settings, message: str, level: str = "info") -> bool:
    """Fan out ``message`` to the configured channels. Returns True if anything was sent."""
    cfg = settings.notify
    if not _enabled(cfg, level):
        return False
    text = f"[CryptoTrader/{level.upper()}] {message}"
    sent = False
    try:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            if cfg.webhook_url:
                try:
                    await s.post(cfg.webhook_url, json={"text": text})
                    sent = True
                except Exception:
                    logger.warning("webhook notify failed", exc_info=True)
            if cfg.telegram_bot_token and cfg.telegram_chat_id:
                try:
                    url = f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendMessage"
                    await s.get(url, params={"chat_id": cfg.telegram_chat_id, "text": text})
                    sent = True
                except Exception:
                    logger.warning("telegram notify failed", exc_info=True)
    except Exception:  # pragma: no cover - aiohttp missing / unexpected
        logger.warning("notify unavailable", exc_info=True)
    if not sent:
        logger.info("NOTIFY[%s] %s", level, message)   # at least log it
    return sent
