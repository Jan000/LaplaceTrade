# src/cryptotrader/api/auth.py
"""Optional HTTP Basic-auth ASGI middleware for the dashboard + API + WebSocket.

Enabled only when a password is configured (``dashboard.auth_password``). Browser-native:
the browser prompts once and replays the credential on every request (HTML, API, and the
WS handshake), so no per-request client changes are needed. ``/api/health`` is exempt so
uptime/load-balancer probes work without credentials.
"""

from __future__ import annotations

import base64
import hmac

from starlette.responses import PlainTextResponse

_EXEMPT = {"/api/health"}


class BasicAuthMiddleware:
    def __init__(self, app, get_auth, exempt: set[str] | None = None) -> None:
        self.app = app
        self._get_auth = get_auth                 # () -> (user, password|None)
        self._exempt = exempt or _EXEMPT

    def _authorized(self, headers: list[tuple[bytes, bytes]], user: str, password: str) -> bool:
        raw = dict(headers).get(b"authorization")
        if not raw:
            return False
        try:
            scheme, _, b64 = raw.decode().partition(" ")
            if scheme.lower() != "basic":
                return False
            got_user, _, got_pw = base64.b64decode(b64).decode("utf-8").partition(":")
        except Exception:
            return False
        # constant-time comparison on both fields
        return (hmac.compare_digest(got_user, user) and hmac.compare_digest(got_pw, password))

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        user, password = self._get_auth()
        if not password:                          # auth disabled -> pass through
            return await self.app(scope, receive, send)
        if scope.get("path") in self._exempt:
            return await self.app(scope, receive, send)
        if self._authorized(scope.get("headers") or [], user, password):
            return await self.app(scope, receive, send)
        # unauthorized
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        resp = PlainTextResponse(
            "Authentication required.", status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="CryptoTrader"'})
        await resp(scope, receive, send)
