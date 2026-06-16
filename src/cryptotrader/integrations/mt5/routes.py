# src/cryptotrader/integrations/mt5/routes.py
"""FastAPI routes for the MetaTrader 5 bridge (token-protected, prefix /api/mt5).

The CPU work (feature transform + model inference) runs in a threadpool so it never
blocks the event loop / WebSocket broadcasts. Auth is a shared ``mt5.api_token`` sent
by the EA as ``X-API-Token`` (the prefix is exempt from the dashboard Basic-auth)."""

from __future__ import annotations

import hmac
import logging

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)


def _auth(settings, request: Request) -> None:
    if not settings.mt5.enabled:
        raise HTTPException(status_code=404, detail="MT5 bridge disabled")
    token = settings.mt5.api_token
    if token:
        got = request.headers.get("x-api-token", "")
        if not hmac.compare_digest(got, token):
            raise HTTPException(status_code=401, detail="invalid MT5 token")


def register_mt5_routes(app, get_settings) -> None:
    """Wire /api/mt5/* onto ``app``. ``get_settings`` returns the live Settings."""

    @app.get("/api/mt5/status")
    async def mt5_status(request: Request) -> JSONResponse:
        s = get_settings()
        _auth(s, request)
        from cryptotrader.integrations.mt5.store import capture_counts

        return JSONResponse({
            "enabled": s.mt5.enabled,
            "require_model": s.mt5.require_model,
            "capture_bars": s.mt5.capture_bars,
            "poll_seconds": s.mt5.poll_seconds,
            "symbol_map": s.mt5.symbol_map,
            "captured": await run_in_threadpool(capture_counts, s),
        })

    @app.post("/api/mt5/decide")
    async def mt5_decide(request: Request) -> JSONResponse:
        s = get_settings()
        _auth(s, request)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="invalid JSON body")

        symbol = body.get("symbol")
        if not symbol:
            raise HTTPException(status_code=400, detail="missing 'symbol'")
        timeframe = body.get("timeframe")
        bars = body.get("bars") or []
        position = body.get("position")
        account = body.get("account")

        from cryptotrader.integrations.mt5.bridge import decide

        result = await run_in_threadpool(
            decide, s, symbol=symbol, timeframe=timeframe, bars=bars,
            position=position, account=account,
        )

        # Data-source role: passively grow a training set from what the EA streams.
        if s.mt5.capture_bars and result.get("model_symbol") and bars:
            from cryptotrader.integrations.mt5.bridge import bars_to_df
            from cryptotrader.integrations.mt5.store import append_bars

            tf = result.get("timeframe") or s.exchange.timeframe
            try:
                df = await run_in_threadpool(bars_to_df, bars)
                if not df.empty:
                    await run_in_threadpool(append_bars, s, result["model_symbol"], tf, df)
            except Exception:  # pragma: no cover - capture must not break a decision
                logger.warning("MT5 capture failed", exc_info=True)

        return JSONResponse(result)

    @app.post("/api/mt5/ingest")
    async def mt5_ingest(request: Request) -> JSONResponse:
        """Capture bars without asking for a decision (pure data feed)."""
        s = get_settings()
        _auth(s, request)
        body = await request.json()
        symbol = body.get("symbol")
        bars = body.get("bars") or []
        if not symbol or not bars:
            raise HTTPException(status_code=400, detail="need 'symbol' and 'bars'")

        from cryptotrader.integrations.mt5.bridge import bars_to_df, map_symbol
        from cryptotrader.integrations.mt5.store import append_bars

        model_symbol = map_symbol(s, symbol)
        if model_symbol is None:
            return JSONResponse({"stored": 0, "reason": "unmapped_symbol"}, status_code=200)
        tf = body.get("timeframe") or s.exchange.timeframe
        df = await run_in_threadpool(bars_to_df, bars)
        total = await run_in_threadpool(append_bars, s, model_symbol, tf, df)
        return JSONResponse({"stored": int(len(df)), "total_rows": total,
                             "symbol": model_symbol, "timeframe": tf})
