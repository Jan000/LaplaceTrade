# src/cryptotrader/api/server.py
"""FastAPI application: REST + WebSocket endpoints and the dashboard."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from cryptotrader.api.controller import EngineController
from cryptotrader.api.management import register_management_routes
from cryptotrader.config import Settings
from cryptotrader.persistence import TradeStore

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


class StartRequest(BaseModel):
    mode: str = "simulation"          # "simulation" | "live"
    real_orders: bool = False         # place REAL ccxt orders (live mode only)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory (keeps state out of import-time globals)."""
    settings = settings or Settings.load()
    app = FastAPI(title="CryptoTrader Control Center", version="0.1.0")
    controller = EngineController(settings)
    app.state.controller = controller
    app.state.settings = settings
    register_management_routes(app, controller)  # /api/config + /api/train

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "dashboard.html")

    @app.get("/api/state")
    async def get_state() -> JSONResponse:
        return JSONResponse(controller.state.snapshot())

    @app.get("/api/model")
    async def model_info() -> JSONResponse:
        """Which predictor the engine will load: the trained model or the baseline."""
        from datetime import datetime, timezone

        s = app.state.settings
        path = s.strategy.model_path
        p = Path(path) if path else None
        exists = bool(p and p.exists())
        info: dict = {
            "configured_path": str(path) if path else None,
            "exists": exists,
            "active": "trained model" if exists else "momentum baseline",
            "trained_at": None,
            "size_bytes": None,
            "train_symbols": s.data.train_symbols,
            "timeframe": s.exchange.timeframe,
        }
        if exists:
            stat = p.stat()
            info["trained_at"] = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()
            info["size_bytes"] = stat.st_size
        return JSONResponse(info)

    @app.post("/api/start")
    async def start(req: StartRequest) -> JSONResponse:
        try:
            await controller.start(mode=req.mode, real_orders=req.real_orders)
        except Exception as exc:
            controller.state.status = "error"
            logger.exception("Failed to start engine")
            return JSONResponse({"status": "error", "error": str(exc)}, status_code=400)
        return JSONResponse({"status": "running", "mode": req.mode})

    @app.post("/api/stop")
    async def stop() -> JSONResponse:
        await controller.stop()
        return JSONResponse({"status": "stopped"})

    @app.get("/api/trades")
    async def trades(limit: int = 200, run_id: str | None = None) -> JSONResponse:
        if run_id == "all":
            rows = await _read_store_all(app.state.settings, lambda s: s.get_all_trades(limit))
            return JSONResponse(rows)
        rid = int(run_id) if run_id not in (None, "", "latest") else None
        rows = await _read_store(app.state.settings, lambda s, r: s.get_trades(r, limit), rid)
        return JSONResponse(rows)

    @app.get("/api/equity")
    async def equity(limit: int = 5000, run_id: str | None = None) -> JSONResponse:
        rid = int(run_id) if run_id not in (None, "", "latest", "all") else None
        rows = await _read_store(
            app.state.settings, lambda s, r: s.get_equity_curve(r, limit), rid
        )
        return JSONResponse(rows)

    @app.get("/api/stats")
    async def stats(run_id: str | None = None) -> JSONResponse:
        """Rich trade statistics for one run, the latest run, or all runs (run_id=all)."""
        from cryptotrader.backtest.analytics import summarize_trades

        if run_id == "all":
            trades_rows = await _read_store_all(
                app.state.settings, lambda s: s.get_all_trades(50000)
            )
            return JSONResponse(summarize_trades(trades_rows, equity=None))
        rid = int(run_id) if run_id not in (None, "", "latest") else None
        trades_rows = await _read_store(
            app.state.settings, lambda s, r: s.get_trades(r, 50000), rid
        )
        eq_rows = await _read_store(
            app.state.settings, lambda s, r: s.get_equity_curve(r, 100000), rid
        )
        return JSONResponse(summarize_trades(trades_rows, eq_rows))

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = controller.broadcaster.subscribe()
        await websocket.send_json(controller.state.snapshot())
        try:
            while True:
                payload = await queue.get()
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            controller.broadcaster.unsubscribe(queue)

    return app


async def _read_store(settings: Settings, fn, run_id: int | None = None) -> list:
    store = await TradeStore(settings.persistence.db_path).connect()
    try:
        rid = run_id if run_id is not None else await store.latest_run_id()
        if rid is None:
            return []
        return await fn(store, rid)
    finally:
        await store.close()


async def _read_store_all(settings: Settings, fn) -> list:
    """Like _read_store but not scoped to a single run (aggregate across all runs)."""
    store = await TradeStore(settings.persistence.db_path).connect()
    try:
        return await fn(store)
    finally:
        await store.close()


app = create_app()


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run("cryptotrader.api.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
