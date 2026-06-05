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
    async def trades(limit: int = 200) -> JSONResponse:
        rows = await _read_store(app.state.settings, lambda s, rid: s.get_trades(rid, limit))
        return JSONResponse(rows)

    @app.get("/api/equity")
    async def equity(limit: int = 5000) -> JSONResponse:
        rows = await _read_store(app.state.settings, lambda s, rid: s.get_equity_curve(rid, limit))
        return JSONResponse(rows)

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


async def _read_store(settings: Settings, fn) -> list:
    store = await TradeStore(settings.persistence.db_path).connect()
    try:
        run_id = await store.latest_run_id()
        if run_id is None:
            return []
        return await fn(store, run_id)
    finally:
        await store.close()


app = create_app()


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run("cryptotrader.api.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
