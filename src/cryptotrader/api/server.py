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
    sim_days: int | None = None       # simulation: accelerated test on the last N days of real data


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
        return JSONResponse(controller.snapshot())

    _COMMON_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                       "XRP/USDT", "ADA/USDT", "DOGE/USDT", "LTC/USDT"]

    @app.get("/api/symbols")
    async def symbols() -> JSONResponse:
        """Per-symbol model status + realized trade stats + active flag (Symbols table)."""
        from cryptotrader.ml.registry import (
            list_models, model_path_for, read_meta, read_validation,
        )

        s = app.state.settings
        trade_set = set(s.data.trade_symbols or [])
        # DB trade stats per symbol.
        summaries = await _read_store_all(app.state.settings, lambda st: st.symbol_summaries())
        stats = {r["symbol"]: r for r in summaries}
        # Models on disk.
        models = {m["symbol"]: m for m in list_models() if m.get("symbol")}

        names = list(dict.fromkeys(
            [*_COMMON_SYMBOLS, *models.keys(), *stats.keys(),
             *trade_set, s.exchange.symbol]))
        out = []
        for sym in names:
            meta = (models.get(sym) or {}).get("meta")
            if meta is None:  # also resolve in case naming differs
                meta = read_meta(model_path_for(sym))
            st = stats.get(sym, {})
            n = int(st.get("n_trades") or 0)
            gw, gl = float(st.get("gross_win") or 0.0), float(st.get("gross_loss") or 0.0)
            out.append({
                "symbol": sym,
                "active": sym == s.exchange.symbol,
                "trade": sym in trade_set,
                "has_model": meta is not None,
                "model_timeframe": (meta or {}).get("timeframe"),
                "trained_at": (meta or {}).get("saved_at"),
                "train_symbols": (meta or {}).get("train_symbols"),
                "n_train_rows": (meta or {}).get("n_train_rows"),
                "matches": bool(meta and meta.get("symbol") == sym
                                and meta.get("timeframe") == s.exchange.timeframe),
                "n_trades": n,
                "win_rate": round((st.get("wins") or 0) / n, 4) if n else 0.0,
                "net_pnl": round(float(st.get("net_pnl") or 0.0), 2),
                "avg_efficiency": round(float(st["avg_efficiency"]), 4)
                                  if st.get("avg_efficiency") is not None else None,
                "profit_factor": round(gw / gl, 3) if gl > 0 else (None if gw > 0 else 0.0),
                "walkforward": read_validation("walkforward", sym),
                "holdout": read_validation("holdout", sym),
            })
        active_set = trade_set or {s.exchange.symbol}
        real_ok = any(r["matches"] for r in out if r["symbol"] in active_set)
        return JSONResponse({"symbols": out, "configured": s.exchange.symbol,
                             "configured_timeframe": s.exchange.timeframe,
                             "trade_symbols": sorted(trade_set),
                             "trade_count": len(active_set), "real_ok": real_ok})

    @app.get("/api/experiments")
    async def experiments(limit: int = 300) -> JSONResponse:
        """Append-only history of training / walk-forward / holdout runs (settings + result)."""
        from cryptotrader.ml.experiments import read_experiments

        return JSONResponse(read_experiments(limit))

    @app.get("/api/observations")
    async def observations() -> JSONResponse:
        """Per-symbol count of recorded live observations (the forward dataset)."""
        counts = await _read_store_all(app.state.settings, lambda s: s.observation_count())
        return JSONResponse({"counts": counts, "total": sum(counts.values())})

    @app.get("/api/model")
    async def model_info() -> JSONResponse:
        """Which model the engine will load for the configured symbol, and whether it matches."""
        from datetime import datetime, timezone

        from cryptotrader.ml.registry import resolve_model

        s = app.state.settings
        path, meta = resolve_model(s)
        exists = path is not None
        meta = meta or {}
        model_symbol = meta.get("symbol")
        model_tf = meta.get("timeframe")
        # Strict match: metadata must confirm the exact symbol+timeframe (this gates real
        # orders). A model without metadata cannot be verified and does not match.
        matches = exists and model_symbol == s.exchange.symbol and model_tf == s.exchange.timeframe
        trained_at = meta.get("saved_at")
        size_bytes = None
        if exists:
            stat = path.stat()
            size_bytes = stat.st_size
            trained_at = trained_at or datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc).isoformat()
        return JSONResponse({
            "exists": exists,
            "active": "trained model" if exists else "momentum baseline",
            "model_path": str(path) if path else None,
            "model_symbol": model_symbol,
            "model_timeframe": model_tf,
            "configured_symbol": s.exchange.symbol,
            "configured_timeframe": s.exchange.timeframe,
            "train_symbols": meta.get("train_symbols", s.data.train_symbols),
            "trained_at": trained_at,
            "size_bytes": size_bytes,
            "matches": bool(matches),
            "real_ok": bool(matches),  # real orders allowed only when a matching model exists
        })

    @app.post("/api/start")
    async def start(req: StartRequest) -> JSONResponse:
        try:
            await controller.start(mode=req.mode, real_orders=req.real_orders,
                                   sim_days=req.sim_days)
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
    async def trades(
        limit: int = 200, run_id: str | None = None, env: str | None = None
    ) -> JSONResponse:
        if run_id == "all":
            rows = await _read_store_all(
                app.state.settings, lambda s: s.get_all_trades(limit, env or None)
            )
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
    async def stats(run_id: str | None = None, env: str | None = None) -> JSONResponse:
        """Rich trade statistics for one run, the latest run, or all runs (run_id=all)."""
        from cryptotrader.backtest.analytics import summarize_trades

        if run_id == "all":
            trades_rows = await _read_store_all(
                app.state.settings, lambda s: s.get_all_trades(50000, env or None)
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
