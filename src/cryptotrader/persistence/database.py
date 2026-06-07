# src/cryptotrader/persistence/database.py
"""Asynchronous SQLite persistence layer.

A single :class:`TradeStore` owns one ``aiosqlite`` connection and exposes
non-blocking writes/reads for the three things the system needs to durably record:

* **runs**            — one row per backtest or live session (mode, symbol, config).
* **trades**          — every closed round-trip trade, incl. the efficiency ratio.
* **equity_snapshots**— the mark-to-market equity curve (for PnL charts / drawdown).
* **feature_rows**    — optional storage of computed feature vectors (model audit /
  offline retraining), serialised as JSON to stay schema-agnostic.

Why async + SQLite
-------------------
The live engine runs inside one ``asyncio`` loop; a blocking ``sqlite3`` write on
the hot path would stall data ingestion. ``aiosqlite`` runs the DB on a worker
thread and hands back awaitables, so logging never blocks tick processing. SQLite
itself is more than sufficient for an MVP's write volume (a few rows per minute).

WAL mode is enabled so the FastAPI dashboard can read the DB concurrently while
the engine writes to it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from cryptotrader.core.types import Trade

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT    NOT NULL,
    mode        TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    exchange    TEXT    NOT NULL,
    initial_equity REAL NOT NULL,
    config_json TEXT,
    environment TEXT    DEFAULT 'simulation'   -- simulation | paper | live (real money)
);

CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    symbol      TEXT    NOT NULL,
    side        INTEGER NOT NULL,
    quantity    REAL    NOT NULL,
    entry_time  TEXT    NOT NULL,
    entry_price REAL    NOT NULL,
    exit_time   TEXT    NOT NULL,
    exit_price  REAL    NOT NULL,
    fees        REAL    NOT NULL,
    gross_pnl   REAL    NOT NULL,
    net_pnl     REAL    NOT NULL,
    best_price  REAL    NOT NULL,
    exit_reason TEXT    NOT NULL,
    efficiency_ratio REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    INTEGER NOT NULL REFERENCES runs(id),
    timestamp TEXT    NOT NULL,
    equity    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_equity_run ON equity_snapshots(run_id);

CREATE TABLE IF NOT EXISTS feature_rows (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    INTEGER NOT NULL REFERENCES runs(id),
    timestamp TEXT    NOT NULL,
    features  TEXT    NOT NULL   -- JSON {name: value}
);
CREATE INDEX IF NOT EXISTS idx_features_run ON feature_rows(run_id);
"""


def _iso(ts: datetime) -> str:
    """Serialise a datetime to a UTC ISO-8601 string."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


class TradeStore:
    """Async data-access object backed by a single SQLite database.

    Use as an async context manager::

        async with TradeStore("data/cryptotrader.sqlite") as store:
            run_id = await store.start_run(mode="live", symbol="BTC/USDT", ...)
            await store.record_trade(run_id, trade)
    """

    def __init__(self, db_path: str | Path, equity_commit_every: int = 50) -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None
        # High-frequency equity samples are buffered and committed in batches so
        # the live hot loop never pays a per-bar fsync. Trades always commit
        # immediately (and flush any pending equity) so nothing important is lost.
        self._equity_commit_every = max(1, equity_commit_every)
        self._pending_equity = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def connect(self) -> "TradeStore":
        """Open the connection, apply pragmas and ensure the schema exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        # WAL gives concurrent dashboard reads, but some filesystems (network
        # shares, certain mounts) reject the shared-memory file WAL needs. Fall
        # back to the default rollback journal there instead of failing hard.
        try:
            await self._db.execute("PRAGMA journal_mode=WAL;")
            await self._db.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            logger.warning("WAL journal unavailable; using default journal mode.")
        # Wait (rather than fail with "database is locked") when another short-lived
        # connection still holds the WAL write lock — common with our per-request stores.
        await self._db.execute("PRAGMA busy_timeout=5000;")
        await self._db.executescript(_SCHEMA)
        # Migration: older DBs predate the runs.environment column.
        await self._ensure_column("runs", "environment", "TEXT DEFAULT 'simulation'")
        await self._db.commit()
        logger.info("TradeStore connected at %s", self.db_path)
        return self

    async def _ensure_column(self, table: str, col: str, decl: str) -> None:
        async with self._db.execute(f"PRAGMA table_info({table})") as cur:
            cols = [r["name"] for r in await cur.fetchall()]
        if col not in cols:
            await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    async def close(self) -> None:
        if self._db is not None:
            await self._db.commit()  # flush any buffered equity samples
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> "TradeStore":
        return await self.connect()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("TradeStore is not connected; call connect() first.")
        return self._db

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    async def start_run(
        self,
        mode: str,
        symbol: str,
        exchange: str,
        initial_equity: float,
        config: dict[str, Any] | None = None,
        environment: str = "simulation",
    ) -> int:
        """Create a run row and return its id (foreign key for all other rows)."""
        cur = await self._conn.execute(
            "INSERT INTO runs (started_at, mode, symbol, exchange, initial_equity, "
            "config_json, environment) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                _iso(datetime.now(tz=timezone.utc)),
                mode,
                symbol,
                exchange,
                initial_equity,
                json.dumps(config or {}, default=str),
                environment,
            ),
        )
        await self._conn.commit()
        return int(cur.lastrowid)

    async def record_trade(self, run_id: int, trade: Trade) -> None:
        """Persist a single closed trade."""
        await self._conn.execute(
            "INSERT INTO trades (run_id, symbol, side, quantity, entry_time, "
            "entry_price, exit_time, exit_price, fees, gross_pnl, net_pnl, "
            "best_price, exit_reason, efficiency_ratio) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                trade.symbol,
                int(trade.side),
                trade.quantity,
                _iso(trade.entry_time),
                trade.entry_price,
                _iso(trade.exit_time),
                trade.exit_price,
                trade.fees,
                trade.gross_pnl,
                trade.net_pnl,
                trade.best_price,
                trade.exit_reason,
                trade.efficiency_ratio,
            ),
        )
        # A trade is significant: commit it together with any buffered equity.
        self._pending_equity = 0
        await self._conn.commit()

    async def record_equity(self, run_id: int, timestamp: datetime, equity: float) -> None:
        """Append one equity-curve sample (committed in batches, see __init__)."""
        await self._conn.execute(
            "INSERT INTO equity_snapshots (run_id, timestamp, equity) VALUES (?,?,?)",
            (run_id, _iso(timestamp), equity),
        )
        self._pending_equity += 1
        if self._pending_equity >= self._equity_commit_every:
            self._pending_equity = 0
            await self._conn.commit()

    async def flush(self) -> None:
        """Commit any buffered (uncommitted) writes."""
        if self._db is not None:
            self._pending_equity = 0
            await self._db.commit()

    async def record_features(
        self, run_id: int, timestamp: datetime, features: dict[str, float]
    ) -> None:
        """Store one feature vector as JSON (model audit / offline retraining)."""
        await self._conn.execute(
            "INSERT INTO feature_rows (run_id, timestamp, features) VALUES (?,?,?)",
            (run_id, _iso(timestamp), json.dumps(features)),
        )
        await self._conn.commit()

    # ------------------------------------------------------------------ #
    # Reads (consumed by the dashboard API)
    # ------------------------------------------------------------------ #
    async def latest_run_id(self) -> int | None:
        """Id of the most recently started run, or ``None`` if there are none."""
        async with self._conn.execute("SELECT MAX(id) AS m FROM runs") as cur:
            row = await cur.fetchone()
        return int(row["m"]) if row and row["m"] is not None else None

    async def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Run metadata (newest first) with trade count + last equity, for the run picker."""
        async with self._conn.execute(
            "SELECT r.id, r.started_at, r.mode, r.symbol, r.exchange, r.initial_equity, "
            "COALESCE(r.environment, CASE r.mode WHEN 'backtest' THEN 'simulation' "
            " ELSE 'paper' END) AS environment, "
            "(SELECT COUNT(*) FROM trades t WHERE t.run_id = r.id) AS n_trades, "
            "(SELECT e.equity FROM equity_snapshots e WHERE e.run_id = r.id "
            " ORDER BY e.id DESC LIMIT 1) AS final_equity "
            "FROM runs r ORDER BY r.id DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_trades(self, run_id: int, limit: int = 500) -> list[dict[str, Any]]:
        """Return the most recent ``limit`` trades for a run (newest first)."""
        async with self._conn.execute(
            "SELECT * FROM trades WHERE run_id = ? ORDER BY id DESC LIMIT ?",
            (run_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def symbol_summaries(self, include_simulation: bool = False) -> list[dict[str, Any]]:
        """Per-symbol realized trade stats for the Symbols table.

        By default EXCLUDES simulation runs: accelerated replays accumulate huge,
        non-representative trade counts that would drown the decision-relevant signal.
        So these columns reflect paper + real (live) behaviour; judge the *edge* from the
        walk-forward / holdout results instead.
        """
        where = ("" if include_simulation else
                 "WHERE COALESCE(r.environment, CASE r.mode WHEN 'backtest' THEN "
                 "'simulation' ELSE 'paper' END) != 'simulation' ")
        async with self._conn.execute(
            "SELECT t.symbol AS symbol, COUNT(*) AS n_trades, "
            "SUM(CASE WHEN t.net_pnl > 0 THEN 1 ELSE 0 END) AS wins, "
            "SUM(t.net_pnl) AS net_pnl, AVG(t.efficiency_ratio) AS avg_efficiency, "
            "SUM(CASE WHEN t.net_pnl > 0 THEN t.net_pnl ELSE 0 END) AS gross_win, "
            "SUM(CASE WHEN t.net_pnl < 0 THEN -t.net_pnl ELSE 0 END) AS gross_loss "
            "FROM trades t JOIN runs r ON t.run_id = r.id " + where + "GROUP BY t.symbol"
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def clear_runs(self, environment: str | None = None) -> int:
        """Delete runs (and their trades / equity / feature rows) for an environment.

        ``environment`` of ``None`` or ``"all"`` wipes everything. Returns the number of
        runs removed. Used by the dashboard's "reset simulation / clear data" action.
        """
        if environment in (None, "all"):
            async with self._conn.execute("SELECT id FROM runs") as cur:
                ids = [r["id"] for r in await cur.fetchall()]
        else:
            async with self._conn.execute(
                "SELECT id FROM runs WHERE COALESCE(environment, CASE mode WHEN 'backtest' "
                "THEN 'simulation' ELSE 'paper' END) = ?", (environment,),
            ) as cur:
                ids = [r["id"] for r in await cur.fetchall()]
        if not ids:
            return 0
        qm = ",".join("?" * len(ids))
        for tbl in ("feature_rows", "equity_snapshots", "trades"):
            await self._conn.execute(f"DELETE FROM {tbl} WHERE run_id IN ({qm})", ids)
        await self._conn.execute(f"DELETE FROM runs WHERE id IN ({qm})", ids)
        await self._conn.commit()
        return len(ids)

    async def get_all_trades(
        self, limit: int = 10000, environment: str | None = None
    ) -> list[dict[str, Any]]:
        """Most recent ``limit`` trades across all runs (optionally one environment)."""
        if environment:
            sql = (
                "SELECT t.* FROM trades t JOIN runs r ON t.run_id = r.id "
                "WHERE COALESCE(r.environment, CASE r.mode WHEN 'backtest' "
                "THEN 'simulation' ELSE 'paper' END) = ? "
                "ORDER BY t.id DESC LIMIT ?"
            )
            params: tuple = (environment, limit)
        else:
            sql = "SELECT * FROM trades ORDER BY id DESC LIMIT ?"
            params = (limit,)
        async with self._conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_equity_curve(self, run_id: int, limit: int = 5000) -> list[dict[str, Any]]:
        """Return equity samples for a run in chronological order."""
        async with self._conn.execute(
            "SELECT timestamp, equity FROM equity_snapshots WHERE run_id = ? "
            "ORDER BY id ASC LIMIT ?",
            (run_id, limit),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]
