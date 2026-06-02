# src/cryptotrader/data/ingestion.py
"""Asynchronous market-data ingestion.

This module provides three things:

* :class:`MarketDataFeed`  — a thin async wrapper around ccxt that pulls paginated
  historical OHLCV (with on-disk caching) and streams live candles.
* :class:`HistoricalDataHandler` — replays a cached DataFrame as ordered
  :class:`MarketEvent` objects for the backtester.
* :class:`LiveDataHandler` — bridges the live websocket stream into the same
  :class:`MarketEvent` interface, so the rest of the system is mode-agnostic.

Design notes
------------
* ``ccxt`` is imported lazily so backtests on synthetic/cached data need no
  network dependency installed.
* The low-latency live path uses ``ccxt.pro``'s ``watch_ohlcv`` (true websocket)
  when available, transparently falling back to short-interval REST polling.
* Only *closed* candles are ever emitted, which is what protects downstream
  feature/label computation from look-ahead bias.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from cryptotrader.core.events import MarketEvent
from cryptotrader.core.interfaces import DataHandler
from cryptotrader.core.types import Bar

logger = logging.getLogger(__name__)

# Canonical column order for every OHLCV DataFrame in the system.
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Map common ccxt timeframes to their millisecond duration for pagination.
_TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
}


class MarketDataFeed:
    """Async ccxt data source for one symbol/timeframe on one exchange.

    Parameters
    ----------
    exchange_id:
        Any ccxt exchange id (``"binance"``, ``"bybit"``, ...).
    symbol:
        Unified ccxt symbol, e.g. ``"BTC/USDT"``.
    timeframe:
        Candle timeframe, e.g. ``"1m"``.
    cache_dir:
        Directory for parquet OHLCV caches. ``None`` disables caching.
    api_key, api_secret:
        Optional credentials; only required for live private endpoints.
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        symbol: str = "BTC/USDT",
        timeframe: str = "1m",
        cache_dir: Path | None = Path(".cache/ohlcv"),
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.cache_dir = cache_dir
        self._api_key = api_key
        self._api_secret = api_secret
        self._tf_ms = _TIMEFRAME_MS.get(timeframe, 60_000)
        self._client: object | None = None  # lazily created ccxt client

    # ------------------------------------------------------------------ #
    # Client lifecycle
    # ------------------------------------------------------------------ #
    def _make_client(self, pro: bool = False):
        """Instantiate a ccxt (or ccxt.pro) async client lazily."""
        module_name = "ccxt.pro" if pro else "ccxt.async_support"
        try:
            module = __import__(module_name, fromlist=["dummy"])
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                f"{module_name} is required for live/network data ingestion."
            ) from exc
        klass = getattr(module, self.exchange_id)
        return klass(
            {
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "enableRateLimit": True,
            }
        )

    async def close(self) -> None:
        """Close the underlying ccxt client's network sessions."""
        if self._client is not None and hasattr(self._client, "close"):
            await self._client.close()  # type: ignore[func-returns-value]
            self._client = None

    # ------------------------------------------------------------------ #
    # Historical data
    # ------------------------------------------------------------------ #
    def _cache_path(self, start: datetime, end: datetime) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_symbol = self.symbol.replace("/", "")
        name = (
            f"{self.exchange_id}_{safe_symbol}_{self.timeframe}"
            f"_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
        )
        return self.cache_dir / name

    async def fetch_history(
        self,
        start: datetime,
        end: datetime | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch paginated OHLCV in ``[start, end]`` as a UTC-indexed DataFrame.

        Returns a DataFrame indexed by candle close time with columns
        :data:`OHLCV_COLUMNS`. Results are cached to parquet keyed by the range.
        """
        end = end or datetime.now(tz=timezone.utc)
        cache_path = self._cache_path(start, end)
        if use_cache and cache_path is not None and cache_path.exists():
            logger.info("Loading OHLCV cache %s", cache_path)
            return pd.read_parquet(cache_path)

        self._client = self._client or self._make_client(pro=False)
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        rows: list[list[float]] = []
        limit = 1000  # exchange max page size

        while since < end_ms:
            batch = await self._client.fetch_ohlcv(  # type: ignore[attr-defined]
                self.symbol, timeframe=self.timeframe, since=since, limit=limit
            )
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1][0] + self._tf_ms
            # Respect rate limits; ccxt also throttles internally.
            await asyncio.sleep(getattr(self._client, "rateLimit", 200) / 1000.0)
            if len(batch) < limit:
                break

        df = self._rows_to_frame(rows)
        df = df.loc[(df.index >= start) & (df.index <= end)]
        if cache_path is not None and not df.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
        logger.info("Fetched %d candles for %s %s", len(df), self.symbol, self.timeframe)
        return df

    @staticmethod
    def _rows_to_frame(rows: list[list[float]]) -> pd.DataFrame:
        """Convert raw ccxt OHLCV rows into a clean, de-duplicated DataFrame."""
        if not rows:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.DataFrame(rows, columns=["ts", *OHLCV_COLUMNS])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.astype(float)

    # ------------------------------------------------------------------ #
    # Live data
    # ------------------------------------------------------------------ #
    async def stream_live(self, poll_interval: float = 1.0) -> AsyncIterator[Bar]:
        """Yield each newly *closed* candle as a :class:`Bar`.

        Prefers ``ccxt.pro``'s websocket ``watch_ohlcv``; if unavailable, falls
        back to REST polling. Only candles strictly older than the most recent
        (still-forming) one are emitted, so no partial bar ever leaks downstream.
        """
        try:
            client = self._make_client(pro=True)
            self._client = client
            async for bar in self._stream_ws(client):
                yield bar
        except RuntimeError:
            logger.warning("ccxt.pro unavailable; falling back to REST polling.")
            async for bar in self._stream_poll(poll_interval):
                yield bar

    async def _stream_ws(self, client) -> AsyncIterator[Bar]:
        last_ts: int | None = None
        while True:
            ohlcv = await client.watch_ohlcv(self.symbol, self.timeframe)
            # The final element is the still-forming candle -> emit the prior one.
            for row in ohlcv[:-1]:
                if last_ts is None or row[0] > last_ts:
                    last_ts = row[0]
                    yield Bar.from_ccxt(row)

    async def _stream_poll(self, poll_interval: float) -> AsyncIterator[Bar]:
        self._client = self._client or self._make_client(pro=False)
        last_ts: int | None = None
        while True:
            batch = await self._client.fetch_ohlcv(  # type: ignore[attr-defined]
                self.symbol, timeframe=self.timeframe, limit=2
            )
            if len(batch) >= 2:
                closed = batch[-2]  # last fully closed candle
                if last_ts is None or closed[0] > last_ts:
                    last_ts = closed[0]
                    yield Bar.from_ccxt(closed)
            await asyncio.sleep(poll_interval)


class HistoricalDataHandler(DataHandler):
    """Replays a cached OHLCV DataFrame as ordered :class:`MarketEvent` objects."""

    def __init__(self, ohlcv: pd.DataFrame) -> None:
        if not ohlcv.index.is_monotonic_increasing:
            ohlcv = ohlcv.sort_index()
        self._ohlcv = ohlcv
        self._bars: list[Bar] = [
            Bar(
                timestamp=ts.to_pydatetime(),
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
            )
            for ts, row in ohlcv.iterrows()
        ]

    @property
    def bars(self) -> list[Bar]:
        return self._bars

    async def stream(self) -> AsyncIterator[MarketEvent]:
        for bar in self._bars:
            yield MarketEvent(bar)


class LiveDataHandler(DataHandler):
    """Adapts a :class:`MarketDataFeed` live stream to the event interface."""

    def __init__(self, feed: MarketDataFeed, poll_interval: float = 1.0) -> None:
        self._feed = feed
        self._poll_interval = poll_interval

    async def stream(self) -> AsyncIterator[MarketEvent]:
        async for bar in self._feed.stream_live(self._poll_interval):
            yield MarketEvent(bar)


def make_synthetic_ohlcv(
    n: int = 5_000,
    start: datetime | None = None,
    seed: int = 7,
    start_price: float = 30_000.0,
) -> pd.DataFrame:
    """Generate a realistic-ish synthetic 1m OHLCV frame for tests/demos.

    Uses a GBM-like random walk with volatility clustering and volume that
    correlates with absolute returns — enough structure to exercise the feature
    engine and backtester without any network access.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    index = pd.date_range(start=start, periods=n, freq="1min", tz="UTC")

    # Volatility clustering via a simple AR(1) on log-variance.
    vol = np.empty(n)
    vol[0] = 0.0006
    for i in range(1, n):
        vol[i] = max(1e-5, 0.95 * vol[i - 1] + 0.05 * 0.0006 + rng.normal(0, 5e-5))
    rets = rng.normal(0, 1, n) * vol
    close = start_price * np.exp(np.cumsum(rets))

    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0004, n)))
    volume = 5 + 200 * np.abs(rets) / vol.mean() + np.abs(rng.normal(0, 3, n))

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
