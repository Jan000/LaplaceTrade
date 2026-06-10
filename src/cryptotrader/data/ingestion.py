# src/cryptotrader/data/ingestion.py
"""Asynchronous market-data ingestion (ccxt history + live, replay, synthetic)."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cryptotrader.core.events import MarketEvent
from cryptotrader.core.interfaces import DataHandler
from cryptotrader.core.types import Bar

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

_TIMEFRAME_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


class MarketDataFeed:
    """Async ccxt data source for one symbol/timeframe on one exchange."""

    def __init__(
        self,
        exchange_id: str = "binance",
        symbol: str = "BTC/USDT",
        timeframe: str = "15m",
        cache_dir: Path | None = Path(".cache/ohlcv"),
        api_key: str | None = None,
        api_secret: str | None = None,
        default_type: str = "spot",
    ) -> None:
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.cache_dir = cache_dir
        self._api_key = api_key
        self._api_secret = api_secret
        self.default_type = default_type  # "spot" | "future" (USDⓈ-M perps: funding/OI)
        self._tf_ms = _TIMEFRAME_MS.get(timeframe, 60_000)
        self._client: object | None = None

    def _make_client(self, pro: bool = False):
        module_name = "ccxt.pro" if pro else "ccxt.async_support"
        try:
            module = __import__(module_name, fromlist=["dummy"])
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(f"{module_name} is required for live/network data.") from exc
        klass = getattr(module, self.exchange_id)
        # defaultType / fetchMarkets are Binance-specific knobs; other venues (Coinbase,
        # Kraken, …) reject or mis-load markets with them, so only apply on Binance.
        options: dict = {}
        if self.exchange_id.startswith("binance"):
            options["defaultType"] = self.default_type
            if self.default_type == "spot":
                options["fetchMarkets"] = ["spot"]  # faster market load; futures load all
        client = klass(
            {
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "enableRateLimit": True,
                "timeout": 30_000,
                "options": options,
                "aiohttp_trust_env": True,
            }
        )
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if proxy:
            try:
                client.httpsProxy = proxy
            except Exception:  # pragma: no cover
                client.proxies = {"http": proxy, "https": proxy}
        if os.environ.get("CT_INSECURE_SSL", "").lower() in {"1", "true", "yes"}:
            client.verify = False
        if not pro:
            self._force_os_dns_resolver(client)
        return client

    @staticmethod
    def _force_os_dns_resolver(client) -> None:
        """Use aiohttp's OS resolver instead of aiodns (avoids c-ares DNS failures)."""
        try:
            import aiohttp
            from aiohttp.resolver import ThreadedResolver

            ssl_opt = False if getattr(client, "verify", True) is False else None
            connector = aiohttp.TCPConnector(
                resolver=ThreadedResolver(), ssl=ssl_opt, ttl_dns_cache=300
            )
            client.session = aiohttp.ClientSession(connector=connector, trust_env=True)
        except Exception:  # pragma: no cover
            pass

    async def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            await self._client.close()  # type: ignore[func-returns-value]
            self._client = None

    def _cache_path(self, start: datetime, end: datetime) -> Path | None:
        if self.cache_dir is None:
            return None
        safe = self.symbol.replace("/", "")
        name = f"{self.exchange_id}_{safe}_{self.timeframe}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
        return self.cache_dir / name

    async def _fetch_ohlcv_retry(self, since: int, limit: int, attempts: int = 4) -> list:
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await self._client.fetch_ohlcv(  # type: ignore[attr-defined]
                    self.symbol, timeframe=self.timeframe, since=since, limit=limit
                )
            except Exception as exc:
                last_exc = exc
                if attempt == attempts:
                    break
                logger.warning("fetch_ohlcv failed (%s), retry %d/%d in %.1fs",
                               type(exc).__name__, attempt, attempts - 1, delay)
                await asyncio.sleep(delay)
                delay *= 2.0
        assert last_exc is not None
        raise last_exc

    async def fetch_history(self, start: datetime, end: datetime | None = None,
                            use_cache: bool = True) -> pd.DataFrame:
        end = end or datetime.now(tz=timezone.utc)
        cache_path = self._cache_path(start, end)
        if use_cache and cache_path is not None and cache_path.exists():
            logger.info("Loading OHLCV cache %s", cache_path)
            return pd.read_parquet(cache_path)
        self._client = self._client or self._make_client(pro=False)
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        rows: list = []
        limit = 1000
        while since < end_ms:
            batch = await self._fetch_ohlcv_retry(since, limit)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1][0] + self._tf_ms
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
    def _rows_to_frame(rows: list) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = pd.DataFrame(rows, columns=["ts", *OHLCV_COLUMNS])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.astype(float)

    async def stream_live(self, poll_interval: float = 1.0) -> AsyncIterator[Bar]:
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
            for row in ohlcv[:-1]:
                if last_ts is None or row[0] > last_ts:
                    last_ts = row[0]
                    yield Bar.from_ccxt(row)

    async def _stream_poll(self, poll_interval: float) -> AsyncIterator[Bar]:
        self._client = self._client or self._make_client(pro=False)
        last_ts: int | None = None
        while True:
            batch = await self._client.fetch_ohlcv(  # type: ignore[attr-defined]
                self.symbol, timeframe=self.timeframe, limit=2)
            if len(batch) >= 2:
                closed = batch[-2]
                if last_ts is None or closed[0] > last_ts:
                    last_ts = closed[0]
                    yield Bar.from_ccxt(closed)
            await asyncio.sleep(poll_interval)


class HistoricalDataHandler(DataHandler):
    """Replays a cached OHLCV DataFrame as ordered MarketEvent objects."""

    def __init__(self, ohlcv: pd.DataFrame) -> None:
        if not ohlcv.index.is_monotonic_increasing:
            ohlcv = ohlcv.sort_index()
        self._ohlcv = ohlcv
        self._bars: list[Bar] = [
            Bar(ts.to_pydatetime(), row.open, row.high, row.low, row.close, row.volume)
            for ts, row in ohlcv.iterrows()
        ]

    @property
    def bars(self) -> list[Bar]:
        return self._bars

    async def stream(self) -> AsyncIterator[MarketEvent]:
        for bar in self._bars:
            yield MarketEvent(bar)


class ReplayDataHandler(DataHandler):
    """Replays cached OHLCV at an accelerated, wall-clock-paced cadence."""

    def __init__(self, ohlcv: pd.DataFrame, delay: float = 0.02) -> None:
        self._inner = HistoricalDataHandler(ohlcv)
        self._delay = delay

    @property
    def bars(self) -> list[Bar]:
        return self._inner.bars

    async def stream(self) -> AsyncIterator[MarketEvent]:
        for bar in self._inner.bars:
            yield MarketEvent(bar)
            if self._delay > 0:
                await asyncio.sleep(self._delay)


class LiveDataHandler(DataHandler):
    """Adapts a MarketDataFeed live stream to the event interface."""

    def __init__(self, feed: MarketDataFeed, poll_interval: float = 1.0) -> None:
        self._feed = feed
        self._poll_interval = poll_interval

    async def stream(self) -> AsyncIterator[MarketEvent]:
        async for bar in self._feed.stream_live(self._poll_interval):
            yield MarketEvent(bar)


def make_synthetic_ohlcv(n: int = 5_000, start: datetime | None = None, seed: int = 7,
                         start_price: float = 30_000.0) -> pd.DataFrame:
    """Synthetic 1m OHLCV with volatility clustering for tests/demos."""
    import numpy as np

    rng = np.random.default_rng(seed)
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    index = pd.date_range(start=start, periods=n, freq="1min", tz="UTC")
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
