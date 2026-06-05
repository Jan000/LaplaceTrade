# src/cryptotrader/data/sources.py
"""Optional alternative data sources, merged onto the OHLCV bar index.

Everything here is best-effort and fail-safe: any source that errors (network,
unsupported market, geo-block) logs a warning and is skipped, so a missing source
never breaks training — the feature engine zero-fills absent columns.

Sources
-------
* taker_flow   : taker-buy base volume + trade count (Binance klines carry these
                 for free; this is the single most useful add — bar-level order
                 flow with full history).
* funding      : perpetual funding rate (ccxt ``fetch_funding_rate_history``).
* open_interest: perpetual open interest (ccxt ``fetch_open_interest_history``).
* cross_asset  : a second asset's close (e.g. ETH) for return + correlation.

All series are forward-filled onto the OHLCV index (lower-frequency sources like
funding/OI are stepped forward, never interpolated — no look-ahead).
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def _align(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill a (possibly lower-frequency) series onto ``index``."""
    if series is None or series.empty:
        return pd.Series(index=index, dtype=float)
    s = series[~series.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(index)).ffill().reindex(index).ffill().bfill()


async def fetch_taker_flow(client, market_id: str, timeframe: str,
                           start_ms: int, end_ms: int) -> pd.DataFrame:
    """Binance raw klines -> taker_buy_base + num_trades columns (UTC-indexed)."""
    getter = getattr(client, "publicGetKlines", None) or getattr(client, "public_get_klines", None)
    if getter is None:
        raise RuntimeError(
            f"{client.id} has no raw-klines endpoint; taker flow is Binance-only. "
            "Disable use_taker_flow or use exchange=binance."
        )
    rows: list = []
    since = start_ms
    while since < end_ms:
        batch = await getter(
            {"symbol": market_id, "interval": timeframe, "startTime": since, "limit": 1000}
        )
        if not batch:
            break
        rows.extend(batch)
        since = int(batch[-1][0]) + 1
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame()
    # kline = [openTime,o,h,l,c,vol,closeTime,quoteVol,numTrades,takerBuyBase,takerBuyQuote,ignore]
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "taker_buy_base": df[9].astype(float),
            "num_trades": df[8].astype(float),
        },
        index=pd.to_datetime(df[0].astype("int64"), unit="ms", utc=True),
    )
    return out[~out.index.duplicated(keep="last")].sort_index()


async def fetch_funding(client, symbol: str, start_ms: int) -> pd.Series:
    """Perpetual funding-rate history as a time-indexed Series."""
    out: list = []
    since = start_ms
    while True:
        batch = await client.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not batch:
            break
        out.extend(batch)
        since = int(batch[-1]["timestamp"]) + 1
        if len(batch) < 1000:
            break
    if not out:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r["timestamp"] for r in out], unit="ms", utc=True)
    return pd.Series([float(r["fundingRate"]) for r in out], index=idx, name="funding_rate")


async def fetch_open_interest(client, symbol: str, timeframe: str, start_ms: int) -> pd.Series:
    """Perpetual open-interest history as a time-indexed Series."""
    out: list = []
    since = start_ms
    while True:
        batch = await client.fetch_open_interest_history(symbol, timeframe, since=since, limit=500)
        if not batch:
            break
        out.extend(batch)
        since = int(batch[-1]["timestamp"]) + 1
        if len(batch) < 500:
            break
    if not out:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r["timestamp"] for r in out], unit="ms", utc=True)
    vals = [float(r.get("openInterestAmount") or r.get("openInterestValue") or 0.0) for r in out]
    return pd.Series(vals, index=idx, name="open_interest")


async def enrich_ohlcv(settings, ohlcv: pd.DataFrame, start: datetime, feed) -> pd.DataFrame:
    """Merge the enabled optional sources onto ``ohlcv`` (best-effort, fail-safe)."""
    f = settings.features
    if not (f.use_taker_flow or f.use_funding or f.use_open_interest or f.use_cross_asset):
        return ohlcv
    if ohlcv.empty:
        return ohlcv

    out = ohlcv.copy()
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(out.index[-1].timestamp() * 1000) + 1
    client = feed._client or feed._make_client(pro=False)
    feed._client = client
    tf = settings.exchange.timeframe
    symbol = settings.exchange.symbol

    if f.use_taker_flow:
        try:
            market_id = symbol.replace("/", "")
            flow = await fetch_taker_flow(client, market_id, tf, start_ms, end_ms)
            if not flow.empty:
                out["taker_buy_base"] = flow["taker_buy_base"].reindex(out.index).ffill().bfill()
                out["num_trades"] = flow["num_trades"].reindex(out.index).ffill().bfill()
                logger.info("Merged taker-flow (%d rows)", len(flow))
        except Exception:
            logger.warning("taker_flow source unavailable; skipping.", exc_info=True)

    if f.use_funding:
        try:
            fr = await fetch_funding(client, symbol, start_ms)
            out["funding_rate"] = _align(fr, out.index)
            logger.info("Merged funding rate (%d points)", len(fr))
        except Exception:
            logger.warning("funding source unavailable; skipping.", exc_info=True)

    if f.use_open_interest:
        try:
            oi = await fetch_open_interest(client, symbol, tf, start_ms)
            out["open_interest"] = _align(oi, out.index)
            logger.info("Merged open interest (%d points)", len(oi))
        except Exception:
            logger.warning("open_interest source unavailable; skipping.", exc_info=True)

    if f.use_cross_asset:
        try:
            from cryptotrader.data.ingestion import MarketDataFeed

            cross_feed = MarketDataFeed(
                exchange_id=settings.exchange.id, symbol=f.cross_symbol,
                timeframe=tf, cache_dir=settings.data.cache_dir,
            )
            cross = await cross_feed.fetch_history(start)
            await cross_feed.close()
            if not cross.empty:
                out["cross_close"] = cross["close"].reindex(out.index).ffill()
                logger.info("Merged cross-asset %s (%d rows)", f.cross_symbol, len(cross))
        except Exception:
            logger.warning("cross_asset source unavailable; skipping.", exc_info=True)

    return out
