# src/cryptotrader/data/sources.py
"""Optional alternative data sources, merged onto the OHLCV bar index.

Everything here is best-effort and fail-safe: any source that errors or fails to
align logs a warning and is skipped, so a missing source never breaks training —
the feature engine zero-fills absent/empty columns.

Sources: taker_flow (Binance klines: taker-buy base volume + trade count),
funding (perp funding rate), open_interest (perp OI), cross_asset (2nd asset).
Lower-frequency sources are forward-filled onto the bar index (no look-ahead).
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def _to_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    # Normalise to nanosecond resolution. pandas 3.0's reindex/align hashtable can
    # silently fail to match otherwise-identical datetime64[ms]/[us] tz-aware
    # indexes — Index.intersection matches but Series.reindex returns all-NaN
    # (the cause of the "fetched but 0 aligned" taker-flow bug). Forcing a common
    # ns unit on both sides routes the lookup through the working code path.
    try:
        idx = idx.as_unit("ns")
    except (AttributeError, ValueError):  # pragma: no cover - older pandas
        pass
    return idx


def _exact_or_nearest(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Align a same-frequency series onto ``index``: exact, else nearest within 1 bar."""
    if series is None or series.empty:
        return pd.Series(index=index, dtype=float)
    s = series[~series.index.duplicated(keep="last")].sort_index()
    s.index = _to_utc(s.index)
    col = s.reindex(index)
    if col.notna().sum() == 0 and len(index) > 1:
        tol = index[1] - index[0]
        col = s.reindex(index, method="nearest", tolerance=tol)
    return col


def _align_ffill(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill a lower-frequency series onto ``index`` (step forward, no leak)."""
    if series is None or series.empty:
        return pd.Series(index=index, dtype=float)
    s = series[~series.index.duplicated(keep="last")].sort_index()
    s.index = _to_utc(s.index)
    return s.reindex(s.index.union(index)).ffill().reindex(index).ffill().bfill()


async def fetch_taker_flow(client, market_id: str, timeframe: str,
                           start_ms: int, end_ms: int) -> pd.DataFrame:
    """Binance raw klines -> taker_buy_base + num_trades columns (UTC-indexed)."""
    getter = getattr(client, "publicGetKlines", None) or getattr(client, "public_get_klines", None)
    if getter is None:
        raise RuntimeError(
            f"{client.id} has no raw-klines endpoint; taker flow is Binance-only."
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
    df = pd.DataFrame(rows)
    # kline = [openTime,o,h,l,c,vol,closeTime,quoteVol,numTrades,takerBuyBase,takerBuyQuote,ignore]
    # NB: take .to_numpy() before building the frame. df[9] etc. carry a RangeIndex;
    # passing them as Series alongside an explicit datetime ``index=`` makes pandas
    # re-align them onto that index (no overlap) and silently fill every value with
    # NaN — the real cause of the "fetched but 0 aligned" taker-flow bug.
    out = pd.DataFrame(
        {
            "taker_buy_base": df[9].astype(float).to_numpy(),
            "num_trades": df[8].astype(float).to_numpy(),
        },
        index=pd.to_datetime(df[0].astype("int64").to_numpy(), unit="ms", utc=True),
    )
    return out[~out.index.duplicated(keep="last")].sort_index()


async def fetch_funding(client, symbol: str, start_ms: int) -> pd.Series:
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


def _diag_misalign(name: str, src_index, out_index) -> None:
    logger.warning(
        "%s fetched but did not align with the bars (0 overlap). "
        "First source ts=%s ; first bar ts=%s. Check timeframe/timezone.",
        name, list(src_index[:3]), list(out_index[:3]),
    )


async def enrich_ohlcv(settings, ohlcv: pd.DataFrame, start: datetime, feed) -> pd.DataFrame:
    """Merge the enabled optional sources onto ``ohlcv`` (best-effort, fail-safe)."""
    f = settings.features
    if not (f.use_taker_flow or f.use_funding or f.use_open_interest or f.use_cross_asset):
        return ohlcv
    if ohlcv.empty:
        return ohlcv

    out = ohlcv.copy()
    out.index = _to_utc(out.index)  # normalise so merges align
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(out.index[-1].timestamp() * 1000) + 1
    client = feed._client or feed._make_client(pro=False)
    feed._client = client
    tf = settings.exchange.timeframe
    symbol = settings.exchange.symbol

    if f.use_taker_flow:
        try:
            flow = await fetch_taker_flow(client, symbol.replace("/", ""), tf, start_ms, end_ms)
            if not flow.empty:
                tb = _exact_or_nearest(flow["taker_buy_base"], out.index)
                nt = _exact_or_nearest(flow["num_trades"], out.index)
                if tb.notna().sum() == 0:
                    _diag_misalign("taker_flow", flow.index, out.index)
                out["taker_buy_base"] = tb
                out["num_trades"] = nt
                logger.info("Merged taker-flow (%d rows, %d aligned)", len(flow), int(tb.notna().sum()))
        except Exception:
            logger.warning("taker_flow source unavailable; skipping.", exc_info=True)

    if f.use_funding:
        try:
            fr = await fetch_funding(client, symbol, start_ms)
            out["funding_rate"] = _align_ffill(fr, out.index)
            logger.info("Merged funding rate (%d points)", len(fr))
        except Exception:
            logger.warning("funding source unavailable; skipping.", exc_info=True)

    if f.use_open_interest:
        try:
            oi = await fetch_open_interest(client, symbol, tf, start_ms)
            out["open_interest"] = _align_ffill(oi, out.index)
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
                cc = cross["close"].copy()
                cc.index = _to_utc(cc.index)
                out["cross_close"] = _exact_or_nearest(cc, out.index).ffill()
                logger.info("Merged cross-asset %s (%d rows)", f.cross_symbol, len(cross))
        except Exception:
            logger.warning("cross_asset source unavailable; skipping.", exc_info=True)

    return out
