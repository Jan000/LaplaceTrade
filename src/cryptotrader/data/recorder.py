# src/cryptotrader/data/recorder.py
"""Live market-data recorder.

Records detailed microstructure signals that free historical APIs cannot provide —
multi-level order-book imbalance, microprice, depth/liquidity, spread, recent taker flow,
the cross-venue (Coinbase) premium, funding and open interest — into the ``observations``
table, building a forward dataset that becomes an exclusive training source over time.

Best-effort: each source is fetched in its own try/except so one failing venue or symbol
never stops the recorder.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from cryptotrader.config import Settings
from cryptotrader.data.ingestion import MarketDataFeed
from cryptotrader.persistence import TradeStore

logger = logging.getLogger(__name__)


def _imb(b: float, a: float) -> float | None:
    return (b - a) / (b + a) if (b + a) else None


def _order_book_metrics(ob: dict) -> dict:
    """Multi-level imbalance, microprice, spread and depth/liquidity from an order book."""
    bids, asks = ob.get("bids") or [], ob.get("asks") or []
    if not bids or not asks:
        return {}
    bb, ba = float(bids[0][0]), float(asks[0][0])
    bbv, bav = float(bids[0][1]), float(asks[0][1])
    mid = (bb + ba) / 2.0

    def vol(levels, side):
        return sum(float(a) for _p, a, *_ in side[:levels])

    bv5, av5, bv20, av20 = vol(5, bids), vol(5, asks), vol(20, bids), vol(20, asks)
    micro = (bb * bav + ba * bbv) / (bbv + bav) if (bbv + bav) else mid   # size-weighted mid
    lo, hi = mid * 0.99, mid * 1.01                                       # depth within ±1%
    dbid = sum(float(a) for p, a, *_ in bids if float(p) >= lo)
    dask = sum(float(a) for p, a, *_ in asks if float(p) <= hi)
    return {
        "mid_price": mid, "best_bid": bb, "best_ask": ba,
        "spread_bps": (ba - bb) / mid * 1e4 if mid else None,
        "ob_imbalance": _imb(bv20, av20),          # headline (top 20)
        "imbalance_top5": _imb(bv5, av5),
        "microprice_dev_bps": (micro - mid) / mid * 1e4 if mid else None,
        "bid_vol20": bv20, "ask_vol20": av20,
        "depth_bid_1pct": dbid, "depth_ask_1pct": dask,
        "depth_imbalance_1pct": _imb(dbid, dask),
    }


def _trade_flow_metrics(trades: list) -> dict:
    """Recent aggressive (taker) order flow from the trade tape."""
    if not trades:
        return {}
    buy = sum(float(t.get("amount") or 0) for t in trades if t.get("side") == "buy")
    tot = sum(float(t.get("amount") or 0) for t in trades)
    return {
        "taker_buy_ratio": (buy / tot) if tot else None,
        "trade_count": len(trades),
        "avg_trade_size": (tot / len(trades)) if trades else None,
    }


class MarketRecorder:
    """Periodically samples detailed live microstructure signals and stores them."""

    def __init__(self, settings: Settings, symbols: list[str], interval: float = 120.0,
                 ob_levels: int = 50, call_timeout: float = 20.0) -> None:
        self.settings = settings
        self.symbols = symbols
        self.interval = interval
        self.ob_levels = ob_levels
        self.call_timeout = call_timeout
        self._spot = MarketDataFeed(exchange_id=settings.exchange.id, timeframe="1m",
                                    cache_dir=None)._make_client(pro=False)
        self._cb = MarketDataFeed(exchange_id="coinbase", timeframe="1m",
                                  cache_dir=None)._make_client(pro=False)
        self._fut = MarketDataFeed(exchange_id=settings.exchange.id, timeframe="1m",
                                   cache_dir=None, default_type="future")._make_client(pro=False)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    async def _call(self, coro):
        """Await a network coroutine with a hard timeout so a hung venue can't stall the
        recorder (ccxt's own timeout is belt; this is braces)."""
        return await asyncio.wait_for(coro, timeout=self.call_timeout)

    async def sample_symbol(self, symbol: str) -> dict:
        """Collect one detailed observation dict for ``symbol`` (best-effort per source;
        each source is timeout-guarded and isolated so one slow venue never blocks)."""
        m: dict = {}
        try:
            ob = await self._call(self._spot.fetch_order_book(symbol, limit=self.ob_levels))
            m.update(_order_book_metrics(ob))
        except Exception:
            logger.debug("order book unavailable for %s", symbol, exc_info=True)
        try:
            trades = await self._call(self._spot.fetch_trades(symbol, limit=100))
            m.update(_trade_flow_metrics(trades))
        except Exception:
            logger.debug("trades unavailable for %s", symbol, exc_info=True)
        try:
            t = await self._call(self._spot.fetch_ticker(symbol))
            m["last"] = t.get("last")
            m["quote_volume_24h"] = t.get("quoteVolume")
            m["pct_change_24h"] = t.get("percentage")
        except Exception:
            logger.debug("ticker unavailable for %s", symbol, exc_info=True)
        try:  # cross-venue premium: Coinbase USD vs the Binance USDT price
            base = symbol.split("/")[0]
            cb = await self._call(self._cb.fetch_ticker(f"{base}/USD"))
            ref = m.get("mid_price") or m.get("last")
            if cb.get("last") and ref:
                m["cb_premium"] = (float(cb["last"]) - float(ref)) / float(ref)
        except Exception:
            logger.debug("premium unavailable for %s", symbol, exc_info=True)
        try:
            fr = await self._call(self._fut.fetch_funding_rate(symbol))
            m["funding_rate"] = fr.get("fundingRate")
        except Exception:
            logger.debug("funding unavailable for %s", symbol, exc_info=True)
        try:
            oi = await self._call(self._fut.fetch_open_interest(symbol))
            m["open_interest"] = oi.get("openInterestAmount") or oi.get("openInterestValue")
        except Exception:
            logger.debug("open interest unavailable for %s", symbol, exc_info=True)
        return m

    async def _sample(self, store: TradeStore, symbol: str) -> None:
        m = await self.sample_symbol(symbol)
        if m:
            await store.record_observation(datetime.now(tz=timezone.utc), symbol, **m)

    async def run(self) -> None:
        store = await TradeStore(self.settings.persistence.db_path).connect()
        logger.info("MarketRecorder started: %d symbols every %.0fs", len(self.symbols), self.interval)
        try:
            while not self._stop:
                for sym in self.symbols:
                    if self._stop:
                        break
                    await self._sample(store, sym)
                # responsive stop: sleep in short slices
                slept = 0.0
                while slept < self.interval and not self._stop:
                    await asyncio.sleep(min(1.0, self.interval - slept))
                    slept += 1.0
        finally:
            await store.close()
            await self.aclose()

    async def aclose(self) -> None:
        for c in (self._spot, self._cb, self._fut):
            try:
                await c.close()
            except Exception:
                pass
