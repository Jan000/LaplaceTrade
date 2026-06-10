# src/cryptotrader/data/recorder.py
"""Live market-data recorder.

Records the signals that are only available *live* — order-book imbalance & spread, the
cross-venue (Coinbase) premium, and current funding — into the ``observations`` table, to
build a forward dataset that free historical APIs cannot provide. Run it continuously
alongside (or independently of) trading; after a few weeks/months the accumulated rows
become a new, exclusive training source.

Best-effort by design: each source is fetched in its own try/except so one failing venue
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


def _order_book_metrics(ob: dict, levels: int = 20) -> dict:
    bids, asks = ob.get("bids") or [], ob.get("asks") or []
    if not bids or not asks:
        return {}
    bid_vol = sum(float(a) for _, a, *_ in bids[:levels])
    ask_vol = sum(float(a) for _, a, *_ in asks[:levels])
    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0
    tot = bid_vol + ask_vol
    return {
        "mid_price": mid,
        "spread_bps": (best_ask - best_bid) / mid * 1e4 if mid else None,
        "ob_imbalance": (bid_vol - ask_vol) / tot if tot else None,
    }


class MarketRecorder:
    """Periodically samples live microstructure signals and stores them."""

    def __init__(self, settings: Settings, symbols: list[str], interval: float = 300.0,
                 ob_levels: int = 20) -> None:
        self.settings = settings
        self.symbols = symbols
        self.interval = interval
        self.ob_levels = ob_levels
        self._spot = MarketDataFeed(exchange_id=settings.exchange.id, timeframe="1m",
                                    cache_dir=None)._make_client(pro=False)
        self._cb = MarketDataFeed(exchange_id="coinbase", timeframe="1m",
                                  cache_dir=None)._make_client(pro=False)
        self._fut = MarketDataFeed(exchange_id=settings.exchange.id, timeframe="1m",
                                   cache_dir=None, default_type="future")._make_client(pro=False)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    async def _sample(self, store: TradeStore, symbol: str) -> None:
        metrics: dict = {}
        try:
            ob = await self._spot.fetch_order_book(symbol, limit=self.ob_levels)
            metrics.update(_order_book_metrics(ob, self.ob_levels))
        except Exception:
            logger.debug("order book unavailable for %s", symbol, exc_info=True)
        try:  # cross-venue premium: Coinbase USD vs the spot USDT price
            base = symbol.split("/")[0]
            cb = await self._cb.fetch_ticker(f"{base}/USD")
            binp = metrics.get("mid_price") or (await self._spot.fetch_ticker(symbol)).get("last")
            cbp = cb.get("last")
            if cbp and binp:
                metrics["cb_premium"] = (float(cbp) - float(binp)) / float(binp)
        except Exception:
            logger.debug("premium unavailable for %s", symbol, exc_info=True)
        try:
            fr = await self._fut.fetch_funding_rate(symbol)
            metrics["funding_rate"] = fr.get("fundingRate")
        except Exception:
            logger.debug("funding unavailable for %s", symbol, exc_info=True)

        if metrics:
            await store.record_observation(datetime.now(tz=timezone.utc), symbol, **metrics)
            logger.info("recorded %s: %s", symbol,
                        {k: round(v, 6) for k, v in metrics.items() if v is not None})

    async def run(self) -> None:
        store = await TradeStore(self.settings.persistence.db_path).connect()
        logger.info("MarketRecorder started: %s every %.0fs", ", ".join(self.symbols), self.interval)
        try:
            while not self._stop:
                for sym in self.symbols:
                    await self._sample(store, sym)
                await asyncio.sleep(self.interval)
        finally:
            await store.close()
            for c in (self._spot, self._cb, self._fut):
                try:
                    await c.close()
                except Exception:
                    pass
