#!/usr/bin/env python
"""Continuously record live microstructure signals into the database.

These signals (order-book imbalance/spread, Coinbase premium, funding) are NOT available
from free historical APIs, so we record them forward to build a future training set. Run
this alongside the bot (or on its own), e.g.:

    python scripts/record_market.py --symbols BTC/USDT ETH/USDT --interval 300

It appends to the same SQLite DB (table ``observations``); inspect progress via the
dashboard ("Experiments"/status) or query the table directly. Stop with Ctrl-C.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cryptotrader.config import Settings  # noqa: E402
from cryptotrader.data.recorder import MarketRecorder  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Record live microstructure signals")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="symbols to record (default: config trade_symbols or exchange.symbol)")
    p.add_argument("--interval", type=float, default=300.0, help="seconds between samples")
    p.add_argument("--ob-levels", type=int, default=20, help="order-book depth levels")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.load()
    symbols = args.symbols or settings.data.trade_symbols or [settings.exchange.symbol]
    recorder = MarketRecorder(settings, symbols, interval=args.interval, ob_levels=args.ob_levels)
    try:
        asyncio.run(recorder.run())
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
