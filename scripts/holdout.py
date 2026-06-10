# scripts/holdout.py
"""Out-of-sample honesty check — train ONCE, test on untouched data.

Two forms of out-of-sample, both stricter than the rolling walk-forward (which can
quietly be over-read across many folds/configs):

1. **Temporal holdout** — split the history once at ``--train-frac``; train on the
   oldest part, test on the most recent contiguous block we never trained on.
2. **Cross-asset holdout** — apply the *same* BTC-trained pipeline, unchanged, to
   assets that were never in the training set (SOL, BNB, XRP, ADA ...). If the edge
   survives on coins the model has never seen, it is a general microstructure effect
   rather than something fit to BTC.

The pipeline (features, labels, regularised seed-ensemble, ETH pooling) is exactly the
one in walkforward.py — imported, not re-implemented, so this can't drift from it.

Usage:
    python scripts/holdout.py
    python scripts/holdout.py --train-frac 0.7 --extra SOL/USDT BNB/USDT XRP/USDT ADA/USDT
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so we can reuse walkforward.py

from cryptotrader.config import Settings  # noqa: E402
from cryptotrader.data.ingestion import MarketDataFeed  # noqa: E402
from walkforward import backtest, train_predictor  # noqa: E402

logger = logging.getLogger("holdout")


async def fetch(settings: Settings, symbol: str, days: int):
    feed = MarketDataFeed(
        exchange_id=settings.exchange.id, symbol=symbol,
        timeframe=settings.exchange.timeframe, cache_dir=settings.data.cache_dir,
    )
    try:
        return await feed.fetch_history(datetime.now(tz=timezone.utc) - timedelta(days=days))
    finally:
        await feed.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Out-of-sample holdout check")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--symbol", type=str, default=None, help="primary symbol (overrides config)")
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="fraction of history used for the single training split")
    parser.add_argument("--extra", nargs="*", default=["SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT"],
                        help="unseen symbols to cross-asset test (not in training)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.load()
    if args.symbol:
        settings.exchange.symbol = args.symbol
    days = args.days if args.days is not None else settings.data.history_days
    primary = settings.exchange.symbol
    pool = list(settings.data.train_symbols)

    # Symbols we need data for: primary + pooled (training) + unseen (test only).
    unseen = [s for s in args.extra if s not in {primary, *pool}]
    need = list(dict.fromkeys([primary, *pool, *unseen]))
    data = {s: asyncio.run(fetch(settings, s, days)) for s in need}
    data = {s: d for s, d in data.items() if d is not None and not d.empty}

    btc = data[primary]
    cutoff = int(len(btc) * args.train_frac)
    cutoff_ts = btc.index[cutoff]

    # Train ONCE: primary up to the cutoff + each pooled symbol sliced to the same cutoff.
    train_frames = [btc.iloc[:cutoff]]
    for s in pool:
        if s in data:
            train_frames.append(data[s].loc[data[s].index < cutoff_ts])
    print(f"Train once: {primary} + pool {pool or 'none'} up to {cutoff_ts:%Y-%m-%d} "
          f"({sum(len(f) for f in train_frames)} pooled bars). "
          f"Holdout = bars on/after the cutoff.\n")
    predictor = train_predictor(settings, train_frames)

    print(f"{'symbol':<12}{'role':<14}{'bars':>7}{'return%':>10}{'PF':>7}"
          f"{'trades':>8}{'win%':>7}{'maxDD%':>8}")
    print("-" * 73)
    for s in need:
        df = data.get(s)
        if df is None:
            continue
        test = df.loc[df.index >= cutoff_ts]
        if len(test) < 50:
            continue
        rep = backtest(settings, test, predictor)
        role = "primary" if s == primary else ("pooled" if s in pool else "UNSEEN asset")
        pf = rep["profit_factor"]
        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        print(f"{s:<12}{role:<14}{len(test):>7}{rep['total_return_pct']:>10.1f}{pf_s:>7}"
              f"{rep['n_trades']:>8}{rep['win_rate'] * 100:>6.1f}{rep['max_drawdown_pct']:>8.1f}")
        if s == primary:  # persist the (out-of-time) result for the dashboard + experiment log
            result = {
                "timeframe": settings.exchange.timeframe,
                "return_pct": round(rep["total_return_pct"], 2),
                "profit_factor": None if pf == float("inf") else round(pf, 3),
                "win_rate": round(rep["win_rate"], 4),
                "n_trades": rep["n_trades"],
                "max_drawdown_pct": round(rep["max_drawdown_pct"], 2),
            }
            try:
                from cryptotrader.ml.registry import write_validation

                write_validation("holdout", primary, result)
            except Exception:
                pass
            from cryptotrader.ml.experiments import log_experiment

            log_experiment("holdout", primary, settings, result)
    print("\nUNSEEN-asset rows are a strict out-of-sample check: those coins were never "
          "in training. A positive edge there means the signal generalises.")


if __name__ == "__main__":
    main()
