# scripts/walkforward.py
"""Walk-forward (out-of-sample) validation — the honest test of an edge.

History is split into N sequential folds. For each fold the model is trained
*only* on the data that precedes it (an expanding/anchored window) and then
backtested on the fold's unseen bars, using the single configured parameter set
(no per-fold tuning — that would re-introduce look-ahead/selection bias). The
out-of-sample fold returns are then compounded, as if you had retrained and kept
trading forward through time.

If the strategy stays positive across most folds, the edge generalises. If only
one fold carries it, the sweep optimum was overfit to that period.

Usage:
    python scripts/walkforward.py                       # uses config (1h, 730d)
    python scripts/walkforward.py --splits 6 --train-frac 0.4
    python scripts/walkforward.py --synthetic --bars 30000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cryptotrader.backtest.engine import EventDrivenBacktester  # noqa: E402
from cryptotrader.config import Settings  # noqa: E402
from cryptotrader.data.features import MicrostructureFeatureEngine  # noqa: E402
from cryptotrader.data.ingestion import MarketDataFeed, make_synthetic_ohlcv  # noqa: E402
from cryptotrader.execution.simulated import SimulatedExecutionHandler  # noqa: E402
from cryptotrader.ml.model import (  # noqa: E402
    LightGBMPredictor,
    make_sample_weights,
    make_triple_barrier_labels,
)
from cryptotrader.risk.manager import ATRRiskManager  # noqa: E402
from cryptotrader.strategy.ml_strategy import MLStrategy  # noqa: E402

logger = logging.getLogger("walkforward")


def feature_engine(settings: Settings) -> MicrostructureFeatureEngine:
    return MicrostructureFeatureEngine(**settings.features.model_dump())


async def load_ohlcv(settings: Settings, args: argparse.Namespace):
    if args.synthetic:
        return make_synthetic_ohlcv(n=args.bars, seed=args.seed)
    days = args.days if args.days is not None else settings.data.history_days
    feed = MarketDataFeed(
        exchange_id=args.exchange or settings.exchange.id,
        symbol=args.symbol or settings.exchange.symbol,
        timeframe=args.timeframe or settings.exchange.timeframe,
        cache_dir=settings.data.cache_dir,
    )
    start = datetime.now(tz=timezone.utc) - timedelta(days=days)
    try:
        df = await feed.fetch_history(start)
        if not df.empty:
            from cryptotrader.data.sources import enrich_ohlcv

            df = await enrich_ohlcv(settings, df, start, feed)
        return df
    finally:
        await feed.close()


def train_predictor(settings: Settings, train_ohlcv):
    feats = feature_engine(settings).transform(train_ohlcv)
    label_tp, label_sl = settings.barriers.label_barriers
    labels, t1 = make_triple_barrier_labels(
        train_ohlcv, feats["atr"], horizon=settings.barriers.horizon,
        tp_mult=label_tp, sl_mult=label_sl,
        return_events=True,
    )
    weights = make_sample_weights(t1)
    valid = labels.index[: len(labels) - settings.barriers.horizon]
    if settings.model.use_meta_labeling:
        from cryptotrader.ml.meta import train_meta_labeled

        predictor, _ = train_meta_labeled(
            feats.loc[valid], labels.loc[valid], settings.model.to_lgbm_params(),
            eval_fraction=settings.model.eval_fraction, embargo=settings.barriers.horizon,
            sample_weight=weights.loc[valid],
        )
        return predictor
    n_members = max(1, settings.model.ensemble_size)
    members = []
    for k in range(n_members):
        params = settings.model.to_lgbm_params()
        params["random_state"] = settings.model.random_state + k
        m = LightGBMPredictor(params, drop_features=settings.model.drop_features)
        m.train(feats.loc[valid], labels.loc[valid],
                eval_fraction=settings.model.eval_fraction, sample_weight=weights.loc[valid])
        members.append(m)
    if n_members == 1:
        return members[0]
    from cryptotrader.ml.model import EnsemblePredictor

    return EnsemblePredictor(members)


def backtest(settings: Settings, test_ohlcv, predictor) -> dict:
    bt = EventDrivenBacktester(
        ohlcv=test_ohlcv,
        feature_engine=feature_engine(settings),
        strategy=MLStrategy(predictor, settings.strategy, settings.exchange.symbol),
        risk_manager=ATRRiskManager(settings.risk, settings.barriers, settings.execution),
        execution_handler=SimulatedExecutionHandler(settings.execution),
        settings=settings,
    )
    return bt.run().report.as_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--exchange", type=str, default=None)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--bars", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--splits", type=int, default=5, help="number of OOS folds")
    parser.add_argument("--train-frac", type=float, default=0.5,
                        help="fraction used for the first (smallest) training window")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # Surface data-source merge info even at WARNING level.
    logging.getLogger("cryptotrader.data.sources").setLevel(logging.INFO)
    settings = Settings.load()
    ohlcv = asyncio.run(load_ohlcv(settings, args))

    # --- Data-source diagnostic: make it unmistakable what is actually active.
    fe_probe = feature_engine(settings)
    flags = {
        "taker_flow": settings.features.use_taker_flow,
        "funding": settings.features.use_funding,
        "open_interest": settings.features.use_open_interest,
        "cross_asset": settings.features.use_cross_asset,
    }
    on = [k for k, v in flags.items() if v]
    print(f"Extra sources enabled: {on or 'none'}  |  model features: {len(fe_probe.feature_names)}")
    for col in ("taker_buy_base", "num_trades", "funding_rate", "open_interest", "cross_close"):
        if col in ohlcv.columns:
            nn = int(ohlcv[col].notna().sum())
            print(f"  column '{col}': {nn}/{len(ohlcv)} rows non-null"
                  + ("  <-- LOADED" if nn > len(ohlcv) // 2 else "  <-- MOSTLY EMPTY"))
    if on and not any(c in ohlcv.columns for c in
                      ("taker_buy_base", "funding_rate", "open_interest", "cross_close")):
        print("  WARNING: sources enabled but NO source columns present "
              "-> the fetch returned nothing (features will be neutral 0).")
    print()

    total = len(ohlcv)
    initial = int(total * args.train_frac)
    test_len = (total - initial) // args.splits
    if test_len < 200:
        print(f"WARNING: only {test_len} bars per OOS fold — results will be noisy. "
              f"Use more --days or fewer --splits.")

    print(f"Walk-forward: {total} bars, {args.splits} folds, "
          f"first train {initial} bars, ~{test_len} test bars/fold\n")
    print(f"{'fold':>4}{'train_bars':>12}{'test_bars':>11}{'return%':>10}"
          f"{'PF':>7}{'trades':>8}{'win%':>7}{'maxDD%':>8}")
    print("-" * 67)

    equity_mult = 1.0
    returns: list[float] = []
    pfs: list[float] = []
    for i in range(args.splits):
        train_end = initial + i * test_len
        test_start = train_end
        test_end = total if i == args.splits - 1 else test_start + test_len
        train_ohlcv = ohlcv.iloc[:train_end]
        test_ohlcv = ohlcv.iloc[test_start:test_end]

        predictor = train_predictor(settings, train_ohlcv)
        rep = backtest(settings, test_ohlcv, predictor)

        r = rep["total_return_pct"]
        equity_mult *= 1.0 + r / 100.0
        returns.append(r)
        pf = rep["profit_factor"]
        if pf != float("inf"):
            pfs.append(pf)
        print(f"{i + 1:>4}{train_end:>12}{len(test_ohlcv):>11}{r:>10.1f}"
              f"{pf:>7.2f}{rep['n_trades']:>8}{rep['win_rate'] * 100:>6.1f}"
              f"{rep['max_drawdown_pct']:>8.1f}")

    compounded = (equity_mult - 1.0) * 100.0
    pos_folds = sum(1 for r in returns if r > 0)
    mean_r = sum(returns) / len(returns) if returns else 0.0
    mean_pf = sum(pfs) / len(pfs) if pfs else 0.0

    print("-" * 67)
    print(f"\nOut-of-sample summary ({args.splits} folds):")
    print(f"  compounded OOS return : {compounded:>8.1f} %")
    print(f"  mean fold return      : {mean_r:>8.1f} %")
    print(f"  positive folds        : {pos_folds}/{args.splits}")
    print(f"  mean profit factor    : {mean_pf:>8.2f}")
    verdict = (
        "ROBUST — edge holds out of sample." if pos_folds >= args.splits - 1 and compounded > 0
        else "MIXED — positive overall but unstable across folds." if compounded > 0
        else "NOT ROBUST — the sweep optimum did not generalise."
    )
    print(f"  verdict               : {verdict}\n")


if __name__ == "__main__":
    main()
