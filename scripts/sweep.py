# scripts/sweep.py
"""Grid-search backtest parameters and print a ranked table.

For each ``tp_mult`` the model is (re)trained once on the training slice, then the
held-out slice is backtested across every combination of ``min_edge_cost_ratio``,
entry ``threshold`` and ``taker_fee`` — those are backtest-only knobs, so they
need no retraining. Results are ranked so the best operating point is obvious.

Edit the GRID_* lists below, then run:
    python scripts/sweep.py --days 365
    python scripts/sweep.py --synthetic --bars 20000      # offline smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
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

logger = logging.getLogger("sweep")

# ---- Grids to search (edit freely) -------------------------------------
GRID_TP_MULT = [1.5, 2.0, 2.5, 3.0]          # retrains per value (affects labels)
GRID_MIN_EDGE = [2.0, 4.0, 8.0]              # backtest-only
GRID_THRESHOLD = [0.58, 0.64, 0.70]          # backtest-only
GRID_TAKER_FEE = [0.0004, 0.0002]            # backtest-only (taker vs maker-ish)


def feature_engine(settings: Settings) -> MicrostructureFeatureEngine:
    return MicrostructureFeatureEngine(**settings.features.model_dump())


async def load_ohlcv(settings: Settings, args: argparse.Namespace):
    if args.synthetic:
        return make_synthetic_ohlcv(n=args.bars, seed=args.seed)
    feed = MarketDataFeed(
        exchange_id=args.exchange or settings.exchange.id,
        symbol=args.symbol or settings.exchange.symbol,
        timeframe=args.timeframe or settings.exchange.timeframe,
        cache_dir=settings.data.cache_dir,
    )
    start = datetime.now(tz=timezone.utc) - timedelta(days=args.days)
    try:
        return await feed.fetch_history(start)
    finally:
        await feed.close()


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
    parser = argparse.ArgumentParser(description="Parameter sweep for CryptoTrader")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--exchange", type=str, default=None)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--bars", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.load()
    ohlcv = asyncio.run(load_ohlcv(settings, args))

    split = int(len(ohlcv) * (1.0 - settings.model.test_fraction))
    train_ohlcv, test_ohlcv = ohlcv.iloc[:split], ohlcv.iloc[split:]
    print(f"Data: {len(train_ohlcv)} train / {len(test_ohlcv)} test bars\n")

    rows: list[tuple] = []
    for tp in GRID_TP_MULT:
        s = settings.model_copy(deep=True)
        s.barriers.tp_mult = tp
        feats = feature_engine(s).transform(train_ohlcv)
        labels, t1 = make_triple_barrier_labels(
            train_ohlcv, feats["atr"], horizon=s.barriers.horizon,
            tp_mult=tp, sl_mult=s.barriers.sl_mult, return_events=True,
        )
        weights = make_sample_weights(t1)
        valid = labels.index[: len(labels) - s.barriers.horizon]
        if s.model.use_meta_labeling:
            from cryptotrader.ml.meta import train_meta_labeled

            predictor, _ = train_meta_labeled(
                feats.loc[valid], labels.loc[valid], s.model.to_lgbm_params(),
                eval_fraction=s.model.eval_fraction, embargo=s.barriers.horizon,
                sample_weight=weights.loc[valid],
            )
        else:
            predictor = LightGBMPredictor(s.model.to_lgbm_params())
            predictor.train(feats.loc[valid], labels.loc[valid],
                            eval_fraction=s.model.eval_fraction, sample_weight=weights.loc[valid])
        print(f"trained tp_mult={tp} ...")

        for min_edge, thr, fee in itertools.product(GRID_MIN_EDGE, GRID_THRESHOLD, GRID_TAKER_FEE):
            cfg = s.model_copy(deep=True)
            cfg.risk.min_edge_cost_ratio = min_edge
            cfg.strategy.long_threshold = cfg.strategy.short_threshold = thr
            cfg.execution.taker_fee = fee
            rep = backtest(cfg, test_ohlcv, predictor)
            rows.append((
                tp, min_edge, thr, fee, rep["total_return_pct"],
                rep["profit_factor"], rep["n_trades"], rep["win_rate"],
            ))

    rows.sort(key=lambda r: r[4], reverse=True)  # best total_return_pct first
    print(f"\n{'tp':>4}{'min_edge':>10}{'thresh':>8}{'fee':>9}"
          f"{'return%':>10}{'PF':>7}{'trades':>8}{'win%':>7}")
    print("-" * 63)
    for tp, me, thr, fee, ret, pf, n, win in rows[:25]:
        print(f"{tp:>4}{me:>10}{thr:>8}{fee:>9}{ret:>10.1f}{pf:>7.2f}{n:>8}{win*100:>6.1f}")
    best = rows[0]
    print(f"\nBest: tp_mult={best[0]} min_edge={best[1]} threshold={best[2]} "
          f"taker_fee={best[3]} -> return {best[4]:.1f}%  PF {best[5]:.2f}  ({best[6]} trades)")


if __name__ == "__main__":
    main()
