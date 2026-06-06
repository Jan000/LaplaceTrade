# scripts/run_backtest.py
"""Run an end-to-end backtest on synthetic data (no API keys / network needed).

Usage
-----
    python scripts/run_backtest.py                 # rule-based baseline predictor
    python scripts/run_backtest.py --lgbm          # train + use LightGBM
    python scripts/run_backtest.py --bars 20000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from a source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cryptotrader.backtest.engine import EventDrivenBacktester  # noqa: E402
from cryptotrader.config import Settings  # noqa: E402
from cryptotrader.data.features import MicrostructureFeatureEngine  # noqa: E402
from cryptotrader.data.ingestion import make_synthetic_ohlcv  # noqa: E402
from cryptotrader.execution.simulated import SimulatedExecutionHandler  # noqa: E402
from cryptotrader.ml.model import (  # noqa: E402
    LightGBMPredictor,
    MomentumBaselinePredictor,
    make_sample_weights,
    make_triple_barrier_labels,
)
from cryptotrader.risk.manager import ATRRiskManager  # noqa: E402
from cryptotrader.strategy.ml_strategy import MLStrategy  # noqa: E402


def build_feature_engine(settings: Settings) -> MicrostructureFeatureEngine:
    return MicrostructureFeatureEngine(**settings.features.model_dump())


def main() -> None:
    parser = argparse.ArgumentParser(description="CryptoTrader backtest runner")
    parser.add_argument("--bars", type=int, default=8000, help="synthetic bar count")
    parser.add_argument("--lgbm", action="store_true", help="train and use LightGBM")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = Settings.load()
    ohlcv = make_synthetic_ohlcv(n=args.bars, seed=args.seed)
    feature_engine = build_feature_engine(settings)

    if args.lgbm:
        features = feature_engine.transform(ohlcv)
        label_tp, label_sl = settings.barriers.label_barriers
        labels, t1 = make_triple_barrier_labels(
            ohlcv, features["atr"], horizon=settings.barriers.horizon,
            tp_mult=label_tp, sl_mult=label_sl,
            return_events=True,
        )
        weights = make_sample_weights(t1)
        predictor: object = LightGBMPredictor(settings.model.to_lgbm_params())
        predictor.train(features, labels, eval_fraction=settings.model.eval_fraction,  # type: ignore[attr-defined]
                        sample_weight=weights)
    else:
        predictor = MomentumBaselinePredictor()

    strategy = MLStrategy(predictor, settings.strategy, settings.exchange.symbol)
    risk = ATRRiskManager(settings.risk, settings.barriers, settings.execution)
    execution = SimulatedExecutionHandler(settings.execution)

    backtester = EventDrivenBacktester(
        ohlcv=ohlcv,
        feature_engine=feature_engine,
        strategy=strategy,
        risk_manager=risk,
        execution_handler=execution,
        settings=settings,
    )
    result = backtester.run()
    print(json.dumps(result.report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
