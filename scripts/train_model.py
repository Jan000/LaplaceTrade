# scripts/train_model.py
"""Train the LightGBM intraday model on real exchange data and evaluate it.

Pipeline
--------
1. **Fetch** real 1m OHLCV from the configured exchange via ccxt (cached to
   parquet). Use ``--synthetic`` to run the whole pipeline offline on generated
   data (useful for smoke-testing without network).
2. **Split** chronologically into a training slice and a held-out test slice —
   strictly no shuffling, so evaluation never sees the future.
3. **Features + labels**: compute the micro-structure feature matrix and
   triple-barrier labels (forward-looking targets are fine; they only define the
   *label*, never leak into features).
4. **Train** LightGBM on the training slice and **save** the model.
5. **Backtest** the trained model on the held-out slice and compare it against
   the rule-based baseline on the very same data.
6. **Persist** the held-out slice to parquet so the dashboard's *simulation* mode
   can replay exactly what the model was evaluated on.

Usage
-----
    python scripts/train_model.py --days 45                 # real Binance data
    python scripts/train_model.py --synthetic --bars 20000  # offline smoke test
    python scripts/train_model.py --days 60 --symbol ETH/USDT
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cryptotrader.backtest.engine import EventDrivenBacktester  # noqa: E402
from cryptotrader.config import Settings  # noqa: E402
from cryptotrader.data.features import MicrostructureFeatureEngine  # noqa: E402
from cryptotrader.data.ingestion import MarketDataFeed, make_synthetic_ohlcv  # noqa: E402
from cryptotrader.execution.simulated import SimulatedExecutionHandler  # noqa: E402
from cryptotrader.ml.model import (  # noqa: E402
    LightGBMPredictor,
    MomentumBaselinePredictor,
    make_labels,
    make_sample_weights,
)
from cryptotrader.risk.manager import ATRRiskManager  # noqa: E402
from cryptotrader.strategy.ml_strategy import MLStrategy  # noqa: E402

logger = logging.getLogger("train")


def build_feature_engine(settings: Settings) -> MicrostructureFeatureEngine:
    # Every indicator window comes straight from FeatureConfig -> fully tunable.
    return MicrostructureFeatureEngine(**settings.features.model_dump())


async def load_ohlcv(settings: Settings, args: argparse.Namespace) -> pd.DataFrame:
    """Return the OHLCV frame to train on (real exchange data or synthetic)."""
    if args.synthetic:
        logger.info("Using synthetic OHLCV (%d bars)", args.bars)
        return make_synthetic_ohlcv(n=args.bars, seed=args.seed)

    exchange_id = args.exchange or settings.exchange.id
    symbol = args.symbol or settings.exchange.symbol
    feed = MarketDataFeed(
        exchange_id=exchange_id,
        symbol=symbol,
        timeframe=settings.exchange.timeframe,
        cache_dir=settings.data.cache_dir,
        api_key=settings.exchange.api_key,
        api_secret=settings.exchange.api_secret,
    )
    days = args.days if args.days is not None else settings.data.history_days
    start = datetime.now(tz=timezone.utc) - timedelta(days=days)
    try:
        df = await feed.fetch_history(start)
        if not df.empty:
            from cryptotrader.data.sources import enrich_ohlcv

            df = await enrich_ohlcv(settings, df, start, feed)  # optional extra sources
    except Exception as exc:  # network / TLS / proxy / geo-block / bad symbol
        cause = exc.__cause__
        cause_txt = f"{type(cause).__name__}: {cause}" if cause is not None else "(none)"
        raise SystemExit(
            f"\nCould not fetch data from '{exchange_id}'.\n"
            f"  error      : {type(exc).__name__}: {exc}\n"
            f"  root cause : {cause_txt}\n\n"
            "If the root cause mentions SSL/CERTIFICATE (very common on corporate or\n"
            "antivirus-protected Windows that intercept TLS), Python doesn't trust your\n"
            "network's root CA. Fixes, in order of preference:\n"
            "  1) Use the Windows certificate store:   pip install pip-system-certs\n"
            "  2) Point Python at a CA bundle:          set SSL_CERT_FILE=C:\\path\\to\\ca.pem\n"
            "  3) Quick confirmation only (insecure):   set CT_INSECURE_SSL=1\n\n"
            "If it mentions PROXY/timeout, set HTTPS_PROXY to your corporate proxy.\n"
            "If it really is a geo-block (HTTP 451): --exchange kraken | coinbase | binanceus.\n"
            "To work fully offline: --synthetic.\n"
        ) from exc
    finally:
        await feed.close()
    if df.empty:
        raise SystemExit(
            f"No data returned from '{exchange_id}' for {symbol}. Check the symbol "
            "(e.g. BTC/USD on Kraken/Coinbase), or run with --synthetic."
        )
    logger.info("Fetched %d real candles from %s for %s", len(df), exchange_id, symbol)
    return df


async def load_extra_symbols(settings: Settings, args: argparse.Namespace) -> dict:
    """Fetch each ``data.train_symbols`` symbol (same timeframe/window) for training pooling."""
    out: dict = {}
    days = args.days if args.days is not None else settings.data.history_days
    start = datetime.now(tz=timezone.utc) - timedelta(days=days)
    primary = args.symbol or settings.exchange.symbol
    for sym in settings.data.pool_for(primary):
        feed = MarketDataFeed(
            exchange_id=args.exchange or settings.exchange.id, symbol=sym,
            timeframe=settings.exchange.timeframe, cache_dir=settings.data.cache_dir,
        )
        try:
            d = await feed.fetch_history(start)
            if not d.empty:
                out[sym] = d
        finally:
            await feed.close()
    return out


def backtest(settings: Settings, ohlcv: pd.DataFrame, predictor) -> dict:
    """Run a backtest of ``predictor`` over ``ohlcv`` and return the report dict."""
    fe = build_feature_engine(settings)
    strategy = MLStrategy(predictor, settings.strategy, settings.exchange.symbol)
    bt = EventDrivenBacktester(
        ohlcv=ohlcv,
        feature_engine=fe,
        strategy=strategy,
        risk_manager=ATRRiskManager(settings.risk, settings.barriers, settings.execution),
        execution_handler=SimulatedExecutionHandler(settings.execution),
        settings=settings,
    )
    return bt.run().report.as_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate the LightGBM model")
    parser.add_argument("--days", type=int, default=None,
                        help="days of history to fetch (default: config data.history_days)")
    parser.add_argument("--symbol", type=str, default=None, help="override config symbol")
    parser.add_argument(
        "--exchange", type=str, default=None,
        help="override config exchange id (e.g. kraken, coinbase, binanceus)",
    )
    parser.add_argument(
        "--timeframe", type=str, default=None,
        help="override config timeframe (e.g. 5m, 15m) — larger = lower cost drag",
    )
    parser.add_argument("--synthetic", action="store_true", help="offline synthetic data")
    parser.add_argument("--bars", type=int, default=20000, help="bars when --synthetic")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-frac", type=float, default=None,
                        help="held-out fraction (default: config model.test_fraction)")
    parser.add_argument("--horizon", type=int, default=None, help="triple-barrier horizon (bars)")
    parser.add_argument("--tp-mult", type=float, default=None, help="take-profit in ATR")
    parser.add_argument("--sl-mult", type=float, default=None, help="stop-loss in ATR")
    parser.add_argument("--out", type=str, default=None,
                        help="model output path (default: models/model_<SYMBOL>.pkl)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = Settings.load()

    # CLI overrides. The barrier params are written back into settings so that the
    # training labels and the backtest/live EXITS use one identical definition.
    if args.symbol:
        settings.exchange.symbol = args.symbol
    if args.timeframe:
        settings.exchange.timeframe = args.timeframe
    if args.horizon is not None:
        settings.barriers.horizon = args.horizon
    if args.tp_mult is not None:
        settings.barriers.tp_mult = args.tp_mult
    if args.sl_mult is not None:
        settings.barriers.sl_mult = args.sl_mult
    logger.info(
        "Timeframe=%s  barriers: tp=%.2f sl=%.2f horizon=%d",
        settings.exchange.timeframe, settings.barriers.tp_mult,
        settings.barriers.sl_mult, settings.barriers.horizon,
    )

    ohlcv = asyncio.run(load_ohlcv(settings, args))

    # --- Chronological split (no shuffling, no leakage) --------------------
    test_frac = args.test_frac if args.test_frac is not None else settings.model.test_fraction
    split = int(len(ohlcv) * (1.0 - test_frac))
    train_ohlcv = ohlcv.iloc[:split]
    test_ohlcv = ohlcv.iloc[split:]
    logger.info("Split: %d train bars, %d test bars", len(train_ohlcv), len(test_ohlcv))

    # --- Features + triple-barrier labels (+ uniqueness weights) ----------
    # Pool any data.train_symbols into training (same as the walk-forward), sliced to the
    # same cutoff timestamp so no future leaks in. The primary symbol is still tested.
    def _prepare_one(df: pd.DataFrame):
        fe = build_feature_engine(settings)
        feats = fe.transform(df)
        labels, t1 = make_labels(df, feats["atr"], settings.barriers, return_events=True)
        w = make_sample_weights(t1)
        valid = labels.index[: len(labels) - settings.barriers.lookahead]
        return feats.loc[valid], labels.loc[valid], w.loc[valid]

    train_frames = [train_ohlcv]
    primary = args.symbol or settings.exchange.symbol
    if not args.synthetic and settings.data.pool_for(primary):
        cutoff_ts = ohlcv.index[split] if split < len(ohlcv) else ohlcv.index[-1]
        extra = asyncio.run(load_extra_symbols(settings, args))
        for df in extra.values():
            train_frames.append(df.loc[df.index < cutoff_ts])
        logger.info("Pooled %d training symbol(s) into the fit", len(train_frames) - 1)

    parts = [_prepare_one(df) for df in train_frames if not df.empty]

    # Calibration: hold out the recent tail of the PRIMARY symbol (never fit by the trees)
    # so the temperature is fit on data the model didn't train on.
    cal_feats = cal_labels = None
    if settings.model.use_calibration and parts:
        f0, l0, w0 = parts[0]
        cut = int(len(f0) * (1.0 - settings.model.calibration_fraction))
        if cut > 50 and len(f0) - cut >= 30:
            cal_feats, cal_labels = f0.iloc[cut:], l0.iloc[cut:]
            parts[0] = (f0.iloc[:cut], l0.iloc[:cut], w0.iloc[:cut])

    train_feats = pd.concat([p[0] for p in parts]).reset_index(drop=True)
    train_labels = pd.concat([p[1] for p in parts]).reset_index(drop=True)
    train_weights = pd.concat([p[2] for p in parts]).reset_index(drop=True)

    # --- Train + save (all hyperparameters come from MLConfig) -------------
    from cryptotrader.ml.registry import holdout_path_for, model_path_for, write_meta

    out_path = Path(args.out) if args.out else model_path_for(settings.exchange.symbol)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if settings.model.use_meta_labeling:
        from cryptotrader.ml.meta import train_meta_labeled

        predictor, info = train_meta_labeled(
            train_feats, train_labels, settings.model.to_lgbm_params(),
            eval_fraction=settings.model.eval_fraction, embargo=settings.barriers.horizon,
            sample_weight=train_weights,
        )
        predictor.save(out_path)
        logger.info(
            "Saved META-LABELED model to %s (meta acc=%.3f, base win-rate=%.3f)",
            out_path, info["meta_train_accuracy"], info["primary_win_base_rate"],
        )
        train_result = {"val_accuracy": round(float(info["meta_train_accuracy"]), 4),
                        "meta_labeling": True}
    else:
        from cryptotrader.ml.model import build_ensemble

        predictor, metrics = build_ensemble(
            settings, train_feats, train_labels, train_weights, cal_feats, cal_labels)
        predictor.save(out_path)
        temp = getattr(predictor, "temperature", 1.0)
        logger.info("Saved %d-member model to %s (val_accuracy=%.3f, temperature=%.3f)",
                    max(1, settings.model.ensemble_size), out_path,
                    metrics["val_accuracy"], temp)
        train_result = {"val_accuracy": round(float(metrics["val_accuracy"]), 4),
                        "temperature": round(float(temp), 3)}

    # Metadata sidecar — powers the dashboard's symbol guardrail.
    write_meta(out_path, {
        "symbol": settings.exchange.symbol,
        "timeframe": settings.exchange.timeframe,
        "train_symbols": settings.data.train_symbols,
        "ensemble_size": settings.model.ensemble_size,
        "meta_labeling": settings.model.use_meta_labeling,
        "n_train_rows": int(len(train_feats)),
        "barriers": {"tp_mult": settings.barriers.tp_mult, "sl_mult": settings.barriers.sl_mult,
                     "horizon": settings.barriers.horizon},
    })

    # Experiment log: which settings produced this model (auditable tuning history).
    if not args.synthetic:
        from cryptotrader.ml.experiments import log_experiment

        train_result["n_train_rows"] = int(len(train_feats))
        log_experiment("train", settings.exchange.symbol, settings, train_result)

    # --- Persist the held-out slice for the dashboard (per symbol) ---------
    replay_path = holdout_path_for(settings.exchange.symbol, out_path.parent)
    test_ohlcv.to_parquet(replay_path)
    logger.info("Saved held-out slice for dashboard replay to %s", replay_path)

    # --- Evaluate on the held-out slice: model vs baseline -----------------
    from cryptotrader.ml.meta import load_predictor  # auto-detects meta vs plain

    model_report = backtest(settings, test_ohlcv, load_predictor(out_path))
    base_report = backtest(settings, test_ohlcv, MomentumBaselinePredictor())

    print("\n================ HOLD-OUT EVALUATION ================")
    print(f"{'metric':<22}{'LightGBM':>14}{'Baseline':>14}")
    keys = [
        "n_trades", "win_rate", "profit_factor", "total_return_pct",
        "max_drawdown_pct", "sharpe_ratio", "avg_efficiency_ratio",
    ]
    for k in keys:
        print(f"{k:<22}{str(model_report[k]):>14}{str(base_report[k]):>14}")
    print("====================================================")
    print(
        "\nTo use this model in the dashboard, set in config/config.yaml:\n"
        f"  strategy:\n    model_path: {out_path}\n"
        f"  data:\n    replay_file: {replay_path}\n"
    )
    print(json.dumps({"model": model_report, "baseline": base_report}, indent=2))


if __name__ == "__main__":
    main()
