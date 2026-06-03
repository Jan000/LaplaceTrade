# CryptoTrader

AI-driven, event-driven **intraday** crypto trading bot (MVP).

The same `Strategy` / `RiskManager` / `Portfolio` stack runs unchanged in
**backtest** and **live** mode — only the `DataHandler` and `ExecutionHandler`
implementations are swapped. This guarantees that what you backtest is what you
trade.

```
            ┌──────────────┐   MarketEvent   ┌────────────┐
   data ───▶│ DataHandler  │────────────────▶│  Strategy  │
            └──────────────┘                 │ (Features  │
                   ▲                          │  + ML)     │
                   │                          └─────┬──────┘
                   │                                │ SignalEvent
                   │                          ┌─────▼──────┐
            ┌──────┴───────┐  FillEvent       │ RiskManager│
            │  Execution   │◀───┐             │ + Portfolio│
            │  Handler     │    │ OrderEvent  └─────┬──────┘
            └──────────────┘    └───────────────────┘
```

## Quick start

```bash
pip install -e ".[dev,dashboard]"
python scripts/run_backtest.py            # runs on synthetic data, no keys needed
pytest -q
```

## Control Center (dashboard)

A FastAPI backend serves a single-page dashboard (HTML/JS + Chart.js) with a
live/sim toggle, account & PnL cards, an equity chart, and a trade history fed in
real time over a WebSocket.

```bash
uvicorn cryptotrader.api.server:app          # then open http://127.0.0.1:8000
# or
python -m cryptotrader.api.server
```

* **Simulation** mode replays synthetic data through the *paper* execution
  handler — fully offline, no keys, great for demos.
* **Live** mode streams real ccxt market data through the *paper* handler by
  default (no real orders). Real order placement uses `CCXTExecutionHandler` and
  is opt-in (`real_orders=true` + API keys); see `execution/live.py`.

## Live engine

`cryptotrader.live.LiveTradingEngine` is the streaming counterpart of the
backtester. It consumes the same `MarketEvent` interface and reuses the same
`Portfolio`, `Strategy` and `RiskManager`, awaiting an async execution handler so
the identical loop drives both paper fills and real ccxt orders.

## Persistence

`cryptotrader.persistence.TradeStore` (async SQLite via `aiosqlite`) durably logs
runs, trades, the equity curve and optional feature vectors. High-frequency
equity samples are committed in batches so the live hot loop never pays a per-bar
fsync; trades always commit immediately. The dashboard reads trade/equity history
from this DB.

See `config/config.yaml` for all tunables. API keys come from environment
variables (`CT_EXCHANGE__API_KEY`, `CT_EXCHANGE__API_SECRET`), never from disk.

## Training the model

The dashboard's default predictor is a rule-based **baseline** — it does not
learn. To train the real LightGBM model on **real** exchange data and plug it
into the dashboard:

```bash
# 1) Fetch real Binance data, train, evaluate on a held-out slice, save model
python scripts/train_model.py --days 45              # or --symbol ETH/USDT
#    -> writes models/model.pkl and models/holdout.parquet
#    -> prints a LightGBM-vs-baseline comparison on the held-out slice

# 2) Point the bot at the trained model + the held-out data (config/config.yaml)
#    strategy:
#      model_path: models/model.pkl
#    data:
#      replay_file: models/holdout.parquet

# 3) Restart the dashboard; "Simulation" now replays the real held-out slice
#    through the trained model (no look-ahead — it never trained on this slice).
python -m cryptotrader.api.server
```

Offline smoke test without network: `python scripts/train_model.py --synthetic`.

Note: on a pure random walk (synthetic data) no strategy can be profitable — the
edge has to come from real micro-structure. Training is what lets the model
*find* that edge; the held-out evaluation is what tells you whether it did.

## Configuration — everything is tunable

Every setting resolves in three layers (later wins):

1. **Defaults** in `src/cryptotrader/config.py`
2. **`config/config.yaml`** — the main place to tune
3. **Environment variables** — prefix `CT_`, nested via `__` (great for sweeps)

```bash
# one-off experiment without editing any file:
CT_MODEL__N_ESTIMATORS=800 CT_MODEL__LEARNING_RATE=0.02 \
CT_BARRIERS__TP_MULT=2.5 CT_STRATEGY__LONG_THRESHOLD=0.66 \
python scripts/train_model.py --days 365
```

Tunable groups: `exchange`, `data`, **`features`** (every indicator window — ATR,
VWAP, RSI fast/slow, MACD, Stochastic, Bollinger, ADX, Donchian, Parkinson,
fracdiff order/window, …), **`model`** (all LightGBM hyperparameters +
`class_weight`, `eval_fraction`, `test_fraction`), **`risk`** (sizing + cost
control: `max_leverage`, `min_edge_cost_ratio`, `cooldown_bars`), `execution`
(fees/slippage), `strategy` (entry thresholds), **`barriers`** (`tp_mult`,
`sl_mult`, `horizon` — shared by labels *and* exits).

### Tuning toward profitability

The held-out backtest is your scoreboard. Levers, in order of impact:

| Goal | Knob | Direction |
|------|------|-----------|
| Cut overtrading / costs | `strategy.long_threshold`/`short_threshold` | up (0.65–0.72) |
| Skip low-edge trades | `risk.min_edge_cost_ratio` | up (2.5–3.5) |
| Bigger wins vs fixed cost | `barriers.tp_mult` | up (2.0–3.0) |
| Fewer round-trips | `risk.cooldown_bars` | up (5–15) |
| Better directional recall | `model.class_weight` | `balanced` |
| Stronger fit | `model.n_estimators` ↑, `model.learning_rate` ↓ | |
| Lower real costs | `execution.taker_fee` | your maker/VIP rate |

Raising `tp_mult` changes the break-even win rate to `sl_mult / (tp_mult + sl_mult)`,
so keep an eye on win-rate vs that threshold in the evaluation table.
