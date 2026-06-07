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

A FastAPI backend serves a single-page dashboard (HTML/JS + Chart.js). The whole
lifecycle — configure, train, validate, trade, review — is operable from the UI.

```bash
uvicorn cryptotrader.api.server:app          # then open http://127.0.0.1:8000
# or
python -m cryptotrader.api.server
```

**Monitor tab** — account & PnL cards, equity chart and trade history streamed in
real time over a WebSocket. A **View** selector browses any past run from the
database (equity, trades and a frozen summary); on a page refresh it seeds from the
last persisted run so the panel is never blank.

**Symbols tab** — a table of every symbol with its model status, last walk-forward /
holdout result, average efficiency and realized trade stats, plus per-row controls
(**Train / Walk-fwd / Holdout / Set active**). Tick **Trade** on several symbols to trade
them **concurrently**: the engine runs one model per symbol at once and splits account
equity equally across them (real orders are still gated per symbol by the model guardrail).
The Monitor then shows aggregate totals plus a per-symbol breakdown.

**Trades & Analytics tab** — pick a source (latest run / **all runs** / any specific
run) and get a full performance breakdown: 18 stat cards (win rate, profit factor,
expectancy, avg/largest win & loss, payoff, max drawdown, total fees, avg efficiency,
avg hold, win/loss streaks, by-side and by-exit-reason), a cumulative-PnL chart, and
the **complete trade log** — filterable (side, win/loss/break-even, exit reason,
free-text), sortable by any column, with CSV export.

**Settings & Training tab**
* **All Parameters** — edit and save the entire `config/config.yaml` (every section,
  including `train_symbols` / `drop_features` as comma-separated lists).
* **Exchange API Keys** — enter API key/secret for real trading. They are stored only
  in a git-ignored `config/secrets.yaml` (merged into Settings below env vars,
  `chmod 600` best-effort), are **never** written to `config.yaml` and **never** echoed
  back — the UI shows only set/unset. `/api/config` redacts them too.
* **Training** — "Train now" runs `scripts/train_model.py` on the saved config with a
  live log; the model-status line shows whether the trained model or the baseline is
  active, when it was trained and which symbols were pooled.
* **Validation** — run **walk-forward** or **holdout** out-of-sample checks
  (optional `--days`) directly, with live log output. One background job runs at a time.

**Modes (header)**
* **Simulation** replays held-out/synthetic data through the *paper* handler — offline,
  no keys.
* **Live** streams real ccxt market data; *paper* fills by default. Tick the
  confirm-gated **real orders** toggle (live mode only) to place REAL orders via
  `CCXTExecutionHandler` — this requires saved API keys and trades live funds.

## Using it as a trader — step by step

The whole flow lives in the dashboard. **Environments**: every run is tagged
`simulation` (replay + paper fills), `paper` (real market data + simulated fills) or
`live` (REAL orders, real money). The Trades & Analytics tab lets you view each
separately, so test runs never pollute your real track record.

1. **Launch & open** — `uvicorn cryptotrader.api.server:app`, open
   http://127.0.0.1:8000.
2. **Configure** (Settings & Training → All Parameters): set `exchange.id`,
   `exchange.symbol`, `exchange.timeframe`, and review **risk** —
   `risk.account_equity` (your starting capital), `risk.risk_per_trade` (e.g. 0.005 =
   0.5%/trade), `risk.max_leverage` (keep 1.0 to start) — and `execution.taker_fee` /
   `slippage_bps` to match your exchange tier. **Save to config.yaml**.
3. **Train** — set an optional **Target symbol** (blank = configured symbol) and click
   **Train now**. Each coin gets its own model file `models/model_<SYMBOL>.pkl` plus a
   metadata sidecar; wait until the model-status line shows a matching *trained* model.
   (Re-train after any feature/model/barrier change.)
4. **Validate** — run **walk-forward** (and **holdout**) for the same target. Only go
   further if the verdict is ROBUST / positive out of sample. This is your edge check.
5. **Simulate (fast, real data)** — switch to the **Simulation** mode and **Start**: it
   replays the held-out real slice through the live engine in seconds. Watch *Trades &
   Analytics* (Environment = Simulation). This is the quick way to *see* behaviour —
   far more useful than waiting in real time on Live data.
5b. **Paper-test on live data** (optional) — mode **Live**, **real orders OFF**, **Start**:
   confirms the real-time plumbing (connectivity, the price appears immediately via
   warm-up). On higher timeframes the next decision only comes when a candle closes.
6. **Connect your account** — Settings → **Exchange API Keys**: paste the API key/secret
   for `exchange.id`. On the exchange, create a key that **allows spot trading but NOT
   withdrawals**, and ideally restrict it to your server's IP. Keys are stored only in
   git-ignored `config/secrets.yaml`, never echoed back.
7. **Go live (real money)** — header: mode **Live**, tick **real orders**, press
   **Start**, confirm the prompt. The header shows **⚠ REAL MONEY** while it runs. Start
   small (low `risk_per_trade`, `max_leverage` 1.0). **Guardrail:** real orders are
   *refused* unless a model trained for the exact configured symbol+timeframe exists —
   so you can't accidentally trade a coin with another coin's model (or the baseline).
8. **Monitor & analyse** — Monitor tab for the live run; **Trades & Analytics → Environment
   = Real money** for your real track record (stats, full filterable trade log, CSV).
9. **Maintain** — re-train on a rolling window periodically and re-validate; **Stop** to
   flatten new decisions.

> ⚠️ **Operational risk.** Exits (stop-loss / take-profit / time-exit) are driven by the
> running engine, not by native exchange orders. **If the server is stopped while a real
> position is open, that position is left unmanaged.** Keep the process running (and the
> position size small) until native exchange-side protective orders are added. This is an
> MVP — paper-trade extensively before risking real capital, and never trade money you
> can't afford to lose.

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
learn. The real edge comes from a trained **LightGBM** model. This section is the
full picture: how a training run works, how to validate it, how data flows, and
what the shipped defaults are.

### What one training run does

`scripts/train_model.py` runs this pipeline (every step is config-driven):

1. **Fetch** OHLCV for the primary symbol (`exchange.symbol`, `exchange.timeframe`)
   over the last `data.history_days`, via ccxt, cached to `.cache/ohlcv/*.parquet`.
2. **Pool extra symbols.** Each symbol in `data.train_symbols` (default `[ETH/USDT]`)
   is fetched and its history — sliced to the same cutoff so nothing leaks — is
   added to the *training* set only. The primary symbol stays what you trade/test.
   Pooling the most BTC-correlated major roughly doubles the training data and
   fights overfitting on the small higher-timeframe history.
3. **Chronological split** into train / held-out test (`model.test_fraction`,
   default 0.25). No shuffling — evaluation never sees the past's future.
4. **Features** — the backward-looking micro-structure matrix (`FeatureConfig`),
   minus `model.drop_features` (low-importance features pruned for robustness).
5. **Labels** — the **triple-barrier** method with **symmetric** label barriers
   (`barriers.label_tp_mult` / `label_sl_mult`) so the direction target is unbiased,
   plus average-uniqueness sample weights for overlapping labels. (Trade *exits* use
   the separate, asymmetric `barriers.tp_mult`/`sl_mult` — see the barriers note.)
6. **Train** a **seed-ensemble** of `model.ensemble_size` LightGBM models (different
   seeds + bagging) and average them, to cancel the seed/sampling variance that
   dominates a few-thousand-row training set.
7. **Save & evaluate** — writes `models/model.pkl` (+ `models/holdout.parquet`) and
   prints a model-vs-baseline backtest on the untouched held-out slice.

```bash
python scripts/train_model.py                 # uses config/config.yaml (4h, pool ETH)
python scripts/train_model.py --days 365      # override history window
python scripts/train_model.py --synthetic     # offline smoke test, no network
```

`strategy.model_path: models/model.pkl` is already set, so once a model exists the
dashboard's **live/simulation** engine loads it automatically on start (it falls
back to the momentum baseline while no model file is present).

### Validate before you trust it — the walk-forward

A single train/test split is one sample and easy to fool. The honest test is
**walk-forward** (anchored, expanding window; retrain per fold; never tuned per
fold):

```bash
python scripts/walkforward.py                 # 5 OOS folds on the shipped config
```

It prints per-fold return / PF / win% / drawdown and a verdict
(`ROBUST` / `MIXED` / `NOT ROBUST`). The shipped config is **ROBUST**:
**+16.1 % compounded OOS, PF 1.45, 4–5 of 5 folds positive, max drawdown < 5 %**
on 4h BTC with ETH pooled.

For an even stricter check, `scripts/holdout.py` trains **once** on the oldest 70 %
and tests the untouched recent 30 % of *several* assets — including SOL/BNB/XRP/ADA
that were never in training. On the current config BTC holds out-of-time (**+6.5 %**)
but cross-asset is mixed: the edge is strongest on BTC and is **regime/asset
dependent**, not a universal crypto effect. Deploy on BTC with measured expectations.

### How much history? More is *not* better

Counter-intuitively, a **longer** window hurts here — the 2022–23 bear regime has
different micro-structure and degrades the recent edge:

| `--days` | Walk-forward (4h, pool ETH) |
|----------|------------------------------|
| **730** (shipped) | **+16.1 %, PF 1.45, ROBUST** |
| 1095 | −2.3 %, PF 1.05, NOT ROBUST |
| 1460 | −17.9 %, PF 0.91, NOT ROBUST |

So "train on more data" was tested and rejected — it degrades monotonically as the
2022–23 bear regime enters the window; ~2 years on 4h is the sweet spot.
The lever that *did* add data without that regime cost was pooling a correlated
**symbol** (ETH), not a longer time window.

### Does live data feed training? (No — and the recommended workflow)

Training is a **batch, offline** job: it fetches history *at the moment you run it*
and fits a static model. The **live** engine only *infers* with the loaded model and
**never learns online** — live ticks drive predictions and trades, not the training
set. New market data therefore enters the model only when you **retrain**.

**Recommended cadence:** retrain periodically on a rolling window (e.g. weekly or
monthly) — exactly what the walk-forward simulates — then restart the engine to pick
up the fresh `models/model.pkl`. Re-run `walkforward.py` after any change to confirm
the edge still generalises before going live.

### Model lifecycle — retraining overwrites; config changes need a retrain

* Each training **overwrites** `models/model.pkl` (and `holdout.parquet`) in place —
  no versioning, nothing to delete by hand to retrain. (Keep a manual copy if you
  want history; `models/` is git-ignored as a build artifact.)
* The model is **not** auto-invalidated when you change config. A saved model stores
  its own feature list, so after changing `features`, `drop_features`, `barriers`,
  `train_symbols`, timeframe, etc. you **must retrain** — otherwise the engine keeps
  serving the stale model. Treat "edit config → retrain → re-validate → restart" as
  one atomic loop.

### Managing it all from the dashboard

Open **Settings & Training** in the dashboard:

* **Edit & save the entire config** (`exchange`, `data`, `features`, `model`, `risk`,
  `execution`, `strategy`, `barriers`) straight to `config/config.yaml` — including
  `train_symbols` and `drop_features` (entered as comma-separated lists).
* **Train now** launches `scripts/train_model.py` as a background job using the saved
  config, with live log tail and status. One run at a time.
* After it finishes, **Stop/Start** the engine so it reloads the new model.

What the dashboard does **not** (yet) expose: per-run CLI overrides (`--days`,
`--symbol`) — it always trains on the saved config — and it does not run the
walk-forward / holdout validators (use the CLI for those). API keys are never shown
or written to `config.yaml`; they come from `CT_EXCHANGE__API_KEY` /
`CT_EXCHANGE__API_SECRET`.

> Note: on a pure random walk (synthetic data) no strategy can be profitable — the
> edge must come from real micro-structure. Training lets the model *find* it; the
> walk-forward/holdout is what tells you whether it did.

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

**Validate with `walkforward.py`, not a single backtest** — one split is easy to
overfit. The shipped config already reflects the biggest lessons learned (4h beats
1h on cost drag; symmetric labels; meta-labeling OFF; *regularise* rather than
"stronger fit" — fewer leaves/trees generalised better; pool ETH; ~730 days). Treat
the table below as a search space to *re-validate*, not as guaranteed wins:

Lower-impact levers, roughly in order:

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
