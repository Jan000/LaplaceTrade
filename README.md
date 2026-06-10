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

**Symbols tab** — a **sortable** table of every symbol with its model status, latest
walk-forward / holdout result, average efficiency and realized trade stats. (The realized
columns — trades, win %, PF, efficiency, net PnL — count **paper + real** runs only;
simulation replays are excluded so they don't drown the decision signal. Sort by any stat
header to pick what to trade.) Per-row controls launch jobs **without leaving the page**:
**T+T** (one click = train → walk-forward → holdout in sequence), or individual **Train /
WF / HO**; a **Train + test all** button pipelines *every* symbol (max 2 at a time). Jobs
run **concurrently**, each with its own live log (Jobs panel), and the **Status** column
shows live per-symbol progress (e.g. `⏳ walk-forward… (2/3)`). Tick **Trade** on several
symbols to trade them **concurrently** — one model per symbol, account equity split
equally (real orders still gated per symbol by the model guardrail). The Monitor then
shows aggregate totals plus a per-symbol breakdown.

**Trades & Analytics tab** — pick a source (latest run / **all runs** / any specific
run) and an **Environment** filter (simulation / paper / real), and get a full
performance breakdown: 18 stat cards (win rate, profit factor, expectancy, avg/largest
win & loss, payoff, max drawdown, total fees, avg efficiency, avg hold, win/loss streaks,
by-side and by-exit-reason), a cumulative-PnL chart, and the **complete trade log** —
filterable (side, win/loss/break-even, exit reason, free-text), sortable by any column,
with CSV export. A **Clear data…** button wipes the persisted runs of the selected
environment (e.g. reset all simulations) after a confirm — blocked while the engine runs.

**Settings & Training tab**
* **Model status** — shows whether the trained per-symbol model or the baseline is
  active for the configured symbol, when it was trained and which symbols were pooled,
  with a shortcut to the Symbols tab. (Training and walk-forward / holdout now live in
  the **Symbols** tab — see above — where several run concurrently with live logs.)
* **All Parameters** — edit and save the entire `config/config.yaml` (every section,
  including `train_symbols` / `drop_features` as comma-separated lists).
* **Feature modules** — a clear checklist of every optional input group (market breadth,
  funding rate, cross-asset, higher-timeframe, taker flow, open interest, trend filter)
  with a one-line description and the tested-evidence note; toggle one and **retrain** the
  affected symbols to apply (each toggle changes the model's input set).
* **Exchange API Keys** — enter API key/secret for real trading. They are stored only
  in a git-ignored `config/secrets.yaml` (merged into Settings below env vars,
  `chmod 600` best-effort), are **never** written to `config.yaml` and **never** echoed
  back — the UI shows only set/unset. `/api/config` redacts them too.

**Modes (header)**
* **Simulation** replays data through the *paper* handler — **accelerated** (seconds), no
  keys. A **test-window** selector picks what to replay: **Held-out (OOS)** slice
  (default) or the **last 24h / 7 / 14 / 30 / 90 days** of real data — a quick accelerated
  test of the model on recent history. (Warm-up bars are fetched *before* the window, so
  they never trade and all trades land inside the requested window.)
* **Live** streams real ccxt market data in **real time** — it only acts when a candle
  **closes**, so on 4h the next decision can be up to 4 h away (the header says
  *"live — waits for the next bar to close"*). *Paper* fills by default; tick the
  confirm-gated **real orders** toggle (live mode only) to place REAL orders via the ccxt
  execution handler — this requires saved API keys and trades live funds.

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
3. **Train & test in one click** (Symbols tab) — find your symbol and click **T+T**
   (train → walk-forward → holdout), or **Train + test all** to pipeline every symbol.
   Each coin gets its own model file `models/model_<SYMBOL>.pkl` plus a metadata sidecar;
   the **Status** column shows live progress and the table fills in once done.
   (Re-train after any feature/model/barrier change — the model stores its own input set.)
4. **Pick the edge** — sort the table by **Walk-fwd** / **Holdout**. Only trade a symbol
   whose walk-forward is **ROBUST** and whose holdout is positive (see *Validation* below
   for exactly what each measures). The edge is strongest on BTC.
5. **Simulate (fast, real data)** — switch to **Simulation** and **Start**. The
   **test-window** selector replays either the held-out OOS slice (default) or the
   **last 24h / 7 / 14 / 30 / 90 days** of real data through the engine in seconds. Watch
   *Trades & Analytics* (Environment = Simulation). This is the quick way to *see*
   behaviour — far more useful than waiting in real time on Live data.
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

See `config/config.yaml` for all tunables. API keys come from environment variables
(`CT_EXCHANGE__API_KEY`, `CT_EXCHANGE__API_SECRET`) or a git-ignored `config/secrets.yaml`
(written by the dashboard) — **never** from `config.yaml`, and never echoed back.

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
   minus `model.drop_features` (low-importance features pruned for robustness), plus the
   enabled **external modules** (see *Feature modules* below — funding + breadth are on).
5. **Labels** — the **triple-barrier** method with **symmetric** label barriers
   (`barriers.label_tp_mult` / `label_sl_mult`) so the direction target is unbiased,
   plus average-uniqueness sample weights for overlapping labels. (Trade *exits* use
   the separate, asymmetric `barriers.tp_mult`/`sl_mult` — see the barriers note.)
6. **Train** a **seed-ensemble** of `model.ensemble_size` LightGBM models (different
   seeds + bagging) and average them, to cancel the seed/sampling variance that
   dominates a few-thousand-row training set.
7. **Save & evaluate** — writes the **per-symbol** model `models/model_<SYMBOL>.pkl` (+ a
   `.meta.json` sidecar recording symbol/timeframe/pooled-symbols/feature-set, and a
   per-symbol `holdout_<SYMBOL>.parquet`) and prints a model-vs-baseline backtest on the
   untouched held-out slice.

```bash
python scripts/train_model.py                 # uses config/config.yaml (4h, pool ETH)
python scripts/train_model.py --symbol ETH/USDT   # train a specific coin
python scripts/train_model.py --days 365      # override history window
python scripts/train_model.py --synthetic     # offline smoke test, no network
```

Models are stored **per symbol**, so each coin has its own model side by side. The
engine resolves `models/model_<SYMBOL>.pkl` for the configured/traded symbol on start
(falling back to the momentum baseline when no model file is present), and a metadata
guardrail refuses REAL orders unless the loaded model was trained for that exact
symbol + timeframe.

### Validation: walk-forward (WF) & holdout (HO) — how they work

A single train/test split is one sample and easy to fool. Two complementary
out-of-sample tests decide whether an edge is real. **Both retrain from scratch** (no
tuning on the test data) and slice any pooled symbols to the same cutoff timestamp, so
nothing leaks. In the dashboard the **Symbols** table shows each symbol's latest WF/HO
result and the per-row **WF / HO** (or one-click **T+T**) buttons run them; the CLI does
the same:

```bash
python scripts/walkforward.py                 # anchored expanding-window WF (5 OOS folds)
python scripts/holdout.py                      # single-split out-of-time + cross-asset HO
python scripts/walkforward.py --symbol ETH/USDT --splits 6 --train-frac 0.4
```

**Walk-forward — anchored, expanding window.** Take the full history (≈730 days). The
first `--train-frac` (0.5) is the initial training window; the rest is cut into
`--splits` (5) equal out-of-sample folds. For each fold the script:

1. **trains** a fresh seed-ensemble on everything from the start up to the fold boundary,
2. **tests** it on the next untouched block — the engine trades that block exactly as
   live would (same features, barriers, costs), and
3. **expands** the training window to include that block and repeats.

The model is retrained every fold and never sees its test data. It prints per-fold
return / PF / trades / win % / max-drawdown, then a summary that **compounds** the fold
returns (as if traded in sequence) and counts positive folds. **Verdict:** `ROBUST`
(≥ `splits−1` folds positive **and** compounded > 0), `MIXED` (positive overall but
unstable), or `NOT ROBUST`. WF answers: *does the edge keep generalising over time under
periodic retraining?* — exactly the recommended deploy-and-retrain workflow.

**Holdout — one split, two strict checks.** `scripts/holdout.py`:

1. **Out-of-time:** split the history **once** at `--train-frac` (0.7). Train on the
   oldest 70 %, test on the most recent contiguous 30 % the model has **never** seen —
   the toughest "what happens next on truly recent data" check.
2. **Out-of-asset (cross-asset):** take that *same* primary-trained model, unchanged, and
   run it on coins that were **never in training** (e.g. SOL/BNB/XRP/ADA). If the edge
   survives on unseen assets it's a general micro-structure effect; if only the primary
   holds, the edge is **asset/regime specific**.

**WF stresses time** (many rolling retrains); **HO stresses the most recent unseen period
and unseen assets.** Trust a config only when WF is `ROBUST` **and** the primary's HO is
positive. On 4h BTC with ETH pooled (+ funding/breadth) the walk-forward is consistently
**ROBUST** and BTC holds out-of-time; cross-asset is mixed — the edge is **strongest on
BTC** and is regime/asset dependent, not a universal crypto effect. Note the numbers are
**window-sensitive** (a one-day shift of the 730-day window has moved BTC WF between
~+11 % and ~+18 %), so judge by the *verdict* and PF, not a single headline figure, and
re-validate after any change.

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
up the fresh `models/model_<SYMBOL>.pkl`. Re-run the walk-forward after any change to
confirm the edge still generalises before going live.

### Model lifecycle — retraining overwrites; config changes need a retrain

* Each training **overwrites** that symbol's `models/model_<SYMBOL>.pkl` (and
  `holdout_<SYMBOL>.parquet`, `.meta.json`) in place — no versioning, nothing to delete
  by hand to retrain. (`models/` is git-ignored as a build artifact.)
* The model is **not** auto-invalidated when you change config. A saved model stores its
  own feature list, so after changing `features` (including the feature modules),
  `drop_features`, `barriers`, `train_symbols`, timeframe, etc. you **must retrain** —
  otherwise the engine keeps serving the stale model. Treat "edit config → retrain →
  re-validate → restart" as one atomic loop.

### Managing it all from the dashboard

The whole lifecycle is operable from the UI:

* **Settings & Training → All Parameters** — edit & save the entire config (`exchange`,
  `data`, `features`, `model`, `risk`, `execution`, `strategy`, `barriers`) straight to
  `config/config.yaml`, including `train_symbols` and `drop_features` (comma-separated).
* **Settings & Training → Feature modules** — flip optional input groups on/off.
* **Symbols tab** — **Train / WF / HO** per symbol, **T+T** (full pipeline) or **Train +
  test all**; jobs run **concurrently**, each with a live log, and progress shows in the
  table. After training, **Stop/Start** the engine so it reloads the new model.

API keys are never shown or written to `config.yaml`; they live in git-ignored
`config/secrets.yaml` (or `CT_EXCHANGE__API_KEY` / `CT_EXCHANGE__API_SECRET`).

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

### Feature modules (optional external inputs)

Beyond the price/volume indicators, the model can pull in extra, orthogonal data. Each is
a `features.use_*` flag (toggle in the dashboard's **Feature modules** card, then retrain).
Every one was tested in the walk-forward; only those that **robustly** helped are on by
default — adding inputs otherwise overfits the small 4h sample.

| Module | Source | Status | Why |
|--------|--------|--------|-----|
| **Market breadth** (`use_breadth`) | basket of alts (ETH/SOL/BNB/XRP) | **ON** | avg return, % positive & BTC-vs-market — market-wide flow |
| **Funding rate** (`use_funding`) | Binance USDⓈ-M futures (free) | **ON** | perp positioning / sentiment, full history |
| Cross-asset (`use_cross_asset`) | one second symbol | off | overfit / redundant with pooling |
| Higher-timeframe (`use_htf`) | daily trend/RSI/return | off | didn't help on 4h |
| Taker flow (`use_taker_flow`) | Binance klines | off | no robust gain |
| Open interest (`use_open_interest`) | Binance futures | off | Binance caps OI history at ~30 days (flat for training) |
| Fear & Greed (`use_fear_greed`) | alternative.me (free daily) | off | too coarse (daily) & redundant with momentum |

Funding/OI use a **free Binance USDⓈ-M futures** client automatically (no extra keys).
The **funding + breadth combination** is the one that passed a multi-fold-structure
robustness check (both folds positive, profit factor up) and is enabled in the shipped
config; the rest are kept available but off. Mapping is **leak-free** (daily/8h sources
use only the last *completed* value, forward-filled onto the bars).

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
