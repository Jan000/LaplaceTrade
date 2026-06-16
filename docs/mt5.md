# MetaTrader 5 bridge

Trade on a MetaTrader 5 broker while keeping the whole ML/decision brain on your
Linux/Coolify deployment. A small MQL5 Expert Advisor (EA) runs inside the MT5
terminal and talks to the dashboard over HTTPS:

```
  MT5 terminal (Windows / broker VPS)            CryptoTrader (Linux / Coolify)
  ┌────────────────────────────┐   POST bars+position    ┌────────────────────────┐
  │  CryptoTraderBridge.mq5     │ ──────────────────────► │  /api/mt5/decide       │
  │  • CopyRates (closed bars)  │                         │  features → model →    │
  │  • places broker orders     │ ◄────────────────────── │  strategy → ATR risk   │
  │  • native SL / TP           │   action + SL/TP + risk │  → decision            │
  └────────────────────────────┘                         └────────────────────────┘
```

* **Why this design?** The official `MetaTrader5` Python package is Windows-only and
  needs a running terminal — incompatible with a Linux/Coolify deployment. MQL5's
  built-in `WebRequest()` lets the EA call your dashboard directly, so nothing about
  the Python service has to change or move.
* **Roles:** the EA is both the *data source* (it streams broker bars, captured under
  `data/mt5/`) and the *execution venue* (it places the orders). The server decides.
* **What's reused:** the decision uses the exact same feature engine, per-symbol model,
  strategy thresholds/regime filters and the ATR expected-value risk gate as live crypto
  trading — so "what you backtest is what MT5 trades" still holds. MT5 brokers have no L2
  order book, so the recorded micro-structure features are automatically disabled.

## 1. Server configuration

In `config/config.yaml` (persists on the `ct_config` volume) or `config/secrets.yaml`
for the token:

```yaml
mt5:
  enabled: true
  api_token: "<long-random-token>"     # put this in secrets.yaml, not config.yaml
  require_model: true                   # never trade a symbol without a matching model
  capture_bars: true                    # record streamed bars for offline training
  default_timeframe: "1h"
  symbol_map:                           # MT5 broker symbol -> internal model symbol
    BTCUSD: "BTC/USDT"
    ETHUSD: "ETH/USDT"
    XAUUSD: "XAU/USD"                   # non-crypto MUST be mapped explicitly
```

`enabled`/`require_model` can also be set via env (`CT_MT5__ENABLED`, …). The nested
`symbol_map` must live in YAML. The `/api/mt5/*` routes are protected by the token (sent
as the `X-API-Token` header) and are exempt from the dashboard Basic-auth.

## 2. Install the EA

1. Copy `src/cryptotrader/integrations/mt5/expert/CryptoTraderBridge.mq5` into your
   terminal's `MQL5/Experts/` folder (MetaEditor → *File → Open Data Folder*), then
   compile it in MetaEditor (F7).
2. In the terminal: **Tools → Options → Expert Advisors →** tick *Allow WebRequest for
   listed URL* and add your dashboard origin (e.g. `https://luciphy.com`).
3. Drag the EA onto a chart whose **symbol and timeframe match a trained model** (e.g.
   `BTCUSD`, H4). Set inputs:
   * `ServerURL` — your dashboard base URL.
   * `ApiToken` — the same `mt5.api_token`.
   * `BarsToSend` — ≥ the model warm-up (≈ 320 is safe).
   * `PollSeconds` — decision cadence (match your timeframe; e.g. 300 for H4 is plenty).
   * `AllowShort`, `MaxSpreadPoints`, `FallbackRiskPct`, `MaxLots` — risk controls.
4. Enable **Algo Trading** in the toolbar.

The EA acts immediately on attach, then every `PollSeconds`.

## 3. How decisions map to orders

| server `action` | EA behaviour                                                        |
|-----------------|---------------------------------------------------------------------|
| `open_long` / `open_short` | open a position sized from `risk_fraction` and the ATR stop distance, with the server's `stop_loss`/`take_profit` attached natively |
| `close`         | close the EA's open position (opposite signal or time-exit)         |
| `hold`          | keep the open position (its broker SL/TP/time manage it)            |
| `none`          | stay flat                                                           |

Lot size is computed **on the EA** from your broker's contract specs:
`lots = (balance × risk_fraction) / (stop_distance / tickSize × tickValue)`, rounded to
the lot step and clamped to the symbol's min/max (and `MaxLots`). Price barriers are the
broker's native SL/TP; the vertical (time) barrier is enforced by the server via
`bars_held ≥ horizon → close`.

## 4. Training a model on MT5 data

With `capture_bars: true`, every `/api/mt5/decide` call appends the streamed bars to
`data/mt5/<SYMBOL>_<tf>.parquet` (deduplicated, capped at `capture_max_rows`). Once enough
history is captured:

```bash
python scripts/train_model.py --mt5 --symbol BTC/USDT --candidate
```

This trains a **candidate** on the captured bars without touching the active model;
validate it (walk-forward / holdout) and promote it from the Symbols tab. You can also
feed bars without asking for a decision via `POST /api/mt5/ingest`.

## 5. Endpoints (all require `X-API-Token`)

* `POST /api/mt5/decide` — `{symbol, timeframe?, bars[], position?, account?}` → decision.
* `POST /api/mt5/ingest` — `{symbol, timeframe?, bars[]}` → captures bars only.
* `GET  /api/mt5/status` — bridge config + captured-row counts.

## Security notes

* Use a long random `api_token` and always run the dashboard over HTTPS (Coolify gives you
  TLS automatically). The token is the only thing standing between the internet and order
  placement on your account.
* Keep `require_model: true` so a misconfigured symbol can never trade on the momentum
  baseline.
* Create the broker/API credentials with trade-only permissions and test on a demo account
  first (point the EA's `ServerURL` at the same dashboard; use a demo MT5 login).
