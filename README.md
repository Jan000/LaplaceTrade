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
pip install -e ".[dev]"
python scripts/run_backtest.py            # runs on synthetic data, no keys needed
pytest -q
```

See `config/config.yaml` for all tunables. API keys come from environment
variables (`CT_EXCHANGE__API_KEY`, `CT_EXCHANGE__API_SECRET`), never from disk.
