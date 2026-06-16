# src/cryptotrader/config.py
"""Typed, layered runtime configuration.

Every field can be overridden three ways (later wins):
  1. defaults below
  2. ``config/config.yaml``
  3. environment variables, prefix ``CT_``, nested via ``__``
     e.g. ``CT_RISK__MAX_LEVERAGE=2.0``  or  ``CT_MODEL__N_ESTIMATORS=800``
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


def _clean_str_list(v):
    """Coerce a config value to a clean list[str]: drop None/blank entries, stringify.

    Guards against a malformed config.yaml (e.g. ``train_symbols: [null]`` written by an
    earlier dashboard bug) crashing the whole Settings load — and thus the config API."""
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if x is not None and str(x).strip() != ""]
    return v

# Set by Settings.load() so the YAML source is only active for an explicit
# load() (keeps a bare Settings() on pure defaults+env, which the tests rely on).
_YAML_PATH: str | None = None
# Optional secrets file (API keys), kept OUT of config.yaml and git-ignored. Managed
# via the dashboard's /api/keys. Loaded with higher priority than config.yaml but below
# environment variables, so CT_EXCHANGE__API_KEY still wins in CI.
_SECRETS_PATH: str | None = None
SECRETS_FILE = "config/secrets.yaml"


class RunMode(str, enum.Enum):
    BACKTEST = "backtest"
    LIVE = "live"


class ExchangeConfig(BaseModel):
    id: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    api_key: str | None = None
    api_secret: str | None = None
    testnet: bool = False           # route REAL orders / account calls to the exchange sandbox


class DataConfig(BaseModel):
    history_days: int = 365
    cache_dir: Path = Path(".cache/ohlcv")
    replay_file: Path | None = None
    # Simulation source (accelerated replay through the live engine). Precedence:
    # replay_file (e.g. the held-out OOS slice train_model writes) -> real recent
    # `sim_days` of data -> synthetic (offline fallback). "synthetic" forces offline.
    sim_source: str = "auto"   # auto | real | synthetic
    sim_days: int = 90         # days of real data to replay when no replay_file is set
    # Extra symbols pooled into the TRAINING set (the primary exchange.symbol is still
    # what gets traded/tested). More + more diverse data fights overfitting on the small
    # higher-timeframe history. Empty list = single-symbol behaviour.
    train_symbols: list[str] = Field(default_factory=list)
    # Symbols to TRADE concurrently in live/simulation. Empty = just exchange.symbol.
    # Account equity is split equally across them; each needs its own trained model.
    trade_symbols: list[str] = Field(default_factory=list)
    # Live market recorder: auto-start it when the server boots (records all symbols).
    recorder_autostart: bool = False
    recorder_interval: float = 120.0   # seconds between recorder samples
    # Scheduled auto-retrain: re-train + walk-forward the traded symbols every N days so the
    # deployed model stays fresh on a rolling window (0 = off). Restart the engine to apply.
    retrain_interval_days: float = 0.0

    _clean_symbols = field_validator("train_symbols", "trade_symbols", mode="before")(
        _clean_str_list)

    def pool_for(self, primary: str) -> list[str]:
        """Training-pool symbols for ``primary`` (``train_symbols`` minus itself).

        A symbol that equals the only configured pool symbol (e.g. ETH when
        train_symbols=[ETH]) would otherwise train **un-augmented** — empirically the
        worst case. So when pooling is intended but the primary collides with it, fall
        back to the market leader (BTC for alts, ETH for BTC), never leaving it empty.
        An explicitly empty ``train_symbols`` (no pooling intended) is respected.
        """
        pool = [s for s in self.train_symbols if s != primary]
        if pool or not self.train_symbols:
            return pool
        return ["ETH/USDT"] if primary == "BTC/USDT" else ["BTC/USDT"]


class FeatureConfig(BaseModel):
    """Every indicator window — fully tunable. Names match the engine kwargs."""

    atr_period: int = 14
    vwap_window: int = 60
    momentum_windows: list[int] = Field(default_factory=lambda: [3, 5, 15])
    extra_momentum: int = 30
    volume_spike_window: int = 30
    zscore_window: int = 60
    rsi_fast: int = 7
    rsi_slow: int = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k: int = 14
    stoch_d: int = 3
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    adx_period: int = 14
    donchian: int = 20
    parkinson: int = 20
    obv_z: int = 30
    amihud: int = 20
    fracdiff_d: float = 0.4
    fracdiff_window: int = 60
    trend_ema: int = 50              # slow EMA span (bars) for the regime/trend filter
    # Higher-timeframe context features (e.g. daily trend/RSI/return fed into the 4h model).
    use_htf: bool = False
    htf_rule: str = "1D"             # pandas resample rule for the higher timeframe
    htf_ema: int = 10                # EMA span (in HTF bars) for the HTF trend regime
    htf_rsi: int = 14                # RSI period (in HTF bars)
    htf_lookback_bars: int = 200     # warmup (base bars) needed to form the HTF features
    # Market breadth: how the broader crypto basket is moving this bar (orthogonal to BTC TA).
    use_breadth: bool = False
    breadth_symbols: list[str] = Field(
        default_factory=lambda: ["ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
    )
    _clean_breadth = field_validator("breadth_symbols", mode="before")(_clean_str_list)
    # Crypto Fear & Greed index (alternative.me, free daily sentiment) — orthogonal to price TA.
    use_fear_greed: bool = False
    # Cross-venue premium: price on a USD exchange (Coinbase) vs the USDT exchange (Binance) —
    # a proxy for US/institutional demand & USDT de-peg. Orthogonal to perp funding.
    use_coinbase_premium: bool = False
    premium_exchange: str = "coinbase"
    premium_fetch_tf: str = "1h"     # Coinbase has no 4h; fetch finer and resample to the base tf
    # Use the live recorder's microstructure observations (order-book imbalance, microprice,
    # depth, taker flow, spread) as features. Only populated where data was recorded — older
    # bars are zero-filled, so train on the recorded window (optionally + history).
    use_recorded: bool = False
    # --- Optional extra data sources (all opt-in; features computed only when
    # the source columns are present in the OHLCV frame) ---
    use_taker_flow: bool = False     # taker buy/sell volume + trade count (Binance klines)
    use_funding: bool = False        # perpetual funding rate
    use_open_interest: bool = False  # perpetual open interest
    use_cross_asset: bool = False    # a second asset's return + rolling correlation
    cross_symbol: str = "ETH/USDT"   # symbol for the cross-asset features
    cross_corr_window: int = 60      # rolling window for cross-asset correlation


class MLConfig(BaseModel):
    """LightGBM hyperparameters + training controls (all tunable)."""

    n_estimators: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 63
    max_depth: int = -1
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    class_weight: str | None = None         # None | "balanced" (tunable)
    eval_fraction: float = 0.2
    test_fraction: float = 0.25             # held-out slice for evaluation
    random_state: int = 42                  # fix for reproducible runs
    use_meta_labeling: bool = False         # add a secondary "should I act?" model
    drop_features: list[str] = Field(default_factory=list)  # feature names to exclude (anti-overfit pruning)
    _clean_drop = field_validator("drop_features", mode="before")(_clean_str_list)
    ensemble_size: int = 1                  # >1 averages N seed-varied models (variance reduction)
    bagging_freq: int = 0                   # LightGBM bagging frequency; >0 makes `subsample` actually bag
    # Probability calibration: temperature-scale the ensemble's class probabilities, fit on a
    # held-out tail of the primary training data, so the entry thresholds / EV gate are meaningful.
    use_calibration: bool = False
    calibration_fraction: float = 0.2       # tail of primary train data held out to fit the temperature
    # LightGBM threads per fit. 0 = leave headroom (half the cores) so concurrent jobs (train +
    # walk-forward), the recorder and the live engine don't oversubscribe the CPU on a small
    # server/container; -1 = all cores; N = exactly N. Bound this on Coolify (e.g. 1–2).
    n_jobs: int = 0

    def resolved_n_jobs(self) -> int:
        if self.n_jobs > 0 or self.n_jobs == -1:
            return self.n_jobs
        import os
        return max(1, (os.cpu_count() or 4) // 2)   # 0 -> half the cores (concurrency headroom)

    def to_lgbm_params(self) -> dict[str, Any]:
        return {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "subsample_freq": self.bagging_freq,  # subsample only bags when this is > 0
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "n_jobs": self.resolved_n_jobs(),
            "verbosity": -1,
        }


class RiskConfig(BaseModel):
    account_equity: float = 10_000.0
    risk_per_trade: float = 0.005
    atr_stop_mult: float = 2.0
    atr_trail_mult: float = 1.5
    max_open_positions: int = 1
    max_leverage: float = 1.0
    min_edge_cost_ratio: float = 2.0   # legacy filter (used when use_ev_filter=False)
    cooldown_bars: int = 3
    use_ev_filter: bool = False        # EV gate: P(win)*tp-(1-P(win))*sl-cost > min_ev
    min_expected_value: float = 0.0    # minimum expected value per unit to trade
    # --- Circuit breakers (real-money safety; 0 = disabled) ---
    # If the aggregate account drops by this fraction intraday or from its session peak,
    # the controller flattens everything and HALTS new entries until you resume.
    max_daily_loss_pct: float = 0.0    # e.g. 0.05 = halt after a 5% loss since UTC day start
    max_drawdown_pct: float = 0.0      # e.g. 0.10 = halt after a 10% drop from session peak
    max_consecutive_losses: int = 0    # halt after N losing trades in a row (0 = disabled)


class ExecutionConfig(BaseModel):
    taker_fee: float = 0.0004
    slippage_bps: float = 1.0
    # Maker/limit entries: post a passive limit just inside the market instead of
    # paying the taker fee + slippage. On 4h the intrabar range almost always sweeps
    # a few-bps offset, so fill rate stays high while entry cost drops to the maker fee
    # with zero slippage. Exits stay taker (market) — conservative.
    entry_order_type: str = "market"   # "market" | "limit"
    maker_fee: float = 0.0002          # Binance maker (2 bps); used for LIMIT entries
    limit_offset_bps: float = 2.0      # how far inside the market to post the entry limit


class StrategyConfig(BaseModel):
    long_threshold: float = 0.62
    short_threshold: float = 0.62
    allow_short: bool = True
    model_path: Path | None = None
    # Regime filter: when on, only take signals that agree with the slow-EMA trend
    # (long only above the EMA, short only below). Symmetric, so it stays regime-
    # adaptive rather than just betting on the prevailing direction.
    trend_filter: bool = False
    # Volatility-regime gate: only act when the bar's realized-vol percentile (rolling,
    # backward-looking) is within [low, high]. Skips dead-low-vol chop / extreme-vol bars.
    vol_gate: bool = False
    vol_gate_low: float = 0.0        # skip below this realized-vol percentile (0 = no floor)
    vol_gate_high: float = 1.0       # skip above this percentile (1 = no cap)


class BarrierConfig(BaseModel):
    """Triple-barrier params.

    Two *separate* barrier sets, because labelling and trading want opposite things:

    * **Trade exits** (``tp_mult`` / ``sl_mult``) — may be asymmetric to give a
      favourable reward:risk (let winners run, cut losers).
    * **Labels** (``label_tp_mult`` / ``label_sl_mult``) — should be **symmetric**.
      An asymmetric tp/sl makes the nearer barrier far easier to touch, which biases
      the training labels (and therefore the model's direction calls) toward one
      side — e.g. a tight stop + wide target yields mostly ``-1`` labels and a
      chronically short-biased model that bleeds in an uptrend. Defaulting them to
      ``None`` falls back to ``max(tp_mult, sl_mult)`` on both sides (symmetric).
    """

    tp_mult: float = 2.0
    sl_mult: float = 1.0
    horizon: int = 18
    label_tp_mult: float | None = None
    label_sl_mult: float | None = None
    # Labelling method for the *training target* (exits always use the ATR barriers above):
    #   "triple_barrier" — which ATR barrier (tp/sl) is touched first within `horizon`.
    #   "trend_scan"     — sign of the most statistically-significant forward trend.
    label_method: str = "triple_barrier"
    ts_min_window: int = 5           # trend-scan: shortest forward window (bars)
    ts_max_window: int = 20          # trend-scan: longest forward window (bars)
    ts_t_threshold: float = 0.0      # trend-scan: min |t-stat| for a non-zero label (0 = always sign)

    @property
    def label_barriers(self) -> tuple[float, float]:
        """(tp, sl) multipliers for *labelling* — symmetric unless overridden."""
        sym = max(self.tp_mult, self.sl_mult)
        return (self.label_tp_mult or sym, self.label_sl_mult or sym)

    @property
    def lookahead(self) -> int:
        """How many trailing bars have no full forward label window (to trim in training)."""
        return self.ts_max_window if self.label_method == "trend_scan" else self.horizon


class NotificationConfig(BaseModel):
    """Out-of-band alerts (best-effort) for unattended 24/7 operation.

    Secrets (telegram_bot_token) belong in the git-ignored config/secrets.yaml, not
    config.yaml. ``min_level`` gates which messages are sent: info < warning < critical.
    """

    webhook_url: str | None = None         # generic JSON POST {"text": ...} (Slack/Discord/…)
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    notify_trades: bool = False            # also alert on every fill (noisy)
    daily_summary: bool = False            # once-a-day equity/PnL summary while trading
    min_level: str = "warning"             # info | warning | critical


class DashboardConfig(BaseModel):
    """Optional HTTP Basic-auth for the dashboard/API (for safe remote exposure).

    Auth is OFF unless ``auth_password`` is set. Put the password in the git-ignored
    config/secrets.yaml, not config.yaml. ``/api/health`` stays open for uptime probes.
    """

    auth_user: str = "admin"
    auth_password: str | None = None


class MT5Config(BaseModel):
    """MetaTrader 5 bridge: a MQL5 Expert Advisor calls /api/mt5/* over HTTP.

    The EA pushes its broker bars/position and asks this service for a trading
    decision; the EA then places the broker order itself. This keeps the whole ML
    brain on Linux/Coolify (no Windows-only MetaTrader5 Python package needed).

    Auth is a shared token (``api_token``) the EA sends as the ``X-API-Token`` header;
    put it in the git-ignored config/secrets.yaml, not config.yaml. ``symbol_map`` maps
    each MT5 broker symbol to the internal model symbol, e.g. {"BTCUSD": "BTC/USDT"} —
    explicit mapping is required for anything the crypto heuristic can't infer (forex,
    indices, commodities). ``require_model`` refuses to trade a symbol that has no
    matching trained model (never falls back to the momentum baseline for real orders).
    """

    enabled: bool = False
    api_token: str | None = None                  # shared secret for /api/mt5/* (secrets.yaml)
    symbol_map: dict[str, str] = Field(default_factory=dict)   # MT5 symbol -> internal symbol
    default_timeframe: str | None = None          # fallback if the EA omits one (else exchange.timeframe)
    require_model: bool = True                     # don't trade symbols without a matching model
    capture_bars: bool = True                     # persist bars the EA sends (offline training set)
    capture_max_rows: int = 200_000               # cap per symbol/timeframe parquet
    poll_seconds: float = 60.0                     # advisory cadence reported to the EA


class PersistenceConfig(BaseModel):
    db_path: Path = Path("data/cryptotrader.sqlite")


class Settings(BaseSettings):
    """Root settings object. Combine YAML defaults with environment overrides."""

    model_config = SettingsConfigDict(
        env_prefix="CT_",
        env_nested_delimiter="__",
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    mode: RunMode = RunMode.BACKTEST
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: MLConfig = Field(default_factory=MLConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    barriers: BarrierConfig = Field(default_factory=BarrierConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    notify: NotificationConfig = Field(default_factory=NotificationConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    mt5: MT5Config = Field(default_factory=MT5Config)

    # Source priority (first = highest): init kwargs > environment > YAML > defaults.
    # This is what makes CT_* env vars actually override config.yaml.
    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings,
                                   env_settings, dotenv_settings, file_secret_settings):
        sources = [init_settings, env_settings]
        if _SECRETS_PATH is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=_SECRETS_PATH))
        if _YAML_PATH is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=_YAML_PATH))
        sources.append(file_secret_settings)
        return tuple(sources)

    @classmethod
    def load(cls, path: str | Path = "config/config.yaml") -> "Settings":
        """Load with precedence: environment overrides YAML overrides defaults."""
        global _YAML_PATH, _SECRETS_PATH
        cfg_path = Path(path)
        _YAML_PATH = str(cfg_path) if cfg_path.exists() else None
        secrets_path = Path(SECRETS_FILE)
        _SECRETS_PATH = str(secrets_path) if secrets_path.exists() else None
        try:
            return cls()
        finally:
            _YAML_PATH = None
            _SECRETS_PATH = None
