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
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Set by Settings.load() so the YAML source is only active for an explicit
# load() (keeps a bare Settings() on pure defaults+env, which the tests rely on).
_YAML_PATH: str | None = None


class RunMode(str, enum.Enum):
    BACKTEST = "backtest"
    LIVE = "live"


class ExchangeConfig(BaseModel):
    id: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    api_key: str | None = None
    api_secret: str | None = None


class DataConfig(BaseModel):
    history_days: int = 365
    cache_dir: Path = Path(".cache/ohlcv")
    replay_file: Path | None = None


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
    ensemble_size: int = 1                  # >1 averages N seed-varied models (variance reduction)
    bagging_freq: int = 0                   # LightGBM bagging frequency; >0 makes `subsample` actually bag

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
            "n_jobs": -1,
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


class ExecutionConfig(BaseModel):
    taker_fee: float = 0.0004
    slippage_bps: float = 1.0


class StrategyConfig(BaseModel):
    long_threshold: float = 0.62
    short_threshold: float = 0.62
    allow_short: bool = True
    model_path: Path | None = None
    # Regime filter: when on, only take signals that agree with the slow-EMA trend
    # (long only above the EMA, short only below). Symmetric, so it stays regime-
    # adaptive rather than just betting on the prevailing direction.
    trend_filter: bool = False


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

    @property
    def label_barriers(self) -> tuple[float, float]:
        """(tp, sl) multipliers for *labelling* — symmetric unless overridden."""
        sym = max(self.tp_mult, self.sl_mult)
        return (self.label_tp_mult or sym, self.label_sl_mult or sym)


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

    # Source priority (first = highest): init kwargs > environment > YAML > defaults.
    # This is what makes CT_* env vars actually override config.yaml.
    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings,
                                   env_settings, dotenv_settings, file_secret_settings):
        sources = [init_settings, env_settings]
        if _YAML_PATH is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=_YAML_PATH))
        sources.append(file_secret_settings)
        return tuple(sources)

    @classmethod
    def load(cls, path: str | Path = "config/config.yaml") -> "Settings":
        """Load with precedence: environment overrides YAML overrides defaults."""
        global _YAML_PATH
        cfg_path = Path(path)
        _YAML_PATH = str(cfg_path) if cfg_path.exists() else None
        try:
            return cls()
        finally:
            _YAML_PATH = None
