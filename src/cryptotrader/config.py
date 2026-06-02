# src/cryptotrader/config.py
"""Typed, layered runtime configuration.

Configuration is loaded from ``config/config.yaml`` and overlaid with environment
variables (prefix ``CT_``, nested via ``__``). Secrets such as exchange API keys
MUST be supplied via the environment, never committed to the YAML file.

Example
-------
    CT_EXCHANGE__API_KEY=... CT_MODE=live python -m cryptotrader.api.server
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RunMode(str, enum.Enum):
    """Top-level execution mode of the bot."""

    BACKTEST = "backtest"
    LIVE = "live"


class ExchangeConfig(BaseModel):
    id: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1m"
    api_key: str | None = None
    api_secret: str | None = None


class DataConfig(BaseModel):
    history_days: int = 30
    cache_dir: Path = Path(".cache/ohlcv")


class FeatureConfig(BaseModel):
    atr_period: int = 14
    vwap_window: int = 60
    momentum_windows: list[int] = Field(default_factory=lambda: [3, 5, 15])
    volume_spike_window: int = 30
    zscore_window: int = 60


class RiskConfig(BaseModel):
    account_equity: float = 10_000.0
    risk_per_trade: float = 0.005
    atr_stop_mult: float = 2.0
    atr_trail_mult: float = 1.5
    max_open_positions: int = 1


class ExecutionConfig(BaseModel):
    taker_fee: float = 0.0004
    slippage_bps: float = 1.0


class StrategyConfig(BaseModel):
    long_threshold: float = 0.58
    short_threshold: float = 0.58
    allow_short: bool = True


class PersistenceConfig(BaseModel):
    db_path: Path = Path("data/cryptotrader.sqlite")


class Settings(BaseSettings):
    """Root settings object. Combine YAML defaults with environment overrides."""

    model_config = SettingsConfigDict(
        env_prefix="CT_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    mode: RunMode = RunMode.BACKTEST
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)

    @classmethod
    def load(cls, path: str | Path = "config/config.yaml") -> "Settings":
        """Load YAML defaults, then let environment variables take precedence."""
        raw: dict[str, Any] = {}
        cfg_path = Path(path)
        if cfg_path.exists():
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return cls(**raw)
