# tests/test_config.py
"""Config-level logic that isn't tied to the network or the model."""

from __future__ import annotations

from cryptotrader.config import DataConfig


def test_pool_for_never_leaves_a_symbol_unaugmented() -> None:
    d = DataConfig(train_symbols=["ETH/USDT"])
    assert d.pool_for("BTC/USDT") == ["ETH/USDT"]          # alt/leader: pool the configured major
    assert d.pool_for("SOL/USDT") == ["ETH/USDT"]          # other alts: unchanged
    # ETH would self-skip to an empty pool -> fall back to the market leader (BTC).
    assert d.pool_for("ETH/USDT") == ["BTC/USDT"]
    # BTC falling back picks ETH, not itself.
    assert DataConfig(train_symbols=["BTC/USDT"]).pool_for("BTC/USDT") == ["ETH/USDT"]
    # An explicitly empty pool (no augmentation intended) is respected.
    assert DataConfig(train_symbols=[]).pool_for("ETH/USDT") == []
