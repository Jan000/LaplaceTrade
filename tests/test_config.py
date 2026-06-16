# tests/test_config.py
"""Config-level logic that isn't tied to the network or the model."""

from __future__ import annotations

from cryptotrader.config import DataConfig, MLConfig


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


def test_resolved_n_jobs_bounds_threads() -> None:
    import os

    assert MLConfig(n_jobs=2).resolved_n_jobs() == 2          # explicit
    assert MLConfig(n_jobs=-1).resolved_n_jobs() == -1        # all cores (opt-in)
    auto = MLConfig(n_jobs=0).resolved_n_jobs()               # default: half the cores, >=1
    assert auto == max(1, (os.cpu_count() or 4) // 2) and auto >= 1
    assert MLConfig(n_jobs=0).to_lgbm_params()["n_jobs"] == auto


def test_candidate_promote_and_discard(tmp_path, monkeypatch) -> None:
    import cryptotrader.ml.registry as reg

    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path)
    sym = "BTC/USDT"
    cand = reg.candidate_path_for(sym)
    cand.write_bytes(b"candidate-model")
    reg.meta_path_for(cand).write_text('{"symbol": "BTC/USDT"}', encoding="utf-8")
    assert reg.has_candidate(sym)

    assert reg.promote_candidate(sym) is True            # candidate -> active
    assert reg.model_path_for(sym).read_bytes() == b"candidate-model"
    assert not reg.has_candidate(sym)                    # candidate consumed
    assert reg.promote_candidate(sym) is False           # nothing left to promote

    cand.write_bytes(b"x")
    assert reg.discard_candidate(sym) is True and not reg.has_candidate(sym)
