# tests/test_mt5_bridge.py
"""MT5 bridge decision logic — fully offline (synthetic bars, stubbed predictor)."""

from __future__ import annotations

import cryptotrader.integrations.mt5.bridge as br
from cryptotrader.config import Settings
from cryptotrader.core.types import Prediction, Side
from cryptotrader.data.ingestion import make_synthetic_ohlcv


def _settings() -> Settings:
    s = Settings()
    s.mt5.enabled = True
    s.mt5.require_model = False        # fall back to baseline; tests stub the predictor anyway
    s.mt5.symbol_map = {"US30": "DOW/USD"}
    # Permissive strategy/risk so a stubbed direction deterministically becomes an entry.
    s.strategy.trend_filter = False
    s.strategy.vol_gate = False
    s.strategy.long_threshold = 0.0
    s.strategy.short_threshold = 0.0
    s.strategy.allow_short = True
    s.risk.use_ev_filter = False
    s.risk.min_edge_cost_ratio = 0.0
    return s


def _bars(n: int = 400) -> list[dict]:
    df = make_synthetic_ohlcv(n=n, seed=3)
    return [{"t": ts.isoformat(), "o": r.open, "h": r.high, "l": r.low,
             "c": r.close, "v": r.volume} for ts, r in df.iterrows()]


class _Stub:
    """Predictor returning a fixed direction with high conviction."""

    def __init__(self, direction: Side) -> None:
        self.direction = direction

    def predict(self, row):
        return Prediction(self.direction, 0.99, (0.0, 0.0, 1.0))

    def predict_batch(self, X):
        return [Prediction(self.direction, 0.99, ()) for _ in range(len(X))]


# --------------------------------------------------------------------------- #
# Symbol mapping
# --------------------------------------------------------------------------- #
def test_map_symbol_explicit_heuristic_and_unmapped() -> None:
    s = _settings()
    assert br.map_symbol(s, "US30") == "DOW/USD"          # explicit map
    assert br.map_symbol(s, "us30") == "DOW/USD"          # case-insensitive
    assert br.map_symbol(s, "BTCUSDT") == "BTC/USDT"      # heuristic
    assert br.map_symbol(s, "BTCUSD") == "BTC/USDT"       # fiat USD -> USDT model space
    assert br.map_symbol(s, "ETHUSDC") == "ETH/USDC"
    assert br.map_symbol(s, "RANDOMX") is None            # nothing inferable


# --------------------------------------------------------------------------- #
# Decision branches
# --------------------------------------------------------------------------- #
def test_insufficient_bars() -> None:
    out = br.decide(_settings(), symbol="BTCUSD", bars=_bars(10))
    assert out["action"] == "none" and out["reason"] == "insufficient_bars"


def test_flat_plus_long_signal_opens(monkeypatch) -> None:
    monkeypatch.setattr(br, "_load_predictor", lambda s, sym: (_Stub(Side.LONG), {"matches": True}))
    out = br.decide(_settings(), symbol="BTCUSD", bars=_bars())
    assert out["action"] == "open_long"
    assert out["direction"] == "long"
    assert out["stop_loss"] > 0 and out["take_profit"] > out["stop_loss"]
    assert out["risk_fraction"] > 0


def test_in_position_same_side_holds(monkeypatch) -> None:
    monkeypatch.setattr(br, "_load_predictor", lambda s, sym: (_Stub(Side.LONG), {}))
    out = br.decide(_settings(), symbol="BTCUSD", bars=_bars(),
                    position={"side": "long", "volume": 0.1, "bars_held": 1})
    assert out["action"] == "hold" and out["reason"] == "in_position"


def test_in_position_opposite_signal_closes(monkeypatch) -> None:
    monkeypatch.setattr(br, "_load_predictor", lambda s, sym: (_Stub(Side.SHORT), {}))
    out = br.decide(_settings(), symbol="BTCUSD", bars=_bars(),
                    position={"side": "long", "volume": 0.1, "bars_held": 1})
    assert out["action"] == "close" and out["reason"] == "reverse_signal"
    assert out["next_direction"] == "short"


def test_time_exit_closes_aged_position(monkeypatch) -> None:
    monkeypatch.setattr(br, "_load_predictor", lambda s, sym: (_Stub(Side.LONG), {}))
    s = _settings()
    horizon = s.barriers.horizon
    out = br.decide(s, symbol="BTCUSD", bars=_bars(),
                    position={"side": "long", "volume": 0.1, "bars_held": horizon + 5})
    assert out["action"] == "close" and out["reason"] == "time_exit"


def test_require_model_refuses_without_model(monkeypatch, tmp_path) -> None:
    import cryptotrader.ml.registry as reg

    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path)     # empty registry -> no model on disk
    s = _settings()
    s.mt5.require_model = True
    out = br.decide(s, symbol="BTCUSD", bars=_bars())
    assert out["action"] == "none" and out["reason"] == "no_model"
    assert out["model"]["matches"] is False
