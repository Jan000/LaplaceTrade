# tests/test_analytics.py
"""Unit tests for the trade-log analytics used by the dashboard."""

from __future__ import annotations

from cryptotrader.backtest.analytics import summarize_trades


def _t(side, pnl, reason, fees=1.0, eff=0.5, h=1):
    return {
        "side": side, "net_pnl": pnl, "gross_pnl": pnl + fees, "fees": fees,
        "efficiency_ratio": eff, "exit_reason": reason,
        "entry_time": "2024-01-01T00:00:00+00:00",
        "exit_time": f"2024-01-01T0{h}:00:00+00:00",
        "entry_price": 100.0, "exit_price": 110.0, "quantity": 1.0,
    }


def test_summarize_empty() -> None:
    s = summarize_trades([], [])
    assert s["n_trades"] == 0
    assert s["profit_factor"] == 0.0 and s["win_rate"] == 0.0


def test_summarize_basic() -> None:
    trades = [
        _t(1, 10.0, "take_profit"),
        _t(-1, -5.0, "stop_loss"),
        _t(1, 0.0, "time_exit"),
    ]
    equity = [{"timestamp": "t0", "equity": 10000.0},
              {"timestamp": "t1", "equity": 9900.0},
              {"timestamp": "t2", "equity": 10005.0}]
    s = summarize_trades(trades, equity)
    assert s["n_trades"] == 3 and s["wins"] == 1 and s["losses"] == 1 and s["breakeven"] == 1
    assert s["gross_profit"] == 10.0 and s["gross_loss"] == 5.0
    assert s["profit_factor"] == 2.0
    assert s["net_pnl"] == 5.0
    assert "LONG" in s["by_side"] and "SHORT" in s["by_side"]
    assert s["by_reason"]["take_profit"]["n"] == 1
    assert s["max_drawdown_pct"] > 0  # dipped to 9900 from 10000 peak
    assert s["avg_hold_seconds"] == 3600.0


def test_profit_factor_no_losses_is_none() -> None:
    s = summarize_trades([_t(1, 10.0, "take_profit"), _t(1, 5.0, "take_profit")])
    assert s["profit_factor"] is None  # undefined (no losses) -> rendered as infinity in UI
    assert s["wins"] == 2 and s["losses"] == 0
