# src/cryptotrader/backtest/analytics.py
"""Trade-log analytics computed from persisted rows (dashboard Trades & Analytics).

Operates on plain dicts as returned by ``TradeStore`` (so it works for any run, or
aggregated across all runs) rather than on live ``Trade`` objects. Everything is
derived from the closed-trade rows; equity-curve-dependent metrics (drawdown,
return) are only filled when an equity series is supplied (a single run).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _side_name(side: Any) -> str:
    if side in (1, "1", "LONG"):
        return "LONG"
    if side in (-1, "-1", "SHORT"):
        return "SHORT"
    return str(side)


def _subset_stats(pnls: list[float]) -> dict[str, Any]:
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "n": n,
        "net_pnl": round(sum(pnls), 2),
        "win_rate": round(len(wins) / n, 4) if n else 0.0,
        # None == undefined (no losing trades); the UI renders it as ∞.
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
    }


def _max_drawdown_pct(equity: list[float]) -> float:
    peak = -math.inf
    mdd = 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            mdd = max(mdd, (peak - e) / peak)
    return round(mdd * 100.0, 4)


def summarize_trades(
    trades: list[dict], equity: list[dict] | None = None
) -> dict[str, Any]:
    """Compute a rich statistics dict from closed-trade rows (+ optional equity)."""
    n = len(trades)
    base: dict[str, Any] = {
        "n_trades": n, "wins": 0, "losses": 0, "breakeven": 0, "win_rate": 0.0,
        "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 0.0,
        "net_pnl": 0.0, "avg_trade": 0.0, "expectancy": 0.0,
        "avg_win": 0.0, "avg_loss": 0.0, "payoff_ratio": 0.0,
        "largest_win": 0.0, "largest_loss": 0.0, "total_fees": 0.0,
        "avg_efficiency": 0.0, "avg_hold_seconds": 0.0,
        "max_consec_wins": 0, "max_consec_losses": 0,
        "by_side": {}, "by_reason": {},
        "max_drawdown_pct": 0.0, "return_pct": 0.0,
        "first_trade": None, "last_trade": None,
    }
    if equity:
        eq = [float(p["equity"]) for p in equity]
        base["max_drawdown_pct"] = _max_drawdown_pct(eq)
        if len(eq) >= 2 and eq[0]:
            base["return_pct"] = round((eq[-1] / eq[0] - 1.0) * 100.0, 4)
    if n == 0:
        return base

    pnls = [float(t["net_pnl"]) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    holds = [
        (_parse(t.get("exit_time")) - _parse(t.get("entry_time"))).total_seconds()
        for t in trades
        if _parse(t.get("exit_time")) and _parse(t.get("entry_time"))
    ]
    # longest win/loss streaks (chronological — rows arrive newest-first, so reverse)
    streak_w = streak_l = mcw = mcl = 0
    for p in reversed(pnls):
        if p > 0:
            streak_w += 1; streak_l = 0; mcw = max(mcw, streak_w)
        elif p < 0:
            streak_l += 1; streak_w = 0; mcl = max(mcl, streak_l)
        else:
            streak_w = streak_l = 0
    effs = [float(t["efficiency_ratio"]) for t in trades if t.get("efficiency_ratio") is not None]
    times = [t.get("entry_time") for t in trades if t.get("entry_time")]

    by_side: dict[str, Any] = {}
    for name in ("LONG", "SHORT"):
        sub = [float(t["net_pnl"]) for t in trades if _side_name(t.get("side")) == name]
        if sub:
            by_side[name] = _subset_stats(sub)
    by_reason: dict[str, Any] = {}
    for t in trades:
        rk = t.get("exit_reason") or "unknown"
        d = by_reason.setdefault(rk, {"n": 0, "net_pnl": 0.0})
        d["n"] += 1
        d["net_pnl"] = round(d["net_pnl"] + float(t["net_pnl"]), 2)

    base.update({
        "wins": len(wins), "losses": len(losses), "breakeven": n - len(wins) - len(losses),
        "win_rate": round(len(wins) / n, 4),
        "gross_profit": round(gross_win, 2), "gross_loss": round(gross_loss, 2),
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else None,
        "net_pnl": round(sum(pnls), 2), "avg_trade": round(sum(pnls) / n, 4),
        "avg_win": round(gross_win / len(wins), 4) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 4) if losses else 0.0,
        "largest_win": round(max(pnls), 2), "largest_loss": round(min(pnls), 2),
        "total_fees": round(sum(float(t.get("fees", 0.0)) for t in trades), 4),
        "avg_efficiency": round(sum(effs) / len(effs), 4) if effs else 0.0,
        "avg_hold_seconds": round(sum(holds) / len(holds), 1) if holds else 0.0,
        "max_consec_wins": mcw, "max_consec_losses": mcl,
        "by_side": by_side, "by_reason": by_reason,
        "first_trade": min(times) if times else None,
        "last_trade": max(times) if times else None,
    })
    aw, al = base["avg_win"], abs(base["avg_loss"])
    base["payoff_ratio"] = round(aw / al, 4) if al > 0 else None
    wr = base["win_rate"]
    base["expectancy"] = round(wr * aw - (1 - wr) * al, 4)
    return base
