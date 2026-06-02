# src/cryptotrader/backtest/metrics.py
"""Performance analytics for a completed backtest.

Includes the bespoke **Max Efficiency Ratio**: the share of each trade's best-case
favourable move (entry -> MFE) that the strategy actually captured, averaged
across trades. It isolates *exit quality* from *entry quality* — a strategy can
have great entries (high hit rate) yet bleed edge through poor exits (low
efficiency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from cryptotrader.core.types import Trade

# 1-minute bars per year, used to annualise the per-bar Sharpe ratio.
_BARS_PER_YEAR = 525_600


@dataclass(slots=True)
class PerformanceReport:
    """Aggregate backtest statistics."""

    initial_equity: float
    final_equity: float
    total_return_pct: float
    n_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    max_drawdown_pct: float
    sharpe_ratio: float
    # The headline custom metric.
    avg_efficiency_ratio: float
    median_efficiency_ratio: float
    total_fees: float
    exit_reason_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | int | dict[str, int]]:
        return {
            "initial_equity": round(self.initial_equity, 2),
            "final_equity": round(self.final_equity, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_trade_pnl": round(self.avg_trade_pnl, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "avg_efficiency_ratio": round(self.avg_efficiency_ratio, 4),
            "median_efficiency_ratio": round(self.median_efficiency_ratio, 4),
            "total_fees": round(self.total_fees, 4),
            "exit_reason_counts": self.exit_reason_counts,
        }


def _max_drawdown(equity: list[float]) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction."""
    peak = -math.inf
    max_dd = 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)
    return max_dd


def _sharpe(equity: list[float]) -> float:
    """Annualised Sharpe from the per-bar equity curve (rf = 0)."""
    if len(equity) < 3:
        return 0.0
    rets = [
        (equity[i] - equity[i - 1]) / equity[i - 1]
        for i in range(1, len(equity))
        if equity[i - 1] != 0
    ]
    n = len(rets)
    if n < 2:
        return 0.0
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(_BARS_PER_YEAR)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def compute_metrics(
    trades: list[Trade],
    equity_curve: list[tuple[datetime, float]],
    initial_equity: float,
) -> PerformanceReport:
    """Build a :class:`PerformanceReport` from trades and the equity curve."""
    equity_values = [e for _, e in equity_curve] or [initial_equity]
    final_equity = equity_values[-1]

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]
    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    n = len(trades)

    efficiencies = [t.efficiency_ratio for t in trades]
    exit_counts: dict[str, int] = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    return PerformanceReport(
        initial_equity=initial_equity,
        final_equity=final_equity,
        total_return_pct=(final_equity / initial_equity - 1.0) * 100.0,
        n_trades=n,
        win_rate=len(wins) / n if n else 0.0,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else math.inf,
        avg_trade_pnl=(sum(t.net_pnl for t in trades) / n) if n else 0.0,
        max_drawdown_pct=_max_drawdown(equity_values) * 100.0,
        sharpe_ratio=_sharpe(equity_values),
        avg_efficiency_ratio=(sum(efficiencies) / n) if n else 0.0,
        median_efficiency_ratio=_median(efficiencies),
        total_fees=sum(t.fees for t in trades),
        exit_reason_counts=exit_counts,
    )
