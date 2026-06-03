# src/cryptotrader/backtest/portfolio.py
"""Portfolio state: position lifecycle, equity tracking, and trade analytics.

The portfolio is intentionally PnL-centric rather than cash/margin-centric: for an
MVP backtest we track *realized equity* (seeded capital +/- closed PnL and fees)
plus the *unrealized* PnL of any open position. This is exchange-model agnostic
(works for spot and linear perps alike) and keeps the accounting auditable.

The portfolio also owns the **Max Efficiency Ratio** computation: while a position
is open it records the most favourable price excursion (MFE), and on close it
reports what fraction of that best-case move the strategy actually banked.
"""

from __future__ import annotations

from datetime import datetime

from cryptotrader.core.events import FillEvent
from cryptotrader.core.types import Position, Side, Trade

_EPS = 1e-12


class Portfolio:
    """Tracks equity, the (single) open position, and closed trades."""

    def __init__(self, initial_equity: float, symbol: str) -> None:
        self.initial_equity = initial_equity
        self.realized_equity = initial_equity
        self.symbol = symbol
        self.position: Position | None = None
        self.trades: list[Trade] = []
        # (timestamp, equity) samples for the equity curve / drawdown analysis.
        self.equity_curve: list[tuple[datetime, float]] = []

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def has_position(self) -> bool:
        return self.position is not None

    @property
    def position_side(self) -> Side:
        return self.position.side if self.position else Side.FLAT

    def equity(self, price: float) -> float:
        """Mark-to-market equity at ``price``."""
        unreal = self.position.unrealized_pnl(price) if self.position else 0.0
        return self.realized_equity + unreal

    def mark(self, timestamp: datetime, price: float) -> None:
        """Append a sample to the equity curve."""
        self.equity_curve.append((timestamp, self.equity(price)))

    # ------------------------------------------------------------------ #
    # Position lifecycle
    # ------------------------------------------------------------------ #
    def open_position(
        self,
        fill: FillEvent,
        stop_distance: float,
        tp_distance: float,
        max_hold_bars: int = 0,
    ) -> None:
        """Open a new position from an entry fill.

        Protective levels (the triple barriers) are anchored to the *actual* fill
        price (not the signal-bar close), so slippage and next-bar gaps are
        reflected: a stop-loss at ``stop_distance``, a take-profit at
        ``tp_distance``, and a vertical (time) barrier of ``max_hold_bars`` bars.
        """
        if self.position is not None:
            raise RuntimeError("Cannot open a position while one is already open.")
        self.realized_equity -= fill.fee
        direction = int(fill.side)
        entry = fill.fill_price
        self.position = Position(
            symbol=fill.symbol,
            side=fill.side,
            quantity=fill.quantity,
            entry_price=entry,
            entry_time=fill.timestamp,
            stop_loss=entry - direction * stop_distance,
            take_profit=entry + direction * tp_distance,
            trail_distance=0.0,  # fixed-barrier exits; no trailing
            max_hold_bars=max_hold_bars,
            mfe_price=entry,
            mae_price=entry,
        )

    def update_excursion(self, high: float, low: float) -> None:
        """Update the most-favourable / most-adverse excursion for the open pos."""
        pos = self.position
        if pos is None:
            return
        if pos.side is Side.LONG:
            pos.mfe_price = max(pos.mfe_price, high)
            pos.mae_price = min(pos.mae_price, low)
        else:  # SHORT
            pos.mfe_price = min(pos.mfe_price, low)
            pos.mae_price = max(pos.mae_price, high)

    def trailing_stop(self) -> float:
        """Current effective stop level.

        The hard ATR stop protects the trade until it has run far enough in our
        favour that trailing the peak by ``trail_distance`` would lock in at least
        break-even. From that point the stop ratchets monotonically with the MFE,
        turning into a trailing take-profit. It never loosens.
        """
        pos = self.position
        if pos is None:
            return 0.0
        if pos.trail_distance <= 0.0:
            return pos.stop_loss

        if pos.side is Side.LONG:
            candidate = pos.mfe_price - pos.trail_distance
            if candidate <= pos.entry_price:  # not yet break-even -> hard stop
                return pos.stop_loss
            return max(pos.stop_loss, candidate)

        # SHORT
        candidate = pos.mfe_price + pos.trail_distance
        if candidate >= pos.entry_price:
            return pos.stop_loss
        return min(pos.stop_loss, candidate)

    def close_position(self, fill: FillEvent, exit_reason: str) -> Trade:
        """Close the open position from an exit fill and return the Trade."""
        pos = self.position
        if pos is None:
            raise RuntimeError("No open position to close.")

        direction = int(pos.side)
        gross_pnl = direction * (fill.fill_price - pos.entry_price) * pos.quantity
        # Entry fee already deducted at open; subtract the exit fee now.
        net_pnl = gross_pnl - fill.fee
        self.realized_equity += net_pnl

        best_price = pos.mfe_price
        potential = direction * (best_price - pos.entry_price)
        realized = direction * (fill.fill_price - pos.entry_price)
        # Share of the best-case favourable move captured. Clamped to [-1, 1]:
        # 1.0 = exited at the high; 0 = no favourable move existed; negative =
        # gave back the run and then some (poor exit). Clamping keeps the
        # aggregate robust to tiny-denominator outliers.
        if potential > _EPS:
            efficiency = max(-1.0, min(1.0, realized / potential))
        else:
            efficiency = 0.0

        trade = Trade(
            symbol=pos.symbol,
            side=pos.side,
            quantity=pos.quantity,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=fill.timestamp,
            exit_price=fill.fill_price,
            fees=fill.fee,  # exit-leg fee (entry fee already realized)
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            best_price=best_price,
            exit_reason=exit_reason,
            efficiency_ratio=efficiency,
        )
        self.trades.append(trade)
        self.position = None
        return trade
