"""Advanced Position Sizing — Kelly Criterion, Risk-of-Ruin, Portfolio Heat.

Inspired by freqtrade's Edge positioning (Kelly-based sizing) and
professional risk management frameworks. Provides optimal position
sizing beyond simple percent-of-account.

Features:
- Kelly Criterion with half-Kelly conservative mode
- Risk-of-ruin calculator
- Portfolio heat tracking (aggregate risk)
- Correlation-adjusted sizing
- Sector concentration limits
- ATR-volatility-adjusted sizing
- Regime-aware size multiplier application
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from loguru import logger


@dataclass
class TradeRecord:
    """Historical trade outcome for sizing calculations."""
    symbol: str
    strategy: str
    pnl: float
    entry_price: float
    exit_price: float
    quantity: float
    side: str = "LONG"
    sector: str = ""
    timestamp: str = ""


@dataclass
class SizingResult:
    """Output of a position sizing calculation."""
    quantity: Decimal
    method: str              # "kelly" | "fixed_risk" | "atr" | "regime"
    risk_per_trade: float    # Dollar risk
    kelly_fraction: float    # Raw kelly %
    applied_fraction: float  # Actual fraction used (after caps/adjustments)
    portfolio_heat: float    # Total portfolio heat after this trade
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "quantity": str(self.quantity),
            "method": self.method,
            "risk_per_trade": self.risk_per_trade,
            "kelly_fraction": round(self.kelly_fraction, 4),
            "applied_fraction": round(self.applied_fraction, 4),
            "portfolio_heat": round(self.portfolio_heat, 4),
            "notes": self.notes,
        }


class PositionSizer:
    """Advanced position sizing with Kelly Criterion and portfolio constraints.

    Usage:
        sizer = PositionSizer(account_size=100_000, max_portfolio_heat=0.06)
        sizer.add_trade_record(TradeRecord(...))
        result = sizer.calculate_size(
            symbol="AAPL", entry_price=150.0, stop_loss=147.0,
            strategy="momentum"
        )
    """

    def __init__(
        self,
        account_size: float = 100_000.0,
        max_risk_per_trade: float = 0.02,     # 2% of account per trade
        max_portfolio_heat: float = 0.06,     # 6% total portfolio heat
        max_position_pct: float = 0.10,       # 10% max single position
        kelly_fraction: float = 0.5,          # Half-Kelly (conservative)
        min_trade_history: int = 20,          # Minimum trades for Kelly
        max_sector_concentration: float = 0.25,  # 25% max in one sector
        regime_multiplier: float = 1.0,       # From regime detector
    ):
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_heat = max_portfolio_heat
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        self.min_trade_history = min_trade_history
        self.max_sector_concentration = max_sector_concentration
        self.regime_multiplier = regime_multiplier

        # Trade history (per strategy)
        self._trade_history: list[TradeRecord] = []
        self._open_positions: dict[str, dict] = {}  # symbol -> {qty, risk$}
        self._sector_exposure: dict[str, float] = {}  # sector -> notional$
        self._correlation_penalties: dict[str, float] = {}  # symbol -> scale 0-1

    # ── Core Sizing ───────────────────────────────────────────────────────────

    def calculate_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        strategy: str = "",
        atr: float | None = None,
        sector: str = "",
    ) -> SizingResult:
        """Calculate optimal position size considering all constraints.

        Priority chain:
        1. Kelly criterion (if enough history)
        2. Fixed fractional risk (fallback)
        3. ATR-based adjustment
        4. Portfolio heat cap
        5. Sector concentration cap
        6. Regime multiplier
        """
        notes = []
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            risk_per_share = entry_price * 0.02  # Use 2% default
            notes.append("No stop distance, using 2% default risk")

        # ── Step 1: Kelly or Fixed Fractional ─────────────────────────────
        strategy_trades = [t for t in self._trade_history if t.strategy == strategy]
        if len(strategy_trades) >= self.min_trade_history:
            kelly = self._kelly_criterion(strategy_trades)
            fraction = kelly * self.kelly_fraction  # Half-Kelly
            method = "kelly"
            notes.append(f"Kelly raw={kelly:.4f}, applied={fraction:.4f} (×{self.kelly_fraction})")
        else:
            fraction = self.max_risk_per_trade
            kelly = 0.0
            method = "fixed_risk"
            notes.append(f"Not enough trades for Kelly ({len(strategy_trades)}/{self.min_trade_history})")

        # Cap fraction
        fraction = max(0.001, min(fraction, self.max_risk_per_trade))

        # ── Step 2: Calculate base quantity ───────────────────────────────
        risk_dollars = self.account_size * fraction
        base_qty = risk_dollars / risk_per_share

        # ── Step 3: ATR adjustment ────────────────────────────────────────
        if atr and atr > 0:
            atr_risk = atr * 2  # 2×ATR as risk measure
            atr_qty = risk_dollars / atr_risk
            if atr_qty < base_qty:
                base_qty = atr_qty
                notes.append(f"ATR-adjusted: using 2×ATR={atr_risk:.2f}")

        # ── Step 4: Position size cap ─────────────────────────────────────
        max_notional = self.account_size * self.max_position_pct
        max_qty_by_notional = max_notional / entry_price
        if base_qty > max_qty_by_notional:
            base_qty = max_qty_by_notional
            notes.append(f"Capped by max position ({self.max_position_pct:.0%} of account)")

        # ── Step 5: Portfolio heat check ──────────────────────────────────
        current_heat = self._current_portfolio_heat()
        new_heat = current_heat + (risk_dollars / self.account_size)
        if new_heat > self.max_portfolio_heat:
            available_heat = max(0, self.max_portfolio_heat - current_heat)
            available_risk = available_heat * self.account_size
            heat_qty = available_risk / risk_per_share if risk_per_share > 0 else 0
            if heat_qty < base_qty:
                base_qty = heat_qty
                risk_dollars = available_risk
                notes.append(f"Portfolio heat limited: {current_heat:.2%} → {new_heat:.2%} (max {self.max_portfolio_heat:.2%})")

        # ── Step 6: Sector concentration ──────────────────────────────────
        if sector:
            sector_usd = self._sector_exposure.get(sector, 0.0)
            sector_pct = sector_usd / self.account_size if self.account_size > 0 else 0
            if sector_pct >= self.max_sector_concentration:
                notes.append(f"Sector {sector} at {sector_pct:.1%} — max reached")
                base_qty = 0

        # ── Step 6b: Correlation-adjusted size reduction ──────────────────
        if self._correlation_penalties and symbol in self._correlation_penalties:
            penalty = self._correlation_penalties[symbol]
            if penalty < 1.0:
                base_qty *= penalty
                notes.append(f"Correlation penalty ×{penalty:.2f}")

        # ── Step 7: Regime multiplier ─────────────────────────────────────
        if self.regime_multiplier != 1.0:
            base_qty *= self.regime_multiplier
            notes.append(f"Regime multiplier ×{self.regime_multiplier:.2f}")

        # Floor at 1 share (or 0 if rejected)
        final_qty = max(0, int(base_qty))
        final_heat = current_heat + (risk_dollars / self.account_size if self.account_size > 0 else 0)

        return SizingResult(
            quantity=Decimal(str(final_qty)),
            method=method,
            risk_per_trade=round(risk_dollars, 2),
            kelly_fraction=kelly,
            applied_fraction=fraction,
            portfolio_heat=round(final_heat, 4),
            notes=notes,
        )

    # ── Kelly Criterion ───────────────────────────────────────────────────────

    def _kelly_criterion(self, trades: list[TradeRecord]) -> float:
        """Calculate Kelly fraction: f* = (bp - q) / b

        Where:
            b = average win / average loss (win/loss ratio)
            p = probability of winning
            q = probability of losing (1 - p)
        """
        if not trades:
            return 0.0

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        if not losses:
            return self.max_risk_per_trade  # All wins = max allocation
        if not wins:
            return 0.0  # All losses = don't trade

        p = len(wins) / len(trades)
        q = 1 - p
        avg_win = sum(t.pnl for t in wins) / len(wins)
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses))

        if avg_loss == 0:
            return self.max_risk_per_trade

        b = avg_win / avg_loss  # Win/loss ratio
        kelly = (b * p - q) / b

        # Kelly can be negative (don't trade!)
        return max(0.0, kelly)

    # ── Risk of Ruin ──────────────────────────────────────────────────────────

    def risk_of_ruin(self, win_rate: float | None = None,
                     win_loss_ratio: float | None = None,
                     risk_per_trade: float | None = None,
                     ruin_threshold: float = 0.5) -> float:
        """Calculate probability of losing ruin_threshold% of account.

        Uses simplified formula: RoR = ((1-edge)/(1+edge))^(account_units)
        where edge = win_rate × win_loss_ratio - (1 - win_rate)
        and account_units = ruin_threshold / risk_per_trade
        """
        if win_rate is None or win_loss_ratio is None:
            trades = self._trade_history
            if len(trades) < 10:
                return 1.0  # Unknown = assume high risk
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]
            win_rate = len(wins) / len(trades)
            avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
            avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 1
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1

        risk = risk_per_trade or self.max_risk_per_trade
        edge = win_rate * win_loss_ratio - (1 - win_rate)

        if edge <= 0:
            return 1.0  # Negative edge = certain ruin

        # Number of "risk units" to ruin
        units = ruin_threshold / risk if risk > 0 else 100

        ror = ((1 - edge) / (1 + edge)) ** units
        return min(1.0, max(0.0, ror))

    # ── Expectancy ────────────────────────────────────────────────────────────

    def expectancy(self, strategy: str = "") -> dict:
        """Calculate system expectancy per trade.

        Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
        Positive expectancy = profitable system over time.
        """
        trades = self._trade_history
        if strategy:
            trades = [t for t in trades if t.strategy == strategy]

        if not trades:
            return {"expectancy": 0, "trades": 0, "sufficient_data": False}

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        win_rate = len(wins) / len(trades)
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.pnl for t in losses) / len(losses)) if losses else 0

        expectancy_val = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        profit_factor = (sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))
                         if losses and sum(t.pnl for t in losses) != 0 else float("inf"))

        return {
            "expectancy": round(expectancy_val, 2),
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
            "win_loss_ratio": round(avg_win / avg_loss, 2) if avg_loss > 0 else "∞",
            "trades": len(trades),
            "sufficient_data": len(trades) >= self.min_trade_history,
            "risk_of_ruin": round(self.risk_of_ruin(win_rate,
                                                     avg_win / avg_loss if avg_loss > 0 else 1), 4),
        }

    # ── Portfolio Heat ────────────────────────────────────────────────────────

    def _current_portfolio_heat(self) -> float:
        """Sum of all open position risks as fraction of account."""
        total_risk = sum(pos.get("risk_dollars", 0) for pos in self._open_positions.values())
        return total_risk / self.account_size if self.account_size > 0 else 0

    def register_position(self, symbol: str, quantity: float,
                          entry_price: float, stop_loss: float,
                          sector: str = "") -> None:
        """Register an open position for heat tracking."""
        risk_per_share = abs(entry_price - stop_loss)
        risk_dollars = risk_per_share * quantity
        notional = entry_price * quantity

        self._open_positions[symbol] = {
            "quantity": quantity,
            "entry": entry_price,
            "stop": stop_loss,
            "risk_dollars": risk_dollars,
            "notional": notional,
            "sector": sector,
        }

        if sector:
            self._sector_exposure[sector] = self._sector_exposure.get(sector, 0) + notional

        logger.debug("Position registered: {} | risk=${:.2f} | heat={:.2%}",
                      symbol, risk_dollars, self._current_portfolio_heat())

    def unregister_position(self, symbol: str) -> None:
        """Remove a closed position from heat tracking."""
        pos = self._open_positions.pop(symbol, None)
        if pos and pos.get("sector"):
            sector = pos["sector"]
            self._sector_exposure[sector] = max(
                0, self._sector_exposure.get(sector, 0) - pos["notional"]
            )

    # ── Trade History ─────────────────────────────────────────────────────────

    def add_trade_record(self, record: TradeRecord) -> None:
        """Add a completed trade to history for sizing calculations."""
        self._trade_history.append(record)
        # Keep last 500 trades
        if len(self._trade_history) > 500:
            self._trade_history = self._trade_history[-500:]

    def update_account_size(self, new_size: float) -> None:
        """Update account size (call after balance changes)."""
        self.account_size = new_size

    def update_regime_multiplier(self, multiplier: float) -> None:
        """Update the regime-based sizing multiplier."""
        self.regime_multiplier = max(0.1, min(2.0, multiplier))
        logger.debug("Regime multiplier updated: {:.2f}", self.regime_multiplier)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return current sizing state and statistics."""
        return {
            "account_size": self.account_size,
            "portfolio_heat": round(self._current_portfolio_heat(), 4),
            "open_positions": len(self._open_positions),
            "total_trades": len(self._trade_history),
            "regime_multiplier": self.regime_multiplier,
            "sector_exposure": dict(self._sector_exposure),
            "strategies": self._per_strategy_stats(),
        }

    def _per_strategy_stats(self) -> dict:
        """Compute per-strategy Kelly and expectancy."""
        strategies = set(t.strategy for t in self._trade_history)
        result = {}
        for s in strategies:
            if not s:
                continue
            result[s] = self.expectancy(strategy=s)
        return result

    def update_correlation_penalties(
        self, penalties: dict[str, float]
    ) -> None:
        """Set per-symbol sizing penalties based on portfolio correlation.

        Args:
            penalties: mapping of symbol → multiplier (0.0–1.0).
                       e.g. {"NVDA": 0.6} means size for NVDA is reduced to 60%.
        """
        self._correlation_penalties = penalties
        if penalties:
            logger.debug("Correlation penalties updated: {}", penalties)

    def reset_daily(self) -> None:
        """Reset daily state (keep trade history)."""
        self._open_positions.clear()
        self._sector_exposure.clear()
        self._correlation_penalties.clear()
