"""Performance Analyzer — tracks and evaluates trading performance.

Reads from the same SQLite database as the bot.  Computes:
- Daily / weekly / monthly P&L
- Win rate, profit factor, avg win vs avg loss
- Per-strategy breakdown
- Sharpe ratio (annualised)
- Max drawdown
- Consecutive loss detection (tilt alert)

Produces a PerformanceReport that the AutoTuner uses to decide adjustments.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from loguru import logger

try:
    import aiosqlite
except ImportError:
    aiosqlite = None  # type: ignore


# ── Report ────────────────────────────────────────────────────────────────────


@dataclass
class StrategyMetrics:
    """Performance metrics for a single strategy."""

    name: str = ""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: Decimal = Decimal("0")
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    win_rate: float = 0.0
    profit_factor: float = 0.0  # gross_profit / gross_loss
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")
    avg_hold_time_s: float = 0.0


@dataclass
class PerformanceReport:
    """Full performance snapshot."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_days: int = 7  # Look-back window

    # Aggregate
    total_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    daily_pnl_series: list[Decimal] = field(default_factory=list)
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    consecutive_losses: int = 0

    # Per-strategy
    strategy_metrics: dict[str, StrategyMetrics] = field(default_factory=dict)

    # Alerts
    alerts: list[str] = field(default_factory=list)

    @property
    def is_degraded(self) -> bool:
        """True if performance is notably poor."""
        return (
            self.win_rate < 0.35
            or self.profit_factor < 0.8
            or self.consecutive_losses >= 5
            or self.total_pnl < Decimal("-100")
        )

    def __str__(self) -> str:
        return (
            f"[PERF {self.period_days}d] trades={self.total_trades} "
            f"pnl=${self.total_pnl:.2f} wr={self.win_rate:.1%} "
            f"pf={self.profit_factor:.2f} sharpe={self.sharpe_ratio:.2f} "
            f"dd={self.max_drawdown_pct:.1f}% consec_loss={self.consecutive_losses}"
        )


# ── Analyzer ──────────────────────────────────────────────────────────────────


class PerformanceAnalyzer:
    """Reads trade history and computes performance metrics."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def analyze(self, period_days: int = 7) -> PerformanceReport:
        """Analyze performance over the last N days."""
        report = PerformanceReport(period_days=period_days)

        if aiosqlite is None:
            report.alerts.append("aiosqlite not installed — cannot analyze")
            return report

        cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                # Check if trades table exists
                cursor = await db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
                )
                if not await cursor.fetchone():
                    report.alerts.append("trades table not yet created — bot hasn't traded yet")
                    return report

                # Fetch trades in period
                cursor = await db.execute(
                    "SELECT * FROM trades WHERE executed_at >= ? ORDER BY executed_at",
                    (cutoff.isoformat(),),
                )
                trades = await cursor.fetchall()

                if not trades:
                    report.alerts.append(f"No trades in last {period_days} days")
                    return report

                # Compute metrics
                self._compute_aggregate(report, trades)
                self._compute_per_strategy(report, trades)
                await self._compute_daily_series(report, db, cutoff)
                self._compute_sharpe(report)
                self._compute_drawdown(report)
                self._detect_consecutive_losses(report, trades)
                self._generate_alerts(report)

        except Exception as e:
            report.alerts.append(f"Analysis error: {e}")
            logger.error("Performance analysis failed: {}", e)

        return report

    def _compute_aggregate(self, report: PerformanceReport, trades: list) -> None:
        """Compute overall win rate, P&L, profit factor."""
        wins, losses = [], []

        for t in trades:
            pnl = Decimal(t["pnl"]) if t["pnl"] else Decimal("0")
            report.total_pnl += pnl
            report.total_trades += 1

            if pnl > 0:
                wins.append(pnl)
            elif pnl < 0:
                losses.append(pnl)

        total = len(wins) + len(losses)
        report.win_rate = len(wins) / total if total > 0 else 0

        gross_profit = sum(wins, Decimal("0"))
        gross_loss = abs(sum(losses, Decimal("0")))
        report.profit_factor = (
            float(gross_profit / gross_loss) if gross_loss > 0 else 999.0
        )

    def _compute_per_strategy(
        self, report: PerformanceReport, trades: list
    ) -> None:
        """Break down metrics by strategy."""
        by_strat: dict[str, list] = {}
        for t in trades:
            strat = t["strategy"] or "unknown"
            by_strat.setdefault(strat, []).append(t)

        for name, strat_trades in by_strat.items():
            m = StrategyMetrics(name=name, total_trades=len(strat_trades))
            wins_pnl, losses_pnl = [], []

            for t in strat_trades:
                pnl = Decimal(t["pnl"]) if t["pnl"] else Decimal("0")
                m.total_pnl += pnl
                if pnl > 0:
                    m.wins += 1
                    wins_pnl.append(pnl)
                    m.largest_win = max(m.largest_win, pnl)
                elif pnl < 0:
                    m.losses += 1
                    losses_pnl.append(pnl)
                    m.largest_loss = min(m.largest_loss, pnl)

            total = m.wins + m.losses
            m.win_rate = m.wins / total if total > 0 else 0
            m.avg_win = (
                sum(wins_pnl, Decimal("0")) / len(wins_pnl) if wins_pnl else Decimal("0")
            )
            m.avg_loss = (
                sum(losses_pnl, Decimal("0")) / len(losses_pnl) if losses_pnl else Decimal("0")
            )

            gp = sum(wins_pnl, Decimal("0"))
            gl = abs(sum(losses_pnl, Decimal("0")))
            m.profit_factor = float(gp / gl) if gl > 0 else 999.0

            report.strategy_metrics[name] = m

    async def _compute_daily_series(
        self, report: PerformanceReport, db: aiosqlite.Connection, cutoff: datetime
    ) -> None:
        """Build daily P&L series from daily_stats table."""
        try:
            cutoff_str = cutoff.strftime("%Y-%m-%d")
            cursor = await db.execute(
                "SELECT date, pnl FROM daily_stats WHERE date >= ? ORDER BY date",
                (cutoff_str,),
            )
            rows = await cursor.fetchall()
            report.daily_pnl_series = [Decimal(r["pnl"]) for r in rows]
        except Exception:
            pass  # Table may not exist yet

    def _compute_sharpe(self, report: PerformanceReport) -> None:
        """Annualised Sharpe ratio from daily P&L series."""
        if len(report.daily_pnl_series) < 2:
            report.sharpe_ratio = 0.0
            return

        daily_returns = [float(d) for d in report.daily_pnl_series]
        mean = statistics.mean(daily_returns)
        stdev = statistics.stdev(daily_returns)
        if stdev == 0:
            report.sharpe_ratio = 0.0
        else:
            # Annualise: ~252 trading days
            report.sharpe_ratio = (mean / stdev) * (252 ** 0.5)

    def _compute_drawdown(self, report: PerformanceReport) -> None:
        """Max drawdown percentage from daily P&L series."""
        if not report.daily_pnl_series:
            return

        cumulative = Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")

        for daily in report.daily_pnl_series:
            cumulative += daily
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        # Express as % of starting equity (approximate)
        report.max_drawdown_pct = float(max_dd / 100000 * 100) if max_dd else 0

    def _detect_consecutive_losses(
        self, report: PerformanceReport, trades: list
    ) -> None:
        """Count current consecutive losing streak."""
        streak = 0
        max_streak = 0

        for t in reversed(trades):
            pnl = Decimal(t["pnl"]) if t["pnl"] else Decimal("0")
            if pnl < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                break  # Streak ended

        report.consecutive_losses = streak

    def _generate_alerts(self, report: PerformanceReport) -> None:
        """Add alerts for concerning patterns."""
        if report.win_rate < 0.35:
            report.alerts.append(
                f"Low win rate: {report.win_rate:.1%} (expected >40%)"
            )

        if report.profit_factor < 0.8:
            report.alerts.append(
                f"Poor profit factor: {report.profit_factor:.2f} (expected >1.0)"
            )

        if report.consecutive_losses >= 5:
            report.alerts.append(
                f"Losing streak: {report.consecutive_losses} consecutive losses"
            )

        if report.sharpe_ratio < -0.5:
            report.alerts.append(
                f"Negative Sharpe: {report.sharpe_ratio:.2f}"
            )

        # Per-strategy alerts
        for name, m in report.strategy_metrics.items():
            if m.total_trades >= 10 and m.win_rate < 0.30:
                report.alerts.append(
                    f"Strategy '{name}' severely underperforming: "
                    f"WR={m.win_rate:.1%}, PnL=${m.total_pnl:.2f}"
                )
