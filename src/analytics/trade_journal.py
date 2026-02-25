"""Trade Journal & Analytics — comprehensive trade tracking and analysis.

Provides professional-grade trade analytics beyond basic PnL:
- Per-trade execution quality (slippage measurement)
- Per-strategy performance metrics
- Time-of-day pattern analysis
- Setup type tracking and scoring
- Win streaks, drawdown tracking
- Trade grading (A/B/C/D/F)
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class JournalEntry:
    """A single completed trade in the journal."""

    # Identity
    symbol: str
    strategy: str
    side: str  # "LONG" or "SHORT"

    # Prices / Qty
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: str = ""
    exit_time: str = ""

    # Execution quality
    expected_entry: float = 0.0    # Price at signal time
    expected_exit: float = 0.0     # Target/stop price
    entry_slippage: float = 0.0    # actual - expected
    exit_slippage: float = 0.0

    # PnL
    gross_pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0
    pnl_percent: float = 0.0

    # Context
    setup_type: str = ""           # "RSI_bounce", "VWAP_reversion", etc.
    market_regime: str = ""        # "trending", "range", "volatile"
    edge_score: float = 0.0        # Scanner edge score at entry
    ai_confidence: float = 0.0     # AI advisor confidence at entry
    confluence_score: float = 0.0  # Multi-indicator score at entry
    exit_reason: str = ""          # "TP1", "SL", "trailing_stop", "EOD"

    # Outcome
    holding_time_minutes: float = 0.0
    max_favorable: float = 0.0     # Max unrealized profit
    max_adverse: float = 0.0       # Max unrealized loss (MAE)
    grade: str = ""                # A/B/C/D/F

    # Tags
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class TradeJournal:
    """Tracks and analyzes all completed trades.

    Usage:
        journal = TradeJournal(data_dir="data")
        journal.record_trade(JournalEntry(...))
        stats = journal.get_performance_report()
    """

    def __init__(self, data_dir: str = "data"):
        self._entries: list[JournalEntry] = []
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._journal_file = self._data_dir / "trade_journal.jsonl"

        # Running stats
        self._peak_equity: float = 0
        self._current_drawdown: float = 0
        self._max_drawdown: float = 0
        self._win_streak: int = 0
        self._lose_streak: int = 0
        self._max_win_streak: int = 0
        self._max_lose_streak: int = 0
        self._cumulative_pnl: float = 0

    # ── Record ────────────────────────────────────────────────────────────────

    def record_trade(self, entry: JournalEntry) -> JournalEntry:
        """Record a completed trade and compute derived fields."""
        # Compute PnL
        if entry.side == "LONG":
            entry.gross_pnl = (entry.exit_price - entry.entry_price) * entry.quantity
        else:
            entry.gross_pnl = (entry.entry_price - entry.exit_price) * entry.quantity
        entry.net_pnl = entry.gross_pnl - entry.fees
        cost = entry.entry_price * entry.quantity
        entry.pnl_percent = (entry.net_pnl / cost * 100) if cost > 0 else 0

        # Slippage
        if entry.expected_entry > 0:
            entry.entry_slippage = abs(entry.entry_price - entry.expected_entry)
        if entry.expected_exit > 0:
            entry.exit_slippage = abs(entry.exit_price - entry.expected_exit)

        # Grade the trade
        entry.grade = self._grade_trade(entry)

        # Update streaks
        if entry.net_pnl > 0:
            self._win_streak += 1
            self._lose_streak = 0
            self._max_win_streak = max(self._max_win_streak, self._win_streak)
        else:
            self._lose_streak += 1
            self._win_streak = 0
            self._max_lose_streak = max(self._max_lose_streak, self._lose_streak)

        # Update equity tracking
        self._cumulative_pnl += entry.net_pnl
        if self._cumulative_pnl > self._peak_equity:
            self._peak_equity = self._cumulative_pnl
        self._current_drawdown = self._peak_equity - self._cumulative_pnl
        self._max_drawdown = max(self._max_drawdown, self._current_drawdown)

        self._entries.append(entry)
        self._persist(entry)

        logger.info(
            "Journal: {} {} {} | PnL=${:.2f} ({:.1f}%) | Grade={} | {}",
            entry.symbol, entry.strategy, entry.side,
            entry.net_pnl, entry.pnl_percent, entry.grade, entry.exit_reason,
        )
        return entry

    # ── Performance Report ────────────────────────────────────────────────────

    def get_performance_report(self, strategy: str = "",
                                period_days: int = 0) -> dict:
        """Comprehensive performance report.

        Optionally filter by strategy and/or recent N days.
        """
        entries = self._filter(strategy, period_days)
        if not entries:
            return {"trades": 0, "sufficient_data": False}

        wins = [e for e in entries if e.net_pnl > 0]
        losses = [e for e in entries if e.net_pnl <= 0]

        pnls = [e.net_pnl for e in entries]
        win_pnls = [e.net_pnl for e in wins]
        loss_pnls = [abs(e.net_pnl) for e in losses]

        total_gross_profit = sum(win_pnls) if win_pnls else 0
        total_gross_loss = sum(loss_pnls) if loss_pnls else 0

        return {
            "trades": len(entries),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(entries), 4),
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(statistics.mean(pnls), 2),
            "median_pnl": round(statistics.median(pnls), 2),
            "std_dev_pnl": round(statistics.stdev(pnls), 2) if len(pnls) > 1 else 0,
            "avg_win": round(statistics.mean(win_pnls), 2) if win_pnls else 0,
            "avg_loss": round(statistics.mean(loss_pnls), 2) if loss_pnls else 0,
            "largest_win": round(max(win_pnls), 2) if win_pnls else 0,
            "largest_loss": round(max(loss_pnls), 2) if loss_pnls else 0,
            "profit_factor": round(total_gross_profit / total_gross_loss, 2) if total_gross_loss > 0 else float("inf"),
            "expectancy": round(statistics.mean(pnls), 2),
            "sharpe_ratio": self._sharpe(pnls),
            "sortino_ratio": self._sortino(pnls),
            "max_drawdown": round(self._max_drawdown, 2),
            "max_win_streak": self._max_win_streak,
            "max_lose_streak": self._max_lose_streak,
            "avg_holding_minutes": round(statistics.mean([e.holding_time_minutes for e in entries]), 1)
                if entries else 0,
            "avg_slippage": round(statistics.mean(
                [e.entry_slippage for e in entries if e.entry_slippage > 0]
            ), 4) if any(e.entry_slippage > 0 for e in entries) else 0,
            "grade_distribution": self._grade_distribution(entries),
        }

    # ── Pattern Analysis ──────────────────────────────────────────────────────

    def time_of_day_analysis(self) -> dict:
        """Analyze performance by hour of day (ET)."""
        hourly: dict[int, list[float]] = defaultdict(list)
        for e in self._entries:
            if e.entry_time:
                try:
                    dt = datetime.fromisoformat(e.entry_time)
                    hour = dt.hour
                    hourly[hour].append(e.net_pnl)
                except (ValueError, TypeError):
                    pass

        result = {}
        for hour in sorted(hourly.keys()):
            pnls = hourly[hour]
            result[f"{hour:02d}:00"] = {
                "trades": len(pnls),
                "total_pnl": round(sum(pnls), 2),
                "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 4) if pnls else 0,
                "avg_pnl": round(statistics.mean(pnls), 2) if pnls else 0,
            }
        return result

    def strategy_comparison(self) -> dict:
        """Compare performance across all strategies."""
        strategies = set(e.strategy for e in self._entries)
        return {s: self.get_performance_report(strategy=s) for s in strategies if s}

    def setup_analysis(self) -> dict:
        """Analyze which setup types are most profitable."""
        setups: dict[str, list[float]] = defaultdict(list)
        for e in self._entries:
            if e.setup_type:
                setups[e.setup_type].append(e.net_pnl)

        result = {}
        for setup, pnls in sorted(setups.items(), key=lambda x: -sum(x[1])):
            wins = [p for p in pnls if p > 0]
            result[setup] = {
                "trades": len(pnls),
                "total_pnl": round(sum(pnls), 2),
                "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0,
                "avg_pnl": round(statistics.mean(pnls), 2) if pnls else 0,
            }
        return result

    def exit_reason_analysis(self) -> dict:
        """Analyze which exit reasons have best outcomes."""
        reasons: dict[str, list[float]] = defaultdict(list)
        for e in self._entries:
            if e.exit_reason:
                reasons[e.exit_reason].append(e.net_pnl)

        result = {}
        for reason, pnls in sorted(reasons.items()):
            result[reason] = {
                "trades": len(pnls),
                "total_pnl": round(sum(pnls), 2),
                "avg_pnl": round(statistics.mean(pnls), 2) if pnls else 0,
            }
        return result

    def regime_analysis(self) -> dict:
        """Performance by market regime."""
        regimes: dict[str, list[float]] = defaultdict(list)
        for e in self._entries:
            regime = e.market_regime or "unknown"
            regimes[regime].append(e.net_pnl)

        result = {}
        for regime, pnls in sorted(regimes.items()):
            wins = [p for p in pnls if p > 0]
            result[regime] = {
                "trades": len(pnls),
                "total_pnl": round(sum(pnls), 2),
                "win_rate": round(len(wins) / len(pnls), 4) if pnls else 0,
            }
        return result

    # ── Execution Quality ─────────────────────────────────────────────────────

    def execution_quality_report(self) -> dict:
        """Detailed execution quality analysis."""
        entries_with_slip = [e for e in self._entries if e.entry_slippage > 0 or e.exit_slippage > 0]
        if not entries_with_slip:
            return {"message": "No slippage data recorded"}

        entry_slips = [e.entry_slippage for e in entries_with_slip if e.entry_slippage > 0]
        exit_slips = [e.exit_slippage for e in entries_with_slip if e.exit_slippage > 0]

        # MAE/MFE analysis
        mae_list = [e.max_adverse for e in self._entries if e.max_adverse > 0]
        mfe_list = [e.max_favorable for e in self._entries if e.max_favorable > 0]

        return {
            "avg_entry_slippage": round(statistics.mean(entry_slips), 4) if entry_slips else 0,
            "avg_exit_slippage": round(statistics.mean(exit_slips), 4) if exit_slips else 0,
            "total_slippage_cost": round(
                sum(e.entry_slippage * e.quantity + e.exit_slippage * e.quantity
                    for e in entries_with_slip), 2),
            "avg_mae": round(statistics.mean(mae_list), 2) if mae_list else 0,
            "avg_mfe": round(statistics.mean(mfe_list), 2) if mfe_list else 0,
            "efficiency": round(
                statistics.mean(
                    [e.net_pnl / (e.max_favorable * e.quantity) for e in self._entries
                     if e.max_favorable > 0 and e.quantity > 0]
                ), 4) if any(e.max_favorable > 0 for e in self._entries) else 0,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _grade_trade(self, entry: JournalEntry) -> str:
        """Grade a trade A-F based on execution quality and edge alignment.

        A = Followed plan perfectly, good profit, low slippage
        B = Good execution, reasonable profit
        C = Acceptable, small profit or small loss
        D = Poor execution or plan deviation
        F = Large loss, high slippage, or no edge
        """
        score = 50  # Start at C

        # PnL contribution (±20)
        if entry.pnl_percent > 1.5:
            score += 20
        elif entry.pnl_percent > 0.5:
            score += 10
        elif entry.pnl_percent > 0:
            score += 5
        elif entry.pnl_percent > -0.5:
            score -= 5
        elif entry.pnl_percent > -1.5:
            score -= 15
        else:
            score -= 20

        # Slippage penalty (±10)
        total_slip = entry.entry_slippage + entry.exit_slippage
        cost = entry.entry_price * entry.quantity
        if cost > 0:
            slip_pct = total_slip * entry.quantity / cost * 100
            if slip_pct < 0.05:
                score += 10
            elif slip_pct < 0.1:
                score += 5
            elif slip_pct > 0.5:
                score -= 10

        # Edge alignment (±10)
        if entry.edge_score >= 70:
            score += 10
        elif entry.edge_score >= 50:
            score += 5
        elif 0 < entry.edge_score < 30:
            score -= 10

        # AI confidence (±10)
        if entry.ai_confidence >= 0.8:
            score += 10
        elif entry.ai_confidence >= 0.6:
            score += 5

        if score >= 80:
            return "A"
        elif score >= 65:
            return "B"
        elif score >= 45:
            return "C"
        elif score >= 30:
            return "D"
        else:
            return "F"

    def _filter(self, strategy: str = "", period_days: int = 0) -> list[JournalEntry]:
        entries = self._entries
        if strategy:
            entries = [e for e in entries if e.strategy == strategy]
        if period_days > 0:
            cutoff = datetime.now(timezone.utc).isoformat()
            # Simple string comparison works for ISO format
            entries = [e for e in entries if e.exit_time >= cutoff]
        return entries

    def _grade_distribution(self, entries: list[JournalEntry]) -> dict:
        dist: dict[str, int] = defaultdict(int)
        for e in entries:
            dist[e.grade] += 1
        return dict(dist)

    def _sharpe(self, pnls: list[float], risk_free: float = 0.0) -> float:
        """Annualized Sharpe Ratio (assuming ~252 trading days)."""
        if len(pnls) < 2:
            return 0.0
        mean_ret = statistics.mean(pnls) - risk_free
        std_ret = statistics.stdev(pnls)
        if std_ret == 0:
            return 0.0
        # Daily Sharpe × sqrt(252) for annualization
        return round(mean_ret / std_ret * (252 ** 0.5), 2)

    def _sortino(self, pnls: list[float], risk_free: float = 0.0) -> float:
        """Sortino Ratio — penalizes only downside deviation."""
        if len(pnls) < 2:
            return 0.0
        mean_ret = statistics.mean(pnls) - risk_free
        downside = [p for p in pnls if p < 0]
        if not downside:
            return float("inf") if mean_ret > 0 else 0.0
        down_dev = (sum(d ** 2 for d in downside) / len(downside)) ** 0.5
        if down_dev == 0:
            return 0.0
        return round(mean_ret / down_dev * (252 ** 0.5), 2)

    def _persist(self, entry: JournalEntry) -> None:
        """Append trade to JSONL journal file."""
        try:
            with open(self._journal_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as exc:
            logger.warning("Failed to persist journal entry: {}", exc)

    def load_history(self) -> int:
        """Load journal from disk. Returns count of entries loaded."""
        if not self._journal_file.exists():
            return 0
        try:
            with open(self._journal_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    entry = JournalEntry(**{
                        k: v for k, v in data.items()
                        if k in JournalEntry.__dataclass_fields__
                    })
                    self._entries.append(entry)
            logger.info("Loaded {} journal entries from {}", len(self._entries), self._journal_file)
            return len(self._entries)
        except Exception as exc:
            logger.warning("Failed to load journal: {}", exc)
            return 0

    def get_full_report(self) -> dict:
        """Generate comprehensive analytics report."""
        return {
            "performance": self.get_performance_report(),
            "by_strategy": self.strategy_comparison(),
            "by_time_of_day": self.time_of_day_analysis(),
            "by_setup": self.setup_analysis(),
            "by_exit_reason": self.exit_reason_analysis(),
            "by_regime": self.regime_analysis(),
            "execution_quality": self.execution_quality_report(),
            "current_streak": {
                "wins": self._win_streak,
                "losses": self._lose_streak,
            },
        }
