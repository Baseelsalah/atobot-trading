"""Tests for Trade Journal (src/analytics/trade_journal.py)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.analytics.trade_journal import TradeJournal, JournalEntry


# ── Helpers ───────────────────────────────────────────────────────────────────


def _entry(pnl_direction: str = "win", strategy: str = "momentum",
           symbol: str = "AAPL", exit_reason: str = "TP1",
           setup_type: str = "RSI_bounce",
           regime: str = "trending",
           hour: int = 10) -> JournalEntry:
    """Create a journal entry. pnl_direction is 'win' or 'loss'."""
    if pnl_direction == "win":
        entry_p, exit_p = 150.0, 153.0
    else:
        entry_p, exit_p = 150.0, 148.0

    return JournalEntry(
        symbol=symbol, strategy=strategy, side="LONG",
        entry_price=entry_p, exit_price=exit_p, quantity=10,
        entry_time=f"2026-01-15T{hour:02d}:30:00+00:00",
        exit_time=f"2026-01-15T{hour + 1:02d}:00:00+00:00",
        expected_entry=entry_p, expected_exit=exit_p + 0.05,
        fees=0.50, setup_type=setup_type, market_regime=regime,
        edge_score=75.0, ai_confidence=0.85, confluence_score=70.0,
        exit_reason=exit_reason, holding_time_minutes=30,
        max_favorable=3.5, max_adverse=0.5,
    )


@pytest.fixture
def journal(tmp_path):
    """Journal using a temp directory for persistence."""
    return TradeJournal(data_dir=str(tmp_path))


# ═══════════════════════════════════════════════════════════════════════════════
# RECORD TRADE
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecordTrade:
    def test_record_winning_trade(self, journal):
        e = journal.record_trade(_entry("win"))
        assert e.gross_pnl > 0
        assert e.net_pnl > 0
        assert e.pnl_percent > 0
        assert e.grade in ("A", "B", "C", "D", "F")

    def test_record_losing_trade(self, journal):
        e = journal.record_trade(_entry("loss"))
        assert e.gross_pnl < 0
        assert e.net_pnl < 0
        assert e.pnl_percent < 0

    def test_pnl_calculation_long(self, journal):
        entry = JournalEntry(
            symbol="AAPL", strategy="test", side="LONG",
            entry_price=100.0, exit_price=105.0, quantity=10, fees=1.0,
        )
        e = journal.record_trade(entry)
        assert e.gross_pnl == 50.0  # (105-100)*10
        assert e.net_pnl == 49.0    # 50 - 1 fee

    def test_pnl_calculation_short(self, journal):
        entry = JournalEntry(
            symbol="AAPL", strategy="test", side="SHORT",
            entry_price=100.0, exit_price=95.0, quantity=10, fees=1.0,
        )
        e = journal.record_trade(entry)
        assert e.gross_pnl == 50.0  # (100-95)*10
        assert e.net_pnl == 49.0

    def test_slippage_computed(self, journal):
        entry = JournalEntry(
            symbol="AAPL", strategy="test", side="LONG",
            entry_price=100.05, exit_price=105.0, quantity=10,
            expected_entry=100.0, expected_exit=105.10,
        )
        e = journal.record_trade(entry)
        assert e.entry_slippage == pytest.approx(0.05, abs=0.001)
        assert e.exit_slippage == pytest.approx(0.10, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# STREAKS & DRAWDOWN
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreaks:
    def test_win_streak(self, journal):
        for _ in range(5):
            journal.record_trade(_entry("win"))
        assert journal._win_streak == 5
        assert journal._max_win_streak == 5

    def test_lose_streak(self, journal):
        for _ in range(3):
            journal.record_trade(_entry("loss"))
        assert journal._lose_streak == 3
        assert journal._max_lose_streak == 3

    def test_streak_reset(self, journal):
        for _ in range(3):
            journal.record_trade(_entry("win"))
        journal.record_trade(_entry("loss"))
        assert journal._win_streak == 0
        assert journal._lose_streak == 1
        assert journal._max_win_streak == 3

    def test_drawdown_tracking(self, journal):
        # Win to build equity
        for _ in range(5):
            journal.record_trade(_entry("win"))
        peak = journal._peak_equity
        assert peak > 0

        # Lose to create drawdown
        for _ in range(3):
            journal.record_trade(_entry("loss"))
        assert journal._current_drawdown > 0
        assert journal._max_drawdown > 0


# ═══════════════════════════════════════════════════════════════════════════════
# GRADING
# ═══════════════════════════════════════════════════════════════════════════════


class TestGrading:
    def test_high_grade_for_good_trade(self, journal):
        entry = JournalEntry(
            symbol="AAPL", strategy="test", side="LONG",
            entry_price=100.0, exit_price=102.0, quantity=10,
            expected_entry=100.0, edge_score=80, ai_confidence=0.9,
        )
        e = journal.record_trade(entry)
        assert e.grade in ("A", "B")

    def test_low_grade_for_bad_trade(self, journal):
        entry = JournalEntry(
            symbol="AAPL", strategy="test", side="LONG",
            entry_price=100.0, exit_price=97.0, quantity=10,
            expected_entry=99.0, edge_score=10, ai_confidence=0.2,
        )
        e = journal.record_trade(entry)
        assert e.grade in ("D", "F")


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformanceReport:
    def test_empty_report(self, journal):
        report = journal.get_performance_report()
        assert report["trades"] == 0
        assert report["sufficient_data"] is False

    def test_basic_report(self, journal):
        for _ in range(7):
            journal.record_trade(_entry("win"))
        for _ in range(3):
            journal.record_trade(_entry("loss"))

        report = journal.get_performance_report()
        assert report["trades"] == 10
        assert report["wins"] == 7
        assert report["losses"] == 3
        assert report["win_rate"] == 0.7
        assert report["total_pnl"] != 0
        assert "profit_factor" in report
        assert "sharpe_ratio" in report
        assert "sortino_ratio" in report
        assert "grade_distribution" in report

    def test_report_by_strategy(self, journal):
        for _ in range(5):
            journal.record_trade(_entry("win", strategy="momentum"))
        for _ in range(5):
            journal.record_trade(_entry("loss", strategy="vwap"))

        report = journal.get_performance_report(strategy="momentum")
        assert report["trades"] == 5
        assert report["win_rate"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS METHODS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnalysis:
    def _populate(self, journal):
        journal.record_trade(_entry("win", strategy="momentum", hour=9, setup_type="RSI_bounce", exit_reason="TP1"))
        journal.record_trade(_entry("win", strategy="momentum", hour=10, setup_type="RSI_bounce", exit_reason="TP2"))
        journal.record_trade(_entry("loss", strategy="vwap", hour=11, setup_type="VWAP_revert", exit_reason="SL"))
        journal.record_trade(_entry("win", strategy="orb", hour=9, setup_type="ORB_break", exit_reason="trailing"))
        journal.record_trade(_entry("loss", strategy="momentum", hour=14, setup_type="RSI_bounce", exit_reason="SL", regime="volatile"))

    def test_time_of_day_analysis(self, journal):
        self._populate(journal)
        result = journal.time_of_day_analysis()
        assert "09:00" in result
        assert result["09:00"]["trades"] == 2

    def test_strategy_comparison(self, journal):
        self._populate(journal)
        result = journal.strategy_comparison()
        assert "momentum" in result
        assert "vwap" in result
        assert "orb" in result

    def test_setup_analysis(self, journal):
        self._populate(journal)
        result = journal.setup_analysis()
        assert "RSI_bounce" in result
        assert result["RSI_bounce"]["trades"] == 3

    def test_exit_reason_analysis(self, journal):
        self._populate(journal)
        result = journal.exit_reason_analysis()
        assert "TP1" in result
        assert "SL" in result

    def test_regime_analysis(self, journal):
        self._populate(journal)
        result = journal.regime_analysis()
        assert "trending" in result

    def test_execution_quality_report(self, journal):
        self._populate(journal)
        result = journal.execution_quality_report()
        # All entries have slippage from expected_exit
        assert "avg_entry_slippage" in result or "message" in result


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════


class TestPersistence:
    def test_persist_and_load(self, tmp_path):
        journal = TradeJournal(data_dir=str(tmp_path))
        journal.record_trade(_entry("win"))
        journal.record_trade(_entry("loss"))

        # Create new journal and load
        journal2 = TradeJournal(data_dir=str(tmp_path))
        count = journal2.load_history()
        assert count == 2

    def test_journal_file_is_jsonl(self, tmp_path):
        journal = TradeJournal(data_dir=str(tmp_path))
        journal.record_trade(_entry("win"))
        path = tmp_path / "trade_journal.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["symbol"] == "AAPL"

    def test_to_dict(self):
        e = _entry("win")
        d = e.to_dict()
        assert "symbol" in d
        assert "strategy" in d


# ═══════════════════════════════════════════════════════════════════════════════
# FULL REPORT
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullReport:
    def test_full_report(self, journal):
        for _ in range(5):
            journal.record_trade(_entry("win"))
        journal.record_trade(_entry("loss"))

        report = journal.get_full_report()
        assert "performance" in report
        assert "by_strategy" in report
        assert "by_time_of_day" in report
        assert "current_streak" in report
