"""Tests for Position Sizer (src/risk/position_sizer.py)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.risk.position_sizer import PositionSizer, TradeRecord, SizingResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trade(pnl: float, strategy: str = "momentum",
                symbol: str = "AAPL", side: str = "LONG") -> TradeRecord:
    entry = 150.0
    exit_p = entry + pnl / 10
    return TradeRecord(
        symbol=symbol, strategy=strategy, pnl=pnl,
        entry_price=entry, exit_price=exit_p, quantity=10, side=side,
    )


def _populate_history(sizer: PositionSizer, n_wins: int = 15,
                      n_losses: int = 5, strategy: str = "momentum") -> None:
    """Add n_wins winning trades and n_losses losing trades."""
    for _ in range(n_wins):
        sizer.add_trade_record(_make_trade(50.0, strategy))
    for _ in range(n_losses):
        sizer.add_trade_record(_make_trade(-30.0, strategy))


@pytest.fixture
def sizer():
    return PositionSizer(
        account_size=100_000,
        max_risk_per_trade=0.02,
        max_portfolio_heat=0.06,
        max_position_pct=0.10,
        kelly_fraction=0.5,
        min_trade_history=20,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIZING — FIXED RISK FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════


class TestFixedRiskSizing:
    def test_basic_sizing(self, sizer):
        result = sizer.calculate_size(
            symbol="AAPL", entry_price=150.0, stop_loss=147.0,
            strategy="momentum",
        )
        assert isinstance(result, SizingResult)
        assert result.quantity > 0
        assert result.method == "fixed_risk"
        assert "Not enough trades for Kelly" in result.notes[0]

    def test_zero_risk_distance(self, sizer):
        result = sizer.calculate_size(
            symbol="AAPL", entry_price=150.0, stop_loss=150.0,
        )
        assert result.quantity > 0
        assert any("2% default" in n for n in result.notes)

    def test_to_dict(self, sizer):
        result = sizer.calculate_size(
            symbol="AAPL", entry_price=150.0, stop_loss=147.0,
        )
        d = result.to_dict()
        assert "quantity" in d
        assert "method" in d
        assert "notes" in d


# ═══════════════════════════════════════════════════════════════════════════════
# SIZING — KELLY CRITERION
# ═══════════════════════════════════════════════════════════════════════════════


class TestKellySizing:
    def test_kelly_with_history(self, sizer):
        _populate_history(sizer, n_wins=15, n_losses=5)
        result = sizer.calculate_size(
            symbol="AAPL", entry_price=150.0, stop_loss=147.0,
            strategy="momentum",
        )
        assert result.method == "kelly"
        assert result.kelly_fraction > 0

    def test_kelly_all_wins(self, sizer):
        _populate_history(sizer, n_wins=20, n_losses=0)
        result = sizer.calculate_size(
            symbol="AAPL", entry_price=150.0, stop_loss=147.0,
            strategy="momentum",
        )
        assert result.kelly_fraction > 0

    def test_kelly_all_losses(self, sizer):
        _populate_history(sizer, n_wins=0, n_losses=20)
        result = sizer.calculate_size(
            symbol="AAPL", entry_price=150.0, stop_loss=147.0,
            strategy="momentum",
        )
        assert result.quantity == Decimal("0") or result.kelly_fraction == 0


# ═══════════════════════════════════════════════════════════════════════════════
# ATR ADJUSTMENT
# ═══════════════════════════════════════════════════════════════════════════════


class TestATRAdjustment:
    def test_atr_reduces_size(self, sizer):
        # High ATR should reduce size
        result_no_atr = sizer.calculate_size(
            symbol="AAPL", entry_price=150.0, stop_loss=147.0,
        )
        result_with_atr = sizer.calculate_size(
            symbol="AAPL", entry_price=150.0, stop_loss=147.0, atr=5.0,
        )
        # With high ATR, quantity should potentially be lower
        assert result_with_atr.quantity <= result_no_atr.quantity


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION CAP
# ═══════════════════════════════════════════════════════════════════════════════


class TestPositionCap:
    def test_max_position_pct(self, sizer):
        # Very wide stop → large position request → capped by max_position_pct
        result = sizer.calculate_size(
            symbol="AAPL", entry_price=150.0, stop_loss=149.99,  # tiny risk
        )
        max_notional = sizer.account_size * sizer.max_position_pct
        actual_notional = float(result.quantity) * 150.0
        assert actual_notional <= max_notional + 150  # Allow 1 share rounding


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO HEAT
# ═══════════════════════════════════════════════════════════════════════════════


class TestPortfolioHeat:
    def test_heat_tracking(self, sizer):
        sizer.register_position("AAPL", 100, 150.0, 147.0)
        heat = sizer._current_portfolio_heat()
        assert heat > 0  # 3 * 100 = $300 risk

    def test_heat_limits_sizing(self, sizer):
        # Fill up most of the heat
        sizer.register_position("AAPL", 100, 150.0, 147.0)  # $300 risk
        sizer.register_position("TSLA", 50, 200.0, 190.0)   # $500 risk
        sizer.register_position("NVDA", 20, 500.0, 470.0)   # $600 risk
        sizer.register_position("MSFT", 100, 350.0, 335.0)  # $1500 risk
        sizer.register_position("GOOG", 50, 150.0, 130.0)   # $1000 risk
        sizer.register_position("AMZN", 50, 180.0, 160.0)   # $1000 risk
        # Total heat ≈ $4900 / $100000 = 4.9%, near 6% cap

        result = sizer.calculate_size(
            symbol="META", entry_price=300.0, stop_loss=290.0,
        )
        # Should be constrained by remaining heat
        assert result.portfolio_heat <= sizer.max_portfolio_heat + 0.01

    def test_unregister_position(self, sizer):
        sizer.register_position("AAPL", 100, 150.0, 147.0, sector="tech")
        sizer.unregister_position("AAPL")
        assert "AAPL" not in sizer._open_positions
        assert sizer._current_portfolio_heat() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTOR CONCENTRATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestSectorConcentration:
    def test_sector_blocks_when_full(self, sizer):
        # Register positions filling the sector
        sizer.register_position("AAPL", 100, 150.0, 147.0, sector="tech")
        sizer.register_position("MSFT", 100, 350.0, 340.0, sector="tech")
        # tech notional = 15000 + 35000 = 50000 = 50% of account → over 25% limit

        result = sizer.calculate_size(
            symbol="NVDA", entry_price=500.0, stop_loss=490.0,
            sector="tech",
        )
        assert result.quantity == Decimal("0")


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME MULTIPLIER
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegimeMultiplier:
    def test_regime_scales_size(self, sizer):
        sizer.update_regime_multiplier(0.5)
        result_half = sizer.calculate_size(
            symbol="AAPL", entry_price=150.0, stop_loss=147.0,
        )
        sizer.update_regime_multiplier(1.0)
        result_full = sizer.calculate_size(
            symbol="AAPL", entry_price=150.0, stop_loss=147.0,
        )
        assert result_half.quantity <= result_full.quantity

    def test_regime_multiplier_clamped(self, sizer):
        sizer.update_regime_multiplier(5.0)
        assert sizer.regime_multiplier == 2.0
        sizer.update_regime_multiplier(0.01)
        assert sizer.regime_multiplier == 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# RISK OF RUIN
# ═══════════════════════════════════════════════════════════════════════════════


class TestRiskOfRuin:
    def test_positive_edge(self, sizer):
        ror = sizer.risk_of_ruin(win_rate=0.6, win_loss_ratio=1.5, risk_per_trade=0.02)
        assert 0 <= ror <= 1
        assert ror < 0.5  # Positive edge → low ruin

    def test_negative_edge(self, sizer):
        ror = sizer.risk_of_ruin(win_rate=0.3, win_loss_ratio=0.5, risk_per_trade=0.02)
        assert ror == 1.0  # Negative edge → certain ruin

    def test_ror_from_history(self, sizer):
        _populate_history(sizer, 8, 2)
        ror = sizer.risk_of_ruin()
        assert 0 <= ror <= 1

    def test_ror_insufficient_history(self, sizer):
        sizer.add_trade_record(_make_trade(10))
        ror = sizer.risk_of_ruin()
        assert ror == 1.0  # Unknown = high risk


# ═══════════════════════════════════════════════════════════════════════════════
# EXPECTANCY
# ═══════════════════════════════════════════════════════════════════════════════


class TestExpectancy:
    def test_positive_expectancy(self, sizer):
        _populate_history(sizer, 15, 5)
        exp = sizer.expectancy("momentum")
        assert exp["expectancy"] > 0
        assert exp["win_rate"] == 0.75
        assert exp["trades"] == 20

    def test_empty_expectancy(self, sizer):
        exp = sizer.expectancy("unknown")
        assert exp["expectancy"] == 0
        assert exp["sufficient_data"] is False

    def test_per_strategy_stats(self, sizer):
        _populate_history(sizer, 15, 5, "momentum")
        _populate_history(sizer, 10, 10, "vwap")
        stats = sizer._per_strategy_stats()
        assert "momentum" in stats
        assert "vwap" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


class TestLifecycle:
    def test_update_account_size(self, sizer):
        sizer.update_account_size(200_000)
        assert sizer.account_size == 200_000

    def test_reset_daily(self, sizer):
        sizer.register_position("AAPL", 10, 150.0, 147.0, sector="tech")
        sizer.reset_daily()
        assert len(sizer._open_positions) == 0
        assert len(sizer._sector_exposure) == 0

    def test_trade_history_cap(self, sizer):
        for i in range(600):
            sizer.add_trade_record(_make_trade(10.0))
        assert len(sizer._trade_history) == 500

    def test_get_stats(self, sizer):
        _populate_history(sizer, 10, 5)
        sizer.register_position("AAPL", 10, 150.0, 147.0)
        stats = sizer.get_stats()
        assert stats["total_trades"] == 15
        assert stats["open_positions"] == 1
        assert stats["portfolio_heat"] >= 0
