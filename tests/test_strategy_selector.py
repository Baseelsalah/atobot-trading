"""Tests for Adaptive Strategy Selector (src/strategies/strategy_selector.py)."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest

from src.strategies.strategy_selector import AdaptiveStrategySelector, StrategyState


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def selector():
    sel = AdaptiveStrategySelector(
        max_consecutive_losses=3,
        cooldown_minutes=15,
        min_weight_threshold=0.2,
    )
    sel.register_strategy("momentum")
    sel.register_strategy("vwap")
    sel.register_strategy("orb")
    return sel


@pytest.fixture
def mock_regime():
    """Mock regime detector with controllable outputs."""
    rd = MagicMock()
    rd.get_strategy_weights.return_value = {
        "momentum": 1.2,
        "vwap": 0.8,
        "orb": 0.5,
    }
    rd.get_size_multiplier.return_value = 0.9
    regime = MagicMock()
    regime.direction = "both"
    regime.preferred_direction = "both"
    regime.risk_on = True
    regime.trend = "trending_bullish"
    rd.get_current_regime.return_value = regime
    return rd


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistration:
    def test_register_strategy(self, selector):
        assert "momentum" in selector._strategies
        assert "vwap" in selector._strategies
        assert "orb" in selector._strategies

    def test_register_duplicate(self, selector):
        selector.register_strategy("momentum")
        # Should not create duplicate
        assert len([s for s in selector._strategies if s == "momentum"]) == 1

    def test_enable_disable(self, selector):
        selector.disable_strategy("momentum")
        assert not selector._strategies["momentum"].enabled
        selector.enable_strategy("momentum")
        assert selector._strategies["momentum"].enabled


# ═══════════════════════════════════════════════════════════════════════════════
# SHOULD_TRADE
# ═══════════════════════════════════════════════════════════════════════════════


class TestShouldTrade:
    def test_allowed_by_default(self, selector):
        allowed, reason = selector.should_trade("momentum")
        assert allowed is True
        assert reason == ""

    def test_unknown_strategy_allowed(self, selector):
        allowed, _ = selector.should_trade("unknown_strat")
        assert allowed is True

    def test_disabled_strategy_blocked(self, selector):
        selector.disable_strategy("vwap")
        allowed, reason = selector.should_trade("vwap")
        assert allowed is False
        assert "disabled" in reason

    def test_low_weight_blocked(self, selector):
        selector._strategies["orb"].weight = 0.05
        allowed, reason = selector.should_trade("orb")
        assert allowed is False

    def test_weight_below_threshold(self, selector):
        selector._strategies["orb"].weight = 0.15  # Below 0.2 threshold
        allowed, reason = selector.should_trade("orb")
        assert allowed is False
        assert "weight" in reason

    def test_direction_bias_blocks_buy(self, selector):
        selector._direction_bias = "short"
        allowed, reason = selector.should_trade("momentum", side="BUY")
        assert allowed is False
        assert "short-only" in reason

    def test_direction_bias_blocks_sell(self, selector):
        selector._direction_bias = "long"
        allowed, reason = selector.should_trade("momentum", side="SELL")
        assert allowed is False
        assert "long-only" in reason

    def test_direction_none_blocks_all(self, selector):
        selector._direction_bias = "none"
        allowed, reason = selector.should_trade("momentum")
        assert allowed is False
        assert "choppy" in reason

    def test_risk_off_blocks_low_weight(self, selector):
        selector._risk_on = False
        selector._strategies["momentum"].weight = 0.5
        allowed, reason = selector.should_trade("momentum")
        assert allowed is False
        assert "Risk-off" in reason

    def test_risk_off_allows_high_weight(self, selector):
        selector._risk_on = False
        selector._strategies["momentum"].weight = 1.0
        allowed, _ = selector.should_trade("momentum")
        assert allowed is True


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME UPDATES
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegimeUpdates:
    def test_update_from_regime(self, selector, mock_regime):
        selector.update_from_regime(mock_regime)
        assert selector._strategies["momentum"].weight == 1.2
        assert selector._strategies["vwap"].weight == 0.8
        assert selector._size_multiplier == 0.9
        assert selector._direction_bias == "both"

    def test_update_weights_manual(self, selector):
        selector.update_weights_manual({"momentum": 0.3, "vwap": 1.5})
        assert selector._strategies["momentum"].weight == 0.3
        assert selector._strategies["vwap"].weight == 1.5


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE FEEDBACK / COOLDOWN
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformanceFeedback:
    def test_record_win(self, selector):
        selector.record_trade_result("momentum", 50.0)
        state = selector._strategies["momentum"]
        assert state.recent_wins == 1
        assert state.total_pnl == 50.0
        assert state.consecutive_losses == 0

    def test_record_loss(self, selector):
        selector.record_trade_result("momentum", -30.0)
        state = selector._strategies["momentum"]
        assert state.recent_losses == 1
        assert state.consecutive_losses == 1

    def test_auto_cooldown_on_streak(self, selector):
        for _ in range(3):  # max_consecutive_losses=3
            selector.record_trade_result("momentum", -20.0)
        state = selector._strategies["momentum"]
        assert state.cooldown_until is not None
        assert state.cooldown_until > datetime.now(timezone.utc)

    def test_cooldown_blocks_trading(self, selector):
        for _ in range(3):
            selector.record_trade_result("momentum", -20.0)
        allowed, reason = selector.should_trade("momentum")
        assert allowed is False
        assert "cooldown" in reason

    def test_cooldown_expires(self, selector):
        for _ in range(3):
            selector.record_trade_result("momentum", -20.0)
        # Manually expire cooldown
        selector._strategies["momentum"].cooldown_until = datetime.now(timezone.utc) - timedelta(minutes=1)
        allowed, _ = selector.should_trade("momentum")
        assert allowed is True

    def test_rolling_window(self, selector):
        # Add lots of trades to trigger rolling halving
        for _ in range(50):
            selector.record_trade_result("momentum", 10.0)
        state = selector._strategies["momentum"]
        # After halving, counters should be reduced
        total = state.recent_wins + state.recent_losses
        assert total < 50


# ═══════════════════════════════════════════════════════════════════════════════
# SIZE MULTIPLIER
# ═══════════════════════════════════════════════════════════════════════════════


class TestSizeMultiplier:
    def test_get_size_multiplier(self, selector, mock_regime):
        selector.update_from_regime(mock_regime)
        assert selector.get_size_multiplier() == 0.9

    def test_apply_size_multiplier_no_strategy(self, selector):
        selector._size_multiplier = 0.8
        qty = selector.apply_size_multiplier(100)
        assert qty == pytest.approx(80.0)

    def test_apply_size_multiplier_with_strategy(self, selector):
        selector._size_multiplier = 1.0
        selector._strategies["momentum"].weight = 0.5
        qty = selector.apply_size_multiplier(100, "momentum")
        assert qty == pytest.approx(50.0)

    def test_apply_size_multiplier_floors_at_1(self, selector):
        selector._size_multiplier = 0.001
        qty = selector.apply_size_multiplier(1)
        assert qty >= 1.0

    def test_get_strategy_weight(self, selector):
        selector._strategies["momentum"].weight = 1.3
        assert selector.get_strategy_weight("momentum") == 1.3
        assert selector.get_strategy_weight("unknown") == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY / STATUS
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatus:
    def test_get_active_strategies(self, selector):
        active = selector.get_active_strategies()
        assert "momentum" in active
        selector.disable_strategy("momentum")
        active = selector.get_active_strategies()
        assert "momentum" not in active

    def test_get_status(self, selector):
        selector.record_trade_result("momentum", 50.0)
        status = selector.get_status()
        assert "regime" in status
        assert "size_multiplier" in status
        assert "strategies" in status
        assert "momentum" in status["strategies"]
        s = status["strategies"]["momentum"]
        assert s["active"] is True
        assert "weight" in s
        assert "win_rate" in s


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY STATE
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategyState:
    def test_is_active_default(self):
        state = StrategyState(name="test")
        assert state.is_active is True

    def test_is_active_disabled(self):
        state = StrategyState(name="test", enabled=False)
        assert state.is_active is False

    def test_is_active_zero_weight(self):
        state = StrategyState(name="test", weight=0.0)
        assert state.is_active is False

    def test_is_active_in_cooldown(self):
        state = StrategyState(
            name="test",
            cooldown_until=datetime.now(timezone.utc) + timedelta(minutes=10),
        )
        assert state.is_active is False

    def test_win_rate(self):
        state = StrategyState(name="test", recent_wins=7, recent_losses=3)
        assert state.win_rate == pytest.approx(0.7)

    def test_win_rate_no_trades(self):
        state = StrategyState(name="test")
        assert state.win_rate == 0.5
