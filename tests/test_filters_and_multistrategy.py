"""Tests for new features: multi-strategy, trend filter, time filter, trailing stop."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.config.settings import Settings
from src.core.bot import AtoBot
from src.models.order import Order, OrderSide, OrderType
from src.models.position import Position
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy
from src.strategies.vwap_strategy import VWAPScalpStrategy


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def vwap_strategy(mock_settings: Settings, mock_exchange_client: AsyncMock) -> VWAPScalpStrategy:
    """Create a VWAPScalpStrategy with mocked deps and filters disabled."""
    mock_settings.DEFAULT_STRATEGY = "vwap_scalp"
    rm = RiskManager(mock_settings)
    return VWAPScalpStrategy(mock_exchange_client, rm, mock_settings)


@pytest.fixture
def vwap_strategy_with_trailing(
    mock_settings: Settings, mock_exchange_client: AsyncMock
) -> VWAPScalpStrategy:
    """Create a VWAPScalpStrategy with trailing stop enabled."""
    mock_settings.DEFAULT_STRATEGY = "vwap_scalp"
    mock_settings.TRAILING_STOP_ENABLED = True
    mock_settings.TRAILING_STOP_ACTIVATION_PCT = 0.5
    mock_settings.TRAILING_STOP_DISTANCE_PCT = 0.3
    rm = RiskManager(mock_settings)
    return VWAPScalpStrategy(mock_exchange_client, rm, mock_settings)


@pytest.fixture
def vwap_strategy_with_trend(
    mock_settings: Settings, mock_exchange_client: AsyncMock
) -> VWAPScalpStrategy:
    """Create a VWAPScalpStrategy with EMA trend filter enabled."""
    mock_settings.DEFAULT_STRATEGY = "vwap_scalp"
    mock_settings.TREND_FILTER_ENABLED = True
    mock_settings.TREND_FILTER_EMA_PERIOD = 5
    mock_settings.TREND_FILTER_TIMEFRAME = "15m"
    rm = RiskManager(mock_settings)
    return VWAPScalpStrategy(mock_exchange_client, rm, mock_settings)


@pytest.fixture
def vwap_strategy_with_midday(
    mock_settings: Settings, mock_exchange_client: AsyncMock
) -> VWAPScalpStrategy:
    """Create a VWAPScalpStrategy with midday filter enabled."""
    mock_settings.DEFAULT_STRATEGY = "vwap_scalp"
    mock_settings.AVOID_MIDDAY = True
    mock_settings.MIDDAY_START_HOUR = 12
    mock_settings.MIDDAY_END_HOUR = 14
    rm = RiskManager(mock_settings)
    return VWAPScalpStrategy(mock_exchange_client, rm, mock_settings)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Strategy Configuration Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMultiStrategySettings:
    """Test STRATEGIES setting parsing and validation."""

    def test_strategies_from_json_list(self) -> None:
        s = Settings(
            ALPACA_API_KEY="k",
            ALPACA_API_SECRET="s",
            STRATEGIES='["vwap_scalp", "ema_pullback"]',
            NOTIFICATIONS_ENABLED=False,
        )
        assert s.STRATEGIES == ["vwap_scalp", "ema_pullback"]

    def test_strategies_from_comma_string(self) -> None:
        s = Settings(
            ALPACA_API_KEY="k",
            ALPACA_API_SECRET="s",
            STRATEGIES="vwap_scalp, ema_pullback",
            NOTIFICATIONS_ENABLED=False,
        )
        assert s.STRATEGIES == ["vwap_scalp", "ema_pullback"]

    def test_strategies_empty_falls_back_to_default(self) -> None:
        s = Settings(
            ALPACA_API_KEY="k",
            ALPACA_API_SECRET="s",
            STRATEGIES="",
            DEFAULT_STRATEGY="vwap_scalp",
            NOTIFICATIONS_ENABLED=False,
        )
        assert s.STRATEGIES == ["vwap_scalp"]

    def test_strategies_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            Settings(
                ALPACA_API_KEY="k",
                ALPACA_API_SECRET="s",
                STRATEGIES='["bad_strategy"]',
                NOTIFICATIONS_ENABLED=False,
            )

    def test_strategies_all_three(self) -> None:
        s = Settings(
            ALPACA_API_KEY="k",
            ALPACA_API_SECRET="s",
            STRATEGIES='["vwap_scalp", "ema_pullback", "momentum"]',
            NOTIFICATIONS_ENABLED=False,
        )
        assert len(s.STRATEGIES) == 3
        assert "vwap_scalp" in s.STRATEGIES
        assert "ema_pullback" in s.STRATEGIES
        assert "momentum" in s.STRATEGIES


class TestMultiStrategyBot:
    """Test bot creates multiple strategy instances."""

    @pytest.mark.asyncio
    async def test_bot_strategies_empty_on_init(self, mock_settings: Settings) -> None:
        bot = AtoBot(mock_settings)
        assert bot.strategies == []

    @pytest.mark.asyncio
    async def test_bot_creates_strategies_from_settings(
        self, mock_settings: Settings
    ) -> None:
        """_create_strategies should populate strategies list."""
        mock_settings.STRATEGIES = ["vwap_scalp", "ema_pullback"]
        bot = AtoBot(mock_settings)
        # Mock exchange and risk_manager to avoid real connection
        bot.exchange = AsyncMock()
        bot.risk_manager = RiskManager(mock_settings)
        bot.strategies = bot._create_strategies()
        assert len(bot.strategies) == 2
        names = [s.name for s in bot.strategies]
        assert "vwap_scalp" in names
        assert "ema_pullback" in names

    @pytest.mark.asyncio
    async def test_bot_single_strategy_from_default(
        self, mock_settings: Settings
    ) -> None:
        """When STRATEGIES is just  [DEFAULT_STRATEGY], create one."""
        mock_settings.STRATEGIES = ["momentum"]
        bot = AtoBot(mock_settings)
        bot.exchange = AsyncMock()
        bot.risk_manager = RiskManager(mock_settings)
        bot.strategies = bot._create_strategies()
        assert len(bot.strategies) == 1
        assert bot.strategies[0].name == "momentum"


# ══════════════════════════════════════════════════════════════════════════════
# EMA Trend Filter Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestTrendFilter:
    """Test _passes_trend_filter method."""

    @pytest.mark.asyncio
    async def test_trend_filter_disabled_always_passes(
        self, vwap_strategy: VWAPScalpStrategy
    ) -> None:
        """When TREND_FILTER_ENABLED=False, filter always returns True."""
        assert vwap_strategy.settings.TREND_FILTER_ENABLED is False
        result = await vwap_strategy._passes_trend_filter("AAPL", Decimal("100"))
        assert result is True

    @pytest.mark.asyncio
    async def test_trend_filter_passes_when_price_above_ema(
        self, vwap_strategy_with_trend: VWAPScalpStrategy
    ) -> None:
        """Price above EMA → filter passes."""
        # Mock get_klines to return ascending prices (EMA will be below current)
        bars = [
            {
                "timestamp": 1700000000000 + i * 60000,
                "open": Decimal(str(180 + i)),
                "high": Decimal(str(181 + i)),
                "low": Decimal(str(179 + i)),
                "close": Decimal(str(180 + i)),
                "volume": Decimal("10000"),
            }
            for i in range(15)
        ]
        vwap_strategy_with_trend.exchange.get_klines.return_value = bars
        # Current price is 200, EMA of ascending 180..194 will be well below 200
        result = await vwap_strategy_with_trend._passes_trend_filter(
            "AAPL", Decimal("200")
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_trend_filter_blocks_when_price_below_ema(
        self, vwap_strategy_with_trend: VWAPScalpStrategy
    ) -> None:
        """Price below EMA → filter blocks."""
        # Mock descending prices that will have a high EMA
        bars = [
            {
                "timestamp": 1700000000000 + i * 60000,
                "open": Decimal(str(200 - i)),
                "high": Decimal(str(201 - i)),
                "low": Decimal(str(199 - i)),
                "close": Decimal(str(200 - i)),
                "volume": Decimal("10000"),
            }
            for i in range(15)
        ]
        vwap_strategy_with_trend.exchange.get_klines.return_value = bars
        # Current price is 170, EMA of descending 200..186 will be above 170
        result = await vwap_strategy_with_trend._passes_trend_filter(
            "AAPL", Decimal("170")
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_trend_filter_allows_on_error(
        self, vwap_strategy_with_trend: VWAPScalpStrategy
    ) -> None:
        """On API error, trend filter should not block (fail-open)."""
        vwap_strategy_with_trend.exchange.get_klines.side_effect = Exception("API down")
        result = await vwap_strategy_with_trend._passes_trend_filter(
            "AAPL", Decimal("185")
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_trend_filter_allows_on_insufficient_data(
        self, vwap_strategy_with_trend: VWAPScalpStrategy
    ) -> None:
        """With too few bars, trend filter should not block."""
        vwap_strategy_with_trend.exchange.get_klines.return_value = [
            {"timestamp": 1, "open": 1, "high": 2, "low": 0, "close": 1, "volume": 100}
        ]
        result = await vwap_strategy_with_trend._passes_trend_filter(
            "AAPL", Decimal("185")
        )
        assert result is True


# ══════════════════════════════════════════════════════════════════════════════
# Time-of-Day (Midday) Filter Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestTimeFilter:
    """Test _passes_time_filter method."""

    def test_time_filter_disabled_always_passes(
        self, vwap_strategy: VWAPScalpStrategy
    ) -> None:
        """When AVOID_MIDDAY=False, filter always returns True."""
        assert vwap_strategy.settings.AVOID_MIDDAY is False
        assert vwap_strategy._passes_time_filter() is True

    @patch("src.strategies.base_strategy.datetime")
    def test_time_filter_blocks_during_midday(
        self, mock_dt: MagicMock, vwap_strategy_with_midday: VWAPScalpStrategy
    ) -> None:
        """At 12:30 ET (midday), filter should block."""
        from zoneinfo import ZoneInfo

        fake_now = datetime(2024, 6, 15, 12, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        mock_dt.now.return_value = fake_now
        result = vwap_strategy_with_midday._passes_time_filter()
        assert result is False

    @patch("src.strategies.base_strategy.datetime")
    def test_time_filter_passes_morning(
        self, mock_dt: MagicMock, vwap_strategy_with_midday: VWAPScalpStrategy
    ) -> None:
        """At 10:00 ET (morning), filter should pass."""
        from zoneinfo import ZoneInfo

        fake_now = datetime(2024, 6, 15, 10, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        mock_dt.now.return_value = fake_now
        result = vwap_strategy_with_midday._passes_time_filter()
        assert result is True

    @patch("src.strategies.base_strategy.datetime")
    def test_time_filter_passes_afternoon(
        self, mock_dt: MagicMock, vwap_strategy_with_midday: VWAPScalpStrategy
    ) -> None:
        """At 14:00 ET (2 PM, end boundary), filter should pass."""
        from zoneinfo import ZoneInfo

        fake_now = datetime(2024, 6, 15, 14, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        mock_dt.now.return_value = fake_now
        result = vwap_strategy_with_midday._passes_time_filter()
        assert result is True

    @patch("src.strategies.base_strategy.datetime")
    def test_time_filter_blocks_at_13(
        self, mock_dt: MagicMock, vwap_strategy_with_midday: VWAPScalpStrategy
    ) -> None:
        """At 13:00 ET, filter should block."""
        from zoneinfo import ZoneInfo

        fake_now = datetime(2024, 6, 15, 13, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        mock_dt.now.return_value = fake_now
        result = vwap_strategy_with_midday._passes_time_filter()
        assert result is False


# ══════════════════════════════════════════════════════════════════════════════
# Trailing Stop Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestTrailingStop:
    """Test _check_trailing_stop method."""

    def test_trailing_stop_disabled_returns_false(
        self, vwap_strategy: VWAPScalpStrategy
    ) -> None:
        """When disabled, trailing stop never triggers."""
        pos = Position(
            symbol="AAPL",
            side="LONG",
            entry_price=Decimal("185.00"),
            current_price=Decimal("200.00"),
            quantity=Decimal("3"),
            strategy="vwap_scalp",
        )
        result = vwap_strategy._check_trailing_stop("AAPL", pos, Decimal("180.00"))
        assert result is False

    def test_trailing_stop_not_activated_below_threshold(
        self, vwap_strategy_with_trailing: VWAPScalpStrategy
    ) -> None:
        """Trailing stop doesn't activate until profit exceeds activation %."""
        pos = Position(
            symbol="AAPL",
            side="LONG",
            entry_price=Decimal("100.00"),
            current_price=Decimal("100.30"),
            quantity=Decimal("10"),
            strategy="vwap_scalp",
        )
        # Price at 100.30 = 0.3% profit, below 0.5% activation
        result = vwap_strategy_with_trailing._check_trailing_stop(
            "AAPL", pos, Decimal("100.30")
        )
        assert result is False

    def test_trailing_stop_activated_but_not_triggered(
        self, vwap_strategy_with_trailing: VWAPScalpStrategy
    ) -> None:
        """After activation, price still above trail level → no trigger."""
        pos = Position(
            symbol="AAPL",
            side="LONG",
            entry_price=Decimal("100.00"),
            current_price=Decimal("100.60"),
            quantity=Decimal("10"),
            strategy="vwap_scalp",
        )
        # Price at 100.60 = 0.6% profit, above 0.5% activation
        # Trail stop = 100.60 * (1 - 0.003) = 100.2982
        # Current price 100.50 > 100.2982 → NOT triggered
        result = vwap_strategy_with_trailing._check_trailing_stop(
            "AAPL", pos, Decimal("100.60")
        )
        assert result is False
        # Price slightly below high but above trail
        result = vwap_strategy_with_trailing._check_trailing_stop(
            "AAPL", pos, Decimal("100.50")
        )
        assert result is False

    def test_trailing_stop_triggered(
        self, vwap_strategy_with_trailing: VWAPScalpStrategy
    ) -> None:
        """Price rises past activation, then falls past trail distance → trigger."""
        pos = Position(
            symbol="AAPL",
            side="LONG",
            entry_price=Decimal("100.00"),
            current_price=Decimal("101.00"),
            quantity=Decimal("10"),
            strategy="vwap_scalp",
        )
        # First update: track high at 101.00 (1.0% profit > 0.5% activation)
        result = vwap_strategy_with_trailing._check_trailing_stop(
            "AAPL", pos, Decimal("101.00")
        )
        assert result is False  # Not triggered, just activated
        assert vwap_strategy_with_trailing._trailing_highs["AAPL"] == Decimal("101.00")

        # Second: price drops to 100.65 → trail = 101 * (1-0.003) = 100.697 → 100.65 < 100.697 → TRIGGERED
        result = vwap_strategy_with_trailing._check_trailing_stop(
            "AAPL", pos, Decimal("100.65")
        )
        assert result is True

    def test_trailing_stop_tracks_highest_price(
        self, vwap_strategy_with_trailing: VWAPScalpStrategy
    ) -> None:
        """Trailing high updates as price rises."""
        pos = Position(
            symbol="AAPL",
            side="LONG",
            entry_price=Decimal("100.00"),
            current_price=Decimal("101.00"),
            quantity=Decimal("10"),
            strategy="vwap_scalp",
        )
        vwap_strategy_with_trailing._check_trailing_stop(
            "AAPL", pos, Decimal("100.80")
        )
        assert vwap_strategy_with_trailing._trailing_highs["AAPL"] == Decimal("100.80")

        # Higher price updates the high
        vwap_strategy_with_trailing._check_trailing_stop(
            "AAPL", pos, Decimal("101.50")
        )
        assert vwap_strategy_with_trailing._trailing_highs["AAPL"] == Decimal("101.50")

        # Lower price doesn't update the high
        vwap_strategy_with_trailing._check_trailing_stop(
            "AAPL", pos, Decimal("101.20")
        )
        assert vwap_strategy_with_trailing._trailing_highs["AAPL"] == Decimal("101.50")

    def test_reset_trailing_high(
        self, vwap_strategy_with_trailing: VWAPScalpStrategy
    ) -> None:
        """_reset_trailing_high clears the tracked high for a symbol."""
        vwap_strategy_with_trailing._trailing_highs["AAPL"] = Decimal("200")
        vwap_strategy_with_trailing._reset_trailing_high("AAPL")
        assert "AAPL" not in vwap_strategy_with_trailing._trailing_highs

    def test_reset_trailing_high_nonexistent_symbol(
        self, vwap_strategy_with_trailing: VWAPScalpStrategy
    ) -> None:
        """Resetting a symbol not tracked should not raise."""
        vwap_strategy_with_trailing._reset_trailing_high("NOPE")  # No error


# ══════════════════════════════════════════════════════════════════════════════
# Integration: filters initialized correctly on BaseStrategy
# ══════════════════════════════════════════════════════════════════════════════


class TestBaseStrategyInit:
    """Test that BaseStrategy properly initializes filter-related state."""

    def test_trailing_highs_initialized(
        self, vwap_strategy: VWAPScalpStrategy
    ) -> None:
        """_trailing_highs dict exists and is empty on init."""
        assert hasattr(vwap_strategy, "_trailing_highs")
        assert vwap_strategy._trailing_highs == {}

    def test_positions_dict_initialized(
        self, vwap_strategy: VWAPScalpStrategy
    ) -> None:
        assert hasattr(vwap_strategy, "positions")
        assert vwap_strategy.positions == {}

    def test_get_position_returns_none_for_unknown(
        self, vwap_strategy: VWAPScalpStrategy
    ) -> None:
        assert vwap_strategy._get_position("AAPL") is None

    def test_get_position_returns_position(
        self, vwap_strategy: VWAPScalpStrategy
    ) -> None:
        pos = Position(
            symbol="AAPL",
            side="LONG",
            entry_price=Decimal("185.00"),
            current_price=Decimal("185.50"),
            quantity=Decimal("3"),
            strategy="vwap_scalp",
        )
        vwap_strategy.positions["AAPL"] = pos
        assert vwap_strategy._get_position("AAPL") is pos
