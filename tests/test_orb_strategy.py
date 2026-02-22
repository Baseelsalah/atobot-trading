"""Tests for the Opening Range Breakout (ORB) day-trading strategy."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.config.settings import Settings
from src.exchange.base_client import BaseExchangeClient
from src.models.order import Order, OrderSide, OrderType
from src.models.position import Position
from src.risk.risk_manager import RiskManager
from src.strategies.orb_strategy import ORBStrategy


@pytest.fixture
def strategy(mock_settings: Settings, mock_exchange_client: AsyncMock) -> ORBStrategy:
    """Create an ORBStrategy with mocked deps."""
    mock_settings.DEFAULT_STRATEGY = "orb"
    mock_settings.ORB_RANGE_MINUTES = 3  # Small for testing
    mock_settings.ORB_BREAKOUT_PERCENT = 0.1
    mock_settings.ORB_TAKE_PROFIT_PERCENT = 1.5
    mock_settings.ORB_STOP_LOSS_PERCENT = 0.75
    mock_settings.ORB_ORDER_SIZE_USD = 500.0
    rm = RiskManager(mock_settings)
    s = ORBStrategy(mock_exchange_client, rm, mock_settings)
    return s


class TestORBInit:
    """Tests for initialization."""

    def test_strategy_name(self, strategy: ORBStrategy) -> None:
        assert strategy.name == "orb"

    @pytest.mark.asyncio
    async def test_initialize_symbol(self, strategy: ORBStrategy) -> None:
        await strategy.initialize("AAPL")
        assert "AAPL" in strategy._initialized_symbols
        assert strategy._range_set["AAPL"] is False
        assert strategy._traded_today["AAPL"] is False
        assert strategy.is_running is True


class TestORBRange:
    """Tests for opening range establishment."""

    @pytest.mark.asyncio
    async def test_range_set_from_bars(self, strategy: ORBStrategy) -> None:
        """Opening range should be set from first N minutes of bars."""
        bars = [
            {"timestamp": 1700000000000 + i * 60000,
             "open": Decimal("184"), "high": Decimal(str(186 + i)),
             "low": Decimal(str(183 - i)), "close": Decimal("185"),
             "volume": Decimal("10000")}
            for i in range(5)
        ]
        strategy.exchange.get_klines = AsyncMock(return_value=bars)
        await strategy.initialize("AAPL")

        orders = await strategy.on_tick("AAPL", Decimal("185.00"))
        assert strategy._range_set["AAPL"] is True
        # Range high = max of first 3 bars: 186, 187, 188 → 188
        assert strategy._range_high["AAPL"] == Decimal("188")
        # Range low = min of first 3 bars: 183, 182, 181 → 181
        assert strategy._range_low["AAPL"] == Decimal("181")
        assert orders == []  # Just setting range, no trade yet


class TestORBBreakout:
    """Tests for breakout detection and entry."""

    @pytest.mark.asyncio
    async def test_bullish_breakout_buy(self, strategy: ORBStrategy) -> None:
        """BUY on breakout above range high."""
        from datetime import datetime, timezone
        await strategy.initialize("AAPL")
        # Manually set the range and prevent daily reset from clearing it
        strategy._last_reset_date = datetime.now(timezone.utc)
        strategy._range_set["AAPL"] = True
        strategy._range_high["AAPL"] = Decimal("186.00")
        strategy._range_low["AAPL"] = Decimal("183.00")

        # Price above breakout level: 186 * (1 + 0.001) ≈ 186.186
        orders = await strategy.on_tick("AAPL", Decimal("187.00"))
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].symbol == "AAPL"
        assert orders[0].strategy == "orb"
        assert strategy._traded_today["AAPL"] is True

    @pytest.mark.asyncio
    async def test_no_entry_within_range(self, strategy: ORBStrategy) -> None:
        """No order when price is within the opening range."""
        from datetime import datetime, timezone
        await strategy.initialize("AAPL")
        strategy._last_reset_date = datetime.now(timezone.utc)
        strategy._range_set["AAPL"] = True
        strategy._range_high["AAPL"] = Decimal("186.00")
        strategy._range_low["AAPL"] = Decimal("183.00")

        orders = await strategy.on_tick("AAPL", Decimal("184.50"))
        assert orders == []

    @pytest.mark.asyncio
    async def test_no_double_trade(self, strategy: ORBStrategy) -> None:
        """Once traded today, no second entry on same symbol."""
        from datetime import datetime, timezone
        await strategy.initialize("AAPL")
        strategy._last_reset_date = datetime.now(timezone.utc)
        strategy._range_set["AAPL"] = True
        strategy._range_high["AAPL"] = Decimal("186.00")
        strategy._range_low["AAPL"] = Decimal("183.00")
        strategy._traded_today["AAPL"] = True

        orders = await strategy.on_tick("AAPL", Decimal("187.00"))
        assert orders == []


class TestORBExit:
    """Tests for ORB exit logic."""

    @pytest.mark.asyncio
    async def test_take_profit(self, strategy: ORBStrategy) -> None:
        """SELL when take-profit threshold is met."""
        await strategy.initialize("AAPL")
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("186.00"), current_price=Decimal("186.00"),
            quantity=Decimal("3"), strategy="orb",
        )
        strategy.settings.ORB_TAKE_PROFIT_PERCENT = 1.0

        # Price up ~2.15% → exceeds TP = 1%
        orders = await strategy.on_tick("AAPL", Decimal("190.00"))
        assert len(orders) == 1
        assert orders[0].side == "SELL"

    @pytest.mark.asyncio
    async def test_stop_loss(self, strategy: ORBStrategy) -> None:
        """SELL when stop-loss threshold is breached."""
        await strategy.initialize("AAPL")
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("186.00"), current_price=Decimal("186.00"),
            quantity=Decimal("3"), strategy="orb",
        )
        strategy.settings.ORB_STOP_LOSS_PERCENT = 0.5

        # Price down ~1.1% → exceeds SL = 0.5%
        orders = await strategy.on_tick("AAPL", Decimal("184.00"))
        assert len(orders) == 1
        assert orders[0].side == "SELL"


class TestORBOrderFilled:
    """Tests for on_order_filled."""

    @pytest.mark.asyncio
    async def test_buy_creates_position(self, strategy: ORBStrategy) -> None:
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            price=Decimal("187.00"), quantity=Decimal("3"), strategy="orb",
        )
        order.filled_quantity = Decimal("3")
        await strategy.on_order_filled(order)
        assert "AAPL" in strategy.positions

    @pytest.mark.asyncio
    async def test_sell_reduces_position(self, strategy: ORBStrategy) -> None:
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("186.00"), current_price=Decimal("190.00"),
            quantity=Decimal("3"), strategy="orb",
        )
        order = Order(
            symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET,
            price=Decimal("190.00"), quantity=Decimal("3"), strategy="orb",
        )
        order.filled_quantity = Decimal("3")
        await strategy.on_order_filled(order)
        assert strategy.positions["AAPL"].quantity == Decimal("0")


class TestORBStatus:
    """Tests for get_status."""

    @pytest.mark.asyncio
    async def test_get_status(self, strategy: ORBStrategy) -> None:
        await strategy.initialize("AAPL")
        status = await strategy.get_status()
        assert status["strategy"] == "orb"
        assert "ranges" in status
        assert "AAPL" in status["ranges"]


class TestORBDailyReset:
    """Tests for daily reset logic."""

    @pytest.mark.asyncio
    async def test_daily_reset_clears_state(self, strategy: ORBStrategy) -> None:
        """New day should reset ranges and traded_today flags."""
        from datetime import datetime, timedelta, timezone
        await strategy.initialize("AAPL")
        strategy._range_set["AAPL"] = True
        strategy._traded_today["AAPL"] = True
        strategy._range_high["AAPL"] = Decimal("186.00")

        # Simulate moving to next day
        strategy._last_reset_date = datetime.now(timezone.utc) - timedelta(days=1)
        strategy._maybe_reset_daily()

        assert strategy._range_set["AAPL"] is False
        assert strategy._traded_today["AAPL"] is False
        assert strategy._range_high["AAPL"] == Decimal("0")
