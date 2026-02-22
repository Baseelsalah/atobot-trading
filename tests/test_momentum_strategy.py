"""Tests for the Momentum day-trading strategy."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import Settings
from src.exchange.base_client import BaseExchangeClient
from src.models.order import Order, OrderSide, OrderType
from src.models.position import Position
from src.risk.risk_manager import RiskManager
from src.strategies.momentum_strategy import MomentumStrategy


@pytest.fixture
def strategy(mock_settings: Settings, mock_exchange_client: AsyncMock) -> MomentumStrategy:
    """Create a MomentumStrategy with mocked deps."""
    rm = RiskManager(mock_settings)
    s = MomentumStrategy(mock_exchange_client, rm, mock_settings)
    return s


class TestMomentumInit:
    """Tests for initialization."""

    def test_strategy_name(self, strategy: MomentumStrategy) -> None:
        assert strategy.name == "momentum"

    @pytest.mark.asyncio
    async def test_initialize_symbol(self, strategy: MomentumStrategy) -> None:
        await strategy.initialize("AAPL")
        assert "AAPL" in strategy._initialized_symbols
        assert strategy.is_running is True
        strategy.exchange.get_symbol_filters.assert_awaited_once_with("AAPL")


class TestMomentumEntry:
    """Tests for momentum entry signals."""

    @pytest.mark.asyncio
    async def test_no_entry_without_data(self, strategy: MomentumStrategy) -> None:
        """Should return no orders if not enough kline data."""
        strategy.exchange.get_klines = AsyncMock(return_value=[])
        await strategy.initialize("AAPL")
        orders = await strategy.on_tick("AAPL", Decimal("185.00"))
        assert orders == []

    @pytest.mark.asyncio
    async def test_no_entry_when_position_exists(self, strategy: MomentumStrategy) -> None:
        """No new entry when already holding."""
        await strategy.initialize("AAPL")
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("180"), current_price=Decimal("182"),
            quantity=Decimal("3"), strategy="momentum",
        )
        orders = await strategy.on_tick("AAPL", Decimal("182"))
        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_entry_on_oversold_rsi_with_volume(
        self, strategy: MomentumStrategy
    ) -> None:
        """Should generate BUY order when RSI oversold + volume spike."""
        import numpy as np
        # Build fake 5m bars where price drops → RSI goes oversold, volume spikes
        bars = []
        for i in range(30):
            # Declining prices to push RSI low
            price = 185.0 - i * 0.5
            bars.append({
                "timestamp": 1700000000000 + i * 300000,
                "open": Decimal(str(price + 0.1)),
                "high": Decimal(str(price + 0.5)),
                "low": Decimal(str(price - 0.3)),
                "close": Decimal(str(price)),
                "volume": Decimal("5000") if i < 29 else Decimal("50000"),  # Volume spike on last bar
            })
        strategy.exchange.get_klines = AsyncMock(return_value=bars)
        strategy.settings.MOMENTUM_RSI_OVERSOLD = 99.0  # Force RSI to be "oversold"
        strategy.settings.MOMENTUM_VOLUME_MULTIPLIER = 1.5
        await strategy.initialize("AAPL")

        orders = await strategy.on_tick("AAPL", Decimal("170.50"))
        # Should generate a BUY
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].symbol == "AAPL"
        assert orders[0].strategy == "momentum"


class TestMomentumExit:
    """Tests for momentum exit signals."""

    @pytest.mark.asyncio
    async def test_take_profit_triggers_sell(self, strategy: MomentumStrategy) -> None:
        """SELL when unrealized PnL passes TP threshold."""
        await strategy.initialize("AAPL")
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("180.00"), current_price=Decimal("180.00"),
            quantity=Decimal("3"), strategy="momentum",
        )
        strategy.settings.MOMENTUM_TAKE_PROFIT_PERCENT = 2.0

        # Price up ~3.6% → exceeds TP
        orders = await strategy.on_tick("AAPL", Decimal("186.50"))
        assert len(orders) == 1
        assert orders[0].side == "SELL"
        assert orders[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_stop_loss_triggers_sell(self, strategy: MomentumStrategy) -> None:
        """SELL when unrealized loss passes SL threshold."""
        await strategy.initialize("AAPL")
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("190.00"), current_price=Decimal("190.00"),
            quantity=Decimal("3"), strategy="momentum",
        )
        strategy.settings.MOMENTUM_STOP_LOSS_PERCENT = 1.0

        # Price down ~2.1% → exceeds SL
        orders = await strategy.on_tick("AAPL", Decimal("186.00"))
        assert len(orders) == 1
        assert orders[0].side == "SELL"

    @pytest.mark.asyncio
    async def test_hold_when_within_range(self, strategy: MomentumStrategy) -> None:
        """No orders when position PnL is within TP/SL range."""
        await strategy.initialize("AAPL")
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("185.00"), current_price=Decimal("185.00"),
            quantity=Decimal("3"), strategy="momentum",
        )
        strategy.settings.MOMENTUM_TAKE_PROFIT_PERCENT = 5.0
        strategy.settings.MOMENTUM_STOP_LOSS_PERCENT = 5.0

        orders = await strategy.on_tick("AAPL", Decimal("186.00"))
        assert len(orders) == 0


class TestMomentumOrderFilled:
    """Tests for on_order_filled."""

    @pytest.mark.asyncio
    async def test_buy_creates_position(self, strategy: MomentumStrategy) -> None:
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            price=Decimal("185.00"), quantity=Decimal("3"), strategy="momentum",
        )
        order.filled_quantity = Decimal("3")
        await strategy.on_order_filled(order)
        assert "AAPL" in strategy.positions
        assert strategy.positions["AAPL"].quantity == Decimal("3")

    @pytest.mark.asyncio
    async def test_sell_reduces_position(self, strategy: MomentumStrategy) -> None:
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("185.00"), current_price=Decimal("190.00"),
            quantity=Decimal("3"), strategy="momentum",
        )
        order = Order(
            symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET,
            price=Decimal("190.00"), quantity=Decimal("3"), strategy="momentum",
        )
        order.filled_quantity = Decimal("3")
        await strategy.on_order_filled(order)
        assert strategy.positions["AAPL"].quantity == Decimal("0")


class TestMomentumStatus:
    """Tests for get_status."""

    @pytest.mark.asyncio
    async def test_get_status(self, strategy: MomentumStrategy) -> None:
        await strategy.initialize("AAPL")
        status = await strategy.get_status()
        assert status["strategy"] == "momentum"
        assert status["is_running"] is True
