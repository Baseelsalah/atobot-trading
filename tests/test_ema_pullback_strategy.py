"""Tests for the EMA Pullback day-trading strategy."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.config.settings import Settings
from src.exchange.base_client import BaseExchangeClient
from src.models.order import Order, OrderSide, OrderType
from src.models.position import Position
from src.risk.risk_manager import RiskManager
from src.strategies.ema_pullback_strategy import EMAPullbackStrategy


@pytest.fixture
def strategy(mock_settings: Settings, mock_exchange_client: AsyncMock) -> EMAPullbackStrategy:
    """Create an EMAPullbackStrategy with mocked deps."""
    mock_settings.DEFAULT_STRATEGY = "ema_pullback"
    mock_settings.EMA_PULLBACK_FAST_PERIOD = 9
    mock_settings.EMA_PULLBACK_SLOW_PERIOD = 21
    mock_settings.EMA_PULLBACK_TREND_PERIOD = 50
    mock_settings.EMA_PULLBACK_RSI_OVERSOLD = 40.0
    mock_settings.EMA_PULLBACK_RSI_OVERBOUGHT = 70.0
    mock_settings.EMA_PULLBACK_VOLUME_MULTIPLIER = 1.2
    mock_settings.EMA_PULLBACK_TAKE_PROFIT_PERCENT = 1.5
    mock_settings.EMA_PULLBACK_STOP_LOSS_PERCENT = 0.75
    mock_settings.EMA_PULLBACK_ORDER_SIZE_USD = 500.0
    mock_settings.TRAILING_STOP_ENABLED = False
    rm = RiskManager(mock_settings)
    s = EMAPullbackStrategy(mock_exchange_client, rm, mock_settings)
    return s


def _make_uptrend_bars(count: int = 60, start_price: float = 180.0,
                       trend_step: float = 0.15, volume: float = 50000) -> list[dict]:
    """Build 5-min bars in a steady uptrend for EMA testing.

    Prices rise gradually so that:
    - 50 EMA (trend) stays below price
    - 9 EMA > 21 EMA (momentum aligned)
    """
    bars = []
    for i in range(count):
        price = start_price + i * trend_step
        bars.append({
            "timestamp": 1700000000000 + i * 300000,
            "open": Decimal(str(round(price - 0.05, 2))),
            "high": Decimal(str(round(price + 0.3, 2))),
            "low": Decimal(str(round(price - 0.2, 2))),
            "close": Decimal(str(round(price, 2))),
            "volume": Decimal(str(int(volume))),
        })
    return bars


def _make_pullback_bars(count: int = 60, pullback_start: int = 55) -> list[dict]:
    """Build bars that trend up then pull back to 9 EMA for entry signal.

    First 55 bars trend up, then 5 bars pull back toward the 9 EMA.
    """
    bars = []
    start_price = 180.0
    for i in range(pullback_start):
        price = start_price + i * 0.15
        bars.append({
            "timestamp": 1700000000000 + i * 300000,
            "open": Decimal(str(round(price - 0.05, 2))),
            "high": Decimal(str(round(price + 0.3, 2))),
            "low": Decimal(str(round(price - 0.2, 2))),
            "close": Decimal(str(round(price, 2))),
            "volume": Decimal(str(60000)),
        })
    # Pullback: price drops back toward the fast EMA
    last_price = start_price + (pullback_start - 1) * 0.15
    for i in range(count - pullback_start):
        price = last_price - (i + 1) * 0.08
        bars.append({
            "timestamp": 1700000000000 + (pullback_start + i) * 300000,
            "open": Decimal(str(round(price + 0.02, 2))),
            "high": Decimal(str(round(price + 0.15, 2))),
            "low": Decimal(str(round(price - 0.1, 2))),
            "close": Decimal(str(round(price, 2))),
            "volume": Decimal(str(70000)),  # Above-avg volume
        })
    return bars


class TestEMAPullbackInit:
    """Tests for initialization."""

    def test_strategy_name(self, strategy: EMAPullbackStrategy) -> None:
        assert strategy.name == "ema_pullback"

    @pytest.mark.asyncio
    async def test_initialize_symbol(self, strategy: EMAPullbackStrategy) -> None:
        await strategy.initialize("AAPL")
        assert "AAPL" in strategy._initialized_symbols
        assert strategy.is_running is True
        strategy.exchange.get_symbol_filters.assert_awaited_once_with("AAPL")


class TestEMAPullbackEntry:
    """Tests for entry signals."""

    @pytest.mark.asyncio
    async def test_no_entry_without_data(self, strategy: EMAPullbackStrategy) -> None:
        """Should return no orders if not enough kline data."""
        strategy.exchange.get_klines = AsyncMock(return_value=[])
        await strategy.initialize("AAPL")
        orders = await strategy.on_tick("AAPL", Decimal("185.00"))
        assert orders == []

    @pytest.mark.asyncio
    async def test_no_entry_when_position_exists(self, strategy: EMAPullbackStrategy) -> None:
        """No new entry when already holding a position."""
        await strategy.initialize("AAPL")
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("185"), current_price=Decimal("186"),
            quantity=Decimal("2.7"), strategy="ema_pullback",
        )
        orders = await strategy.on_tick("AAPL", Decimal("186"))
        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_no_entry_below_trend_ema(self, strategy: EMAPullbackStrategy) -> None:
        """Should not enter when price is below 50 EMA (no uptrend)."""
        # Build flat/downtrend bars: price below 50 EMA
        bars = []
        for i in range(60):
            price = 190.0 - i * 0.1  # Declining
            bars.append({
                "timestamp": 1700000000000 + i * 300000,
                "open": Decimal(str(round(price + 0.05, 2))),
                "high": Decimal(str(round(price + 0.3, 2))),
                "low": Decimal(str(round(price - 0.2, 2))),
                "close": Decimal(str(round(price, 2))),
                "volume": Decimal("50000"),
            })
        strategy.exchange.get_klines = AsyncMock(return_value=bars)
        await strategy.initialize("AAPL")
        orders = await strategy.on_tick("AAPL", Decimal("184.00"))
        assert orders == []

    @pytest.mark.asyncio
    async def test_no_entry_when_ema_not_aligned(self, strategy: EMAPullbackStrategy) -> None:
        """No entry when 9 EMA <= 21 EMA (no momentum)."""
        # Build choppy bars where EMAs interleave
        bars = []
        for i in range(60):
            price = 185.0 + (0.1 if i % 2 == 0 else -0.1)
            bars.append({
                "timestamp": 1700000000000 + i * 300000,
                "open": Decimal(str(round(price, 2))),
                "high": Decimal(str(round(price + 0.3, 2))),
                "low": Decimal(str(round(price - 0.3, 2))),
                "close": Decimal(str(round(price, 2))),
                "volume": Decimal("50000"),
            })
        strategy.exchange.get_klines = AsyncMock(return_value=bars)
        await strategy.initialize("AAPL")
        orders = await strategy.on_tick("AAPL", Decimal("185.00"))
        assert orders == []


class TestEMAPullbackExit:
    """Tests for exit signals."""

    @pytest.mark.asyncio
    async def test_take_profit_triggers_sell(self, strategy: EMAPullbackStrategy) -> None:
        """SELL when unrealized PnL passes TP threshold."""
        await strategy.initialize("AAPL")
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("185"), current_price=Decimal("185"),
            quantity=Decimal("2.7"), strategy="ema_pullback",
        )
        # Price rises 1.5% from entry
        exit_price = Decimal("187.78")  # 1.5% above 185
        orders = await strategy.on_tick("AAPL", exit_price)
        assert len(orders) == 1
        assert orders[0].side == "SELL"
        assert orders[0].strategy == "ema_pullback"

    @pytest.mark.asyncio
    async def test_stop_loss_triggers_sell(self, strategy: EMAPullbackStrategy) -> None:
        """SELL when unrealized loss exceeds SL threshold."""
        await strategy.initialize("AAPL")
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("185"), current_price=Decimal("185"),
            quantity=Decimal("2.7"), strategy="ema_pullback",
        )
        # Price drops 0.75% from entry
        exit_price = Decimal("183.61")  # 0.75% below 185
        orders = await strategy.on_tick("AAPL", exit_price)
        assert len(orders) == 1
        assert orders[0].side == "SELL"

    @pytest.mark.asyncio
    async def test_trailing_stop_exit(self, strategy: EMAPullbackStrategy) -> None:
        """SELL when trailing stop triggers after price reversal."""
        strategy.settings.TRAILING_STOP_ENABLED = True
        strategy.settings.TRAILING_STOP_ACTIVATION_PCT = 0.5
        strategy.settings.TRAILING_STOP_DISTANCE_PCT = 0.3
        await strategy.initialize("AAPL")

        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("185"), current_price=Decimal("185"),
            quantity=Decimal("2.7"), strategy="ema_pullback",
        )

        # Price goes up first (activate trailing) — 1% above entry
        strategy._trailing_highs["AAPL"] = Decimal("186.85")

        # Now drop 0.4% from trailing high (needs to be clearly below trail stop)
        drop_price = Decimal("186.10")  # ~0.40% below 186.85 -> triggers trail
        orders = await strategy.on_tick("AAPL", drop_price)
        assert len(orders) == 1
        assert orders[0].side == "SELL"


class TestEMAPullbackOrderFilled:
    """Tests for order fill handling."""

    @pytest.mark.asyncio
    async def test_buy_fill_creates_position(self, strategy: EMAPullbackStrategy) -> None:
        """BUY fill should create a new position."""
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            price=Decimal("185.50"), quantity=Decimal("2.7"),
            strategy="ema_pullback",
        )
        order.filled_quantity = Decimal("2.7")
        await strategy.on_order_filled(order)
        assert "AAPL" in strategy.positions
        pos = strategy.positions["AAPL"]
        assert pos.entry_price == Decimal("185.50")
        assert pos.quantity == Decimal("2.7")
        assert pos.strategy == "ema_pullback"

    @pytest.mark.asyncio
    async def test_sell_fill_closes_position(self, strategy: EMAPullbackStrategy) -> None:
        """SELL fill should reduce/close position."""
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("185"), current_price=Decimal("187"),
            quantity=Decimal("2.7"), strategy="ema_pullback",
        )
        order = Order(
            symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET,
            price=Decimal("187.00"), quantity=Decimal("2.7"),
            strategy="ema_pullback",
        )
        order.filled_quantity = Decimal("2.7")
        await strategy.on_order_filled(order)
        pos = strategy.positions["AAPL"]
        assert pos.is_closed


class TestEMAPullbackStatus:
    """Tests for strategy status."""

    @pytest.mark.asyncio
    async def test_get_status(self, strategy: EMAPullbackStrategy) -> None:
        status = await strategy.get_status()
        assert status["strategy"] == "ema_pullback"
        assert status["is_running"] is False  # not initialized yet

    @pytest.mark.asyncio
    async def test_get_status_with_position(self, strategy: EMAPullbackStrategy) -> None:
        strategy.is_running = True
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("185"), current_price=Decimal("187"),
            quantity=Decimal("2.7"), strategy="ema_pullback",
        )
        status = await strategy.get_status()
        assert status["is_running"] is True
        assert "AAPL" in status["positions"]
