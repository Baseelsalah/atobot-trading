"""Tests for the VWAP Scalp day-trading strategy."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.config.settings import Settings
from src.exchange.base_client import BaseExchangeClient
from src.models.order import Order, OrderSide, OrderType
from src.models.position import Position
from src.risk.risk_manager import RiskManager
from src.strategies.vwap_strategy import VWAPScalpStrategy


@pytest.fixture
def strategy(mock_settings: Settings, mock_exchange_client: AsyncMock) -> VWAPScalpStrategy:
    """Create a VWAPScalpStrategy with mocked deps."""
    mock_settings.DEFAULT_STRATEGY = "vwap_scalp"
    mock_settings.VWAP_BOUNCE_PERCENT = 0.3
    mock_settings.VWAP_TAKE_PROFIT_PERCENT = 0.5
    mock_settings.VWAP_STOP_LOSS_PERCENT = 0.3
    mock_settings.VWAP_ORDER_SIZE_USD = 500.0
    rm = RiskManager(mock_settings)
    s = VWAPScalpStrategy(mock_exchange_client, rm, mock_settings)
    return s


def _make_bars(prices: list[tuple[str, str, str, str]], volume: str = "10000") -> list[dict]:
    """Return a simple list of bar dicts for testing."""
    bars = []
    for i, (o, h, l, c) in enumerate(prices):
        bars.append({
            "timestamp": 1700000000000 + i * 300000,
            "open": Decimal(o), "high": Decimal(h),
            "low": Decimal(l), "close": Decimal(c),
            "volume": Decimal(volume),
        })
    return bars


class TestVWAPInit:
    """Tests for initialization."""

    def test_strategy_name(self, strategy: VWAPScalpStrategy) -> None:
        assert strategy.name == "vwap_scalp"

    @pytest.mark.asyncio
    async def test_initialize_symbol(self, strategy: VWAPScalpStrategy) -> None:
        await strategy.initialize("AAPL")
        assert "AAPL" in strategy._initialized_symbols
        assert strategy.is_running is True


class TestVWAPCompute:
    """Tests for internal VWAP calculation."""

    def test_compute_vwap_basic(self, strategy: VWAPScalpStrategy) -> None:
        """VWAP = cumulative(typical_price * volume) / cumulative(volume)."""
        bars = _make_bars([
            ("184", "186", "183", "185"),
            ("185", "187", "184", "186"),
        ])
        vwap = strategy._compute_vwap(bars)
        # typical1 = (186+183+185)/3 = 184.666...
        # typical2 = (187+184+186)/3 = 185.666...
        # VWAP = (184.666*10000 + 185.666*10000) / 20000 = 185.166...
        assert vwap is not None
        assert abs(float(vwap) - 185.166) < 0.5

    def test_compute_vwap_empty(self, strategy: VWAPScalpStrategy) -> None:
        vwap = strategy._compute_vwap([])
        assert vwap is None or vwap == Decimal("0")


class TestVWAPEntry:
    """Tests for entry signals."""

    @pytest.mark.asyncio
    async def test_buy_below_vwap(self, strategy: VWAPScalpStrategy) -> None:
        """BUY when price is below VWAP by bounce percent."""
        bars = _make_bars([
            ("184", "186", "183", "185"),
            ("185", "187", "184", "186"),
            ("186", "188", "185", "187"),
        ])
        strategy.exchange.get_klines = AsyncMock(return_value=bars)
        await strategy.initialize("AAPL")

        # VWAP ≈ 186. Price well below VWAP - bounce_pct
        orders = await strategy.on_tick("AAPL", Decimal("183.00"))
        assert len(orders) == 1
        assert orders[0].side == "BUY"
        assert orders[0].symbol == "AAPL"
        assert orders[0].strategy == "vwap_scalp"

    @pytest.mark.asyncio
    async def test_no_entry_at_vwap(self, strategy: VWAPScalpStrategy) -> None:
        """No entry if price is at or above VWAP."""
        bars = _make_bars([
            ("184", "186", "183", "185"),
            ("185", "187", "184", "186"),
        ])
        strategy.exchange.get_klines = AsyncMock(return_value=bars)
        await strategy.initialize("AAPL")

        # Price right around VWAP — no entry
        orders = await strategy.on_tick("AAPL", Decimal("186.00"))
        assert orders == []


class TestVWAPExit:
    """Tests for exit / profit-take / stop-loss."""

    @pytest.mark.asyncio
    async def test_take_profit(self, strategy: VWAPScalpStrategy) -> None:
        """SELL when price rises above TP threshold from entry."""
        await strategy.initialize("AAPL")
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("183.00"), current_price=Decimal("183.00"),
            quantity=Decimal("3"), strategy="vwap_scalp",
        )
        strategy.settings.VWAP_TAKE_PROFIT_PERCENT = 0.5
        bars = _make_bars([
            ("184", "186", "183", "185"),
            ("185", "187", "184", "186"),
        ])
        strategy.exchange.get_klines = AsyncMock(return_value=bars)

        # Price up > 0.5% from 183 entry → ~184.0+
        orders = await strategy.on_tick("AAPL", Decimal("185.00"))
        assert len(orders) == 1
        assert orders[0].side == "SELL"

    @pytest.mark.asyncio
    async def test_stop_loss(self, strategy: VWAPScalpStrategy) -> None:
        """SELL when price falls below SL threshold from entry."""
        await strategy.initialize("AAPL")
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("185.00"), current_price=Decimal("185.00"),
            quantity=Decimal("3"), strategy="vwap_scalp",
        )
        strategy.settings.VWAP_STOP_LOSS_PERCENT = 0.3
        bars = _make_bars([
            ("184", "186", "183", "185"),
        ])
        strategy.exchange.get_klines = AsyncMock(return_value=bars)

        # Price down > 0.3% from 185 → ~184.44
        orders = await strategy.on_tick("AAPL", Decimal("183.00"))
        assert len(orders) == 1
        assert orders[0].side == "SELL"


class TestVWAPOrderFilled:
    """Tests for on_order_filled."""

    @pytest.mark.asyncio
    async def test_buy_creates_position(self, strategy: VWAPScalpStrategy) -> None:
        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            price=Decimal("183.00"), quantity=Decimal("3"), strategy="vwap_scalp",
        )
        order.filled_quantity = Decimal("3")
        await strategy.on_order_filled(order)
        assert "AAPL" in strategy.positions

    @pytest.mark.asyncio
    async def test_sell_closes_position(self, strategy: VWAPScalpStrategy) -> None:
        strategy.positions["AAPL"] = Position(
            symbol="AAPL", side="LONG",
            entry_price=Decimal("183.00"), current_price=Decimal("185.00"),
            quantity=Decimal("3"), strategy="vwap_scalp",
        )
        order = Order(
            symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET,
            price=Decimal("185.00"), quantity=Decimal("3"), strategy="vwap_scalp",
        )
        order.filled_quantity = Decimal("3")
        await strategy.on_order_filled(order)
        assert strategy.positions["AAPL"].quantity == Decimal("0")


class TestVWAPStatus:
    """Tests for get_status."""

    @pytest.mark.asyncio
    async def test_get_status(self, strategy: VWAPScalpStrategy) -> None:
        await strategy.initialize("AAPL")
        status = await strategy.get_status()
        assert status["strategy"] == "vwap_scalp"
        assert "positions" in status or "running" in status
