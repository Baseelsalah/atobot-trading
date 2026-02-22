"""Tests for the RiskManager module."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.config.settings import Settings
from src.models.order import Order, OrderSide, OrderStatus, OrderType
from src.risk.risk_manager import RiskManager


@pytest.fixture
def risk_manager(mock_settings: Settings) -> RiskManager:
    """Create a RiskManager instance for testing."""
    return RiskManager(mock_settings)


def _make_order(
    side: OrderSide = OrderSide.BUY,
    price: Decimal = Decimal("185.50"),
    quantity: Decimal = Decimal("3"),
) -> Order:
    return Order(
        symbol="AAPL",
        side=side,
        order_type=OrderType.LIMIT,
        price=price,
        quantity=quantity,
        strategy="momentum",
    )


class TestCanPlaceOrder:
    """Tests for can_place_order()."""

    @pytest.mark.asyncio
    async def test_order_allowed_when_healthy(
        self, risk_manager: RiskManager
    ) -> None:
        """Order should be allowed under normal conditions."""
        order = _make_order()
        balance = {"USD": Decimal("50000"), "EQUITY": Decimal("50000"), "DAYTRADE_COUNT": Decimal("0")}
        allowed, reason = await risk_manager.can_place_order(order, balance)
        assert allowed is True
        assert reason == ""

    @pytest.mark.asyncio
    async def test_order_rejected_when_halted(
        self, risk_manager: RiskManager
    ) -> None:
        """Order should be rejected when trading is halted."""
        risk_manager.is_halted = True
        risk_manager.halt_reason = "test halt"
        order = _make_order()
        balance = {"USD": Decimal("50000"), "EQUITY": Decimal("50000")}
        allowed, reason = await risk_manager.can_place_order(order, balance)
        assert allowed is False

    @pytest.mark.asyncio
    async def test_order_rejected_daily_loss_exceeded(
        self, risk_manager: RiskManager
    ) -> None:
        """Order should be rejected when daily loss limit exceeded."""
        risk_manager.daily_pnl = Decimal("-250")
        order = _make_order()
        balance = {"USD": Decimal("50000"), "EQUITY": Decimal("50000"), "DAYTRADE_COUNT": Decimal("0")}
        allowed, reason = await risk_manager.can_place_order(order, balance)
        assert allowed is False

    @pytest.mark.asyncio
    async def test_order_rejected_insufficient_balance(
        self, risk_manager: RiskManager
    ) -> None:
        """Order should be rejected when USD balance is too low."""
        order = _make_order(price=Decimal("185.50"), quantity=Decimal("5"))
        balance = {"USD": Decimal("500"), "EQUITY": Decimal("50000"), "DAYTRADE_COUNT": Decimal("0")}
        allowed, reason = await risk_manager.can_place_order(order, balance)
        assert allowed is False
        assert "Insufficient USD" in reason

    @pytest.mark.asyncio
    async def test_pdt_protection_blocks_trade(
        self, risk_manager: RiskManager
    ) -> None:
        """PDT rule should block 4th day trade when equity < 25k."""
        order = _make_order()
        balance = {"USD": Decimal("10000"), "EQUITY": Decimal("20000"), "DAYTRADE_COUNT": Decimal("3")}
        allowed, reason = await risk_manager.can_place_order(order, balance)
        assert allowed is False
        assert "PDT" in reason

    @pytest.mark.asyncio
    async def test_pdt_no_block_above_25k(
        self, risk_manager: RiskManager
    ) -> None:
        """PDT should not block when equity >= 25k."""
        order = _make_order()
        balance = {"USD": Decimal("30000"), "EQUITY": Decimal("30000"), "DAYTRADE_COUNT": Decimal("5")}
        allowed, reason = await risk_manager.can_place_order(order, balance)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_daily_trade_limit(
        self, risk_manager: RiskManager
    ) -> None:
        """Order should be rejected when MAX_DAILY_TRADES is reached."""
        risk_manager._daily_trade_count = 20
        order = _make_order()
        balance = {"USD": Decimal("50000"), "EQUITY": Decimal("50000"), "DAYTRADE_COUNT": Decimal("0")}
        allowed, reason = await risk_manager.can_place_order(order, balance)
        assert allowed is False
        assert "Daily trade limit" in reason


class TestStopLoss:
    """Tests for check_stop_loss()."""

    @pytest.mark.asyncio
    async def test_stop_loss_triggered(self, risk_manager: RiskManager) -> None:
        """Stop-loss should trigger when position loss exceeds threshold."""
        from src.models.position import Position

        pos = Position(
            symbol="AAPL",
            side="LONG",
            entry_price=Decimal("200"),
            quantity=Decimal("10"),
            current_price=Decimal("185"),
        )
        # Default STOP_LOSS_PERCENT is 5, loss here is 7.5%
        result = await risk_manager.check_stop_loss(pos)
        assert result is True

    @pytest.mark.asyncio
    async def test_stop_loss_not_triggered(self, risk_manager: RiskManager) -> None:
        """Stop-loss should not trigger when loss is small."""
        from src.models.position import Position

        pos = Position(
            symbol="AAPL",
            side="LONG",
            entry_price=Decimal("200"),
            quantity=Decimal("10"),
            current_price=Decimal("198"),
        )
        result = await risk_manager.check_stop_loss(pos)
        assert result is False


class TestEmergency:
    """Tests for emergency_shutdown() and resume_trading()."""

    @pytest.mark.asyncio
    async def test_emergency_shutdown(self, risk_manager: RiskManager) -> None:
        """emergency_shutdown should halt trading."""
        await risk_manager.emergency_shutdown("test reason")
        assert risk_manager.is_halted is True

    @pytest.mark.asyncio
    async def test_resume_trading(self, risk_manager: RiskManager) -> None:
        """resume_trading should un-halt trading."""
        risk_manager.is_halted = True
        await risk_manager.resume_trading()
        assert risk_manager.is_halted is False


class TestDrawdown:
    """Tests for drawdown tracking."""

    @pytest.mark.asyncio
    async def test_drawdown_updates(self, risk_manager: RiskManager) -> None:
        """Drawdown percent should update correctly."""
        await risk_manager.update_balance(Decimal("50000"))
        assert risk_manager.peak_balance == Decimal("50000")
        await risk_manager.update_balance(Decimal("47500"))
        assert risk_manager.current_drawdown_percent == Decimal("5")

    @pytest.mark.asyncio
    async def test_peak_balance_rises(self, risk_manager: RiskManager) -> None:
        """Peak balance should increase with new highs."""
        await risk_manager.update_balance(Decimal("50000"))
        await risk_manager.update_balance(Decimal("55000"))
        assert risk_manager.peak_balance == Decimal("55000")
