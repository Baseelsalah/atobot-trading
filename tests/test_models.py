"""Tests for the Pydantic data models."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.models.order import Order, OrderSide, OrderStatus, OrderType
from src.models.position import Position
from src.models.trade import Trade


class TestOrderModel:
    """Tests for the Order model."""

    def test_order_creation(self) -> None:
        """Order should be created with default values."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("185.50"),
            quantity=Decimal("3"),
            strategy="momentum",
        )
        assert order.symbol == "AAPL"
        assert order.status == OrderStatus.PENDING
        assert order.internal_id is not None

    def test_notional_value(self) -> None:
        """notional_value should be price * quantity."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("200"),
            quantity=Decimal("10"),
            strategy="momentum",
        )
        assert order.notional_value == Decimal("2000")

    def test_is_active(self) -> None:
        """is_active should be True for PENDING and OPEN orders."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("185.50"),
            quantity=Decimal("3"),
            strategy="momentum",
        )
        assert order.is_active is True
        order.mark_filled()
        assert order.is_active is False

    def test_mark_filled(self) -> None:
        """mark_filled should update status and filled_quantity."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("185.50"),
            quantity=Decimal("3"),
            strategy="momentum",
        )
        order.mark_filled(Decimal("3"))
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("3")

    def test_mark_cancelled(self) -> None:
        """mark_cancelled should update status."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("185.50"),
            quantity=Decimal("3"),
            strategy="momentum",
        )
        order.mark_cancelled()
        assert order.status == OrderStatus.CANCELLED

    def test_mark_failed(self) -> None:
        """mark_failed should update status and store response."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("185.50"),
            quantity=Decimal("3"),
            strategy="momentum",
        )
        order.mark_failed({"reason": "Insufficient balance"})
        assert order.status == OrderStatus.FAILED

    def test_remaining_quantity(self) -> None:
        """remaining_quantity should reflect unfilled amount."""
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("185.50"),
            quantity=Decimal("10"),
            filled_quantity=Decimal("4"),
            strategy="momentum",
        )
        assert order.remaining_quantity == Decimal("6")


class TestPositionModel:
    """Tests for the Position model."""

    def test_position_creation(self) -> None:
        """Position should be created with basic fields."""
        pos = Position(
            symbol="AAPL",
            side="LONG",
            entry_price=Decimal("185.00"),
            quantity=Decimal("10"),
            current_price=Decimal("185.00"),
        )
        assert pos.symbol == "AAPL"
        assert pos.quantity == Decimal("10")

    def test_unrealized_pnl_long(self) -> None:
        """Unrealized PnL % should be positive when price goes up for long."""
        pos = Position(
            symbol="AAPL",
            side="LONG",
            entry_price=Decimal("100"),
            quantity=Decimal("10"),
            current_price=Decimal("100"),
        )
        pos.update_price(Decimal("110"))
        assert pos.unrealized_pnl_percent == Decimal("10")

    def test_update_price(self) -> None:
        """update_price should set current_price."""
        pos = Position(
            symbol="AAPL",
            side="LONG",
            entry_price=Decimal("185.00"),
            quantity=Decimal("10"),
            current_price=Decimal("185.00"),
        )
        pos.update_price(Decimal("190.00"))
        assert pos.current_price == Decimal("190.00")

    def test_add_to_position(self) -> None:
        """add_to_position should compute weighted average entry."""
        pos = Position(
            symbol="AAPL",
            side="LONG",
            entry_price=Decimal("200"),
            quantity=Decimal("10"),
            current_price=Decimal("200"),
        )
        pos.add_to_position(Decimal("180"), Decimal("10"))
        assert pos.quantity == Decimal("20")
        assert pos.entry_price == Decimal("190")  # weighted avg

    def test_reduce_position(self) -> None:
        """reduce_position should decrease quantity and return realized PnL."""
        pos = Position(
            symbol="AAPL",
            side="LONG",
            entry_price=Decimal("180"),
            quantity=Decimal("10"),
            current_price=Decimal("190"),
        )
        pnl = pos.reduce_position(Decimal("5"), Decimal("190"))
        assert pos.quantity == Decimal("5")
        # PnL = (190 - 180) * 5 = 50
        assert pnl == Decimal("50")


class TestTradeModel:
    """Tests for the Trade model."""

    def test_trade_creation(self) -> None:
        """Trade should be created with required fields."""
        trade = Trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            price=Decimal("185.50"),
            quantity=Decimal("3"),
            fee=Decimal("0.00"),
        )
        assert trade.symbol == "AAPL"

    def test_notional_value(self) -> None:
        """notional_value should be price * quantity."""
        trade = Trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            price=Decimal("200"),
            quantity=Decimal("10"),
            fee=Decimal("0"),
        )
        assert trade.notional_value == Decimal("2000")

    def test_net_value(self) -> None:
        """net_value should subtract fee from notional."""
        trade = Trade(
            symbol="AAPL",
            side=OrderSide.BUY,
            price=Decimal("200"),
            quantity=Decimal("10"),
            fee=Decimal("5"),
        )
        assert trade.net_value == Decimal("1995")
