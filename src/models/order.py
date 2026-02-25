"""Order model for AtoBot Trading."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"    # Open a short position
    COVER = "COVER"    # Close a short position (buy-to-cover)


class OrderStatus(str, Enum):
    """Order lifecycle status."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class OrderType(str, Enum):
    """Order type enumeration."""

    LIMIT = "LIMIT"
    MARKET = "MARKET"


class Order(BaseModel):
    """Represents a trading order."""

    id: str | None = None  # Exchange order ID (assigned after placement)
    internal_id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: Decimal
    quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    status: OrderStatus = OrderStatus.PENDING
    strategy: str  # Which strategy created this order
    grid_level: int | None = None  # Strategy-specific level index
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    exchange_response: dict[str, Any] | None = None

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    @property
    def notional_value(self) -> Decimal:
        """Calculate total order value in quote currency."""
        return self.price * self.quantity

    @property
    def is_active(self) -> bool:
        """Check if order is still active (pending or open)."""
        return self.status in (
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
        )

    @property
    def remaining_quantity(self) -> Decimal:
        """Return unfilled quantity."""
        return self.quantity - self.filled_quantity

    def mark_filled(self, filled_qty: Decimal | None = None) -> None:
        """Mark the order as filled (fully or partially)."""
        now = datetime.now(timezone.utc)
        if filled_qty is not None:
            self.filled_quantity = filled_qty
        else:
            self.filled_quantity = self.quantity

        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        self.updated_at = now

    def mark_cancelled(self) -> None:
        """Mark the order as cancelled."""
        self.status = OrderStatus.CANCELLED
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self, response: dict[str, Any] | None = None) -> None:
        """Mark the order as failed."""
        self.status = OrderStatus.FAILED
        self.exchange_response = response
        self.updated_at = datetime.now(timezone.utc)
