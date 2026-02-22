"""Completed trade model for AtoBot Trading."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from pydantic import BaseModel, Field

from src.models.order import OrderSide


class Trade(BaseModel):
    """Represents a completed (executed) trade."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    symbol: str
    side: OrderSide
    price: Decimal
    quantity: Decimal
    fee: Decimal = Decimal("0")
    fee_asset: str = "USD"
    pnl: Decimal | None = None
    strategy: str = ""
    order_id: str | None = None  # Link back to the originating order
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    @property
    def notional_value(self) -> Decimal:
        """Total value of the trade in quote currency."""
        return self.price * self.quantity

    @property
    def net_value(self) -> Decimal:
        """Notional value minus fees (approximation when fee is in quote)."""
        return self.notional_value - self.fee
