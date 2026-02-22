"""Position tracking model for AtoBot Trading."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from pydantic import BaseModel, Field


class Position(BaseModel):
    """Represents an open trading position."""

    symbol: str
    side: str = "LONG"  # "LONG" or "SHORT"
    entry_price: Decimal
    current_price: Decimal
    quantity: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    total_invested: Decimal = Decimal("0")
    order_count: int = 0
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    strategy: str = ""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def update_price(self, new_price: Decimal) -> None:
        """Update current price and recalculate unrealized PnL."""
        self.current_price = new_price
        if self.side == "LONG":
            self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - new_price) * self.quantity
        self.updated_at = datetime.now(timezone.utc)

    def add_to_position(self, price: Decimal, quantity: Decimal) -> None:
        """Add to the position (DCA-style). Recalculates weighted average entry."""
        total_cost = (self.entry_price * self.quantity) + (price * quantity)
        self.quantity += quantity
        if self.quantity > Decimal("0"):
            self.entry_price = total_cost / self.quantity
        self.total_invested += price * quantity
        self.order_count += 1
        self.updated_at = datetime.now(timezone.utc)

    def reduce_position(self, quantity: Decimal, exit_price: Decimal) -> Decimal:
        """Reduce position by given quantity. Returns realized PnL for the reduction."""
        if self.side == "LONG":
            pnl = (exit_price - self.entry_price) * quantity
        else:
            pnl = (self.entry_price - exit_price) * quantity
        self.realized_pnl += pnl
        self.quantity -= quantity
        self.updated_at = datetime.now(timezone.utc)
        return pnl

    @property
    def is_closed(self) -> bool:
        """Check if position is fully closed."""
        return self.quantity <= Decimal("0")

    @property
    def unrealized_pnl_percent(self) -> Decimal:
        """Return unrealized PnL as a percentage of entry cost."""
        cost = self.entry_price * self.quantity
        if cost == Decimal("0"):
            return Decimal("0")
        return (self.unrealized_pnl / cost) * Decimal("100")

    @property
    def total_pnl(self) -> Decimal:
        """Return combined realized + unrealized PnL."""
        return self.realized_pnl + self.unrealized_pnl
