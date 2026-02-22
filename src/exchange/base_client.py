"""Abstract exchange client interface for AtoBot Trading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal


class BaseExchangeClient(ABC):
    """Abstract base class that every exchange adapter must implement."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish the connection / session to the exchange."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the connection."""
        ...

    @abstractmethod
    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Return the latest ticker price for *symbol*."""
        ...

    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 10) -> dict:
        """Return the top-of-book bids/asks for *symbol*."""
        ...

    @abstractmethod
    async def place_limit_order(
        self, symbol: str, side: str, price: Decimal, quantity: Decimal
    ) -> dict:
        """Place a limit order and return the exchange response."""
        ...

    @abstractmethod
    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal
    ) -> dict:
        """Place a market order and return the exchange response."""
        ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an open order by its exchange *order_id*."""
        ...

    @abstractmethod
    async def get_order_status(self, symbol: str, order_id: str) -> dict:
        """Return the current status of an order."""
        ...

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Return a list of all open orders (optionally filtered by symbol)."""
        ...

    @abstractmethod
    async def get_account_balance(self) -> dict[str, Decimal]:
        """Return mapping of asset -> free balance."""
        ...

    @abstractmethod
    async def get_klines(
        self, symbol: str, interval: str, limit: int = 100
    ) -> list:
        """Return historical kline/OHLCV data."""
        ...

    @abstractmethod
    async def get_exchange_info(self, symbol: str) -> dict:
        """Return raw exchange info for *symbol*."""
        ...

    @abstractmethod
    async def get_symbol_filters(self, symbol: str) -> dict:
        """Return parsed symbol filters.

        Expected keys::

            {
                "tick_size":    Decimal,  # price precision
                "step_size":    Decimal,  # quantity precision
                "min_notional": Decimal,  # minimum order value
                "min_qty":      Decimal,  # minimum quantity
            }
        """
        ...
