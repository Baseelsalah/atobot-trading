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

    # ── Optional capabilities (default no-op for exchanges that lack them) ──

    async def start_streams(self, symbols: list[str]) -> None:
        """Start WebSocket streams for price/order/news data (optional)."""

    async def replace_order(
        self, order_id: str, *, quantity: Decimal | None = None, limit_price: Decimal | None = None
    ) -> dict:
        """Atomically replace an order (optional). Default raises."""
        raise NotImplementedError("replace_order not supported by this exchange")

    async def get_order_by_client_id(self, client_order_id: str) -> dict | None:
        """Look up an order by client-assigned ID (optional)."""
        return None

    async def get_market_calendar(self, start: str = "", end: str = "") -> list:
        """Return market calendar entries (optional)."""
        return []

    async def get_next_market_open(self):  # noqa: ANN201
        """Return the next market open datetime (optional)."""
        return None

    async def get_next_market_close(self):  # noqa: ANN201
        """Return the next market close datetime (optional)."""
        return None

    async def get_account_config(self) -> dict:
        """Return account configuration (PDT flag, margin, etc.)."""
        return {}

    async def get_portfolio_history(self, **kwargs) -> dict:
        """Return portfolio equity history (optional)."""
        return {}

    async def get_corporate_announcements(self, **kwargs) -> list:
        """Return corporate announcements (optional)."""
        return []

    def is_streaming(self) -> bool:
        """Whether the price data stream is connected."""
        return False

    def is_trade_streaming(self) -> bool:
        """Whether the trade-update stream is connected."""
        return False

    def is_symbol_halted(self, symbol: str) -> bool:
        """Whether a symbol is currently halted."""
        return False

    async def drain_trade_updates(self) -> list[dict]:
        """Return and clear all queued trade-update events."""
        return []

    async def drain_news_events(self) -> list[dict]:
        """Return and clear all queued news events."""
        return []
