"""Binance exchange client implementation for AtoBot Trading."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException, BinanceRequestException
from loguru import logger

from src.config.settings import Settings
from src.exchange.base_client import BaseExchangeClient
from src.utils.helpers import decimal_from_str
from src.utils.retry import retry


class BinanceClientError(Exception):
    """Raised when a Binance API call fails."""

    def __init__(self, message: str, original: Exception | None = None):
        super().__init__(message)
        self.original = original


class BinanceClient(BaseExchangeClient):
    """Binance exchange adapter using python-binance AsyncClient."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncClient | None = None
        self._symbol_filters: dict[str, dict] = {}

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the AsyncClient and prefetch symbol filters."""
        try:
            if self._settings.BINANCE_TESTNET:
                self._client = await AsyncClient.create(
                    api_key=self._settings.BINANCE_API_KEY,
                    api_secret=self._settings.BINANCE_API_SECRET,
                    testnet=True,
                )
                logger.info("Connected to Binance TESTNET")
            else:
                self._client = await AsyncClient.create(
                    api_key=self._settings.BINANCE_API_KEY,
                    api_secret=self._settings.BINANCE_API_SECRET,
                )
                logger.info("Connected to Binance LIVE")

            # Prefetch symbol filters for configured pairs
            for symbol in self._settings.TRADING_PAIRS:
                try:
                    await self.get_symbol_filters(symbol)
                except Exception as exc:
                    logger.warning(
                        "Could not prefetch filters for {}: {}", symbol, exc
                    )
        except (BinanceAPIException, BinanceRequestException) as exc:
            raise BinanceClientError(
                f"Failed to connect to Binance: {exc}", original=exc
            ) from exc

    async def disconnect(self) -> None:
        """Close the async client session."""
        if self._client:
            await self._client.close_connection()
            self._client = None
            logger.info("Disconnected from Binance")

    def _ensure_connected(self) -> AsyncClient:
        """Return the live client or raise."""
        if self._client is None:
            raise BinanceClientError("BinanceClient is not connected. Call connect() first.")
        return self._client

    # ── Market Data ───────────────────────────────────────────────────────────

    @retry(max_attempts=3, delay=1.0, exceptions=(BinanceAPIException, BinanceRequestException, Exception))
    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Return latest ticker price."""
        client = self._ensure_connected()
        logger.debug("get_ticker_price({})", symbol)
        ticker = await client.get_symbol_ticker(symbol=symbol)
        return decimal_from_str(ticker["price"])

    @retry(max_attempts=3, delay=1.0, exceptions=(BinanceAPIException, BinanceRequestException, Exception))
    async def get_order_book(self, symbol: str, limit: int = 10) -> dict:
        """Return top-of-book bids + asks."""
        client = self._ensure_connected()
        logger.debug("get_order_book({}, limit={})", symbol, limit)
        book = await client.get_order_book(symbol=symbol, limit=limit)
        return {
            "bids": [
                {"price": decimal_from_str(b[0]), "quantity": decimal_from_str(b[1])}
                for b in book["bids"]
            ],
            "asks": [
                {"price": decimal_from_str(a[0]), "quantity": decimal_from_str(a[1])}
                for a in book["asks"]
            ],
        }

    @retry(max_attempts=3, delay=1.0, exceptions=(BinanceAPIException, BinanceRequestException, Exception))
    async def get_klines(self, symbol: str, interval: str, limit: int = 100) -> list:
        """Fetch historical klines."""
        client = self._ensure_connected()
        logger.debug("get_klines({}, {}, limit={})", symbol, interval, limit)
        raw = await client.get_klines(symbol=symbol, interval=interval, limit=limit)
        # Each kline: [open_time, open, high, low, close, volume, ...]
        result: list[dict[str, Any]] = []
        for k in raw:
            result.append(
                {
                    "timestamp": k[0],
                    "open": decimal_from_str(k[1]),
                    "high": decimal_from_str(k[2]),
                    "low": decimal_from_str(k[3]),
                    "close": decimal_from_str(k[4]),
                    "volume": decimal_from_str(k[5]),
                }
            )
        return result

    # ── Orders ────────────────────────────────────────────────────────────────

    @retry(max_attempts=3, delay=1.0, exceptions=(BinanceAPIException, BinanceRequestException, Exception))
    async def place_limit_order(
        self, symbol: str, side: str, price: Decimal, quantity: Decimal
    ) -> dict:
        """Place a limit order on Binance."""
        client = self._ensure_connected()
        logger.info(
            "place_limit_order | symbol={} side={} price={} qty={}",
            symbol,
            side,
            price,
            quantity,
        )
        try:
            response = await client.create_order(
                symbol=symbol,
                side=side.upper(),
                type="LIMIT",
                timeInForce="GTC",
                price=str(price),
                quantity=str(quantity),
            )
            logger.debug("Limit order response: {}", response)
            return response
        except (BinanceAPIException, BinanceRequestException) as exc:
            raise BinanceClientError(
                f"Failed to place limit order: {exc}", original=exc
            ) from exc

    @retry(max_attempts=3, delay=1.0, exceptions=(BinanceAPIException, BinanceRequestException, Exception))
    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal
    ) -> dict:
        """Place a market order on Binance."""
        client = self._ensure_connected()
        logger.info(
            "place_market_order | symbol={} side={} qty={}", symbol, side, quantity
        )
        try:
            response = await client.create_order(
                symbol=symbol,
                side=side.upper(),
                type="MARKET",
                quantity=str(quantity),
            )
            logger.debug("Market order response: {}", response)
            return response
        except (BinanceAPIException, BinanceRequestException) as exc:
            raise BinanceClientError(
                f"Failed to place market order: {exc}", original=exc
            ) from exc

    @retry(max_attempts=3, delay=1.0, exceptions=(BinanceAPIException, BinanceRequestException, Exception))
    async def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an open order."""
        client = self._ensure_connected()
        logger.info("cancel_order | symbol={} order_id={}", symbol, order_id)
        try:
            response = await client.cancel_order(symbol=symbol, orderId=int(order_id))
            return response
        except (BinanceAPIException, BinanceRequestException) as exc:
            raise BinanceClientError(
                f"Failed to cancel order {order_id}: {exc}", original=exc
            ) from exc

    @retry(max_attempts=3, delay=1.0, exceptions=(BinanceAPIException, BinanceRequestException, Exception))
    async def get_order_status(self, symbol: str, order_id: str) -> dict:
        """Fetch the status of a specific order."""
        client = self._ensure_connected()
        logger.debug("get_order_status | symbol={} order_id={}", symbol, order_id)
        try:
            response = await client.get_order(symbol=symbol, orderId=int(order_id))
            return response
        except (BinanceAPIException, BinanceRequestException) as exc:
            raise BinanceClientError(
                f"Failed to get order status: {exc}", original=exc
            ) from exc

    @retry(max_attempts=3, delay=1.0, exceptions=(BinanceAPIException, BinanceRequestException, Exception))
    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Return list of open orders."""
        client = self._ensure_connected()
        logger.debug("get_open_orders(symbol={})", symbol)
        kwargs: dict[str, Any] = {}
        if symbol:
            kwargs["symbol"] = symbol
        try:
            return await client.get_open_orders(**kwargs)
        except (BinanceAPIException, BinanceRequestException) as exc:
            raise BinanceClientError(
                f"Failed to get open orders: {exc}", original=exc
            ) from exc

    # ── Account ───────────────────────────────────────────────────────────────

    @retry(max_attempts=3, delay=1.0, exceptions=(BinanceAPIException, BinanceRequestException, Exception))
    async def get_account_balance(self) -> dict[str, Decimal]:
        """Return mapping of asset -> free balance (only non-zero)."""
        client = self._ensure_connected()
        logger.debug("get_account_balance()")
        try:
            info = await client.get_account()
            balances: dict[str, Decimal] = {}
            for b in info.get("balances", []):
                free = decimal_from_str(b["free"])
                if free > Decimal("0"):
                    balances[b["asset"]] = free
            return balances
        except (BinanceAPIException, BinanceRequestException) as exc:
            raise BinanceClientError(
                f"Failed to get account balance: {exc}", original=exc
            ) from exc

    # ── Exchange Info / Filters ───────────────────────────────────────────────

    @retry(max_attempts=3, delay=1.0, exceptions=(BinanceAPIException, BinanceRequestException, Exception))
    async def get_exchange_info(self, symbol: str) -> dict:
        """Return raw exchange info for a symbol."""
        client = self._ensure_connected()
        logger.debug("get_exchange_info({})", symbol)
        info = await client.get_exchange_info()
        for s in info.get("symbols", []):
            if s["symbol"] == symbol:
                return s
        raise BinanceClientError(f"Symbol {symbol} not found in exchange info")

    async def get_symbol_filters(self, symbol: str) -> dict:
        """Parse and cache symbol filters (tick_size, step_size, etc.)."""
        if symbol in self._symbol_filters:
            return self._symbol_filters[symbol]

        info = await self.get_exchange_info(symbol)
        filters: dict[str, Decimal] = {
            "tick_size": Decimal("0.01"),
            "step_size": Decimal("0.001"),
            "min_notional": Decimal("10"),
            "min_qty": Decimal("0.001"),
        }
        for f in info.get("filters", []):
            if f["filterType"] == "PRICE_FILTER":
                filters["tick_size"] = decimal_from_str(f["tickSize"])
            elif f["filterType"] == "LOT_SIZE":
                filters["step_size"] = decimal_from_str(f["stepSize"])
                filters["min_qty"] = decimal_from_str(f["minQty"])
            elif f["filterType"] == "NOTIONAL":
                filters["min_notional"] = decimal_from_str(f.get("minNotional", "10"))
            elif f["filterType"] == "MIN_NOTIONAL":
                filters["min_notional"] = decimal_from_str(f.get("minNotional", "10"))

        self._symbol_filters[symbol] = filters
        logger.debug("Symbol filters for {}: {}", symbol, filters)
        return filters
