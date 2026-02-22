"""Alpaca exchange client for AtoBot — **stock day trading**.

Uses the alpaca-py SDK for:
- Paper / live equity trading via ``TradingClient``
- Historical & real-time stock market data via ``StockHistoricalDataClient``

All order helpers follow the ``BaseExchangeClient`` interface so the rest
of the bot is exchange-agnostic.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from decimal import Decimal
from typing import Any

from loguru import logger

from src.config.settings import Settings
from src.exchange.base_client import BaseExchangeClient
from src.utils.retry import retry


class AlpacaClientError(Exception):
    """Raised when an Alpaca API call fails."""

    def __init__(self, message: str, original: Exception | None = None):
        super().__init__(message)
        self.original = original


class _RateLimiter:
    """Sliding-window rate limiter for Alpaca API calls.

    Alpaca free tier allows ~200 requests/minute. We default to 180 to
    leave headroom.
    """

    def __init__(self, max_calls: int = 180, window_seconds: float = 60.0) -> None:
        self._max_calls = max_calls
        self._window = window_seconds
        self._timestamps: deque[float] = deque()

    async def acquire(self) -> None:
        """Wait until a call slot is available."""
        now = time.monotonic()
        # Prune old entries outside the window
        while self._timestamps and self._timestamps[0] < now - self._window:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max_calls:
            sleep_for = self._window - (now - self._timestamps[0])
            if sleep_for > 0:
                logger.debug("Rate limiter: sleeping {:.1f}s", sleep_for)
                await asyncio.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


class AlpacaClient(BaseExchangeClient):
    """Alpaca exchange adapter for **equities** (stocks).

    Behaviour:
    * ``connect()`` creates ``TradingClient`` + ``StockHistoricalDataClient``
    * Market data comes from Alpaca's free IEX feed (or SIP if subscribed)
    * Orders use ``DAY`` time-in-force by default (day-trading)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._trading_client: Any | None = None
        self._data_client: Any | None = None
        self._symbol_filters: dict[str, dict] = {}
        self._rate_limiter = _RateLimiter(max_calls=180, window_seconds=60.0)

    # ── Connection ───────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create the Alpaca REST clients for stock trading."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical.stock import StockHistoricalDataClient

            api_key = self._settings.ALPACA_API_KEY
            api_secret = self._settings.ALPACA_API_SECRET
            paper = self._settings.ALPACA_PAPER

            self._trading_client = TradingClient(
                api_key=api_key,
                secret_key=api_secret,
                paper=paper,
            )
            self._data_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=api_secret,
            )

            # Verify connectivity
            account = self._trading_client.get_account()
            mode = "PAPER" if paper else "LIVE"
            logger.info(
                "Connected to Alpaca {} | equity=${} | buying_power=${} | "
                "day_trade_count={}",
                mode,
                account.equity,
                account.buying_power,
                account.daytrade_count,
            )
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to connect to Alpaca: {exc}", original=exc
            ) from exc

    async def disconnect(self) -> None:
        """Alpaca REST clients don't need explicit close."""
        self._trading_client = None
        self._data_client = None
        logger.info("Disconnected from Alpaca")

    def _ensure_connected(self) -> Any:
        """Return the live trading client or raise."""
        if self._trading_client is None:
            raise AlpacaClientError(
                "AlpacaClient is not connected. Call connect() first."
            )
        return self._trading_client

    # ── Market Data ──────────────────────────────────────────────────────────

    @retry(max_attempts=3, delay=1.0, exceptions=(Exception,))
    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Return latest trade price for a stock symbol (e.g. AAPL)."""
        from alpaca.data.requests import StockSnapshotRequest

        await self._rate_limiter.acquire()
        logger.debug("get_ticker_price({})", symbol)
        try:
            request = StockSnapshotRequest(symbol_or_symbols=symbol)
            snapshots = self._data_client.get_stock_snapshot(request)
            if isinstance(snapshots, dict):
                snap = snapshots.get(symbol)
            else:
                snap = snapshots
            if snap is None:
                raise AlpacaClientError(f"No snapshot for {symbol}")
            return Decimal(str(snap.latest_trade.price))
        except AlpacaClientError:
            raise
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to get ticker price for {symbol}: {exc}", original=exc
            ) from exc

    @retry(max_attempts=3, delay=1.0, exceptions=(Exception,))
    async def get_order_book(self, symbol: str, limit: int = 10) -> dict:
        """Return latest NBBO quote as a simplified order book."""
        from alpaca.data.requests import StockSnapshotRequest

        await self._rate_limiter.acquire()
        logger.debug("get_order_book({}, limit={})", symbol, limit)
        try:
            request = StockSnapshotRequest(symbol_or_symbols=symbol)
            snapshots = self._data_client.get_stock_snapshot(request)
            snap = snapshots.get(symbol) if isinstance(snapshots, dict) else snapshots
            if snap is None:
                return {"bids": [], "asks": []}
            quote = snap.latest_quote
            return {
                "bids": [
                    {
                        "price": Decimal(str(quote.bid_price)),
                        "quantity": Decimal(str(quote.bid_size)),
                    }
                ],
                "asks": [
                    {
                        "price": Decimal(str(quote.ask_price)),
                        "quantity": Decimal(str(quote.ask_size)),
                    }
                ],
            }
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to get order book for {symbol}: {exc}", original=exc
            ) from exc

    @retry(max_attempts=3, delay=1.0, exceptions=(Exception,))
    async def get_klines(
        self, symbol: str, interval: str, limit: int = 100
    ) -> list:
        """Fetch historical stock bars."""
        from datetime import datetime, timedelta, timezone
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        await self._rate_limiter.acquire()
        logger.debug("get_klines({}, {}, limit={})", symbol, interval, limit)

        tf_map: dict[str, TimeFrame] = {
            "1m": TimeFrame.Minute,
            "5m": TimeFrame(5, TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "30m": TimeFrame(30, TimeFrameUnit.Minute),
            "1h": TimeFrame.Hour,
            "4h": TimeFrame(4, TimeFrameUnit.Hour),
            "1d": TimeFrame.Day,
        }
        timeframe = tf_map.get(interval, TimeFrame.Minute)

        multipliers = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440,
        }
        minutes_per_bar = multipliers.get(interval, 1)
        start = datetime.now(timezone.utc) - timedelta(
            minutes=minutes_per_bar * limit * 1.5
        )

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                limit=limit,
            )
            bars_response = self._data_client.get_stock_bars(request)

            bars_list = (
                bars_response[symbol]
                if isinstance(bars_response, dict)
                else bars_response.data.get(symbol, [])
            )

            result: list[dict[str, Any]] = []
            for bar in bars_list[-limit:]:
                result.append(
                    {
                        "timestamp": int(bar.timestamp.timestamp() * 1000),
                        "open": Decimal(str(bar.open)),
                        "high": Decimal(str(bar.high)),
                        "low": Decimal(str(bar.low)),
                        "close": Decimal(str(bar.close)),
                        "volume": Decimal(str(bar.volume)),
                        "vwap": Decimal(str(bar.vwap)) if hasattr(bar, "vwap") and bar.vwap else None,
                    }
                )
            return result
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to get klines for {symbol}: {exc}", original=exc
            ) from exc

    # ── Orders ───────────────────────────────────────────────────────────────

    @retry(max_attempts=3, delay=1.0, exceptions=(Exception,))
    async def place_limit_order(
        self, symbol: str, side: str, price: Decimal, quantity: Decimal
    ) -> dict:
        """Place a limit order (DAY time-in-force for day trading)."""
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client = self._ensure_connected()
        await self._rate_limiter.acquire()
        logger.info(
            "place_limit_order | symbol={} side={} price={} qty={}",
            symbol, side, price, quantity,
        )
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            request = LimitOrderRequest(
                symbol=symbol,
                qty=float(quantity),
                side=order_side,
                type="limit",
                time_in_force=TimeInForce.DAY,
                limit_price=float(price),
            )
            order = client.submit_order(request)
            response = {
                "orderId": str(order.id),
                "clientOrderId": str(order.client_order_id),
                "status": order.status.value if hasattr(order.status, "value") else str(order.status),
                "symbol": symbol,
            }
            logger.debug("Limit order response: {}", response)
            return response
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to place limit order: {exc}", original=exc
            ) from exc

    @retry(max_attempts=3, delay=1.0, exceptions=(Exception,))
    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal
    ) -> dict:
        """Place a market order."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client = self._ensure_connected()
        await self._rate_limiter.acquire()
        logger.info(
            "place_market_order | symbol={} side={} qty={}",
            symbol, side, quantity,
        )
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            request = MarketOrderRequest(
                symbol=symbol,
                qty=float(quantity),
                side=order_side,
                type="market",
                time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(request)
            response = {
                "orderId": str(order.id),
                "clientOrderId": str(order.client_order_id),
                "status": order.status.value if hasattr(order.status, "value") else str(order.status),
                "symbol": symbol,
            }
            logger.debug("Market order response: {}", response)
            return response
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to place market order: {exc}", original=exc
            ) from exc

    @retry(max_attempts=3, delay=1.0, exceptions=(Exception,))
    async def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an open order by its Alpaca order ID."""
        client = self._ensure_connected()
        await self._rate_limiter.acquire()
        logger.info("cancel_order | order_id={}", order_id)
        try:
            client.cancel_order_by_id(order_id)
            return {"orderId": order_id, "status": "CANCELED"}
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to cancel order {order_id}: {exc}", original=exc
            ) from exc

    @retry(max_attempts=3, delay=1.0, exceptions=(Exception,))
    async def get_order_status(self, symbol: str, order_id: str) -> dict:
        """Fetch the status of a specific order."""
        client = self._ensure_connected()
        await self._rate_limiter.acquire()
        logger.debug("get_order_status | order_id={}", order_id)
        try:
            order = client.get_order_by_id(order_id)
            filled_qty = Decimal(str(order.filled_qty or 0))
            try:
                filled_avg_price = (
                    Decimal(str(order.filled_avg_price))
                    if order.filled_avg_price is not None
                    else None
                )
            except Exception:
                filled_avg_price = None
            return {
                "orderId": str(order.id),
                "status": order.status.value if hasattr(order.status, "value") else str(order.status),
                "executedQty": str(filled_qty),
                "filledAvgPrice": str(filled_avg_price) if filled_avg_price else None,
                "side": order.side.value if hasattr(order.side, "value") else str(order.side),
                "type": order.type.value if hasattr(order.type, "value") else str(order.type),
                "symbol": str(order.symbol),
            }
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to get order status: {exc}", original=exc
            ) from exc

    @retry(max_attempts=3, delay=1.0, exceptions=(Exception,))
    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Return list of all open orders."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        client = self._ensure_connected()
        await self._rate_limiter.acquire()
        logger.debug("get_open_orders(symbol={})", symbol)
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = client.get_orders(request)
            result: list[dict] = []
            for o in orders:
                o_symbol = str(o.symbol)
                if symbol and o_symbol != symbol:
                    continue
                result.append(
                    {
                        "orderId": str(o.id),
                        "symbol": o_symbol,
                        "side": o.side.value if hasattr(o.side, "value") else str(o.side),
                        "status": o.status.value if hasattr(o.status, "value") else str(o.status),
                        "executedQty": str(o.filled_qty or 0),
                    }
                )
            return result
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to get open orders: {exc}", original=exc
            ) from exc

    # ── Account ──────────────────────────────────────────────────────────────

    @retry(max_attempts=3, delay=1.0, exceptions=(Exception,))
    async def get_account_balance(self) -> dict[str, Decimal]:
        """Return account balances and equity positions."""
        client = self._ensure_connected()
        await self._rate_limiter.acquire()
        logger.debug("get_account_balance()")
        try:
            account = client.get_account()
            balances: dict[str, Decimal] = {
                "USD": Decimal(str(account.cash)),
                "EQUITY": Decimal(str(account.equity)),
                "BUYING_POWER": Decimal(str(account.buying_power)),
                "DAYTRADE_COUNT": Decimal(str(account.daytrade_count)),
            }

            # Add stock positions
            try:
                positions = client.get_all_positions()
                for pos in positions:
                    balances[str(pos.symbol)] = Decimal(str(pos.qty))
                    balances[f"{pos.symbol}_MKT_VALUE"] = Decimal(str(pos.market_value))
                    balances[f"{pos.symbol}_UNREALIZED_PL"] = Decimal(str(pos.unrealized_pl))
            except Exception:
                pass

            return balances
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to get account balance: {exc}", original=exc
            ) from exc

    # ── Exchange Info / Filters ──────────────────────────────────────────────

    @retry(max_attempts=3, delay=1.0, exceptions=(Exception,))
    async def get_exchange_info(self, symbol: str) -> dict:
        """Return asset info from Alpaca."""
        client = self._ensure_connected()
        await self._rate_limiter.acquire()
        logger.debug("get_exchange_info({})", symbol)
        try:
            asset = client.get_asset(symbol)
            return {
                "symbol": str(asset.symbol),
                "name": str(asset.name),
                "exchange": str(asset.exchange),
                "asset_class": str(asset.asset_class),
                "tradable": asset.tradable,
                "fractionable": asset.fractionable,
                "shortable": asset.shortable,
                "easy_to_borrow": asset.easy_to_borrow,
            }
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to get exchange info for {symbol}: {exc}", original=exc
            ) from exc

    async def get_symbol_filters(self, symbol: str) -> dict:
        """Return symbol filters for stocks."""
        if symbol in self._symbol_filters:
            return self._symbol_filters[symbol]

        try:
            info = await self.get_exchange_info(symbol)
            fractionable = info.get("fractionable", False)
            filters = {
                "tick_size": Decimal("0.01"),
                "step_size": Decimal("0.001") if fractionable else Decimal("1"),
                "min_notional": Decimal("1"),
                "min_qty": Decimal("0.001") if fractionable else Decimal("1"),
            }
        except Exception:
            logger.warning(
                "Could not fetch filters for {} — using stock defaults", symbol
            )
            filters = {
                "tick_size": Decimal("0.01"),
                "step_size": Decimal("1"),
                "min_notional": Decimal("1"),
                "min_qty": Decimal("1"),
            }

        self._symbol_filters[symbol] = filters
        logger.debug("Symbol filters for {}: {}", symbol, filters)
        return filters

    # ── Stock-specific helpers ───────────────────────────────────────────────

    async def get_account_info(self) -> dict:
        """Return full Alpaca account detail for dashboard / risk checks."""
        client = self._ensure_connected()
        account = client.get_account()
        return {
            "equity": Decimal(str(account.equity)),
            "cash": Decimal(str(account.cash)),
            "buying_power": Decimal(str(account.buying_power)),
            "portfolio_value": Decimal(str(account.portfolio_value)),
            "daytrade_count": int(account.daytrade_count),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
        }

    async def get_positions(self) -> list[dict]:
        """Return all open stock positions."""
        client = self._ensure_connected()
        positions = client.get_all_positions()
        return [
            {
                "symbol": str(p.symbol),
                "qty": Decimal(str(p.qty)),
                "avg_entry_price": Decimal(str(p.avg_entry_price)),
                "market_value": Decimal(str(p.market_value)),
                "unrealized_pl": Decimal(str(p.unrealized_pl)),
                "unrealized_plpc": Decimal(str(p.unrealized_plpc)),
                "current_price": Decimal(str(p.current_price)),
                "side": str(p.side),
            }
            for p in positions
        ]

    async def close_position(self, symbol: str) -> dict:
        """Close an entire position for a symbol (flatten)."""
        client = self._ensure_connected()
        logger.info("close_position({})", symbol)
        try:
            order = client.close_position(symbol)
            return {"orderId": str(order.id), "symbol": symbol, "status": "closing"}
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to close position for {symbol}: {exc}", original=exc
            ) from exc

    async def close_all_positions(self) -> list[dict]:
        """Close ALL open positions (end-of-day flatten)."""
        client = self._ensure_connected()
        logger.info("close_all_positions()")
        try:
            responses = client.close_all_positions(cancel_orders=True)
            return [
                {"symbol": str(getattr(r, "symbol", "unknown")), "status": "closing"}
                for r in responses
            ]
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to close all positions: {exc}", original=exc
            ) from exc

    async def is_market_open(self) -> bool:
        """Check if the stock market is currently open."""
        client = self._ensure_connected()
        clock = client.get_clock()
        return clock.is_open

    async def get_market_clock(self) -> dict:
        """Return market clock info."""
        client = self._ensure_connected()
        clock = client.get_clock()
        return {
            "is_open": clock.is_open,
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
            "timestamp": str(clock.timestamp),
        }
