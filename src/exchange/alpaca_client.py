"""Alpaca exchange client for AtoBot — **stock + crypto trading**.

Uses the alpaca-py SDK for:
- Paper / live equity trading via ``TradingClient``
- Historical & real-time stock market data via ``StockHistoricalDataClient``
- Historical & real-time crypto market data via ``CryptoHistoricalDataClient``
- **WebSocket streaming** via ``StockDataStream`` + ``CryptoDataStream``
- **Trade updates streaming** via ``TradingStream`` (order fills/rejects)
- **Market calendar** for schedule-aware sleep/wake
- **News streaming** via ``NewsDataStream`` (optional, for AI advisor)

Crypto symbols use ``/`` separator (e.g. ``BTC/USD``, ``ETH/USD``) and
route automatically to the crypto data pipeline.  Orders use ``GTC``
time-in-force (crypto markets are 24/7).

All order helpers follow the ``BaseExchangeClient`` interface so the rest
of the bot is exchange-agnostic.
"""

from __future__ import annotations

import asyncio
import time
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Coroutine

from loguru import logger

from src.config.settings import Settings
from src.exchange.base_client import BaseExchangeClient


class AlpacaClientError(Exception):
    """Raised when an Alpaca API call fails."""

    def __init__(self, message: str, original: Exception | None = None):
        super().__init__(message)
        self.original = original


class _RateLimiter:
    """Sliding-window rate limiter for Alpaca REST calls (backup only).

    The SDK handles 429 retries natively.  We keep a lighter limiter
    (raised from 180 → 190) as a courtesy throttle to avoid hitting the
    SDK retry path in normal operation.
    """

    def __init__(self, max_calls: int = 190, window_seconds: float = 60.0) -> None:
        self._max_calls = max_calls
        self._window = window_seconds
        self._timestamps: deque[float] = deque()

    async def acquire(self) -> None:
        now = time.monotonic()
        while self._timestamps and self._timestamps[0] < now - self._window:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max_calls:
            sleep_for = self._window - (now - self._timestamps[0])
            if sleep_for > 0:
                logger.debug("Rate limiter: sleeping {:.1f}s", sleep_for)
                await asyncio.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


# ── Streaming helpers ─────────────────────────────────────────────────────────

class StreamingPriceCache:
    """Thread-safe cache of latest prices pushed by WebSocket.

    When the ``StockDataStream`` is running it pushes trade events here.
    ``get_ticker_price`` reads from the cache instead of making a REST call.
    """

    def __init__(self) -> None:
        self._prices: dict[str, Decimal] = {}
        self._lock = threading.Lock()
        self._updated_at: dict[str, float] = {}

    def update(self, symbol: str, price: Decimal) -> None:
        with self._lock:
            self._prices[symbol] = price
            self._updated_at[symbol] = time.monotonic()

    def get(self, symbol: str, max_age: float = 30.0) -> Decimal | None:
        """Return cached price if fresh enough, else None."""
        with self._lock:
            ts = self._updated_at.get(symbol, 0.0)
            if time.monotonic() - ts > max_age:
                return None
            return self._prices.get(symbol)

    def get_all(self) -> dict[str, Decimal]:
        with self._lock:
            return dict(self._prices)


class StreamingBarCache:
    """Thread-safe cache of latest 1-min bars from WebSocket."""

    def __init__(self) -> None:
        self._bars: dict[str, dict] = {}  # symbol -> latest bar dict
        self._lock = threading.Lock()

    def update(self, symbol: str, bar: dict) -> None:
        with self._lock:
            self._bars[symbol] = bar

    def get(self, symbol: str) -> dict | None:
        with self._lock:
            return self._bars.get(symbol)


class HaltedSymbolTracker:
    """Thread-safe tracker for halted/paused symbols."""

    def __init__(self) -> None:
        self._halted: set[str] = set()
        self._lock = threading.Lock()

    def set_halted(self, symbol: str) -> None:
        with self._lock:
            self._halted.add(symbol)
            logger.warning("HALT detected: {} — trading paused for this symbol", symbol)

    def set_resumed(self, symbol: str) -> None:
        with self._lock:
            self._halted.discard(symbol)
            logger.info("RESUME detected: {} — trading resumed", symbol)

    def is_halted(self, symbol: str) -> bool:
        with self._lock:
            return symbol in self._halted

    def get_all_halted(self) -> set[str]:
        with self._lock:
            return set(self._halted)


class TradeUpdateQueue:
    """Thread-safe queue for trade/order update events from TradingStream."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict] = asyncio.Queue()

    def put_nowait(self, event: dict) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Trade update queue full — dropping event")

    async def get(self, timeout: float = 0.1) -> dict | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def qsize(self) -> int:
        return self._queue.qsize()


class NewsEventQueue:
    """Thread-safe queue for real-time news events."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)

    def put_nowait(self, event: dict) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop oldest news silently

    async def get(self, timeout: float = 0.1) -> dict | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


# ── Main Client ───────────────────────────────────────────────────────────────

class AlpacaClient(BaseExchangeClient):
    """Alpaca exchange adapter for **equities + crypto**.

    Supports two operational modes:
    - **Polling** (legacy): REST calls every N seconds.
    - **Streaming** (new, preferred): WebSocket for prices, order events,
      halt detection, and news. Falls back to REST when needed.

    Crypto symbols (containing ``/``) automatically route to the crypto
    data pipeline (``CryptoHistoricalDataClient``, ``CryptoDataStream``)
    and use ``GTC`` time-in-force instead of ``DAY``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._trading_client: Any | None = None
        self._data_client: Any | None = None
        self._crypto_data_client: Any | None = None   # CryptoHistoricalDataClient
        self._symbol_filters: dict[str, dict] = {}
        self._rate_limiter = _RateLimiter(max_calls=190, window_seconds=60.0)

        # ── Streaming infrastructure ──────────────────────────────────────
        self._stock_stream: Any | None = None       # StockDataStream
        self._crypto_stream: Any | None = None      # CryptoDataStream
        self._trading_stream: Any | None = None      # TradingStream
        self._news_stream: Any | None = None         # NewsDataStream
        self._stream_thread: threading.Thread | None = None
        self._crypto_stream_thread: threading.Thread | None = None
        self._trade_stream_thread: threading.Thread | None = None
        self._news_stream_thread: threading.Thread | None = None

        # Shared thread-safe caches
        self.price_cache = StreamingPriceCache()
        self.bar_cache = StreamingBarCache()
        self.halt_tracker = HaltedSymbolTracker()
        self.trade_updates = TradeUpdateQueue()
        self.news_events = NewsEventQueue()

        # Market calendar cache
        self._calendar_cache: list[dict] | None = None
        self._calendar_fetched_at: float = 0.0

        # Corporate announcements cache
        self._announcements_cache: list[dict] = []
        self._announcements_fetched_at: float = 0.0

        # Streaming enabled flags (from settings)
        self._streaming_enabled = getattr(settings, "STREAMING_ENABLED", True)
        self._trade_stream_enabled = getattr(settings, "TRADE_STREAM_ENABLED", True)
        self._news_stream_enabled = getattr(settings, "NEWS_STREAM_ENABLED", False)
        self._data_feed = getattr(settings, "DATA_FEED", "iex")

    # ── Connection ───────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create Alpaca REST clients + start WebSocket streams."""
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

            # ── Crypto data client (separate from stock data) ─────────────
            try:
                from alpaca.data.historical.crypto import CryptoHistoricalDataClient

                self._crypto_data_client = CryptoHistoricalDataClient(
                    api_key=api_key,
                    secret_key=api_secret,
                )
                logger.info("CryptoHistoricalDataClient initialized")
            except Exception as exc:
                logger.warning("Could not init CryptoHistoricalDataClient: {}", exc)
                self._crypto_data_client = None

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

            # Pre-load market calendar for the next 7 days
            await self._load_market_calendar()

        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to connect to Alpaca: {exc}", original=exc
            ) from exc

    async def start_streams(self, symbols: list[str]) -> None:
        """Start all WebSocket streams (call after connect()).

        Launched in background threads so they don't block the async engine.
        Stock and crypto symbols are separated and routed to their respective
        WebSocket endpoints.
        """
        api_key = self._settings.ALPACA_API_KEY
        api_secret = self._settings.ALPACA_API_SECRET

        # Split symbols by asset class
        stock_symbols = [s for s in symbols if not self._is_crypto(s)]
        crypto_symbols = [s for s in symbols if self._is_crypto(s)]

        if self._streaming_enabled and stock_symbols:
            await self._start_stock_stream(stock_symbols, api_key, api_secret)

        if self._streaming_enabled and crypto_symbols:
            await self._start_crypto_stream(crypto_symbols, api_key, api_secret)

        if self._trade_stream_enabled:
            await self._start_trading_stream(api_key, api_secret)

        if self._news_stream_enabled and stock_symbols:
            await self._start_news_stream(stock_symbols, api_key, api_secret)

    @staticmethod
    def _is_crypto(symbol: str) -> bool:
        """Return True if the symbol is a crypto trading pair (e.g. BTC/USD)."""
        return "/" in symbol

    async def _start_stock_stream(
        self, symbols: list[str], api_key: str, api_secret: str
    ) -> None:
        """Start the StockDataStream in a background thread."""
        try:
            from alpaca.data.live.stock import StockDataStream
            from alpaca.data.enums import DataFeed

            feed = DataFeed.SIP if self._data_feed.lower() == "sip" else DataFeed.IEX

            self._stock_stream = StockDataStream(
                api_key=api_key,
                secret_key=api_secret,
                feed=feed,
                websocket_params={
                    "ping_interval": 10,
                    "ping_timeout": 180,
                    "max_queue": 1024,
                },
            )

            # Handler: real-time trade events → price cache
            async def _on_trade(data):
                try:
                    symbol = str(data.symbol)
                    price = Decimal(str(data.price))
                    self.price_cache.update(symbol, price)
                except Exception as exc:
                    logger.debug("Stream trade handler error: {}", exc)

            # Handler: 1-min bar events → bar cache
            async def _on_bar(data):
                try:
                    symbol = str(data.symbol)
                    self.bar_cache.update(symbol, {
                        "timestamp": int(data.timestamp.timestamp() * 1000),
                        "open": Decimal(str(data.open)),
                        "high": Decimal(str(data.high)),
                        "low": Decimal(str(data.low)),
                        "close": Decimal(str(data.close)),
                        "volume": Decimal(str(data.volume)),
                    })
                except Exception as exc:
                    logger.debug("Stream bar handler error: {}", exc)

            # Handler: trading status (halts/resumes)
            async def _on_status(data):
                try:
                    symbol = str(data.symbol)
                    status_code = str(getattr(data, "status_code", "")).upper()
                    if status_code in ("H", "T"):  # Halted / Trading pause
                        self.halt_tracker.set_halted(symbol)
                    elif status_code in ("Q", "R", "B"):  # Quotation resumed / Trading resumed / Book
                        self.halt_tracker.set_resumed(symbol)
                    else:
                        logger.debug("Trading status {} for {}: {}", status_code, symbol, data)
                except Exception as exc:
                    logger.debug("Stream status handler error: {}", exc)

            self._stock_stream.subscribe_trades(_on_trade, *symbols)
            self._stock_stream.subscribe_bars(_on_bar, *symbols)
            self._stock_stream.subscribe_trading_statuses(_on_status, *symbols)

            def _run_stream():
                try:
                    self._stock_stream.run()
                except Exception as exc:
                    logger.error("StockDataStream crashed: {}", exc)

            self._stream_thread = threading.Thread(
                target=_run_stream, daemon=True, name="alpaca-data-stream"
            )
            self._stream_thread.start()
            logger.info(
                "StockDataStream started | feed={} | symbols={}",
                feed.value, symbols,
            )

        except Exception as exc:
            logger.warning(
                "Could not start StockDataStream (falling back to polling): {}", exc
            )
            self._streaming_enabled = False

    async def _start_crypto_stream(
        self, symbols: list[str], api_key: str, api_secret: str
    ) -> None:
        """Start the CryptoDataStream in a background thread for crypto symbols."""
        try:
            from alpaca.data.live.crypto import CryptoDataStream

            self._crypto_stream = CryptoDataStream(
                api_key=api_key,
                secret_key=api_secret,
                websocket_params={
                    "ping_interval": 10,
                    "ping_timeout": 180,
                    "max_queue": 1024,
                },
            )

            # Handler: real-time crypto trade events → price cache
            async def _on_crypto_trade(data):
                try:
                    symbol = str(data.symbol)
                    price = Decimal(str(data.price))
                    self.price_cache.update(symbol, price)
                except Exception as exc:
                    logger.debug("Crypto stream trade handler error: {}", exc)

            # Handler: crypto bar events → bar cache
            async def _on_crypto_bar(data):
                try:
                    symbol = str(data.symbol)
                    self.bar_cache.update(symbol, {
                        "timestamp": int(data.timestamp.timestamp() * 1000),
                        "open": Decimal(str(data.open)),
                        "high": Decimal(str(data.high)),
                        "low": Decimal(str(data.low)),
                        "close": Decimal(str(data.close)),
                        "volume": Decimal(str(data.volume)),
                    })
                except Exception as exc:
                    logger.debug("Crypto stream bar handler error: {}", exc)

            self._crypto_stream.subscribe_trades(_on_crypto_trade, *symbols)
            self._crypto_stream.subscribe_bars(_on_crypto_bar, *symbols)

            def _run_crypto_stream():
                try:
                    self._crypto_stream.run()
                except Exception as exc:
                    logger.error("CryptoDataStream crashed: {}", exc)

            self._crypto_stream_thread = threading.Thread(
                target=_run_crypto_stream, daemon=True, name="alpaca-crypto-stream"
            )
            self._crypto_stream_thread.start()
            logger.info(
                "CryptoDataStream started | symbols={}",
                symbols,
            )

        except Exception as exc:
            logger.warning(
                "Could not start CryptoDataStream (falling back to polling): {}", exc
            )

    async def _start_trading_stream(self, api_key: str, api_secret: str) -> None:
        """Start the TradingStream for order fill/reject events."""
        try:
            from alpaca.trading.stream import TradingStream

            self._trading_stream = TradingStream(
                api_key=api_key,
                secret_key=api_secret,
                paper=self._settings.ALPACA_PAPER,
                websocket_params={
                    "ping_interval": 10,
                    "ping_timeout": 180,
                    "max_queue": 512,
                },
            )

            async def _on_trade_update(data):
                try:
                    event = data.event if hasattr(data, "event") else str(data.get("event", ""))
                    order = data.order if hasattr(data, "order") else data.get("order", {})

                    update = {
                        "event": str(event),
                        "order_id": str(getattr(order, "id", "")),
                        "client_order_id": str(getattr(order, "client_order_id", "")),
                        "symbol": str(getattr(order, "symbol", "")),
                        "side": str(getattr(order, "side", "")),
                        "status": str(getattr(order, "status", "")),
                        "filled_qty": str(getattr(order, "filled_qty", "0")),
                        "filled_avg_price": str(getattr(order, "filled_avg_price", "")),
                        "qty": str(getattr(order, "qty", "0")),
                        "type": str(getattr(order, "type", "")),
                    }
                    self.trade_updates.put_nowait(update)
                    logger.info(
                        "Trade update: {} {} {} qty={} @ {}",
                        update["event"], update["side"],
                        update["symbol"], update["filled_qty"],
                        update["filled_avg_price"],
                    )
                except Exception as exc:
                    logger.debug("Trade update handler error: {}", exc)

            self._trading_stream.subscribe_trade_updates(_on_trade_update)

            def _run_trade_stream():
                try:
                    self._trading_stream.run()
                except Exception as exc:
                    logger.error("TradingStream crashed: {}", exc)

            self._trade_stream_thread = threading.Thread(
                target=_run_trade_stream, daemon=True, name="alpaca-trade-stream"
            )
            self._trade_stream_thread.start()
            logger.info("TradingStream started (order fill/reject events)")

        except Exception as exc:
            logger.warning(
                "Could not start TradingStream (falling back to polling): {}", exc
            )
            self._trade_stream_enabled = False

    async def _start_news_stream(
        self, symbols: list[str], api_key: str, api_secret: str
    ) -> None:
        """Start the NewsDataStream for real-time news events."""
        try:
            from alpaca.data.live.news import NewsDataStream

            self._news_stream = NewsDataStream(
                api_key=api_key,
                secret_key=api_secret,
            )

            async def _on_news(data):
                try:
                    news_item = {
                        "headline": str(getattr(data, "headline", "")),
                        "summary": str(getattr(data, "summary", "")),
                        "source": str(getattr(data, "source", "")),
                        "symbols": [str(s) for s in getattr(data, "symbols", [])],
                        "timestamp": str(getattr(data, "created_at", "")),
                        "url": str(getattr(data, "url", "")),
                    }
                    self.news_events.put_nowait(news_item)
                    logger.info(
                        "News: [{}] {} ({})",
                        ", ".join(news_item["symbols"]),
                        news_item["headline"][:80],
                        news_item["source"],
                    )
                except Exception as exc:
                    logger.debug("News handler error: {}", exc)

            self._news_stream.subscribe_news(_on_news, *symbols)

            def _run_news_stream():
                try:
                    self._news_stream.run()
                except Exception as exc:
                    logger.error("NewsDataStream crashed: {}", exc)

            self._news_stream_thread = threading.Thread(
                target=_run_news_stream, daemon=True, name="alpaca-news-stream"
            )
            self._news_stream_thread.start()
            logger.info("NewsDataStream started | symbols={}", symbols)

        except Exception as exc:
            logger.warning("Could not start NewsDataStream: {}", exc)
            self._news_stream_enabled = False

    async def disconnect(self) -> None:
        """Shut down all streams and REST clients."""
        # Stop WebSocket streams
        for stream, name in [
            (self._stock_stream, "StockDataStream"),
            (self._crypto_stream, "CryptoDataStream"),
            (self._trading_stream, "TradingStream"),
            (self._news_stream, "NewsDataStream"),
        ]:
            if stream is not None:
                try:
                    stream.stop()
                    logger.info("{} stopped", name)
                except Exception as exc:
                    logger.debug("Error stopping {}: {}", name, exc)

        # Wait for threads to finish
        for thread in [
            self._stream_thread, self._crypto_stream_thread,
            self._trade_stream_thread, self._news_stream_thread,
        ]:
            if thread is not None and thread.is_alive():
                thread.join(timeout=5)

        self._trading_client = None
        self._data_client = None
        self._crypto_data_client = None
        self._stock_stream = None
        self._crypto_stream = None
        self._trading_stream = None
        self._news_stream = None
        logger.info("Disconnected from Alpaca")

    def _ensure_connected(self) -> Any:
        """Return the live trading client or raise."""
        if self._trading_client is None:
            raise AlpacaClientError(
                "AlpacaClient is not connected. Call connect() first."
            )
        return self._trading_client

    # ── Market Data ──────────────────────────────────────────────────────────

    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Return latest trade price — from stream cache or REST fallback.

        Crypto symbols route to ``CryptoHistoricalDataClient``.
        Priority: streaming cache → latest_trade REST call.
        """
        # Try streaming cache first (sub-ms)
        cached = self.price_cache.get(symbol, max_age=30.0)
        if cached is not None:
            logger.debug("get_ticker_price({}) = {} [from stream cache]", symbol, cached)
            return cached

        await self._rate_limiter.acquire()
        logger.debug("get_ticker_price({}) [REST fallback]", symbol)

        if self._is_crypto(symbol):
            # ── Crypto REST fallback ──────────────────────────────────
            try:
                from alpaca.data.requests import CryptoLatestTradeRequest

                if self._crypto_data_client is None:
                    raise AlpacaClientError("CryptoHistoricalDataClient not initialized")
                request = CryptoLatestTradeRequest(symbol_or_symbols=symbol)
                trades = self._crypto_data_client.get_crypto_latest_trade(request)
                if isinstance(trades, dict):
                    trade = trades.get(symbol)
                else:
                    trade = trades
                if trade is None:
                    raise AlpacaClientError(f"No latest crypto trade for {symbol}")
                price = Decimal(str(trade.price))
                self.price_cache.update(symbol, price)
                return price
            except AlpacaClientError:
                raise
            except Exception as exc:
                raise AlpacaClientError(
                    f"Failed to get crypto ticker price for {symbol}: {exc}", original=exc
                ) from exc
        else:
            # ── Stock REST fallback ───────────────────────────────────
            from alpaca.data.requests import StockLatestTradeRequest

            try:
                request = StockLatestTradeRequest(symbol_or_symbols=symbol)
                trades = self._data_client.get_stock_latest_trade(request)
                if isinstance(trades, dict):
                    trade = trades.get(symbol)
                else:
                    trade = trades
                if trade is None:
                    raise AlpacaClientError(f"No latest trade for {symbol}")
                price = Decimal(str(trade.price))
                self.price_cache.update(symbol, price)
                return price
            except AlpacaClientError:
                raise
            except Exception as exc:
                raise AlpacaClientError(
                    f"Failed to get ticker price for {symbol}: {exc}", original=exc
                ) from exc

    async def get_order_book(self, symbol: str, limit: int = 10) -> dict:
        """Return latest quote — NBBO for stocks, latest quote for crypto."""
        await self._rate_limiter.acquire()
        logger.debug("get_order_book({}, limit={})", symbol, limit)

        if self._is_crypto(symbol):
            try:
                from alpaca.data.requests import CryptoLatestQuoteRequest

                if self._crypto_data_client is None:
                    return {"bids": [], "asks": []}
                request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self._crypto_data_client.get_crypto_latest_quote(request)
                quote = quotes.get(symbol) if isinstance(quotes, dict) else quotes
                if quote is None:
                    return {"bids": [], "asks": []}
                return {
                    "bids": [{"price": Decimal(str(quote.bid_price)), "quantity": Decimal(str(quote.bid_size))}],
                    "asks": [{"price": Decimal(str(quote.ask_price)), "quantity": Decimal(str(quote.ask_size))}],
                }
            except Exception as exc:
                raise AlpacaClientError(
                    f"Failed to get crypto order book for {symbol}: {exc}", original=exc
                ) from exc
        else:
            from alpaca.data.requests import StockLatestQuoteRequest

            try:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quotes = self._data_client.get_stock_latest_quote(request)
                quote = quotes.get(symbol) if isinstance(quotes, dict) else quotes
                if quote is None:
                    return {"bids": [], "asks": []}
                return {
                    "bids": [{"price": Decimal(str(quote.bid_price)), "quantity": Decimal(str(quote.bid_size))}],
                    "asks": [{"price": Decimal(str(quote.ask_price)), "quantity": Decimal(str(quote.ask_size))}],
                }
            except Exception as exc:
                raise AlpacaClientError(
                    f"Failed to get order book for {symbol}: {exc}", original=exc
                ) from exc

    async def get_klines(
        self, symbol: str, interval: str, limit: int = 100
    ) -> list:
        """Fetch historical bars — routes to stock or crypto data client."""
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
            "1D": TimeFrame.Day,
        }
        timeframe = tf_map.get(interval, TimeFrame.Minute)

        multipliers = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440, "1D": 1440,
        }
        minutes_per_bar = multipliers.get(interval, 1)
        start = datetime.now(timezone.utc) - timedelta(
            minutes=minutes_per_bar * limit * 1.5
        )

        if self._is_crypto(symbol):
            return await self._get_crypto_klines(symbol, timeframe, start, limit)
        else:
            return await self._get_stock_klines(symbol, timeframe, start, limit)

    async def _get_stock_klines(
        self, symbol: str, timeframe: Any, start: datetime, limit: int
    ) -> list:
        """Fetch historical stock bars."""
        from alpaca.data.requests import StockBarsRequest

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

    async def _get_crypto_klines(
        self, symbol: str, timeframe: Any, start: datetime, limit: int
    ) -> list:
        """Fetch historical crypto bars via CryptoHistoricalDataClient."""
        from alpaca.data.requests import CryptoBarsRequest

        if self._crypto_data_client is None:
            raise AlpacaClientError("CryptoHistoricalDataClient not initialized")

        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                limit=limit,
            )
            bars_response = self._crypto_data_client.get_crypto_bars(request)

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
                f"Failed to get crypto klines for {symbol}: {exc}", original=exc
            ) from exc

    # ── Orders ───────────────────────────────────────────────────────────────

    async def place_limit_order(
        self, symbol: str, side: str, price: Decimal, quantity: Decimal
    ) -> dict:
        """Place a limit order (DAY for stocks, GTC for crypto)."""
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client = self._ensure_connected()
        await self._rate_limiter.acquire()
        tif = TimeInForce.GTC if self._is_crypto(symbol) else TimeInForce.DAY
        logger.info(
            "place_limit_order | symbol={} side={} price={} qty={} tif={}",
            symbol, side, price, quantity, tif,
        )
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            request = LimitOrderRequest(
                symbol=symbol,
                qty=float(quantity),
                side=order_side,
                type="limit",
                time_in_force=tif,
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

    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal
    ) -> dict:
        """Place a market order (GTC for crypto, DAY for stocks)."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        client = self._ensure_connected()
        await self._rate_limiter.acquire()
        tif = TimeInForce.GTC if self._is_crypto(symbol) else TimeInForce.DAY
        logger.info(
            "place_market_order | symbol={} side={} qty={} tif={}",
            symbol, side, quantity, tif,
        )
        try:
            order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            request = MarketOrderRequest(
                symbol=symbol,
                qty=float(quantity),
                side=order_side,
                type="market",
                time_in_force=tif,
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

    async def get_exchange_info(self, symbol: str) -> dict:
        """Return asset info from Alpaca (stocks and crypto)."""
        client = self._ensure_connected()
        await self._rate_limiter.acquire()
        logger.debug("get_exchange_info({})", symbol)
        try:
            asset = client.get_asset(symbol)
            info = {
                "symbol": str(asset.symbol),
                "name": str(asset.name),
                "exchange": str(asset.exchange),
                "asset_class": str(asset.asset_class),
                "tradable": asset.tradable,
                "fractionable": asset.fractionable,
                "shortable": getattr(asset, "shortable", False),
                "easy_to_borrow": getattr(asset, "easy_to_borrow", False),
            }
            # Crypto assets have min_order_size and min_trade_increment
            if hasattr(asset, "min_order_size") and asset.min_order_size:
                info["min_order_size"] = str(asset.min_order_size)
            if hasattr(asset, "min_trade_increment") and asset.min_trade_increment:
                info["min_trade_increment"] = str(asset.min_trade_increment)
            if hasattr(asset, "price_increment") and asset.price_increment:
                info["price_increment"] = str(asset.price_increment)
            return info
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to get exchange info for {symbol}: {exc}", original=exc
            ) from exc

    async def get_symbol_filters(self, symbol: str) -> dict:
        """Return symbol filters for stocks or crypto."""
        if symbol in self._symbol_filters:
            return self._symbol_filters[symbol]

        if self._is_crypto(symbol):
            # Crypto: always fractionable, min order sizes from Alpaca
            try:
                info = await self.get_exchange_info(symbol)
                min_order = Decimal(str(info.get("min_order_size", "0.0001")))
                min_increment = Decimal(str(info.get("min_trade_increment", "0.0001")))
                price_increment = Decimal(str(info.get("price_increment", "0.01")))
                filters = {
                    "tick_size": price_increment,
                    "step_size": min_increment,
                    "min_notional": Decimal("1"),
                    "min_qty": min_order,
                }
            except Exception:
                logger.warning("Could not fetch crypto filters for {} — using defaults", symbol)
                filters = {
                    "tick_size": Decimal("0.01"),
                    "step_size": Decimal("0.0001"),
                    "min_notional": Decimal("1"),
                    "min_qty": Decimal("0.0001"),
                }
        else:
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

    # ── New SDK Features ─────────────────────────────────────────────────────

    async def replace_order(
        self, order_id: str, qty: Decimal | None = None,
        limit_price: Decimal | None = None,
    ) -> dict:
        """Atomic order modification (no cancel+resubmit race condition).

        Uses ``replace_order_by_id()`` from the SDK.
        """
        from alpaca.trading.requests import ReplaceOrderRequest

        client = self._ensure_connected()
        await self._rate_limiter.acquire()
        logger.info("replace_order | id={} qty={} price={}", order_id, qty, limit_price)
        try:
            request_params: dict[str, Any] = {}
            if qty is not None:
                request_params["qty"] = float(qty)
            if limit_price is not None:
                request_params["limit_price"] = float(limit_price)

            request = ReplaceOrderRequest(**request_params)
            order = client.replace_order_by_id(order_id, request)
            return {
                "orderId": str(order.id),
                "clientOrderId": str(order.client_order_id),
                "status": order.status.value if hasattr(order.status, "value") else str(order.status),
                "symbol": str(order.symbol),
            }
        except Exception as exc:
            raise AlpacaClientError(
                f"Failed to replace order {order_id}: {exc}", original=exc
            ) from exc

    async def get_order_by_client_id(self, client_order_id: str) -> dict | None:
        """Look up an order by our own client_order_id (crash recovery)."""
        client = self._ensure_connected()
        await self._rate_limiter.acquire()
        try:
            order = client.get_order_by_client_id(client_order_id)
            return {
                "orderId": str(order.id),
                "clientOrderId": str(order.client_order_id),
                "symbol": str(order.symbol),
                "side": order.side.value if hasattr(order.side, "value") else str(order.side),
                "status": order.status.value if hasattr(order.status, "value") else str(order.status),
                "filled_qty": str(order.filled_qty or 0),
                "filled_avg_price": str(order.filled_avg_price) if order.filled_avg_price else None,
            }
        except Exception:
            return None

    async def _load_market_calendar(self, days_ahead: int = 7) -> None:
        """Pre-load market calendar for the next N days."""
        client = self._ensure_connected()
        try:
            from alpaca.trading.requests import GetCalendarRequest

            start = datetime.now(timezone.utc).date()
            end = start + timedelta(days=days_ahead)
            request = GetCalendarRequest(start=str(start), end=str(end))
            calendar = client.get_calendar(request)

            self._calendar_cache = [
                {
                    "date": str(day.date),
                    "open": str(day.open),
                    "close": str(day.close),
                }
                for day in calendar
            ]
            self._calendar_fetched_at = time.monotonic()
            logger.info(
                "Market calendar loaded: {} trading days ahead",
                len(self._calendar_cache),
            )
        except Exception as exc:
            logger.warning("Could not load market calendar: {}", exc)
            self._calendar_cache = None

    async def get_market_calendar(self) -> list[dict]:
        """Return cached market calendar (auto-refreshes daily)."""
        age = time.monotonic() - self._calendar_fetched_at
        if self._calendar_cache is None or age > 86400:  # Refresh every 24h
            await self._load_market_calendar()
        return self._calendar_cache or []

    async def get_next_market_open(self) -> datetime | None:
        """Return the next market open time from calendar cache."""
        calendar = await self.get_market_calendar()
        if not calendar:
            return None
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
            now = datetime.now(et)
            for day in calendar:
                open_str = f"{day['date']} {day['open']}"
                open_dt = datetime.strptime(open_str, "%Y-%m-%d %H:%M").replace(tzinfo=et)
                if open_dt > now:
                    return open_dt
        except Exception as exc:
            logger.debug("get_next_market_open error: {}", exc)
        return None

    async def get_next_market_close(self) -> datetime | None:
        """Return the next market close time from calendar cache."""
        calendar = await self.get_market_calendar()
        if not calendar:
            return None
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
            now = datetime.now(et)
            for day in calendar:
                close_str = f"{day['date']} {day['close']}"
                close_dt = datetime.strptime(close_str, "%Y-%m-%d %H:%M").replace(tzinfo=et)
                if close_dt > now:
                    return close_dt
        except Exception as exc:
            logger.debug("get_next_market_close error: {}", exc)
        return None

    async def get_account_config(self) -> dict:
        """Read Alpaca account configuration (PDT, margin, shorting settings)."""
        client = self._ensure_connected()
        try:
            config = client.get_account_configurations()
            return {
                "dtbp_check": str(getattr(config, "dtbp_check", "")),
                "no_shorting": getattr(config, "no_shorting", False),
                "suspend_trade": getattr(config, "suspend_trade", False),
                "trade_confirm_email": str(getattr(config, "trade_confirm_email", "")),
                "fractional_trading": getattr(config, "fractional_trading", True),
                "pdt_check": str(getattr(config, "pdt_check", "")),
            }
        except Exception as exc:
            logger.warning("Could not get account config: {}", exc)
            return {}

    async def get_portfolio_history(
        self, period: str = "1W", timeframe: str = "1D"
    ) -> dict:
        """Return native Alpaca portfolio performance history."""
        client = self._ensure_connected()
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest

            request = GetPortfolioHistoryRequest(
                period=period,
                timeframe=timeframe,
            )
            history = client.get_portfolio_history(request)
            return {
                "equity": [float(e) for e in history.equity] if history.equity else [],
                "profit_loss": [float(p) for p in history.profit_loss] if history.profit_loss else [],
                "profit_loss_pct": [float(p) for p in history.profit_loss_pct] if history.profit_loss_pct else [],
                "timestamps": [int(t) for t in history.timestamp] if history.timestamp else [],
                "base_value": float(history.base_value) if history.base_value else 0,
            }
        except Exception as exc:
            logger.warning("Could not get portfolio history: {}", exc)
            return {}

    async def get_corporate_announcements(
        self, symbols: list[str] | None = None
    ) -> list[dict]:
        """Check for upcoming corporate events (earnings, splits, dividends)."""
        client = self._ensure_connected()
        # Cache for 6 hours
        age = time.monotonic() - self._announcements_fetched_at
        if self._announcements_cache and age < 21600:
            return self._announcements_cache

        try:
            from alpaca.trading.requests import GetCorporateAnnouncementsRequest
            from alpaca.trading.enums import CorporateActionType, CorporateActionDateType

            now = datetime.now(timezone.utc).date()
            request = GetCorporateAnnouncementsRequest(
                ca_types=[
                    CorporateActionType.DIVIDEND,
                    CorporateActionType.SPLIT,
                ],
                since=now,
                until=now + timedelta(days=7),
                date_type=CorporateActionDateType.DECLARATION_DATE,
            )
            announcements = client.get_corporate_announcements(request)
            results = []
            for a in announcements:
                entry = {
                    "id": str(a.id),
                    "type": str(a.ca_type),
                    "sub_type": str(getattr(a, "ca_sub_type", "")),
                    "symbol": str(getattr(a, "symbol", "")),
                    "declaration_date": str(getattr(a, "declaration_date", "")),
                    "ex_date": str(getattr(a, "ex_date", "")),
                    "record_date": str(getattr(a, "record_date", "")),
                }
                if symbols and entry["symbol"] not in symbols:
                    continue
                results.append(entry)

            self._announcements_cache = results
            self._announcements_fetched_at = time.monotonic()
            if results:
                logger.info(
                    "Found {} corporate announcements for tracked symbols",
                    len(results),
                )
            return results
        except Exception as exc:
            logger.debug("Could not get corporate announcements: {}", exc)
            return []

    # ── Stream status helpers ─────────────────────────────────────────────────

    def is_streaming(self) -> bool:
        """Return True if any market data websocket is connected."""
        stock_ok = self._stream_thread is not None and self._stream_thread.is_alive()
        crypto_ok = self._crypto_stream_thread is not None and self._crypto_stream_thread.is_alive()
        return stock_ok or crypto_ok

    def is_trade_streaming(self) -> bool:
        """Return True if the trading event websocket is connected."""
        return (
            self._trade_stream_thread is not None
            and self._trade_stream_thread.is_alive()
        )

    def is_symbol_halted(self, symbol: str) -> bool:
        """Return True if symbol is currently halted."""
        return self.halt_tracker.is_halted(symbol)

    async def drain_trade_updates(self) -> list[dict]:
        """Drain all pending trade update events from the queue."""
        events: list[dict] = []
        while True:
            event = await self.trade_updates.get(timeout=0.01)
            if event is None:
                break
            events.append(event)
        return events

    async def drain_news_events(self) -> list[dict]:
        """Drain all pending news events from the queue."""
        events: list[dict] = []
        while True:
            event = await self.news_events.get(timeout=0.01)
            if event is None:
                break
            events.append(event)
        return events
