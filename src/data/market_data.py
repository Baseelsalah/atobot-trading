"""Market data provider for AtoBot Trading."""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Any

import pandas as pd
from loguru import logger

from src.exchange.base_client import BaseExchangeClient


class MarketDataProvider:
    """Fetches and caches market data from the exchange."""

    def __init__(self, exchange_client: BaseExchangeClient, cache_size: int = 500) -> None:
        self.exchange = exchange_client
        self._price_cache: dict[str, deque[Decimal]] = {}
        self._kline_cache: dict[str, pd.DataFrame] = {}
        self._cache_size = cache_size

    async def get_current_price(self, symbol: str) -> Decimal:
        """Fetch and cache the latest ticker price.

        Args:
            symbol: Trading symbol (e.g. "AAPL").

        Returns:
            Current price as Decimal.
        """
        price = await self.exchange.get_ticker_price(symbol)
        if symbol not in self._price_cache:
            self._price_cache[symbol] = deque(maxlen=self._cache_size)
        self._price_cache[symbol].append(price)
        return price

    async def get_ohlcv(
        self, symbol: str, interval: str = "1h", limit: int = 100
    ) -> pd.DataFrame:
        """Fetch OHLCV kline data and return as a DataFrame.

        Args:
            symbol: Trading pair symbol.
            interval: Kline interval (e.g. "1m", "5m", "1h", "1d").
            limit: Number of klines to fetch (max 1000).

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        cache_key = f"{symbol}_{interval}"
        raw_klines: list[dict[str, Any]] = await self.exchange.get_klines(
            symbol=symbol, interval=interval, limit=limit
        )

        rows: list[dict[str, Any]] = []
        for k in raw_klines:
            rows.append(
                {
                    "timestamp": k["timestamp"],
                    "open": float(k["open"]),
                    "high": float(k["high"]),
                    "low": float(k["low"]),
                    "close": float(k["close"]),
                    "volume": float(k["volume"]),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")

        self._kline_cache[cache_key] = df
        logger.debug(
            "Fetched {} klines for {} ({})", len(df), symbol, interval
        )
        return df

    async def get_price_history(self, symbol: str, limit: int = 50) -> list[Decimal]:
        """Return the last *limit* cached prices for *symbol*.

        If the cache has fewer entries, returns whatever is available.

        Args:
            symbol: Trading pair symbol.
            limit: Maximum number of prices.

        Returns:
            List of Decimal prices (oldest first).
        """
        cache = self._price_cache.get(symbol, deque())
        history = list(cache)
        return history[-limit:]

    def get_cached_klines(self, symbol: str, interval: str) -> pd.DataFrame | None:
        """Return cached klines if available (no API call)."""
        cache_key = f"{symbol}_{interval}"
        return self._kline_cache.get(cache_key)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._price_cache.clear()
        self._kline_cache.clear()
        logger.debug("Market data cache cleared")
