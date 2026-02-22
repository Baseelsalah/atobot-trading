"""Tests for the MarketDataProvider module."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.config.settings import Settings
from src.data.market_data import MarketDataProvider


@pytest.fixture
def market_data(mock_exchange_client: AsyncMock) -> MarketDataProvider:
    """Create a MarketDataProvider with mocked exchange."""
    return MarketDataProvider(mock_exchange_client)


@pytest.mark.asyncio
async def test_get_current_price(market_data: MarketDataProvider) -> None:
    """get_current_price should return price from exchange."""
    market_data.exchange.get_ticker_price = AsyncMock(
        return_value=Decimal("185.50")
    )
    price = await market_data.get_current_price("AAPL")
    assert price == Decimal("185.50")
    market_data.exchange.get_ticker_price.assert_awaited_once_with("AAPL")


@pytest.mark.asyncio
async def test_price_caching(market_data: MarketDataProvider) -> None:
    """Prices should be cached in the deque."""
    market_data.exchange.get_ticker_price = AsyncMock(
        return_value=Decimal("185.50")
    )
    await market_data.get_current_price("AAPL")
    assert len(market_data._price_cache["AAPL"]) == 1


@pytest.mark.asyncio
async def test_get_price_history(market_data: MarketDataProvider) -> None:
    """get_price_history should return cached prices."""
    market_data.exchange.get_ticker_price = AsyncMock(
        side_effect=[Decimal("185"), Decimal("186"), Decimal("187")]
    )
    await market_data.get_current_price("AAPL")
    await market_data.get_current_price("AAPL")
    await market_data.get_current_price("AAPL")
    history = await market_data.get_price_history("AAPL")
    assert len(history) == 3
    assert history[-1] == Decimal("187")


@pytest.mark.asyncio
async def test_get_ohlcv(market_data: MarketDataProvider) -> None:
    """get_ohlcv should fetch klines from exchange and return DataFrame."""
    import pandas as pd

    mock_klines = [
        {
            "timestamp": 1625000000000,
            "open": "185.00",
            "high": "186.00",
            "low": "184.00",
            "close": "185.50",
            "volume": "50000.0",
        }
    ]
    market_data.exchange.get_klines = AsyncMock(return_value=mock_klines)
    df = await market_data.get_ohlcv("AAPL")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "close" in df.columns
