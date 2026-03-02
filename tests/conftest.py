"""Shared pytest fixtures for AtoBot Trading tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.config.settings import Settings
from src.exchange.base_client import BaseExchangeClient
from src.models.order import Order, OrderSide, OrderType
from src.models.position import Position
from src.models.trade import Trade
from src.risk.risk_manager import RiskManager


@pytest.fixture
def mock_settings() -> Settings:
    """Return a Settings object with safe test defaults (stocks day-trading)."""
    return Settings(
        EXCHANGE="alpaca",
        ALPACA_API_KEY="test-key",
        ALPACA_API_SECRET="test-secret",
        ALPACA_PAPER=True,
        BINANCE_API_KEY="test_key",
        BINANCE_API_SECRET="test_secret",
        BINANCE_TESTNET=True,
        SYMBOLS=["AAPL"],
        DEFAULT_STRATEGY="momentum",
        BASE_ORDER_SIZE_USD=500.0,
        MAX_OPEN_ORDERS=10,
        MARKET_HOURS_ONLY=False,
        FLATTEN_EOD=False,
        FLATTEN_MINUTES_BEFORE_CLOSE=5,
        MOMENTUM_RSI_PERIOD=14,
        MOMENTUM_RSI_OVERSOLD=30.0,
        MOMENTUM_RSI_OVERBOUGHT=70.0,
        MOMENTUM_VOLUME_MULTIPLIER=1.5,
        MOMENTUM_TAKE_PROFIT_PERCENT=2.0,
        MOMENTUM_STOP_LOSS_PERCENT=1.0,
        MAX_DRAWDOWN_PERCENT=10.0,
        STOP_LOSS_PERCENT=5.0,
        MAX_POSITION_SIZE_USD=2000.0,
        DAILY_LOSS_LIMIT_USD=200.0,
        MAX_DAILY_TRADES=20,
        PDT_PROTECTION=True,
        TELEGRAM_BOT_TOKEN="",
        TELEGRAM_CHAT_ID="",
        NOTIFICATIONS_ENABLED=False,
        DATABASE_URL="sqlite+aiosqlite:///data/test_atobot.db",
        LOG_LEVEL="DEBUG",
        DRY_RUN=True,
        POLL_INTERVAL_SECONDS=1,
        # Disable new filters by default in tests (test them separately)
        TREND_FILTER_ENABLED=False,
        AVOID_MIDDAY=False,
        TRAILING_STOP_ENABLED=False,
    )


@pytest.fixture
def mock_exchange_client() -> AsyncMock:
    """Return an AsyncMock of BaseExchangeClient."""
    client = AsyncMock(spec=BaseExchangeClient)

    # Default return values — stock prices
    client.get_ticker_price.return_value = Decimal("185.50")
    client.get_order_book.return_value = {
        "bids": [{"price": Decimal("185.49"), "quantity": Decimal("100")}],
        "asks": [{"price": Decimal("185.51"), "quantity": Decimal("100")}],
    }
    client.get_symbol_filters.return_value = {
        "tick_size": Decimal("0.01"),
        "step_size": Decimal("0.01"),
        "min_notional": Decimal("1"),
        "min_qty": Decimal("1"),
    }
    client.get_account_balance.return_value = {
        "USD": Decimal("50000"),
        "EQUITY": Decimal("50000"),
        "BUYING_POWER": Decimal("100000"),
        "DAYTRADE_COUNT": Decimal("0"),
    }
    client.place_limit_order.return_value = {"orderId": "123456", "status": "NEW"}
    client.place_market_order.return_value = {"orderId": "123457", "status": "FILLED"}
    client.cancel_order.return_value = {"orderId": "123456", "status": "CANCELED"}
    client.get_order_status.return_value = {"orderId": "123456", "status": "NEW", "executedQty": "0"}
    client.get_open_orders.return_value = []
    client.get_klines.return_value = [
        {
            "timestamp": 1700000000000 + i * 300000,
            "open": Decimal("185") + Decimal(str(i)),
            "high": Decimal("186") + Decimal(str(i)),
            "low": Decimal("184") + Decimal(str(i)),
            "close": Decimal("185.50") + Decimal(str(i)),
            "volume": Decimal("50000"),
        }
        for i in range(100)
    ]
    client.get_exchange_info.return_value = {
        "symbol": "AAPL",
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "1"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "1"},
        ],
    }

    # Streaming defaults (not halted, no queued events)
    client.is_symbol_halted.return_value = False
    client.is_streaming.return_value = False
    client.is_trade_streaming.return_value = False
    client.drain_trade_updates.return_value = []
    client.drain_news_events.return_value = []

    return client


@pytest.fixture
def mock_risk_manager(mock_settings: Settings) -> RiskManager:
    """Return a RiskManager with test settings."""
    rm = RiskManager(mock_settings)
    rm.peak_balance = Decimal("50000")
    rm.current_balance = Decimal("50000")
    return rm


@pytest.fixture
def sample_order() -> Order:
    """Return a sample BUY limit order for AAPL."""
    return Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("185.50"),
        quantity=Decimal("3"),
        strategy="momentum",
    )


@pytest.fixture
def sample_trade() -> Trade:
    """Return a sample completed trade."""
    return Trade(
        symbol="AAPL",
        side=OrderSide.BUY,
        price=Decimal("185.50"),
        quantity=Decimal("3"),
        fee=Decimal("0.00"),
        fee_asset="USD",
        pnl=Decimal("5.50"),
        strategy="momentum",
    )


@pytest.fixture
def sample_position() -> Position:
    """Return a sample open position."""
    return Position(
        symbol="AAPL",
        side="LONG",
        entry_price=Decimal("185.00"),
        current_price=Decimal("187.50"),
        quantity=Decimal("3"),
        strategy="momentum",
    )
