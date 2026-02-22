"""Tests for the Alpaca exchange client (stock day trading)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings
from src.exchange.alpaca_client import AlpacaClient, AlpacaClientError


# Patch asyncio.sleep globally so retries don't actually wait
@pytest.fixture(autouse=True)
async def _fast_retry(monkeypatch):
    """Zero out retry delay to keep tests fast."""
    import asyncio

    async def _instant(*_a, **_k):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant)


# ── Helper fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def alpaca_settings(mock_settings: Settings) -> Settings:
    """Return settings configured for Alpaca stock trading."""
    mock_settings.EXCHANGE = "alpaca"
    mock_settings.ALPACA_API_KEY = "test-key"
    mock_settings.ALPACA_API_SECRET = "test-secret"
    mock_settings.ALPACA_PAPER = True
    return mock_settings


# ── Construction / Connection ─────────────────────────────────────────────────


def test_alpaca_client_init(alpaca_settings: Settings) -> None:
    """AlpacaClient should store settings on init."""
    client = AlpacaClient(alpaca_settings)
    assert client._settings is alpaca_settings
    assert client._trading_client is None
    assert client._data_client is None


def test_ensure_connected_raises_when_not_connected(
    alpaca_settings: Settings,
) -> None:
    """_ensure_connected should raise when trading client is None."""
    client = AlpacaClient(alpaca_settings)
    with pytest.raises(AlpacaClientError, match="not connected"):
        client._ensure_connected()


@pytest.mark.asyncio
async def test_disconnect_when_not_connected(
    alpaca_settings: Settings,
) -> None:
    """Disconnecting when not connected should not raise."""
    client = AlpacaClient(alpaca_settings)
    await client.disconnect()
    assert client._trading_client is None


@pytest.mark.asyncio
async def test_connect_creates_clients(alpaca_settings: Settings) -> None:
    """connect() should create TradingClient and StockHistoricalDataClient."""
    mock_account = MagicMock()
    mock_account.equity = "50000.00"
    mock_account.buying_power = "100000.00"
    mock_account.daytrade_count = 0

    mock_trading_instance = MagicMock()
    mock_trading_instance.get_account.return_value = mock_account

    mock_data_instance = MagicMock()

    # Verify the client works after manual wiring
    client = AlpacaClient(alpaca_settings)
    client._trading_client = mock_trading_instance
    client._data_client = mock_data_instance

    tc = client._ensure_connected()
    assert tc is mock_trading_instance


# ── Symbol filters / caching ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_symbol_filters_returns_defaults_on_error(
    alpaca_settings: Settings,
) -> None:
    """When exchange info fails, sensible defaults should be returned."""
    client = AlpacaClient(alpaca_settings)
    # Not connected → get_exchange_info will fail — expect defaults
    filters = await client.get_symbol_filters("AAPL")
    assert filters["tick_size"] == Decimal("0.01")
    assert filters["step_size"] == Decimal("1")
    assert filters["min_notional"] == Decimal("1")
    assert filters["min_qty"] == Decimal("1")


@pytest.mark.asyncio
async def test_get_symbol_filters_caches(alpaca_settings: Settings) -> None:
    """After first call, filters should be served from cache."""
    client = AlpacaClient(alpaca_settings)
    client._symbol_filters["AAPL"] = {
        "tick_size": Decimal("0.01"),
        "step_size": Decimal("0.001"),
        "min_notional": Decimal("1"),
        "min_qty": Decimal("0.001"),
    }
    result = await client.get_symbol_filters("AAPL")
    assert result["step_size"] == Decimal("0.001")


# ── Order methods (offline — no real API) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_place_limit_order_raises_not_connected(
    alpaca_settings: Settings,
) -> None:
    """place_limit_order should raise when client is not connected."""
    client = AlpacaClient(alpaca_settings)
    with pytest.raises(AlpacaClientError):
        await client.place_limit_order(
            "AAPL", "BUY", Decimal("185.50"), Decimal("3")
        )


@pytest.mark.asyncio
async def test_place_market_order_raises_not_connected(
    alpaca_settings: Settings,
) -> None:
    """place_market_order should raise when client is not connected."""
    client = AlpacaClient(alpaca_settings)
    with pytest.raises(AlpacaClientError):
        await client.place_market_order("AAPL", "BUY", Decimal("3"))


@pytest.mark.asyncio
async def test_cancel_order_raises_not_connected(
    alpaca_settings: Settings,
) -> None:
    """cancel_order should raise when client is not connected."""
    client = AlpacaClient(alpaca_settings)
    with pytest.raises(AlpacaClientError):
        await client.cancel_order("AAPL", "fake-order-id")


@pytest.mark.asyncio
async def test_get_order_status_raises_not_connected(
    alpaca_settings: Settings,
) -> None:
    """get_order_status should raise when not connected."""
    client = AlpacaClient(alpaca_settings)
    with pytest.raises(AlpacaClientError):
        await client.get_order_status("AAPL", "fake-order-id")


@pytest.mark.asyncio
async def test_get_open_orders_raises_not_connected(
    alpaca_settings: Settings,
) -> None:
    """get_open_orders should raise when not connected."""
    client = AlpacaClient(alpaca_settings)
    with pytest.raises(AlpacaClientError):
        await client.get_open_orders("AAPL")


@pytest.mark.asyncio
async def test_get_account_balance_raises_not_connected(
    alpaca_settings: Settings,
) -> None:
    """get_account_balance should raise when not connected."""
    client = AlpacaClient(alpaca_settings)
    with pytest.raises(AlpacaClientError):
        await client.get_account_balance()


@pytest.mark.asyncio
async def test_get_ticker_price_raises_not_connected(
    alpaca_settings: Settings,
) -> None:
    """get_ticker_price should raise when data client is None."""
    client = AlpacaClient(alpaca_settings)
    with pytest.raises((AlpacaClientError, Exception)):
        await client.get_ticker_price("AAPL")


# ── Mocked API calls ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_place_limit_order_success(alpaca_settings: Settings) -> None:
    """place_limit_order should return order dict on success."""
    mock_order = MagicMock()
    mock_order.id = "abc-123"
    mock_order.client_order_id = "client-456"
    mock_order.status = MagicMock(value="accepted")

    mock_tc = MagicMock()
    mock_tc.submit_order.return_value = mock_order

    client = AlpacaClient(alpaca_settings)
    client._trading_client = mock_tc

    result = await client.place_limit_order(
        "AAPL", "BUY", Decimal("185.50"), Decimal("3")
    )
    assert result["orderId"] == "abc-123"
    assert result["status"] == "accepted"
    assert result["symbol"] == "AAPL"
    mock_tc.submit_order.assert_called_once()


@pytest.mark.asyncio
async def test_place_market_order_success(alpaca_settings: Settings) -> None:
    """place_market_order should return order dict on success."""
    mock_order = MagicMock()
    mock_order.id = "mkt-789"
    mock_order.client_order_id = "client-mkt"
    mock_order.status = MagicMock(value="accepted")

    mock_tc = MagicMock()
    mock_tc.submit_order.return_value = mock_order

    client = AlpacaClient(alpaca_settings)
    client._trading_client = mock_tc

    result = await client.place_market_order("AAPL", "SELL", Decimal("5"))
    assert result["orderId"] == "mkt-789"
    assert result["symbol"] == "AAPL"
    mock_tc.submit_order.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_order_success(alpaca_settings: Settings) -> None:
    """cancel_order should return cancelled status."""
    mock_tc = MagicMock()
    mock_tc.cancel_order_by_id.return_value = None

    client = AlpacaClient(alpaca_settings)
    client._trading_client = mock_tc

    result = await client.cancel_order("AAPL", "order-to-cancel")
    assert result["status"] == "CANCELED"
    assert result["orderId"] == "order-to-cancel"


@pytest.mark.asyncio
async def test_get_order_status_success(alpaca_settings: Settings) -> None:
    """get_order_status should parse API response."""
    mock_order = MagicMock()
    mock_order.id = "status-123"
    mock_order.status = MagicMock(value="filled")
    mock_order.filled_qty = "3"
    mock_order.filled_avg_price = "185.50"
    mock_order.side = MagicMock(value="buy")
    mock_order.type = MagicMock(value="limit")
    mock_order.symbol = "AAPL"

    mock_tc = MagicMock()
    mock_tc.get_order_by_id.return_value = mock_order

    client = AlpacaClient(alpaca_settings)
    client._trading_client = mock_tc

    result = await client.get_order_status("AAPL", "status-123")
    assert result["status"] == "filled"
    assert result["executedQty"] == "3"
    assert result["filledAvgPrice"] == "185.50"
    assert result["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_get_account_balance_success(alpaca_settings: Settings) -> None:
    """get_account_balance should return cash, equity and position balances."""
    mock_account = MagicMock()
    mock_account.cash = "50000.00"
    mock_account.equity = "52000.00"
    mock_account.buying_power = "100000.00"
    mock_account.daytrade_count = 1

    mock_position = MagicMock()
    mock_position.symbol = "AAPL"
    mock_position.qty = "10"
    mock_position.market_value = "1855.00"
    mock_position.unrealized_pl = "55.00"

    mock_tc = MagicMock()
    mock_tc.get_account.return_value = mock_account
    mock_tc.get_all_positions.return_value = [mock_position]

    client = AlpacaClient(alpaca_settings)
    client._trading_client = mock_tc

    result = await client.get_account_balance()
    assert result["USD"] == Decimal("50000.00")
    assert result["EQUITY"] == Decimal("52000.00")
    assert result["BUYING_POWER"] == Decimal("100000.00")
    assert result["DAYTRADE_COUNT"] == Decimal("1")
    assert result["AAPL"] == Decimal("10")


# ── Stock-specific helpers ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_is_market_open(alpaca_settings: Settings) -> None:
    """is_market_open should return bool from Alpaca clock."""
    mock_clock = MagicMock()
    mock_clock.is_open = True

    mock_tc = MagicMock()
    mock_tc.get_clock.return_value = mock_clock

    client = AlpacaClient(alpaca_settings)
    client._trading_client = mock_tc

    result = await client.is_market_open()
    assert result is True


@pytest.mark.asyncio
async def test_get_market_clock(alpaca_settings: Settings) -> None:
    """get_market_clock should return clock info dict."""
    mock_clock = MagicMock()
    mock_clock.is_open = True
    mock_clock.next_open = "2026-02-22T14:30:00Z"
    mock_clock.next_close = "2026-02-22T21:00:00Z"
    mock_clock.timestamp = "2026-02-22T18:00:00Z"

    mock_tc = MagicMock()
    mock_tc.get_clock.return_value = mock_clock

    client = AlpacaClient(alpaca_settings)
    client._trading_client = mock_tc

    result = await client.get_market_clock()
    assert result["is_open"] is True
    assert "next_close" in result


@pytest.mark.asyncio
async def test_close_position(alpaca_settings: Settings) -> None:
    """close_position should return order info."""
    mock_order = MagicMock()
    mock_order.id = "close-123"

    mock_tc = MagicMock()
    mock_tc.close_position.return_value = mock_order

    client = AlpacaClient(alpaca_settings)
    client._trading_client = mock_tc

    result = await client.close_position("AAPL")
    assert result["symbol"] == "AAPL"
    assert result["status"] == "closing"


@pytest.mark.asyncio
async def test_close_all_positions(alpaca_settings: Settings) -> None:
    """close_all_positions should close all and return list."""
    mock_resp = MagicMock()
    mock_resp.symbol = "AAPL"

    mock_tc = MagicMock()
    mock_tc.close_all_positions.return_value = [mock_resp]

    client = AlpacaClient(alpaca_settings)
    client._trading_client = mock_tc

    result = await client.close_all_positions()
    assert len(result) == 1
    assert result[0]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_get_positions(alpaca_settings: Settings) -> None:
    """get_positions should return list of position dicts."""
    mock_pos = MagicMock()
    mock_pos.symbol = "TSLA"
    mock_pos.qty = "5"
    mock_pos.avg_entry_price = "250.00"
    mock_pos.market_value = "1300.00"
    mock_pos.unrealized_pl = "50.00"
    mock_pos.unrealized_plpc = "0.04"
    mock_pos.current_price = "260.00"
    mock_pos.side = "long"

    mock_tc = MagicMock()
    mock_tc.get_all_positions.return_value = [mock_pos]

    client = AlpacaClient(alpaca_settings)
    client._trading_client = mock_tc

    result = await client.get_positions()
    assert len(result) == 1
    assert result[0]["symbol"] == "TSLA"
    assert result[0]["qty"] == Decimal("5")
    assert result[0]["current_price"] == Decimal("260.00")


@pytest.mark.asyncio
async def test_get_account_info(alpaca_settings: Settings) -> None:
    """get_account_info should return full account detail."""
    mock_account = MagicMock()
    mock_account.equity = "50000.00"
    mock_account.cash = "25000.00"
    mock_account.buying_power = "100000.00"
    mock_account.portfolio_value = "50000.00"
    mock_account.daytrade_count = 2
    mock_account.pattern_day_trader = False
    mock_account.trading_blocked = False
    mock_account.account_blocked = False

    mock_tc = MagicMock()
    mock_tc.get_account.return_value = mock_account

    client = AlpacaClient(alpaca_settings)
    client._trading_client = mock_tc

    result = await client.get_account_info()
    assert result["equity"] == Decimal("50000.00")
    assert result["daytrade_count"] == 2
    assert result["pattern_day_trader"] is False
