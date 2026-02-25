"""Tests for streaming infrastructure (WebSocket caches, halt tracker, queues)."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings
from src.core.engine import TradingEngine
from src.data.market_data import MarketDataProvider
from src.exchange.alpaca_client import (
    HaltedSymbolTracker,
    NewsEventQueue,
    StreamingBarCache,
    StreamingPriceCache,
    TradeUpdateQueue,
)
from src.models.order import Order, OrderSide, OrderStatus, OrderType
from src.persistence.repository import TradingRepository
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy


# ── Unit tests for streaming helper classes ──────────────────────────────────


class TestStreamingPriceCache:
    def test_update_and_get(self):
        cache = StreamingPriceCache()
        cache.update("AAPL", Decimal("185.50"))
        assert cache.get("AAPL") == Decimal("185.50")

    def test_get_returns_none_for_unknown(self):
        cache = StreamingPriceCache()
        assert cache.get("TSLA") is None

    def test_get_returns_none_for_stale(self):
        cache = StreamingPriceCache()
        cache.update("AAPL", Decimal("185.50"))
        # Force staleness by backdating
        cache._updated_at["AAPL"] = 0.0
        assert cache.get("AAPL", max_age=1.0) is None

    def test_get_all(self):
        cache = StreamingPriceCache()
        cache.update("AAPL", Decimal("185.50"))
        cache.update("TSLA", Decimal("200.00"))
        all_prices = cache.get_all()
        assert len(all_prices) == 2
        assert all_prices["AAPL"] == Decimal("185.50")
        assert all_prices["TSLA"] == Decimal("200.00")


class TestStreamingBarCache:
    def test_update_and_get(self):
        cache = StreamingBarCache()
        bar = {"open": Decimal("185"), "close": Decimal("186")}
        cache.update("AAPL", bar)
        assert cache.get("AAPL") == bar

    def test_get_returns_none_for_unknown(self):
        cache = StreamingBarCache()
        assert cache.get("TSLA") is None


class TestHaltedSymbolTracker:
    def test_halted_and_resumed(self):
        tracker = HaltedSymbolTracker()
        assert not tracker.is_halted("AAPL")
        tracker.set_halted("AAPL")
        assert tracker.is_halted("AAPL")
        tracker.set_resumed("AAPL")
        assert not tracker.is_halted("AAPL")

    def test_get_all_halted(self):
        tracker = HaltedSymbolTracker()
        tracker.set_halted("AAPL")
        tracker.set_halted("TSLA")
        assert tracker.get_all_halted() == {"AAPL", "TSLA"}

    def test_resume_unknown_no_error(self):
        tracker = HaltedSymbolTracker()
        tracker.set_resumed("AAPL")  # Should not raise
        assert not tracker.is_halted("AAPL")


class TestTradeUpdateQueue:
    @pytest.mark.asyncio
    async def test_put_and_get(self):
        queue = TradeUpdateQueue()
        queue.put_nowait({"event": "fill", "symbol": "AAPL"})
        item = await queue.get(timeout=1.0)
        assert item is not None
        assert item["event"] == "fill"

    @pytest.mark.asyncio
    async def test_get_timeout_returns_none(self):
        queue = TradeUpdateQueue()
        item = await queue.get(timeout=0.01)
        assert item is None

    def test_qsize(self):
        queue = TradeUpdateQueue()
        assert queue.qsize() == 0
        queue.put_nowait({"event": "fill"})
        assert queue.qsize() == 1


class TestNewsEventQueue:
    @pytest.mark.asyncio
    async def test_put_and_get(self):
        queue = NewsEventQueue()
        queue.put_nowait({"headline": "AAPL beats earnings"})
        item = await queue.get(timeout=1.0)
        assert item is not None
        assert item["headline"] == "AAPL beats earnings"


# ── Integration tests: engine + streaming ────────────────────────────────────


def _make_streaming_engine(
    mock_settings: Settings,
    mock_exchange: AsyncMock,
    mock_risk_manager: RiskManager,
) -> TradingEngine:
    strategy = AsyncMock(spec=BaseStrategy)
    strategy.name = "momentum"
    strategy.active_orders = []
    strategy.positions = {}
    strategy.on_tick = AsyncMock(return_value=[])
    strategy.on_order_filled = AsyncMock(return_value=[])
    strategy.on_order_cancelled = AsyncMock(return_value=[])
    strategy.get_status = AsyncMock(return_value={"strategy": "momentum", "active_orders": 0})

    market_data = AsyncMock(spec=MarketDataProvider)
    market_data.get_current_price = AsyncMock(return_value=Decimal("185.50"))

    repo = AsyncMock(spec=TradingRepository)

    engine = TradingEngine(
        exchange=mock_exchange,
        strategy=strategy,
        risk_manager=mock_risk_manager,
        market_data=market_data,
        repository=repo,
        notifier=None,
        settings=mock_settings,
    )
    return engine


@pytest.mark.asyncio
async def test_engine_skips_halted_symbol(
    mock_settings: Settings,
    mock_exchange_client: AsyncMock,
    mock_risk_manager: RiskManager,
) -> None:
    """Engine skips symbols that are halted."""
    mock_exchange_client.is_symbol_halted.return_value = True

    engine = _make_streaming_engine(mock_settings, mock_exchange_client, mock_risk_manager)
    mock_settings.POLL_INTERVAL_SECONDS = 1

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.2)
    await engine.stop()
    await asyncio.wait_for(task, timeout=5)

    # Strategy on_tick should NOT have been called (symbol is halted)
    engine.strategy.on_tick.assert_not_called()


@pytest.mark.asyncio
async def test_engine_processes_non_halted_symbol(
    mock_settings: Settings,
    mock_exchange_client: AsyncMock,
    mock_risk_manager: RiskManager,
) -> None:
    """Engine processes symbols that are not halted."""
    mock_exchange_client.is_symbol_halted.return_value = False

    engine = _make_streaming_engine(mock_settings, mock_exchange_client, mock_risk_manager)
    mock_settings.POLL_INTERVAL_SECONDS = 1

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.2)
    await engine.stop()
    await asyncio.wait_for(task, timeout=5)

    # Strategy on_tick should have been called
    engine.strategy.on_tick.assert_called()


@pytest.mark.asyncio
async def test_engine_processes_trade_update_fill(
    mock_settings: Settings,
    mock_exchange_client: AsyncMock,
    mock_risk_manager: RiskManager,
) -> None:
    """Engine processes a fill event from the trade update queue."""
    engine = _make_streaming_engine(mock_settings, mock_exchange_client, mock_risk_manager)

    # Create an active order
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        price=Decimal("185.50"),
        quantity=Decimal("10"),
        strategy="momentum",
    )
    order.id = "test-order-123"
    order.status = OrderStatus.OPEN
    engine.strategies[0].active_orders.append(order)

    # Simulate a fill event from the stream
    fill_event = {
        "event": "fill",
        "order_id": "test-order-123",
        "symbol": "AAPL",
        "side": "buy",
        "filled_qty": "10",
        "filled_avg_price": "185.50",
        "qty": "10",
    }
    mock_exchange_client.drain_trade_updates.return_value = [fill_event]

    await engine._process_trade_updates()

    # Order should be marked as filled
    assert order.status == OrderStatus.FILLED
    # Strategy callback should have been called
    engine.strategies[0].on_order_filled.assert_called_once_with(order)


@pytest.mark.asyncio
async def test_engine_processes_trade_update_cancel(
    mock_settings: Settings,
    mock_exchange_client: AsyncMock,
    mock_risk_manager: RiskManager,
) -> None:
    """Engine processes a cancel event from the trade update queue."""
    engine = _make_streaming_engine(mock_settings, mock_exchange_client, mock_risk_manager)

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("184.00"),
        quantity=Decimal("5"),
        strategy="momentum",
    )
    order.id = "test-order-456"
    order.status = OrderStatus.OPEN
    engine.strategies[0].active_orders.append(order)

    cancel_event = {
        "event": "canceled",
        "order_id": "test-order-456",
        "symbol": "AAPL",
        "side": "buy",
        "filled_qty": "0",
        "filled_avg_price": "",
        "qty": "5",
    }
    mock_exchange_client.drain_trade_updates.return_value = [cancel_event]

    await engine._process_trade_updates()

    assert order.status == OrderStatus.CANCELLED
    engine.strategies[0].on_order_cancelled.assert_called_once_with(order)


@pytest.mark.asyncio
async def test_engine_ignores_unknown_order_update(
    mock_settings: Settings,
    mock_exchange_client: AsyncMock,
    mock_risk_manager: RiskManager,
) -> None:
    """Engine ignores stream updates for orders it doesn't track."""
    engine = _make_streaming_engine(mock_settings, mock_exchange_client, mock_risk_manager)

    unknown_event = {
        "event": "fill",
        "order_id": "unknown-order-999",
        "symbol": "TSLA",
        "side": "buy",
        "filled_qty": "10",
        "filled_avg_price": "200.00",
    }
    mock_exchange_client.drain_trade_updates.return_value = [unknown_event]

    # Should not raise
    await engine._process_trade_updates()

    # No callbacks should have been made
    engine.strategies[0].on_order_filled.assert_not_called()


@pytest.mark.asyncio
async def test_settings_streaming_defaults() -> None:
    """Verify streaming settings have correct defaults."""
    with patch.dict("os.environ", {}, clear=False):
        settings = Settings(
            ALPACA_API_KEY="test",
            ALPACA_API_SECRET="test",
        )
        assert settings.STREAMING_ENABLED is True
        assert settings.TRADE_STREAM_ENABLED is True
        assert settings.NEWS_STREAM_ENABLED is False
        assert settings.DATA_FEED == "sip"


@pytest.mark.asyncio
async def test_settings_data_feed_validation() -> None:
    """DATA_FEED must be 'iex' or 'sip'."""
    with pytest.raises(Exception):
        Settings(
            ALPACA_API_KEY="test",
            ALPACA_API_SECRET="test",
            DATA_FEED="invalid",
        )
