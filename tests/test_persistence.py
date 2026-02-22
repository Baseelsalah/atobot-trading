"""Tests for the persistence layer (database + repository)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from src.models.order import Order, OrderSide, OrderStatus, OrderType
from src.models.trade import Trade
from src.persistence.database import init_database, close_database
from src.persistence.repository import TradingRepository


@pytest.fixture
async def repo():
    """Create an in-memory SQLite database and return a TradingRepository."""
    session_factory = await init_database("sqlite+aiosqlite:///:memory:")
    repository = TradingRepository(session_factory)
    yield repository
    await close_database()


def _make_order(**overrides) -> Order:
    """Create a test order with sensible defaults."""
    defaults = {
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "order_type": OrderType.LIMIT,
        "price": Decimal("185.50"),
        "quantity": Decimal("3"),
        "strategy": "momentum",
    }
    defaults.update(overrides)
    return Order(**defaults)


def _make_trade(**overrides) -> Trade:
    """Create a test trade with sensible defaults."""
    defaults = {
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "price": Decimal("185.50"),
        "quantity": Decimal("3"),
        "fee": Decimal("0.00"),
        "fee_asset": "USD",
        "strategy": "momentum",
    }
    defaults.update(overrides)
    return Trade(**defaults)


class TestOrderPersistence:
    """Tests for saving and retrieving orders."""

    @pytest.mark.asyncio
    async def test_save_and_retrieve_order(self, repo: TradingRepository) -> None:
        """save_order + get_open_orders round-trip."""
        order = _make_order()
        await repo.save_order(order)
        open_orders = await repo.get_open_orders("AAPL")
        assert len(open_orders) == 1
        assert open_orders[0].internal_id == order.internal_id
        assert open_orders[0].price == Decimal("185.50")

    @pytest.mark.asyncio
    async def test_update_order(self, repo: TradingRepository) -> None:
        """update_order should change status and filled_quantity."""
        order = _make_order()
        await repo.save_order(order)

        order.mark_filled(Decimal("3"))
        await repo.update_order(order)

        # Filled orders should not appear in open orders
        open_orders = await repo.get_open_orders("AAPL")
        assert len(open_orders) == 0

    @pytest.mark.asyncio
    async def test_get_open_orders_filters_by_symbol(
        self, repo: TradingRepository
    ) -> None:
        """get_open_orders should only return orders for the given symbol."""
        await repo.save_order(_make_order(symbol="AAPL"))
        await repo.save_order(_make_order(symbol="TSLA"))

        aapl_orders = await repo.get_open_orders("AAPL")
        assert len(aapl_orders) == 1
        assert aapl_orders[0].symbol == "AAPL"


class TestTradePersistence:
    """Tests for saving and retrieving trades."""

    @pytest.mark.asyncio
    async def test_save_and_get_trades(self, repo: TradingRepository) -> None:
        """save_trade + get_trades round-trip."""
        trade = _make_trade()
        await repo.save_trade(trade)
        trades = await repo.get_trades("AAPL")
        assert len(trades) == 1
        assert trades[0].price == Decimal("185.50")

    @pytest.mark.asyncio
    async def test_get_total_pnl(self, repo: TradingRepository) -> None:
        """get_total_pnl should sum all trade PnLs."""
        await repo.save_trade(_make_trade(pnl=Decimal("100")))
        await repo.save_trade(_make_trade(pnl=Decimal("-30")))
        total = await repo.get_total_pnl()
        assert total == Decimal("70")


class TestDailyStats:
    """Tests for daily statistics."""

    @pytest.mark.asyncio
    async def test_update_and_get_daily_pnl(
        self, repo: TradingRepository
    ) -> None:
        """update_daily_stats + get_daily_pnl round-trip."""
        today = date.today()
        await repo.update_daily_stats(today, Decimal("50"), is_win=True)
        await repo.update_daily_stats(today, Decimal("-20"), is_win=False)
        pnl = await repo.get_daily_pnl(today)
        assert pnl == Decimal("30")


class TestBotState:
    """Tests for bot state persistence."""

    @pytest.mark.asyncio
    async def test_save_and_load_state(self, repo: TradingRepository) -> None:
        """save_bot_state + load_bot_state round-trip."""
        state = {"strategy": "momentum", "running": True, "balance": "50000"}
        await repo.save_bot_state(state)
        loaded = await repo.load_bot_state()
        assert loaded is not None
        assert loaded["strategy"] == "momentum"
        assert loaded["running"] is True

    @pytest.mark.asyncio
    async def test_load_state_empty(self, repo: TradingRepository) -> None:
        """load_bot_state should return None when empty."""
        loaded = await repo.load_bot_state()
        assert loaded is None

    @pytest.mark.asyncio
    async def test_update_existing_state(self, repo: TradingRepository) -> None:
        """save_bot_state should update existing keys."""
        await repo.save_bot_state({"key1": "value1"})
        await repo.save_bot_state({"key1": "value2"})
        loaded = await repo.load_bot_state()
        assert loaded["key1"] == "value2"
