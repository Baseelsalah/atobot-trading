"""Tests for the trading engine."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings
from src.core.engine import TradingEngine
from src.data.market_data import MarketDataProvider
from src.models.order import Order, OrderSide, OrderType
from src.persistence.repository import TradingRepository
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy


def _make_engine(
    mock_settings: Settings,
    mock_exchange_client: AsyncMock,
    mock_risk_manager: RiskManager,
) -> TradingEngine:
    """Helper to create a TradingEngine with mocked deps."""
    strategy = AsyncMock(spec=BaseStrategy)
    strategy.name = "momentum"
    strategy.active_orders = []
    strategy.positions = {}
    strategy.on_tick = AsyncMock(return_value=[])
    strategy.on_order_filled = AsyncMock(return_value=[])
    strategy.get_status = AsyncMock(return_value={"strategy": "momentum", "active_orders": 0})

    market_data = AsyncMock(spec=MarketDataProvider)
    market_data.get_current_price = AsyncMock(return_value=Decimal("185.50"))

    repo = AsyncMock(spec=TradingRepository)

    engine = TradingEngine(
        exchange=mock_exchange_client,
        strategy=strategy,
        risk_manager=mock_risk_manager,
        market_data=market_data,
        repository=repo,
        notifier=None,
        settings=mock_settings,
    )
    return engine


@pytest.mark.asyncio
async def test_engine_stop(
    mock_settings: Settings,
    mock_exchange_client: AsyncMock,
    mock_risk_manager: RiskManager,
) -> None:
    """Engine stops when stop() is called."""
    engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager)
    # Start engine in background, then immediately stop
    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.1)
    await engine.stop()
    await asyncio.wait_for(task, timeout=5)
    # Should have completed without error


@pytest.mark.asyncio
async def test_engine_calls_on_tick(
    mock_settings: Settings,
    mock_exchange_client: AsyncMock,
    mock_risk_manager: RiskManager,
) -> None:
    """Engine calls strategy.on_tick at least once before stopping."""
    engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager)
    mock_settings.POLL_INTERVAL_SECONDS = 1

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.2)
    await engine.stop()
    await asyncio.wait_for(task, timeout=5)

    engine.strategy.on_tick.assert_called()


@pytest.mark.asyncio
async def test_engine_risk_rejected_order(
    mock_settings: Settings,
    mock_exchange_client: AsyncMock,
    mock_risk_manager: RiskManager,
) -> None:
    """Orders rejected by risk manager are not placed."""
    engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager)

    # Risk manager rejects everything
    mock_risk_manager.is_halted = True
    mock_risk_manager.halt_reason = "test halt"

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.2)
    await engine.stop()
    await asyncio.wait_for(task, timeout=5)

    # No orders should have been placed
    mock_exchange_client.place_limit_order.assert_not_called()
    mock_exchange_client.place_market_order.assert_not_called()


@pytest.mark.asyncio
async def test_engine_dry_run_no_exchange_calls(
    mock_settings: Settings,
    mock_exchange_client: AsyncMock,
    mock_risk_manager: RiskManager,
) -> None:
    """In dry-run mode, orders are simulated and not sent to exchange."""
    engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager)
    mock_settings.DRY_RUN = True

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        price=Decimal("185.00"),
        quantity=Decimal("3"),
        strategy="momentum",
    )
    engine.strategy.on_tick = AsyncMock(return_value=[order])

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.3)
    await engine.stop()
    await asyncio.wait_for(task, timeout=5)

    # Exchange should NOT have been called for order placement
    mock_exchange_client.place_limit_order.assert_not_called()
