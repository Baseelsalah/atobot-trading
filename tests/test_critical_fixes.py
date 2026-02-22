"""Tests for the 6 critical bug fixes."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings
from src.core.bot import AtoBot
from src.core.engine import TradingEngine
from src.data.market_data import MarketDataProvider
from src.models.order import Order, OrderSide, OrderStatus, OrderType
from src.models.position import Position
from src.models.trade import Trade
from src.persistence.repository import TradingRepository
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy
from src.utils.helpers import calculate_pnl


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_strategy_mock(name: str = "vwap_scalp") -> AsyncMock:
    """Create a mock strategy with all required attributes."""
    strat = AsyncMock(spec=BaseStrategy)
    strat.name = name
    strat.active_orders = []
    strat.positions = {}
    strat._trailing_highs = {}
    strat.on_tick = AsyncMock(return_value=[])
    strat.on_order_filled = AsyncMock(return_value=[])
    strat.on_order_cancelled = AsyncMock(return_value=None)
    strat.get_status = AsyncMock(return_value={"strategy": name, "active_orders": 0})
    return strat


def _make_engine(
    mock_settings: Settings,
    mock_exchange_client: AsyncMock,
    mock_risk_manager: RiskManager,
    strategy: AsyncMock | None = None,
) -> TradingEngine:
    """Create a TradingEngine with mocked deps."""
    strat = strategy or _make_strategy_mock()
    market_data = AsyncMock(spec=MarketDataProvider)
    market_data.get_current_price = AsyncMock(return_value=Decimal("185.50"))
    repo = AsyncMock(spec=TradingRepository)

    engine = TradingEngine(
        exchange=mock_exchange_client,
        strategy=strat,
        risk_manager=mock_risk_manager,
        market_data=market_data,
        repository=repo,
        notifier=None,
        settings=mock_settings,
    )
    return engine


# ══════════════════════════════════════════════════════════════════════════════
# Fix #1: Position Reconciliation on Startup
# ══════════════════════════════════════════════════════════════════════════════


class TestPositionReconciliation:
    """Test _reconcile_positions in AtoBot."""

    @pytest.mark.asyncio
    async def test_reconcile_loads_exchange_positions(
        self, mock_settings: Settings
    ) -> None:
        """Untracked exchange positions are loaded into the first strategy."""
        bot = AtoBot(mock_settings)
        exchange = AsyncMock()
        exchange.get_positions.return_value = [
            {
                "symbol": "AAPL",
                "qty": Decimal("10"),
                "avg_entry_price": Decimal("185.00"),
                "current_price": Decimal("187.50"),
                "market_value": Decimal("1875.00"),
                "unrealized_pl": Decimal("25.00"),
                "unrealized_plpc": Decimal("0.0135"),
                "side": "long",
            }
        ]
        bot.exchange = exchange

        strat = _make_strategy_mock()
        bot.strategies = [strat]

        await bot._reconcile_positions()

        assert "AAPL" in strat.positions
        pos = strat.positions["AAPL"]
        assert pos.entry_price == Decimal("185.00")
        assert pos.quantity == Decimal("10")
        assert pos.side == "LONG"

    @pytest.mark.asyncio
    async def test_reconcile_skips_tracked_positions(
        self, mock_settings: Settings
    ) -> None:
        """Positions already tracked by a strategy are not duplicated."""
        bot = AtoBot(mock_settings)
        exchange = AsyncMock()
        exchange.get_positions.return_value = [
            {
                "symbol": "AAPL",
                "qty": Decimal("5"),
                "avg_entry_price": Decimal("190.00"),
                "current_price": Decimal("192.00"),
                "side": "long",
            }
        ]
        bot.exchange = exchange

        strat = _make_strategy_mock()
        existing_pos = Position(
            symbol="AAPL", side="LONG", entry_price=Decimal("185"),
            current_price=Decimal("187"), quantity=Decimal("3"), strategy="vwap_scalp",
        )
        strat.positions = {"AAPL": existing_pos}
        bot.strategies = [strat]

        await bot._reconcile_positions()

        # Should keep the original position, not overwrite
        assert strat.positions["AAPL"].entry_price == Decimal("185")
        assert strat.positions["AAPL"].quantity == Decimal("3")

    @pytest.mark.asyncio
    async def test_reconcile_handles_no_positions(
        self, mock_settings: Settings
    ) -> None:
        """No error when exchange has no open positions."""
        bot = AtoBot(mock_settings)
        exchange = AsyncMock()
        exchange.get_positions.return_value = []
        bot.exchange = exchange
        bot.strategies = [_make_strategy_mock()]

        await bot._reconcile_positions()  # No error

    @pytest.mark.asyncio
    async def test_reconcile_handles_api_error(
        self, mock_settings: Settings
    ) -> None:
        """API error during reconciliation doesn't crash the bot."""
        bot = AtoBot(mock_settings)
        exchange = AsyncMock()
        exchange.get_positions.side_effect = Exception("API down")
        bot.exchange = exchange
        bot.strategies = [_make_strategy_mock()]

        await bot._reconcile_positions()  # No error, just logs warning


# ══════════════════════════════════════════════════════════════════════════════
# Fix #2: Trade PnL Calculation
# ══════════════════════════════════════════════════════════════════════════════


class TestTradePnLCalculation:
    """Trade PnL is computed on sell fills using entry price."""

    def test_calculate_pnl_long_profit(self) -> None:
        """Long entry at 100, exit at 105, qty 10 = +$50."""
        pnl = calculate_pnl(Decimal("100"), Decimal("105"), Decimal("10"), "BUY")
        assert pnl == Decimal("50")

    def test_calculate_pnl_long_loss(self) -> None:
        """Long entry at 100, exit at 95, qty 10 = -$50."""
        pnl = calculate_pnl(Decimal("100"), Decimal("95"), Decimal("10"), "BUY")
        assert pnl == Decimal("-50")

    @pytest.mark.asyncio
    async def test_engine_computes_pnl_on_sell_fill(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """When a SELL order fills, trade.pnl is computed from position entry price."""
        strat = _make_strategy_mock()
        # Simulate an existing position
        strat.positions = {
            "AAPL": Position(
                symbol="AAPL", side="LONG", entry_price=Decimal("180.00"),
                current_price=Decimal("185.50"), quantity=Decimal("3"),
                strategy="vwap_scalp",
            )
        }

        # Create a filled sell order
        sell_order = Order(
            symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET,
            price=Decimal("185.50"), quantity=Decimal("3"), strategy="vwap_scalp",
        )
        sell_order.id = "order-123"
        sell_order.status = OrderStatus.OPEN
        strat.active_orders = [sell_order]

        # Mock exchange to return FILLED with fill price
        mock_exchange_client.get_order_status.return_value = {
            "orderId": "order-123",
            "status": "FILLED",
            "executedQty": "3",
            "filledAvgPrice": "186.00",
            "commission": "0",
            "commissionAsset": "USD",
        }

        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)
        mock_settings.DRY_RUN = False

        await engine._check_open_orders("AAPL", strat)

        # Verify trade was saved with PnL = (186 - 180) * 3 = $18
        save_trade_call = engine.repository.save_trade
        assert save_trade_call.called
        trade: Trade = save_trade_call.call_args[0][0]
        assert trade.pnl == Decimal("18.00")
        assert trade.price == Decimal("186.00")  # Fix #5: actual fill price


# ══════════════════════════════════════════════════════════════════════════════
# Fix #3: Daily Trade Count
# ══════════════════════════════════════════════════════════════════════════════


class TestDailyTradeCount:
    """Daily trade counter is incremented and reset properly."""

    def test_record_trade_increments_count(self, mock_settings: Settings) -> None:
        rm = RiskManager(mock_settings)
        assert rm._daily_trade_count == 0
        rm.record_trade()
        assert rm._daily_trade_count == 1
        rm.record_trade()
        assert rm._daily_trade_count == 2

    @pytest.mark.asyncio
    async def test_daily_trade_limit_blocks(self, mock_settings: Settings) -> None:
        """After hitting MAX_DAILY_TRADES, orders are rejected."""
        mock_settings.MAX_DAILY_TRADES = 3
        rm = RiskManager(mock_settings)
        rm.current_balance = Decimal("50000")
        rm.peak_balance = Decimal("50000")

        rm._daily_trade_count = 3  # At limit

        order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            price=Decimal("185"), quantity=Decimal("2"), strategy="test",
        )

        allowed, reason = await rm.can_place_order(
            order,
            {"USD": Decimal("50000"), "EQUITY": Decimal("50000"), "DAYTRADE_COUNT": Decimal("0")},
        )
        assert not allowed
        assert "Daily trade limit" in reason

    def test_daily_count_resets_on_new_day(self, mock_settings: Settings) -> None:
        """Trade count resets when UTC date changes."""
        from datetime import datetime, timezone, timedelta
        rm = RiskManager(mock_settings)
        rm._daily_trade_count = 15
        rm.daily_pnl = Decimal("-100")
        # Force last reset to yesterday
        rm._last_reset_date = datetime.now(timezone.utc) - timedelta(days=1)

        rm.record_trade()  # Should trigger reset first

        # Count should be 1 (reset to 0, then incremented)
        assert rm._daily_trade_count == 1
        assert rm.daily_pnl == Decimal("0")


# ══════════════════════════════════════════════════════════════════════════════
# Fix #4: Dry-Run Fill Simulation
# ══════════════════════════════════════════════════════════════════════════════


class TestDryRunSimulation:
    """Dry-run mode simulates fills instead of leaving orders stuck."""

    @pytest.mark.asyncio
    async def test_dry_run_market_order_fills_immediately(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """Market orders in dry-run fill immediately at current price."""
        strat = _make_strategy_mock()
        buy_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            price=Decimal("185.50"), quantity=Decimal("3"), strategy="vwap_scalp",
        )
        buy_order.id = "DRY-12345678"
        buy_order.status = OrderStatus.OPEN
        strat.active_orders = [buy_order]

        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)
        mock_settings.DRY_RUN = True

        await engine._check_open_orders("AAPL", strat)

        # Order should now be filled
        assert buy_order.status == OrderStatus.FILLED
        # Strategy's on_order_filled should have been called
        strat.on_order_filled.assert_called_once_with(buy_order)
        # Trade count should be incremented
        assert mock_risk_manager._daily_trade_count == 1

    @pytest.mark.asyncio
    async def test_dry_run_limit_buy_fills_when_price_drops(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """Limit BUY at 180 fills when current price is 179."""
        strat = _make_strategy_mock()
        buy_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            price=Decimal("180.00"), quantity=Decimal("5"), strategy="vwap_scalp",
        )
        buy_order.id = "DRY-limit-buy"
        buy_order.status = OrderStatus.OPEN
        strat.active_orders = [buy_order]

        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)
        mock_settings.DRY_RUN = True
        # Price below limit → should fill
        engine.market_data.get_current_price = AsyncMock(return_value=Decimal("179.00"))

        await engine._check_open_orders("AAPL", strat)

        assert buy_order.status == OrderStatus.FILLED
        strat.on_order_filled.assert_called_once()

    @pytest.mark.asyncio
    async def test_dry_run_limit_buy_stays_open_when_price_above(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """Limit BUY at 180 does NOT fill when current price is 185."""
        strat = _make_strategy_mock()
        buy_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            price=Decimal("180.00"), quantity=Decimal("5"), strategy="vwap_scalp",
        )
        buy_order.id = "DRY-limit-buy2"
        buy_order.status = OrderStatus.OPEN
        strat.active_orders = [buy_order]

        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)
        mock_settings.DRY_RUN = True
        engine.market_data.get_current_price = AsyncMock(return_value=Decimal("185.00"))

        await engine._check_open_orders("AAPL", strat)

        assert buy_order.status == OrderStatus.OPEN  # Still open
        strat.on_order_filled.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_sell_computes_pnl(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """Dry-run sell fills compute PnL and update daily PnL."""
        strat = _make_strategy_mock()
        strat.positions = {
            "AAPL": Position(
                symbol="AAPL", side="LONG", entry_price=Decimal("180.00"),
                current_price=Decimal("190.00"), quantity=Decimal("5"),
                strategy="vwap_scalp",
            )
        }

        sell_order = Order(
            symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.MARKET,
            price=Decimal("190.00"), quantity=Decimal("5"), strategy="vwap_scalp",
        )
        sell_order.id = "DRY-sell"
        sell_order.status = OrderStatus.OPEN
        strat.active_orders = [sell_order]

        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)
        mock_settings.DRY_RUN = True
        engine.market_data.get_current_price = AsyncMock(return_value=Decimal("190.00"))

        await engine._check_open_orders("AAPL", strat)

        # PnL = (190 - 180) * 5 = $50
        assert mock_risk_manager.daily_pnl == Decimal("50")


# ══════════════════════════════════════════════════════════════════════════════
# Fix #5: Use Actual Fill Prices
# ══════════════════════════════════════════════════════════════════════════════


class TestActualFillPrices:
    """Engine uses filledAvgPrice from exchange, not proposed price."""

    @pytest.mark.asyncio
    async def test_fill_price_used_for_trade_record(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """Trade record uses actual fill price, not order.price."""
        strat = _make_strategy_mock()
        buy_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            price=Decimal("185.50"), quantity=Decimal("3"), strategy="vwap_scalp",
        )
        buy_order.id = "order-fill"
        buy_order.status = OrderStatus.OPEN
        strat.active_orders = [buy_order]

        # Exchange filled at 186.00, not 185.50
        mock_exchange_client.get_order_status.return_value = {
            "orderId": "order-fill",
            "status": "FILLED",
            "executedQty": "3",
            "filledAvgPrice": "186.00",
            "commission": "0.50",
            "commissionAsset": "USD",
        }

        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)
        mock_settings.DRY_RUN = False

        await engine._check_open_orders("AAPL", strat)

        trade: Trade = engine.repository.save_trade.call_args[0][0]
        assert trade.price == Decimal("186.00")  # Actual fill price
        assert trade.fee == Decimal("0.50")

    @pytest.mark.asyncio
    async def test_fallback_to_proposed_price_when_no_fill_price(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """When exchange doesn't return filledAvgPrice, fall back to order.price."""
        strat = _make_strategy_mock()
        buy_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            price=Decimal("185.50"), quantity=Decimal("3"), strategy="vwap_scalp",
        )
        buy_order.id = "order-nofill"
        buy_order.status = OrderStatus.OPEN
        strat.active_orders = [buy_order]

        mock_exchange_client.get_order_status.return_value = {
            "orderId": "order-nofill",
            "status": "FILLED",
            "executedQty": "3",
            # No filledAvgPrice
            "commission": "0",
        }

        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)
        mock_settings.DRY_RUN = False

        await engine._check_open_orders("AAPL", strat)

        trade: Trade = engine.repository.save_trade.call_args[0][0]
        assert trade.price == Decimal("185.50")  # Fallback to proposed


# ══════════════════════════════════════════════════════════════════════════════
# Fix #6: Emergency Shutdown Flattens Positions
# ══════════════════════════════════════════════════════════════════════════════


class TestEmergencyShutdownFlattens:
    """Emergency shutdown now closes all positions via exchange."""

    @pytest.mark.asyncio
    async def test_emergency_calls_close_all_positions(
        self, mock_settings: Settings
    ) -> None:
        """When exchange is provided, emergency_shutdown tries to flatten."""
        rm = RiskManager(mock_settings)
        exchange = AsyncMock()
        exchange.close_all_positions = AsyncMock(return_value=[])

        await rm.emergency_shutdown("drawdown exceeded", exchange=exchange)

        assert rm.is_halted is True
        assert rm.halt_reason == "drawdown exceeded"
        exchange.close_all_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_emergency_without_exchange_still_halts(
        self, mock_settings: Settings
    ) -> None:
        """Without exchange, emergency_shutdown just halts (no flatten)."""
        rm = RiskManager(mock_settings)
        await rm.emergency_shutdown("test halt")

        assert rm.is_halted is True
        assert rm.halt_reason == "test halt"

    @pytest.mark.asyncio
    async def test_emergency_flatten_failure_doesnt_crash(
        self, mock_settings: Settings
    ) -> None:
        """If flatten fails, shutdown still completes (is_halted = True)."""
        rm = RiskManager(mock_settings)
        exchange = AsyncMock()
        exchange.close_all_positions.side_effect = Exception("API down")

        await rm.emergency_shutdown("drawdown exceeded", exchange=exchange)

        assert rm.is_halted is True  # Still halted even if flatten fails

    @pytest.mark.asyncio
    async def test_engine_passes_exchange_to_emergency(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """Engine's emergency shutdown passes its exchange to risk manager."""
        strat = _make_strategy_mock()
        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)
        mock_settings.DRY_RUN = False
        mock_settings.POLL_INTERVAL_SECONDS = 0
        mock_settings.MARKET_HOURS_ONLY = False

        # Make the entire _tick raise to trigger consecutive error count
        engine._tick = AsyncMock(side_effect=RuntimeError("boom"))

        # Capture emergency_shutdown calls
        calls: list[dict] = []
        async def capture_shutdown(reason, exchange=None):
            calls.append({"reason": reason, "exchange": exchange})
            mock_risk_manager.is_halted = True

        mock_risk_manager.emergency_shutdown = capture_shutdown

        task = asyncio.create_task(engine.run())
        await asyncio.wait_for(task, timeout=10)

        # Should have called emergency_shutdown with exchange
        assert len(calls) == 1
        assert calls[0]["exchange"] is mock_exchange_client
