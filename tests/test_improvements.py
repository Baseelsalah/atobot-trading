"""Tests for the 6 new improvements:
1. Stale order cleanup
2. Daily summary notification
3. Graceful shutdown (signal handlers + daily summary in stop)
4. Partial fill handling
5. ATR-based position sizing
6. Rate-limit / retry layer
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import Settings
from src.core.engine import TradingEngine
from src.data.market_data import MarketDataProvider
from src.exchange.alpaca_client import _RateLimiter
from src.models.order import Order, OrderSide, OrderStatus, OrderType
from src.models.position import Position
from src.models.trade import Trade
from src.persistence.repository import TradingRepository
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_strategy_mock(name: str = "vwap_scalp") -> AsyncMock:
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
    settings: Settings,
    exchange: AsyncMock,
    risk: RiskManager,
    strategy: AsyncMock | None = None,
    notifier: AsyncMock | None = None,
) -> TradingEngine:
    strat = strategy or _make_strategy_mock()
    md = AsyncMock(spec=MarketDataProvider)
    md.get_current_price = AsyncMock(return_value=Decimal("185.50"))
    repo = AsyncMock(spec=TradingRepository)

    return TradingEngine(
        exchange=exchange,
        strategy=strat,
        risk_manager=risk,
        market_data=md,
        repository=repo,
        notifier=notifier,
        settings=settings,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. Stale Order Cleanup
# ══════════════════════════════════════════════════════════════════════════════


class TestStaleOrderCleanup:
    """Limit orders older than STALE_ORDER_MAX_AGE_SECONDS are auto-cancelled."""

    @pytest.mark.asyncio
    async def test_stale_limit_order_is_cancelled(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        strat = _make_strategy_mock()
        old_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            price=Decimal("180.00"), quantity=Decimal("3"), strategy="vwap_scalp",
        )
        old_order.id = "order-stale"
        old_order.status = OrderStatus.OPEN
        old_order.created_at = datetime.now(timezone.utc) - timedelta(seconds=2000)
        strat.active_orders = [old_order]

        mock_settings.STALE_ORDER_MAX_AGE_SECONDS = 1800
        mock_settings.DRY_RUN = False
        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)

        await engine._cancel_stale_orders()

        assert old_order.status == OrderStatus.CANCELLED
        mock_exchange_client.cancel_order.assert_called_once_with("AAPL", "order-stale")
        strat.on_order_cancelled.assert_called_once_with(old_order)

    @pytest.mark.asyncio
    async def test_fresh_limit_order_not_cancelled(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        strat = _make_strategy_mock()
        new_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            price=Decimal("180.00"), quantity=Decimal("3"), strategy="vwap_scalp",
        )
        new_order.id = "order-fresh"
        new_order.status = OrderStatus.OPEN
        # Created just now — should NOT be cancelled
        strat.active_orders = [new_order]

        mock_settings.STALE_ORDER_MAX_AGE_SECONDS = 1800
        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)

        await engine._cancel_stale_orders()

        assert new_order.status == OrderStatus.OPEN
        mock_exchange_client.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_market_orders_never_cancelled_as_stale(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        strat = _make_strategy_mock()
        old_market = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET,
            price=Decimal("185.00"), quantity=Decimal("2"), strategy="vwap_scalp",
        )
        old_market.id = "order-market-old"
        old_market.status = OrderStatus.OPEN
        old_market.created_at = datetime.now(timezone.utc) - timedelta(seconds=5000)
        strat.active_orders = [old_market]

        mock_settings.STALE_ORDER_MAX_AGE_SECONDS = 1800
        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)

        await engine._cancel_stale_orders()

        assert old_market.status == OrderStatus.OPEN  # Not touched

    @pytest.mark.asyncio
    async def test_dry_run_stale_cleanup(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """In dry-run, stale orders are cancelled locally without exchange call."""
        strat = _make_strategy_mock()
        old_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            price=Decimal("180.00"), quantity=Decimal("3"), strategy="vwap_scalp",
        )
        old_order.id = "DRY-stale"
        old_order.status = OrderStatus.OPEN
        old_order.created_at = datetime.now(timezone.utc) - timedelta(seconds=2000)
        strat.active_orders = [old_order]

        mock_settings.STALE_ORDER_MAX_AGE_SECONDS = 1800
        mock_settings.DRY_RUN = True
        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)

        await engine._cancel_stale_orders()

        assert old_order.status == OrderStatus.CANCELLED
        mock_exchange_client.cancel_order.assert_not_called()  # No exchange call


# ══════════════════════════════════════════════════════════════════════════════
# 2. Daily Summary Notification
# ══════════════════════════════════════════════════════════════════════════════


class TestDailySummary:
    """send_daily_summary builds and sends an EOD report."""

    @pytest.mark.asyncio
    async def test_daily_summary_sent(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        notifier = AsyncMock()
        notifier.send_daily_summary = AsyncMock(return_value=True)
        engine = _make_engine(
            mock_settings, mock_exchange_client, mock_risk_manager,
            notifier=notifier,
        )
        mock_risk_manager.daily_pnl = Decimal("125.50")
        mock_risk_manager._daily_trade_count = 8
        mock_risk_manager._daily_wins = 5
        mock_risk_manager.current_balance = Decimal("50125.50")

        await engine.send_daily_summary()

        notifier.send_daily_summary.assert_called_once()
        summary = notifier.send_daily_summary.call_args[0][0]
        assert summary["trades"] == 8
        assert summary["win_rate"] == 62.5
        assert "125.50" in summary["pnl"]

    @pytest.mark.asyncio
    async def test_daily_summary_no_notifier(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """No error when notifier is None."""
        engine = _make_engine(
            mock_settings, mock_exchange_client, mock_risk_manager,
        )
        engine.notifier = None
        await engine.send_daily_summary()  # Should not raise

    @pytest.mark.asyncio
    async def test_daily_summary_zero_trades(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """Win rate is 0% when no trades happened."""
        notifier = AsyncMock()
        notifier.send_daily_summary = AsyncMock(return_value=True)
        engine = _make_engine(
            mock_settings, mock_exchange_client, mock_risk_manager,
            notifier=notifier,
        )
        mock_risk_manager.daily_pnl = Decimal("0")
        mock_risk_manager._daily_trade_count = 0
        mock_risk_manager._daily_wins = 0
        mock_risk_manager.current_balance = Decimal("50000")

        await engine.send_daily_summary()

        summary = notifier.send_daily_summary.call_args[0][0]
        assert summary["win_rate"] == 0.0
        assert summary["trades"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. Graceful Shutdown
# ══════════════════════════════════════════════════════════════════════════════


class TestGracefulShutdown:
    """Bot stop() sends daily summary and flattens positions."""

    @pytest.mark.asyncio
    async def test_stop_sends_daily_summary(
        self, mock_settings: Settings
    ) -> None:
        from src.core.bot import AtoBot

        bot = AtoBot(mock_settings)
        bot.engine = MagicMock()
        bot.engine.send_daily_summary = AsyncMock(return_value=None)
        bot.engine.stop = AsyncMock()
        bot.strategies = []
        bot.exchange = AsyncMock()
        bot.repository = None
        bot.notifier = None

        await bot.stop()

        bot.engine.send_daily_summary.assert_called_once()
        bot.engine.stop.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# 4. Partial Fill Handling
# ══════════════════════════════════════════════════════════════════════════════


class TestPartialFillHandling:
    """PARTIALLY_FILLED orders create trade records for new slices."""

    @pytest.mark.asyncio
    async def test_partial_fill_creates_trade(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        strat = _make_strategy_mock()
        buy_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            price=Decimal("185.00"), quantity=Decimal("10"), strategy="vwap_scalp",
        )
        buy_order.id = "order-partial"
        buy_order.status = OrderStatus.OPEN
        buy_order.filled_quantity = Decimal("0")  # Nothing filled yet
        strat.active_orders = [buy_order]

        mock_exchange_client.get_order_status.return_value = {
            "orderId": "order-partial",
            "status": "PARTIALLY_FILLED",
            "executedQty": "5",
            "filledAvgPrice": "185.25",
        }

        mock_settings.DRY_RUN = False
        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)

        await engine._check_open_orders("AAPL", strat)

        assert buy_order.status == OrderStatus.PARTIALLY_FILLED
        assert buy_order.filled_quantity == Decimal("5")
        # A trade for the partial fill should have been saved
        engine.repository.save_trade.assert_called_once()
        trade: Trade = engine.repository.save_trade.call_args[0][0]
        assert trade.quantity == Decimal("5")
        assert trade.price == Decimal("185.25")

    @pytest.mark.asyncio
    async def test_partial_fill_no_duplicate_on_same_qty(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """If executedQty hasn't changed, no new trade is created."""
        strat = _make_strategy_mock()
        buy_order = Order(
            symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            price=Decimal("185.00"), quantity=Decimal("10"), strategy="vwap_scalp",
        )
        buy_order.id = "order-partial2"
        buy_order.status = OrderStatus.PARTIALLY_FILLED
        buy_order.filled_quantity = Decimal("5")  # Already partially filled
        strat.active_orders = [buy_order]

        mock_exchange_client.get_order_status.return_value = {
            "orderId": "order-partial2",
            "status": "PARTIALLY_FILLED",
            "executedQty": "5",  # Same as before
            "filledAvgPrice": "185.25",
        }

        mock_settings.DRY_RUN = False
        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)

        await engine._check_open_orders("AAPL", strat)

        engine.repository.save_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_fill_sell_pnl(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """Partial sell fill computes PnL for the filled portion."""
        strat = _make_strategy_mock()
        strat.positions = {
            "AAPL": Position(
                symbol="AAPL", side="LONG", entry_price=Decimal("180.00"),
                current_price=Decimal("190.00"), quantity=Decimal("10"),
                strategy="vwap_scalp",
            )
        }

        sell_order = Order(
            symbol="AAPL", side=OrderSide.SELL, order_type=OrderType.LIMIT,
            price=Decimal("190.00"), quantity=Decimal("10"), strategy="vwap_scalp",
        )
        sell_order.id = "order-partial-sell"
        sell_order.status = OrderStatus.OPEN
        sell_order.filled_quantity = Decimal("0")
        strat.active_orders = [sell_order]

        mock_exchange_client.get_order_status.return_value = {
            "orderId": "order-partial-sell",
            "status": "PARTIALLY_FILLED",
            "executedQty": "4",
            "filledAvgPrice": "190.50",
        }

        mock_settings.DRY_RUN = False
        engine = _make_engine(mock_settings, mock_exchange_client, mock_risk_manager, strat)

        await engine._check_open_orders("AAPL", strat)

        trade: Trade = engine.repository.save_trade.call_args[0][0]
        # PnL = (190.50 - 180) * 4 = $42
        assert trade.pnl == Decimal("42.00")
        assert trade.quantity == Decimal("4")


# ══════════════════════════════════════════════════════════════════════════════
# 5. ATR-based Position Sizing
# ══════════════════════════════════════════════════════════════════════════════


class TestATRPositionSizing:
    """compute_atr_quantity adjusts size based on ATR volatility."""

    @pytest.mark.asyncio
    async def test_atr_sizing_disabled_uses_fallback(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
    ) -> None:
        """When ATR_SIZING_ENABLED=False, uses fallback USD / price."""
        mock_settings.ATR_SIZING_ENABLED = False
        rm = RiskManager(mock_settings)
        from src.strategies.vwap_strategy import VWAPScalpStrategy
        strat = VWAPScalpStrategy(mock_exchange_client, rm, mock_settings)

        qty = await strat.compute_atr_quantity("AAPL", Decimal("200"), 500.0)
        assert qty == Decimal("500.0") / Decimal("200")

    @pytest.mark.asyncio
    async def test_atr_sizing_enabled(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
    ) -> None:
        """ATR sizing: qty = risk_dollars / ATR."""
        mock_settings.ATR_SIZING_ENABLED = True
        mock_settings.ATR_SIZING_PERIOD = 14
        mock_settings.ATR_SIZING_TIMEFRAME = "5m"
        mock_settings.ATR_RISK_DOLLARS = 50.0
        mock_settings.MAX_POSITION_SIZE_USD = 2000.0

        # Return enough bars for ATR(14)
        bars = []
        for i in range(20):
            bars.append({
                "timestamp": 1700000000000 + i * 300000,
                "open": Decimal("185") + Decimal(str(i % 3)),
                "high": Decimal("187") + Decimal(str(i % 3)),
                "low": Decimal("183") + Decimal(str(i % 3)),
                "close": Decimal("185.50") + Decimal(str(i % 3)),
                "volume": Decimal("50000"),
            })
        mock_exchange_client.get_klines.return_value = bars

        rm = RiskManager(mock_settings)
        from src.strategies.vwap_strategy import VWAPScalpStrategy
        strat = VWAPScalpStrategy(mock_exchange_client, rm, mock_settings)

        qty = await strat.compute_atr_quantity("AAPL", Decimal("185"), 500.0)
        # ATR based on the bars, qty = 50 / ATR, should be > 0
        assert qty > Decimal("0")
        assert qty <= Decimal("2000") / Decimal("185")  # Capped at max position

    @pytest.mark.asyncio
    async def test_atr_sizing_insufficient_bars(
        self, mock_settings: Settings, mock_exchange_client: AsyncMock,
    ) -> None:
        """Falls back to fixed size when not enough bars for ATR."""
        mock_settings.ATR_SIZING_ENABLED = True
        mock_settings.ATR_SIZING_PERIOD = 14
        mock_settings.ATR_SIZING_TIMEFRAME = "5m"
        mock_settings.ATR_RISK_DOLLARS = 50.0
        mock_exchange_client.get_klines.return_value = [
            {"timestamp": 1700000000000, "open": 185, "high": 186,
             "low": 184, "close": 185.5, "volume": 50000}
        ]

        rm = RiskManager(mock_settings)
        from src.strategies.vwap_strategy import VWAPScalpStrategy
        strat = VWAPScalpStrategy(mock_exchange_client, rm, mock_settings)

        qty = await strat.compute_atr_quantity("AAPL", Decimal("185"), 500.0)
        assert qty == Decimal("500.0") / Decimal("185")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Rate Limiter
# ══════════════════════════════════════════════════════════════════════════════


class TestRateLimiter:
    """_RateLimiter correctly throttles API calls."""

    @pytest.mark.asyncio
    async def test_under_limit_no_wait(self) -> None:
        """Calls within rate limit proceed immediately."""
        rl = _RateLimiter(max_calls=10, window_seconds=60.0)
        start = time.monotonic()
        for _ in range(5):
            await rl.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 1.0  # Should be near-instant

    @pytest.mark.asyncio
    async def test_at_limit_throttles(self) -> None:
        """When limit is hit, acquire() waits."""
        rl = _RateLimiter(max_calls=3, window_seconds=1.0)
        # Fill up the window
        for _ in range(3):
            await rl.acquire()

        start = time.monotonic()
        await rl.acquire()  # 4th call should wait ~1s
        elapsed = time.monotonic() - start
        assert elapsed >= 0.5  # Should have waited


# ══════════════════════════════════════════════════════════════════════════════
# Risk Manager: _daily_wins tracking
# ══════════════════════════════════════════════════════════════════════════════


class TestDailyWinsTracking:
    """record_trade() tracks winning and losing trades."""

    def test_winning_trade_increments_wins(self, mock_settings: Settings) -> None:
        rm = RiskManager(mock_settings)
        rm.record_trade(Decimal("10"))
        assert rm._daily_wins == 1
        assert rm._daily_trade_count == 1

    def test_losing_trade_no_win(self, mock_settings: Settings) -> None:
        rm = RiskManager(mock_settings)
        rm.record_trade(Decimal("-5"))
        assert rm._daily_wins == 0
        assert rm._daily_trade_count == 1

    def test_none_pnl_no_win(self, mock_settings: Settings) -> None:
        rm = RiskManager(mock_settings)
        rm.record_trade(None)
        assert rm._daily_wins == 0
        assert rm._daily_trade_count == 1

    def test_daily_wins_reset_on_new_day(self, mock_settings: Settings) -> None:
        rm = RiskManager(mock_settings)
        rm._daily_wins = 10
        rm._daily_trade_count = 15
        rm._last_reset_date = datetime.now(timezone.utc) - timedelta(days=1)

        rm.record_trade(Decimal("5"))

        assert rm._daily_wins == 1  # Reset to 0, then +1
        assert rm._daily_trade_count == 1


# ══════════════════════════════════════════════════════════════════════════════
# Settings: New fields have correct defaults
# ══════════════════════════════════════════════════════════════════════════════


class TestNewSettings:
    def test_stale_order_default(self, mock_settings: Settings) -> None:
        s = Settings(
            ALPACA_API_KEY="k", ALPACA_API_SECRET="s",
            NOTIFICATIONS_ENABLED=False,
        )
        assert s.STALE_ORDER_MAX_AGE_SECONDS == 1800

    def test_atr_sizing_defaults(self, mock_settings: Settings) -> None:
        s = Settings(
            ALPACA_API_KEY="k", ALPACA_API_SECRET="s",
            NOTIFICATIONS_ENABLED=False,
        )
        assert s.ATR_SIZING_ENABLED is False
        assert s.ATR_SIZING_PERIOD == 14
        assert s.ATR_RISK_DOLLARS == 500.0
