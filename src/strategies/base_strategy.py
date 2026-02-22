"""Abstract base strategy for AtoBot Trading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
from loguru import logger

from src.config.settings import Settings
from src.data import indicators
from src.exchange.base_client import BaseExchangeClient
from src.models.order import Order
from src.models.position import Position
from src.risk.risk_manager import RiskManager


class BaseStrategy(ABC):
    """Every trading strategy must inherit from this class."""

    def __init__(
        self,
        exchange_client: BaseExchangeClient,
        risk_manager: RiskManager,
        settings: Settings,
    ) -> None:
        self.exchange = exchange_client
        self.risk = risk_manager
        self.settings = settings
        self.active_orders: list[Order] = []
        self.positions: dict[str, Position] = {}  # symbol -> Position
        self.is_running: bool = False
        # Trailing stop tracking: symbol -> highest price since entry
        self._trailing_highs: dict[str, Decimal] = {}

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    async def initialize(self, symbol: str) -> None:
        """Set up the strategy for a symbol."""
        ...

    @abstractmethod
    async def on_tick(self, symbol: str, current_price: Decimal) -> list[Order]:
        """Called every poll interval. Return list of orders to place."""
        ...

    @abstractmethod
    async def on_order_filled(self, order: Order) -> list[Order]:
        """Called when an order is filled. Return follow-up orders."""
        ...

    @abstractmethod
    async def on_order_cancelled(self, order: Order) -> None:
        """Called when an order is cancelled."""
        ...

    @abstractmethod
    async def get_status(self) -> dict:
        """Return current strategy status for dashboard / notifications."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    # ── Shared helpers ────────────────────────────────────────────────────────

    async def cancel_all(self, symbol: str) -> None:
        """Cancel all active orders for this strategy on *symbol*."""
        to_cancel = [
            o for o in self.active_orders if o.symbol == symbol and o.is_active
        ]
        for order in to_cancel:
            if order.id:
                try:
                    await self.exchange.cancel_order(symbol, order.id)
                    order.mark_cancelled()
                    logger.info(
                        "[{}] Cancelled order {} for {}",
                        self.name,
                        order.internal_id,
                        symbol,
                    )
                except Exception as exc:
                    logger.error(
                        "[{}] Failed to cancel order {}: {}",
                        self.name,
                        order.internal_id,
                        exc,
                    )
            else:
                order.mark_cancelled()
        self.active_orders = [o for o in self.active_orders if o.is_active]

    def _get_active_orders_for(self, symbol: str) -> list[Order]:
        """Return active orders for a specific symbol."""
        return [o for o in self.active_orders if o.symbol == symbol and o.is_active]

    def _get_position(self, symbol: str) -> Position | None:
        """Return the current position for *symbol*, or None."""
        return self.positions.get(symbol)

    # ── Shared entry filters ──────────────────────────────────────────────────

    async def _passes_trend_filter(self, symbol: str, current_price: Decimal) -> bool:
        """Return True if price is above the trend EMA (uptrend) or filter is disabled."""
        if not self.settings.TREND_FILTER_ENABLED:
            return True
        try:
            tf = self.settings.TREND_FILTER_TIMEFRAME  # e.g. "15m"
            period = self.settings.TREND_FILTER_EMA_PERIOD
            bars = await self.exchange.get_klines(symbol, tf, period + 5)
            if len(bars) < period:
                return True  # Not enough data → allow trade
            df = pd.DataFrame(bars)
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)
            ema_series = indicators.ema(df, period)
            ema_value = Decimal(str(ema_series.iloc[-1]))
            passes = current_price >= ema_value
            if not passes:
                logger.debug(
                    "[{}] Trend filter BLOCKED {} | price={} < EMA({})={}",
                    self.name, symbol, current_price, period, ema_value,
                )
            return passes
        except Exception as exc:
            logger.warning("[{}] Trend filter error for {}: {}", self.name, symbol, exc)
            return True  # On error, don't block

    def _passes_time_filter(self) -> bool:
        """Return True if current time is outside the midday dead zone, or filter is disabled."""
        if not self.settings.AVOID_MIDDAY:
            return True
        try:
            from zoneinfo import ZoneInfo
            now_et = datetime.now(ZoneInfo("America/New_York"))
            hour = now_et.hour
            if self.settings.MIDDAY_START_HOUR <= hour < self.settings.MIDDAY_END_HOUR:
                logger.debug(
                    "[{}] Time filter BLOCKED — midday dead zone ({:02d}:00 ET)",
                    self.name, hour,
                )
                return False
            return True
        except Exception:
            return True  # On error, don't block

    def _check_trailing_stop(
        self, symbol: str, pos: Position, current_price: Decimal
    ) -> bool:
        """Return True if trailing stop triggered. Updates highest price tracker."""
        if not self.settings.TRAILING_STOP_ENABLED:
            return False

        # Track highest price since entry
        prev_high = self._trailing_highs.get(symbol, pos.entry_price)
        if current_price > prev_high:
            self._trailing_highs[symbol] = current_price
            prev_high = current_price

        activation = Decimal(str(self.settings.TRAILING_STOP_ACTIVATION_PCT))
        distance = Decimal(str(self.settings.TRAILING_STOP_DISTANCE_PCT))

        # Only activate after minimum profit threshold
        profit_pct = ((prev_high - pos.entry_price) / pos.entry_price) * Decimal("100")
        if profit_pct < activation:
            return False

        # Trailing stop: price has fallen distance% from the high
        trail_stop_price = prev_high * (Decimal("1") - distance / Decimal("100"))
        if current_price <= trail_stop_price:
            logger.info(
                "[{}] TRAILING STOP {} | price={} < trail={} (high={})",
                self.name, symbol, current_price, trail_stop_price, prev_high,
            )
            return True
        return False

    def _reset_trailing_high(self, symbol: str) -> None:
        """Reset trailing high when position is closed."""
        self._trailing_highs.pop(symbol, None)

    # ── ATR-based position sizing ─────────────────────────────────────────────

    async def compute_atr_quantity(
        self, symbol: str, current_price: Decimal, fallback_usd: float
    ) -> Decimal:
        """Return position quantity adjusted for volatility via ATR.

        If ATR sizing is disabled or data is unavailable, falls back to
        ``fallback_usd / current_price``.

        Formula: ``quantity = ATR_RISK_DOLLARS / ATR_per_share``
        Capped at ``MAX_POSITION_SIZE_USD / current_price``.
        """
        if not getattr(self.settings, "ATR_SIZING_ENABLED", False):
            return Decimal(str(fallback_usd)) / current_price

        try:
            tf = self.settings.ATR_SIZING_TIMEFRAME  # e.g. "5m"
            period = self.settings.ATR_SIZING_PERIOD
            risk_dollars = Decimal(str(self.settings.ATR_RISK_DOLLARS))

            bars = await self.exchange.get_klines(symbol, tf, period + 5)
            if len(bars) < period + 1:
                logger.debug(
                    "[{}] ATR sizing: not enough bars for {} ({}), using fallback",
                    self.name, symbol, len(bars),
                )
                return Decimal(str(fallback_usd)) / current_price

            df = pd.DataFrame(bars)
            for col in ("high", "low", "close"):
                df[col] = df[col].astype(float)

            atr_series = indicators.atr(df, period)
            atr_value = Decimal(str(atr_series.iloc[-1]))

            if atr_value <= Decimal("0"):
                return Decimal(str(fallback_usd)) / current_price

            # qty = risk$ / ATR, capped at max position size
            qty = risk_dollars / atr_value
            max_qty = Decimal(str(self.settings.MAX_POSITION_SIZE_USD)) / current_price
            qty = min(qty, max_qty)
            qty = max(qty, Decimal("0.01"))  # minimum floor

            logger.debug(
                "[{}] ATR sizing {} | ATR={:.2f} | risk=${} | qty={:.2f} (max {:.2f})",
                self.name, symbol, atr_value, risk_dollars, qty, max_qty,
            )
            return qty
        except Exception as exc:
            logger.warning("[{}] ATR sizing error for {}: {} — using fallback", self.name, symbol, exc)
            return Decimal(str(fallback_usd)) / current_price
