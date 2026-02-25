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
from src.risk.position_sizer import PositionSizer, SizingResult
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

        # ── Position Sizer (v5: wired for Kelly + 2% risk) ───────────────
        self._position_sizer: PositionSizer | None = None  # Set by engine

        # ── Progressive Risk Scaling (v5: reduce size after losses) ──────
        self._consecutive_losses: int = 0
        self._today_date: str = ""  # Reset losses each new day

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
        """Multi-layer trend filter (inspired by Alpaca chatGPT example).

        Layer 1 – EMA trend on intraday timeframe (original filter).
        Layer 2 – Multi-MA daily trend (50/100/200 SMA alignment).

        Both layers must agree for a BUY to proceed.  If either dataset
        is unavailable, that layer is skipped (fail-open).
        """
        if not self.settings.TREND_FILTER_ENABLED:
            return True

        # ── Layer 1: Intraday EMA ─────────────────────────────────────────
        try:
            tf = self.settings.TREND_FILTER_TIMEFRAME  # e.g. "15m"
            period = self.settings.TREND_FILTER_EMA_PERIOD
            bars = await self.exchange.get_klines(symbol, tf, period + 5)
            if len(bars) >= period:
                df = pd.DataFrame(bars)
                for col in ("open", "high", "low", "close", "volume"):
                    df[col] = df[col].astype(float)
                ema_series = indicators.ema(df, period)
                ema_value = Decimal(str(ema_series.iloc[-1]))
                if current_price < ema_value:
                    logger.info(
                        "[{}] Trend filter BLOCKED {} | price={} < EMA({})={}",
                        self.name, symbol, current_price, period, ema_value,
                    )
                    return False
        except Exception as exc:
            logger.warning("[{}] Intraday EMA filter error for {}: {}", self.name, symbol, exc)

        # ── Layer 2: Daily multi-MA alignment (50/100/200) ─────────────────
        try:
            daily_bars = await self.exchange.get_klines(symbol, "1D", 210)
            if len(daily_bars) >= 200:
                df_daily = pd.DataFrame(daily_bars)
                for col in ("open", "high", "low", "close", "volume"):
                    df_daily[col] = df_daily[col].astype(float)
                trend = indicators.multi_ma_trend(df_daily)
                # Require at least price > MA-50 AND MA-50 > MA-100
                if not trend["above_ma50"]:
                    logger.info(
                        "[{}] Daily MA filter BLOCKED {} | price below 50-day MA",
                        self.name, symbol,
                    )
                    return False
                if not trend["ma50_above_ma100"]:
                    logger.info(
                        "[{}] Daily MA filter BLOCKED {} | 50-MA below 100-MA (bearish structure)",
                        self.name, symbol,
                    )
                    return False
                if trend["aligned_bullish"]:
                    logger.debug(
                        "[{}] Daily MA filter STRONG for {} — fully aligned bullish",
                        self.name, symbol,
                    )
        except Exception as exc:
            logger.warning("[{}] Daily MA filter error for {}: {}", self.name, symbol, exc)

        return True  # all layers passed (or failed-open)

    def _passes_time_filter(self) -> bool:
        """Return True if current time is outside the midday dead zone, or filter is disabled."""
        if not self.settings.AVOID_MIDDAY:
            return True
        try:
            from zoneinfo import ZoneInfo
            now_et = datetime.now(ZoneInfo("America/New_York"))
            hour = now_et.hour
            if self.settings.MIDDAY_START_HOUR <= hour < self.settings.MIDDAY_END_HOUR:
                logger.info(
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

    # ── Dynamic Position Sizing (v5: Kelly + 2% risk + progressive) ──────────

    def set_position_sizer(self, sizer: PositionSizer) -> None:
        """Inject the shared PositionSizer (called by engine at startup)."""
        self._position_sizer = sizer

    async def compute_dynamic_quantity(
        self,
        symbol: str,
        current_price: Decimal,
        fallback_usd: float,
        stop_loss_pct: float | None = None,
    ) -> Decimal:
        """Compute position size using PositionSizer (Kelly/fixed-risk/heat).

        Falls back to ``fallback_usd / current_price`` if PositionSizer is
        unavailable or returns zero.

        Args:
            symbol: Trading symbol.
            current_price: Current market price.
            fallback_usd: Dollar amount for fallback sizing.
            stop_loss_pct: Stop-loss percent for this strategy (e.g. 1.0).
                           Used to compute risk-per-share for Kelly/fixed-risk.
        """
        price_f = float(current_price)
        fallback_qty = Decimal(str(fallback_usd)) / current_price

        # Reset consecutive losses at day boundary
        self._reset_daily_losses()

        if self._position_sizer is None or not getattr(self.settings, "KELLY_SIZING_ENABLED", False):
            qty = fallback_qty
        else:
            # Compute stop-loss price from configured %
            sl_pct = stop_loss_pct or getattr(self.settings, "STOP_LOSS_PERCENT", 2.0)
            stop_price = price_f * (1.0 - sl_pct / 100.0)

            try:
                result: SizingResult = self._position_sizer.calculate_size(
                    symbol=symbol,
                    entry_price=price_f,
                    stop_loss=stop_price,
                    strategy=self.name,
                )
                if result.quantity > Decimal("0"):
                    qty = result.quantity
                    logger.info(
                        "[{}] Dynamic sizing {} | method={} qty={} risk=${:.0f} kelly={:.4f} heat={:.2%}",
                        self.name, symbol, result.method, qty,
                        result.risk_per_trade, result.kelly_fraction,
                        result.portfolio_heat,
                    )
                else:
                    logger.info(
                        "[{}] PositionSizer returned 0 for {} — using fallback | notes={}",
                        self.name, symbol, result.notes,
                    )
                    qty = fallback_qty
            except Exception as exc:
                logger.warning(
                    "[{}] PositionSizer error for {}: {} — using fallback",
                    self.name, symbol, exc,
                )
                qty = fallback_qty

        # ── Progressive risk scaling (reduce after consecutive losses) ────
        if getattr(self.settings, "PROGRESSIVE_RISK_ENABLED", False) and self._consecutive_losses > 0:
            multiplier = getattr(self.settings, "PROGRESSIVE_LOSS_MULTIPLIER", 0.75)
            min_mult = getattr(self.settings, "PROGRESSIVE_MIN_MULTIPLIER", 0.25)
            scale = max(min_mult, multiplier ** self._consecutive_losses)
            qty = Decimal(str(float(qty) * scale))
            logger.info(
                "[{}] Progressive risk: {} consec losses -> {:.0%} size for {}",
                self.name, self._consecutive_losses, scale, symbol,
            )

        # Floor
        qty = max(qty, Decimal("0.01"))
        return qty

    # ── Confluence Gate (v5: multi-indicator quality filter) ──────────────────

    async def passes_confluence_gate(self, symbol: str) -> bool:
        """Return True if confluence score meets minimum threshold.

        Fetches 5-min bars and computes the 8-indicator confluence score.
        If disabled or data unavailable, returns True (fail-open).
        """
        if not getattr(self.settings, "CONFLUENCE_GATE_ENABLED", False):
            return True

        min_score = getattr(self.settings, "CONFLUENCE_MIN_SCORE", 30)
        bars_needed = getattr(self.settings, "CONFLUENCE_BARS_NEEDED", 60)

        try:
            from src.data.indicators_advanced import confluence_score

            bars = await self.exchange.get_klines(symbol, "5m", bars_needed)
            if len(bars) < 55:
                return True  # Not enough data, fail-open

            df = pd.DataFrame(bars)
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)

            result = confluence_score(df)
            score = result["score"]

            if score < min_score:
                logger.info(
                    "[{}] Confluence BLOCKED {} | score={}/100 (min={}) signals={}",
                    self.name, symbol, score, min_score, result["signals"],
                )
                return False

            logger.debug(
                "[{}] Confluence OK {} | score={}/100 ({})",
                self.name, symbol, score,
                "strong" if result["strong"] else "moderate" if result["moderate"] else "weak",
            )
            return True
        except Exception as exc:
            logger.warning("[{}] Confluence gate error for {}: {} — allowing", self.name, symbol, exc)
            return True  # Fail-open

    # ── Progressive Loss Tracking (v5) ────────────────────────────────────────

    def record_loss(self) -> None:
        """Record a losing trade for progressive risk scaling."""
        self._consecutive_losses += 1
        logger.info(
            "[{}] Consecutive losses: {}", self.name, self._consecutive_losses,
        )

    def record_win(self) -> None:
        """Record a winning trade — resets consecutive loss counter."""
        if self._consecutive_losses > 0:
            logger.info(
                "[{}] Win after {} losses — resetting progressive risk",
                self.name, self._consecutive_losses,
            )
        self._consecutive_losses = 0

    def _reset_daily_losses(self) -> None:
        """Reset consecutive loss counter at day boundary."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._today_date != today:
            if self._consecutive_losses > 0:
                logger.info(
                    "[{}] New day — resetting {} consecutive losses",
                    self.name, self._consecutive_losses,
                )
            self._consecutive_losses = 0
            self._today_date = today
