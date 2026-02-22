"""Risk management module for AtoBot Trading."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from loguru import logger

from src.config.settings import Settings
from src.models.order import Order
from src.models.position import Position


class RiskManager:
    """Enforces risk limits before and during trading."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.daily_pnl: Decimal = Decimal("0")
        self.peak_balance: Decimal = Decimal("0")
        self.current_balance: Decimal = Decimal("0")
        self.is_halted: bool = False
        self.halt_reason: str = ""
        self._open_order_count: int = 0
        self._daily_trade_count: int = 0
        self._daily_wins: int = 0
        self._last_reset_date: datetime = datetime.now(timezone.utc)

    # ── Pre-trade checks ──────────────────────────────────────────────────────

    async def can_place_order(
        self,
        order: Order,
        current_balance: dict[str, Decimal],
    ) -> tuple[bool, str]:
        """Validate that an order passes all risk checks.

        Returns:
            (True, "") if the order is allowed.
            (False, reason) if the order is rejected.
        """
        # 1. Is the bot halted?
        if self.is_halted:
            return False, f"Trading halted: {self.halt_reason}"

        # 2. Does this order exceed MAX_POSITION_SIZE_USD?
        notional = order.price * order.quantity
        if notional > Decimal(str(self.settings.MAX_POSITION_SIZE_USD)):
            return False, (
                f"Order notional {notional} exceeds MAX_POSITION_SIZE_USD "
                f"({self.settings.MAX_POSITION_SIZE_USD})"
            )

        # 3. Does adding this order exceed MAX_OPEN_ORDERS?
        if self._open_order_count >= self.settings.MAX_OPEN_ORDERS:
            return False, (
                f"Open order count ({self._open_order_count}) "
                f"already at MAX_OPEN_ORDERS ({self.settings.MAX_OPEN_ORDERS})"
            )

        # 4. Has DAILY_LOSS_LIMIT_USD been reached?
        daily_limit = Decimal(str(self.settings.DAILY_LOSS_LIMIT_USD))
        if self.daily_pnl < Decimal("0") and abs(self.daily_pnl) >= daily_limit:
            return False, (
                f"Daily loss limit reached: {self.daily_pnl} "
                f"(limit: -{daily_limit})"
            )

        # 5. Has MAX_DRAWDOWN_PERCENT been reached?
        dd = self.current_drawdown_percent
        max_dd = Decimal(str(self.settings.MAX_DRAWDOWN_PERCENT))
        if dd >= max_dd:
            return False, (
                f"Max drawdown reached: {dd:.2f}% (limit: {max_dd}%)"
            )

        # 6. Sufficient balance?
        usd_balance = current_balance.get("USD", Decimal("0"))
        if order.side == "BUY" and notional > usd_balance:
            return False, (
                f"Insufficient USD balance: need {notional}, have {usd_balance}"
            )

        # 7. PDT protection (< 25k equity, max 3 day trades per 5 days)
        if self.settings.PDT_PROTECTION:
            dt_count = current_balance.get("DAYTRADE_COUNT", Decimal("0"))
            equity = current_balance.get("EQUITY", Decimal("0"))
            if equity < Decimal("25000") and dt_count >= Decimal("3"):
                return False, (
                    f"PDT protection: {dt_count} day trades, equity ${equity} < $25k"
                )

        # 8. Daily trade limit
        if hasattr(self.settings, 'MAX_DAILY_TRADES'):
            if self._daily_trade_count >= self.settings.MAX_DAILY_TRADES:
                return False, (
                    f"Daily trade limit reached: {self._daily_trade_count} "
                    f"(max: {self.settings.MAX_DAILY_TRADES})"
                )

        return True, ""

    # ── Stop-loss ─────────────────────────────────────────────────────────────

    async def check_stop_loss(self, position: Position) -> bool:
        """Return True if the position should be stopped out."""
        if position.is_closed:
            return False

        stop_pct = Decimal(str(self.settings.STOP_LOSS_PERCENT))
        if position.side == "LONG":
            loss_pct = (
                (position.entry_price - position.current_price)
                / position.entry_price
                * Decimal("100")
            )
        else:
            loss_pct = (
                (position.current_price - position.entry_price)
                / position.entry_price
                * Decimal("100")
            )

        if loss_pct >= stop_pct:
            logger.warning(
                "Stop-loss triggered for {} | loss={:.2f}% | limit={}%",
                position.symbol,
                loss_pct,
                stop_pct,
            )
            return True
        return False

    # ── PnL tracking ─────────────────────────────────────────────────────────

    async def update_daily_pnl(self, pnl: Decimal) -> None:
        """Add to running daily PnL. Resets at midnight UTC."""
        self._maybe_reset_daily()
        self.daily_pnl += pnl
        logger.debug("Daily PnL updated: {}", self.daily_pnl)

    def record_trade(self, pnl: Decimal | None = None) -> None:
        """Increment the daily trade counter. Call on every fill."""
        self._maybe_reset_daily()
        self._daily_trade_count += 1
        if pnl is not None and pnl > Decimal("0"):
            self._daily_wins += 1
        logger.debug("Daily trade count: {} (wins: {})", self._daily_trade_count, self._daily_wins)

    async def update_balance(self, balance: Decimal) -> None:
        """Track current balance and update peak for drawdown calc."""
        self.current_balance = balance
        if balance > self.peak_balance:
            self.peak_balance = balance
            logger.debug("New peak balance: {}", self.peak_balance)

    def set_open_order_count(self, count: int) -> None:
        """Update the tracked open order count."""
        self._open_order_count = count

    # ── Drawdown ──────────────────────────────────────────────────────────────

    @property
    def current_drawdown_percent(self) -> Decimal:
        """Calculate current drawdown from peak balance."""
        if self.peak_balance <= Decimal("0"):
            return Decimal("0")
        dd = (
            (self.peak_balance - self.current_balance) / self.peak_balance
        ) * Decimal("100")
        return max(dd, Decimal("0"))

    # ── Emergency ─────────────────────────────────────────────────────────────

    async def emergency_shutdown(
        self, reason: str, exchange: Any | None = None
    ) -> None:
        """Halt all trading and optionally flatten positions."""
        self.is_halted = True
        self.halt_reason = reason
        logger.critical("EMERGENCY SHUTDOWN: {}", reason)

        # Attempt to close all positions if exchange is available
        if exchange is not None and hasattr(exchange, "close_all_positions"):
            try:
                await exchange.close_all_positions()
                logger.info("Emergency flatten: all positions closed")
            except Exception as exc:
                logger.error("Emergency flatten FAILED: {}", exc)

    async def resume_trading(self) -> None:
        """Resume trading after a halt."""
        self.is_halted = False
        self.halt_reason = ""
        logger.info("Trading resumed")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _maybe_reset_daily(self) -> None:
        """Reset daily PnL and trade count if a new UTC day has started."""
        now = datetime.now(timezone.utc)
        if now.date() > self._last_reset_date.date():
            logger.info(
                "New trading day — resetting daily PnL (was {}) and trade count (was {})",
                self.daily_pnl,
                self._daily_trade_count,
            )
            self.daily_pnl = Decimal("0")
            self._daily_trade_count = 0
            self._daily_wins = 0
            self._last_reset_date = now
