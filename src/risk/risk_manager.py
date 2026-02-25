"""Risk management module for AtoBot Trading."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np
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

        # Correlation risk — daily returns cache: symbol -> np.array of returns
        self._returns_cache: dict[str, np.ndarray] = {}
        self._correlation_matrix: dict[tuple[str, str], float] = {}

        # VaR — portfolio daily returns history
        self._portfolio_returns: list[float] = []

        # ATR-adaptive stops — current ATR % for each symbol
        self._symbol_atr_pct: dict[str, float] = {}

        # Circuit breaker state
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_until: datetime | None = None
        self._spy_open_price: float | None = None  # SPY price at market open

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

        # 9. Correlation-based exposure limit
        if getattr(self.settings, "CORRELATION_RISK_ENABLED", False):
            corr_ok, corr_reason = self._check_correlation_exposure(
                order.symbol, float(notional)
            )
            if not corr_ok:
                return False, corr_reason

        # 10. Value-at-Risk check
        if getattr(self.settings, "VAR_ENABLED", False):
            var_ok, var_reason = self._check_var(float(notional))
            if not var_ok:
                return False, var_reason

        return True, ""

    # ── Stop-loss ─────────────────────────────────────────────────────────────

    async def check_stop_loss(self, position: Position) -> bool:
        """Return True if the position should be stopped out.

        Uses ATR-adaptive widening: when a symbol's ATR% is elevated
        (above ATR_NORMAL_BASELINE_PCT), the SL is widened proportionally
        up to ATR_STOP_MAX_WIDENING × baseline.  This prevents getting
        whipsawed out by normal noise in high-vol regimes.
        """
        if position.is_closed:
            return False

        stop_pct = Decimal(str(self.settings.STOP_LOSS_PERCENT))

        # ATR-adaptive widening
        if getattr(self.settings, "ATR_ADAPTIVE_STOPS", False):
            atr_pct = self._symbol_atr_pct.get(position.symbol, 0.0)
            baseline = getattr(self.settings, "ATR_NORMAL_BASELINE_PCT", 0.3)
            max_widening = getattr(self.settings, "ATR_STOP_MAX_WIDENING", 3.0)
            multiplier = getattr(self.settings, "ATR_STOP_MULTIPLIER", 1.5)

            if atr_pct > 0 and baseline > 0 and atr_pct > baseline:
                # How many times higher is current ATR vs normal?
                vol_ratio = atr_pct / baseline
                # Widen SL proportionally: e.g. 2x vol → 2x SL, capped
                widening = min(vol_ratio * multiplier, max_widening)
                widened_pct = Decimal(str(float(stop_pct) * widening))
                if widened_pct > stop_pct:
                    logger.debug(
                        "ATR-adaptive SL for {}: {:.2f}% -> {:.2f}% "
                        "(ATR={:.2f}% vs baseline={:.2f}%, widening={:.1f}x)",
                        position.symbol, stop_pct, widened_pct,
                        atr_pct, baseline, widening,
                    )
                    stop_pct = widened_pct

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

    def update_symbol_atr(self, symbol: str, atr_pct: float) -> None:
        """Store the latest ATR % for a symbol (called from engine)."""
        self._symbol_atr_pct[symbol] = atr_pct

    # ── Circuit Breaker ───────────────────────────────────────────────────────

    def update_spy_open(self, price: float) -> None:
        """Set SPY's opening price for circuit breaker calculation."""
        self._spy_open_price = price

    def check_circuit_breaker(self, spy_price: float) -> bool:
        """Check if SPY has dropped enough to trigger a circuit breaker pause.

        Returns True if the circuit breaker is ACTIVE (trading should pause).
        """
        if not getattr(self.settings, "CIRCUIT_BREAKER_ENABLED", True):
            return False

        # Check if existing circuit breaker has expired
        if self._circuit_breaker_until:
            if datetime.now(timezone.utc) >= self._circuit_breaker_until:
                if self._circuit_breaker_active:
                    logger.info("Circuit breaker cooldown expired — resuming entries")
                self._circuit_breaker_active = False
                self._circuit_breaker_until = None
            else:
                return True  # Still paused

        if self._spy_open_price and self._spy_open_price > 0:
            drop_pct = (self._spy_open_price - spy_price) / self._spy_open_price * 100
            threshold = getattr(self.settings, "CIRCUIT_BREAKER_SPY_DROP_PCT", 4.0)
            if drop_pct >= threshold:
                pause_min = getattr(self.settings, "CIRCUIT_BREAKER_PAUSE_MINUTES", 30)
                self._circuit_breaker_active = True
                from datetime import timedelta as _td
                self._circuit_breaker_until = datetime.now(timezone.utc) + \
                    _td(minutes=pause_min)
                logger.warning(
                    "CIRCUIT BREAKER TRIGGERED — SPY dropped {:.1f}% "
                    "(open={:.2f}, now={:.2f}). Pausing entries for {}min",
                    drop_pct, self._spy_open_price, spy_price, pause_min,
                )
                return True

        return False

    @property
    def is_circuit_breaker_active(self) -> bool:
        """Return True if the circuit breaker pause is currently active."""
        if self._circuit_breaker_until:
            if datetime.now(timezone.utc) < self._circuit_breaker_until:
                return True
            self._circuit_breaker_active = False
            self._circuit_breaker_until = None
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

    # ── Correlation Risk ────────────────────────────────────────────────────

    def update_returns_cache(
        self, symbol: str, daily_returns: list[float] | np.ndarray
    ) -> None:
        """Store daily returns for a symbol (called from engine after fetching OHLCV)."""
        self._returns_cache[symbol] = np.array(daily_returns, dtype=float)

    def _check_correlation_exposure(
        self, new_symbol: str, new_notional: float
    ) -> tuple[bool, str]:
        """Block if adding this symbol would breach correlated-exposure limit.

        For every open position whose Pearson r with *new_symbol* exceeds
        CORRELATION_THRESHOLD, sum their notionals + the new trade's notional.
        If that sum exceeds MAX_CORRELATED_EXPOSURE × account, reject.
        """
        threshold = getattr(self.settings, "CORRELATION_THRESHOLD", 0.70)
        max_pct = getattr(self.settings, "MAX_CORRELATED_EXPOSURE", 0.40)
        if self.current_balance <= 0:
            return True, ""

        account = float(self.current_balance)
        max_usd = account * max_pct

        # Gather notionals of correlated open positions
        correlated_notional = new_notional
        new_ret = self._returns_cache.get(new_symbol)
        if new_ret is None or len(new_ret) < 10:
            # Not enough data to compute correlation — allow trade
            return True, ""

        from src.risk.position_sizer import PositionSizer  # avoid circular at module level

        for sym, pos_info in list(self._open_positions_snapshot.items()):
            if sym == new_symbol:
                continue
            cached_ret = self._returns_cache.get(sym)
            if cached_ret is None or len(cached_ret) < 10:
                continue
            # Align lengths
            n = min(len(new_ret), len(cached_ret))
            r = np.corrcoef(new_ret[-n:], cached_ret[-n:])[0, 1]
            if math.isnan(r):
                continue
            if abs(r) >= threshold:
                correlated_notional += pos_info.get("notional", 0.0)

        if correlated_notional > max_usd:
            return False, (
                f"Correlated exposure ${correlated_notional:,.0f} would exceed "
                f"limit ${max_usd:,.0f} ({max_pct:.0%} of ${account:,.0f})"
            )
        return True, ""

    def set_open_positions_snapshot(
        self, positions: dict[str, dict]
    ) -> None:
        """Receive open-positions map from PositionSizer for correlation checks."""
        self._open_positions_snapshot = positions

    # ── Value-at-Risk ─────────────────────────────────────────────────────────

    def record_portfolio_return(self, daily_return_pct: float) -> None:
        """Append one daily portfolio return (e.g. 0.012 = +1.2%)."""
        self._portfolio_returns.append(daily_return_pct)
        # Keep last 252 trading days
        if len(self._portfolio_returns) > 252:
            self._portfolio_returns = self._portfolio_returns[-252:]

    def _check_var(self, new_notional: float) -> tuple[bool, str]:
        """Block trade if adding it would push daily VaR above limit.

        Uses historical VaR on portfolio returns.  If insufficient history,
        allows trade (conservative bootstrap phase).
        """
        lookback = getattr(self.settings, "VAR_LOOKBACK_DAYS", 30)
        confidence = getattr(self.settings, "VAR_CONFIDENCE", 0.95)
        max_var_pct = getattr(self.settings, "VAR_MAX_PORTFOLIO_PCT", 0.03)

        if len(self._portfolio_returns) < lookback:
            return True, ""  # Not enough data yet

        account = float(self.current_balance) if self.current_balance > 0 else 1.0
        returns = np.array(self._portfolio_returns[-lookback:])

        method = getattr(self.settings, "VAR_METHOD", "historical")
        if method == "parametric":
            # Gaussian VaR
            from scipy.stats import norm  # lightweight import
            mu = float(np.mean(returns))
            sigma = float(np.std(returns, ddof=1))
            if sigma == 0:
                return True, ""
            z = norm.ppf(1 - confidence)
            var_pct = -(mu + z * sigma)
        else:
            # Historical VaR — percentile of losses
            var_pct = float(-np.percentile(returns, (1 - confidence) * 100))

        if var_pct < 0:
            var_pct = 0.0  # No risk? Allow.

        if var_pct > max_var_pct:
            var_usd = var_pct * account
            limit_usd = max_var_pct * account
            return False, (
                f"Portfolio VaR {var_pct:.2%} (${var_usd:,.0f}) exceeds limit "
                f"{max_var_pct:.2%} (${limit_usd:,.0f})"
            )
        return True, ""

    def current_var(self) -> dict:
        """Return current VaR metrics (for dashboard / logging)."""
        lookback = getattr(self.settings, "VAR_LOOKBACK_DAYS", 30)
        confidence = getattr(self.settings, "VAR_CONFIDENCE", 0.95)
        account = float(self.current_balance) if self.current_balance > 0 else 0.0

        if len(self._portfolio_returns) < lookback:
            return {"var_pct": None, "var_usd": None, "es_pct": None, "data_days": len(self._portfolio_returns)}

        returns = np.array(self._portfolio_returns[-lookback:])
        cutoff = (1 - confidence) * 100
        var_pct = float(-np.percentile(returns, cutoff))
        # Expected shortfall (CVaR) — average of losses beyond VaR
        tail = returns[returns <= -var_pct]
        es_pct = float(-np.mean(tail)) if len(tail) > 0 else var_pct

        return {
            "var_pct": round(var_pct, 5),
            "var_usd": round(var_pct * account, 2),
            "es_pct": round(es_pct, 5),
            "es_usd": round(es_pct * account, 2),
            "data_days": len(self._portfolio_returns),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    @property
    def _open_positions_snapshot(self) -> dict[str, dict]:
        return getattr(self, "_positions_snap", {})

    @_open_positions_snapshot.setter
    def _open_positions_snapshot(self, value: dict[str, dict]) -> None:
        self._positions_snap = value

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
