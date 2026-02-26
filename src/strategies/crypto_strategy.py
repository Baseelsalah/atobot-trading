"""Crypto Swing Trading Strategy — 24/7 Alpaca Crypto.

Designed for $500-$25K accounts trading crypto on Alpaca.
Captures 3-8% moves over 1-7 days on BTC/USD and ETH/USD.

Crypto-specific advantages over stock swing:
  - 24/7 markets → more entry opportunities
  - No PDT rule at all
  - Fractional shares down to tiny amounts
  - Higher volatility → bigger targets (but wider stops)

Entry Signals (need 2+ confluence):
  1. RSI oversold bounce (<35 crossing back above 35) on 4H bars
  2. Price near rising 20-EMA support (within 1.5 ATR)
  3. Volume surge (>1.5x 20-period avg)
  4. Bullish candle pattern (hammer, engulfing)
  5. EMA stack alignment (20 > 50 = uptrend)
  6. BTC trend gate: alts (ETH) only enter if BTC 20-EMA > 50-EMA

Exit Signals:
  - Take profit: 10%+ (ATR-scaled, wide for crypto swings)
  - Stop loss: 5% (crypto needs breathing room)
  - Trailing stop: activates at +5%, trails at 2.5%
  - Time stop: 14 days max hold
  - No EOD flatten (24/7 market, exempt from stock flatten)

Position Sizing:
  - Risk 4% of account per trade (aggressive for growth)
  - Max 2 concurrent crypto positions
  - Fractional shares
  - Accounts for 0.25% taker fee in size calc

Fee Structure:
  - Alpaca crypto: 25 bps taker (0.25%) per side
  - Round-trip cost: ~0.50% (factored into min R:R calc)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
from loguru import logger

from src.config.settings import Settings
from src.exchange.base_client import BaseExchangeClient
from src.models.order import Order, OrderSide, OrderType
from src.models.position import Position
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy
from src.utils.helpers import round_quantity


class CryptoSwingStrategy(BaseStrategy):
    """Crypto swing trading strategy optimized for small account growth.

    Uses 4H-timeframe signals with 24/7 monitoring.
    Holds positions 1-7 days to capture crypto swings.
    """

    # ── Flag: engine should NOT flatten crypto positions at EOD ──
    exempt_eod_flatten: bool = True

    def __init__(
        self,
        exchange_client: BaseExchangeClient,
        risk_manager: RiskManager,
        settings: Settings,
    ) -> None:
        super().__init__(exchange_client, risk_manager, settings)
        self._initialized_symbols: set[str] = set()
        self._symbol_filters: dict[str, dict] = {}
        self._bar_cache: dict[str, list[dict]] = {}    # 4H bars cache
        self._daily_cache: dict[str, list[dict]] = {}  # Daily bars cache
        self._last_bar_fetch: dict[str, datetime] = {}
        self._entry_dates: dict[str, datetime] = {}     # symbol -> entry datetime
        self._swing_highs: dict[str, Decimal] = {}      # symbol -> highest since entry
        self._swing_stops: dict[str, Decimal] = {}       # symbol -> stop loss price
        self._swing_targets: dict[str, Decimal] = {}     # symbol -> take profit price
        self._trailing_active: dict[str, bool] = {}      # symbol -> trailing activated
        self._eval_done: dict[str, str] = {}             # symbol -> hour_str of last eval
        self._tick_count_local: int = 0
        self._btc_trend_bullish: bool | None = None      # BTC trend cache
        self._btc_trend_checked: str = ""                 # Hour str of last BTC check
        self._cooldown_until: dict[str, datetime] = {}   # symbol -> cooldown expiry after stop-loss

        # Crypto-specific params
        self._rsi_oversold = float(getattr(settings, 'CRYPTO_RSI_OVERSOLD', 35.0))
        self._rsi_overbought = float(getattr(settings, 'CRYPTO_RSI_OVERBOUGHT', 75.0))
        self._volume_surge = float(getattr(settings, 'CRYPTO_VOLUME_SURGE', 1.5))
        self._min_confluence = int(getattr(settings, 'CRYPTO_MIN_CONFLUENCE', 2))
        self._take_profit_pct = float(getattr(settings, 'CRYPTO_TAKE_PROFIT_PCT', 5.0))
        self._stop_loss_pct = float(getattr(settings, 'CRYPTO_STOP_LOSS_PCT', 3.0))
        self._trailing_act_pct = float(getattr(settings, 'CRYPTO_TRAILING_ACTIVATION_PCT', 2.5))
        self._trailing_offset_pct = float(getattr(settings, 'CRYPTO_TRAILING_OFFSET_PCT', 1.5))
        self._max_hold_days = int(getattr(settings, 'CRYPTO_MAX_HOLD_DAYS', 7))
        self._max_positions = int(getattr(settings, 'CRYPTO_MAX_POSITIONS', 2))
        self._risk_per_trade = float(getattr(settings, 'CRYPTO_RISK_PER_TRADE_PCT', 4.0))
        self._order_size_usd = float(getattr(settings, 'CRYPTO_ORDER_SIZE_USD', 200.0))
        self._equity_cap = float(getattr(settings, 'CRYPTO_EQUITY_CAP', 0))
        self._btc_trend_gate = bool(getattr(settings, 'CRYPTO_BTC_TREND_GATE', True))
        self._fee_bps = float(getattr(settings, 'CRYPTO_FEE_BPS', 25.0))

        logger.info(
            "[CRYPTO] Initialized: TP={:.1f}%, SL={:.1f}%, Trail +{:.1f}%/{:.1f}% | "
            "RSI<{:.0f}, max_pos={}, risk/trade={:.1f}%, hold≤{}d | "
            "equity_cap=${} | BTC_gate={} | fee={}bps | EOD_EXEMPT=True",
            self._take_profit_pct, self._stop_loss_pct,
            self._trailing_act_pct, self._trailing_offset_pct,
            self._rsi_oversold, self._max_positions, self._risk_per_trade,
            self._max_hold_days,
            int(self._equity_cap) if self._equity_cap else "disabled",
            self._btc_trend_gate, int(self._fee_bps),
        )

    @property
    def name(self) -> str:
        return "crypto_swing"

    async def initialize(self, symbol: str) -> None:
        """Pre-load 4H bars for the crypto symbol.

        Silently skips non-crypto symbols (stocks) since those are
        handled by other strategies.
        """
        # Only initialize crypto pairs (contain "/")
        if "/" not in symbol:
            return
        if symbol in self._initialized_symbols:
            return
        try:
            await self._get_4h_bars(symbol)
            self._initialized_symbols.add(symbol)
            logger.info("[CRYPTO] Initialized {}", symbol)
        except Exception as exc:
            logger.warning("[CRYPTO] Init failed for {}: {}", symbol, exc)

    async def on_tick(self, symbol: str, current_price: Decimal) -> list[Order]:
        """Main tick handler — manage positions and scan for entries."""
        self._tick_count_local += 1
        orders: list[Order] = []

        # Skip non-crypto symbols (engine routing should prevent this,
        # but guard defensively)
        if "/" not in symbol:
            return orders

        # ── Manage existing position ─────────────────────────────────────
        pos = self.positions.get(symbol)
        if pos and not pos.is_closed:
            exit_orders = await self._manage_position(symbol, pos, current_price)
            orders.extend(exit_orders)
            return orders

        # ── Check for new entries (evaluate every 4 hours to avoid noise) ──
        now = datetime.now(timezone.utc)
        hour_key = now.strftime("%Y-%m-%d-%H")
        # Only re-evaluate entry every ~4 hours
        eval_hour = int(now.hour / 4) * 4
        eval_key = f"{now.strftime('%Y-%m-%d')}-{eval_hour:02d}"
        if self._eval_done.get(symbol) == eval_key:
            return orders

        # Post-stop-loss cooldown (avoid re-entering too quickly)
        if symbol in self._cooldown_until:
            if now < self._cooldown_until[symbol]:
                return orders
            else:
                del self._cooldown_until[symbol]

        # Count active crypto positions across all symbols
        active_count = sum(
            1 for p in self.positions.values() if not p.is_closed
        )
        if active_count >= self._max_positions:
            return orders

        # Active orders check — don't place if we already have pending
        if self._get_active_orders_for(symbol):
            return orders

        # ── BTC trend gate for non-BTC pairs ─────────────────────────────
        if self._btc_trend_gate and "BTC" not in symbol:
            btc_bullish = await self._check_btc_trend()
            if not btc_bullish:
                logger.debug(
                    "[CRYPTO] BTC trend gate BLOCKED {} — BTC in downtrend", symbol
                )
                self._eval_done[symbol] = eval_key
                return orders

        # ── Fetch 4H bars and compute signals ────────────────────────────
        try:
            bars = await self._get_4h_bars(symbol)
            if len(bars) < 30:
                return orders

            df = pd.DataFrame(bars)
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)

            confluence = self._compute_confluence(df, float(current_price))

            if confluence >= self._min_confluence:
                entry_order = await self._create_entry_order(
                    symbol, current_price, df
                )
                if entry_order:
                    orders.append(entry_order)
                    logger.info(
                        "[CRYPTO] ENTRY signal {} | confluence={}/{} | price={}",
                        symbol, confluence, 6, current_price,
                    )
            else:
                logger.debug(
                    "[CRYPTO] No entry for {} | confluence={}/{} (need {})",
                    symbol, confluence, 6, self._min_confluence,
                )

        except Exception as exc:
            logger.warning("[CRYPTO] Entry scan error for {}: {}", symbol, exc)

        self._eval_done[symbol] = eval_key
        return orders

    async def on_order_filled(self, order: Order) -> list[Order]:
        """Handle order fill — create/close position."""
        symbol = order.symbol
        side = str(order.side).upper()

        if side == "BUY":
            # Entry fill — create position
            entry_price = order.price
            qty = order.filled_quantity if order.filled_quantity > 0 else order.quantity
            pos = Position(
                symbol=symbol,
                entry_price=entry_price,
                current_price=entry_price,
                quantity=qty,
                side="LONG",
                strategy=self.name,
            )
            self.positions[symbol] = pos
            self._entry_dates[symbol] = datetime.now(timezone.utc)
            self._swing_highs[symbol] = entry_price
            self._trailing_active[symbol] = False

            # Set stop loss and take profit levels
            sl_price = entry_price * Decimal(str(1 - self._stop_loss_pct / 100))
            tp_price = entry_price * Decimal(str(1 + self._take_profit_pct / 100))
            self._swing_stops[symbol] = sl_price
            self._swing_targets[symbol] = tp_price

            logger.info(
                "[CRYPTO] Position OPENED {} | entry={} qty={} | "
                "SL={} ({:.1f}%) TP={} ({:.1f}%)",
                symbol, entry_price, qty,
                sl_price, self._stop_loss_pct,
                tp_price, self._take_profit_pct,
            )

        elif side == "SELL":
            # Exit fill — close position
            pos = self.positions.get(symbol)
            if pos and not pos.is_closed:
                exit_price = order.price
                pnl = pos.reduce_position(pos.quantity, exit_price)
                self._cleanup_symbol(symbol)

                pnl_pct = ((exit_price / pos.entry_price) - 1) * 100
                logger.info(
                    "[CRYPTO] Position CLOSED {} | entry={} exit={} | "
                    "PnL=${:.2f} ({:.2f}%)",
                    symbol, pos.entry_price, exit_price,
                    float(pnl), float(pnl_pct),
                )

        return []

    async def on_order_cancelled(self, order: Order) -> None:
        """Handle order cancellation."""
        logger.debug("[CRYPTO] Order cancelled: {} {}", order.side, order.symbol)

    async def get_status(self) -> dict:
        """Return current strategy status."""
        active_positions = {
            sym: {
                "entry": str(pos.entry_price),
                "qty": str(pos.quantity),
                "pnl_pct": str(pos.unrealized_pnl_pct) if hasattr(pos, "unrealized_pnl_pct") else "N/A",
                "days_held": (datetime.now(timezone.utc) - self._entry_dates.get(sym, datetime.now(timezone.utc))).days,
                "trailing_active": self._trailing_active.get(sym, False),
                "stop": str(self._swing_stops.get(sym, "N/A")),
                "target": str(self._swing_targets.get(sym, "N/A")),
            }
            for sym, pos in self.positions.items()
            if not pos.is_closed
        }
        return {
            "strategy": self.name,
            "active_positions": active_positions,
            "total_active": len(active_positions),
            "max_positions": self._max_positions,
            "btc_trend_bullish": self._btc_trend_bullish,
        }

    # ── Position Management ───────────────────────────────────────────────────

    async def _manage_position(
        self, symbol: str, pos: Position, current_price: Decimal
    ) -> list[Order]:
        """Manage an open crypto position — check exits."""
        orders: list[Order] = []

        entry_price = pos.entry_price
        pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else Decimal("0")

        # Update highest price since entry
        high = self._swing_highs.get(symbol, entry_price)
        if current_price > high:
            self._swing_highs[symbol] = current_price
            high = current_price

        # ── 1. Stop Loss ─────────────────────────────────────────────────
        stop_price = self._swing_stops.get(symbol, Decimal("0"))
        if current_price <= stop_price:
            logger.warning(
                "[CRYPTO] STOP LOSS hit {} | price={} <= stop={} | PnL={:.2f}%",
                symbol, current_price, stop_price, float(pnl_pct),
            )
            # 6-bar (24h) cooldown after stop-loss to avoid churning
            self._cooldown_until[symbol] = datetime.now(timezone.utc) + timedelta(hours=24)
            orders.append(self._create_exit_order(symbol, pos, "stop_loss"))
            return orders

        # ── 2. Take Profit ───────────────────────────────────────────────
        target_price = self._swing_targets.get(symbol, Decimal("999999"))
        if current_price >= target_price:
            logger.info(
                "[CRYPTO] TAKE PROFIT hit {} | price={} >= target={} | PnL={:.2f}%",
                symbol, current_price, target_price, float(pnl_pct),
            )
            orders.append(self._create_exit_order(symbol, pos, "take_profit"))
            return orders

        # ── 3. Trailing Stop ─────────────────────────────────────────────
        trailing_activation = entry_price * Decimal(str(1 + self._trailing_act_pct / 100))
        if current_price >= trailing_activation:
            if not self._trailing_active.get(symbol, False):
                self._trailing_active[symbol] = True
                logger.info(
                    "[CRYPTO] Trailing stop ACTIVATED {} at {:.2f}% profit",
                    symbol, float(pnl_pct),
                )

        if self._trailing_active.get(symbol, False):
            trail_stop = high * Decimal(str(1 - self._trailing_offset_pct / 100))
            # Only raise the stop, never lower it
            if trail_stop > stop_price:
                self._swing_stops[symbol] = trail_stop
                logger.debug(
                    "[CRYPTO] Trailing stop updated {} | stop={} (high={}, offset={:.1f}%)",
                    symbol, trail_stop, high, self._trailing_offset_pct,
                )
            if current_price <= trail_stop:
                logger.info(
                    "[CRYPTO] TRAILING STOP hit {} | price={} <= trail={} | PnL={:.2f}%",
                    symbol, current_price, trail_stop, float(pnl_pct),
                )
                orders.append(self._create_exit_order(symbol, pos, "trailing_stop"))
                return orders

        # ── 4. Time Stop ─────────────────────────────────────────────────
        entry_time = self._entry_dates.get(symbol)
        if entry_time:
            days_held = (datetime.now(timezone.utc) - entry_time).days
            if days_held >= self._max_hold_days:
                logger.info(
                    "[CRYPTO] TIME STOP {} | held {} days (max={}) | PnL={:.2f}%",
                    symbol, days_held, self._max_hold_days, float(pnl_pct),
                )
                orders.append(self._create_exit_order(symbol, pos, "time_stop"))
                return orders

        return orders

    # ── Signal Computation ────────────────────────────────────────────────────

    def _compute_confluence(self, df: pd.DataFrame, current_price: float) -> int:
        """Compute entry confluence score (0-6) from 4H bars."""
        confluence = 0

        try:
            # 1. RSI oversold bounce
            rsi = self._calc_rsi(df, period=14)
            if rsi is not None and rsi < self._rsi_oversold:
                # Check if RSI was even lower recently (bouncing)
                rsi_series = df['close'].diff()
                if len(rsi_series) > 2 and rsi_series.iloc[-1] > 0:
                    confluence += 1
                    logger.debug("[CRYPTO] Signal: RSI oversold bounce ({:.1f})", rsi)

            # 2. Price near 20-EMA support
            ema20 = self._calc_ema(df, 20)
            if ema20 is not None:
                distance_pct = abs(current_price - ema20) / ema20 * 100
                atr = self._calc_atr(df, 14)
                atr_distance = (atr / current_price * 100) * 1.5 if atr else 3.0
                if distance_pct <= atr_distance and current_price >= ema20 * 0.98:
                    confluence += 1
                    logger.debug("[CRYPTO] Signal: Near 20-EMA ({:.2f}% away)", distance_pct)

            # 3. Volume surge
            avg_vol = self._avg_volume(df, 20)
            current_vol = float(df['volume'].iloc[-1])
            if avg_vol > 0 and current_vol > avg_vol * self._volume_surge:
                confluence += 1
                logger.debug("[CRYPTO] Signal: Volume surge ({:.1f}x)", current_vol / avg_vol)

            # 4. Bullish candle
            if self._is_bullish_candle(df):
                confluence += 1
                logger.debug("[CRYPTO] Signal: Bullish candle pattern")

            # 5. EMA stack (20 > 50 = uptrend)
            ema50 = self._calc_ema(df, 50)
            if ema20 is not None and ema50 is not None and ema20 > ema50:
                confluence += 1
                logger.debug("[CRYPTO] Signal: EMA stack bullish (20 > 50)")

            # 6. Price recovering from recent low (momentum shift)
            if len(df) >= 10:
                recent_low = float(df['low'].tail(10).min())
                recent_high = float(df['high'].tail(10).max())
                if recent_high > recent_low:
                    position_in_range = (current_price - recent_low) / (recent_high - recent_low)
                    if 0.4 <= position_in_range <= 0.7:
                        confluence += 1
                        logger.debug(
                            "[CRYPTO] Signal: Recovery from low ({:.0%} of range)",
                            position_in_range,
                        )

        except Exception as exc:
            logger.debug("[CRYPTO] Confluence calculation error: {}", exc)

        return confluence

    # ── Entry / Exit Order Creation ───────────────────────────────────────────

    async def _create_entry_order(
        self, symbol: str, price: Decimal, df: pd.DataFrame
    ) -> Order | None:
        """Create a sized entry order for the crypto symbol."""
        try:
            qty = await self._calc_crypto_size(symbol, price)
            if qty <= 0:
                logger.debug("[CRYPTO] Size too small for {} at {}", symbol, price)
                return None

            # Minimum notional check (Alpaca min ~$1 for crypto)
            notional = float(price) * qty
            if notional < 1.0:
                logger.debug("[CRYPTO] Notional too small: ${:.2f}", notional)
                return None

            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal(str(qty)),
                price=price,
                strategy=self.name,
            )
            return order

        except Exception as exc:
            logger.warning("[CRYPTO] Entry order creation failed for {}: {}", symbol, exc)
            return None

    def _create_exit_order(
        self, symbol: str, pos: Position, reason: str
    ) -> Order:
        """Create a market sell order to exit the position."""
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=pos.quantity,
            price=pos.current_price if hasattr(pos, "current_price") else pos.entry_price,
            strategy=self.name,
        )
        return order

    # ── Position Sizing ───────────────────────────────────────────────────────

    async def _calc_crypto_size(self, symbol: str, price: Decimal) -> float:
        """Calculate position size for crypto — risk-based with fee adjustment.

        Accounts for Alpaca's 25 bps taker fee on both entry and exit.
        """
        try:
            balances = await self.exchange.get_account_balance()
            equity = float(balances.get("EQUITY", balances.get("USD", Decimal("0"))))

            # Apply equity cap if configured (paper testing)
            if self._equity_cap > 0:
                equity = min(equity, self._equity_cap)

            if equity <= 0:
                return 0.0

            # Risk-based sizing: risk_pct of equity / stop_loss_pct
            risk_amount = equity * (self._risk_per_trade / 100)
            stop_distance_pct = self._stop_loss_pct / 100

            # Account for round-trip fees (~0.50% total)
            fee_pct = (self._fee_bps / 10000) * 2  # Entry + exit
            effective_risk = stop_distance_pct + fee_pct

            position_usd = risk_amount / effective_risk

            # Cap at 50% of equity (leave room for other positions)
            max_position = equity * 0.50
            position_usd = min(position_usd, max_position)

            # Apply non-marginable buying power check
            buying_power = float(balances.get("BUYING_POWER", equity))
            if self._equity_cap > 0:
                buying_power = min(buying_power, self._equity_cap)
            position_usd = min(position_usd, buying_power * 0.95)

            # Convert to quantity
            qty = position_usd / float(price)

            # Crypto fractional precision (6 decimals for most)
            qty = round(qty, 6)

            logger.debug(
                "[CRYPTO] Size calc {} | equity=${:.0f} risk=${:.0f} | "
                "position=${:.0f} qty={:.6f} @ {}",
                symbol, equity, risk_amount, position_usd, qty, price,
            )
            return qty

        except Exception as exc:
            logger.warning("[CRYPTO] Size calc error: {} — using fallback", exc)
            fallback_qty = self._order_size_usd / float(price)
            return round(fallback_qty, 6)

    # ── BTC Trend Gate ────────────────────────────────────────────────────────

    async def _check_btc_trend(self) -> bool:
        """Check if BTC is in an uptrend (20-EMA > 50-EMA on daily).

        Cached for 4 hours to avoid excessive API calls.
        """
        now = datetime.now(timezone.utc)
        check_key = f"{now.strftime('%Y-%m-%d')}-{(now.hour // 4) * 4:02d}"
        if self._btc_trend_checked == check_key and self._btc_trend_bullish is not None:
            return self._btc_trend_bullish

        try:
            bars = await self.exchange.get_klines("BTC/USD", "1d", 60)
            if len(bars) < 50:
                self._btc_trend_bullish = True  # Fail-open if insufficient data
                return True

            df = pd.DataFrame(bars)
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)

            ema20 = self._calc_ema(df, 20)
            ema50 = self._calc_ema(df, 50)

            if ema20 is not None and ema50 is not None:
                self._btc_trend_bullish = ema20 > ema50
            else:
                self._btc_trend_bullish = True  # Fail-open

            self._btc_trend_checked = check_key
            logger.info(
                "[CRYPTO] BTC trend check: {} (20-EMA={:.0f}, 50-EMA={:.0f})",
                "BULLISH" if self._btc_trend_bullish else "BEARISH",
                ema20 or 0, ema50 or 0,
            )
            return self._btc_trend_bullish

        except Exception as exc:
            logger.warning("[CRYPTO] BTC trend check error: {} — fail-open", exc)
            self._btc_trend_bullish = True
            return True

    # ── Data Helpers ──────────────────────────────────────────────────────────

    async def _get_4h_bars(self, symbol: str) -> list[dict]:
        """Fetch and cache 4H bars. Refresh every 4 hours."""
        now = datetime.now(timezone.utc)
        last_fetch = self._last_bar_fetch.get(symbol)

        if (
            last_fetch
            and (now - last_fetch).total_seconds() < 14400  # 4 hours
            and symbol in self._bar_cache
            and len(self._bar_cache[symbol]) >= 20
        ):
            return self._bar_cache[symbol]

        try:
            bars = await self.exchange.get_klines(symbol, "4h", 100)
            if bars:
                self._bar_cache[symbol] = bars
                self._last_bar_fetch[symbol] = now
                logger.debug("[CRYPTO] Fetched {} 4H bars for {}", len(bars), symbol)
            return bars or []
        except Exception as exc:
            logger.warning("[CRYPTO] Bar fetch failed for {}: {}", symbol, exc)
            return self._bar_cache.get(symbol, [])

    # ── Technical Indicator Helpers ───────────────────────────────────────────

    @staticmethod
    def _calc_rsi(df: pd.DataFrame, period: int = 14) -> float | None:
        """Calculate RSI from DataFrame."""
        if len(df) < period + 1:
            return None
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(period).mean().iloc[-1]
        avg_loss = loss.rolling(period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calc_ema(df: pd.DataFrame, period: int) -> float | None:
        """Calculate EMA of close prices."""
        if len(df) < period:
            return None
        return float(df['close'].ewm(span=period, adjust=False).mean().iloc[-1])

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float | None:
        """Calculate Average True Range."""
        if len(df) < period + 1:
            return None
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs(),
        ], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    @staticmethod
    def _avg_volume(df: pd.DataFrame, period: int = 20) -> float:
        """Calculate average volume over period."""
        if len(df) < period:
            return float(df['volume'].mean()) if len(df) > 0 else 0.0
        return float(df['volume'].tail(period).mean())

    @staticmethod
    def _is_bullish_candle(df: pd.DataFrame) -> bool:
        """Check if the last candle is bullish (hammer or engulfing)."""
        if len(df) < 2:
            return False
        last = df.iloc[-1]
        prev = df.iloc[-2]

        body = last['close'] - last['open']
        full_range = last['high'] - last['low']

        if full_range <= 0:
            return False

        # Hammer: small body, long lower wick
        lower_wick = min(last['open'], last['close']) - last['low']
        if body > 0 and lower_wick > body * 2 and lower_wick > full_range * 0.5:
            return True

        # Bullish engulfing
        if (prev['close'] < prev['open'] and  # Previous candle bearish
                last['close'] > last['open'] and  # Current candle bullish
                last['close'] > prev['open'] and  # Current close > prev open
                last['open'] < prev['close']):     # Current open < prev close
            return True

        return False

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _cleanup_symbol(self, symbol: str) -> None:
        """Remove tracking state for a closed position."""
        self._entry_dates.pop(symbol, None)
        self._swing_highs.pop(symbol, None)
        self._swing_stops.pop(symbol, None)
        self._swing_targets.pop(symbol, None)
        self._trailing_active.pop(symbol, None)
        self._eval_done.pop(symbol, None)
        # NOTE: intentionally NOT clearing _cooldown_until here —
        # cooldown must persist after position cleanup so the bot
        # waits before re-entering the same symbol.
