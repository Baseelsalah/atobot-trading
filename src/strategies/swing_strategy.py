"""Swing Trading Strategy for Small Accounts ($500-$25K).

Designed to grow small accounts fast by capturing 2-5% moves over 1-5 days.
This avoids the PDT rule entirely since positions are held overnight.

Entry Signals (need 2+ confluence):
  1. RSI oversold bounce (<38 crossing back above 38)
  2. Price near rising 20-EMA support (within 1 ATR)
  3. Volume surge (>1.3x 20-day avg)
  4. Bullish candle pattern (hammer, engulfing)
  5. EMA stack alignment (20 > 50 = uptrend)
  6. Price bouncing off 50-EMA support

Exit Signals:
  - Take profit: 3%+ (ATR-scaled, min 2.5:1 R:R)
  - Stop loss: 1.5% (tight to cut losers fast)
  - Trailing stop: activates at +1.5%, trails at 0.75%
  - Time stop: 5 days max hold
  - Gap protection: instant exit if open gaps past SL

Position Sizing:
  - Risk 3% of account per trade (aggressive for growth)
  - Max 3 concurrent positions
  - Fractional shares supported (Alpaca)
  - Max 50% of buying power per position
  - Equity cap for paper testing (simulates small account)
"""

from __future__ import annotations

from datetime import datetime, timezone
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


class SwingStrategy(BaseStrategy):
    """Swing trading strategy optimized for small account growth.

    Uses daily-timeframe signals with intraday entry timing.
    Holds positions 1-5 days to avoid PDT restrictions.
    """

    # ── Flag: engine should NOT flatten swing positions at EOD ──
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
        self._daily_cache: dict[str, list[dict]] = {}
        self._last_daily_fetch: dict[str, datetime] = {}
        self._entry_dates: dict[str, datetime] = {}  # symbol -> entry datetime
        self._swing_highs: dict[str, Decimal] = {}  # symbol -> highest since entry
        self._swing_lows: dict[str, Decimal] = {}   # symbol -> lowest since entry
        self._swing_stops: dict[str, Decimal] = {}   # symbol -> stop loss price
        self._swing_targets: dict[str, Decimal] = {}  # symbol -> take profit price
        self._trailing_active: dict[str, bool] = {}   # symbol -> trailing activated
        self._daily_eval_done: dict[str, str] = {}    # symbol -> date string of last eval
        self._tick_count_local: int = 0

        # Swing params
        self._rsi_oversold = float(getattr(settings, 'SWING_RSI_OVERSOLD', 38.0))
        self._rsi_overbought = float(getattr(settings, 'SWING_RSI_OVERBOUGHT', 70.0))
        self._volume_surge = float(getattr(settings, 'SWING_VOLUME_SURGE', 1.3))
        self._min_confluence = int(getattr(settings, 'SWING_MIN_CONFLUENCE', 2))
        self._take_profit_pct = float(getattr(settings, 'SWING_TAKE_PROFIT_PCT', 3.0))
        self._stop_loss_pct = float(getattr(settings, 'SWING_STOP_LOSS_PCT', 1.5))
        self._trailing_act_pct = float(getattr(settings, 'SWING_TRAILING_ACTIVATION_PCT', 1.5))
        self._trailing_offset_pct = float(getattr(settings, 'SWING_TRAILING_OFFSET_PCT', 0.75))
        self._max_hold_days = int(getattr(settings, 'SWING_MAX_HOLD_DAYS', 5))
        self._max_positions = int(getattr(settings, 'SWING_MAX_POSITIONS', 3))
        self._risk_per_trade = float(getattr(settings, 'SWING_RISK_PER_TRADE_PCT', 3.0))
        self._order_size_usd = float(getattr(settings, 'SWING_ORDER_SIZE_USD', 250.0))
        # Equity cap — on paper accounts, pretend equity is capped to simulate
        # small account sizing.  0 = use real equity.
        self._equity_cap = float(getattr(settings, 'SWING_EQUITY_CAP', 0))
        # Gap protection — max acceptable overnight gap %. If a position
        # gaps down more than this at open, exit immediately.
        self._max_gap_pct = float(getattr(settings, 'SWING_MAX_GAP_PCT', 5.0))
        # Track previous close for gap detection
        self._prev_closes: dict[str, float] = {}
        self._gap_checked_today: dict[str, str] = {}  # symbol -> date_str

        logger.info(
            "[SWING] Initialized: TP={:.1f}%, SL={:.1f}%, Trail +{:.1f}%/{:.1f}% | "
            "RSI<{:.0f}, max_pos={}, risk/trade={:.1f}%, hold≤{}d | "
            "equity_cap=${:.0f} | EOD_EXEMPT=True",
            self._take_profit_pct, self._stop_loss_pct,
            self._trailing_act_pct, self._trailing_offset_pct,
            self._rsi_oversold, self._max_positions, self._risk_per_trade,
            self._max_hold_days, self._equity_cap,
        )

    @property
    def name(self) -> str:
        return "swing"

    async def initialize(self, symbol: str) -> None:
        logger.info("[SWING] Initialising for {}", symbol)
        self._symbol_filters[symbol] = await self.exchange.get_symbol_filters(symbol)
        self._initialized_symbols.add(symbol)
        self.is_running = True

    # ── Indicator helpers ─────────────────────────────────────────────────────

    def _calc_rsi(self, closes: list[float], period: int = 14) -> float:
        """Calculate RSI from closing prices."""
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        recent = deltas[-period:]
        gains = [d for d in recent if d > 0]
        losses = [-d for d in recent if d < 0]
        avg_gain = sum(gains) / period if gains else 0.0001
        avg_loss = sum(losses) / period if losses else 0.0001
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calc_ema(self, closes: list[float], period: int = 20) -> float:
        """Calculate EMA from closing prices."""
        if len(closes) < period:
            return closes[-1] if closes else 0
        multiplier = 2 / (period + 1)
        ema = sum(closes[:period]) / period
        for price in closes[period:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def _calc_atr(self, bars: list[dict], period: int = 14) -> float:
        """Calculate ATR from daily bars."""
        if len(bars) < period + 1:
            return 0
        true_ranges = []
        for i in range(1, len(bars)):
            h = float(bars[i]['high'])
            l = float(bars[i]['low'])
            pc = float(bars[i - 1]['close'])
            tr = max(h - l, abs(h - pc), abs(l - pc))
            true_ranges.append(tr)
        return sum(true_ranges[-period:]) / period

    def _avg_volume(self, bars: list[dict], period: int = 20) -> float:
        """Calculate average volume."""
        if len(bars) < period:
            return float(bars[-1].get('volume', 0)) if bars else 0
        vols = [float(b.get('volume', 0)) for b in bars[-period:]]
        return sum(vols) / len(vols) if vols else 0

    def _is_bullish_candle(self, bar: dict) -> bool:
        """Check for bullish candle pattern (hammer, engulfing)."""
        o, h, l, c = float(bar['open']), float(bar['high']), float(bar['low']), float(bar['close'])
        body = abs(c - o)
        full_range = h - l
        if full_range == 0:
            return False
        lower_wick = min(o, c) - l
        # Hammer
        if c > o and lower_wick > body * 2 and body / full_range < 0.35:
            return True
        # Strong bullish body
        if c > o and body / full_range > 0.6:
            return True
        return False

    # ── Daily data fetcher (cached, refreshes once per day) ───────────────────

    async def _get_daily_bars(self, symbol: str) -> list[dict]:
        """Fetch daily bars, caching for the day."""
        now = datetime.now(timezone.utc)
        last_fetch = self._last_daily_fetch.get(symbol)

        # Refresh daily bars at most every 4 hours (or on first call)
        if last_fetch and (now - last_fetch).total_seconds() < 14400:
            cached = self._daily_cache.get(symbol)
            if cached:
                return cached

        try:
            bars = await self.exchange.get_klines(symbol, "1D", 200)
            if bars:
                self._daily_cache[symbol] = bars
                self._last_daily_fetch[symbol] = now
                return bars
        except Exception as exc:
            logger.warning("[SWING] Failed to fetch daily bars for {}: {}", symbol, exc)

        return self._daily_cache.get(symbol, [])

    # ── Position size calculation ─────────────────────────────────────────────

    def _calc_swing_size(
        self, price: float, sl_price: float, equity: float,
    ) -> Decimal:
        """Risk-based position sizing for swing trades.

        Risks SWING_RISK_PER_TRADE_PCT of equity per trade.
        Supports fractional shares for small accounts.
        """
        risk_dollars = equity * (self._risk_per_trade / 100)
        sl_distance = abs(price - sl_price)
        if sl_distance <= 0:
            sl_distance = price * (self._stop_loss_pct / 100)

        shares = risk_dollars / sl_distance

        # Cap at 50% of buying power (2x equity for margin) per position
        max_position = equity * 2 * 0.50 / price
        shares = min(shares, max_position)

        # Fractional shares for small accounts (Alpaca supports)
        if equity < 10000:
            shares = round(shares, 4)
        else:
            shares = max(1, int(shares))

        return Decimal(str(max(0.001, shares)))

    # ── Main tick handler ─────────────────────────────────────────────────────

    async def on_tick(self, symbol: str, current_price: Decimal) -> list[Order]:
        """Called every poll interval. Handles swing entry and exit logic."""
        if symbol not in self._initialized_symbols:
            await self.initialize(symbol)

        self._tick_count_local += 1
        orders: list[Order] = []
        filters = self._symbol_filters[symbol]
        pos = self.positions.get(symbol)
        price_f = float(current_price)

        # ── GAP PROTECTION (check once at open each day) ────────────────
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        today_str = now_et.strftime("%Y-%m-%d")

        if pos and not pos.is_closed:
            gap_key = f"{symbol}_{today_str}"
            if self._gap_checked_today.get(symbol) != gap_key:
                self._gap_checked_today[symbol] = gap_key
                prev_close = self._prev_closes.get(symbol)
                if prev_close and prev_close > 0:
                    gap_pct = ((price_f - prev_close) / prev_close) * 100
                    sl_price_gap = self._swing_stops.get(symbol)
                    # If gapped below stop loss, exit immediately
                    if sl_price_gap and pos.side == "LONG" and current_price < sl_price_gap:
                        exit_side = OrderSide.SELL if pos.side == "LONG" else OrderSide.COVER
                        qty = pos.quantity
                        if float(qty) > 0:
                            orders.append(Order(
                                symbol=symbol, side=exit_side,
                                order_type=OrderType.MARKET, price=current_price,
                                quantity=qty, strategy=self.name,
                            ))
                            logger.warning(
                                "[SWING] GAP EXIT {} | gap={:.1f}% | "
                                "open=${:.2f} < SL=${:.2f} | IMMEDIATE EXIT",
                                symbol, gap_pct, price_f, float(sl_price_gap),
                            )
                        return orders

        # ── EXIT LOGIC (check every tick) ─────────────────────────────────
        if pos and not pos.is_closed:
            pos.update_price(current_price)

            exit_side = OrderSide.SELL if pos.side == "LONG" else OrderSide.COVER

            # Track high/low since entry
            prev_high = self._swing_highs.get(symbol, pos.entry_price)
            prev_low = self._swing_lows.get(symbol, pos.entry_price)
            if current_price > prev_high:
                self._swing_highs[symbol] = current_price
                prev_high = current_price
            if current_price < prev_low:
                self._swing_lows[symbol] = current_price
                prev_low = current_price

            sl_price = self._swing_stops.get(symbol)
            tp_price = self._swing_targets.get(symbol)

            exit_reason = None

            # 1. Stop loss
            if sl_price and pos.side == "LONG" and current_price <= sl_price:
                exit_reason = f"STOP_LOSS (${price_f:.2f} <= ${float(sl_price):.2f})"
            elif sl_price and pos.side == "SHORT" and current_price >= sl_price:
                exit_reason = f"STOP_LOSS (${price_f:.2f} >= ${float(sl_price):.2f})"

            # 2. Take profit
            if not exit_reason:
                if tp_price and pos.side == "LONG" and current_price >= tp_price:
                    exit_reason = f"TAKE_PROFIT (${price_f:.2f} >= ${float(tp_price):.2f})"
                elif tp_price and pos.side == "SHORT" and current_price <= tp_price:
                    exit_reason = f"TAKE_PROFIT (${price_f:.2f} <= ${float(tp_price):.2f})"

            # 3. Trailing stop (after activation)
            if not exit_reason and pos.side == "LONG":
                activation_price = pos.entry_price * (
                    Decimal("1") + Decimal(str(self._trailing_act_pct / 100))
                )
                if prev_high >= activation_price:
                    self._trailing_active[symbol] = True
                    trail_stop = prev_high * (
                        Decimal("1") - Decimal(str(self._trailing_offset_pct / 100))
                    )
                    if current_price <= trail_stop:
                        exit_reason = f"TRAILING_STOP (${price_f:.2f} <= ${float(trail_stop):.2f})"

            # 4. Time stop (max hold days)
            if not exit_reason:
                entry_dt = self._entry_dates.get(symbol)
                if entry_dt:
                    days_held = (datetime.now(timezone.utc) - entry_dt).days
                    if days_held >= self._max_hold_days:
                        exit_reason = f"TIME_STOP ({days_held}d >= {self._max_hold_days}d max)"

            # Execute exit
            if exit_reason:
                qty = pos.quantity
                # Allow fractional
                if float(qty) > 0:
                    orders.append(Order(
                        symbol=symbol, side=exit_side,
                        order_type=OrderType.MARKET, price=current_price,
                        quantity=qty, strategy=self.name,
                    ))
                    logger.info(
                        "[SWING] EXIT {} {} | {} | PnL%={:.2f}%",
                        exit_side.value, symbol, exit_reason,
                        pos.unrealized_pnl_percent,
                    )
                return orders

            return orders  # Holding, no exit triggered

        # ── ENTRY LOGIC (evaluate daily signals, throttled) ───────────────

        # Only evaluate entries once per day per symbol (not every 5s tick)
        eval_key = f"{symbol}_{today_str}"

        if self._daily_eval_done.get(symbol) == eval_key:
            return orders  # Already evaluated today

        # Only evaluate near market open (9:30-10:30 ET) for best swing entries
        if now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 35):
            return orders
        if now_et.hour > 10 or (now_et.hour == 10 and now_et.minute > 30):
            # After 10:30, mark as done for today (missed window)
            self._daily_eval_done[symbol] = eval_key
            return orders

        # Gate: max positions
        open_count = sum(
            1 for p in self.positions.values()
            if not p.is_closed
        )
        if open_count >= self._max_positions:
            self._daily_eval_done[symbol] = eval_key
            return orders

        # Fetch daily bars
        daily_bars = await self._get_daily_bars(symbol)
        if len(daily_bars) < 55:
            self._daily_eval_done[symbol] = eval_key
            return orders

        # Mark evaluated
        self._daily_eval_done[symbol] = eval_key

        # Calculate indicators
        closes = [float(b['close']) for b in daily_bars]
        rsi_now = self._calc_rsi(closes)
        rsi_prev = self._calc_rsi(closes[:-1]) if len(closes) > 15 else rsi_now
        ema_20 = self._calc_ema(closes, 20)
        ema_50 = self._calc_ema(closes, 50)
        atr = self._calc_atr(daily_bars)
        avg_vol = self._avg_volume(daily_bars)
        latest = daily_bars[-1]

        if atr <= 0 or avg_vol <= 0:
            return orders

        # ── Score confluence signals ──────────────────────────────────────
        reasons: list[str] = []
        confidence = 0

        # 1. RSI oversold bounce
        if rsi_prev < self._rsi_oversold and rsi_now >= self._rsi_oversold:
            reasons.append(f"RSI bounce {rsi_prev:.0f}->{rsi_now:.0f}")
            confidence += 25
        elif rsi_now < self._rsi_oversold + 5 and rsi_now > rsi_prev:
            reasons.append(f"RSI recovering {rsi_now:.0f}")
            confidence += 15

        # 2. Price near rising 20-EMA support
        dist_to_ema = (price_f - ema_20) / ema_20 * 100
        if -1.5 < dist_to_ema < 0.5 and ema_20 > ema_50:
            reasons.append(f"Near rising 20-EMA ({dist_to_ema:+.1f}%)")
            confidence += 20

        # 3. Volume surge
        today_vol = float(latest.get('volume', 0))
        if today_vol > avg_vol * self._volume_surge:
            vol_ratio = today_vol / avg_vol
            reasons.append(f"Volume surge {vol_ratio:.1f}x")
            confidence += 15

        # 4. Bullish candle pattern
        if self._is_bullish_candle(latest):
            reasons.append("Bullish candle")
            confidence += 20

        # 5. Above rising EMA stack (trend)
        if price_f > ema_20 and ema_20 > ema_50:
            reasons.append("Above rising 20/50 EMA")
            confidence += 15

        # 6. Price bouncing off 50-EMA support
        dist_to_50 = (price_f - ema_50) / ema_50 * 100
        if 0 < dist_to_50 < 1.0 and ema_20 > ema_50:
            reasons.append(f"Bouncing off 50-EMA ({dist_to_50:+.1f}%)")
            confidence += 15

        # ── Check confluence threshold ────────────────────────────────────
        if len(reasons) < self._min_confluence:
            if self._tick_count_local % 120 == 0:
                logger.debug(
                    "[SWING] No entry {} | confluence={}/{} RSI={:.0f}",
                    symbol, len(reasons), self._min_confluence, rsi_now,
                )
            return orders

        # ── Build entry order ─────────────────────────────────────────────
        atr_pct = (atr / price_f) * 100
        sl_pct = max(self._stop_loss_pct, atr_pct * 1.5)
        sl_pct = min(sl_pct, 5.0)  # Cap at 5%

        tp_pct = max(self._take_profit_pct, sl_pct * 2.5)  # Min 2.5:1 R:R

        sl_price = Decimal(str(round(price_f * (1 - sl_pct / 100), 2)))
        tp_price = Decimal(str(round(price_f * (1 + tp_pct / 100), 2)))

        # Get account equity for sizing (apply cap for paper testing)
        try:
            balance = await self.exchange.get_account_balance()
            equity = float(balance.get("equity", balance.get("USD", Decimal("500"))))
            if self._equity_cap > 0:
                equity = min(equity, self._equity_cap)
        except Exception:
            equity = 500.0

        # Store prev close for gap detection next day
        if daily_bars:
            self._prev_closes[symbol] = float(daily_bars[-1]['close'])

        quantity = self._calc_swing_size(price_f, float(sl_price), equity)

        if float(quantity) <= 0:
            return orders

        # Use market order for immediate fill
        orders.append(Order(
            symbol=symbol, side=OrderSide.BUY,
            order_type=OrderType.MARKET, price=current_price,
            quantity=quantity, strategy=self.name,
        ))

        # Store swing levels
        self._swing_stops[symbol] = sl_price
        self._swing_targets[symbol] = tp_price
        self._trailing_active[symbol] = False

        logger.info(
            "[SWING] BUY {} | ${:.2f} | SL=${:.2f} ({:.1f}%) | TP=${:.2f} ({:.1f}%) | "
            "qty={} | conf={} | {}",
            symbol, price_f, float(sl_price), sl_pct, float(tp_price), tp_pct,
            quantity, confidence, " + ".join(reasons),
        )

        return orders

    # ── Order fill handler ────────────────────────────────────────────────────

    async def on_order_filled(self, order: Order) -> list[Order]:
        symbol = order.symbol
        if order.side == OrderSide.BUY:
            pos = self.positions.get(symbol)
            if pos is None or pos.is_closed:
                self.positions[symbol] = Position(
                    symbol=symbol, side="LONG",
                    entry_price=order.price, current_price=order.price,
                    quantity=order.filled_quantity, strategy=self.name,
                )
                self._entry_dates[symbol] = datetime.now(timezone.utc)
                self._swing_highs[symbol] = order.price
                self._swing_lows[symbol] = order.price
                logger.info(
                    "[SWING] Position opened: {} {} shares @ ${:.2f}",
                    symbol, order.filled_quantity, order.price,
                )
            else:
                pos.add_to_position(order.price, order.filled_quantity)
        elif order.side in (OrderSide.SELL, OrderSide.COVER):
            pos = self.positions.get(symbol)
            if pos and not pos.is_closed:
                pnl = pos.unrealized_pnl
                pos.reduce_position(order.filled_quantity, order.price)
                if pos.is_closed:
                    pnl_pct = float(pos.unrealized_pnl_percent)
                    if pnl > Decimal("0"):
                        self.record_win()
                        logger.info(
                            "[SWING] WIN {} | +${:.2f} ({:+.1f}%)",
                            symbol, pnl, pnl_pct,
                        )
                    else:
                        self.record_loss()
                        logger.info(
                            "[SWING] LOSS {} | -${:.2f} ({:+.1f}%)",
                            symbol, abs(pnl), pnl_pct,
                        )
                    # Cleanup
                    self._swing_stops.pop(symbol, None)
                    self._swing_targets.pop(symbol, None)
                    self._swing_highs.pop(symbol, None)
                    self._swing_lows.pop(symbol, None)
                    self._entry_dates.pop(symbol, None)
                    self._trailing_active.pop(symbol, None)
                    self._reset_trailing_high(symbol)

        self.active_orders = [
            o for o in self.active_orders if o.internal_id != order.internal_id
        ]
        return []

    async def on_order_cancelled(self, order: Order) -> None:
        self.active_orders = [
            o for o in self.active_orders if o.internal_id != order.internal_id
        ]

    async def get_status(self) -> dict:
        open_positions = {
            s: {
                "entry": str(p.entry_price),
                "current": str(p.current_price),
                "qty": str(p.quantity),
                "pnl": str(p.unrealized_pnl),
                "pnl_pct": f"{p.unrealized_pnl_percent:.2f}%",
                "days_held": (datetime.now(timezone.utc) - self._entry_dates.get(s, datetime.now(timezone.utc))).days
                if s in self._entry_dates else 0,
                "sl": str(self._swing_stops.get(s, "N/A")),
                "tp": str(self._swing_targets.get(s, "N/A")),
            }
            for s, p in self.positions.items() if not p.is_closed
        }
        return {
            "strategy": self.name,
            "is_running": self.is_running,
            "mode": "swing_hold",
            "eod_exempt": True,
            "max_hold_days": self._max_hold_days,
            "max_positions": self._max_positions,
            "active_orders": len([o for o in self.active_orders if o.is_active]),
            "positions": open_positions,
        }
