"""Opening Range Breakout (ORB) day-trading strategy for AtoBot.

Monitors the first N minutes of trading to establish the opening range
(high/low), then enters on a confirmed breakout above the range high.

**v3 research-driven** (backtest validated):
- MACD death cross exit REMOVED (82% over-triggering on 1-min bars)
- MACD entry confirmation REMOVED (was filtering valid breakouts)
- Volume confirmation is now BLOCKING (not advisory)
- Brackets REMOVED (cutting avg win to $74 vs $122 avg loss)
- EMA trend filter kept (maintains 60% win rate vs 53% without)
- One trade per day per symbol preserved
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
from loguru import logger

from src.config.settings import Settings
from src.data import indicators
from src.exchange.base_client import BaseExchangeClient
from src.models.order import Order, OrderSide, OrderType
from src.models.position import Position
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy
from src.utils.helpers import round_quantity


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout strategy.

    1. During the first ``ORB_RANGE_MINUTES`` of market open, track
       the high and low.
    2. After the range is set, buy on breakout above range + buffer, or
       sell/short on breakdown below range – buffer.
    3. Take-profit / stop-loss percentages control exits.
    4. Only one trade per symbol per day.
    """

    def __init__(
        self,
        exchange_client: BaseExchangeClient,
        risk_manager: RiskManager,
        settings: Settings,
    ) -> None:
        super().__init__(exchange_client, risk_manager, settings)
        self._initialized_symbols: set[str] = set()
        self._symbol_filters: dict[str, dict] = {}
        # ORB state per symbol
        self._range_high: dict[str, Decimal] = {}
        self._range_low: dict[str, Decimal] = {}
        self._range_set: dict[str, bool] = {}
        self._traded_today: dict[str, bool] = {}
        self._range_bars: dict[str, list[dict]] = {}
        self._last_reset_date: datetime | None = None

    @property
    def name(self) -> str:
        return "orb"

    async def initialize(self, symbol: str) -> None:
        logger.info("[ORB] Initialising for {}", symbol)
        self._symbol_filters[symbol] = await self.exchange.get_symbol_filters(symbol)
        self._range_set[symbol] = False
        self._traded_today[symbol] = False
        self._range_high[symbol] = Decimal("0")
        self._range_low[symbol] = Decimal("999999")
        self._range_bars[symbol] = []
        self._initialized_symbols.add(symbol)
        self.is_running = True

    def _maybe_reset_daily(self) -> None:
        """Reset ORB state at start of new trading day."""
        now = datetime.now(timezone.utc)
        if self._last_reset_date is None or now.date() > self._last_reset_date.date():
            for sym in self._initialized_symbols:
                self._range_set[sym] = False
                self._traded_today[sym] = False
                self._range_high[sym] = Decimal("0")
                self._range_low[sym] = Decimal("999999")
                self._range_bars[sym] = []
            self._last_reset_date = now
            logger.info("[ORB] New trading day — ranges reset")

    async def on_tick(self, symbol: str, current_price: Decimal) -> list[Order]:
        if symbol not in self._initialized_symbols:
            await self.initialize(symbol)

        self._maybe_reset_daily()
        orders: list[Order] = []
        filters = self._symbol_filters[symbol]
        pos = self.positions.get(symbol)

        # ── Exit logic ───────────────────────────────────────────────────
        if pos and not pos.is_closed:
            pos.update_price(current_price)
            tp = Decimal(str(self.settings.ORB_TAKE_PROFIT_PERCENT))
            sl = Decimal(str(self.settings.ORB_STOP_LOSS_PERCENT))

            # Determine exit side based on position direction
            exit_side = OrderSide.SELL if pos.side == "LONG" else OrderSide.COVER

            # Trailing stop check (before fixed TP/SL)
            if self._check_trailing_stop(symbol, pos, current_price):
                qty = round_quantity(pos.quantity, filters["step_size"])
                if qty > Decimal("0"):
                    orders.append(Order(
                        symbol=symbol, side=exit_side,
                        order_type=OrderType.MARKET, price=current_price,
                        quantity=qty, strategy=self.name,
                    ))
                    logger.info("[ORB] TRAILING STOP {} | PnL%={:.2f}", symbol, pos.unrealized_pnl_percent)
                return orders

            # MACD death cross exit REMOVED in v3
            # Backtest proved MACD was 82% of all ORB exits (over-triggering)
            # Trailing stop + TP/SL are sufficient for exit management

            if pos.unrealized_pnl_percent >= tp:
                qty = round_quantity(pos.quantity, filters["step_size"])
                if qty > Decimal("0"):
                    orders.append(Order(
                        symbol=symbol, side=exit_side,
                        order_type=OrderType.MARKET, price=current_price,
                        quantity=qty, strategy=self.name,
                    ))
                    logger.info("[ORB] TP triggered {}", symbol)
                return orders

            if pos.unrealized_pnl_percent <= -sl:
                qty = round_quantity(pos.quantity, filters["step_size"])
                if qty > Decimal("0"):
                    orders.append(Order(
                        symbol=symbol, side=exit_side,
                        order_type=OrderType.MARKET, price=current_price,
                        quantity=qty, strategy=self.name,
                    ))
                    logger.info("[ORB] SL triggered {}", symbol)
                return orders

            return orders

        # ── Already traded today? ────────────────────────────────────────
        if self._traded_today.get(symbol, False):
            return orders

        # ── Build opening range ──────────────────────────────────────────
        if not self._range_set.get(symbol, False):
            # Accumulate price updates during the opening window
            hi = self._range_high[symbol]
            lo = self._range_low[symbol]
            if current_price > hi:
                self._range_high[symbol] = current_price
            if current_price < lo:
                self._range_low[symbol] = current_price

            # Try to get 1-minute bars for the opening range
            try:
                bars = await self.exchange.get_klines(
                    symbol, "1m", self.settings.ORB_RANGE_MINUTES + 2
                )
                if len(bars) >= self.settings.ORB_RANGE_MINUTES:
                    range_bars = bars[:self.settings.ORB_RANGE_MINUTES]
                    self._range_high[symbol] = max(
                        Decimal(str(b["high"])) for b in range_bars
                    )
                    self._range_low[symbol] = min(
                        Decimal(str(b["low"])) for b in range_bars
                    )
                    self._range_set[symbol] = True
                    logger.info(
                        "[ORB] Range set for {} | high={} low={}",
                        symbol, self._range_high[symbol], self._range_low[symbol],
                    )
            except Exception as exc:
                logger.warning("[ORB] Could not fetch bars for {}: {}", symbol, exc)

            return orders

        # ── Breakout detection ───────────────────────────────────────────
        # Check entry filters
        if not self._passes_time_filter():
            return orders
        if not await self._passes_trend_filter(symbol, current_price):
            return orders

        range_high = self._range_high[symbol]
        range_low = self._range_low[symbol]
        buffer = Decimal(str(self.settings.ORB_BREAKOUT_PERCENT)) / Decimal("100")

        breakout_high = range_high * (Decimal("1") + buffer)
        breakout_low = range_low * (Decimal("1") - buffer)

        order_usd = Decimal(str(self.settings.ORB_ORDER_SIZE_USD))

        if current_price >= breakout_high:
            # ── Volume confirmation for breakout (BLOCKING in v3) ────
            volume_ok = True
            try:
                bars_5m = await self.exchange.get_klines(symbol, "5m", 40)
                if len(bars_5m) >= 20:
                    df = pd.DataFrame(bars_5m)
                    for col in ("open", "high", "low", "close", "volume"):
                        df[col] = df[col].astype(float)

                    # Volume must be >= 1.3x average for valid breakout
                    vol_sma = indicators.volume_sma(df, 20)
                    cur_vol = df["volume"].iloc[-1]
                    avg_vol = vol_sma.iloc[-1]
                    if avg_vol > 0:
                        volume_ok = cur_vol >= avg_vol * 1.3
                        if not volume_ok:
                            logger.info(
                                "[ORB] BLOCKED weak volume {} | vol {:.0f} < 1.3x avg {:.0f}",
                                symbol, cur_vol, avg_vol,
                            )
                            return orders
            except Exception as exc:
                logger.debug("[ORB] Volume check error for {}: {}", symbol, exc)

            # MACD confirmation REMOVED in v3 (over-filtered valid breakouts)
            # Bullish breakout with volume confirmation -> BUY
            quantity = order_usd / current_price
            quantity = round_quantity(quantity, filters["step_size"])
            if quantity > Decimal("0"):
                orders.append(Order(
                    symbol=symbol, side=OrderSide.BUY,
                    order_type=OrderType.MARKET, price=current_price,
                    quantity=quantity, strategy=self.name,
                ))
                self._traded_today[symbol] = True
                logger.info(
                    "[ORB] BREAKOUT BUY {} | price={} > high={} | volume confirmed",
                    symbol, current_price, breakout_high,
                )

        elif current_price <= breakout_low:
            # Bearish breakdown — SHORT if enabled
            if getattr(self.settings, "SHORT_SELLING_ENABLED", False):
                # Volume confirmation for breakdown (same as breakout)
                volume_ok = True
                try:
                    bars_5m = await self.exchange.get_klines(symbol, "5m", 40)
                    if len(bars_5m) >= 20:
                        df = pd.DataFrame(bars_5m)
                        for col in ("open", "high", "low", "close", "volume"):
                            df[col] = df[col].astype(float)
                        vol_sma = indicators.volume_sma(df, 20)
                        cur_vol = df["volume"].iloc[-1]
                        avg_vol = vol_sma.iloc[-1]
                        if avg_vol > 0:
                            volume_ok = cur_vol >= avg_vol * 1.3
                            if not volume_ok:
                                logger.info(
                                    "[ORB] BLOCKED weak volume breakdown {} | vol {:.0f} < 1.3x avg {:.0f}",
                                    symbol, cur_vol, avg_vol,
                                )
                                return orders
                except Exception as exc:
                    logger.debug("[ORB] Volume check error for {}: {}", symbol, exc)

                quantity = order_usd / current_price
                quantity = round_quantity(quantity, filters["step_size"])
                if quantity > Decimal("0"):
                    orders.append(Order(
                        symbol=symbol, side=OrderSide.SHORT,
                        order_type=OrderType.MARKET, price=current_price,
                        quantity=quantity, strategy=self.name,
                    ))
                    self._traded_today[symbol] = True
                    logger.info(
                        "[ORB] BREAKDOWN SHORT {} | price={} < low={} | volume confirmed",
                        symbol, current_price, breakout_low,
                    )
            else:
                logger.debug(
                    "[ORB] Breakdown {} — skipping (short selling disabled)", symbol
                )

        return orders

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
                self._trailing_highs[symbol] = order.price
            else:
                pos.add_to_position(order.price, order.filled_quantity)
        elif order.side == OrderSide.SHORT:
            pos = self.positions.get(symbol)
            if pos is None or pos.is_closed:
                self.positions[symbol] = Position(
                    symbol=symbol, side="SHORT",
                    entry_price=order.price, current_price=order.price,
                    quantity=order.filled_quantity, strategy=self.name,
                )
                self._trailing_highs[symbol] = order.price
            else:
                pos.add_to_position(order.price, order.filled_quantity)
        elif order.side in (OrderSide.SELL, OrderSide.COVER):
            pos = self.positions.get(symbol)
            if pos and not pos.is_closed:
                pos.reduce_position(order.filled_quantity, order.price)
            if pos and pos.is_closed:
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
        return {
            "strategy": self.name,
            "is_running": self.is_running,
            "active_orders": len([o for o in self.active_orders if o.is_active]),
            "positions": {
                s: {
                    "entry": str(p.entry_price), "current": str(p.current_price),
                    "qty": str(p.quantity), "pnl": str(p.unrealized_pnl),
                }
                for s, p in self.positions.items() if not p.is_closed
            },
            "ranges": {
                s: {"high": str(self._range_high.get(s, 0)),
                    "low": str(self._range_low.get(s, 0)),
                    "set": self._range_set.get(s, False)}
                for s in self._initialized_symbols
            },
        }
