"""EMA Pullback day-trading strategy for AtoBot.

Trend-following strategy that enters on pullbacks to fast EMA in the
direction of the higher-timeframe trend. Complements VWAP mean-reversion
by capturing trending moves.

**Research-backed** (Reddit consensus + "Advanced Day Trading" book):
- "100 EMA on 1H for direction, trade 1m/5m with trend"
- "Price action, market structure, liquidity — that's it"
- "Master your stop loss = more good weeks"

Uses 3-layer EMA stack (9/21/50) on 5-min bars:
- 50 EMA defines trend direction (price must be above for longs)
- 9/21 EMA crossover confirms momentum
- Entry on pullback to 9 or 21 EMA with volume confirmation
- RSI in pullback zone (35-55) confirms oversold-in-uptrend

Exit rules:
- Trailing stop (primary profit capture)
- Take-profit at configured %
- Stop-loss at configured %
- RSI overbought exit
- End-of-day flatten (handled by engine)
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
from src.utils.helpers import round_price, round_quantity


class EMAPullbackStrategy(BaseStrategy):
    """EMA Pullback trend-following strategy.

    Entry (BUY):
    * Price is above 50 EMA (uptrend confirmed)
    * 9 EMA > 21 EMA (momentum aligned)
    * Price pulls back to touch or near 9 EMA (within 0.15%)
    * RSI in pullback zone (35-55 — oversold within uptrend)
    * Volume above average (smart money participating)
    * No existing position in the symbol.

    Exit (SELL):
    * Trailing stop (primary — captures trending moves)
    * RSI >= overbought threshold (momentum exhaustion)
    * Take-profit % hit
    * Stop-loss % hit
    * End-of-day flatten (handled by engine)
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

    @property
    def name(self) -> str:
        return "ema_pullback"

    async def initialize(self, symbol: str) -> None:
        logger.info("[EMA-PB] Initialising for {}", symbol)
        self._symbol_filters[symbol] = await self.exchange.get_symbol_filters(symbol)
        self._initialized_symbols.add(symbol)
        self.is_running = True

    async def on_tick(self, symbol: str, current_price: Decimal) -> list[Order]:
        """Evaluate EMA pullback signals and return orders."""
        if symbol not in self._initialized_symbols:
            await self.initialize(symbol)

        orders: list[Order] = []
        filters = self._symbol_filters[symbol]
        pos = self.positions.get(symbol)

        # ── Exit logic ───────────────────────────────────────────────
        if pos and not pos.is_closed:
            pos.update_price(current_price)
            tp = Decimal(str(self.settings.EMA_PULLBACK_TAKE_PROFIT_PERCENT))
            sl = Decimal(str(self.settings.EMA_PULLBACK_STOP_LOSS_PERCENT))

            # Trailing stop (primary exit — lets trends run)
            if self._check_trailing_stop(symbol, pos, current_price):
                sell_qty = round_quantity(pos.quantity, filters["step_size"])
                if sell_qty > Decimal("0"):
                    orders.append(Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        price=current_price,
                        quantity=sell_qty,
                        strategy=self.name,
                    ))
                    logger.info(
                        "[EMA-PB] TRAILING STOP {} | PnL%={:.2f}",
                        symbol, pos.unrealized_pnl_percent,
                    )
                return orders

            # Take-profit
            if pos.unrealized_pnl_percent >= tp:
                sell_qty = round_quantity(pos.quantity, filters["step_size"])
                if sell_qty > Decimal("0"):
                    orders.append(Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        price=current_price,
                        quantity=sell_qty,
                        strategy=self.name,
                    ))
                    logger.info(
                        "[EMA-PB] TP triggered {} | PnL%={:.2f}",
                        symbol, pos.unrealized_pnl_percent,
                    )
                return orders

            # Stop-loss
            if pos.unrealized_pnl_percent <= -sl:
                sell_qty = round_quantity(pos.quantity, filters["step_size"])
                if sell_qty > Decimal("0"):
                    orders.append(Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        price=current_price,
                        quantity=sell_qty,
                        strategy=self.name,
                    ))
                    logger.info(
                        "[EMA-PB] SL triggered {} | PnL%={:.2f}",
                        symbol, pos.unrealized_pnl_percent,
                    )
                return orders

            # RSI overbought exit (momentum exhaustion)
            try:
                ohlcv = await self.exchange.get_klines(symbol, "5m", 30)
                if ohlcv and len(ohlcv) >= 20:
                    df = pd.DataFrame(ohlcv)
                    for col in ("open", "high", "low", "close", "volume"):
                        df[col] = df[col].astype(float)
                    rsi_series = indicators.rsi(df, 14)
                    if len(rsi_series) > 0:
                        current_rsi = rsi_series.iloc[-1]
                        if current_rsi >= self.settings.EMA_PULLBACK_RSI_OVERBOUGHT:
                            sell_qty = round_quantity(pos.quantity, filters["step_size"])
                            if sell_qty > Decimal("0"):
                                orders.append(Order(
                                    symbol=symbol,
                                    side=OrderSide.SELL,
                                    order_type=OrderType.MARKET,
                                    price=current_price,
                                    quantity=sell_qty,
                                    strategy=self.name,
                                ))
                                logger.info(
                                    "[EMA-PB] RSI overbought exit {} | RSI={:.1f} PnL%={:.2f}",
                                    symbol, current_rsi, pos.unrealized_pnl_percent,
                                )
                            return orders
            except Exception:
                pass  # Don't block on data errors

            return orders

        # ── Entry logic ──────────────────────────────────────────────
        try:
            fast_period = self.settings.EMA_PULLBACK_FAST_PERIOD    # 9
            slow_period = self.settings.EMA_PULLBACK_SLOW_PERIOD    # 21
            trend_period = self.settings.EMA_PULLBACK_TREND_PERIOD  # 50

            # Need enough bars for the longest EMA + warmup
            bars_needed = trend_period + 10
            ohlcv = await self.exchange.get_klines(symbol, "5m", bars_needed)
            if len(ohlcv) < trend_period:
                return orders

            df = pd.DataFrame(ohlcv)
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)

            # Compute EMAs
            ema_fast = indicators.ema(df, fast_period)
            ema_slow = indicators.ema(df, slow_period)
            ema_trend = indicators.ema(df, trend_period)

            price_f = float(current_price)
            ema_fast_val = ema_fast.iloc[-1]
            ema_slow_val = ema_slow.iloc[-1]
            ema_trend_val = ema_trend.iloc[-1]

            # ── Condition 1: Price above 50 EMA (uptrend) ──
            if price_f <= ema_trend_val:
                return orders

            # ── Condition 2: 9 EMA > 21 EMA (momentum aligned) ──
            if ema_fast_val <= ema_slow_val:
                return orders

            # ── Condition 3: Price pulling back to fast EMA ──
            # Price must be within 0.15% above the 9 EMA (touching it)
            pullback_zone = ema_fast_val * 1.0015
            if price_f > pullback_zone:
                return orders  # Price too far above — not a pullback
            if price_f < ema_fast_val * 0.998:
                return orders  # Price too far below — broken structure

            # ── Condition 4: RSI in pullback zone ──
            rsi_series = indicators.rsi(df, 14)
            if len(rsi_series) < 1:
                return orders
            current_rsi = rsi_series.iloc[-1]
            if current_rsi < self.settings.EMA_PULLBACK_RSI_OVERSOLD or \
               current_rsi > 55:
                return orders  # RSI not in pullback sweet spot

            # ── Condition 5: Volume confirmation ──
            vol_sma = indicators.volume_sma(df, min(20, len(df) - 1))
            current_vol = df["volume"].iloc[-1]
            avg_vol = vol_sma.iloc[-1]
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0
            if vol_ratio < self.settings.EMA_PULLBACK_VOLUME_MULTIPLIER:
                return orders

            # ── All conditions met — enter ──
            # Confluence gate (v5: multi-indicator quality filter)
            if not await self.passes_confluence_gate(symbol):
                return orders

            # Dynamic position sizing (v5: Kelly + 2% risk + progressive)
            quantity = await self.compute_dynamic_quantity(
                symbol, current_price,
                fallback_usd=getattr(self.settings, "EMA_PULLBACK_ORDER_SIZE_USD",
                                     self.settings.BASE_ORDER_SIZE_USD),
                stop_loss_pct=self.settings.EMA_PULLBACK_STOP_LOSS_PERCENT,
            )
            quantity = round_quantity(quantity, filters["step_size"])

            if quantity > Decimal("0"):
                orders.append(Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    price=current_price,
                    quantity=quantity,
                    strategy=self.name,
                ))
                logger.info(
                    "[EMA-PB] BUY signal {} | RSI={:.1f} vol_ratio={:.1f}x "
                    "price={:.2f} ema9={:.2f} ema21={:.2f} ema50={:.2f}",
                    symbol, current_rsi, vol_ratio, price_f,
                    ema_fast_val, ema_slow_val, ema_trend_val,
                )

        except Exception as exc:
            logger.warning("[EMA-PB] Data fetch error for {}: {}", symbol, exc)

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
        elif order.side == OrderSide.SELL:
            pos = self.positions.get(symbol)
            if pos and not pos.is_closed:
                # Track win/loss for progressive risk scaling (v5)
                pnl = pos.unrealized_pnl
                pos.reduce_position(order.filled_quantity, order.price)
                if pos.is_closed:
                    if pnl > Decimal("0"):
                        self.record_win()
                    else:
                        self.record_loss()
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
        total_unrealized = sum(
            p.unrealized_pnl for p in self.positions.values() if not p.is_closed
        )
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
            "unrealized_pnl": str(total_unrealized),
        }
