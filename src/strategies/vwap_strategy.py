"""VWAP Scalp day-trading strategy for AtoBot.

Buys when price dips below VWAP by a configured percentage and sells
when it bounces back to or above VWAP (mean-reversion scalp).
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from loguru import logger

from src.config.settings import Settings
from src.exchange.base_client import BaseExchangeClient
from src.models.order import Order, OrderSide, OrderType
from src.models.position import Position
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy
from src.utils.helpers import round_quantity


class VWAPScalpStrategy(BaseStrategy):
    """VWAP mean-reversion scalp strategy.

    Entry (BUY):
    * Price is below VWAP by ``VWAP_BOUNCE_PERCENT`` or more.
    * No existing position in the symbol.

    Exit (SELL):
    * Price returns to VWAP (or above) → take-profit.
    * Unrealised loss exceeds ``VWAP_STOP_LOSS_PERCENT`` → stop-loss.
    * End-of-day flatten (handled by engine).
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
        return "vwap_scalp"

    async def initialize(self, symbol: str) -> None:
        logger.info("[VWAP] Initialising for {}", symbol)
        self._symbol_filters[symbol] = await self.exchange.get_symbol_filters(symbol)
        self._initialized_symbols.add(symbol)
        self.is_running = True

    def _compute_vwap(self, bars: list[dict]) -> Decimal | None:
        """Compute VWAP from intraday bars.

        VWAP = Σ(typical_price × volume) / Σ(volume)
        typical_price = (high + low + close) / 3
        """
        if not bars:
            return None
        total_tp_vol = Decimal("0")
        total_vol = Decimal("0")
        for b in bars:
            high = Decimal(str(b["high"]))
            low = Decimal(str(b["low"]))
            close = Decimal(str(b["close"]))
            volume = Decimal(str(b["volume"]))
            if volume <= 0:
                continue
            typical = (high + low + close) / Decimal("3")
            total_tp_vol += typical * volume
            total_vol += volume
        if total_vol <= 0:
            return None
        return total_tp_vol / total_vol

    async def on_tick(self, symbol: str, current_price: Decimal) -> list[Order]:
        if symbol not in self._initialized_symbols:
            await self.initialize(symbol)

        orders: list[Order] = []
        filters = self._symbol_filters[symbol]
        pos = self.positions.get(symbol)

        # ── Fetch intraday bars for VWAP calculation ─────────────────────
        try:
            bars = await self.exchange.get_klines(symbol, "5m", 80)
        except Exception as exc:
            logger.warning("[VWAP] Data error for {}: {}", symbol, exc)
            return orders

        vwap = self._compute_vwap(bars)
        if vwap is None or vwap <= 0:
            return orders

        # ── Exit logic ───────────────────────────────────────────────────
        if pos and not pos.is_closed:
            pos.update_price(current_price)
            tp = Decimal(str(self.settings.VWAP_TAKE_PROFIT_PERCENT))
            sl = Decimal(str(self.settings.VWAP_STOP_LOSS_PERCENT))

            # Trailing stop check (before fixed TP/SL)
            if self._check_trailing_stop(symbol, pos, current_price):
                qty = round_quantity(pos.quantity, filters["step_size"])
                if qty > Decimal("0"):
                    orders.append(Order(
                        symbol=symbol, side=OrderSide.SELL,
                        order_type=OrderType.MARKET, price=current_price,
                        quantity=qty, strategy=self.name,
                    ))
                    logger.info("[VWAP] TRAILING STOP {} | PnL%={:.2f}", symbol, pos.unrealized_pnl_percent)
                return orders

            # Take profit: price returned to VWAP (or above) or PnL target
            price_above_vwap = current_price >= vwap
            if price_above_vwap or pos.unrealized_pnl_percent >= tp:
                qty = round_quantity(pos.quantity, filters["step_size"])
                if qty > Decimal("0"):
                    orders.append(Order(
                        symbol=symbol, side=OrderSide.SELL,
                        order_type=OrderType.MARKET, price=current_price,
                        quantity=qty, strategy=self.name,
                    ))
                    logger.info(
                        "[VWAP] TP {} | price={} vwap={} PnL%={:.2f}",
                        symbol, current_price, vwap, pos.unrealized_pnl_percent,
                    )
                return orders

            # Stop loss
            if pos.unrealized_pnl_percent <= -sl:
                qty = round_quantity(pos.quantity, filters["step_size"])
                if qty > Decimal("0"):
                    orders.append(Order(
                        symbol=symbol, side=OrderSide.SELL,
                        order_type=OrderType.MARKET, price=current_price,
                        quantity=qty, strategy=self.name,
                    ))
                    logger.info("[VWAP] SL {} | PnL%={:.2f}", symbol, pos.unrealized_pnl_percent)
                return orders

            return orders

        # ── Entry logic: price below VWAP ────────────────────────────────
        # Check entry filters first
        if not self._passes_time_filter():
            return orders
        if not await self._passes_trend_filter(symbol, current_price):
            return orders

        bounce_pct = Decimal(str(self.settings.VWAP_BOUNCE_PERCENT))
        deviation = ((vwap - current_price) / vwap) * Decimal("100")

        if deviation >= bounce_pct:
            order_usd = Decimal(str(self.settings.VWAP_ORDER_SIZE_USD))
            quantity = order_usd / current_price
            quantity = round_quantity(quantity, filters["step_size"])

            if quantity > Decimal("0"):
                orders.append(Order(
                    symbol=symbol, side=OrderSide.BUY,
                    order_type=OrderType.MARKET, price=current_price,
                    quantity=quantity, strategy=self.name,
                ))
                logger.info(
                    "[VWAP] BUY signal {} | price={} vwap={} deviation={:.2f}%",
                    symbol, current_price, vwap, deviation,
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
                # Reset trailing high for new position
                self._trailing_highs[symbol] = order.price
            else:
                pos.add_to_position(order.price, order.filled_quantity)
        elif order.side == OrderSide.SELL:
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
        }
