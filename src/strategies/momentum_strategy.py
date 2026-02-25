"""Momentum day-trading strategy for AtoBot.

Scans for stocks showing strong intraday momentum using RSI + volume
confirmation, enters with a market order, and exits on take-profit /
stop-loss / end-of-day flatten.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from loguru import logger

from src.config.settings import Settings
from src.data import indicators
from src.exchange.base_client import BaseExchangeClient
from src.models.order import Order, OrderSide, OrderType
from src.models.position import Position
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy
from src.utils.helpers import round_price, round_quantity


class MomentumStrategy(BaseStrategy):
    """Momentum day-trading strategy.

    Entry rules  (BUY):
    1. RSI < ``MOMENTUM_RSI_OVERSOLD``  **or** RSI crossing back above it.
    2. Current volume bar ≥ ``MOMENTUM_VOLUME_MULTIPLIER`` × average volume.
    3. No existing position in the symbol.

    Exit rules  (SELL):
    - Unrealised PnL ≥ ``MOMENTUM_TAKE_PROFIT_PERCENT``.
    - Unrealised loss ≥ ``MOMENTUM_STOP_LOSS_PERCENT``.
    - End-of-day flatten (handled by engine).
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
        return "momentum"

    async def initialize(self, symbol: str) -> None:
        """Warm up data for a symbol."""
        logger.info("[Momentum] Initialising for {}", symbol)
        self._symbol_filters[symbol] = await self.exchange.get_symbol_filters(symbol)
        self._initialized_symbols.add(symbol)
        self.is_running = True

    async def on_tick(self, symbol: str, current_price: Decimal) -> list[Order]:
        """Evaluate momentum signals and return orders."""
        if symbol not in self._initialized_symbols:
            await self.initialize(symbol)

        orders: list[Order] = []
        filters = self._symbol_filters[symbol]
        pos = self.positions.get(symbol)

        # ── Exit logic: check open position ──────────────────────────────
        if pos and not pos.is_closed:
            pos.update_price(current_price)
            tp = Decimal(str(self.settings.MOMENTUM_TAKE_PROFIT_PERCENT))
            sl = Decimal(str(self.settings.MOMENTUM_STOP_LOSS_PERCENT))

            # Trailing stop check (before fixed TP/SL)
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
                    logger.info("[Momentum] TRAILING STOP {} | PnL%={:.2f}", symbol, pos.unrealized_pnl_percent)
                return orders

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
                        "[Momentum] TP triggered {} | PnL%={:.2f}",
                        symbol, pos.unrealized_pnl_percent,
                    )
                return orders

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
                        "[Momentum] SL triggered {} | PnL%={:.2f}",
                        symbol, pos.unrealized_pnl_percent,
                    )
                return orders

            # MACD death cross early exit (matches VWAP/ORB behavior)
            if getattr(self.settings, "MACD_CONFIRMATION_ENABLED", True):
                try:
                    ohlcv_exit = await self.exchange.get_klines(symbol, "5m", 40)
                    if ohlcv_exit and len(ohlcv_exit) >= 35:
                        import pandas as pd
                        df_exit = pd.DataFrame(ohlcv_exit)
                        for col in ("open", "high", "low", "close", "volume"):
                            df_exit[col] = df_exit[col].astype(float)
                        macd_exit = indicators.macd_signal(df_exit)
                        if macd_exit.get("death_cross"):
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
                                    "[Momentum] MACD death cross exit {} | PnL%={:.2f}",
                                    symbol, pos.unrealized_pnl_percent,
                                )
                            return orders
                except Exception:
                    pass  # Don't block on data errors

            # Already in position, no new entry
            return orders

        # ── Entry logic: need OHLCV data ─────────────────────────────────
        # Check entry filters first
        if not self._passes_time_filter():
            return orders
        if not await self._passes_trend_filter(symbol, current_price):
            return orders

        try:
            from src.data.market_data import MarketDataProvider

            ohlcv = await self.exchange.get_klines(
                symbol, "5m", self.settings.MOMENTUM_LOOKBACK_BARS + 5
            )
            if len(ohlcv) < self.settings.MOMENTUM_LOOKBACK_BARS:
                return orders

            import pandas as pd
            df = pd.DataFrame(ohlcv)
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)

            # RSI check
            rsi_series = indicators.rsi(df, self.settings.MOMENTUM_RSI_PERIOD)
            current_rsi = rsi_series.iloc[-1]

            # Volume check
            vol_sma = indicators.volume_sma(df, min(20, len(df) - 1))
            current_vol = df["volume"].iloc[-1]
            avg_vol = vol_sma.iloc[-1]
            vol_ok = current_vol >= avg_vol * self.settings.MOMENTUM_VOLUME_MULTIPLIER

            # Entry signal: RSI oversold + volume confirmation + MACD
            rsi_buy = current_rsi <= self.settings.MOMENTUM_RSI_OVERSOLD
            # Also check RSI momentum (crossing back up from oversold)
            if len(rsi_series) >= 2:
                prev_rsi = rsi_series.iloc[-2]
                rsi_crossover = prev_rsi <= self.settings.MOMENTUM_RSI_OVERSOLD and current_rsi > prev_rsi
            else:
                rsi_crossover = False

            # MACD confirmation (v2 improvement)
            macd_ok = True
            if getattr(self.settings, "MACD_CONFIRMATION_ENABLED", True) and len(df) >= 35:
                try:
                    macd_info = indicators.macd_signal(df)
                    macd_ok = macd_info["bullish"] or macd_info["golden_cross"]
                    if not macd_ok:
                        logger.info(
                            "[Momentum] BLOCKED {} | RSI={:.1f} but MACD bearish (MACD={:.4f})",
                            symbol, current_rsi, macd_info["macd"],
                        )
                except Exception:
                    pass  # On error, don't block

            if (rsi_buy or rsi_crossover) and vol_ok and macd_ok:
                # Confluence gate (v5: multi-indicator quality filter)
                if not await self.passes_confluence_gate(symbol):
                    return orders

                # Dynamic position sizing (v5: Kelly + 2% risk + progressive)
                quantity = await self.compute_dynamic_quantity(
                    symbol, current_price,
                    fallback_usd=self.settings.BASE_ORDER_SIZE_USD,
                    stop_loss_pct=self.settings.MOMENTUM_STOP_LOSS_PERCENT,
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
                        "[Momentum] BUY signal {} | RSI={:.1f} vol_ratio={:.1f}x",
                        symbol, current_rsi,
                        current_vol / avg_vol if avg_vol else 0,
                    )

        except Exception as exc:
            logger.warning("[Momentum] Data fetch error for {}: {}", symbol, exc)

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
