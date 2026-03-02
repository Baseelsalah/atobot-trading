"""VWAP Scalp day-trading strategy for AtoBot.

Buys when price dips below VWAP by a configured percentage and sells
when it bounces back to or above VWAP (mean-reversion scalp).

**v7 research-driven** (backtest validated):
- MACD death cross exit REMOVED (premature exits hurt avg win)
- MACD/RSI entry confirmation REMOVED (over-filtered VWAP bounces)
- Midday/trend filters REMOVED for VWAP (needs volume from all sessions)
- VWAP touch exit is primary TP signal (proven best for mean reversion)
- Trailing stop provides downside protection
- ATR-based dynamic stop-loss adapts to per-symbol volatility (v7)
- Volume surge detection kept for entry quality
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
from src.utils.helpers import round_quantity


class VWAPScalpStrategy(BaseStrategy):
    """VWAP mean-reversion scalp strategy (v3 research-optimized).

    Entry (BUY):
    * Price is below VWAP by ``VWAP_BOUNCE_PERCENT`` or more.
    * No existing position in the symbol.

    Exit (SELL):
    * Price returns to VWAP (or above) -> take-profit (primary signal).
    * Trailing stop protects gains.
    * Unrealised loss exceeds ``VWAP_STOP_LOSS_PERCENT`` -> stop-loss.
    * End-of-day flatten (handled by engine).

    v3 changes (backtest validated):
    * MACD death cross exit REMOVED (hurt avg win by premature exits).
    * MACD/RSI entry filters REMOVED (over-filtered valid VWAP bounces).
    * Midday avoidance REMOVED for VWAP (needs all-session volume).
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
        # Track MACD/RSI signal windows (from Alpaca example)
        self._signal_window: dict[str, dict] = {}  # symbol -> signal state
        # ATR-based dynamic stop-loss per position (v7)
        self._atr_stops: dict[str, Decimal] = {}  # symbol -> dynamic SL %

    @property
    def name(self) -> str:
        return "vwap_scalp"

    async def initialize(self, symbol: str) -> None:
        logger.info("[VWAP] Initialising for {}", symbol)
        self._symbol_filters[symbol] = await self.exchange.get_symbol_filters(symbol)
        self._signal_window[symbol] = {
            "macd_cross_tick": None,
            "rsi_bounce_tick": None,
            "tick_count": 0,
        }
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

        # Increment tick counter for signal window tracking
        if symbol in self._signal_window:
            self._signal_window[symbol]["tick_count"] += 1

        # ── Fetch intraday bars for VWAP + indicator calculation ─────────
        try:
            bars = await self.exchange.get_klines(symbol, "5m", 80)
        except Exception as exc:
            logger.warning("[VWAP] Data error for {}: {}", symbol, exc)
            return orders

        vwap = self._compute_vwap(bars)
        if vwap is None or vwap <= 0:
            return orders

        # ── Compute MACD and RSI from bar data ───────────────────────────
        import pandas as pd
        macd_info = None
        rsi_info = None
        try:
            df = pd.DataFrame(bars)
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)

            if len(df) >= 35:  # Need enough bars for MACD (26+9)
                macd_info = indicators.macd_signal(df)
            if len(df) >= 16:  # Need enough bars for RSI (14+2)
                rsi_info = indicators.rsi_bounce(df)
        except Exception as exc:
            logger.debug("[VWAP] Indicator calc error for {}: {}", symbol, exc)

        # ── Exit logic ───────────────────────────────────────────────────
        if pos and not pos.is_closed:
            pos.update_price(current_price)
            tp = Decimal(str(self.settings.VWAP_TAKE_PROFIT_PERCENT))
            sl = Decimal(str(self.settings.VWAP_STOP_LOSS_PERCENT))

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
                    logger.info("[VWAP] TRAILING STOP {} | PnL%={:.2f}", symbol, pos.unrealized_pnl_percent)
                return orders

            # MACD death cross exit REMOVED in v3 (backtest proved it hurt avg win)

            # Take profit:
            # v8 fix: VWAP-touch exit now requires the position to be in profit.
            # Previously, `price >= vwap` fired when VWAP drifted below our entry
            # (bearish market) producing only ~0.12% "wins" against 0.50% ATR SLs.
            # Kelly was negative: f* = (0.54×0.12 − 0.46×0.50) / 0.06 = −2.75.
            # Fix: VWAP touch is secondary; primary exit is the hard TP%.
            # VWAP touch only triggers when unrealized PnL > 0 (never exit at loss).
            if pos.side == "LONG":
                tp_triggered = (
                    pos.unrealized_pnl_percent >= tp            # Hard TP% — primary
                    or (current_price >= vwap                   # VWAP return — secondary,
                        and pos.unrealized_pnl_percent > Decimal("0"))  # only when profitable
                )
            else:
                # Short TP: price dropped back to VWAP (or below)
                tp_triggered = (
                    pos.unrealized_pnl_percent >= tp
                    or (current_price <= vwap
                        and pos.unrealized_pnl_percent > Decimal("0"))
                )

            if tp_triggered:
                qty = round_quantity(pos.quantity, filters["step_size"])
                if qty > Decimal("0"):
                    orders.append(Order(
                        symbol=symbol, side=exit_side,
                        order_type=OrderType.MARKET, price=current_price,
                        quantity=qty, strategy=self.name,
                    ))
                    logger.info(
                        "[VWAP] TP {} | price={} vwap={} PnL%={:.2f}",
                        symbol, current_price, vwap, pos.unrealized_pnl_percent,
                    )
                return orders

            # Stop loss (ATR-based dynamic)
            dynamic_sl = self._atr_stops.get(symbol, sl)
            if pos.unrealized_pnl_percent <= -dynamic_sl:
                qty = round_quantity(pos.quantity, filters["step_size"])
                if qty > Decimal("0"):
                    orders.append(Order(
                        symbol=symbol, side=exit_side,
                        order_type=OrderType.MARKET, price=current_price,
                        quantity=qty, strategy=self.name,
                    ))
                    logger.info("[VWAP] SL {} | PnL%={:.2f}", symbol, pos.unrealized_pnl_percent)
                return orders

            return orders

        # ── Entry logic: price below VWAP (v3: no MACD/RSI/midday filters) ──
        # VWAP mean-reversion needs all-session volume; filters removed per backtest
        # Trend/midday filters are NOT applied for VWAP (backtest validated)

        # ── Warm-up guard: block new entries until indicators are valid ───────
        # MACD(12,26,9) = 35 bars minimum. Sub-35-bar VWAP deviations are noisy
        # because the volume-weighted mean is computed from too few data points.
        # Exits are unaffected — the guard is placed BEFORE the entry block.
        _VWAP_MIN_WARMUP = 35  # O(1) constant check
        if len(df) < _VWAP_MIN_WARMUP:
            logger.debug(
                "[VWAP] Warming up {} — {}/{} 5m bars. Entries suppressed.",
                symbol, len(df), _VWAP_MIN_WARMUP,
            )
            return orders

        bounce_pct = Decimal(str(self.settings.VWAP_BOUNCE_PERCENT))
        deviation = ((vwap - current_price) / vwap) * Decimal("100")

        if deviation >= bounce_pct:

            # Confluence gate (v5: multi-indicator quality filter)
            if not await self.passes_confluence_gate(symbol):
                return orders

            # ── ATR-based dynamic stop-loss (v8: cap at 0.35% not 0.50%) ──────
            # v8 fix: old 0.50% cap let ATR SL exceed typical 0.25% wins.
            # Cap at 0.35% keeps SL < TP (0.4%) for positive R:R.
            dynamic_sl_pct = float(self.settings.VWAP_STOP_LOSS_PERCENT)
            if len(df) >= 15:
                tr = pd.concat([
                    df['high'] - df['low'],
                    (df['high'] - df['close'].shift(1)).abs(),
                    (df['low'] - df['close'].shift(1)).abs()
                ], axis=1).max(axis=1)
                atr_14 = tr.rolling(14).mean().iloc[-1]
                if not pd.isna(atr_14):
                    atr_pct = (atr_14 / float(current_price)) * 100
                    baseline_sl = float(self.settings.VWAP_STOP_LOSS_PERCENT)
                    dynamic_sl_pct = max(baseline_sl, min(0.35, atr_pct * 1.5))  # O(1)
            self._atr_stops[symbol] = Decimal(str(round(dynamic_sl_pct, 4)))

            # ── Dynamic position sizing (v5: Kelly + 2% risk + progressive) ──
            quantity = await self.compute_dynamic_quantity(
                symbol, current_price,
                fallback_usd=self.settings.VWAP_ORDER_SIZE_USD,
                stop_loss_pct=dynamic_sl_pct,
            )

            quantity = round_quantity(quantity, filters["step_size"])

            if quantity > Decimal("0"):
                entry_type, entry_price = self._entry_order_type_and_price(current_price, "BUY")
                orders.append(Order(
                    symbol=symbol, side=OrderSide.BUY,
                    order_type=entry_type, price=entry_price,
                    quantity=quantity, strategy=self.name,
                ))
                macd_str = f"MACD={macd_info['macd']:.4f}" if macd_info else "MACD=N/A"
                rsi_str = f"RSI={rsi_info['rsi_now']:.1f}" if rsi_info else "RSI=N/A"
                logger.info(
                    "[VWAP] BUY signal {} | price={} vwap={} dev={:.2f}% | {} | {}",
                    symbol, current_price, vwap, deviation, macd_str, rsi_str,
                )
        else:
            # ── SHORT entry: price ABOVE VWAP (mean-reversion short) ─────
            short_deviation = ((current_price - vwap) / vwap) * Decimal("100")
            if (
                getattr(self.settings, "SHORT_SELLING_ENABLED", False)
                and short_deviation >= bounce_pct
            ):
                # Confluence gate
                if not await self.passes_confluence_gate(symbol):
                    pass  # Fall through to logging
                else:
                    # ATR-based dynamic stop for short (v8: cap at 0.35%)
                    dynamic_sl_pct = float(self.settings.VWAP_STOP_LOSS_PERCENT)
                    if len(df) >= 15:
                        tr = pd.concat([
                            df['high'] - df['low'],
                            (df['high'] - df['close'].shift(1)).abs(),
                            (df['low'] - df['close'].shift(1)).abs()
                        ], axis=1).max(axis=1)
                        atr_14 = tr.rolling(14).mean().iloc[-1]
                        if not pd.isna(atr_14):
                            atr_pct = (atr_14 / float(current_price)) * 100
                            baseline_sl = float(self.settings.VWAP_STOP_LOSS_PERCENT)
                            dynamic_sl_pct = max(baseline_sl, min(0.35, atr_pct * 1.5))
                    self._atr_stops[symbol] = Decimal(str(round(dynamic_sl_pct, 4)))

                    quantity = await self.compute_dynamic_quantity(
                        symbol, current_price,
                        fallback_usd=self.settings.VWAP_ORDER_SIZE_USD,
                        stop_loss_pct=dynamic_sl_pct,
                    )
                    quantity = round_quantity(quantity, filters["step_size"])

                    if quantity > Decimal("0"):
                        entry_type, entry_price = self._entry_order_type_and_price(current_price, "SHORT")
                        orders.append(Order(
                            symbol=symbol, side=OrderSide.SHORT,
                            order_type=entry_type, price=entry_price,
                            quantity=quantity, strategy=self.name,
                        ))
                        logger.info(
                            "[VWAP] SHORT signal {} | price={} vwap={} dev={:.2f}%",
                            symbol, current_price, vwap, short_deviation,
                        )
            elif symbol in self._signal_window and self._signal_window[symbol]["tick_count"] % 60 == 0:
                # Log why no entry (every 60 ticks to avoid spam)
                logger.info(
                    "[VWAP] No entry {} | price={} vwap={} dev={:.2f}% (need {:.2f}%)",
                    symbol, current_price, vwap, deviation, bounce_pct,
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
                pnl = pos.unrealized_pnl
                pos.reduce_position(order.filled_quantity, order.price)
                if pos.is_closed:
                    if pnl > Decimal("0"):
                        self.record_win()
                    else:
                        self.record_loss()
            if pos and pos.is_closed:
                self._reset_trailing_high(symbol)
                self._atr_stops.pop(symbol, None)

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
