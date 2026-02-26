"""Crypto Swing Trading Strategy v3 — 24/7 Alpaca Crypto, Multi-Pair.

Designed for $500-$25K accounts trading crypto on Alpaca.
Captures 3-20% moves over 1-14 days across 8 crypto pairs:
  BTC/USD, ETH/USD, SOL/USD, AVAX/USD, LINK/USD, DOGE/USD, DOT/USD, LTC/USD

Per-asset volatility profiles:
  - BTC: baseline stops/targets/sizing (leader asset)
  - ETH: 1.2x SL, 1.3x TP, 0.9x size
  - SOL/AVAX: 1.5x SL, 1.8x TP, 0.6-0.7x size (high vol alts)
  - DOGE: 1.8x SL, 2.2x TP, 0.5x size, needs 4 confluence (meme coin)
  - LINK/DOT/LTC: 1.3-1.5x SL, 1.5-1.7x TP, 0.6-0.7x size

Gate Filters (must all pass or skip entry):
  - ADX > 20 (skip choppy/ranging markets)
  - Daily EMA20 > EMA50 AND daily RSI > 45 (macro uptrend)
  - BTC trend gate: alts only enter if BTC bullish
  - BTC panic gate: block ALL alt longs when BTC RSI < 30
  - Alt correlation limit: max 3 alts open simultaneously

Entry Signals (need N confluence from 9 possible):
  1. RSI oversold bounce (<35 crossing back above 35) on 4H bars
  2. Price near rising 20-EMA support (within 1.5 ATR)
  3. Volume surge (>1.5x = +1, >3x = +2) — tiered
  4. Bullish candle pattern (hammer, engulfing)
  5. EMA stack alignment (20 > 50 = uptrend)
  6. Price recovery from recent low (momentum shift)
  7. Bollinger Band: price in lower half of BB
  8. MACD histogram rising (momentum confirmation)
  9. RSI bullish divergence (price lower-low, RSI higher-low)

Exit Signals:
  - Multi-level take profit: 33% at TP1, 33% at TP2, 34% at TP3
    (TP levels scaled by per-asset profile multiplier)
  - Stop loss: dynamic ATR-based (3-7% base, scaled by asset profile)
  - Move stop to breakeven after first TP hit
  - Trailing stop: activates at configured %, trails from high
  - Time stop: max hold days
  - No EOD flatten (24/7 market, exempt from stock flatten)

Position Sizing:
  - Risk N% of account per trade (ATR-based stop → adaptive size)
  - Per-asset size multiplier (smaller for volatile alts)
  - Fear & Greed Index adjustment (extreme fear = +25%, extreme greed = -50%)
  - Max 4 concurrent crypto positions, max 3 alts
  - Fractional shares
  - Accounts for 0.25% taker fee in size calc

Fee Structure:
  - Alpaca crypto: 25 bps taker (0.25%) per side
  - Round-trip cost: ~0.50% (factored into min R:R calc)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import aiohttp
import numpy as np
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
    """Crypto swing trading strategy v3 — multi-pair with per-asset profiles.

    Uses 4H-timeframe signals with 24/7 monitoring + daily macro gate.
    Holds positions 1-14 days to capture crypto swings across 8 pairs.
    Features: per-asset volatility profiles, BTC correlation gate,
              ADX regime filter, Bollinger Bands, MACD, RSI divergence,
              multi-level TP, dynamic ATR stops, Fear & Greed sizing.
    """

    # ── Per-asset volatility profiles ─────────────────────────────────────
    # Each coin has different volatility → different SL/TP/sizing
    # sl_mult/tp_mult: multiplier on base stop-loss/take-profit percentages
    # size_mult: multiplier on position size (smaller for higher vol)
    # min_confluence: higher bar for riskier assets
    ASSET_PROFILES: dict[str, dict] = {
        "BTC/USD": {
            "sl_mult": 1.0, "tp_mult": 1.0, "size_mult": 1.0,
            "min_confluence": 3, "label": "Bitcoin", "is_leader": True,
        },
        "ETH/USD": {
            "sl_mult": 1.2, "tp_mult": 1.3, "size_mult": 0.9,
            "min_confluence": 3, "label": "Ethereum", "is_leader": False,
        },
        "SOL/USD": {
            "sl_mult": 1.5, "tp_mult": 1.8, "size_mult": 0.7,
            "min_confluence": 3, "label": "Solana", "is_leader": False,
        },
        "AVAX/USD": {
            "sl_mult": 1.5, "tp_mult": 1.8, "size_mult": 0.6,
            "min_confluence": 3, "label": "Avalanche", "is_leader": False,
        },
        "LINK/USD": {
            "sl_mult": 1.4, "tp_mult": 1.6, "size_mult": 0.7,
            "min_confluence": 3, "label": "Chainlink", "is_leader": False,
        },
        "DOGE/USD": {
            "sl_mult": 1.8, "tp_mult": 2.2, "size_mult": 0.5,
            "min_confluence": 4, "label": "Dogecoin", "is_leader": False,
        },
        "DOT/USD": {
            "sl_mult": 1.5, "tp_mult": 1.7, "size_mult": 0.6,
            "min_confluence": 3, "label": "Polkadot", "is_leader": False,
        },
        "LTC/USD": {
            "sl_mult": 1.3, "tp_mult": 1.5, "size_mult": 0.7,
            "min_confluence": 3, "label": "Litecoin", "is_leader": False,
        },
    }

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
        self._last_daily_fetch: dict[str, datetime] = {}
        self._entry_dates: dict[str, datetime] = {}     # symbol -> entry datetime
        self._swing_highs: dict[str, Decimal] = {}      # symbol -> highest since entry
        self._swing_stops: dict[str, Decimal] = {}       # symbol -> stop loss price
        self._swing_targets: dict[str, Decimal] = {}     # symbol -> final TP price
        self._trailing_active: dict[str, bool] = {}      # symbol -> trailing activated
        self._eval_done: dict[str, str] = {}             # symbol -> hour_str of last eval
        self._tick_count_local: int = 0
        self._btc_trend_bullish: bool | None = None      # BTC trend cache
        self._btc_trend_checked: str = ""                 # Hour str of last BTC check
        self._btc_rsi: float | None = None                # BTC RSI (for panic gate)
        self._cooldown_until: dict[str, datetime] = {}   # symbol -> cooldown expiry after stop-loss

        # ── Multi-level TP tracking ──
        self._tp_levels: dict[str, list[tuple[float, Decimal]]] = {}  # symbol -> [(fraction, price)]
        self._tp_level_hit: dict[str, int] = {}           # symbol -> how many TP levels hit
        self._original_qty: dict[str, Decimal] = {}       # symbol -> original entry qty
        self._breakeven_set: dict[str, bool] = {}         # symbol -> stop moved to breakeven?

        # ── Fear & Greed cache ──
        self._fear_greed_value: int | None = None
        self._fear_greed_fetched: str = ""

        # ── Daily trend cache ──
        self._daily_trend_ok: dict[str, bool] = {}
        self._daily_trend_checked: dict[str, str] = {}

        # Crypto-specific params (base)
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
        self._btc_panic_rsi = float(getattr(settings, 'CRYPTO_BTC_PANIC_RSI', 30.0))
        self._max_alt_positions = int(getattr(settings, 'CRYPTO_MAX_ALT_POSITIONS', 3))
        self._fee_bps = float(getattr(settings, 'CRYPTO_FEE_BPS', 25.0))

        # v2 params: ADX, Bollinger, MACD, multi-TP, Fear & Greed
        self._adx_filter_enabled = bool(getattr(settings, 'CRYPTO_ADX_FILTER_ENABLED', True))
        self._adx_min_trend = float(getattr(settings, 'CRYPTO_ADX_MIN_TREND', 20.0))
        self._bb_filter_enabled = bool(getattr(settings, 'CRYPTO_BB_FILTER_ENABLED', True))
        self._bb_period = int(getattr(settings, 'CRYPTO_BB_PERIOD', 20))
        self._bb_std = float(getattr(settings, 'CRYPTO_BB_STD', 2.0))
        self._macd_enabled = bool(getattr(settings, 'CRYPTO_MACD_ENABLED', True))
        self._daily_trend_gate = bool(getattr(settings, 'CRYPTO_DAILY_TREND_GATE', True))
        self._multi_tp_enabled = bool(getattr(settings, 'CRYPTO_MULTI_TP_ENABLED', True))
        self._tp1_pct = float(getattr(settings, 'CRYPTO_TP1_PCT', 5.0))
        self._tp2_pct = float(getattr(settings, 'CRYPTO_TP2_PCT', 8.0))
        self._tp3_pct = float(getattr(settings, 'CRYPTO_TP3_PCT', 12.0))
        self._dynamic_stops = bool(getattr(settings, 'CRYPTO_DYNAMIC_STOPS', True))
        self._fear_greed_enabled = bool(getattr(settings, 'CRYPTO_FEAR_GREED_ENABLED', True))

        logger.info(
            "[CRYPTO v3] Initialized: TP={:.1f}%/{:.1f}%/{:.1f}% (multi={}), "
            "SL={:.1f}% (dynamic={}), Trail +{:.1f}%/{:.1f}% | "
            "RSI<{:.0f}, ADX>{:.0f} (enabled={}), BB={}/{:.1f} (enabled={}), "
            "MACD={}, DailyGate={}, F&G={} | "
            "max_pos={}, max_alt={}, risk/trade={:.1f}%, hold≤{}d | "
            "equity_cap=${} | BTC_gate={} (panic<{:.0f}) | fee={}bps | "
            "pairs={}",
            self._tp1_pct, self._tp2_pct, self._tp3_pct, self._multi_tp_enabled,
            self._stop_loss_pct, self._dynamic_stops,
            self._trailing_act_pct, self._trailing_offset_pct,
            self._rsi_oversold, self._adx_min_trend, self._adx_filter_enabled,
            self._bb_period, self._bb_std, self._bb_filter_enabled,
            self._macd_enabled, self._daily_trend_gate, self._fear_greed_enabled,
            self._max_positions, self._max_alt_positions, self._risk_per_trade,
            self._max_hold_days,
            int(self._equity_cap) if self._equity_cap else "disabled",
            self._btc_trend_gate, self._btc_panic_rsi,
            int(self._fee_bps),
            getattr(settings, 'CRYPTO_SYMBOLS', 'BTC/USD,ETH/USD'),
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

        # ── Alt exposure limit: max N alts at once (correlation risk) ────
        profile = self._get_asset_profile(symbol)
        if not profile.get("is_leader", False):
            alt_count = sum(
                1 for sym, p in self.positions.items()
                if not p.is_closed and not self._get_asset_profile(sym).get("is_leader", False)
            )
            if alt_count >= self._max_alt_positions:
                logger.debug(
                    "[CRYPTO] Alt limit: {} alts already open, blocking {}",
                    alt_count, symbol,
                )
                return orders

        # Active orders check — don't place if we already have pending
        if self._get_active_orders_for(symbol):
            return orders

        # ── BTC trend gate for non-BTC pairs ─────────────────────────────
        if self._btc_trend_gate and "BTC" not in symbol:
            btc_bullish = await self._check_btc_trend()

            # Hard block: BTC panic selling (RSI < 30) blocks ALL alt longs
            if self._btc_rsi is not None and self._btc_rsi < self._btc_panic_rsi:
                logger.info(
                    "[CRYPTO] BTC PANIC gate BLOCKED {} — BTC RSI={:.0f} < {:.0f}",
                    symbol, self._btc_rsi, self._btc_panic_rsi,
                )
                self._eval_done[symbol] = eval_key
                return orders

            if not btc_bullish:
                logger.debug(
                    "[CRYPTO] BTC trend gate BLOCKED {} — BTC in downtrend", symbol
                )
                self._eval_done[symbol] = eval_key
                return orders

        # ── Daily timeframe macro gate ───────────────────────────────────
        if self._daily_trend_gate:
            daily_ok = await self._check_daily_trend(symbol)
            if not daily_ok:
                logger.debug(
                    "[CRYPTO] Daily trend gate BLOCKED {} — macro downtrend", symbol
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

            confluence, details = self._compute_confluence(df, float(current_price))

            # Per-asset min confluence (e.g., DOGE needs 4, BTC needs 3)
            asset_profile = self._get_asset_profile(symbol)
            min_conf = asset_profile.get("min_confluence", self._min_confluence)

            if confluence >= min_conf:
                entry_order = await self._create_entry_order(
                    symbol, current_price, df
                )
                if entry_order:
                    orders.append(entry_order)
                    logger.info(
                        "[CRYPTO] ENTRY signal {} ({}) | confluence={}/{} (need {}) | price={} | {}",
                        symbol, asset_profile.get("label", symbol),
                        confluence, 9, min_conf, current_price, ", ".join(details),
                    )
            else:
                logger.debug(
                    "[CRYPTO] No entry for {} | confluence={}/{} (need {})",
                    symbol, confluence, 9, min_conf,
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
            self._original_qty[symbol] = qty
            self._tp_level_hit[symbol] = 0
            self._breakeven_set[symbol] = False

            # ── Per-asset profile multipliers ────────────────────────────
            asset_profile = self._get_asset_profile(symbol)
            sl_mult = asset_profile.get("sl_mult", 1.0)
            tp_mult = asset_profile.get("tp_mult", 1.0)

            # ── Dynamic ATR-based stop loss ──────────────────────────────
            if self._dynamic_stops:
                try:
                    bars = self._bar_cache.get(symbol, [])
                    if len(bars) >= 15:
                        df = pd.DataFrame(bars)
                        for col in ("open", "high", "low", "close", "volume"):
                            df[col] = df[col].astype(float)
                        atr = self._calc_atr(df, 14)
                        if atr is not None and float(entry_price) > 0:
                            vol_ratio = atr / float(entry_price)
                            dynamic_sl = max(0.03, min(0.07, vol_ratio * 2)) * 100
                            # Apply per-asset multiplier
                            dynamic_sl *= sl_mult
                            sl_price = entry_price * Decimal(str(1 - dynamic_sl / 100))
                            logger.info(
                                "[CRYPTO] Dynamic SL: ATR={:.0f}, vol_ratio={:.4f}, "
                                "SL={:.2f}% (x{:.1f}) → ${}",
                                atr, vol_ratio, dynamic_sl, sl_mult, sl_price,
                            )
                        else:
                            sl_pct = self._stop_loss_pct * sl_mult
                            sl_price = entry_price * Decimal(str(1 - sl_pct / 100))
                    else:
                        sl_pct = self._stop_loss_pct * sl_mult
                        sl_price = entry_price * Decimal(str(1 - sl_pct / 100))
                except Exception:
                    sl_pct = self._stop_loss_pct * sl_mult
                    sl_price = entry_price * Decimal(str(1 - sl_pct / 100))
            else:
                sl_pct = self._stop_loss_pct * sl_mult
                sl_price = entry_price * Decimal(str(1 - sl_pct / 100))

            self._swing_stops[symbol] = sl_price

            # ── Multi-level take profit setup (scaled by asset profile) ──
            if self._multi_tp_enabled:
                tp1_pct = self._tp1_pct * tp_mult
                tp2_pct = self._tp2_pct * tp_mult
                tp3_pct = self._tp3_pct * tp_mult
                tp1 = entry_price * Decimal(str(1 + tp1_pct / 100))
                tp2 = entry_price * Decimal(str(1 + tp2_pct / 100))
                tp3 = entry_price * Decimal(str(1 + tp3_pct / 100))
                self._tp_levels[symbol] = [
                    (0.33, tp1),   # Take 33% at TP1
                    (0.33, tp2),   # Take 33% at TP2
                    (0.34, tp3),   # Take 34% at TP3
                ]
                final_tp = tp3
                logger.info(
                    "[CRYPTO] Multi-TP {}: TP1=${} (+{:.1f}%), TP2=${} (+{:.1f}%), "
                    "TP3=${} (+{:.1f}%) [x{:.1f}]",
                    symbol, tp1, tp1_pct, tp2, tp2_pct, tp3, tp3_pct, tp_mult,
                )
            else:
                tp_pct = self._take_profit_pct * tp_mult
                tp_price = entry_price * Decimal(str(1 + tp_pct / 100))
                final_tp = tp_price

            self._swing_targets[symbol] = final_tp

            logger.info(
                "[CRYPTO] Position OPENED {} ({}) | entry={} qty={} | "
                "SL={} TP={} | dynamic_sl={} multi_tp={} | "
                "sl_mult={:.1f} tp_mult={:.1f}",
                symbol, asset_profile.get("label", symbol),
                entry_price, qty,
                sl_price, final_tp,
                self._dynamic_stops, self._multi_tp_enabled,
                sl_mult, tp_mult,
            )

        elif side == "SELL":
            # Exit fill — reduce/close position
            pos = self.positions.get(symbol)
            if pos and not pos.is_closed:
                exit_price = order.price
                exit_qty = order.filled_quantity if order.filled_quantity > 0 else order.quantity
                pnl = pos.reduce_position(exit_qty, exit_price)

                pnl_pct = ((exit_price / pos.entry_price) - 1) * 100 if pos.entry_price > 0 else Decimal("0")

                if pos.is_closed:
                    # Fully closed — cleanup
                    self._cleanup_symbol(symbol)
                    logger.info(
                        "[CRYPTO] Position CLOSED {} | entry={} exit={} | "
                        "PnL=${:.2f} ({:.2f}%)",
                        symbol, pos.entry_price, exit_price,
                        float(pnl), float(pnl_pct),
                    )
                else:
                    # Partial exit (multi-TP)
                    logger.info(
                        "[CRYPTO] Partial exit {} | sold {} @ {} | "
                        "remaining={} | PnL=${:.2f}",
                        symbol, exit_qty, exit_price, pos.quantity, float(pnl),
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
            # 24h cooldown after stop-loss to avoid churning
            self._cooldown_until[symbol] = datetime.now(timezone.utc) + timedelta(hours=24)
            orders.append(self._create_exit_order(symbol, pos, "stop_loss"))
            return orders

        # ── 2. Multi-Level Take Profit ───────────────────────────────────
        if self._multi_tp_enabled and symbol in self._tp_levels:
            tp_levels = self._tp_levels[symbol]
            level_hit = self._tp_level_hit.get(symbol, 0)

            if level_hit < len(tp_levels):
                fraction, tp_price = tp_levels[level_hit]
                if current_price >= tp_price:
                    original_qty = self._original_qty.get(symbol, pos.quantity)
                    sell_qty = Decimal(str(float(original_qty) * fraction))
                    sell_qty = max(sell_qty, Decimal("0.000001"))

                    # Don't sell more than we have
                    if sell_qty > pos.quantity:
                        sell_qty = pos.quantity

                    # Check if this is the last level or remaining qty is tiny
                    remaining_after = pos.quantity - sell_qty
                    is_final = (level_hit >= len(tp_levels) - 1) or (
                        remaining_after * current_price < Decimal("1")
                    )
                    if is_final:
                        sell_qty = pos.quantity  # Close entire remaining

                    logger.info(
                        "[CRYPTO] TP{} hit {} | price={} >= {} | "
                        "selling {}/{} ({:.0f}%) | PnL={:.2f}%",
                        level_hit + 1, symbol, current_price, tp_price,
                        sell_qty, pos.quantity, float(fraction) * 100,
                        float(pnl_pct),
                    )

                    self._tp_level_hit[symbol] = level_hit + 1

                    # Move stop to breakeven after first TP hit
                    if level_hit == 0 and not self._breakeven_set.get(symbol, False):
                        # Breakeven = entry + fees (0.5% round trip)
                        breakeven = entry_price * Decimal("1.005")
                        if breakeven > stop_price:
                            self._swing_stops[symbol] = breakeven
                            self._breakeven_set[symbol] = True
                            logger.info(
                                "[CRYPTO] Stop moved to BREAKEVEN {} → {}",
                                symbol, breakeven,
                            )

                    orders.append(self._create_exit_order(
                        symbol, pos, f"tp{level_hit + 1}", sell_qty
                    ))
                    return orders

        # ── 2b. Single Take Profit (fallback if multi-TP disabled or no levels set)
        if not self._multi_tp_enabled or symbol not in self._tp_levels:
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

    def _compute_confluence(self, df: pd.DataFrame, current_price: float) -> tuple[int, list[str]]:
        """Compute entry confluence score (0-10) from 4H bars.

        Returns (score, list_of_signal_names) for logging.
        Includes ADX gate (returns 0 if choppy market).
        """
        confluence = 0
        details: list[str] = []

        try:
            # ── ADX GATE — skip choppy markets ───────────────────────────
            if self._adx_filter_enabled:
                adx = self._calc_adx(df, 14)
                if adx is not None and adx < self._adx_min_trend:
                    logger.debug("[CRYPTO] ADX gate: {:.1f} < {:.0f} — SKIP (choppy)", adx, self._adx_min_trend)
                    return 0, ["ADX_BLOCKED"]
                if adx is not None:
                    details.append(f"ADX={adx:.1f}")

            # 1. RSI oversold bounce
            rsi = self._calc_rsi(df, period=14)
            if rsi is not None and rsi < self._rsi_oversold:
                # Check if RSI was even lower recently (bouncing)
                rsi_series = df['close'].diff()
                if len(rsi_series) > 2 and rsi_series.iloc[-1] > 0:
                    confluence += 1
                    details.append(f"RSI_bounce={rsi:.1f}")

            # 2. Price near 20-EMA support
            ema20 = self._calc_ema(df, 20)
            if ema20 is not None:
                distance_pct = abs(current_price - ema20) / ema20 * 100
                atr = self._calc_atr(df, 14)
                atr_distance = (atr / current_price * 100) * 1.5 if atr else 3.0
                if distance_pct <= atr_distance and current_price >= ema20 * 0.98:
                    confluence += 1
                    details.append(f"EMA20_support={distance_pct:.2f}%")

            # 3. Volume surge (tiered: 3x = +2, 1.5x = +1)
            avg_vol = self._avg_volume(df, 20)
            current_vol = float(df['volume'].iloc[-1])
            if avg_vol > 0:
                vol_ratio = current_vol / avg_vol
                if vol_ratio > 3.0:
                    confluence += 2
                    details.append(f"VOL_massive={vol_ratio:.1f}x")
                elif vol_ratio > self._volume_surge:
                    confluence += 1
                    details.append(f"VOL_surge={vol_ratio:.1f}x")

            # 4. Bullish candle
            if self._is_bullish_candle(df):
                confluence += 1
                details.append("BULLISH_candle")

            # 5. EMA stack (20 > 50 = uptrend)
            ema50 = self._calc_ema(df, 50)
            if ema20 is not None and ema50 is not None and ema20 > ema50:
                confluence += 1
                details.append("EMA_stack_bull")

            # 6. Price recovering from recent low (momentum shift)
            if len(df) >= 10:
                recent_low = float(df['low'].tail(10).min())
                recent_high = float(df['high'].tail(10).max())
                if recent_high > recent_low:
                    position_in_range = (current_price - recent_low) / (recent_high - recent_low)
                    if 0.4 <= position_in_range <= 0.7:
                        confluence += 1
                        details.append(f"recovery={position_in_range:.0%}")

            # 7. Bollinger Band — price in lower half
            if self._bb_filter_enabled:
                bb_upper, bb_middle, bb_lower = self._calc_bollinger(
                    df, self._bb_period, self._bb_std
                )
                if bb_middle is not None and current_price <= bb_middle:
                    confluence += 1
                    details.append("BB_lower_half")

            # 8. MACD histogram rising
            if self._macd_enabled:
                macd_line, signal_line, histogram = self._calc_macd(df, 12, 26, 9)
                if histogram is not None and len(histogram) >= 2:
                    if histogram[-1] > histogram[-2]:
                        confluence += 1
                        details.append(f"MACD_rising={histogram[-1]:.2f}")

            # 9. RSI bullish divergence
            if rsi is not None:
                divergence = self._detect_rsi_divergence(df, lookback=10)
                if divergence == "bullish":
                    confluence += 1
                    details.append("RSI_bull_diverge")

        except Exception as exc:
            logger.debug("[CRYPTO] Confluence calculation error: {}", exc)

        return confluence, details

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
        self, symbol: str, pos: Position, reason: str,
        quantity: Decimal | None = None,
    ) -> Order:
        """Create a market sell order to exit (full or partial) the position."""
        qty = quantity if quantity is not None else pos.quantity
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=qty,
            price=pos.current_price if hasattr(pos, "current_price") else pos.entry_price,
            strategy=self.name,
        )
        return order

    # ── Position Sizing ───────────────────────────────────────────────────────

    async def _calc_crypto_size(self, symbol: str, price: Decimal) -> float:
        """Calculate position size for crypto — risk-based with fee adjustment.

        Accounts for Alpaca's 25 bps taker fee on both entry and exit.
        Adjusts for Fear & Greed sentiment (extreme fear = bigger, extreme greed = smaller).
        Applies per-asset size multiplier (smaller for high-vol alts).
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

            # ── Per-asset size multiplier ────────────────────────────────
            asset_profile = self._get_asset_profile(symbol)
            size_mult = asset_profile.get("size_mult", 1.0)
            position_usd *= size_mult

            # ── Fear & Greed adjustment ──────────────────────────────────
            if self._fear_greed_enabled:
                fg = await self._get_fear_greed()
                if fg is not None:
                    if fg < 25:
                        # Extreme Fear = best time to buy, increase size 25%
                        position_usd *= 1.25
                        logger.debug("[CRYPTO] F&G={} (Extreme Fear) → size +25%", fg)
                    elif fg > 75:
                        # Extreme Greed = risky, cut size 50%
                        position_usd *= 0.50
                        logger.debug("[CRYPTO] F&G={} (Extreme Greed) → size -50%", fg)
                    elif fg > 60:
                        # Greed = slightly reduce
                        position_usd *= 0.75
                        logger.debug("[CRYPTO] F&G={} (Greed) → size -25%", fg)

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
                "[CRYPTO] Size calc {} ({}) | equity=${:.0f} risk=${:.0f} | "
                "position=${:.0f} (x{:.1f}) qty={:.6f} @ {}",
                symbol, asset_profile.get("label", symbol),
                equity, risk_amount, position_usd, size_mult, qty, price,
            )
            return qty

        except Exception as exc:
            logger.warning("[CRYPTO] Size calc error: {} — using fallback", exc)
            fallback_qty = self._order_size_usd / float(price)
            return round(fallback_qty, 6)

    # ── BTC Trend Gate ────────────────────────────────────────────────────────

    async def _check_btc_trend(self) -> bool:
        """Check if BTC is in an uptrend (20-EMA > 50-EMA on daily).

        Also tracks BTC RSI for the panic gate (RSI < 30 = block all alts).
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
                self._btc_rsi = None
                return True

            df = pd.DataFrame(bars)
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)

            ema20 = self._calc_ema(df, 20)
            ema50 = self._calc_ema(df, 50)
            rsi = self._calc_rsi(df, 14)

            # Track RSI for panic gate
            self._btc_rsi = rsi

            if ema20 is not None and ema50 is not None:
                self._btc_trend_bullish = ema20 > ema50
            else:
                self._btc_trend_bullish = True  # Fail-open

            self._btc_trend_checked = check_key
            logger.info(
                "[CRYPTO] BTC trend check: {} (20-EMA={:.0f}, 50-EMA={:.0f}, RSI={:.1f})",
                "BULLISH" if self._btc_trend_bullish else "BEARISH",
                ema20 or 0, ema50 or 0, rsi or 0,
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

    async def _get_daily_bars(self, symbol: str) -> list[dict]:
        """Fetch and cache daily bars. Refresh every 4 hours."""
        now = datetime.now(timezone.utc)
        last_fetch = self._last_daily_fetch.get(symbol)

        if (
            last_fetch
            and (now - last_fetch).total_seconds() < 14400
            and symbol in self._daily_cache
            and len(self._daily_cache[symbol]) >= 20
        ):
            return self._daily_cache[symbol]

        try:
            bars = await self.exchange.get_klines(symbol, "1d", 60)
            if bars:
                self._daily_cache[symbol] = bars
                self._last_daily_fetch[symbol] = now
                logger.debug("[CRYPTO] Fetched {} daily bars for {}", len(bars), symbol)
            return bars or []
        except Exception as exc:
            logger.warning("[CRYPTO] Daily bar fetch failed for {}: {}", symbol, exc)
            return self._daily_cache.get(symbol, [])

    # ── Daily Trend Gate ──────────────────────────────────────────────────────

    async def _check_daily_trend(self, symbol: str) -> bool:
        """Check if the symbol's daily trend is bullish.

        Gate: daily EMA20 > EMA50 AND daily RSI > 45 → allow longs.
        Cached for 4 hours.
        """
        now = datetime.now(timezone.utc)
        check_key = f"{now.strftime('%Y-%m-%d')}-{(now.hour // 4) * 4:02d}"

        if self._daily_trend_checked.get(symbol) == check_key:
            return self._daily_trend_ok.get(symbol, True)

        try:
            bars = await self._get_daily_bars(symbol)
            if len(bars) < 50:
                self._daily_trend_ok[symbol] = True  # Fail-open
                self._daily_trend_checked[symbol] = check_key
                return True

            df = pd.DataFrame(bars)
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)

            ema20 = self._calc_ema(df, 20)
            ema50 = self._calc_ema(df, 50)
            rsi = self._calc_rsi(df, 14)

            if ema20 is not None and ema50 is not None and rsi is not None:
                ok = ema20 > ema50 and rsi > 45
                self._daily_trend_ok[symbol] = ok
                logger.info(
                    "[CRYPTO] Daily trend {} for {}: EMA20={:.0f} vs EMA50={:.0f}, RSI={:.1f}",
                    "BULLISH" if ok else "BEARISH", symbol,
                    ema20, ema50, rsi,
                )
            else:
                self._daily_trend_ok[symbol] = True  # Fail-open

            self._daily_trend_checked[symbol] = check_key
            return self._daily_trend_ok[symbol]

        except Exception as exc:
            logger.warning("[CRYPTO] Daily trend check error for {}: {} — fail-open", symbol, exc)
            self._daily_trend_ok[symbol] = True
            self._daily_trend_checked[symbol] = check_key
            return True

    # ── Fear & Greed Index ────────────────────────────────────────────────────

    async def _get_fear_greed(self) -> int | None:
        """Fetch the crypto Fear & Greed Index (0-100).

        Cached for 4 hours. Returns None on failure (non-blocking).
        """
        now = datetime.now(timezone.utc)
        check_key = f"{now.strftime('%Y-%m-%d')}-{(now.hour // 4) * 4:02d}"

        if self._fear_greed_fetched == check_key and self._fear_greed_value is not None:
            return self._fear_greed_value

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.alternative.me/fng/?limit=1",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        value = int(data["data"][0]["value"])
                        self._fear_greed_value = value
                        self._fear_greed_fetched = check_key
                        logger.info("[CRYPTO] Fear & Greed Index: {} ({})",
                                    value, data["data"][0].get("value_classification", ""))
                        return value
        except Exception as exc:
            logger.debug("[CRYPTO] Fear & Greed fetch failed: {} — using cached", exc)

        return self._fear_greed_value

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
    def _calc_adx(df: pd.DataFrame, period: int = 14) -> float | None:
        """Calculate Average Directional Index (ADX).

        ADX > 25 = trending market, ADX < 20 = choppy/ranging.
        """
        if len(df) < period * 2:
            return None

        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        close = df['close'].values.astype(float)

        # True Range
        tr = np.zeros(len(df))
        tr[0] = high[0] - low[0]
        for i in range(1, len(df)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Directional Movement
        plus_dm = np.zeros(len(df))
        minus_dm = np.zeros(len(df))
        for i in range(1, len(df)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smoothed averages (Wilder's smoothing)
        atr = np.zeros(len(df))
        plus_di_arr = np.zeros(len(df))
        minus_di_arr = np.zeros(len(df))

        atr[period] = np.mean(tr[1:period + 1])
        smooth_plus = np.mean(plus_dm[1:period + 1])
        smooth_minus = np.mean(minus_dm[1:period + 1])

        for i in range(period + 1, len(df)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            smooth_plus = (smooth_plus * (period - 1) + plus_dm[i]) / period
            smooth_minus = (smooth_minus * (period - 1) + minus_dm[i]) / period

            if atr[i] > 0:
                plus_di_arr[i] = (smooth_plus / atr[i]) * 100
                minus_di_arr[i] = (smooth_minus / atr[i]) * 100

        # DX and ADX
        dx = np.zeros(len(df))
        for i in range(period, len(df)):
            di_sum = plus_di_arr[i] + minus_di_arr[i]
            if di_sum > 0:
                dx[i] = abs(plus_di_arr[i] - minus_di_arr[i]) / di_sum * 100

        # ADX = smoothed DX
        adx_start = period * 2
        if adx_start >= len(df):
            return None

        adx_val = np.mean(dx[period:adx_start])
        for i in range(adx_start, len(df)):
            adx_val = (adx_val * (period - 1) + dx[i]) / period

        return float(adx_val)

    @staticmethod
    def _calc_bollinger(
        df: pd.DataFrame, period: int = 20, num_std: float = 2.0
    ) -> tuple[float | None, float | None, float | None]:
        """Calculate Bollinger Bands (upper, middle, lower)."""
        if len(df) < period:
            return None, None, None

        closes = df['close'].astype(float)
        middle = float(closes.rolling(period).mean().iloc[-1])
        std = float(closes.rolling(period).std().iloc[-1])
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower

    @staticmethod
    def _calc_macd(
        df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[float | None, float | None, list[float] | None]:
        """Calculate MACD line, signal line, and histogram series."""
        if len(df) < slow + signal:
            return None, None, None

        closes = df['close'].astype(float)
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return (
            float(macd_line.iloc[-1]),
            float(signal_line.iloc[-1]),
            histogram.values.tolist(),
        )

    @staticmethod
    def _detect_rsi_divergence(df: pd.DataFrame, lookback: int = 10) -> str | None:
        """Detect bullish/bearish RSI divergence.

        Bullish: price makes lower low, RSI makes higher low.
        Returns 'bullish', 'bearish', or None.
        """
        if len(df) < lookback + 14:
            return None

        closes = df['close'].astype(float)
        lows = df['low'].astype(float)

        # Calculate RSI series
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi_series = 100 - (100 / (1 + rs))

        # Look for swing lows in last `lookback` bars
        recent_lows = lows.tail(lookback)
        recent_rsi = rsi_series.tail(lookback)

        if len(recent_lows) < 4:
            return None

        # Find two lowest points (simple approach: split in half)
        half = len(recent_lows) // 2
        first_half_low_idx = recent_lows.iloc[:half].idxmin()
        second_half_low_idx = recent_lows.iloc[half:].idxmin()

        if first_half_low_idx == second_half_low_idx:
            return None

        price_low1 = lows.loc[first_half_low_idx]
        price_low2 = lows.loc[second_half_low_idx]
        rsi_low1 = rsi_series.loc[first_half_low_idx]
        rsi_low2 = rsi_series.loc[second_half_low_idx]

        # Bullish divergence: price lower-low, RSI higher-low
        if price_low2 < price_low1 and rsi_low2 > rsi_low1:
            return "bullish"

        # Bearish divergence: price higher-high, RSI lower-high (using highs)
        highs = df['high'].astype(float)
        recent_highs = highs.tail(lookback)
        first_half_high_idx = recent_highs.iloc[:half].idxmax()
        second_half_high_idx = recent_highs.iloc[half:].idxmax()

        if first_half_high_idx != second_half_high_idx:
            price_high1 = highs.loc[first_half_high_idx]
            price_high2 = highs.loc[second_half_high_idx]
            rsi_high1 = rsi_series.loc[first_half_high_idx]
            rsi_high2 = rsi_series.loc[second_half_high_idx]

            if price_high2 > price_high1 and rsi_high2 < rsi_high1:
                return "bearish"

        return None

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

    # ── Asset Profile Helper ────────────────────────────────────────────────

    def _get_asset_profile(self, symbol: str) -> dict:
        """Get per-asset volatility profile, with safe defaults for unknown pairs."""
        return self.ASSET_PROFILES.get(symbol, {
            "sl_mult": 1.5, "tp_mult": 1.8, "size_mult": 0.5,
            "min_confluence": 4, "label": symbol, "is_leader": False,
        })

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _cleanup_symbol(self, symbol: str) -> None:
        """Remove tracking state for a closed position."""
        self._entry_dates.pop(symbol, None)
        self._swing_highs.pop(symbol, None)
        self._swing_stops.pop(symbol, None)
        self._swing_targets.pop(symbol, None)
        self._trailing_active.pop(symbol, None)
        self._eval_done.pop(symbol, None)
        self._tp_levels.pop(symbol, None)
        self._tp_level_hit.pop(symbol, None)
        self._original_qty.pop(symbol, None)
        self._breakeven_set.pop(symbol, None)
        # NOTE: intentionally NOT clearing _cooldown_until here —
        # cooldown must persist after position cleanup so the bot
        # waits before re-entering the same symbol.
