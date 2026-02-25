"""Pairs / Statistical Arbitrage Strategy for AtoBot Trading.

Trades mean-reversion of the price spread between two correlated assets.
When the z-score of the spread exceeds PAIRS_ENTRY_ZSCORE, we go long
the underperformer and short the outperformer.  We exit when the spread
reverts toward PAIRS_EXIT_ZSCORE (or stop-out at PAIRS_STOP_ZSCORE).

Pair definitions come from settings.PAIRS (e.g. ["NVDA:AMD", "GOOGL:META"]).
Each pair is traded independently with its own spread tracking.

Key features:
- Rolling z-score of log-price spread
- Hedge-ratio via OLS regression (β)
- Both legs entered simultaneously (dollar-neutral)
- Max holding period to avoid stale trades
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import Settings
from src.data import indicators
from src.exchange.base_client import BaseExchangeClient
from src.models.order import Order, OrderSide, OrderType
from src.models.position import Position
from src.risk.position_sizer import PositionSizer
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy


@dataclass
class PairState:
    """Runtime state for a single pair."""

    sym_a: str
    sym_b: str
    hedge_ratio: float = 1.0          # β — how many shares of B per share of A
    spread_mean: float = 0.0
    spread_std: float = 1.0
    current_zscore: float = 0.0
    position_side: str = ""           # "long_spread" or "short_spread" or ""
    entry_zscore: float = 0.0
    entry_time: datetime | None = None
    bars_held: int = 0

    @property
    def is_flat(self) -> bool:
        return self.position_side == ""

    @property
    def pair_key(self) -> str:
        return f"{self.sym_a}:{self.sym_b}"


class PairsTradingStrategy(BaseStrategy):
    """Statistical arbitrage on pre-defined correlated pairs.

    Lifecycle:
        1. initialize() — fetch historical closes, compute hedge ratio & spread stats
        2. on_tick()     — update spread z-score, generate entry/exit orders
        3. on_order_filled() — track pair positions
    """

    name = "pairs"

    def __init__(
        self,
        exchange_client: BaseExchangeClient,
        risk_manager: RiskManager,
        settings: Settings,
    ) -> None:
        super().__init__(exchange_client, risk_manager, settings)
        self._pairs: dict[str, PairState] = {}    # "NVDA:AMD" -> PairState
        self._prices: dict[str, Decimal] = {}     # Latest prices for all pair symbols
        self._initialized: bool = False

    # ── Initialization ────────────────────────────────────────────────────────

    async def initialize(self, symbol: str = "") -> None:  # type: ignore[override]
        """Initialise all configured pairs (symbol arg ignored for pairs)."""
        if self._initialized:
            return

        pairs_list: list[str] = getattr(self.settings, "PAIRS", [])
        if not pairs_list:
            logger.warning("PairsTradingStrategy: no pairs configured")
            return

        for pair_str in pairs_list:
            parts = pair_str.split(":")
            if len(parts) != 2:
                logger.warning("Invalid pair format '{}', expected 'SYM_A:SYM_B'", pair_str)
                continue
            sym_a, sym_b = parts[0].strip().upper(), parts[1].strip().upper()
            state = PairState(sym_a=sym_a, sym_b=sym_b)
            self._pairs[state.pair_key] = state

            # Fetch historical daily closes for spread calculation
            try:
                await self._calibrate_pair(state)
            except Exception as e:
                logger.error("Failed to calibrate pair {}: {}", state.pair_key, e)

        self._initialized = True
        logger.info(
            "PairsTradingStrategy initialised with {} pairs: {}",
            len(self._pairs),
            list(self._pairs.keys()),
        )

    async def _calibrate_pair(self, state: PairState) -> None:
        """Compute hedge ratio and spread statistics from historical data."""
        lookback = getattr(self.settings, "PAIRS_LOOKBACK_DAYS", 60)

        # Fetch daily bars for both legs
        df_a = await self.exchange.get_klines(
            symbol=state.sym_a, interval="1d", limit=lookback
        )
        df_b = await self.exchange.get_klines(
            symbol=state.sym_b, interval="1d", limit=lookback
        )

        if not df_a or not df_b:
            logger.warning("No kline data for pair {}", state.pair_key)
            return

        closes_a = np.array([float(k["close"]) for k in df_a])
        closes_b = np.array([float(k["close"]) for k in df_b])
        n = min(len(closes_a), len(closes_b))
        if n < 20:
            logger.warning("Insufficient data for pair {} ({} bars)", state.pair_key, n)
            return

        closes_a = closes_a[-n:]
        closes_b = closes_b[-n:]

        # OLS hedge ratio: A = β × B + α  → β = cov(A,B) / var(B)
        cov = np.cov(closes_a, closes_b)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
        state.hedge_ratio = round(float(beta), 4)

        # Spread = A - β × B
        spread = closes_a - beta * closes_b
        state.spread_mean = float(np.mean(spread))
        state.spread_std = float(np.std(spread, ddof=1))
        if state.spread_std == 0:
            state.spread_std = 1.0

        logger.info(
            "Pair {} calibrated: β={:.4f}, spread μ={:.2f}, σ={:.2f}",
            state.pair_key, state.hedge_ratio, state.spread_mean, state.spread_std,
        )

    # ── Tick Processing ───────────────────────────────────────────────────────

    async def on_tick(self, symbol: str, current_price: Decimal) -> list[Order]:
        """Update prices for pair symbols and generate orders.

        The engine calls on_tick once per symbol.  We accumulate prices and
        only evaluate a pair when BOTH legs have fresh prices.
        """
        self._prices[symbol] = current_price
        orders: list[Order] = []

        for key, state in self._pairs.items():
            if symbol not in (state.sym_a, state.sym_b):
                continue

            price_a = self._prices.get(state.sym_a)
            price_b = self._prices.get(state.sym_b)
            if price_a is None or price_b is None:
                continue

            # Compute current spread z-score
            spread = float(price_a) - state.hedge_ratio * float(price_b)
            state.current_zscore = (
                (spread - state.spread_mean) / state.spread_std
                if state.spread_std > 0 else 0.0
            )

            if not state.is_flat:
                state.bars_held += 1

            pair_orders = self._evaluate_pair(state, price_a, price_b)
            orders.extend(pair_orders)

        return orders

    def _evaluate_pair(
        self,
        state: PairState,
        price_a: Decimal,
        price_b: Decimal,
    ) -> list[Order]:
        """Generate entry or exit orders for a single pair."""
        orders: list[Order] = []
        z = state.current_zscore

        entry_z = getattr(self.settings, "PAIRS_ENTRY_ZSCORE", 2.0)
        exit_z = getattr(self.settings, "PAIRS_EXIT_ZSCORE", 0.5)
        stop_z = getattr(self.settings, "PAIRS_STOP_ZSCORE", 3.5)
        max_bars = getattr(self.settings, "PAIRS_MAX_HOLDING_BARS", 100)
        per_leg_usd = getattr(self.settings, "PAIRS_ORDER_SIZE_USD", 5000.0)

        # ── EXIT logic ────────────────────────────────────────────────────
        if not state.is_flat:
            should_exit = False
            reason = ""

            # Mean reversion exit
            if state.position_side == "long_spread" and z <= exit_z:
                should_exit = True
                reason = f"z-score reverted to {z:.2f} (exit at {exit_z})"
            elif state.position_side == "short_spread" and z >= -exit_z:
                should_exit = True
                reason = f"z-score reverted to {z:.2f} (exit at -{exit_z})"

            # Stop-loss
            if state.position_side == "long_spread" and z < -stop_z:
                should_exit = True
                reason = f"stop-loss z={z:.2f} (stop at -{stop_z})"
            elif state.position_side == "short_spread" and z > stop_z:
                should_exit = True
                reason = f"stop-loss z={z:.2f} (stop at {stop_z})"

            # Max holding period
            if state.bars_held >= max_bars:
                should_exit = True
                reason = f"max holding period {max_bars} bars"

            if should_exit:
                logger.info(
                    "Pairs EXIT {} [{}] — {}",
                    state.pair_key, state.position_side, reason,
                )
                orders.extend(self._close_pair_orders(state, price_a, price_b))
                state.position_side = ""
                state.bars_held = 0
                state.entry_zscore = 0
                state.entry_time = None

            return orders

        # ── ENTRY logic ───────────────────────────────────────────────────
        if not getattr(self.settings, "PAIRS_TRADING_ENABLED", False):
            return orders

        qty_a = max(1, int(per_leg_usd / float(price_a)))
        qty_b = max(1, int(per_leg_usd / float(price_b)))

        if z > entry_z:
            # Spread is wide (A overpriced relative to B)
            # → SHORT A, LONG B (sell the spread)
            logger.info(
                "Pairs ENTRY short_spread {} | z={:.2f} > {:.1f} | "
                "SHORT {} qty={}, BUY {} qty={}",
                state.pair_key, z, entry_z,
                state.sym_a, qty_a, state.sym_b, qty_b,
            )
            orders.append(self._make_order(
                state.sym_a, OrderSide.SHORT, price_a, qty_a,
                f"pairs_short_{state.pair_key}",
            ))
            orders.append(self._make_order(
                state.sym_b, OrderSide.BUY, price_b, qty_b,
                f"pairs_long_{state.pair_key}",
            ))
            state.position_side = "short_spread"
            state.entry_zscore = z
            state.entry_time = datetime.now(timezone.utc)
            state.bars_held = 0

        elif z < -entry_z:
            # Spread is narrow (A underpriced relative to B)
            # → LONG A, SHORT B (buy the spread)
            logger.info(
                "Pairs ENTRY long_spread {} | z={:.2f} < -{:.1f} | "
                "BUY {} qty={}, SHORT {} qty={}",
                state.pair_key, z, entry_z,
                state.sym_a, qty_a, state.sym_b, qty_b,
            )
            orders.append(self._make_order(
                state.sym_a, OrderSide.BUY, price_a, qty_a,
                f"pairs_long_{state.pair_key}",
            ))
            orders.append(self._make_order(
                state.sym_b, OrderSide.SHORT, price_b, qty_b,
                f"pairs_short_{state.pair_key}",
            ))
            state.position_side = "long_spread"
            state.entry_zscore = z
            state.entry_time = datetime.now(timezone.utc)
            state.bars_held = 0

        return orders

    # ── Order Helpers ─────────────────────────────────────────────────────────

    def _make_order(
        self,
        symbol: str,
        side: OrderSide,
        price: Decimal,
        quantity: int,
        tag: str,
    ) -> Order:
        """Create a market order for a pair leg."""
        return Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=Decimal(str(quantity)),
            price=price,
            strategy="pairs",
            metadata={"tag": tag},
        )

    def _close_pair_orders(
        self,
        state: PairState,
        price_a: Decimal,
        price_b: Decimal,
    ) -> list[Order]:
        """Generate closing orders for both legs."""
        orders: list[Order] = []
        per_leg_usd = getattr(self.settings, "PAIRS_ORDER_SIZE_USD", 5000.0)
        qty_a = max(1, int(per_leg_usd / float(price_a)))
        qty_b = max(1, int(per_leg_usd / float(price_b)))

        if state.position_side == "short_spread":
            # Was SHORT A, LONG B → COVER A, SELL B
            orders.append(self._make_order(
                state.sym_a, OrderSide.COVER, price_a, qty_a,
                f"pairs_cover_{state.pair_key}",
            ))
            orders.append(self._make_order(
                state.sym_b, OrderSide.SELL, price_b, qty_b,
                f"pairs_sell_{state.pair_key}",
            ))
        elif state.position_side == "long_spread":
            # Was LONG A, SHORT B → SELL A, COVER B
            orders.append(self._make_order(
                state.sym_a, OrderSide.SELL, price_a, qty_a,
                f"pairs_sell_{state.pair_key}",
            ))
            orders.append(self._make_order(
                state.sym_b, OrderSide.COVER, price_b, qty_b,
                f"pairs_cover_{state.pair_key}",
            ))

        return orders

    async def on_order_filled(self, order: Order) -> list[Order]:
        """Handle fill notifications for pair legs.

        Pairs trades are market orders so we don't generate follow-up
        bracket orders — exit logic is handled in on_tick via z-score.
        """
        side_str = order.side if isinstance(order.side, str) else order.side.value
        logger.info(
            "Pairs fill: {} {} {} qty={} @ {}",
            side_str, order.symbol, order.strategy,
            order.quantity, order.price,
        )

        # Track in positions dict for the base strategy
        if side_str in ("BUY", "SHORT"):
            self.positions[order.symbol] = Position(
                symbol=order.symbol,
                side="LONG" if side_str == "BUY" else "SHORT",
                entry_price=order.price,
                quantity=order.quantity,
                strategy="pairs",
            )
        elif side_str in ("SELL", "COVER"):
            self.positions.pop(order.symbol, None)

        return []

    # ── Recalibration ─────────────────────────────────────────────────────────

    async def recalibrate(self) -> None:
        """Re-compute hedge ratios and spread stats (call periodically)."""
        for state in self._pairs.values():
            if state.is_flat:
                try:
                    await self._calibrate_pair(state)
                except Exception as e:
                    logger.error("Recalibrate failed for {}: {}", state.pair_key, e)

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_pair_states(self) -> list[dict]:
        """Return current state of all pairs (for dashboard)."""
        return [
            {
                "pair": s.pair_key,
                "hedge_ratio": s.hedge_ratio,
                "zscore": round(s.current_zscore, 3),
                "position": s.position_side,
                "bars_held": s.bars_held,
                "entry_zscore": round(s.entry_zscore, 3) if s.entry_zscore else None,
            }
            for s in self._pairs.values()
        ]
