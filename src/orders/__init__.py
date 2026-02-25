"""Smart Order Manager — bracket orders, partial profit-taking, scale-in.

Inspired by Jesse-AI's smart ordering and freqtrade's custom stoploss.
Provides professional-grade order management on top of AtoBot's
existing exchange client.

Features:
- Bracket orders (entry + stop-loss + take-profit as one logical unit)
- Multi-target exits (TP1 at 50%, TP2 at remainder)
- Scale-in to winners (pyramiding with tighter stops)
- OCO (one-cancels-other) exit management
- Trailing bracket updates
- Slippage tracking per fill
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from loguru import logger


class BracketState(str, Enum):
    PENDING = "PENDING"
    ENTRY_PLACED = "ENTRY_PLACED"
    ACTIVE = "ACTIVE"          # Entry filled, TP/SL managed
    PARTIAL_TP = "PARTIAL_TP"  # TP1 hit, remainder running
    CLOSING = "CLOSING"        # SL or final TP hit
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


@dataclass
class ExitTarget:
    """A single exit target (take-profit leg)."""
    label: str                          # "TP1", "TP2", "SL"
    price: Decimal                      # Target price
    quantity_pct: float                  # Fraction of position (0-1)
    order_id: str | None = None         # Exchange order ID once placed
    filled: bool = False
    filled_price: Decimal | None = None
    filled_at: datetime | None = None


@dataclass
class BracketOrder:
    """A bracket order: entry + multiple exits (TP legs + stop-loss).

    The order manager tracks the lifecycle and adjusts child orders
    as partial fills occur.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    side: str = "BUY"      # Entry side
    strategy: str = ""
    state: BracketState = BracketState.PENDING

    # Entry
    entry_price: Decimal = Decimal("0")  # Expected entry price
    entry_type: str = "MARKET"            # MARKET or LIMIT
    quantity: Decimal = Decimal("0")
    entry_order_id: str | None = None
    filled_price: Decimal | None = None   # Actual fill
    filled_qty: Decimal = Decimal("0")

    # Exits
    stop_loss: Decimal = Decimal("0")     # Hard stop-loss price
    stop_order_id: str | None = None
    targets: list[ExitTarget] = field(default_factory=list)

    # Scale-in tracking
    scale_ins: list[dict] = field(default_factory=list)
    max_scale_ins: int = 2

    # Trailing stop
    trailing_active: bool = False
    trailing_activation_pct: float = 0.01  # 1%
    trailing_distance_pct: float = 0.005   # 0.5%
    highest_price: Decimal = Decimal("0")
    trailing_stop_price: Decimal = Decimal("0")

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime | None = None

    # Execution quality
    expected_slippage: Decimal = Decimal("0")
    realized_slippage: Decimal = Decimal("0")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "state": self.state.value,
            "entry_price": str(self.entry_price),
            "filled_price": str(self.filled_price) if self.filled_price else None,
            "quantity": str(self.quantity),
            "stop_loss": str(self.stop_loss),
            "targets": [
                {"label": t.label, "price": str(t.price), "pct": t.quantity_pct, "filled": t.filled}
                for t in self.targets
            ],
            "state": self.state.value,
            "trailing_stop_price": str(self.trailing_stop_price),
        }


class SmartOrderManager:
    """Manages bracket orders, partial exits, and scale-ins.

    Works alongside the existing exchange client — wraps its
    place_market_order / place_limit_order with orchestration logic.
    """

    def __init__(self, exchange_client, settings=None):
        self.exchange = exchange_client
        self.settings = settings
        self._brackets: dict[str, BracketOrder] = {}  # bracket_id -> BracketOrder
        self._symbol_brackets: dict[str, list[str]] = {}  # symbol -> [bracket_ids]
        self._fills: list[dict] = []  # Execution quality log

    # ── Creation ──────────────────────────────────────────────────────────────

    def create_bracket(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        entry_price: Decimal,
        stop_loss: Decimal,
        targets: list[dict] | None = None,
        entry_type: str = "MARKET",
        strategy: str = "",
        trailing_activation_pct: float = 0.01,
        trailing_distance_pct: float = 0.005,
        max_scale_ins: int = 2,
    ) -> BracketOrder:
        """Create a bracket order with entry + SL + optional TP targets.

        ``targets`` is a list of dicts: [{"price": Decimal, "pct": 0.5}, ...]
        where pct is fraction of total qty for that target.  If omitted,
        a single 100% TP is NOT set (just SL), allowing trailing stop to manage exit.
        """
        bracket = BracketOrder(
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
            entry_price=entry_price,
            entry_type=entry_type.upper(),
            stop_loss=stop_loss,
            strategy=strategy,
            trailing_activation_pct=trailing_activation_pct,
            trailing_distance_pct=trailing_distance_pct,
            max_scale_ins=max_scale_ins,
        )

        if targets:
            for i, t in enumerate(targets):
                bracket.targets.append(ExitTarget(
                    label=f"TP{i + 1}",
                    price=Decimal(str(t["price"])),
                    quantity_pct=float(t["pct"]),
                ))

        self._brackets[bracket.id] = bracket
        self._symbol_brackets.setdefault(symbol, []).append(bracket.id)
        logger.info(
            "Bracket created | {} {} {} qty={} entry={} SL={} targets={}",
            bracket.id[:8], side, symbol, quantity, entry_price, stop_loss,
            len(bracket.targets),
        )
        return bracket

    def create_default_bracket(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        entry_price: Decimal,
        stop_loss_pct: float = 0.02,
        tp1_pct: float = 0.015,
        tp2_pct: float = 0.03,
        tp1_size: float = 0.5,
        strategy: str = "",
    ) -> BracketOrder:
        """Create a bracket with standard TP1/TP2 + SL percent offsets.

        Default: SL at 2% below entry, TP1 at 1.5% (50% size), TP2 at 3% (50% size).
        """
        if side.upper() == "BUY":
            sl = entry_price * (1 - Decimal(str(stop_loss_pct)))
            tp1 = entry_price * (1 + Decimal(str(tp1_pct)))
            tp2 = entry_price * (1 + Decimal(str(tp2_pct)))
        else:
            sl = entry_price * (1 + Decimal(str(stop_loss_pct)))
            tp1 = entry_price * (1 - Decimal(str(tp1_pct)))
            tp2 = entry_price * (1 - Decimal(str(tp2_pct)))

        return self.create_bracket(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=sl,
            targets=[
                {"price": tp1, "pct": tp1_size},
                {"price": tp2, "pct": 1.0 - tp1_size},
            ],
            strategy=strategy,
        )

    # ── Entry Placement ───────────────────────────────────────────────────────

    async def place_entry(self, bracket_id: str) -> dict | None:
        """Place the entry order for a bracket."""
        bracket = self._brackets.get(bracket_id)
        if not bracket:
            logger.error("Bracket {} not found", bracket_id)
            return None

        if bracket.state != BracketState.PENDING:
            logger.warning("Bracket {} in state {}, cannot place entry", bracket_id[:8], bracket.state)
            return None

        try:
            if bracket.entry_type == "MARKET":
                resp = await self.exchange.place_market_order(
                    bracket.symbol, bracket.side, bracket.quantity,
                )
            else:
                resp = await self.exchange.place_limit_order(
                    bracket.symbol, bracket.side, bracket.entry_price, bracket.quantity,
                )
            bracket.entry_order_id = resp.get("orderId", "")
            bracket.state = BracketState.ENTRY_PLACED
            logger.info("Bracket {} entry placed: {}", bracket_id[:8], resp)
            return resp
        except Exception as exc:
            logger.error("Bracket {} entry failed: {}", bracket_id[:8], exc)
            bracket.state = BracketState.CANCELLED
            return None

    # ── Fill Handling ─────────────────────────────────────────────────────────

    async def on_entry_fill(self, bracket_id: str, filled_price: Decimal,
                            filled_qty: Decimal | None = None) -> None:
        """Called when the entry order fills. Activates exit management."""
        bracket = self._brackets.get(bracket_id)
        if not bracket:
            return
        bracket.filled_price = filled_price
        bracket.filled_qty = filled_qty or bracket.quantity
        bracket.state = BracketState.ACTIVE
        bracket.highest_price = filled_price

        # Track slippage
        bracket.realized_slippage = abs(filled_price - bracket.entry_price)
        self._record_fill(bracket, "ENTRY", filled_price, bracket.filled_qty)

        logger.info(
            "Bracket {} entry filled @ {} (expected {}, slip={})",
            bracket_id[:8], filled_price, bracket.entry_price, bracket.realized_slippage,
        )

    async def on_target_fill(self, bracket_id: str, target_label: str,
                             filled_price: Decimal) -> None:
        """Called when a take-profit target fills."""
        bracket = self._brackets.get(bracket_id)
        if not bracket:
            return

        for target in bracket.targets:
            if target.label == target_label and not target.filled:
                target.filled = True
                target.filled_price = filled_price
                target.filled_at = datetime.now(timezone.utc)
                exit_qty = bracket.filled_qty * Decimal(str(target.quantity_pct))
                self._record_fill(bracket, target_label, filled_price, exit_qty)
                logger.info(
                    "Bracket {} {} filled @ {} (qty={})",
                    bracket_id[:8], target_label, filled_price, exit_qty,
                )
                break

        # Check if all targets filled
        unfilled = [t for t in bracket.targets if not t.filled]
        if not unfilled:
            bracket.state = BracketState.CLOSED
            bracket.closed_at = datetime.now(timezone.utc)
        elif len(unfilled) < len(bracket.targets):
            bracket.state = BracketState.PARTIAL_TP

    async def on_stop_hit(self, bracket_id: str, filled_price: Decimal) -> None:
        """Called when stop-loss fires. Closes the bracket."""
        bracket = self._brackets.get(bracket_id)
        if not bracket:
            return

        remaining = bracket.filled_qty
        for t in bracket.targets:
            if t.filled:
                remaining -= bracket.filled_qty * Decimal(str(t.quantity_pct))

        self._record_fill(bracket, "SL", filled_price, remaining)
        bracket.state = BracketState.CLOSED
        bracket.closed_at = datetime.now(timezone.utc)
        logger.warning(
            "Bracket {} STOP HIT @ {} (remaining qty={})",
            bracket_id[:8], filled_price, remaining,
        )

    # ── Price Update / Trailing ───────────────────────────────────────────────

    def update_price(self, symbol: str, current_price: Decimal) -> list[dict]:
        """Update price for all active brackets on this symbol.

        Returns list of actions to be executed (e.g. sell signals).
        """
        actions = []
        bracket_ids = self._symbol_brackets.get(symbol, [])
        for bid in bracket_ids:
            bracket = self._brackets.get(bid)
            if not bracket or bracket.state not in (BracketState.ACTIVE, BracketState.PARTIAL_TP):
                continue

            # ── Check take-profit targets ──
            for target in bracket.targets:
                if target.filled:
                    continue
                if bracket.side == "BUY" and current_price >= target.price:
                    qty = bracket.filled_qty * Decimal(str(target.quantity_pct))
                    actions.append({
                        "type": "TAKE_PROFIT",
                        "bracket_id": bid,
                        "target": target.label,
                        "symbol": symbol,
                        "side": "SELL",
                        "quantity": qty,
                        "price": current_price,
                    })
                elif bracket.side == "SELL" and current_price <= target.price:
                    qty = bracket.filled_qty * Decimal(str(target.quantity_pct))
                    actions.append({
                        "type": "TAKE_PROFIT",
                        "bracket_id": bid,
                        "target": target.label,
                        "symbol": symbol,
                        "side": "BUY",
                        "quantity": qty,
                        "price": current_price,
                    })

            # ── Check stop-loss ──
            remaining = self._remaining_qty(bracket)
            if remaining <= 0:
                continue

            if bracket.side == "BUY" and current_price <= bracket.stop_loss:
                actions.append({
                    "type": "STOP_LOSS",
                    "bracket_id": bid,
                    "symbol": symbol,
                    "side": "SELL",
                    "quantity": remaining,
                    "price": current_price,
                })
            elif bracket.side == "SELL" and current_price >= bracket.stop_loss:
                actions.append({
                    "type": "STOP_LOSS",
                    "bracket_id": bid,
                    "symbol": symbol,
                    "side": "BUY",
                    "quantity": remaining,
                    "price": current_price,
                })

            # ── Update trailing stop ──
            if bracket.trailing_active or self._should_activate_trailing(bracket, current_price):
                bracket.trailing_active = True
                if bracket.side == "BUY":
                    if current_price > bracket.highest_price:
                        bracket.highest_price = current_price
                        bracket.trailing_stop_price = current_price * (
                            1 - Decimal(str(bracket.trailing_distance_pct))
                        )
                    if current_price <= bracket.trailing_stop_price and bracket.trailing_stop_price > bracket.stop_loss:
                        actions.append({
                            "type": "TRAILING_STOP",
                            "bracket_id": bid,
                            "symbol": symbol,
                            "side": "SELL",
                            "quantity": remaining,
                            "price": current_price,
                        })
                else:
                    if current_price < bracket.highest_price:
                        bracket.highest_price = current_price
                        bracket.trailing_stop_price = current_price * (
                            1 + Decimal(str(bracket.trailing_distance_pct))
                        )
                    if current_price >= bracket.trailing_stop_price:
                        actions.append({
                            "type": "TRAILING_STOP",
                            "bracket_id": bid,
                            "symbol": symbol,
                            "side": "BUY",
                            "quantity": remaining,
                            "price": current_price,
                        })

        return actions

    # ── Scale-in (Pyramid) ────────────────────────────────────────────────────

    async def scale_in(self, bracket_id: str, additional_qty: Decimal,
                       current_price: Decimal) -> dict | None:
        """Add to a winning position (pyramiding).

        Only allowed when position is in profit and max_scale_ins not reached.
        Tightens stop-loss to breakeven on previous tranche.
        """
        bracket = self._brackets.get(bracket_id)
        if not bracket or bracket.state != BracketState.ACTIVE:
            return None

        if len(bracket.scale_ins) >= bracket.max_scale_ins:
            logger.info("Bracket {} max scale-ins ({}) reached", bracket_id[:8], bracket.max_scale_ins)
            return None

        # Must be in profit to scale in
        if bracket.side == "BUY" and current_price <= bracket.filled_price:
            logger.debug("Bracket {} not in profit, skip scale-in", bracket_id[:8])
            return None
        if bracket.side == "SELL" and current_price >= bracket.filled_price:
            return None

        try:
            resp = await self.exchange.place_market_order(
                bracket.symbol, bracket.side, additional_qty,
            )
            bracket.scale_ins.append({
                "price": str(current_price),
                "quantity": str(additional_qty),
                "order_id": resp.get("orderId", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # Update filled qty and weighted average
            old_total = bracket.filled_price * bracket.filled_qty
            new_total = old_total + current_price * additional_qty
            bracket.filled_qty += additional_qty
            bracket.filled_price = new_total / bracket.filled_qty

            # Tighten stop to breakeven of previous tranche
            bracket.stop_loss = bracket.filled_price

            logger.info(
                "Bracket {} scale-in #{} | +{} @ {} | new avg={} | SL→{}",
                bracket_id[:8], len(bracket.scale_ins),
                additional_qty, current_price,
                bracket.filled_price, bracket.stop_loss,
            )
            return resp
        except Exception as exc:
            logger.error("Scale-in failed for bracket {}: {}", bracket_id[:8], exc)
            return None

    # ── Cancel / Close ────────────────────────────────────────────────────────

    async def cancel_bracket(self, bracket_id: str) -> None:
        """Cancel a bracket and all its child orders."""
        bracket = self._brackets.get(bracket_id)
        if not bracket:
            return

        if bracket.entry_order_id and bracket.state == BracketState.ENTRY_PLACED:
            try:
                await self.exchange.cancel_order(bracket.symbol, bracket.entry_order_id)
            except Exception:
                pass

        if bracket.stop_order_id:
            try:
                await self.exchange.cancel_order(bracket.symbol, bracket.stop_order_id)
            except Exception:
                pass

        for target in bracket.targets:
            if target.order_id and not target.filled:
                try:
                    await self.exchange.cancel_order(bracket.symbol, target.order_id)
                except Exception:
                    pass

        bracket.state = BracketState.CANCELLED
        bracket.closed_at = datetime.now(timezone.utc)
        logger.info("Bracket {} cancelled", bracket_id[:8])

    async def flatten_symbol(self, symbol: str) -> list[dict]:
        """Close all active brackets for a symbol (EOD flatten)."""
        results = []
        for bid in self._symbol_brackets.get(symbol, []):
            bracket = self._brackets.get(bid)
            if bracket and bracket.state in (BracketState.ACTIVE, BracketState.PARTIAL_TP):
                remaining = self._remaining_qty(bracket)
                if remaining > 0:
                    close_side = "SELL" if bracket.side == "BUY" else "BUY"
                    try:
                        resp = await self.exchange.place_market_order(
                            symbol, close_side, remaining,
                        )
                        bracket.state = BracketState.CLOSED
                        bracket.closed_at = datetime.now(timezone.utc)
                        results.append(resp)
                    except Exception as exc:
                        logger.error("Flatten bracket {} failed: {}", bid[:8], exc)
                else:
                    bracket.state = BracketState.CLOSED
                    bracket.closed_at = datetime.now(timezone.utc)
        return results

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_active_brackets(self, symbol: str | None = None) -> list[BracketOrder]:
        """Return all active brackets, optionally filtered by symbol."""
        active = [
            b for b in self._brackets.values()
            if b.state in (BracketState.ACTIVE, BracketState.PARTIAL_TP, BracketState.ENTRY_PLACED)
        ]
        if symbol:
            active = [b for b in active if b.symbol == symbol]
        return active

    def get_bracket(self, bracket_id: str) -> BracketOrder | None:
        return self._brackets.get(bracket_id)

    def get_execution_stats(self) -> dict:
        """Return execution quality statistics."""
        if not self._fills:
            return {"total_fills": 0}

        slippages = [f["slippage"] for f in self._fills if f.get("slippage") is not None]
        wins = [b for b in self._brackets.values() if b.state == BracketState.CLOSED]
        pnl_list = [self._calc_bracket_pnl(b) for b in wins]

        return {
            "total_fills": len(self._fills),
            "total_brackets": len(self._brackets),
            "active_brackets": len(self.get_active_brackets()),
            "avg_slippage": float(sum(slippages) / len(slippages)) if slippages else 0,
            "max_slippage": float(max(slippages)) if slippages else 0,
            "total_pnl": float(sum(pnl_list)),
            "win_rate": sum(1 for p in pnl_list if p > 0) / len(pnl_list) if pnl_list else 0,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _remaining_qty(self, bracket: BracketOrder) -> Decimal:
        remaining = bracket.filled_qty
        for t in bracket.targets:
            if t.filled:
                remaining -= bracket.filled_qty * Decimal(str(t.quantity_pct))
        return max(Decimal("0"), remaining)

    def _should_activate_trailing(self, bracket: BracketOrder, price: Decimal) -> bool:
        if not bracket.filled_price:
            return False
        if bracket.side == "BUY":
            gain = (price - bracket.filled_price) / bracket.filled_price
        else:
            gain = (bracket.filled_price - price) / bracket.filled_price
        return float(gain) >= bracket.trailing_activation_pct

    def _record_fill(self, bracket: BracketOrder, label: str,
                     price: Decimal, qty: Decimal) -> None:
        slip = float(abs(price - bracket.entry_price)) if label == "ENTRY" else None
        self._fills.append({
            "bracket_id": bracket.id,
            "symbol": bracket.symbol,
            "label": label,
            "price": float(price),
            "quantity": float(qty),
            "slippage": slip,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def _calc_bracket_pnl(self, bracket: BracketOrder) -> float:
        if not bracket.filled_price:
            return 0.0
        total_pnl = 0.0
        for fill in self._fills:
            if fill["bracket_id"] == bracket.id and fill["label"] != "ENTRY":
                exit_price = Decimal(str(fill["price"]))
                exit_qty = Decimal(str(fill["quantity"]))
                if bracket.side == "BUY":
                    total_pnl += float((exit_price - bracket.filled_price) * exit_qty)
                else:
                    total_pnl += float((bracket.filled_price - exit_price) * exit_qty)
        return total_pnl

    def reset_daily(self) -> None:
        """Clear closed brackets (keep active ones). Call at start of day."""
        closed_ids = [
            bid for bid, b in self._brackets.items()
            if b.state in (BracketState.CLOSED, BracketState.CANCELLED)
        ]
        for bid in closed_ids:
            bracket = self._brackets.pop(bid)
            for sym, ids in self._symbol_brackets.items():
                if bid in ids:
                    ids.remove(bid)
        logger.info("Daily reset: cleared {} closed brackets", len(closed_ids))
