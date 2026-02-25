"""Tests for Smart Order Manager (src/orders/__init__.py)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from src.orders import SmartOrderManager, BracketOrder, BracketState, ExitTarget


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def exchange():
    """Minimal async mock exchange client."""
    client = AsyncMock()
    client.place_market_order.return_value = {"orderId": "mkt-001", "status": "FILLED"}
    client.place_limit_order.return_value = {"orderId": "lmt-001", "status": "NEW"}
    client.cancel_order.return_value = {"orderId": "x", "status": "CANCELED"}
    return client


@pytest.fixture
def mgr(exchange):
    return SmartOrderManager(exchange)


# ═══════════════════════════════════════════════════════════════════════════════
# BRACKET CREATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateBracket:
    def test_basic_create(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
            targets=[{"price": Decimal("153"), "pct": 0.5},
                     {"price": Decimal("156"), "pct": 0.5}],
        )
        assert b.state == BracketState.PENDING
        assert b.symbol == "AAPL"
        assert b.quantity == Decimal("10")
        assert b.stop_loss == Decimal("147")
        assert len(b.targets) == 2
        assert b.targets[0].label == "TP1"
        assert b.targets[1].label == "TP2"

    def test_create_without_targets(self, mgr):
        b = mgr.create_bracket(
            symbol="TSLA", side="BUY", quantity=Decimal("5"),
            entry_price=Decimal("200"), stop_loss=Decimal("195"),
        )
        assert len(b.targets) == 0

    def test_create_default_bracket_buy(self, mgr):
        b = mgr.create_default_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("100"),
        )
        assert b.stop_loss < Decimal("100")  # SL below entry for BUY
        assert len(b.targets) == 2
        assert b.targets[0].price > Decimal("100")  # TP1 above entry
        assert b.targets[1].price > b.targets[0].price  # TP2 > TP1

    def test_create_default_bracket_sell(self, mgr):
        b = mgr.create_default_bracket(
            symbol="AAPL", side="SELL", quantity=Decimal("10"),
            entry_price=Decimal("100"),
        )
        assert b.stop_loss > Decimal("100")  # SL above entry for SELL
        assert b.targets[0].price < Decimal("100")  # TP below entry

    def test_bracket_stored(self, mgr):
        b = mgr.create_bracket(
            symbol="NVDA", side="BUY", quantity=Decimal("5"),
            entry_price=Decimal("500"), stop_loss=Decimal("490"),
        )
        assert mgr.get_bracket(b.id) is b
        assert b.id in mgr._symbol_brackets.get("NVDA", [])


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY PLACEMENT
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlaceEntry:
    @pytest.mark.asyncio
    async def test_place_market_entry(self, mgr, exchange):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
            entry_type="MARKET",
        )
        resp = await mgr.place_entry(b.id)
        assert resp is not None
        assert b.state == BracketState.ENTRY_PLACED
        exchange.place_market_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_place_limit_entry(self, mgr, exchange):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
            entry_type="LIMIT",
        )
        resp = await mgr.place_entry(b.id)
        assert resp is not None
        exchange.place_limit_order.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_place_entry_invalid_id(self, mgr):
        resp = await mgr.place_entry("nonexistent")
        assert resp is None

    @pytest.mark.asyncio
    async def test_place_entry_wrong_state(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
        )
        b.state = BracketState.ACTIVE
        resp = await mgr.place_entry(b.id)
        assert resp is None

    @pytest.mark.asyncio
    async def test_place_entry_exchange_error(self, mgr, exchange):
        exchange.place_market_order.side_effect = Exception("API error")
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
        )
        resp = await mgr.place_entry(b.id)
        assert resp is None
        assert b.state == BracketState.CANCELLED


# ═══════════════════════════════════════════════════════════════════════════════
# FILL HANDLING
# ═══════════════════════════════════════════════════════════════════════════════


class TestFillHandling:
    @pytest.mark.asyncio
    async def test_on_entry_fill(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
        )
        await mgr.on_entry_fill(b.id, Decimal("150.05"), Decimal("10"))
        assert b.state == BracketState.ACTIVE
        assert b.filled_price == Decimal("150.05")
        assert b.realized_slippage == Decimal("0.05")

    @pytest.mark.asyncio
    async def test_on_target_fill_partial(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
            targets=[{"price": Decimal("153"), "pct": 0.5},
                     {"price": Decimal("156"), "pct": 0.5}],
        )
        await mgr.on_entry_fill(b.id, Decimal("150"), Decimal("10"))
        await mgr.on_target_fill(b.id, "TP1", Decimal("153"))
        assert b.targets[0].filled is True
        assert b.state == BracketState.PARTIAL_TP

    @pytest.mark.asyncio
    async def test_on_all_targets_filled(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
            targets=[{"price": Decimal("153"), "pct": 0.5},
                     {"price": Decimal("156"), "pct": 0.5}],
        )
        await mgr.on_entry_fill(b.id, Decimal("150"), Decimal("10"))
        await mgr.on_target_fill(b.id, "TP1", Decimal("153"))
        await mgr.on_target_fill(b.id, "TP2", Decimal("156"))
        assert b.state == BracketState.CLOSED

    @pytest.mark.asyncio
    async def test_on_stop_hit(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
        )
        await mgr.on_entry_fill(b.id, Decimal("150"), Decimal("10"))
        await mgr.on_stop_hit(b.id, Decimal("147"))
        assert b.state == BracketState.CLOSED


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE UPDATE / TRAILING
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpdatePrice:
    @pytest.mark.asyncio
    async def test_take_profit_triggered(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
            targets=[{"price": Decimal("153"), "pct": 1.0}],
        )
        await mgr.on_entry_fill(b.id, Decimal("150"), Decimal("10"))
        actions = mgr.update_price("AAPL", Decimal("154"))
        assert len(actions) >= 1
        tp_actions = [a for a in actions if a["type"] == "TAKE_PROFIT"]
        assert len(tp_actions) == 1
        assert tp_actions[0]["side"] == "SELL"

    @pytest.mark.asyncio
    async def test_stop_loss_triggered(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
        )
        await mgr.on_entry_fill(b.id, Decimal("150"), Decimal("10"))
        actions = mgr.update_price("AAPL", Decimal("146"))
        sl_actions = [a for a in actions if a["type"] == "STOP_LOSS"]
        assert len(sl_actions) == 1

    @pytest.mark.asyncio
    async def test_trailing_stop_activation(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("100"), stop_loss=Decimal("97"),
            trailing_activation_pct=0.02,  # 2% gain to activate
            trailing_distance_pct=0.01,    # 1% trailing distance
        )
        await mgr.on_entry_fill(b.id, Decimal("100"), Decimal("10"))

        # Push price up 3% — should activate trailing
        mgr.update_price("AAPL", Decimal("103"))
        assert b.trailing_active is True
        assert b.highest_price == Decimal("103")
        assert b.trailing_stop_price > Decimal("0")

    @pytest.mark.asyncio
    async def test_trailing_stop_fires(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("100"), stop_loss=Decimal("95"),
            trailing_activation_pct=0.02,
            trailing_distance_pct=0.01,
        )
        await mgr.on_entry_fill(b.id, Decimal("100"), Decimal("10"))

        # Push up to activate trailing
        mgr.update_price("AAPL", Decimal("103"))

        # Now drop below trailing stop
        trailing_price = b.trailing_stop_price
        actions = mgr.update_price("AAPL", trailing_price - Decimal("0.01"))
        trail_actions = [a for a in actions if a["type"] == "TRAILING_STOP"]
        assert len(trail_actions) == 1

    @pytest.mark.asyncio
    async def test_no_actions_inactive_bracket(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
        )
        # Bracket still PENDING — no actions
        actions = mgr.update_price("AAPL", Decimal("160"))
        assert len(actions) == 0

    def test_no_actions_unknown_symbol(self, mgr):
        actions = mgr.update_price("UNKNOWN", Decimal("100"))
        assert len(actions) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# SCALE-IN
# ═══════════════════════════════════════════════════════════════════════════════


class TestScaleIn:
    @pytest.mark.asyncio
    async def test_scale_in_success(self, mgr, exchange):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("100"), stop_loss=Decimal("95"),
            max_scale_ins=2,
        )
        await mgr.on_entry_fill(b.id, Decimal("100"), Decimal("10"))
        resp = await mgr.scale_in(b.id, Decimal("5"), Decimal("105"))
        assert resp is not None
        assert len(b.scale_ins) == 1
        assert b.filled_qty == Decimal("15")
        # Stop should tighten to breakeven
        assert b.stop_loss == b.filled_price

    @pytest.mark.asyncio
    async def test_scale_in_not_in_profit(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("100"), stop_loss=Decimal("95"),
        )
        await mgr.on_entry_fill(b.id, Decimal("100"), Decimal("10"))
        resp = await mgr.scale_in(b.id, Decimal("5"), Decimal("99"))
        assert resp is None  # Can't scale in at a loss

    @pytest.mark.asyncio
    async def test_scale_in_max_reached(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("100"), stop_loss=Decimal("95"),
            max_scale_ins=1,
        )
        await mgr.on_entry_fill(b.id, Decimal("100"), Decimal("10"))
        await mgr.scale_in(b.id, Decimal("5"), Decimal("105"))
        resp = await mgr.scale_in(b.id, Decimal("5"), Decimal("110"))
        assert resp is None


# ═══════════════════════════════════════════════════════════════════════════════
# CANCEL / FLATTEN / QUERY
# ═══════════════════════════════════════════════════════════════════════════════


class TestCancelFlattenQuery:
    @pytest.mark.asyncio
    async def test_cancel_bracket(self, mgr, exchange):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
        )
        await mgr.place_entry(b.id)
        b.entry_order_id = "entry-001"
        await mgr.cancel_bracket(b.id)
        assert b.state == BracketState.CANCELLED

    @pytest.mark.asyncio
    async def test_flatten_symbol(self, mgr, exchange):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
        )
        await mgr.on_entry_fill(b.id, Decimal("150"), Decimal("10"))
        results = await mgr.flatten_symbol("AAPL")
        assert len(results) == 1
        assert b.state == BracketState.CLOSED

    def test_get_active_brackets(self, mgr):
        b1 = mgr.create_bracket(symbol="AAPL", side="BUY", quantity=Decimal("10"),
                                 entry_price=Decimal("150"), stop_loss=Decimal("147"))
        b2 = mgr.create_bracket(symbol="TSLA", side="BUY", quantity=Decimal("5"),
                                 entry_price=Decimal("200"), stop_loss=Decimal("195"))
        b1.state = BracketState.ACTIVE
        b2.state = BracketState.CLOSED
        active = mgr.get_active_brackets()
        assert len(active) == 1
        assert active[0].symbol == "AAPL"

    def test_get_active_brackets_by_symbol(self, mgr):
        b = mgr.create_bracket(symbol="AAPL", side="BUY", quantity=Decimal("10"),
                                entry_price=Decimal("150"), stop_loss=Decimal("147"))
        b.state = BracketState.ACTIVE
        assert len(mgr.get_active_brackets("AAPL")) == 1
        assert len(mgr.get_active_brackets("TSLA")) == 0

    def test_execution_stats_empty(self, mgr):
        stats = mgr.get_execution_stats()
        assert stats["total_fills"] == 0

    @pytest.mark.asyncio
    async def test_execution_stats_with_fills(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
            targets=[{"price": Decimal("153"), "pct": 1.0}],
        )
        await mgr.on_entry_fill(b.id, Decimal("150.05"), Decimal("10"))
        await mgr.on_target_fill(b.id, "TP1", Decimal("153"))
        stats = mgr.get_execution_stats()
        assert stats["total_fills"] >= 2
        assert stats["total_pnl"] > 0

    def test_reset_daily(self, mgr):
        b = mgr.create_bracket(symbol="AAPL", side="BUY", quantity=Decimal("10"),
                                entry_price=Decimal("150"), stop_loss=Decimal("147"))
        b.state = BracketState.CLOSED
        mgr.reset_daily()
        assert b.id not in mgr._brackets


# ═══════════════════════════════════════════════════════════════════════════════
# BRACKET ORDER TO_DICT
# ═══════════════════════════════════════════════════════════════════════════════


class TestBracketSerialization:
    def test_to_dict(self, mgr):
        b = mgr.create_bracket(
            symbol="AAPL", side="BUY", quantity=Decimal("10"),
            entry_price=Decimal("150"), stop_loss=Decimal("147"),
            targets=[{"price": Decimal("153"), "pct": 0.5}],
        )
        d = b.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["side"] == "BUY"
        assert d["state"] == "PENDING"
        assert len(d["targets"]) == 1
