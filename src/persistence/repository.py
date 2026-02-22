"""CRUD repository for AtoBot Trading persistence."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.models.order import Order, OrderSide, OrderStatus, OrderType
from src.models.trade import Trade
from src.persistence.database import (
    BotStateRow,
    DailyStatRow,
    OrderRow,
    TradeRow,
)


class TradingRepository:
    """Async CRUD operations for trades, orders, positions, and bot state."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    # ── Orders ────────────────────────────────────────────────────────────────

    async def save_order(self, order: Order) -> None:
        """Insert a new order row."""
        async with self._session_factory() as session:
            row = OrderRow(
                internal_id=order.internal_id,
                exchange_id=order.id,
                symbol=order.symbol,
                side=order.side if isinstance(order.side, str) else order.side.value,
                order_type=order.order_type if isinstance(order.order_type, str) else order.order_type.value,
                price=str(order.price),
                quantity=str(order.quantity),
                filled_quantity=str(order.filled_quantity),
                status=order.status if isinstance(order.status, str) else order.status.value,
                strategy=order.strategy,
                created_at=order.created_at,
                updated_at=order.updated_at,
                exchange_response=(
                    json.dumps(order.exchange_response)
                    if order.exchange_response
                    else None
                ),
            )
            session.add(row)
            await session.commit()
            logger.debug("Saved order {}", order.internal_id)

    async def update_order(self, order: Order) -> None:
        """Update an existing order row by internal_id."""
        async with self._session_factory() as session:
            stmt = (
                update(OrderRow)
                .where(OrderRow.internal_id == order.internal_id)
                .values(
                    exchange_id=order.id,
                    filled_quantity=str(order.filled_quantity),
                    status=order.status if isinstance(order.status, str) else order.status.value,
                    updated_at=order.updated_at or datetime.now(timezone.utc),
                    exchange_response=(
                        json.dumps(order.exchange_response)
                        if order.exchange_response
                        else None
                    ),
                )
            )
            await session.execute(stmt)
            await session.commit()
            logger.debug("Updated order {}", order.internal_id)

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Return all orders with an active status."""
        async with self._session_factory() as session:
            stmt = select(OrderRow).where(
                OrderRow.status.in_(["PENDING", "OPEN", "PARTIALLY_FILLED"])
            )
            if symbol:
                stmt = stmt.where(OrderRow.symbol == symbol)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_order(r) for r in rows]

    # ── Trades ────────────────────────────────────────────────────────────────

    async def save_trade(self, trade: Trade) -> None:
        """Insert a new trade row."""
        async with self._session_factory() as session:
            row = TradeRow(
                trade_id=trade.id,
                symbol=trade.symbol,
                side=trade.side if isinstance(trade.side, str) else trade.side.value,
                price=str(trade.price),
                quantity=str(trade.quantity),
                fee=str(trade.fee),
                fee_asset=trade.fee_asset,
                pnl=str(trade.pnl) if trade.pnl is not None else None,
                strategy=trade.strategy,
                order_id=trade.order_id,
                executed_at=trade.executed_at,
            )
            session.add(row)
            await session.commit()
            logger.debug("Saved trade {}", trade.id)

    async def get_trades(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Trade]:
        """Return trades, optionally filtered by symbol and date range."""
        async with self._session_factory() as session:
            stmt = select(TradeRow)
            if symbol:
                stmt = stmt.where(TradeRow.symbol == symbol)
            if start:
                stmt = stmt.where(TradeRow.executed_at >= start)
            if end:
                stmt = stmt.where(TradeRow.executed_at <= end)
            stmt = stmt.order_by(TradeRow.executed_at.desc())
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._row_to_trade(r) for r in rows]

    async def get_daily_pnl(self, target_date: date) -> Decimal:
        """Sum PnL for all trades on a specific date."""
        async with self._session_factory() as session:
            stmt = select(DailyStatRow).where(
                DailyStatRow.date == target_date.isoformat()
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row:
                return Decimal(row.pnl)
            return Decimal("0")

    async def get_total_pnl(self) -> Decimal:
        """Sum PnL across all trades."""
        async with self._session_factory() as session:
            stmt = select(TradeRow).where(TradeRow.pnl.isnot(None))
            result = await session.execute(stmt)
            rows = result.scalars().all()
            total = Decimal("0")
            for r in rows:
                if r.pnl:
                    total += Decimal(r.pnl)
            return total

    async def update_daily_stats(
        self, target_date: date, pnl: Decimal, is_win: bool
    ) -> None:
        """Upsert daily statistics."""
        async with self._session_factory() as session:
            date_str = target_date.isoformat()
            stmt = select(DailyStatRow).where(DailyStatRow.date == date_str)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row:
                row.pnl = str(Decimal(row.pnl) + pnl)
                row.trades += 1
                if is_win:
                    row.wins += 1
                else:
                    row.losses += 1
            else:
                row = DailyStatRow(
                    date=date_str,
                    pnl=str(pnl),
                    trades=1,
                    wins=1 if is_win else 0,
                    losses=0 if is_win else 1,
                )
                session.add(row)
            await session.commit()

    # ── Bot State ─────────────────────────────────────────────────────────────

    async def save_bot_state(self, state: dict) -> None:
        """Persist bot state as key-value pairs."""
        async with self._session_factory() as session:
            for key, value in state.items():
                stmt = select(BotStateRow).where(BotStateRow.key == key)
                result = await session.execute(stmt)
                row = result.scalar_one_or_none()
                json_value = json.dumps(value, default=str)
                if row:
                    row.value = json_value
                    row.updated_at = datetime.now(timezone.utc)
                else:
                    row = BotStateRow(
                        key=key,
                        value=json_value,
                        updated_at=datetime.now(timezone.utc),
                    )
                    session.add(row)
            await session.commit()
            logger.debug("Bot state saved ({} keys)", len(state))

    async def load_bot_state(self) -> dict | None:
        """Load all bot state key-value pairs."""
        async with self._session_factory() as session:
            stmt = select(BotStateRow)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            if not rows:
                return None
            state: dict = {}
            for row in rows:
                try:
                    state[row.key] = json.loads(row.value)
                except json.JSONDecodeError:
                    state[row.key] = row.value
            return state

    # ── Row → Model converters ────────────────────────────────────────────────

    @staticmethod
    def _row_to_order(row: OrderRow) -> Order:
        """Convert an OrderRow to an Order model."""
        exchange_response = None
        if row.exchange_response:
            try:
                exchange_response = json.loads(row.exchange_response)
            except json.JSONDecodeError:
                exchange_response = None

        return Order(
            id=row.exchange_id,
            internal_id=row.internal_id,
            symbol=row.symbol,
            side=OrderSide(row.side),
            order_type=OrderType(row.order_type),
            price=Decimal(row.price),
            quantity=Decimal(row.quantity),
            filled_quantity=Decimal(row.filled_quantity),
            status=OrderStatus(row.status),
            strategy=row.strategy,
            created_at=row.created_at,
            updated_at=row.updated_at,
            exchange_response=exchange_response,
        )

    @staticmethod
    def _row_to_trade(row: TradeRow) -> Trade:
        """Convert a TradeRow to a Trade model."""
        return Trade(
            id=row.trade_id,
            symbol=row.symbol,
            side=OrderSide(row.side),
            price=Decimal(row.price),
            quantity=Decimal(row.quantity),
            fee=Decimal(row.fee),
            fee_asset=row.fee_asset,
            pnl=Decimal(row.pnl) if row.pnl else None,
            strategy=row.strategy,
            order_id=row.order_id,
            executed_at=row.executed_at,
        )
