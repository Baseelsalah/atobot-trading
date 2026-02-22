"""SQLAlchemy async database setup for AtoBot Trading."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from loguru import logger


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


# ── Table definitions ─────────────────────────────────────────────────────────


class OrderRow(Base):
    """Persisted order record."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    internal_id = Column(String(64), unique=True, nullable=False, index=True)
    exchange_id = Column(String(64), nullable=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    order_type = Column(String(10), nullable=False)
    price = Column(String(40), nullable=False)  # Store as string for Decimal precision
    quantity = Column(String(40), nullable=False)
    filled_quantity = Column(String(40), nullable=False, default="0")
    status = Column(String(20), nullable=False, default="PENDING")
    strategy = Column(String(20), nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, nullable=True)
    exchange_response = Column(Text, nullable=True)  # JSON string


class TradeRow(Base):
    """Persisted trade record."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(64), unique=True, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    price = Column(String(40), nullable=False)
    quantity = Column(String(40), nullable=False)
    fee = Column(String(40), nullable=False, default="0")
    fee_asset = Column(String(10), nullable=False, default="USD")
    pnl = Column(String(40), nullable=True)
    strategy = Column(String(20), nullable=False)
    order_id = Column(String(64), nullable=True)
    executed_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class PositionRow(Base):
    """Persisted position snapshot."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)
    entry_price = Column(String(40), nullable=False)
    current_price = Column(String(40), nullable=False)
    quantity = Column(String(40), nullable=False)
    unrealized_pnl = Column(String(40), nullable=False, default="0")
    realized_pnl = Column(String(40), nullable=False, default="0")
    strategy = Column(String(20), nullable=False)
    opened_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=True)


class DailyStatRow(Base):
    """Daily P&L stats."""

    __tablename__ = "daily_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, unique=True, index=True)  # YYYY-MM-DD
    pnl = Column(String(40), nullable=False, default="0")
    trades = Column(Integer, nullable=False, default=0)
    wins = Column(Integer, nullable=False, default=0)
    losses = Column(Integer, nullable=False, default=0)


class BotStateRow(Base):
    """Key-value store for bot state persistence."""

    __tablename__ = "bot_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), nullable=False, unique=True, index=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


# ── Engine & Session factory ──────────────────────────────────────────────────

_engine = None
_session_factory = None


async def init_database(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Create the async engine, create all tables, and return a session factory.

    Args:
        database_url: SQLAlchemy-style connection string (must use async driver).

    Returns:
        An ``async_sessionmaker`` ready for use.
    """
    global _engine, _session_factory

    _engine = create_async_engine(database_url, echo=False)
    _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialised at {}", database_url)
    return _session_factory


async def close_database() -> None:
    """Dispose of the engine and release connections."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connection closed")


def get_session_factory() -> async_sessionmaker[AsyncSession] | None:
    """Return the current session factory (or None if not initialised)."""
    return _session_factory
