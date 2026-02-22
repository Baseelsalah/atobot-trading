"""Seed historical data into SQLite for testing/development.

Usage:
    python scripts/seed_data.py --symbol BTCUSDT --days 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def seed_trades(db_path: str, symbol: str, count: int = 50) -> None:
    """Insert dummy trade records for dashboard testing."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price TEXT NOT NULL,
            quantity TEXT NOT NULL,
            fee TEXT NOT NULL DEFAULT '0',
            fee_asset TEXT NOT NULL DEFAULT 'USDT',
            pnl TEXT,
            strategy TEXT NOT NULL,
            order_id TEXT,
            executed_at DATETIME NOT NULL
        )"""
    )

    import random

    base_price = 42000.0 if "BTC" in symbol else 2200.0
    now = datetime.now(timezone.utc)

    for i in range(count):
        side = random.choice(["BUY", "SELL"])
        price = base_price + random.uniform(-500, 500)
        qty = round(random.uniform(0.001, 0.01), 6)
        fee = round(price * qty * 0.001, 4)
        pnl = round(random.uniform(-5, 10), 2) if side == "SELL" else None
        ts = now - timedelta(hours=count - i)

        conn.execute(
            "INSERT OR IGNORE INTO trades "
            "(trade_id, symbol, side, price, quantity, fee, fee_asset, pnl, strategy, executed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid4()),
                symbol,
                side,
                str(price),
                str(qty),
                str(fee),
                "USDT",
                str(pnl) if pnl is not None else None,
                "grid",
                ts.isoformat(),
            ),
        )

    conn.commit()
    print(f"Seeded {count} trades for {symbol}")

    # Seed daily stats
    conn.execute(
        """CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            pnl TEXT NOT NULL DEFAULT '0',
            trades INTEGER NOT NULL DEFAULT 0,
            wins INTEGER NOT NULL DEFAULT 0,
            losses INTEGER NOT NULL DEFAULT 0
        )"""
    )
    for d in range(30):
        date_str = (now - timedelta(days=30 - d)).strftime("%Y-%m-%d")
        daily_pnl = round(random.uniform(-20, 40), 2)
        trades_count = random.randint(2, 15)
        wins = random.randint(0, trades_count)
        losses = trades_count - wins
        conn.execute(
            "INSERT OR IGNORE INTO daily_stats (date, pnl, trades, wins, losses) "
            "VALUES (?, ?, ?, ?, ?)",
            (date_str, str(daily_pnl), trades_count, wins, losses),
        )

    conn.commit()
    conn.close()
    print(f"Seeded 30 days of daily stats")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Seed test data for AtoBot")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--db", default="data/atobot.db")
    args = parser.parse_args()

    # Ensure data directory exists
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    seed_trades(args.db, args.symbol, args.count)
    print("Done!")


if __name__ == "__main__":
    main()
