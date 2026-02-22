"""Streamlit dashboard for AtoBot Trading."""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(page_title="AtoBot Dashboard", page_icon="🤖", layout="wide")

# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_bot_state() -> dict:
    """Try to read bot state from the SQLite database synchronously."""
    try:
        import sqlite3

        db_path = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///data/atobot.db")
        # Extract file path from SQLAlchemy URL
        db_file = db_path.split("///")[-1] if "///" in db_path else "data/atobot.db"
        if not os.path.exists(db_file):
            return {}
        conn = sqlite3.connect(db_file)
        cursor = conn.execute("SELECT key, value FROM bot_state")
        state = {}
        for key, value in cursor.fetchall():
            try:
                state[key] = json.loads(value)
            except json.JSONDecodeError:
                state[key] = value
        conn.close()
        return state
    except Exception:
        return {}


def _load_trades() -> list[dict]:
    """Load recent trades from the database."""
    try:
        import sqlite3

        db_path = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///data/atobot.db")
        db_file = db_path.split("///")[-1] if "///" in db_path else "data/atobot.db"
        if not os.path.exists(db_file):
            return []
        conn = sqlite3.connect(db_file)
        cursor = conn.execute(
            "SELECT trade_id, symbol, side, price, quantity, pnl, strategy, executed_at "
            "FROM trades ORDER BY executed_at DESC LIMIT 100"
        )
        columns = [desc[0] for desc in cursor.description]
        trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return trades
    except Exception:
        return []


def _load_daily_stats() -> list[dict]:
    """Load daily statistics."""
    try:
        import sqlite3

        db_path = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///data/atobot.db")
        db_file = db_path.split("///")[-1] if "///" in db_path else "data/atobot.db"
        if not os.path.exists(db_file):
            return []
        conn = sqlite3.connect(db_file)
        cursor = conn.execute(
            "SELECT date, pnl, trades, wins, losses FROM daily_stats ORDER BY date DESC LIMIT 30"
        )
        columns = [desc[0] for desc in cursor.description]
        stats = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return stats
    except Exception:
        return []


# ── Dashboard Layout ──────────────────────────────────────────────────────────

st.title("🤖 AtoBot Trading Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    st.sidebar.info("Dashboard refreshes every 30 seconds")

# Main content
col1, col2, col3, col4 = st.columns(4)

bot_state = _load_bot_state()
strategy_status = bot_state.get("strategy_status", {})

with col1:
    st.metric("Strategy", strategy_status.get("strategy", "N/A"))

with col2:
    st.metric("Active Orders", strategy_status.get("active_orders", 0))

with col3:
    st.metric("Unrealised PnL", strategy_status.get("unrealized_pnl", "0"))

with col4:
    st.metric("Realised PnL", strategy_status.get("realized_pnl", "0"))

st.markdown("---")

# Recent Trades
st.subheader("📈 Recent Trades")
trades = _load_trades()
if trades:
    import pandas as pd

    df = pd.DataFrame(trades)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No trades recorded yet.")

# Daily Stats
st.subheader("📊 Daily Statistics")
daily_stats = _load_daily_stats()
if daily_stats:
    import pandas as pd

    df_stats = pd.DataFrame(daily_stats)
    st.dataframe(df_stats, use_container_width=True)

    # PnL chart
    if "pnl" in df_stats.columns and "date" in df_stats.columns:
        df_stats["pnl_float"] = df_stats["pnl"].astype(float)
        st.line_chart(df_stats.set_index("date")["pnl_float"])
else:
    st.info("No daily statistics yet.")

# Bot State
with st.expander("🔧 Bot State (raw)"):
    st.json(bot_state)

# Auto-refresh
if auto_refresh:
    import time

    time.sleep(30)
    st.rerun()
