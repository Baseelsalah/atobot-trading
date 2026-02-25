"""Streamlit dashboard for AtoBot Trading — Enhanced with VaR, Pairs, ML, Walk-Forward."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(page_title="AtoBot Dashboard", page_icon="🤖", layout="wide")

# ── Data Loaders ──────────────────────────────────────────────────────────────


def _db_path() -> str:
    db_url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///data/atobot.db")
    return db_url.split("///")[-1] if "///" in db_url else "data/atobot.db"


def _get_conn():
    import sqlite3
    db = _db_path()
    if not os.path.exists(db):
        return None
    return sqlite3.connect(db)


def _load_bot_state() -> dict:
    try:
        conn = _get_conn()
        if not conn:
            return {}
        cursor = conn.execute("SELECT key, value FROM bot_state")
        state = {}
        for key, value in cursor.fetchall():
            try:
                state[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                state[key] = value
        conn.close()
        return state
    except Exception:
        return {}


def _load_trades(limit: int = 200) -> list[dict]:
    try:
        conn = _get_conn()
        if not conn:
            return []
        cursor = conn.execute(
            "SELECT trade_id, symbol, side, price, quantity, pnl, strategy, executed_at "
            "FROM trades ORDER BY executed_at DESC LIMIT ?",
            (limit,),
        )
        columns = [desc[0] for desc in cursor.description]
        trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return trades
    except Exception:
        return []


def _load_daily_stats(limit: int = 60) -> list[dict]:
    try:
        conn = _get_conn()
        if not conn:
            return []
        cursor = conn.execute(
            "SELECT date, pnl, trades, wins, losses FROM daily_stats ORDER BY date DESC LIMIT ?",
            (limit,),
        )
        columns = [desc[0] for desc in cursor.description]
        stats = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return stats
    except Exception:
        return []


def _load_positions() -> list[dict]:
    try:
        conn = _get_conn()
        if not conn:
            return []
        cursor = conn.execute(
            "SELECT symbol, side, entry_price, current_price, quantity, "
            "unrealized_pnl, realized_pnl, strategy, opened_at FROM positions"
        )
        columns = [desc[0] for desc in cursor.description]
        positions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return positions
    except Exception:
        return []


def _load_walk_forward() -> list[dict]:
    """Load walk-forward optimization history from JSON file."""
    path = os.path.join("data", "walk_forward_history.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


# ── Dashboard ─────────────────────────────────────────────────────────────────

st.title("🤖 AtoBot Trading Dashboard")

# Sidebar ──────────────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
page = st.sidebar.radio(
    "View",
    ["Overview", "Trades", "Risk & VaR", "Pairs & ML", "Walk-Forward", "Raw State"],
)

st.sidebar.markdown("---")
st.sidebar.caption("AtoBot v2 — Long+Short · Pairs · ML")

# ── Load all data ─────────────────────────────────────────────────────────────
bot_state = _load_bot_state()
strategy_status = bot_state.get("strategy_status", {})
risk_state = bot_state.get("risk_state", {})
pairs_state = bot_state.get("pairs_state", [])
ml_state = bot_state.get("ml_state", {})
correlation_state = bot_state.get("correlation_penalties", {})

# ══════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("---")

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Strategy", strategy_status.get("strategy", "multi"))
    with c2:
        st.metric("Active Orders", strategy_status.get("active_orders", 0))
    with c3:
        rpnl = strategy_status.get("realized_pnl", "0")
        st.metric("Realised PnL", f"${float(rpnl):,.2f}" if rpnl else "$0.00")
    with c4:
        upnl = strategy_status.get("unrealized_pnl", "0")
        st.metric("Unrealised PnL", f"${float(upnl):,.2f}" if upnl else "$0.00")
    with c5:
        halted = risk_state.get("is_halted", False)
        st.metric("Status", "🔴 HALTED" if halted else "🟢 ACTIVE")

    st.markdown("---")

    # Open positions
    st.subheader("📌 Open Positions")
    positions = _load_positions()
    if positions:
        import pandas as pd
        df_pos = pd.DataFrame(positions)
        # Color side column
        st.dataframe(df_pos, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")

    # Recent trades preview
    st.subheader("📈 Recent Trades (last 20)")
    trades = _load_trades(20)
    if trades:
        import pandas as pd
        df_t = pd.DataFrame(trades)
        st.dataframe(df_t, use_container_width=True, hide_index=True)
    else:
        st.info("No trades recorded yet.")

    # Daily PnL chart
    st.subheader("📊 Daily P&L")
    daily_stats = _load_daily_stats()
    if daily_stats:
        import pandas as pd
        df_d = pd.DataFrame(daily_stats)
        if "pnl" in df_d.columns and "date" in df_d.columns:
            df_d["pnl"] = df_d["pnl"].astype(float)
            df_d = df_d.sort_values("date")
            df_d["cumulative_pnl"] = df_d["pnl"].cumsum()
            st.line_chart(df_d.set_index("date")[["pnl", "cumulative_pnl"]])
    else:
        st.info("No daily statistics yet. Start trading to see data.")

    # Quick risk summary
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.subheader("⚠️ Risk Snapshot")
        var_data = risk_state.get("var", {})
        if var_data:
            v_pct = var_data.get("var_pct")
            es_pct = var_data.get("es_pct")
            st.write(f"**VaR (95%):** {v_pct*100:.2f}%" if v_pct else "**VaR:** Calibrating...")
            st.write(f"**CVaR/ES:** {es_pct*100:.2f}%" if es_pct else "**CVaR:** Calibrating...")
            st.write(f"Data days: {var_data.get('data_days', 0)}")
        else:
            st.write("VaR not yet computed — needs 30+ trading days.")

    with col_r2:
        st.subheader("🔗 Correlation Penalties")
        if correlation_state:
            for sym, scale in sorted(correlation_state.items()):
                pct = (1 - scale) * 100
                bar = "🟥" * max(1, int(pct / 10)) if pct > 0 else "🟩"
                st.write(f"**{sym}:** {pct:.0f}% reduction {bar}")
        else:
            st.write("No correlation penalties active.")


# ══════════════════════════════════════════════════════════════════════════════
#  TRADES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Trades":
    st.markdown("---")
    st.subheader("📈 Trade History")

    trades = _load_trades(200)
    if trades:
        import pandas as pd
        df = pd.DataFrame(trades)

        # Filters
        fc1, fc2 = st.columns(2)
        with fc1:
            symbols = ["All"] + sorted(df["symbol"].unique().tolist())
            sym_filter = st.selectbox("Symbol", symbols)
        with fc2:
            sides = ["All", "BUY", "SELL", "SHORT", "COVER"]
            side_filter = st.selectbox("Side", sides)

        filtered = df.copy()
        if sym_filter != "All":
            filtered = filtered[filtered["symbol"] == sym_filter]
        if side_filter != "All":
            filtered = filtered[filtered["side"] == side_filter]

        # Summary
        sc1, sc2, sc3, sc4 = st.columns(4)
        pnl_vals = filtered["pnl"].astype(float)
        with sc1:
            st.metric("Trades", len(filtered))
        with sc2:
            st.metric("Net PnL", f"${pnl_vals.sum():,.2f}")
        with sc3:
            wins = (pnl_vals > 0).sum()
            wr = wins / len(pnl_vals) * 100 if len(pnl_vals) > 0 else 0
            st.metric("Win Rate", f"{wr:.1f}%")
        with sc4:
            longs = (filtered["side"].isin(["BUY"])).sum()
            shorts = (filtered["side"].isin(["SHORT"])).sum()
            st.metric("Long / Short", f"{longs} / {shorts}")

        st.dataframe(filtered, use_container_width=True, hide_index=True)

        # Strategy breakdown
        st.subheader("Strategy Breakdown")
        by_strat = df.groupby("strategy").agg(
            trades=("trade_id", "count"),
            net_pnl=("pnl", lambda x: x.astype(float).sum()),
        ).reset_index()
        st.dataframe(by_strat, use_container_width=True, hide_index=True)
    else:
        st.info("No trades recorded yet.")


# ══════════════════════════════════════════════════════════════════════════════
#  RISK & VAR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Risk & VaR":
    st.markdown("---")
    st.subheader("⚠️ Risk Manager")

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        halted = risk_state.get("is_halted", False)
        st.metric("Circuit Breaker", "🔴 TRIPPED" if halted else "🟢 OK")
        if halted:
            st.error(f"Reason: {risk_state.get('halt_reason', 'Unknown')}")
    with rc2:
        bal = risk_state.get("balance", 0)
        st.metric("Account Balance", f"${float(bal):,.2f}" if bal else "N/A")
    with rc3:
        daily = risk_state.get("daily_pnl", 0)
        st.metric("Today's PnL", f"${float(daily):,.2f}" if daily else "$0.00")

    st.markdown("---")
    st.subheader("📉 Value at Risk (VaR)")

    var_data = risk_state.get("var", {})
    if var_data and var_data.get("var_pct") is not None:
        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            st.metric("VaR (95%)", f"{var_data['var_pct']*100:.2f}%")
        with vc2:
            st.metric("VaR ($)", f"${var_data.get('var_usd', 0):,.2f}")
        with vc3:
            st.metric("CVaR/ES", f"{var_data.get('es_pct', 0)*100:.2f}%")
        with vc4:
            st.metric("ES ($)", f"${var_data.get('es_usd', 0):,.2f}")

        st.progress(
            min(var_data["var_pct"] * 100 / 5.0, 1.0),
            text=f"VaR {var_data['var_pct']*100:.2f}% vs 5% limit",
        )
    else:
        st.info("VaR not yet computed — requires 30+ trading days of portfolio returns.")

    st.markdown("---")
    st.subheader("🔗 Correlation Risk")
    if correlation_state:
        import pandas as pd
        corr_rows = [{"Symbol": sym, "Scale Factor": f"{scale:.2f}", "Size Reduction": f"{(1-scale)*100:.0f}%"}
                     for sym, scale in sorted(correlation_state.items())]
        st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)
    else:
        st.success("No correlation penalties currently active — all positions sized normally.")

    # Pre-trade check results
    checks = risk_state.get("last_checks", {})
    if checks:
        st.markdown("---")
        st.subheader("✅ Pre-Trade Checks")
        for check_name, passed in checks.items():
            icon = "✅" if passed else "❌"
            st.write(f"{icon} {check_name}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAIRS & ML
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Pairs & ML":
    st.markdown("---")

    # Pairs Trading
    st.subheader("🔄 Pairs Trading")
    if pairs_state:
        import pandas as pd
        df_pairs = pd.DataFrame(pairs_state)
        # Highlight active positions
        for _, row in df_pairs.iterrows():
            pair = row.get("pair", "?")
            z = row.get("zscore", 0)
            pos = row.get("position", "")
            held = row.get("bars_held", 0)
            hedge = row.get("hedge_ratio", 1.0)

            if pos:
                st.warning(f"**{pair}** — {pos.upper()} | z={z:.3f} | held {held} bars | β={hedge:.3f}")
            else:
                st.info(f"**{pair}** — Flat | z={z:.3f} | β={hedge:.3f}")
    else:
        from src.config.settings import get_settings
        try:
            s = get_settings()
            pairs_list = getattr(s, "PAIRS_SYMBOLS", [])
            if pairs_list:
                st.write(f"Monitoring pairs: {', '.join(pairs_list)}")
            else:
                st.write("Pairs: NVDA:AMD, GOOGL:META, MSFT:AAPL (default)")
        except Exception:
            st.write("Pairs: NVDA:AMD, GOOGL:META, MSFT:AAPL (default)")
        st.info("No pairs state data yet — bot needs to be running to populate.")

    st.markdown("---")

    # ML Model
    st.subheader("🧠 ML Model Status")
    if ml_state:
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            available = ml_state.get("is_available", False)
            st.metric("Model", "✅ Ready" if available else "⏳ Training")
        with mc2:
            metrics = ml_state.get("metrics", {})
            acc = metrics.get("accuracy")
            st.metric("Accuracy", f"{acc*100:.1f}%" if acc else "N/A")
        with mc3:
            auc = metrics.get("auc_roc")
            st.metric("AUC-ROC", f"{auc:.3f}" if auc else "N/A")
        with mc4:
            f1 = metrics.get("f1")
            st.metric("F1 Score", f"{f1:.3f}" if f1 else "N/A")

        trained_at = metrics.get("trained_at", "")
        if trained_at:
            st.caption(f"Last trained: {trained_at}")

        # Feature importance
        feat_imp = ml_state.get("feature_importance", {})
        if feat_imp:
            st.subheader("Feature Importance (Top 15)")
            import pandas as pd
            top_feats = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15])
            df_feat = pd.DataFrame({"Feature": top_feats.keys(), "Importance": top_feats.values()})
            st.bar_chart(df_feat.set_index("Feature"))
    else:
        st.info("ML model not yet initialized — needs sufficient trade history to train.")
        st.write("**Requirements:** 200+ trades for initial training, LightGBM installed.")


# ══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Walk-Forward":
    st.markdown("---")
    st.subheader("🔬 Walk-Forward Optimization History")

    wf_data = _load_walk_forward()
    if wf_data:
        import pandas as pd

        df_wf = pd.DataFrame(wf_data)

        # Summary
        st.write(f"**{len(df_wf)} optimization runs** recorded")

        # Filter by strategy
        strategies = ["All"] + sorted(df_wf["strategy"].unique().tolist())
        wf_strat = st.selectbox("Strategy", strategies)
        if wf_strat != "All":
            df_wf = df_wf[df_wf["strategy"] == wf_strat]

        # Sharpe comparison chart
        if "train_sharpe" in df_wf.columns and "test_sharpe" in df_wf.columns:
            st.subheader("Train vs Test Sharpe")
            chart_data = df_wf[["strategy", "train_sharpe", "test_sharpe", "timestamp"]].copy()
            chart_data = chart_data.sort_values("timestamp")
            st.line_chart(chart_data.set_index("timestamp")[["train_sharpe", "test_sharpe"]])

        # Best params per strategy
        st.subheader("Latest Best Parameters")
        for strat in df_wf["strategy"].unique():
            strat_rows = df_wf[df_wf["strategy"] == strat].sort_values("timestamp", ascending=False)
            if len(strat_rows) > 0:
                latest = strat_rows.iloc[0]
                with st.expander(f"📋 {strat} — Test Sharpe: {latest.get('test_sharpe', 'N/A')}"):
                    st.json(latest.get("best_params", {}))
                    st.write(f"Train PF: {latest.get('train_pf', 'N/A')} | Test PF: {latest.get('test_pf', 'N/A')}")
                    st.write(f"Train trades: {latest.get('train_trades', 0)} | Test trades: {latest.get('test_trades', 0)}")
                    st.write(f"Candidates tested: {latest.get('candidates_tested', 0)}")

        # Full table
        st.subheader("All Runs")
        st.dataframe(df_wf, use_container_width=True, hide_index=True)
    else:
        st.info("No walk-forward optimization data yet.")
        st.write("Walk-forward runs automatically when the guardian detects performance degradation.")
        st.write("History is saved to `data/walk_forward_history.json`.")


# ══════════════════════════════════════════════════════════════════════════════
#  RAW STATE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Raw State":
    st.markdown("---")
    st.subheader("🔧 Raw Bot State")
    st.json(bot_state)

    # Daily stats table
    st.subheader("📊 Daily Statistics")
    daily_stats = _load_daily_stats()
    if daily_stats:
        import pandas as pd
        df_stats = pd.DataFrame(daily_stats)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
    else:
        st.info("No daily statistics yet.")


# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()
