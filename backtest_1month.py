"""AtoBot 1-Month Backtest — runs the VWAP+ORB combo (live config) over the last month."""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone
from backtest_v2 import (
    fetch_bars, bars_to_5min, group_by_day,
    run_vwap_improved, run_orb_improved, run_vwap_baseline, run_orb_baseline,
    combine_results, print_report,
    SYMBOLS, STARTING_CAPITAL, ORDER_SIZE_USD,
)

def main():
    print()
    print("=" * 65)
    print("  AtoBot — 1-MONTH BACKTEST (live config: VWAP + ORB)")
    print("=" * 65)
    print()
    print(f"  Symbols:  {', '.join(SYMBOLS)}")
    print(f"  Capital:  ${STARTING_CAPITAL:,.0f}")
    print(f"  Order:    ${ORDER_SIZE_USD:,.0f} per trade")

    # Last 1 month: Jan 22, 2026 → Feb 22, 2026
    start = datetime(2026, 1, 22, tzinfo=timezone.utc)
    end   = datetime(2026, 2, 22, tzinfo=timezone.utc)
    print(f"  Period:   {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print()

    # Fetch data
    bars_1m = fetch_bars(SYMBOLS, start, end)
    total_bars = sum(len(b) for b in bars_1m.values())
    if total_bars == 0:
        print("ERROR: No historical data. Check API keys.")
        sys.exit(1)

    all_days: set[str] = set()
    for sym_bars in bars_1m.values():
        for b in sym_bars:
            all_days.add(b["timestamp"].strftime("%Y-%m-%d"))
    trading_days = len(all_days)
    print(f"  Total bars: {total_bars:,} | Trading days: {trading_days}")

    bars_5m = {sym: bars_to_5min(b) for sym, b in bars_1m.items()}

    # Run strategies (improved = what's deployed on VPS)
    print("\n  Running strategies...")
    vwap = run_vwap_improved(bars_5m)
    orb  = run_orb_improved(bars_1m)
    combo = combine_results([vwap, orb], "VWAP+ORB (deployed)")

    # Also run baseline for comparison
    vwap_b = run_vwap_baseline(bars_5m)
    orb_b  = run_orb_baseline(bars_1m)
    combo_b = combine_results([vwap_b, orb_b], "VWAP+ORB (baseline)")

    # Print individual + combo reports
    for r in [vwap, orb, combo]:
        print_report(r, trading_days)

    # Comparison table
    print("\n")
    print("=" * 80)
    print("  1-MONTH PERFORMANCE SUMMARY")
    print("=" * 80)
    header = f"  {'Strategy':<28} {'Net P&L':>12} {'Win Rate':>10} {'Trades':>8} {'Max DD':>8} {'Mo. P&L':>10}"
    print(header)
    print(f"  {'-'*28} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")

    for r in [vwap_b, vwap, orb_b, orb, combo_b, combo]:
        monthly = r.net_pnl / trading_days * 21 if trading_days > 0 else 0
        pnl_str = f"${r.net_pnl:>+10,.2f}"
        mo_str  = f"${monthly:>+8,.0f}"
        print(f"  {r.name:<28} {pnl_str:>12} {r.win_rate:>9.1f}% {len(r.closed_pnl):>8} {r.max_drawdown_pct:>7.2f}% {mo_str:>10}")

    print("=" * 80)

    best = combo
    best_monthly = best.net_pnl / trading_days * 21 if trading_days > 0 else 0
    print(f"\n  YOUR LIVE CONFIG: {best.name}")
    print(f"    Net P&L (1 month): ${best.net_pnl:+,.2f}")
    print(f"    Win Rate:          {best.win_rate:.1f}%")
    print(f"    Total Trades:      {len(best.closed_pnl)}")
    print(f"    Max Drawdown:      {best.max_drawdown_pct:.2f}%")
    print(f"    Est Monthly:       ${best_monthly:+,.0f}/mo")
    print()

if __name__ == "__main__":
    main()
