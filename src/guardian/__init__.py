"""AtoBot Guardian Agent — self-healing, self-improving watchdog.

Runs as a sidecar container alongside the trading bot.  Every cycle it:

1. **Health Monitor** — checks container status, API connectivity, disk/memory.
2. **Self-Healer** — auto-restarts crashed services, clears stuck orders,
   frees disk space, reconnects on API failures.
3. **Performance Analyzer** — tracks daily/weekly P&L, win rate, Sharpe,
   per-strategy breakdown, detects degradation.
4. **Auto-Tuner** — adjusts strategy parameters (within safe bounds) based
   on rolling performance windows.  Changes are logged & reversible.
"""
