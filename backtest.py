"""AtoBot Backtester — replay historical Alpaca data through all 3 strategies.

Usage:
    python backtest.py

Fetches 1 month of 5-minute bars from Alpaca, simulates each strategy
independently, and prints a performance report.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN

# ── Alpaca SDK ────────────────────────────────────────────────────────────────
from alpaca.trading.client import TradingClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ── Config ────────────────────────────────────────────────────────────────────
ALPACA_KEY = "REDACTED_KEY"
ALPACA_SECRET = "REDACTED_SECRET"

SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD"]
STARTING_CAPITAL = 100_000.0
ORDER_SIZE_USD = 17_000.0  # ~17% of capital per trade → targets $5k/month
COMMISSION_PER_TRADE = 0.0  # Alpaca is commission-free

# Momentum params (tuned for higher conviction entries)
MOM_RSI_PERIOD = 14
MOM_RSI_OVERSOLD = 35.0   # Wider entry zone → more trades
MOM_RSI_OVERBOUGHT = 65.0
MOM_VOL_MULT = 1.3        # Slightly lower volume filter → more entries
MOM_TP_PCT = 1.5           # Tighter TP → lock in gains faster
MOM_SL_PCT = 0.75          # Tighter SL → cut losers fast

# ORB params (tuned)
ORB_RANGE_BARS = 3  # first 3 five-min bars = 15 min opening range
ORB_BREAKOUT_PCT = 0.08   # Slightly tighter breakout confirmation
ORB_TP_PCT = 1.2           # Faster profit-taking
ORB_SL_PCT = 0.5           # Tighter stop

# VWAP params (tuned)
VWAP_BOUNCE_PCT = 0.12    # Enter closer to VWAP → more trades
VWAP_TP_PCT = 0.4          # Faster scalps
VWAP_SL_PCT = 0.25         # Tight stop


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rsi(closes: list[float], period: int = 14) -> float | None:
    """Compute RSI from a list of close prices."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    # Wilder smoothed
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _avg_volume(volumes: list[float], period: int = 20) -> float:
    """Simple average of last `period` volume bars."""
    if len(volumes) < period:
        return sum(volumes) / max(len(volumes), 1)
    return sum(volumes[-period:]) / period


def _vwap(bars: list[dict]) -> float | None:
    """Compute VWAP from bar dicts with high, low, close, volume."""
    total_tp_vol = 0.0
    total_vol = 0.0
    for b in bars:
        tp = (b["high"] + b["low"] + b["close"]) / 3.0
        vol = b["volume"]
        if vol <= 0:
            continue
        total_tp_vol += tp * vol
        total_vol += vol
    if total_vol <= 0:
        return None
    return total_tp_vol / total_vol


# ── Trade tracking ────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol: str
    side: str  # "BUY" or "SELL"
    price: float
    quantity: float
    timestamp: datetime
    reason: str = ""


@dataclass
class OpenPosition:
    symbol: str
    entry_price: float
    quantity: float
    entry_time: datetime


@dataclass
class StrategyResult:
    name: str
    trades: list[BacktestTrade] = field(default_factory=list)
    closed_pnl: list[float] = field(default_factory=list)
    open_positions: dict[str, OpenPosition] = field(default_factory=dict)
    peak_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    cash: float = STARTING_CAPITAL

    @property
    def total_trades(self) -> int:
        return len([t for t in self.trades if t.side == "SELL"])

    @property
    def wins(self) -> int:
        return len([p for p in self.closed_pnl if p > 0])

    @property
    def losses(self) -> int:
        return len([p for p in self.closed_pnl if p <= 0])

    @property
    def win_rate(self) -> float:
        if not self.closed_pnl:
            return 0.0
        return self.wins / len(self.closed_pnl) * 100

    @property
    def net_pnl(self) -> float:
        return sum(self.closed_pnl)

    @property
    def avg_win(self) -> float:
        w = [p for p in self.closed_pnl if p > 0]
        return sum(w) / len(w) if w else 0.0

    @property
    def avg_loss(self) -> float:
        l_list = [p for p in self.closed_pnl if p <= 0]
        return sum(l_list) / len(l_list) if l_list else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(p for p in self.closed_pnl if p > 0)
        gross_loss = abs(sum(p for p in self.closed_pnl if p <= 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def update_drawdown(self) -> None:
        equity = self.cash + sum(
            pos.quantity * pos.entry_price for pos in self.open_positions.values()
        )
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity * 100
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_bars(symbols: list[str], start: datetime, end: datetime) -> dict[str, list[dict]]:
    """Fetch 5-min bars for all symbols from Alpaca."""
    print("Fetching historical data from Alpaca...")
    client = StockHistoricalDataClient(api_key=ALPACA_KEY, secret_key=ALPACA_SECRET)

    all_bars: dict[str, list[dict]] = {s: [] for s in symbols}

    for sym in symbols:
        print(f"  {sym}...", end=" ", flush=True)
        req = StockBarsRequest(
            symbol_or_symbols=sym,
            timeframe=TimeFrame.Minute,  # 1-min for ORB; we'll also build 5-min
            start=start,
            end=end,
        )
        barset = client.get_stock_bars(req)

        if sym in barset.data:
            for bar in barset.data[sym]:
                all_bars[sym].append({
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                })
        print(f"{len(all_bars[sym])} bars")

    return all_bars


def bars_to_5min(bars_1m: list[dict]) -> list[dict]:
    """Aggregate 1-min bars into 5-min bars."""
    result = []
    i = 0
    while i + 4 < len(bars_1m):
        chunk = bars_1m[i:i + 5]
        agg = {
            "timestamp": chunk[0]["timestamp"],
            "open": chunk[0]["open"],
            "high": max(b["high"] for b in chunk),
            "low": min(b["low"] for b in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(b["volume"] for b in chunk),
        }
        result.append(agg)
        i += 5
    return result


def group_by_day(bars: list[dict]) -> dict[str, list[dict]]:
    """Group bars by trading day (date string)."""
    days: dict[str, list[dict]] = {}
    for b in bars:
        day_key = b["timestamp"].strftime("%Y-%m-%d")
        days.setdefault(day_key, []).append(b)
    return days


# ── Strategy simulators ──────────────────────────────────────────────────────

def run_momentum(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """Simulate Momentum strategy across all symbols."""
    res = StrategyResult(name="Momentum")

    for sym, bars in bars_5m_by_sym.items():
        days = group_by_day(bars)

        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            closes: list[float] = []
            volumes: list[float] = []

            for bar in day_bars:
                closes.append(bar["close"])
                volumes.append(bar["volume"])
                price = bar["close"]
                ts = bar["timestamp"]

                # ── Exit check ────────────────────────────────────────
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if pnl_pct >= MOM_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "TP"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                    if pnl_pct <= -MOM_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "SL"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue

                # ── Entry check (no position) ─────────────────────────
                if sym not in res.open_positions and len(closes) >= MOM_RSI_PERIOD + 2:
                    current_rsi = _rsi(closes, MOM_RSI_PERIOD)
                    prev_rsi = _rsi(closes[:-1], MOM_RSI_PERIOD)
                    avg_vol = _avg_volume(volumes[:-1], 20)
                    cur_vol = volumes[-1]

                    rsi_oversold = current_rsi is not None and current_rsi <= MOM_RSI_OVERSOLD
                    rsi_crossover = (
                        prev_rsi is not None and current_rsi is not None
                        and prev_rsi <= MOM_RSI_OVERSOLD and current_rsi > prev_rsi
                    )
                    vol_ok = avg_vol > 0 and cur_vol >= avg_vol * MOM_VOL_MULT

                    if (rsi_oversold or rsi_crossover) and vol_ok:
                        qty = ORDER_SIZE_USD / price
                        cost = qty * price
                        if res.cash >= cost:
                            res.cash -= cost
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts)
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "RSI+Vol"))
                            res.update_drawdown()

            # ── EOD flatten ───────────────────────────────────────────
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                eod_price = day_bars[-1]["close"]
                pnl = (eod_price - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += eod_price * pos.quantity
                res.trades.append(BacktestTrade(sym, "SELL", eod_price, pos.quantity, day_bars[-1]["timestamp"], "EOD"))
                del res.open_positions[sym]
                res.update_drawdown()

    return res


def run_orb(bars_1m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """Simulate ORB strategy across all symbols using 1-min bars."""
    res = StrategyResult(name="ORB (Opening Range Breakout)")

    for sym, bars in bars_1m_by_sym.items():
        days = group_by_day(bars)

        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            if len(day_bars) < ORB_RANGE_BARS + 1:
                continue

            # Build opening range from first 15 bars (15 min of 1-min bars)
            range_bars = day_bars[:15]  # 15 minutes
            range_high = max(b["high"] for b in range_bars)
            range_low = min(b["low"] for b in range_bars)
            buffer_h = range_high * (1 + ORB_BREAKOUT_PCT / 100)
            buffer_l = range_low * (1 - ORB_BREAKOUT_PCT / 100)

            traded_today = False

            for bar in day_bars[15:]:
                price = bar["close"]
                ts = bar["timestamp"]

                # ── Exit check ────────────────────────────────────────
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if pnl_pct >= ORB_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "TP"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                    if pnl_pct <= -ORB_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "SL"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue

                # ── Breakout entry ────────────────────────────────────
                if sym not in res.open_positions and not traded_today:
                    if price >= buffer_h:
                        qty = ORDER_SIZE_USD / price
                        cost = qty * price
                        if res.cash >= cost:
                            res.cash -= cost
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts)
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "Breakout"))
                            traded_today = True
                            res.update_drawdown()

            # ── EOD flatten ───────────────────────────────────────────
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                eod_price = day_bars[-1]["close"]
                pnl = (eod_price - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += eod_price * pos.quantity
                res.trades.append(BacktestTrade(sym, "SELL", eod_price, pos.quantity, day_bars[-1]["timestamp"], "EOD"))
                del res.open_positions[sym]
                res.update_drawdown()

    return res


def run_vwap(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """Simulate VWAP Scalp strategy across all symbols."""
    res = StrategyResult(name="VWAP Scalp")

    for sym, bars in bars_5m_by_sym.items():
        days = group_by_day(bars)

        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            intraday_bars: list[dict] = []

            for bar in day_bars:
                intraday_bars.append(bar)
                price = bar["close"]
                ts = bar["timestamp"]

                vwap_val = _vwap(intraday_bars)
                if vwap_val is None:
                    continue

                # ── Exit check ────────────────────────────────────────
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

                    # TP: price returned to VWAP or PnL target
                    if price >= vwap_val or pnl_pct >= VWAP_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "TP/VWAP"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue

                    if pnl_pct <= -VWAP_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "SL"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue

                # ── Entry: price below VWAP by bounce % ──────────────
                if sym not in res.open_positions and vwap_val > 0:
                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= VWAP_BOUNCE_PCT:
                        qty = ORDER_SIZE_USD / price
                        cost = qty * price
                        if res.cash >= cost:
                            res.cash -= cost
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts)
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "VWAP dip"))
                            res.update_drawdown()

            # ── EOD flatten ───────────────────────────────────────────
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                eod_price = day_bars[-1]["close"]
                pnl = (eod_price - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += eod_price * pos.quantity
                res.trades.append(BacktestTrade(sym, "SELL", eod_price, pos.quantity, day_bars[-1]["timestamp"], "EOD"))
                del res.open_positions[sym]
                res.update_drawdown()

    return res


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(result: StrategyResult, trading_days: int) -> None:
    """Print a formatted performance report."""
    final_equity = result.cash
    ret_pct = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    buys = len([t for t in result.trades if t.side == "BUY"])
    sells = len([t for t in result.trades if t.side == "SELL"])
    daily_avg = result.net_pnl / trading_days if trading_days > 0 else 0

    # Count by exit reason
    tp_count = len([t for t in result.trades if t.side == "SELL" and "TP" in t.reason])
    sl_count = len([t for t in result.trades if t.side == "SELL" and t.reason == "SL"])
    eod_count = len([t for t in result.trades if t.side == "SELL" and t.reason == "EOD"])

    # Sharpe approximation (daily returns)
    if result.closed_pnl:
        import statistics
        daily_rets: dict[str, float] = {}
        for t in result.trades:
            if t.side == "SELL":
                day = t.timestamp.strftime("%Y-%m-%d")
                # find matching pnl — approximate by accumulating
        # Simpler: use trade-level returns
        trade_rets = [p / ORDER_SIZE_USD for p in result.closed_pnl]
        if len(trade_rets) > 1:
            mean_r = statistics.mean(trade_rets)
            std_r = statistics.stdev(trade_rets)
            sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0.0
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    bar = "=" * 60
    print()
    print(bar)
    print(f"  {result.name}")
    print(bar)
    print(f"  Starting Capital:     ${STARTING_CAPITAL:>12,.2f}")
    print(f"  Final Equity:         ${final_equity:>12,.2f}")
    print(f"  Net P&L:              ${result.net_pnl:>12,.2f}  ({ret_pct:+.2f}%)")
    print(f"  Max Drawdown:         {result.max_drawdown_pct:>12.2f}%")
    print(f"  Sharpe (approx):      {sharpe:>12.2f}")
    print()
    print(f"  Total Round-Trips:    {len(result.closed_pnl):>6}")
    print(f"  Wins / Losses:        {result.wins:>3} / {result.losses:<3}")
    print(f"  Win Rate:             {result.win_rate:>12.1f}%")
    print(f"  Avg Win:              ${result.avg_win:>12,.2f}")
    print(f"  Avg Loss:             ${result.avg_loss:>12,.2f}")
    print(f"  Profit Factor:        {result.profit_factor:>12.2f}")
    print()
    print(f"  Exits by reason:")
    print(f"    Take-Profit:        {tp_count:>6}")
    print(f"    Stop-Loss:          {sl_count:>6}")
    print(f"    EOD Flatten:        {eod_count:>6}")
    print()
    print(f"  Avg Daily P&L:        ${daily_avg:>12,.2f}")
    print(f"  Trading Days:         {trading_days:>6}")
    print(bar)


def print_trade_log(result: StrategyResult, last_n: int = 15) -> None:
    """Print last N trades."""
    sells = [t for t in result.trades if t.side == "SELL"]
    if not sells:
        print("  No completed trades.")
        return
    print(f"\n  Last {min(last_n, len(sells))} trades:")
    print(f"  {'Date':<20} {'Symbol':<6} {'Entry':>9} {'Exit':>9} {'P&L':>10} {'Reason':<10}")
    print(f"  {'-'*20} {'-'*6} {'-'*9} {'-'*9} {'-'*10} {'-'*10}")

    # Match buys to sells (simple sequential matching per symbol)
    buy_stack: dict[str, list[BacktestTrade]] = {}
    for t in result.trades:
        if t.side == "BUY":
            buy_stack.setdefault(t.symbol, []).append(t)

    paired: list[tuple[BacktestTrade, BacktestTrade]] = []
    sell_stack: dict[str, int] = {s: 0 for s in buy_stack}
    for t in result.trades:
        if t.side == "SELL":
            idx = sell_stack.get(t.symbol, 0)
            buys = buy_stack.get(t.symbol, [])
            if idx < len(buys):
                paired.append((buys[idx], t))
                sell_stack[t.symbol] = idx + 1

    for buy_t, sell_t in paired[-last_n:]:
        pnl = (sell_t.price - buy_t.price) * sell_t.quantity
        marker = "+" if pnl >= 0 else ""
        print(f"  {sell_t.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {sell_t.symbol:<6} "
              f"${buy_t.price:>8.2f} ${sell_t.price:>8.2f} {marker}${pnl:>9.2f} {sell_t.reason:<10}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 60)
    print("  AtoBot Backtester — 3 Month Historical Replay")
    print("=" * 60)
    print()
    print(f"  Symbols:  {', '.join(SYMBOLS)}")
    print(f"  Capital:  ${STARTING_CAPITAL:,.0f}")
    print(f"  Order:    ${ORDER_SIZE_USD:,.0f} per trade")
    print()

    # Date range: last ~3 months of market data
    end = datetime(2026, 2, 20, tzinfo=timezone.utc)  # Yesterday (Feb 21 is today)
    start = datetime(2025, 11, 20, tzinfo=timezone.utc)  # 3 months back

    print(f"  Period:   {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print()

    # Fetch 1-min bars (used for ORB; aggregated to 5-min for Momentum/VWAP)
    bars_1m = fetch_bars(SYMBOLS, start, end)

    total_bars = sum(len(b) for b in bars_1m.values())
    if total_bars == 0:
        print("ERROR: No historical data received. Check API keys and dates.")
        sys.exit(1)

    # Count trading days
    all_days: set[str] = set()
    for sym_bars in bars_1m.values():
        for b in sym_bars:
            all_days.add(b["timestamp"].strftime("%Y-%m-%d"))
    trading_days = len(all_days)
    print(f"\n  Total bars fetched: {total_bars:,}")
    print(f"  Trading days:       {trading_days}")

    # Build 5-min bars
    bars_5m = {sym: bars_to_5min(b) for sym, b in bars_1m.items()}
    total_5m = sum(len(b) for b in bars_5m.values())
    print(f"  5-min bars:         {total_5m:,}")

    # ── Run all 3 strategies ──────────────────────────────────────────────
    print("\n  Running Momentum strategy...")
    mom_result = run_momentum(bars_5m)
    print_report(mom_result, trading_days)
    print_trade_log(mom_result)

    print("\n  Running ORB strategy...")
    orb_result = run_orb(bars_1m)
    print_report(orb_result, trading_days)
    print_trade_log(orb_result)

    print("\n  Running VWAP Scalp strategy...")
    vwap_result = run_vwap(bars_5m)
    print_report(vwap_result, trading_days)
    print_trade_log(vwap_result)

    # ── Summary comparison ────────────────────────────────────────────────
    print("\n")
    print("=" * 60)
    print("  STRATEGY COMPARISON")
    print("=" * 60)
    print(f"  {'Strategy':<25} {'Net P&L':>12} {'Win Rate':>10} {'Trades':>8} {'Max DD':>8}")
    print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")
    for r in [mom_result, orb_result, vwap_result]:
        pnl_str = f"${r.net_pnl:>+10,.2f}"
        print(f"  {r.name:<25} {pnl_str:>12} {r.win_rate:>9.1f}% {len(r.closed_pnl):>8} {r.max_drawdown_pct:>7.2f}%")
    print("=" * 60)

    best = max([mom_result, orb_result, vwap_result], key=lambda r: r.net_pnl)
    print(f"\n  Best strategy: {best.name} (${best.net_pnl:+,.2f})")
    print()


if __name__ == "__main__":
    main()
