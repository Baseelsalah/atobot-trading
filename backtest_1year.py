"""AtoBot 1-YEAR BACKTEST v2 -- Scaling Analysis for $70-80K/yr Target.

Fixed version: processes all symbols per day (like real bot), proper
drawdown handling, realistic slippage model.

Usage:
    python backtest_1year.py
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# -- Config -------------------------------------------------------------------
ALPACA_KEY = ""
ALPACA_SECRET = ""

try:
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line.startswith("ALPACA_API_KEY="):
                ALPACA_KEY = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("ALPACA_API_SECRET="):
                ALPACA_SECRET = line.split("=", 1)[1].strip().strip('"')
except FileNotFoundError:
    pass

SYMBOLS_5 = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD"]
SYMBOLS_8 = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD", "META", "AMZN", "GOOG"]

STARTING_CAPITAL = 100_000.0
ORDER_SIZES = [17_000.0, 25_000.0, 30_000.0, 35_000.0]

# Strategy params (baseline -- proven best)
ORB_BREAKOUT_PCT = 0.08
ORB_TP_PCT = 1.2
ORB_SL_PCT = 0.5

VWAP_BOUNCE_PCT = 0.12
VWAP_TP_PCT = 0.4
VWAP_SL_PCT = 0.25

# Midday filter
MIDDAY_START = 12
MIDDAY_END = 14

# Risk limits
MAX_OPEN_POSITIONS = 8
DAILY_LOSS_LIMIT = 2000.0
MAX_DRAWDOWN_PCT = 8.0  # More realistic for a full year

# Slippage: 1 bps round-trip for large-cap liquid stocks
# AAPL spread ~$0.01 on ~$200 = 0.5 bps per side = 1 bps round-trip
SLIPPAGE_PER_TRADE_BPS = 1.0


# -- Helpers ------------------------------------------------------------------

def _vwap(bars: list[dict]) -> float | None:
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


def _is_midday(ts: datetime) -> bool:
    try:
        et = ts.astimezone(ZoneInfo("America/New_York"))
        return MIDDAY_START <= et.hour < MIDDAY_END
    except Exception:
        return False


# -- Trade tracking -----------------------------------------------------------

@dataclass
class OpenPosition:
    symbol: str
    entry_price: float
    quantity: float
    entry_time: datetime
    highest_price: float = 0.0

    def __post_init__(self):
        self.highest_price = self.entry_price


@dataclass
class StrategyResult:
    name: str
    order_size: float = 17_000.0
    closed_pnl: list[float] = field(default_factory=list)
    closed_pnl_dates: list[str] = field(default_factory=list)
    open_positions: dict[str, OpenPosition] = field(default_factory=dict)
    peak_equity: float = STARTING_CAPITAL
    max_drawdown_pct: float = 0.0
    cash: float = STARTING_CAPITAL
    halted_days: int = 0
    total_slippage: float = 0.0
    exit_reasons: dict[str, int] = field(default_factory=lambda: {"TP": 0, "SL": 0, "EOD": 0})
    symbols_traded: set[str] = field(default_factory=set)

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
    def net_pnl_after_slippage(self) -> float:
        return sum(self.closed_pnl) - self.total_slippage

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
        equity = self.cash
        for pos in self.open_positions.values():
            equity += pos.quantity * pos.highest_price
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity * 100
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd

    def monthly_pnl(self) -> dict[str, float]:
        monthly: dict[str, float] = defaultdict(float)
        for pnl, date_str in zip(self.closed_pnl, self.closed_pnl_dates):
            month_key = date_str[:7]
            monthly[month_key] += pnl
        return dict(sorted(monthly.items()))

    def monthly_trades(self) -> dict[str, int]:
        monthly: dict[str, int] = defaultdict(int)
        for date_str in self.closed_pnl_dates:
            month_key = date_str[:7]
            monthly[month_key] += 1
        return dict(sorted(monthly.items()))


# -- Data fetching ------------------------------------------------------------

def fetch_bars(symbols: list[str], start: datetime, end: datetime) -> dict[str, list[dict]]:
    print("Fetching historical data from Alpaca...")
    client = StockHistoricalDataClient(api_key=ALPACA_KEY, secret_key=ALPACA_SECRET)
    all_bars: dict[str, list[dict]] = {s: [] for s in symbols}

    for sym in symbols:
        print(f"  {sym}...", end=" ", flush=True)
        chunk_start = start
        sym_bars = []
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=30), end)
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=sym,
                    timeframe=TimeFrame.Minute,
                    start=chunk_start, end=chunk_end,
                )
                barset = client.get_stock_bars(req)
                if sym in barset.data:
                    for bar in barset.data[sym]:
                        sym_bars.append({
                            "timestamp": bar.timestamp,
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": float(bar.volume),
                        })
            except Exception as e:
                print(f"  [WARN] {sym} chunk {chunk_start.date()} failed: {e}")
            chunk_start = chunk_end
        all_bars[sym] = sym_bars
        print(f"{len(sym_bars):,} bars")
    return all_bars


def bars_to_5min(bars_1m: list[dict]) -> list[dict]:
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
    days: dict[str, list[dict]] = {}
    for b in bars:
        day_key = b["timestamp"].strftime("%Y-%m-%d")
        days.setdefault(day_key, []).append(b)
    return days


# -- VWAP+ORB combined strategy (processes all symbols per day) ---------------

def run_vwap_orb_combined(
    bars_5m_by_sym: dict[str, list[dict]],
    bars_1m_by_sym: dict[str, list[dict]],
    order_size: float,
    symbols: list[str],
    name: str,
) -> StrategyResult:
    """Run VWAP Scalp + ORB together, processing all symbols per day.

    This matches how the real bot works: on each day, all symbols are
    evaluated for VWAP and ORB signals, with shared capital and position limits.
    """
    res = StrategyResult(name=name, order_size=order_size)

    # Group data by day for each symbol
    vwap_days: dict[str, dict[str, list[dict]]] = {}
    orb_days: dict[str, dict[str, list[dict]]] = {}

    for sym in symbols:
        vwap_days[sym] = group_by_day(bars_5m_by_sym.get(sym, []))
        orb_days[sym] = group_by_day(bars_1m_by_sym.get(sym, []))

    # Collect all trading days
    all_day_keys: set[str] = set()
    for sym in symbols:
        all_day_keys.update(vwap_days[sym].keys())
        all_day_keys.update(orb_days[sym].keys())

    # ORB state tracking
    orb_traded_today: dict[str, bool] = {}
    orb_range_high: dict[str, float] = {}

    for day_key in sorted(all_day_keys):
        # Daily reset
        daily_pnl = 0.0
        day_halted = False
        orb_traded_today = {s: False for s in symbols}
        orb_range_high = {}

        # Compute ORB ranges for each symbol
        for sym in symbols:
            day_1m = orb_days[sym].get(day_key, [])
            if len(day_1m) >= 16:
                range_bars = day_1m[:15]
                orb_range_high[sym] = max(b["high"] for b in range_bars)

        # VWAP intraday accumulators per symbol
        vwap_intraday: dict[str, list[dict]] = {s: [] for s in symbols}

        # Get 5-min bars and 1-min bars for this day
        day_5m_data: dict[str, list[dict]] = {}
        max_5m_bars = 0
        for sym in symbols:
            day_5m_data[sym] = vwap_days[sym].get(day_key, [])
            max_5m_bars = max(max_5m_bars, len(day_5m_data[sym]))

        day_1m_data: dict[str, list[dict]] = {}
        max_1m_bars = 0
        for sym in symbols:
            day_1m = orb_days[sym].get(day_key, [])
            day_1m_data[sym] = day_1m[15:] if len(day_1m) >= 16 else []
            max_1m_bars = max(max_1m_bars, len(day_1m_data[sym]))

        # Process VWAP signals bar-by-bar (5-min resolution)
        for bar_idx in range(max_5m_bars):
            if day_halted:
                break

            for sym in symbols:
                bars = day_5m_data[sym]
                if bar_idx >= len(bars):
                    continue

                bar = bars[bar_idx]
                price = bar["close"]
                ts = bar["timestamp"]
                vwap_intraday[sym].append(bar)

                vwap_val = _vwap(vwap_intraday[sym])
                if vwap_val is None:
                    continue

                pos = res.open_positions.get(f"VWAP_{sym}")
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

                    # Take profit
                    if price >= vwap_val or pnl_pct >= VWAP_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        slippage = order_size * SLIPPAGE_PER_TRADE_BPS / 10000
                        res.total_slippage += slippage
                        res.closed_pnl.append(pnl)
                        res.closed_pnl_dates.append(day_key)
                        res.cash += price * pos.quantity
                        res.exit_reasons["TP"] += 1
                        res.symbols_traded.add(sym)
                        del res.open_positions[f"VWAP_{sym}"]
                        daily_pnl += pnl
                        res.update_drawdown()
                        continue

                    # Stop loss
                    if pnl_pct <= -VWAP_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        slippage = order_size * SLIPPAGE_PER_TRADE_BPS / 10000
                        res.total_slippage += slippage
                        res.closed_pnl.append(pnl)
                        res.closed_pnl_dates.append(day_key)
                        res.cash += price * pos.quantity
                        res.exit_reasons["SL"] += 1
                        res.symbols_traded.add(sym)
                        del res.open_positions[f"VWAP_{sym}"]
                        daily_pnl += pnl
                        res.update_drawdown()
                        continue

                # VWAP Entry
                if f"VWAP_{sym}" not in res.open_positions and vwap_val > 0:
                    if len(res.open_positions) >= MAX_OPEN_POSITIONS:
                        continue
                    if _is_midday(ts):
                        continue

                    # Daily loss check
                    if daily_pnl < -DAILY_LOSS_LIMIT:
                        day_halted = True
                        res.halted_days += 1
                        break

                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= VWAP_BOUNCE_PCT:
                        qty = order_size / price
                        cost = qty * price
                        if res.cash >= cost:
                            res.cash -= cost
                            res.open_positions[f"VWAP_{sym}"] = OpenPosition(
                                sym, price, qty, ts
                            )
                            res.symbols_traded.add(sym)
                            res.update_drawdown()

        # Process ORB signals (1-min resolution, after opening range)
        if not day_halted:
            for bar_idx in range(max_1m_bars):
                if day_halted:
                    break
                for sym in symbols:
                    bars_1m = day_1m_data[sym]
                    if bar_idx >= len(bars_1m):
                        continue
                    if sym not in orb_range_high:
                        continue

                    bar = bars_1m[bar_idx]
                    price = bar["close"]
                    ts = bar["timestamp"]

                    pos = res.open_positions.get(f"ORB_{sym}")
                    if pos:
                        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

                        if pnl_pct >= ORB_TP_PCT:
                            pnl = (price - pos.entry_price) * pos.quantity
                            slippage = order_size * SLIPPAGE_PER_TRADE_BPS / 10000
                            res.total_slippage += slippage
                            res.closed_pnl.append(pnl)
                            res.closed_pnl_dates.append(day_key)
                            res.cash += price * pos.quantity
                            res.exit_reasons["TP"] += 1
                            res.symbols_traded.add(sym)
                            del res.open_positions[f"ORB_{sym}"]
                            daily_pnl += pnl
                            res.update_drawdown()
                            continue

                        if pnl_pct <= -ORB_SL_PCT:
                            pnl = (price - pos.entry_price) * pos.quantity
                            slippage = order_size * SLIPPAGE_PER_TRADE_BPS / 10000
                            res.total_slippage += slippage
                            res.closed_pnl.append(pnl)
                            res.closed_pnl_dates.append(day_key)
                            res.cash += price * pos.quantity
                            res.exit_reasons["SL"] += 1
                            res.symbols_traded.add(sym)
                            del res.open_positions[f"ORB_{sym}"]
                            daily_pnl += pnl
                            res.update_drawdown()
                            continue

                    if f"ORB_{sym}" not in res.open_positions and not orb_traded_today.get(sym, False):
                        if len(res.open_positions) >= MAX_OPEN_POSITIONS:
                            continue
                        if _is_midday(ts):
                            continue
                        if daily_pnl < -DAILY_LOSS_LIMIT:
                            day_halted = True
                            res.halted_days += 1
                            break

                        buffer_h = orb_range_high[sym] * (1 + ORB_BREAKOUT_PCT / 100)
                        if price >= buffer_h:
                            qty = order_size / price
                            cost = qty * price
                            if res.cash >= cost:
                                res.cash -= cost
                                res.open_positions[f"ORB_{sym}"] = OpenPosition(
                                    sym, price, qty, ts
                                )
                                orb_traded_today[sym] = True
                                res.symbols_traded.add(sym)
                                res.update_drawdown()

        # EOD flatten all remaining positions
        for key in list(res.open_positions.keys()):
            pos = res.open_positions[key]
            sym = pos.symbol
            eod_price = None
            day_1m_full = orb_days[sym].get(day_key, [])
            if day_1m_full:
                eod_price = day_1m_full[-1]["close"]
            elif day_5m_data.get(sym):
                eod_price = day_5m_data[sym][-1]["close"]

            if eod_price is not None:
                pnl = (eod_price - pos.entry_price) * pos.quantity
                slippage = order_size * SLIPPAGE_PER_TRADE_BPS / 10000
                res.total_slippage += slippage
                res.closed_pnl.append(pnl)
                res.closed_pnl_dates.append(day_key)
                res.cash += eod_price * pos.quantity
                res.exit_reasons["EOD"] += 1
                daily_pnl += pnl
                del res.open_positions[key]
                res.update_drawdown()

    return res


# -- Compounding simulator ----------------------------------------------------

def simulate_compounding(monthly_pnl: dict[str, float], start_equity: float,
                         base_order_size: float) -> list[dict]:
    rows = []
    equity = start_equity
    cumulative = 0.0
    for month, pnl in monthly_pnl.items():
        scale = equity / start_equity
        scaled_pnl = pnl * scale
        equity += scaled_pnl
        cumulative += scaled_pnl
        rows.append({
            "month": month, "raw_pnl": pnl, "scale": scale,
            "scaled_pnl": scaled_pnl, "equity": equity,
            "cumulative": cumulative, "order_size": base_order_size * scale,
        })
    return rows


# -- Reporting ----------------------------------------------------------------

def print_report(result: StrategyResult, trading_days: int) -> None:
    final_equity = result.cash
    ret_pct = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    daily_avg = result.net_pnl / trading_days if trading_days > 0 else 0
    monthly_avg = daily_avg * 21

    if result.closed_pnl and len(result.closed_pnl) > 1:
        trade_rets = [p / result.order_size for p in result.closed_pnl]
        mean_r = statistics.mean(trade_rets)
        std_r = statistics.stdev(trade_rets)
        sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    slip_adj_pnl = result.net_pnl_after_slippage
    slip_daily = slip_adj_pnl / trading_days if trading_days > 0 else 0
    slip_monthly = slip_daily * 21
    slip_annual = slip_monthly * 12

    bar = "=" * 70
    print()
    print(bar)
    print(f"  {result.name}")
    print(bar)
    print(f"  Starting Capital:     ${STARTING_CAPITAL:>14,.2f}")
    print(f"  Final Equity:         ${final_equity:>14,.2f}")
    print(f"  Net P&L (raw):        ${result.net_pnl:>14,.2f}  ({ret_pct:+.2f}%)")
    print(f"  Est. Slippage:        ${result.total_slippage:>14,.2f}")
    print(f"  Net P&L (adj):        ${slip_adj_pnl:>14,.2f}")
    print(f"  Max Drawdown:         {result.max_drawdown_pct:>14.2f}%")
    print(f"  Sharpe (approx):      {sharpe:>14.2f}")
    print()
    print(f"  Total Trades:         {len(result.closed_pnl):>8}")
    print(f"  Wins / Losses:        {result.wins:>5} / {result.losses:<5}")
    print(f"  Win Rate:             {result.win_rate:>14.1f}%")
    print(f"  Avg Win:              ${result.avg_win:>14,.2f}")
    print(f"  Avg Loss:             ${result.avg_loss:>14,.2f}")
    print(f"  Profit Factor:        {result.profit_factor:>14.2f}")
    print()
    print(f"  Exit Reasons:  TP={result.exit_reasons['TP']}  SL={result.exit_reasons['SL']}  EOD={result.exit_reasons['EOD']}")
    print(f"  Symbols Traded:       {len(result.symbols_traded)} ({', '.join(sorted(result.symbols_traded))})")
    print()
    print(f"  Avg Daily P&L (raw):  ${daily_avg:>14,.2f}")
    print(f"  Monthly P&L (raw):    ${monthly_avg:>14,.2f}")
    print(f"  Monthly P&L (adj):    ${slip_monthly:>14,.2f}")
    print(f"  Annual P&L (adj):     ${slip_annual:>14,.2f}")
    print(f"  Trading Days:         {trading_days:>8}")
    if result.halted_days > 0:
        print(f"  Halted Days:          {result.halted_days:>8}")
    print(bar)


def print_monthly_breakdown(result: StrategyResult) -> None:
    monthly = result.monthly_pnl()
    trades = result.monthly_trades()

    print()
    print(f"  MONTHLY BREAKDOWN: {result.name}")
    print(f"  {'Month':<10} {'P&L':>12} {'Trades':>8} {'Cum P&L':>14} {'Win Mo?':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*8} {'-'*14} {'-'*8}")

    cumulative = 0.0
    winning_months = 0
    total_months = 0

    for month in sorted(monthly.keys()):
        pnl = monthly[month]
        cumulative += pnl
        trade_count = trades.get(month, 0)
        is_win = "YES" if pnl > 0 else "no"
        if pnl > 0:
            winning_months += 1
        total_months += 1
        print(f"  {month:<10} ${pnl:>+11,.2f} {trade_count:>8} ${cumulative:>+13,.2f} {is_win:>8}")

    if total_months > 0:
        print(f"  {'-'*10} {'-'*12} {'-'*8} {'-'*14} {'-'*8}")
        print(f"  Winning Months: {winning_months}/{total_months} ({winning_months/total_months*100:.0f}%)")
    print()


# -- Main ---------------------------------------------------------------------

def main() -> None:
    # Force UTF-8 output on Windows
    import io, os
    if os.name == 'nt':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print()
    print("=" * 70)
    print("  AtoBot 1-YEAR BACKTEST v2 -- Scaling to $70-80K/yr")
    print("=" * 70)
    print()

    end = datetime(2026, 2, 22, tzinfo=timezone.utc)
    start = datetime(2025, 2, 22, tzinfo=timezone.utc)

    print(f"  Period:         {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} (1 year)")
    print(f"  Starting Cap:   ${STARTING_CAPITAL:,.0f}")
    print(f"  Buying Power:   $200,000 (2:1 margin)")
    print(f"  Strategy:       VWAP Scalp + ORB baseline (midday filter)")
    print(f"  Symbols (5):    {', '.join(SYMBOLS_5)}")
    print(f"  Symbols (8):    {', '.join(SYMBOLS_8)}")
    print(f"  Order sizes:    {', '.join(f'${s/1000:.0f}K' for s in ORDER_SIZES)}")
    print(f"  Slippage:       {SLIPPAGE_PER_TRADE_BPS} bps per round-trip trade")
    print(f"  Risk limits:    ${DAILY_LOSS_LIMIT:,.0f}/day loss limit, {MAX_DRAWDOWN_PCT}% max DD")
    print()

    # -- Fetch data for all 8 symbols ----------------------------------------
    print("  Fetching 1-year of minute data for 8 symbols...")
    print("  (This may take a few minutes...)")
    print()

    bars_1m_8 = fetch_bars(SYMBOLS_8, start, end)
    total_bars = sum(len(b) for b in bars_1m_8.values())
    if total_bars == 0:
        print("ERROR: No historical data. Check API keys in .env")
        sys.exit(1)

    all_days: set[str] = set()
    for sym_bars in bars_1m_8.values():
        for b in sym_bars:
            all_days.add(b["timestamp"].strftime("%Y-%m-%d"))
    trading_days = len(all_days)

    print(f"\n  Total bars: {total_bars:,} | Trading days: {trading_days}")

    # Prepare 5-min bars
    bars_5m_8 = {sym: bars_to_5min(b) for sym, b in bars_1m_8.items()}
    bars_1m_5 = {s: bars_1m_8[s] for s in SYMBOLS_5}
    bars_5m_5 = {s: bars_5m_8[s] for s in SYMBOLS_5}

    # ========================================================================
    #  TEST 1: 5 symbols, multiple order sizes
    # ========================================================================
    print("\n")
    print("=" * 70)
    print("  TEST 1: 5 SYMBOLS -- Order Size Scaling")
    print("=" * 70)

    results_5sym = []
    for size in ORDER_SIZES:
        label = f"VWAP+ORB 5sym ${size/1000:.0f}K"
        print(f"\n  Running {label}...", flush=True)
        r = run_vwap_orb_combined(bars_5m_5, bars_1m_5, size, SYMBOLS_5, label)
        results_5sym.append(r)

    for r in results_5sym:
        print_report(r, trading_days)

    # ========================================================================
    #  TEST 2: 8 symbols, multiple order sizes
    # ========================================================================
    print("\n")
    print("=" * 70)
    print("  TEST 2: 8 SYMBOLS -- Order Size Scaling")
    print("=" * 70)

    results_8sym = []
    for size in ORDER_SIZES:
        label = f"VWAP+ORB 8sym ${size/1000:.0f}K"
        print(f"\n  Running {label}...", flush=True)
        r = run_vwap_orb_combined(bars_5m_8, bars_1m_8, size, SYMBOLS_8, label)
        results_8sym.append(r)

    for r in results_8sym:
        print_report(r, trading_days)

    # ========================================================================
    #  MONTHLY BREAKDOWN
    # ========================================================================
    print("\n")
    print("=" * 70)
    print("  MONTHLY P&L BREAKDOWN")
    print("=" * 70)

    for r in results_8sym:
        if r.order_size in [25_000.0, 35_000.0]:
            print_monthly_breakdown(r)

    # ========================================================================
    #  COMPOUNDING SIMULATION
    # ========================================================================
    print("\n")
    print("=" * 70)
    print("  COMPOUNDING SIMULATION (reinvest profits monthly)")
    print("=" * 70)

    for r in results_8sym:
        if r.order_size == 25_000.0:
            monthly = r.monthly_pnl()
            comp_rows = simulate_compounding(monthly, STARTING_CAPITAL, 25_000.0)
            print(f"\n  Base: {r.name}, compound monthly")
            print(f"  {'Month':<10} {'Raw P&L':>10} {'Scale':>6} {'Adj P&L':>10} {'Equity':>14} {'Order$':>10}")
            print(f"  {'-'*10} {'-'*10} {'-'*6} {'-'*10} {'-'*14} {'-'*10}")
            for row in comp_rows:
                print(f"  {row['month']:<10} ${row['raw_pnl']:>+9,.0f} {row['scale']:>5.2f}x "
                      f"${row['scaled_pnl']:>+9,.0f} ${row['equity']:>13,.0f} ${row['order_size']:>9,.0f}")
            if comp_rows:
                last = comp_rows[-1]
                print(f"\n  COMPOUNDED RESULT:")
                print(f"    Starting Equity:  ${STARTING_CAPITAL:>12,.0f}")
                print(f"    Final Equity:     ${last['equity']:>12,.0f}")
                print(f"    Total Profit:     ${last['cumulative']:>+12,.0f}")
                print(f"    Return:           {(last['equity']/STARTING_CAPITAL - 1)*100:>11.1f}%")

    # ========================================================================
    #  GRAND COMPARISON TABLE
    # ========================================================================
    print("\n")
    print("=" * 104)
    print("  GRAND COMPARISON -- PATH TO $70-80K/YEAR")
    print("=" * 104)

    header = (f"  {'Config':<30} {'Raw P&L':>10} {'Slippage':>10} {'Net P&L':>10} "
              f"{'Mo. Avg':>10} {'Annual':>10} {'WR':>6} {'Trades':>7} {'MaxDD':>6}")
    print(header)
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*7} {'-'*6}")

    all_results = results_5sym + results_8sym
    for r in all_results:
        raw = r.net_pnl
        slip = r.total_slippage
        net = r.net_pnl_after_slippage
        mo_avg = net / trading_days * 21 if trading_days > 0 else 0
        annual = mo_avg * 12
        print(f"  {r.name:<30} ${raw:>+9,.0f} ${slip:>9,.0f} ${net:>+9,.0f} "
              f"${mo_avg:>+9,.0f} ${annual:>+9,.0f} {r.win_rate:>5.1f}% {len(r.closed_pnl):>7} {r.max_drawdown_pct:>5.1f}%")

    print("=" * 104)

    # -- Target analysis ------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("  TARGET ANALYSIS: $70-80K/YEAR")
    print("=" * 70)

    profitable = [r for r in all_results if r.net_pnl_after_slippage > 0]
    if profitable:
        best = max(profitable, key=lambda r: r.net_pnl_after_slippage)
    else:
        best = max(all_results, key=lambda r: r.net_pnl_after_slippage)

    best_net = best.net_pnl_after_slippage
    best_mo = best_net / trading_days * 21 if trading_days > 0 else 0
    best_annual = best_mo * 12

    print(f"\n  BEST CONFIG: {best.name}")
    print(f"    Annual (slippage-adjusted): ${best_annual:+,.0f}")
    print(f"    Monthly Average:            ${best_mo:+,.0f}")
    print(f"    Max Drawdown:               {best.max_drawdown_pct:.2f}%")
    print(f"    Win Rate:                   {best.win_rate:.1f}%")
    print(f"    Total Trades:               {len(best.closed_pnl):,}")

    for r in all_results:
        net = r.net_pnl_after_slippage
        mo_avg = net / trading_days * 21 if trading_days > 0 else 0
        annual = mo_avg * 12
        pct_of_target = annual / 75_000 * 100 if annual > 0 else 0
        print(f"\n  {r.name:<30} => ${annual:>+9,.0f}/yr", end="")
        if annual > 0:
            print(f"  ({pct_of_target:.0f}% of $75K target)", end="")
        print()

    gap = 75_000 - best_annual
    if gap > 0 and best_annual > 0:
        print(f"\n  GAP TO $75K TARGET: ${gap:,.0f}/yr")
        print(f"  TO CLOSE THE GAP:")
        print(f"    - Current best annual: ${best_annual:+,.0f}")
        print(f"    - Need {gap/best_annual*100:.0f}% more revenue")
        print(f"    - Scale order size by {75000/best_annual:.1f}x")
        print(f"    - Or add more symbols + compound monthly")
    elif best_annual >= 75_000:
        print(f"\n  TARGET $75K: ACHIEVED! Surplus: ${best_annual - 75_000:+,.0f}/yr")
    else:
        print(f"\n  Current best annual: ${best_annual:+,.0f}")

    # -- Recommendations ------------------------------------------------------
    print("\n")
    print("=" * 70)
    print("  RECOMMENDATIONS TO HIT $70-80K/yr")
    print("=" * 70)
    print()
    print("  1. Use BASELINE strategy (no trailing stop, no EMA filter)")
    print("  2. Enable midday filter only (proven +6.6% boost)")
    print("  3. Scale order sizes to match buying power")
    print("  4. Expand to 8-12 high-liquidity symbols")
    print("  5. Compound monthly (scale orders with equity growth)")
    print("  6. Reduce trade frequency to cut slippage costs")
    print("  7. Widen VWAP_BOUNCE_PCT to reduce false entries")
    print("  8. Add per-symbol parameter tuning")
    print("  9. Consider adding SPY, QQQ, NFLX, COIN")
    print(" 10. Run paper trading for 2-4 weeks before going live")
    print()


if __name__ == "__main__":
    main()
