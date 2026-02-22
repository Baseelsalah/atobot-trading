"""AtoBot Backtester v2 — Improved strategies with EMA trend filter,
time-of-day filter, and trailing stop-loss.

Compares baseline (old) vs. improved (new) across all 3 strategies
and shows a multi-strategy combined result (VWAP + ORB).

Usage:
    python backtest_v2.py
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ── Config ────────────────────────────────────────────────────────────────────
ALPACA_KEY = ""
ALPACA_SECRET = ""

# Try to load from .env
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

SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD"]
STARTING_CAPITAL = 100_000.0
ORDER_SIZE_USD = 17_000.0

# Strategy params (same as v1 baseline)
MOM_RSI_PERIOD = 14
MOM_RSI_OVERSOLD = 35.0
MOM_VOL_MULT = 1.3
MOM_TP_PCT = 1.5
MOM_SL_PCT = 0.75

ORB_BREAKOUT_PCT = 0.08
ORB_TP_PCT = 1.2
ORB_SL_PCT = 0.5

VWAP_BOUNCE_PCT = 0.12
VWAP_TP_PCT = 0.4
VWAP_SL_PCT = 0.25

# ── NEW: Improvement params ────────────────────────────────────────────────
TREND_EMA_PERIOD = 20          # 20-bar EMA on 5-min for trend filter
MIDDAY_START = 12              # Avoid entries 12:00-14:00 ET
MIDDAY_END = 14
TRAILING_ACTIVATION_PCT = 0.5  # Activate trail after 0.5% profit
TRAILING_DISTANCE_PCT = 0.3   # Trail distance from high


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
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
    if len(volumes) < period:
        return sum(volumes) / max(len(volumes), 1)
    return sum(volumes[-period:]) / period


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


def _ema(closes: list[float], period: int) -> float | None:
    """Compute EMA of the last `period` values."""
    if len(closes) < period:
        return None
    multiplier = 2.0 / (period + 1)
    ema_val = sum(closes[:period]) / period  # SMA seed
    for price in closes[period:]:
        ema_val = (price - ema_val) * multiplier + ema_val
    return ema_val


def _is_midday(ts: datetime) -> bool:
    """Check if timestamp falls in the midday dead zone (12-2 PM ET)."""
    try:
        et = ts.astimezone(ZoneInfo("America/New_York"))
        return MIDDAY_START <= et.hour < MIDDAY_END
    except Exception:
        return False


# ── Trade tracking ────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol: str
    side: str
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
    highest_price: float = 0.0  # For trailing stop

    def __post_init__(self):
        self.highest_price = self.entry_price


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
    print("Fetching historical data from Alpaca...")
    client = StockHistoricalDataClient(api_key=ALPACA_KEY, secret_key=ALPACA_SECRET)
    all_bars: dict[str, list[dict]] = {s: [] for s in symbols}

    for sym in symbols:
        print(f"  {sym}...", end=" ", flush=True)
        req = StockBarsRequest(
            symbol_or_symbols=sym,
            timeframe=TimeFrame.Minute,
            start=start, end=end,
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


# ── Trailing stop helper ─────────────────────────────────────────────────────

def _check_trailing_stop(pos: OpenPosition, price: float) -> bool:
    """Update highest price and return True if trailing stop triggered."""
    if price > pos.highest_price:
        pos.highest_price = price

    profit_pct = (pos.highest_price - pos.entry_price) / pos.entry_price * 100
    if profit_pct < TRAILING_ACTIVATION_PCT:
        return False

    trail_stop = pos.highest_price * (1 - TRAILING_DISTANCE_PCT / 100)
    return price <= trail_stop


# ── Strategy simulators (IMPROVED with filters) ──────────────────────────────

def run_momentum_improved(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """Momentum + EMA trend filter + time-of-day filter + trailing stop."""
    res = StrategyResult(name="Momentum (improved)")

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

                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

                    # TRAILING STOP (new)
                    if _check_trailing_stop(pos, price):
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "TRAIL"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue

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

                # Entry
                if sym not in res.open_positions and len(closes) >= MOM_RSI_PERIOD + 2:
                    # TIME-OF-DAY FILTER (new)
                    if _is_midday(ts):
                        continue

                    # EMA TREND FILTER (new)
                    ema_val = _ema(closes, TREND_EMA_PERIOD)
                    if ema_val is not None and price < ema_val:
                        continue

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

            # EOD flatten
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


def run_orb_improved(bars_1m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """ORB + EMA trend filter + time-of-day filter + trailing stop."""
    res = StrategyResult(name="ORB (improved)")

    # Pre-compute 5-min bars for EMA calculation per symbol
    bars_5m = {sym: bars_to_5min(b) for sym, b in bars_1m_by_sym.items()}

    for sym, bars in bars_1m_by_sym.items():
        days = group_by_day(bars)
        days_5m = group_by_day(bars_5m.get(sym, []))

        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            if len(day_bars) < 16:
                continue

            range_bars = day_bars[:15]
            range_high = max(b["high"] for b in range_bars)
            range_low = min(b["low"] for b in range_bars)
            buffer_h = range_high * (1 + ORB_BREAKOUT_PCT / 100)

            traded_today = False

            # Get 5-min closes for EMA trend
            day_5m = days_5m.get(day_key, [])
            closes_5m = [b["close"] for b in day_5m]

            for bar in day_bars[15:]:
                price = bar["close"]
                ts = bar["timestamp"]

                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

                    # TRAILING STOP
                    if _check_trailing_stop(pos, price):
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "TRAIL"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue

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

                if sym not in res.open_positions and not traded_today:
                    # TIME-OF-DAY FILTER
                    if _is_midday(ts):
                        continue

                    # EMA TREND FILTER (use accumulated 5-min closes)
                    if len(closes_5m) >= TREND_EMA_PERIOD:
                        ema_val = _ema(closes_5m, TREND_EMA_PERIOD)
                        if ema_val is not None and price < ema_val:
                            continue

                    if price >= buffer_h:
                        qty = ORDER_SIZE_USD / price
                        cost = qty * price
                        if res.cash >= cost:
                            res.cash -= cost
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts)
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "Breakout"))
                            traded_today = True
                            res.update_drawdown()

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


def run_vwap_improved(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """VWAP Scalp + EMA trend filter + time-of-day filter + trailing stop."""
    res = StrategyResult(name="VWAP Scalp (improved)")

    for sym, bars in bars_5m_by_sym.items():
        days = group_by_day(bars)

        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            intraday_bars: list[dict] = []
            closes: list[float] = []

            for bar in day_bars:
                intraday_bars.append(bar)
                closes.append(bar["close"])
                price = bar["close"]
                ts = bar["timestamp"]

                vwap_val = _vwap(intraday_bars)
                if vwap_val is None:
                    continue

                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

                    # TRAILING STOP (new)
                    if _check_trailing_stop(pos, price):
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "TRAIL"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue

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

                if sym not in res.open_positions and vwap_val > 0:
                    # TIME-OF-DAY FILTER
                    if _is_midday(ts):
                        continue

                    # EMA TREND FILTER
                    if len(closes) >= TREND_EMA_PERIOD:
                        ema_val = _ema(closes, TREND_EMA_PERIOD)
                        if ema_val is not None and price < ema_val:
                            continue

                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= VWAP_BOUNCE_PCT:
                        qty = ORDER_SIZE_USD / price
                        cost = qty * price
                        if res.cash >= cost:
                            res.cash -= cost
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts)
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "VWAP dip"))
                            res.update_drawdown()

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


# ── Baseline (old) strategies for comparison ─────────────────────────────────

def run_momentum_baseline(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """Baseline Momentum (no filters)."""
    res = StrategyResult(name="Momentum (baseline)")
    for sym, bars in bars_5m_by_sym.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            closes, volumes = [], []
            for bar in day_bars:
                closes.append(bar["close"]); volumes.append(bar["volume"])
                price = bar["close"]; ts = bar["timestamp"]
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if pnl_pct >= MOM_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl); res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "TP"))
                        del res.open_positions[sym]; res.update_drawdown(); continue
                    if pnl_pct <= -MOM_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl); res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "SL"))
                        del res.open_positions[sym]; res.update_drawdown(); continue
                if sym not in res.open_positions and len(closes) >= MOM_RSI_PERIOD + 2:
                    current_rsi = _rsi(closes, MOM_RSI_PERIOD)
                    prev_rsi = _rsi(closes[:-1], MOM_RSI_PERIOD)
                    avg_vol = _avg_volume(volumes[:-1], 20); cur_vol = volumes[-1]
                    rsi_oversold = current_rsi is not None and current_rsi <= MOM_RSI_OVERSOLD
                    rsi_crossover = (prev_rsi is not None and current_rsi is not None
                        and prev_rsi <= MOM_RSI_OVERSOLD and current_rsi > prev_rsi)
                    vol_ok = avg_vol > 0 and cur_vol >= avg_vol * MOM_VOL_MULT
                    if (rsi_oversold or rsi_crossover) and vol_ok:
                        qty = ORDER_SIZE_USD / price; cost = qty * price
                        if res.cash >= cost:
                            res.cash -= cost
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts)
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "RSI+Vol"))
                            res.update_drawdown()
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]; eod_price = day_bars[-1]["close"]
                pnl = (eod_price - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl); res.cash += eod_price * pos.quantity
                res.trades.append(BacktestTrade(sym, "SELL", eod_price, pos.quantity, day_bars[-1]["timestamp"], "EOD"))
                del res.open_positions[sym]; res.update_drawdown()
    return res


def run_orb_baseline(bars_1m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """Baseline ORB (no filters)."""
    res = StrategyResult(name="ORB (baseline)")
    for sym, bars in bars_1m_by_sym.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            if len(day_bars) < 16: continue
            range_bars = day_bars[:15]
            range_high = max(b["high"] for b in range_bars)
            buffer_h = range_high * (1 + ORB_BREAKOUT_PCT / 100)
            traded_today = False
            for bar in day_bars[15:]:
                price = bar["close"]; ts = bar["timestamp"]
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if pnl_pct >= ORB_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl); res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "TP"))
                        del res.open_positions[sym]; res.update_drawdown(); continue
                    if pnl_pct <= -ORB_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl); res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "SL"))
                        del res.open_positions[sym]; res.update_drawdown(); continue
                if sym not in res.open_positions and not traded_today:
                    if price >= buffer_h:
                        qty = ORDER_SIZE_USD / price; cost = qty * price
                        if res.cash >= cost:
                            res.cash -= cost
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts)
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "Breakout"))
                            traded_today = True; res.update_drawdown()
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]; eod_price = day_bars[-1]["close"]
                pnl = (eod_price - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl); res.cash += eod_price * pos.quantity
                res.trades.append(BacktestTrade(sym, "SELL", eod_price, pos.quantity, day_bars[-1]["timestamp"], "EOD"))
                del res.open_positions[sym]; res.update_drawdown()
    return res


def run_vwap_baseline(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """Baseline VWAP (no filters)."""
    res = StrategyResult(name="VWAP Scalp (baseline)")
    for sym, bars in bars_5m_by_sym.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]; intraday_bars = []
            for bar in day_bars:
                intraday_bars.append(bar); price = bar["close"]; ts = bar["timestamp"]
                vwap_val = _vwap(intraday_bars)
                if vwap_val is None: continue
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if price >= vwap_val or pnl_pct >= VWAP_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl); res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "TP/VWAP"))
                        del res.open_positions[sym]; res.update_drawdown(); continue
                    if pnl_pct <= -VWAP_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl); res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "SL"))
                        del res.open_positions[sym]; res.update_drawdown(); continue
                if sym not in res.open_positions and vwap_val > 0:
                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= VWAP_BOUNCE_PCT:
                        qty = ORDER_SIZE_USD / price; cost = qty * price
                        if res.cash >= cost:
                            res.cash -= cost
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts)
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "VWAP dip"))
                            res.update_drawdown()
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]; eod_price = day_bars[-1]["close"]
                pnl = (eod_price - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl); res.cash += eod_price * pos.quantity
                res.trades.append(BacktestTrade(sym, "SELL", eod_price, pos.quantity, day_bars[-1]["timestamp"], "EOD"))
                del res.open_positions[sym]; res.update_drawdown()
    return res


# ── Multi-strategy combiner ──────────────────────────────────────────────────

def combine_results(results: list[StrategyResult], name: str) -> StrategyResult:
    """Merge multiple strategy results into one combined result.
    
    Note: This is a simplified combination that sums P&L.
    In reality, strategies would share capital and may conflict on symbols.
    """
    combined = StrategyResult(name=name)
    combined.closed_pnl = []
    combined.trades = []
    for r in results:
        combined.closed_pnl.extend(r.closed_pnl)
        combined.trades.extend(r.trades)
    combined.cash = STARTING_CAPITAL + sum(combined.closed_pnl)
    # Use worst drawdown as conservative estimate
    combined.max_drawdown_pct = max(r.max_drawdown_pct for r in results)
    return combined


# ── Isolated filter tests (VWAP only) ────────────────────────────────────────

def run_vwap_trail_only(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """VWAP + trailing stop ONLY (no EMA, no midday filter)."""
    res = StrategyResult(name="VWAP + Trail Only")
    for sym, bars in bars_5m_by_sym.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]; intraday_bars = []
            for bar in day_bars:
                intraday_bars.append(bar); price = bar["close"]; ts = bar["timestamp"]
                vwap_val = _vwap(intraday_bars)
                if vwap_val is None: continue
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if _check_trailing_stop(pos, price):
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl); res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "TRAIL"))
                        del res.open_positions[sym]; res.update_drawdown(); continue
                    if price >= vwap_val or pnl_pct >= VWAP_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl); res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "TP/VWAP"))
                        del res.open_positions[sym]; res.update_drawdown(); continue
                    if pnl_pct <= -VWAP_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl); res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "SL"))
                        del res.open_positions[sym]; res.update_drawdown(); continue
                if sym not in res.open_positions and vwap_val > 0:
                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= VWAP_BOUNCE_PCT:
                        qty = ORDER_SIZE_USD / price; cost = qty * price
                        if res.cash >= cost:
                            res.cash -= cost
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts)
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "VWAP dip"))
                            res.update_drawdown()
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]; eod_price = day_bars[-1]["close"]
                pnl = (eod_price - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl); res.cash += eod_price * pos.quantity
                res.trades.append(BacktestTrade(sym, "SELL", eod_price, pos.quantity, day_bars[-1]["timestamp"], "EOD"))
                del res.open_positions[sym]; res.update_drawdown()
    return res


def run_vwap_midday_only(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """VWAP + midday filter ONLY (no EMA, no trailing)."""
    res = StrategyResult(name="VWAP + Midday Filter")
    for sym, bars in bars_5m_by_sym.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]; intraday_bars = []
            for bar in day_bars:
                intraday_bars.append(bar); price = bar["close"]; ts = bar["timestamp"]
                vwap_val = _vwap(intraday_bars)
                if vwap_val is None: continue
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if price >= vwap_val or pnl_pct >= VWAP_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl); res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "TP/VWAP"))
                        del res.open_positions[sym]; res.update_drawdown(); continue
                    if pnl_pct <= -VWAP_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl); res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "SL"))
                        del res.open_positions[sym]; res.update_drawdown(); continue
                if sym not in res.open_positions and vwap_val > 0:
                    if _is_midday(ts): continue
                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= VWAP_BOUNCE_PCT:
                        qty = ORDER_SIZE_USD / price; cost = qty * price
                        if res.cash >= cost:
                            res.cash -= cost
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts)
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "VWAP dip"))
                            res.update_drawdown()
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]; eod_price = day_bars[-1]["close"]
                pnl = (eod_price - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl); res.cash += eod_price * pos.quantity
                res.trades.append(BacktestTrade(sym, "SELL", eod_price, pos.quantity, day_bars[-1]["timestamp"], "EOD"))
                del res.open_positions[sym]; res.update_drawdown()
    return res


def run_vwap_ema_only(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """VWAP + EMA trend filter ONLY (no midday, no trailing)."""
    res = StrategyResult(name="VWAP + EMA Filter")
    for sym, bars in bars_5m_by_sym.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]; intraday_bars = []; closes = []
            for bar in day_bars:
                intraday_bars.append(bar); closes.append(bar["close"])
                price = bar["close"]; ts = bar["timestamp"]
                vwap_val = _vwap(intraday_bars)
                if vwap_val is None: continue
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if price >= vwap_val or pnl_pct >= VWAP_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl); res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "TP/VWAP"))
                        del res.open_positions[sym]; res.update_drawdown(); continue
                    if pnl_pct <= -VWAP_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl); res.cash += price * pos.quantity
                        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, "SL"))
                        del res.open_positions[sym]; res.update_drawdown(); continue
                if sym not in res.open_positions and vwap_val > 0:
                    if len(closes) >= TREND_EMA_PERIOD:
                        ema_val = _ema(closes, TREND_EMA_PERIOD)
                        if ema_val is not None and price < ema_val:
                            continue
                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= VWAP_BOUNCE_PCT:
                        qty = ORDER_SIZE_USD / price; cost = qty * price
                        if res.cash >= cost:
                            res.cash -= cost
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts)
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "VWAP dip"))
                            res.update_drawdown()
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]; eod_price = day_bars[-1]["close"]
                pnl = (eod_price - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl); res.cash += eod_price * pos.quantity
                res.trades.append(BacktestTrade(sym, "SELL", eod_price, pos.quantity, day_bars[-1]["timestamp"], "EOD"))
                del res.open_positions[sym]; res.update_drawdown()
    return res


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(result: StrategyResult, trading_days: int) -> None:
    final_equity = result.cash
    ret_pct = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    daily_avg = result.net_pnl / trading_days if trading_days > 0 else 0
    monthly_avg = daily_avg * 21  # ~21 trading days/month

    tp_count = len([t for t in result.trades if t.side == "SELL" and "TP" in t.reason])
    sl_count = len([t for t in result.trades if t.side == "SELL" and t.reason == "SL"])
    eod_count = len([t for t in result.trades if t.side == "SELL" and t.reason == "EOD"])
    trail_count = len([t for t in result.trades if t.side == "SELL" and t.reason == "TRAIL"])

    if result.closed_pnl and len(result.closed_pnl) > 1:
        trade_rets = [p / ORDER_SIZE_USD for p in result.closed_pnl]
        mean_r = statistics.mean(trade_rets)
        std_r = statistics.stdev(trade_rets)
        sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0.0
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
    print(f"    Trailing Stop:      {trail_count:>6}")
    print(f"    EOD Flatten:        {eod_count:>6}")
    print()
    print(f"  Avg Daily P&L:        ${daily_avg:>12,.2f}")
    print(f"  Est. Monthly P&L:     ${monthly_avg:>12,.2f}")
    print(f"  Trading Days:         {trading_days:>6}")
    print(bar)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 60)
    print("  AtoBot Backtester v2 — Baseline vs Improved")
    print("=" * 60)
    print()
    print(f"  Symbols:  {', '.join(SYMBOLS)}")
    print(f"  Capital:  ${STARTING_CAPITAL:,.0f}")
    print(f"  Order:    ${ORDER_SIZE_USD:,.0f} per trade")
    print()
    print("  Improvements:")
    print(f"    EMA Trend Filter:   {TREND_EMA_PERIOD}-period EMA (uptrend only)")
    print(f"    Time-of-Day:        Skip {MIDDAY_START}:00-{MIDDAY_END}:00 ET")
    print(f"    Trailing Stop:      Activate {TRAILING_ACTIVATION_PCT}%, distance {TRAILING_DISTANCE_PCT}%")
    print()

    end = datetime(2026, 2, 20, tzinfo=timezone.utc)
    start = datetime(2025, 11, 20, tzinfo=timezone.utc)
    print(f"  Period:   {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print()

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
    print(f"\n  Total bars: {total_bars:,} | Trading days: {trading_days}")

    bars_5m = {sym: bars_to_5min(b) for sym, b in bars_1m.items()}

    # ── Run baseline ──────────────────────────────────────────────────────
    print("\n  Running BASELINE strategies...")
    mom_base = run_momentum_baseline(bars_5m)
    orb_base = run_orb_baseline(bars_1m)
    vwap_base = run_vwap_baseline(bars_5m)

    # ── Run improved ──────────────────────────────────────────────────────
    print("  Running IMPROVED strategies...")
    mom_imp = run_momentum_improved(bars_5m)
    orb_imp = run_orb_improved(bars_1m)
    vwap_imp = run_vwap_improved(bars_5m)

    # ── Run individual filter tests on VWAP ──────────────────────────────
    print("  Running VWAP filter isolation tests...")

    # VWAP + trailing stop only (no EMA, no midday)
    _save = (TREND_EMA_PERIOD, MIDDAY_START, MIDDAY_END)
    vwap_trail_only = run_vwap_trail_only(bars_5m)

    # VWAP + midday filter only
    vwap_midday_only = run_vwap_midday_only(bars_5m)

    # VWAP + EMA filter only
    vwap_ema_only = run_vwap_ema_only(bars_5m)

    # ── Multi-strategy combos ─────────────────────────────────────────────
    combo_base = combine_results([vwap_base, orb_base], "VWAP+ORB (baseline)")
    combo_imp = combine_results([vwap_imp, orb_imp], "VWAP+ORB (improved)")
    combo_all = combine_results([vwap_imp, orb_imp, mom_imp], "ALL 3 (improved)")

    # Best combo: VWAP+trail + ORB baseline
    combo_trail = combine_results([vwap_trail_only, orb_base], "VWAP(trail)+ORB")
    combo_midday = combine_results([vwap_midday_only, orb_base], "VWAP(midday)+ORB")

    # ── Print reports ─────────────────────────────────────────────────────
    for r in [vwap_base, vwap_trail_only, vwap_midday_only, vwap_ema_only, vwap_imp]:
        print_report(r, trading_days)

    print_report(combo_base, trading_days)
    print_report(combo_trail, trading_days)
    print_report(combo_midday, trading_days)

    # ── Comparison table ──────────────────────────────────────────────────
    print("\n")
    print("=" * 80)
    print("  BASELINE vs IMPROVED COMPARISON")
    print("=" * 80)
    print(f"  {'Strategy':<28} {'Net P&L':>12} {'Win Rate':>10} {'Trades':>8} {'Max DD':>8} {'Mo. P&L':>10}")
    print(f"  {'-'*28} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")

    all_results = [
        vwap_base, vwap_trail_only, vwap_midday_only, vwap_ema_only, vwap_imp,
        orb_base, orb_imp,
        mom_base, mom_imp,
        combo_base, combo_trail, combo_midday, combo_imp, combo_all,
    ]

    for r in all_results:
        pnl_str = f"${r.net_pnl:>+10,.2f}"
        monthly = r.net_pnl / trading_days * 21 if trading_days > 0 else 0
        mo_str = f"${monthly:>+8,.0f}"
        print(f"  {r.name:<28} {pnl_str:>12} {r.win_rate:>9.1f}% {len(r.closed_pnl):>8} {r.max_drawdown_pct:>7.2f}% {mo_str:>10}")

    print("=" * 80)

    best = max(all_results, key=lambda r: r.net_pnl)
    best_monthly = best.net_pnl / trading_days * 21 if trading_days > 0 else 0
    print(f"\n  BEST: {best.name}")
    print(f"    Net P&L: ${best.net_pnl:+,.2f} | Est Monthly: ${best_monthly:+,.0f}/mo")
    print()


if __name__ == "__main__":
    main()
