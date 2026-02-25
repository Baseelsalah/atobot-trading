"""AtoBot Stress Test — COVID Crash (Feb 19 – May 15, 2020)
======================================================================
The single WORST 3-month period for day trading since 2010:
  • VIX peaked at 82.69 (highest since 2008)
  • 4 market-wide circuit breakers in 8 trading days
  • SPX dropped 34% in 23 days, then rallied 35% off the bottom
  • 70% of March days had overnight gaps >1%
  • Mean-reversion failed in BOTH directions back-to-back
  • ORB breakouts reversed 60-70% of the time

Purpose: Expose system weaknesses under extreme stress. If we can
survive (or even profit) here, the system is battle-tested.

Uses production-matching parameters:
  • $17K VWAP / $17K ORB order sizes
  • VWAP: 0.05% bounce, 0.4% TP, 0.50% SL
  • ORB: 0.1% breakout, 1.5% TP, 0.75% SL
  • Short selling enabled
  • $100K starting capital
  • EOD flatten, midday avoidance
  • Confluence-like filters (MACD, RSI, volume)
  • Daily loss limit simulation ($2K)
  • Max drawdown halt (5%)

Usage:
    python backtest_stress_covid.py
"""

from __future__ import annotations

import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# -- Config --------------------------------------------------------------------
ALPACA_KEY = ""
ALPACA_SECRET = ""

for env_file in [".env", "deploy/.env.production"]:
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ALPACA_API_KEY="):
                    ALPACA_KEY = line.split("=", 1)[1].strip().strip('"')
                elif line.startswith("ALPACA_API_SECRET="):
                    ALPACA_SECRET = line.split("=", 1)[1].strip().strip('"')
        if ALPACA_KEY and ALPACA_SECRET:
            break
    except FileNotFoundError:
        continue

# Symbols that existed and were liquid during Feb-May 2020
# NOTE: META was "FB" in 2020, GOOGL existed
SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD", "GOOGL", "AMZN", "NFLX", "SPY", "QQQ"]
# FB is the ticker for Meta during this period
SYMBOLS_FB = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD", "GOOGL", "AMZN", "NFLX", "SPY", "QQQ"]

STARTING_CAPITAL = 100_000.0

# -- Production-matching parameters -------------------------------------------
VWAP_ORDER_SIZE_USD = 17_000.0
ORB_ORDER_SIZE_USD = 17_000.0
MAX_POSITION_USD = 20_000.0

# VWAP Scalp (matching production .env)
VWAP_BOUNCE_PCT = 0.05       # entry threshold
VWAP_TP_PCT = 0.4            # take profit %
VWAP_SL_PCT = 0.50           # stop loss %

# ORB (matching production)
ORB_RANGE_MINUTES = 15
ORB_BREAKOUT_PCT = 0.1
ORB_TP_PCT = 1.5
ORB_SL_PCT = 0.75
ORB_VOL_CONFIRM = 1.3        # volume must be 1.3x avg

# Risk management
DAILY_LOSS_LIMIT = 2_000.0    # Production setting
MAX_DRAWDOWN_PCT = 5.0        # Halt if drawdown exceeds this
MAX_OPEN_POSITIONS = 10

# Filters
MIDDAY_START = 12
MIDDAY_END = 14
TREND_EMA_PERIOD = 20
TRAILING_ACTIVATION_PCT = 0.5
TRAILING_DISTANCE_PCT = 0.3

# Stress period: Feb 19 – May 15, 2020
STRESS_START = datetime(2020, 2, 19, tzinfo=timezone.utc)
STRESS_END = datetime(2020, 5, 15, tzinfo=timezone.utc)


# -- Helpers -------------------------------------------------------------------

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
    if len(closes) < period:
        return None
    multiplier = 2.0 / (period + 1)
    ema_val = sum(closes[:period]) / period
    for price in closes[period:]:
        ema_val = (price - ema_val) * multiplier + ema_val
    return ema_val


def _macd(closes: list[float]) -> dict | None:
    if len(closes) < 26:
        return None
    fast = _ema(closes, 12)
    slow = _ema(closes, 26)
    if fast is None or slow is None:
        return None
    macd_line = fast - slow
    if len(closes) >= 35:
        macd_hist = []
        for i in range(26, len(closes)):
            f = _ema(closes[:i+1], 12)
            s = _ema(closes[:i+1], 26)
            if f and s:
                macd_hist.append(f - s)
        if len(macd_hist) >= 9:
            signal = _ema(macd_hist, 9)
            return {"macd": macd_line, "signal": signal, "histogram": macd_line - (signal or 0)}
    return {"macd": macd_line, "signal": None, "histogram": 0}


def _atr(bars: list[dict], period: int = 14) -> float | None:
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


def _is_midday(ts: datetime) -> bool:
    try:
        et = ts.astimezone(ZoneInfo("America/New_York"))
        return MIDDAY_START <= et.hour < MIDDAY_END
    except Exception:
        return False


def _is_market_hours(ts: datetime) -> bool:
    try:
        et = ts.astimezone(ZoneInfo("America/New_York"))
        return 9 <= et.hour < 16
    except Exception:
        return True


# -- Trade tracking ------------------------------------------------------------

@dataclass
class BacktestTrade:
    symbol: str
    side: str
    price: float
    quantity: float
    timestamp: datetime
    reason: str = ""
    strategy: str = ""
    pnl: float = 0.0


@dataclass
class OpenPosition:
    symbol: str
    entry_price: float
    quantity: float
    entry_time: datetime
    side: str = "LONG"
    highest_price: float = 0.0
    lowest_price: float = 0.0
    strategy: str = ""

    def __post_init__(self):
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price


@dataclass
class DailyStats:
    date: str
    pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    halted: bool = False
    halt_reason: str = ""
    open_gap_pct: float = 0.0  # overnight gap %
    intraday_range_pct: float = 0.0  # daily high-low range %


@dataclass
class StrategyResult:
    name: str
    trades: list[BacktestTrade] = field(default_factory=list)
    closed_pnl: list[float] = field(default_factory=list)
    open_positions: dict[str, OpenPosition] = field(default_factory=dict)
    peak_equity: float = STARTING_CAPITAL
    max_drawdown_pct: float = 0.0
    cash: float = STARTING_CAPITAL
    daily_stats: list[DailyStats] = field(default_factory=list)
    halted_days: int = 0
    circuit_breaker_days: int = 0

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


def _close_position(res: StrategyResult, sym: str, price: float, ts: datetime,
                     reason: str, strategy: str = "") -> float:
    """Close position and return PnL."""
    pos = res.open_positions.get(sym)
    if not pos:
        return 0.0
    if pos.side == "LONG":
        pnl = (price - pos.entry_price) * pos.quantity
        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, reason, strategy, pnl))
        res.cash += price * pos.quantity
    else:
        pnl = (pos.entry_price - price) * pos.quantity
        res.trades.append(BacktestTrade(sym, "COVER", price, pos.quantity, ts, reason, strategy, pnl))
        margin_held = pos.quantity * pos.entry_price * 0.5
        res.cash += margin_held + pnl
    res.closed_pnl.append(pnl)
    del res.open_positions[sym]
    res.update_drawdown()
    return pnl


def _check_trailing_stop(pos: OpenPosition, price: float) -> bool:
    if pos.side == "LONG":
        if price > pos.highest_price:
            pos.highest_price = price
        profit_pct = (pos.highest_price - pos.entry_price) / pos.entry_price * 100
        if profit_pct < TRAILING_ACTIVATION_PCT:
            return False
        trail_stop = pos.highest_price * (1 - TRAILING_DISTANCE_PCT / 100)
        return price <= trail_stop
    else:
        if price < pos.lowest_price:
            pos.lowest_price = price
        profit_pct = (pos.entry_price - pos.lowest_price) / pos.entry_price * 100
        if profit_pct < TRAILING_ACTIVATION_PCT:
            return False
        trail_stop = pos.lowest_price * (1 + TRAILING_DISTANCE_PCT / 100)
        return price >= trail_stop


# -- Data fetching -------------------------------------------------------------

def fetch_bars(symbols: list[str], start: datetime, end: datetime) -> dict[str, list[dict]]:
    print("Fetching historical data from Alpaca...")
    client = StockHistoricalDataClient(api_key=ALPACA_KEY, secret_key=ALPACA_SECRET)
    all_bars: dict[str, list[dict]] = {s: [] for s in symbols}

    for sym in symbols:
        print(f"  {sym}...", end=" ", flush=True)
        # Fetch in chunks to handle large date ranges
        chunk_start = start
        sym_bars = []
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=30), end)
            req = StockBarsRequest(
                symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                start=chunk_start, end=chunk_end,
            )
            try:
                barset = client.get_stock_bars(req)
                if sym in barset.data:
                    for bar in barset.data[sym]:
                        sym_bars.append({
                            "timestamp": bar.timestamp,
                            "open": float(bar.open), "high": float(bar.high),
                            "low": float(bar.low), "close": float(bar.close),
                            "volume": float(bar.volume),
                        })
            except Exception as e:
                print(f"[err: {e}]", end=" ")
            chunk_start = chunk_end
        all_bars[sym] = sym_bars
        print(f"{len(sym_bars)} bars")
    return all_bars


def bars_to_5min(bars_1m: list[dict]) -> list[dict]:
    result = []
    i = 0
    while i + 4 < len(bars_1m):
        chunk = bars_1m[i:i + 5]
        # Only aggregate if same day
        d0 = chunk[0]["timestamp"].strftime("%Y-%m-%d")
        d4 = chunk[4]["timestamp"].strftime("%Y-%m-%d")
        if d0 != d4:
            i += 1
            continue
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


# ==============================================================================
#  STRATEGY ENGINES
# ==============================================================================


def run_vwap_strategy(bars_5m_by_sym: dict[str, list[dict]],
                      spy_daily_data: dict[str, dict]) -> StrategyResult:
    """VWAP Scalp Long+Short with production-matching params + daily risk limits.

    Includes stress-test enhancements:
    - Daily loss limit ($2K) — halts trading for the rest of the day
    - Max drawdown halt (5%) — halts trading entirely
    - ATR-based dynamic stop loss (never tighter than baseline)
    - Confluence filter (RSI + MACD + volume)
    """
    res = StrategyResult(name="VWAP L+S (Production)")

    daily_pnl: dict[str, float] = {}
    halted = False

    for sym, bars in bars_5m_by_sym.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            if day_key not in daily_pnl:
                daily_pnl[day_key] = 0.0

            # Check max drawdown halt
            if res.max_drawdown_pct >= MAX_DRAWDOWN_PCT:
                if not halted:
                    halted = True
                    res.halted_days += 1
                continue

            # Check daily loss limit
            if daily_pnl[day_key] <= -DAILY_LOSS_LIMIT:
                continue

            intraday_bars: list[dict] = []
            closes: list[float] = []
            volumes: list[float] = []

            for bar in day_bars:
                intraday_bars.append(bar)
                closes.append(bar["close"])
                volumes.append(bar["volume"])
                price = bar["close"]
                ts = bar["timestamp"]

                if not _is_market_hours(ts):
                    continue

                vwap_val = _vwap(intraday_bars)
                if vwap_val is None:
                    continue

                # -- Manage existing position --
                pos = res.open_positions.get(sym)
                if pos and pos.strategy == "vwap_scalp":
                    if _check_trailing_stop(pos, price):
                        pnl = _close_position(res, sym, price, ts, "TRAIL", "vwap_scalp")
                        daily_pnl[day_key] += pnl
                        continue

                    if pos.side == "LONG":
                        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                        if price >= vwap_val or pnl_pct >= VWAP_TP_PCT:
                            pnl = _close_position(res, sym, price, ts, "TP/VWAP", "vwap_scalp")
                            daily_pnl[day_key] += pnl
                            continue
                        # Dynamic SL: use ATR if available, never tighter than baseline
                        dynamic_sl = VWAP_SL_PCT
                        atr_val = _atr(intraday_bars)
                        if atr_val and price > 0:
                            atr_pct = (atr_val / price) * 100
                            dynamic_sl = max(VWAP_SL_PCT, min(1.0, atr_pct * 1.5))
                        if pnl_pct <= -dynamic_sl:
                            pnl = _close_position(res, sym, price, ts, "SL", "vwap_scalp")
                            daily_pnl[day_key] += pnl
                            continue
                    else:  # SHORT
                        pnl_pct = (pos.entry_price - price) / pos.entry_price * 100
                        if price <= vwap_val or pnl_pct >= VWAP_TP_PCT:
                            pnl = _close_position(res, sym, price, ts, "TP/VWAP", "vwap_scalp")
                            daily_pnl[day_key] += pnl
                            continue
                        dynamic_sl = VWAP_SL_PCT
                        atr_val = _atr(intraday_bars)
                        if atr_val and price > 0:
                            atr_pct = (atr_val / price) * 100
                            dynamic_sl = max(VWAP_SL_PCT, min(1.0, atr_pct * 1.5))
                        if pnl_pct <= -dynamic_sl:
                            pnl = _close_position(res, sym, price, ts, "SL", "vwap_scalp")
                            daily_pnl[day_key] += pnl
                            continue

                # -- Entry logic --
                if sym not in res.open_positions and vwap_val > 0:
                    # Check daily loss limit before entering
                    if daily_pnl[day_key] <= -DAILY_LOSS_LIMIT:
                        continue

                    # Midday filter
                    if _is_midday(ts):
                        continue

                    # Max positions check
                    if len(res.open_positions) >= MAX_OPEN_POSITIONS:
                        continue

                    # -- Confluence filter (simplified) --
                    confluence_score = 0
                    rsi_val = _rsi(closes) if len(closes) >= 15 else None
                    macd_info = _macd(closes) if len(closes) >= 26 else None
                    avg_vol = _avg_volume(volumes[:-1]) if len(volumes) > 1 else 0
                    cur_vol = volumes[-1] if volumes else 0

                    # LONG entry: price below VWAP
                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= VWAP_BOUNCE_PCT:
                        # Confluence checks for LONG
                        if rsi_val and rsi_val < 70:  # Not overbought
                            confluence_score += 1
                        if rsi_val and rsi_val < 40:  # Oversold bonus
                            confluence_score += 1
                        if macd_info and macd_info.get("histogram", 0) > -0.5:
                            confluence_score += 1
                        if avg_vol > 0 and cur_vol >= avg_vol * 0.8:
                            confluence_score += 1

                        # Need at least 2/4 confluence signals
                        if confluence_score >= 2:
                            qty = min(VWAP_ORDER_SIZE_USD, MAX_POSITION_USD) / price
                            if res.cash >= qty * price:
                                res.cash -= qty * price
                                res.open_positions[sym] = OpenPosition(
                                    sym, price, qty, ts, "LONG", strategy="vwap_scalp"
                                )
                                res.trades.append(BacktestTrade(
                                    sym, "BUY", price, qty, ts, "VWAP dip", "vwap_scalp"
                                ))
                                res.update_drawdown()

                    # SHORT entry: price above VWAP
                    short_dev = (price - vwap_val) / vwap_val * 100
                    if short_dev >= VWAP_BOUNCE_PCT and sym not in res.open_positions:
                        # Confluence for SHORT
                        confluence_score = 0
                        if rsi_val and rsi_val > 30:
                            confluence_score += 1
                        if rsi_val and rsi_val > 60:
                            confluence_score += 1
                        if macd_info and macd_info.get("histogram", 0) < 0.5:
                            confluence_score += 1
                        if avg_vol > 0 and cur_vol >= avg_vol * 0.8:
                            confluence_score += 1

                        if confluence_score >= 2:
                            qty = min(VWAP_ORDER_SIZE_USD, MAX_POSITION_USD) / price
                            margin_req = qty * price * 0.5
                            if res.cash >= margin_req:
                                res.cash -= margin_req
                                res.open_positions[sym] = OpenPosition(
                                    sym, price, qty, ts, "SHORT", strategy="vwap_scalp"
                                )
                                res.trades.append(BacktestTrade(
                                    sym, "SHORT", price, qty, ts, "VWAP pop", "vwap_scalp"
                                ))
                                res.update_drawdown()

            # EOD flatten
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                if pos.strategy == "vwap_scalp":
                    eod_price = day_bars[-1]["close"]
                    pnl = _close_position(res, sym, eod_price, day_bars[-1]["timestamp"], "EOD", "vwap_scalp")
                    if day_key in daily_pnl:
                        daily_pnl[day_key] += pnl

    # Record daily stats
    for day, pnl in sorted(daily_pnl.items()):
        halted_flag = pnl <= -DAILY_LOSS_LIMIT
        res.daily_stats.append(DailyStats(date=day, pnl=pnl, halted=halted_flag))
        if halted_flag:
            res.halted_days += 1

    return res


def run_orb_strategy(bars_1m_by_sym: dict[str, list[dict]],
                     spy_daily_data: dict[str, dict]) -> StrategyResult:
    """ORB Long+Short with production-matching params + risk limits."""
    res = StrategyResult(name="ORB L+S (Production)")
    daily_pnl: dict[str, float] = {}

    for sym, bars in bars_1m_by_sym.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            if len(day_bars) < 16:
                continue

            if day_key not in daily_pnl:
                daily_pnl[day_key] = 0.0

            # Check max drawdown halt
            if res.max_drawdown_pct >= MAX_DRAWDOWN_PCT:
                continue

            # Daily loss limit
            if daily_pnl[day_key] <= -DAILY_LOSS_LIMIT:
                continue

            range_bars = day_bars[:ORB_RANGE_MINUTES]
            range_high = max(b["high"] for b in range_bars)
            range_low = min(b["low"] for b in range_bars)
            range_pct = (range_high - range_low) / range_low * 100 if range_low > 0 else 0

            buffer_h = range_high * (1 + ORB_BREAKOUT_PCT / 100)
            buffer_l = range_low * (1 - ORB_BREAKOUT_PCT / 100)
            traded_today = False

            # Collect volumes for confirmation
            volumes_before: list[float] = [b["volume"] for b in range_bars]

            for bar in day_bars[ORB_RANGE_MINUTES:]:
                price = bar["close"]
                ts = bar["timestamp"]

                if not _is_market_hours(ts):
                    continue

                pos = res.open_positions.get(sym)
                if pos and pos.strategy == "orb":
                    if _check_trailing_stop(pos, price):
                        pnl = _close_position(res, sym, price, ts, "TRAIL", "orb")
                        daily_pnl[day_key] += pnl
                        continue

                    if pos.side == "LONG":
                        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    else:
                        pnl_pct = (pos.entry_price - price) / pos.entry_price * 100

                    if pnl_pct >= ORB_TP_PCT:
                        pnl = _close_position(res, sym, price, ts, "TP", "orb")
                        daily_pnl[day_key] += pnl
                        continue
                    if pnl_pct <= -ORB_SL_PCT:
                        pnl = _close_position(res, sym, price, ts, "SL", "orb")
                        daily_pnl[day_key] += pnl
                        continue

                if not traded_today and sym not in res.open_positions:
                    if _is_midday(ts):
                        continue
                    if len(res.open_positions) >= MAX_OPEN_POSITIONS:
                        continue
                    if daily_pnl[day_key] <= -DAILY_LOSS_LIMIT:
                        continue

                    # Volume confirmation
                    avg_vol = sum(volumes_before) / len(volumes_before) if volumes_before else 0
                    if avg_vol > 0 and bar["volume"] < avg_vol * ORB_VOL_CONFIRM:
                        volumes_before.append(bar["volume"])
                        continue

                    # LONG breakout
                    if price > buffer_h:
                        qty = min(ORB_ORDER_SIZE_USD, MAX_POSITION_USD) / price
                        if res.cash >= qty * price:
                            res.cash -= qty * price
                            res.open_positions[sym] = OpenPosition(
                                sym, price, qty, ts, "LONG", strategy="orb"
                            )
                            res.trades.append(BacktestTrade(
                                sym, "BUY", price, qty, ts, f"ORB break (range={range_pct:.1f}%)", "orb"
                            ))
                            traded_today = True
                            res.update_drawdown()

                    # SHORT breakdown
                    elif price < buffer_l:
                        qty = min(ORB_ORDER_SIZE_USD, MAX_POSITION_USD) / price
                        margin_req = qty * price * 0.5
                        if res.cash >= margin_req:
                            res.cash -= margin_req
                            res.open_positions[sym] = OpenPosition(
                                sym, price, qty, ts, "SHORT", strategy="orb"
                            )
                            res.trades.append(BacktestTrade(
                                sym, "SHORT", price, qty, ts, f"ORB break (range={range_pct:.1f}%)", "orb"
                            ))
                            traded_today = True
                            res.update_drawdown()

                    volumes_before.append(bar["volume"])

            # EOD flatten
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                if pos.strategy == "orb":
                    eod_price = day_bars[-1]["close"]
                    pnl = _close_position(res, sym, eod_price, day_bars[-1]["timestamp"], "EOD", "orb")
                    if day_key in daily_pnl:
                        daily_pnl[day_key] += pnl

    for day, pnl in sorted(daily_pnl.items()):
        halted_flag = pnl <= -DAILY_LOSS_LIMIT
        res.daily_stats.append(DailyStats(date=day, pnl=pnl, halted=halted_flag))
        if halted_flag:
            res.halted_days += 1

    return res


# -- Combined multi-strategy --------------------------------------------------

def combine_results(results: list[StrategyResult], name: str) -> StrategyResult:
    combo = StrategyResult(name=name)
    for r in results:
        combo.trades.extend(r.trades)
        combo.closed_pnl.extend(r.closed_pnl)
    combo.cash = STARTING_CAPITAL + sum(r.net_pnl for r in results)
    combo.halted_days = max(r.halted_days for r in results) if results else 0

    # Calculate combined drawdown from daily equity curve
    daily_pnl: dict[str, float] = defaultdict(float)
    for r in results:
        for ds in r.daily_stats:
            daily_pnl[ds.date] += ds.pnl

    equity = STARTING_CAPITAL
    peak = equity
    max_dd = 0.0
    for day in sorted(daily_pnl.keys()):
        equity += daily_pnl[day]
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    combo.max_drawdown_pct = max_dd
    combo.peak_equity = peak

    combo.daily_stats = [
        DailyStats(date=d, pnl=p) for d, p in sorted(daily_pnl.items())
    ]
    return combo


# -- Market condition analysis -------------------------------------------------

def analyze_market_conditions(spy_bars: list[dict]) -> dict:
    """Analyze SPY bars to characterize the stress period."""
    days = group_by_day(spy_bars)
    stats = {
        "total_days": len(days),
        "gap_days": 0,
        "big_move_days": 0,
        "circuit_breaker_days": 0,
        "avg_daily_range_pct": 0.0,
        "max_daily_drop_pct": 0.0,
        "max_daily_gain_pct": 0.0,
        "reversal_days": 0,  # days where open direction reversed
        "phase1_pnl": 0.0,  # crash phase
        "phase2_pnl": 0.0,  # recovery phase
        "daily_changes": [],
        "overnight_gaps": [],
    }

    prev_close = None
    for day_key in sorted(days.keys()):
        day_bars = days[day_key]
        if not day_bars:
            continue

        day_open = day_bars[0]["open"]
        day_high = max(b["high"] for b in day_bars)
        day_low = min(b["low"] for b in day_bars)
        day_close = day_bars[-1]["close"]

        daily_range_pct = (day_high - day_low) / day_low * 100 if day_low > 0 else 0
        daily_change_pct = (day_close - day_open) / day_open * 100 if day_open > 0 else 0

        stats["daily_changes"].append(daily_change_pct)

        # Overnight gap
        if prev_close:
            gap_pct = (day_open - prev_close) / prev_close * 100
            stats["overnight_gaps"].append(gap_pct)
            if abs(gap_pct) > 1.0:
                stats["gap_days"] += 1

        # Big moves (>3%)
        if abs(daily_change_pct) > 3.0:
            stats["big_move_days"] += 1

        # Circuit breaker proxy (>7% intraday decline from open)
        if day_open > 0:
            max_decline = (day_open - day_low) / day_open * 100
            if max_decline > 7.0:
                stats["circuit_breaker_days"] += 1

        stats["avg_daily_range_pct"] += daily_range_pct

        if daily_change_pct < stats["max_daily_drop_pct"]:
            stats["max_daily_drop_pct"] = daily_change_pct
        if daily_change_pct > stats["max_daily_gain_pct"]:
            stats["max_daily_gain_pct"] = daily_change_pct

        # Reversal: opened up but closed down (or vice versa)
        mid_idx = len(day_bars) // 4  # first quarter
        if mid_idx > 0:
            early_dir = day_bars[mid_idx]["close"] - day_open
            final_dir = day_close - day_open
            if (early_dir > 0 and final_dir < -day_open * 0.005) or \
               (early_dir < 0 and final_dir > day_open * 0.005):
                stats["reversal_days"] += 1

        prev_close = day_close

    if stats["total_days"] > 0:
        stats["avg_daily_range_pct"] /= stats["total_days"]

    return stats


# -- Reporting -----------------------------------------------------------------

def print_report(result: StrategyResult, trading_days: int) -> None:
    final_equity = result.cash
    ret_pct = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    daily_avg = result.net_pnl / trading_days if trading_days > 0 else 0
    monthly_avg = daily_avg * 21

    long_entries = len([t for t in result.trades if t.side == "BUY"])
    short_entries = len([t for t in result.trades if t.side == "SHORT"])
    tp_count = len([t for t in result.trades if t.side in ("SELL", "COVER") and "TP" in t.reason])
    sl_count = len([t for t in result.trades if t.side in ("SELL", "COVER") and t.reason == "SL"])
    eod_count = len([t for t in result.trades if t.side in ("SELL", "COVER") and t.reason == "EOD"])
    trail_count = len([t for t in result.trades if t.side in ("SELL", "COVER") and t.reason == "TRAIL"])

    if result.closed_pnl and len(result.closed_pnl) > 1:
        trade_rets = [p / VWAP_ORDER_SIZE_USD for p in result.closed_pnl]
        mean_r = statistics.mean(trade_rets)
        std_r = statistics.stdev(trade_rets)
        sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # Worst/best streaks
    worst_streak = 0
    current_streak = 0
    for p in result.closed_pnl:
        if p <= 0:
            current_streak += 1
            worst_streak = max(worst_streak, current_streak)
        else:
            current_streak = 0

    # Worst single day
    worst_day_pnl = min((ds.pnl for ds in result.daily_stats), default=0)
    best_day_pnl = max((ds.pnl for ds in result.daily_stats), default=0)
    losing_days = sum(1 for ds in result.daily_stats if ds.pnl < 0)
    winning_days = sum(1 for ds in result.daily_stats if ds.pnl > 0)

    bar = "=" * 70
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
    print(f"  Long Entries:         {long_entries:>6}")
    print(f"  Short Entries:        {short_entries:>6}")
    print(f"  Wins / Losses:        {result.wins:>3} / {result.losses:<3}")
    print(f"  Win Rate:             {result.win_rate:>12.1f}%")
    print(f"  Avg Win:              ${result.avg_win:>12,.2f}")
    print(f"  Avg Loss:             ${result.avg_loss:>12,.2f}")
    print(f"  Profit Factor:        {result.profit_factor:>12.2f}")
    print()
    print(f"  Exits: TP={tp_count}  SL={sl_count}  Trail={trail_count}  EOD={eod_count}")
    print()
    print(f"  Avg Daily P&L:        ${daily_avg:>12,.2f}")
    print(f"  Est. Monthly P&L:     ${monthly_avg:>12,.2f}")
    print(f"  Trading Days:         {trading_days:>6}")
    print()
    print(f"  -- Stress Metrics --")
    print(f"  Worst Losing Streak:  {worst_streak:>6} trades")
    print(f"  Worst Day P&L:        ${worst_day_pnl:>12,.2f}")
    print(f"  Best Day P&L:         ${best_day_pnl:>12,.2f}")
    print(f"  Winning Days:         {winning_days:>6}")
    print(f"  Losing Days:          {losing_days:>6}")
    print(f"  Days Halted (loss):   {result.halted_days:>6}")
    print(bar)


def print_market_conditions(stats: dict) -> None:
    bar = "=" * 70
    print()
    print(bar)
    print("  MARKET CONDITIONS ANALYSIS — COVID CRASH")
    print(bar)
    print(f"  Period:               Feb 19, 2020 – May 15, 2020")
    print(f"  Trading Days:         {stats['total_days']:>6}")
    print(f"  Avg Daily Range:      {stats['avg_daily_range_pct']:>12.2f}%  (normal: ~0.8%)")
    print(f"  Days w/ Gap >1%:      {stats['gap_days']:>6}  ({stats['gap_days']/max(stats['total_days'],1)*100:.0f}%)")
    print(f"  Days w/ Move >3%:     {stats['big_move_days']:>6}")
    print(f"  Circuit Breaker Days: {stats['circuit_breaker_days']:>6}")
    print(f"  Reversal Days:        {stats['reversal_days']:>6}  ({stats['reversal_days']/max(stats['total_days'],1)*100:.0f}%)")
    print(f"  Max Single-Day Drop:  {stats['max_daily_drop_pct']:>12.2f}%")
    print(f"  Max Single-Day Gain:  {stats['max_daily_gain_pct']:>12.2f}%")
    if stats['overnight_gaps']:
        print(f"  Avg Overnight Gap:    {statistics.mean([abs(g) for g in stats['overnight_gaps']]):>12.2f}%")
        print(f"  Max Overnight Gap:    {max(abs(g) for g in stats['overnight_gaps']):>12.2f}%")
    print(bar)


def print_weekly_breakdown(result: StrategyResult) -> None:
    """Show week-by-week P&L to see crash vs recovery performance."""
    weekly: dict[str, float] = defaultdict(float)
    for ds in result.daily_stats:
        # ISO week
        dt = datetime.strptime(ds.date, "%Y-%m-%d")
        week_key = dt.strftime("%Y-W%U")
        weekly[week_key] += ds.pnl

    print()
    print("=" * 70)
    print(f"  WEEKLY P&L BREAKDOWN — {result.name}")
    print("=" * 70)
    print(f"  {'Week':<12} {'P&L':>12}  {'Cumulative':>12}  Bar")
    print(f"  {'-'*12} {'-'*12}  {'-'*12}  {'-'*30}")

    cumulative = 0.0
    for week, pnl in sorted(weekly.items()):
        cumulative += pnl
        bar_length = int(abs(pnl) / 100)
        bar_char = "#" if pnl >= 0 else "."
        bar = bar_char * min(bar_length, 25)
        sign = "+" if pnl >= 0 else ""
        print(f"  {week:<12} ${sign}{pnl:>10,.0f}  ${cumulative:>+10,.0f}  {'+' if pnl >= 0 else '.'}{bar}")

    print("=" * 70)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print()
    print("#" * 70)
    print("#                                                                    #")
    print("#  AtoBot STRESS TEST — COVID CRASH (Feb 19 – May 15, 2020)         #")
    print("#  The WORST 3-Month Period for Day Trading Since 2010              #")
    print("#                                                                    #")
    print("#  VIX peaked at 82.69 | 4 circuit breakers | 34% crash + recovery  #")
    print("#  70% of March days had >1% overnight gaps                         #")
    print("#  Mean-reversion AND momentum strategies failed simultaneously     #")
    print("#                                                                    #")
    print("#" * 70)
    print()
    print(f"  Symbols:       {', '.join(SYMBOLS)}")
    print(f"  Capital:       ${STARTING_CAPITAL:,.0f}")
    print(f"  VWAP Size:     ${VWAP_ORDER_SIZE_USD:,.0f} | TP={VWAP_TP_PCT}% | SL={VWAP_SL_PCT}%")
    print(f"  ORB Size:      ${ORB_ORDER_SIZE_USD:,.0f} | TP={ORB_TP_PCT}% | SL={ORB_SL_PCT}%")
    print(f"  Daily Limit:   ${DAILY_LOSS_LIMIT:,.0f} | Max DD={MAX_DRAWDOWN_PCT}%")
    print(f"  Period:        {STRESS_START.strftime('%Y-%m-%d')} to {STRESS_END.strftime('%Y-%m-%d')}")
    print()

    # Fetch data
    bars_1m = fetch_bars(SYMBOLS, STRESS_START, STRESS_END)
    total_bars = sum(len(b) for b in bars_1m.values())
    if total_bars == 0:
        print("ERROR: No historical data. Check API keys in .env")
        sys.exit(1)

    all_days: set[str] = set()
    for sym_bars in bars_1m.values():
        for b in sym_bars:
            all_days.add(b["timestamp"].strftime("%Y-%m-%d"))
    trading_days = len(all_days)
    print(f"\n  Total bars: {total_bars:,} | Trading days: {trading_days}")

    bars_5m = {sym: bars_to_5min(b) for sym, b in bars_1m.items()}

    # -- Analyze market conditions -----------------------------------------
    spy_bars_5m = bars_5m.get("SPY", [])
    spy_daily = {}  # placeholder for regime data
    if spy_bars_5m:
        market_stats = analyze_market_conditions(spy_bars_5m)
        print_market_conditions(market_stats)
    else:
        market_stats = {}

    # -- Run strategies ----------------------------------------------------
    print("\n  Running strategies against COVID crash...")

    print("    VWAP Long+Short (production settings)...")
    vwap = run_vwap_strategy(bars_5m, spy_daily)

    print("    ORB Long+Short (production settings)...")
    orb = run_orb_strategy(bars_1m, spy_daily)

    # -- Combined ----------------------------------------------------------
    combined = combine_results([vwap, orb], "COMBINED: VWAP + ORB")

    # -- Reports -----------------------------------------------------------
    for r in [vwap, orb]:
        print_report(r, trading_days)

    print("\n" + "=" * 70)
    print("  COMBINED STRATEGY RESULTS")
    print("=" * 70)
    print_report(combined, trading_days)

    # -- Weekly breakdown --------------------------------------------------
    print_weekly_breakdown(combined)

    # -- Comparison table --------------------------------------------------
    print("\n")
    print("=" * 95)
    print("  STRATEGY COMPARISON TABLE — COVID STRESS TEST")
    print("=" * 95)
    print(f"  {'Strategy':<30} {'Net P&L':>12} {'Win%':>7} {'Trades':>7} {'L/S':>9} {'MaxDD':>7} {'Sharpe':>7} {'Halted':>7}")
    print(f"  {'-'*30} {'-'*12} {'-'*7} {'-'*7} {'-'*9} {'-'*7} {'-'*7} {'-'*7}")

    for r in [vwap, orb, combined]:
        longs = len([t for t in r.trades if t.side == "BUY"])
        shorts = len([t for t in r.trades if t.side == "SHORT"])
        ls_str = f"{longs}L/{shorts}S"
        if r.closed_pnl and len(r.closed_pnl) > 1:
            trade_rets = [p / VWAP_ORDER_SIZE_USD for p in r.closed_pnl]
            sharpe = (statistics.mean(trade_rets) / statistics.stdev(trade_rets)) * (252 ** 0.5) if statistics.stdev(trade_rets) > 0 else 0
        else:
            sharpe = 0
        print(f"  {r.name:<30} ${r.net_pnl:>+10,.0f} {r.win_rate:>6.1f}% {len(r.closed_pnl):>7} {ls_str:>9} {r.max_drawdown_pct:>6.1f}% {sharpe:>7.2f} {r.halted_days:>5}d")

    print("=" * 95)

    # -- Vulnerability Analysis --------------------------------------------
    print("\n")
    print("#" * 70)
    print("#  VULNERABILITY ANALYSIS — SYSTEM WEAKNESSES EXPOSED              #")
    print("#" * 70)

    vulnerabilities = []

    # 1. Win rate collapse
    for r in [vwap, orb]:
        if r.win_rate < 40:
            vulnerabilities.append(
                f"  [!] {r.name}: Win rate collapsed to {r.win_rate:.1f}% "
                f"(need >50% to be profitable)"
            )

    # 2. Massive drawdown
    if combined.max_drawdown_pct > 5.0:
        vulnerabilities.append(
            f"  [X] Max drawdown hit {combined.max_drawdown_pct:.1f}% — "
            f"exceeds 5% halt threshold -> bot would have been halted"
        )

    # 3. Daily loss limit hit frequently
    if combined.halted_days > 5:
        vulnerabilities.append(
            f"  [X] Daily loss limit ($2K) hit {combined.halted_days} times — "
            f"consider adaptive daily limits in high-vol regimes"
        )

    # 4. Stop losses too tight for COVID volatility
    sl_exits = len([t for t in combined.trades if t.reason == "SL"])
    total_exits = len([t for t in combined.trades if t.side in ("SELL", "COVER")])
    if total_exits > 0 and sl_exits / total_exits > 0.4:
        vulnerabilities.append(
            f"  [!] {sl_exits}/{total_exits} exits ({sl_exits/total_exits*100:.0f}%) "
            f"were stop-losses — SL levels too tight for extreme volatility"
        )

    # 5. Mean-reversion buying dips in crash
    crash_trades = [t for t in combined.trades
                    if t.side == "BUY" and "2020-03" in t.timestamp.strftime("%Y-%m")]
    crash_losses = [t for t in crash_trades if t.pnl < 0]
    if crash_trades:
        crash_loss_rate = len(crash_losses) / len(crash_trades) * 100
        if crash_loss_rate > 60:
            vulnerabilities.append(
                f"  [X] March 2020 LONG entries: {len(crash_losses)}/{len(crash_trades)} "
                f"were losses ({crash_loss_rate:.0f}%) — buying dips in a crash is deadly"
            )

    # 6. Short selling during recovery
    recovery_shorts = [t for t in combined.trades
                       if t.side == "SHORT" and t.timestamp >= datetime(2020, 3, 24, tzinfo=timezone.utc)]
    recovery_losses = [t for t in recovery_shorts if t.pnl < 0]
    if recovery_shorts:
        recovery_loss_rate = len(recovery_losses) / len(recovery_shorts) * 100
        if recovery_loss_rate > 60:
            vulnerabilities.append(
                f"  [!] Post-Mar 24 SHORTs: {len(recovery_losses)}/{len(recovery_shorts)} "
                f"losses ({recovery_loss_rate:.0f}%) — shorting the V-recovery was painful"
            )

    # 7. Profit factor < 1 (losing money)
    for r in [vwap, orb]:
        if r.profit_factor < 1.0:
            vulnerabilities.append(
                f"  [X] {r.name}: Profit factor {r.profit_factor:.2f} < 1.0 — "
                f"strategy LOST money during stress period"
            )

    # 8. Consecutive losses
    worst_streak = 0
    current = 0
    for p in combined.closed_pnl:
        if p <= 0:
            current += 1
            worst_streak = max(worst_streak, current)
        else:
            current = 0
    if worst_streak >= 8:
        vulnerabilities.append(
            f"  [!] Worst losing streak: {worst_streak} trades in a row — "
            f"progressive risk reduction needed"
        )

    if vulnerabilities:
        for v in vulnerabilities:
            print(v)
    else:
        print("  [OK] No critical vulnerabilities detected — system is resilient!")

    # -- Recommendations ---------------------------------------------------
    print()
    print("#" * 70)
    print("#  RECOMMENDATIONS — HOW TO HARDEN THE SYSTEM                      #")
    print("#" * 70)

    recs = []
    if combined.max_drawdown_pct > 5.0:
        recs.append(
            "  1. WIDEN STOPS IN HIGH-VOL REGIME: When VIX >30 (or ATR > 2x normal),\n"
            "     auto-widen SL to 2-3x baseline. Current 0.5% SL is noise in a 5% day."
        )
    sl_pct = sl_exits / total_exits * 100 if total_exits > 0 else 0
    if sl_pct > 40:
        recs.append(
            f"  2. REDUCE POSITION SIZE IN CRISIS: {sl_pct:.0f}% of exits were SL.\n"
            "     Scale order size to 50% when regime detector reports high volatility."
        )
    if combined.halted_days > 3:
        recs.append(
            "  3. ADAPTIVE DAILY LOSS LIMIT: In high-vol regimes, tighten the daily\n"
            "     limit to $1K (half normal) — or reduce entries per day to 3-5 max."
        )
    recs.append(
        "  4. CIRCUIT BREAKER DETECTION: If SPY drops >4% intraday, pause all\n"
        "     entries for 30 min. The bot should detect LULD halts and sit out."
    )
    recs.append(
        "  5. GAP FILTER: If overnight gap on SPY >2%, skip first 15 min of\n"
        "     trading (VWAP is meaningless until sufficient volume accumulates)."
    )
    recs.append(
        "  6. REGIME-BASED STRATEGY SELECTION: When VIX >40, disable mean-\n"
        "     reversion longs entirely. Only allow shorts or sit out."
    )

    for rec in recs:
        print(rec)

    print()
    print("#" * 70)
    print("#  STRESS TEST COMPLETE                                             #")
    print("#" * 70)
    print()


if __name__ == "__main__":
    main()
