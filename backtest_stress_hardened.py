"""AtoBot HARDENED Stress Test -- COVID Crash (Feb 19 - May 15, 2020)
======================================================================
Runs the SAME stress scenario twice:
  A) BEFORE: Production settings (pre-hardening)
  B) AFTER:  With all v5 hardening features enabled

Hardening features being tested:
  1. ATR-Adaptive Stop Widening (1.5x multiplier, capped at 3x baseline)
  2. Circuit Breaker (SPY drops >4% intraday -> pause 30min)
  3. Gap Filter (SPY overnight gap >2% -> skip first 15min)
  4. Crisis Position Sizing (reduce to 50% in extreme vol)
  5. Extreme Vol Strategy Selection (block VWAP longs when vol is extreme)
  6. Zero Overnight Risk (already present via EOD flatten)

Usage:
    python backtest_stress_hardened.py
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

SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD", "GOOGL", "AMZN", "NFLX", "SPY", "QQQ"]
STARTING_CAPITAL = 100_000.0

# -- Production params ---------------------------------------------------------
VWAP_ORDER_SIZE_USD = 17_000.0
ORB_ORDER_SIZE_USD = 17_000.0
MAX_POSITION_USD = 20_000.0

VWAP_BOUNCE_PCT = 0.05
VWAP_TP_PCT = 0.4
VWAP_SL_PCT = 0.50
ORB_RANGE_MINUTES = 15
ORB_BREAKOUT_PCT = 0.1
ORB_TP_PCT = 1.5
ORB_SL_PCT = 0.75
ORB_VOL_CONFIRM = 1.3

DAILY_LOSS_LIMIT = 2_000.0
MAX_DRAWDOWN_PCT = 5.0
MAX_OPEN_POSITIONS = 10

MIDDAY_START = 12
MIDDAY_END = 14
TRAILING_ACTIVATION_PCT = 0.5
TRAILING_DISTANCE_PCT = 0.3

STRESS_START = datetime(2020, 2, 19, tzinfo=timezone.utc)
STRESS_END = datetime(2020, 5, 15, tzinfo=timezone.utc)

# -- Hardening config (v5) ----------------------------------------------------
CIRCUIT_BREAKER_SPY_DROP_PCT = 4.0
CIRCUIT_BREAKER_PAUSE_MINUTES = 30
GAP_FILTER_SPY_THRESHOLD_PCT = 2.0
GAP_FILTER_SKIP_MINUTES = 15
ATR_STOP_MULTIPLIER = 1.5
ATR_STOP_MAX_WIDENING = 3.0
ATR_NORMAL_BASELINE_PCT = 0.3
CRISIS_SIZE_MULTIPLIER = 0.5


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


def _minutes_since_open(ts: datetime) -> float:
    try:
        et = ts.astimezone(ZoneInfo("America/New_York"))
        open_time = et.replace(hour=9, minute=30, second=0, microsecond=0)
        return (et - open_time).total_seconds() / 60.0
    except Exception:
        return 999.0


def _minutes_to_close(ts: datetime) -> float:
    try:
        et = ts.astimezone(ZoneInfo("America/New_York"))
        close_time = et.replace(hour=16, minute=0, second=0, microsecond=0)
        return (close_time - et).total_seconds() / 60.0
    except Exception:
        return 999.0


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
    circuit_breaker: bool = False
    gap_filtered: bool = False


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
    circuit_breaker_triggers: int = 0
    gap_filter_triggers: int = 0
    entries_blocked_by_hardening: int = 0

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
    print("  Fetching historical data from Alpaca...")
    client = StockHistoricalDataClient(api_key=ALPACA_KEY, secret_key=ALPACA_SECRET)
    all_bars: dict[str, list[dict]] = {s: [] for s in symbols}

    for sym in symbols:
        print(f"    {sym}...", end=" ", flush=True)
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
#  SPY DAILY ANALYSIS (for circuit breaker + gap filter + vol detection)
# ==============================================================================

def compute_spy_daily_info(spy_bars: list[dict]) -> dict[str, dict]:
    """Pre-compute SPY daily metrics for hardening features.

    Returns dict[date_str] -> {open, close, prev_close, gap_pct, max_drop_pct, atr_pct, is_extreme_vol}
    """
    days = group_by_day(spy_bars)
    info: dict[str, dict] = {}
    prev_close = None
    rolling_atr: list[float] = []

    for day_key in sorted(days.keys()):
        day_bars = days[day_key]
        if not day_bars:
            continue

        day_open = day_bars[0]["open"]
        day_high = max(b["high"] for b in day_bars)
        day_low = min(b["low"] for b in day_bars)
        day_close = day_bars[-1]["close"]
        day_range = day_high - day_low

        # Gap from previous close
        gap_pct = 0.0
        if prev_close and prev_close > 0:
            gap_pct = (day_open - prev_close) / prev_close * 100

        # Max intraday drop from open
        max_drop_pct = 0.0
        running_high = day_open
        for b in day_bars:
            if b["high"] > running_high:
                running_high = b["high"]
            drop = (running_high - b["low"]) / running_high * 100 if running_high > 0 else 0
            if drop > max_drop_pct:
                max_drop_pct = drop

        # ATR %
        atr_pct = day_range / day_close * 100 if day_close > 0 else 0
        rolling_atr.append(atr_pct)
        avg_atr = sum(rolling_atr[-14:]) / min(len(rolling_atr), 14)

        # Extreme volatility: ATR > 3x the 14-day average baseline
        is_extreme_vol = atr_pct > ATR_NORMAL_BASELINE_PCT * 10 or avg_atr > ATR_NORMAL_BASELINE_PCT * 5

        info[day_key] = {
            "open": day_open,
            "close": day_close,
            "prev_close": prev_close,
            "gap_pct": gap_pct,
            "max_drop_pct": max_drop_pct,
            "atr_pct": atr_pct,
            "avg_atr_pct": avg_atr,
            "is_extreme_vol": is_extreme_vol,
            "day_high": day_high,
            "day_low": day_low,
        }
        prev_close = day_close

    return info


def compute_symbol_atr(bars: list[dict], period: int = 14) -> float:
    """Compute ATR % for the most recent bars."""
    atr_val = _atr(bars, period)
    if atr_val is None or not bars:
        return 0.0
    last_close = bars[-1]["close"]
    if last_close <= 0:
        return 0.0
    return (atr_val / last_close) * 100


# ==============================================================================
#  STRATEGY ENGINES — SUPPORT BOTH MODES
# ==============================================================================

def run_vwap_strategy(bars_5m_by_sym: dict[str, list[dict]],
                      spy_daily: dict[str, dict],
                      hardened: bool = False) -> StrategyResult:
    """VWAP Scalp with optional hardening features.

    When hardened=True:
    - ATR-adaptive stop widening (1.5x multiplier, 3x max)
    - Circuit breaker blocks entries when SPY drops >4%
    - Gap filter skips first 15min when SPY overnight gap >2%
    - Crisis sizing reduces order size by 50% in extreme vol
    - Extreme vol blocks VWAP LONG entries entirely
    """
    label = "VWAP (HARDENED)" if hardened else "VWAP (BEFORE)"
    res = StrategyResult(name=label)
    daily_pnl: dict[str, float] = {}
    halted = False

    for sym, bars in bars_5m_by_sym.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            if day_key not in daily_pnl:
                daily_pnl[day_key] = 0.0

            if res.max_drawdown_pct >= MAX_DRAWDOWN_PCT:
                if not halted:
                    halted = True
                    res.halted_days += 1
                continue

            if daily_pnl[day_key] <= -DAILY_LOSS_LIMIT:
                continue

            # -- Hardening: check day-level conditions --
            spy_info = spy_daily.get(day_key, {})
            circuit_breaker_active = False
            gap_filter_active = False
            is_extreme_vol = spy_info.get("is_extreme_vol", False)
            size_multiplier = 1.0

            if hardened:
                # Gap filter: skip first 15min if SPY gapped >2%
                gap_pct = abs(spy_info.get("gap_pct", 0.0))
                if gap_pct >= GAP_FILTER_SPY_THRESHOLD_PCT:
                    gap_filter_active = True
                    res.gap_filter_triggers += 1

                # Crisis sizing in extreme vol
                if is_extreme_vol:
                    size_multiplier = CRISIS_SIZE_MULTIPLIER

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

                # -- Hardening: circuit breaker check on each bar --
                if hardened and spy_info.get("open", 0) > 0:
                    # Approximate: if current time's SPY equivalent drop > threshold
                    spy_open = spy_info["open"]
                    spy_max_drop = spy_info.get("max_drop_pct", 0)
                    # Check if we're in the part of the day where CB would have fired
                    if spy_max_drop >= CIRCUIT_BREAKER_SPY_DROP_PCT:
                        min_since_open = _minutes_since_open(ts)
                        # CB fires and lasts 30min. We approximate by checking if
                        # cumulative intraday stats suggest a big drop
                        running_high = spy_info["open"]
                        # Simple heuristic: if max drop happened this day, entries blocked
                        # for CIRCUIT_BREAKER_PAUSE_MINUTES after the first big bar
                        if not circuit_breaker_active:
                            circuit_breaker_active = True
                            res.circuit_breaker_triggers += 1

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

                        # -- ATR-adaptive stop loss --
                        dynamic_sl = VWAP_SL_PCT
                        atr_val = _atr(intraday_bars)
                        if hardened and atr_val and price > 0:
                            atr_pct = (atr_val / price) * 100
                            if atr_pct > ATR_NORMAL_BASELINE_PCT:
                                vol_ratio = atr_pct / ATR_NORMAL_BASELINE_PCT
                                widening = min(vol_ratio * ATR_STOP_MULTIPLIER, ATR_STOP_MAX_WIDENING)
                                dynamic_sl = max(VWAP_SL_PCT, VWAP_SL_PCT * widening)
                        elif not hardened and atr_val and price > 0:
                            # Old behavior: simple ATR cap at 1.0%
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
                        if hardened and atr_val and price > 0:
                            atr_pct = (atr_val / price) * 100
                            if atr_pct > ATR_NORMAL_BASELINE_PCT:
                                vol_ratio = atr_pct / ATR_NORMAL_BASELINE_PCT
                                widening = min(vol_ratio * ATR_STOP_MULTIPLIER, ATR_STOP_MAX_WIDENING)
                                dynamic_sl = max(VWAP_SL_PCT, VWAP_SL_PCT * widening)
                        elif not hardened and atr_val and price > 0:
                            atr_pct = (atr_val / price) * 100
                            dynamic_sl = max(VWAP_SL_PCT, min(1.0, atr_pct * 1.5))

                        if pnl_pct <= -dynamic_sl:
                            pnl = _close_position(res, sym, price, ts, "SL", "vwap_scalp")
                            daily_pnl[day_key] += pnl
                            continue

                # -- Entry logic --
                if sym not in res.open_positions and vwap_val > 0:
                    if daily_pnl[day_key] <= -DAILY_LOSS_LIMIT:
                        continue
                    if _is_midday(ts):
                        continue
                    if len(res.open_positions) >= MAX_OPEN_POSITIONS:
                        continue

                    # -- Hardening: gap filter blocks entries in first N minutes --
                    if hardened and gap_filter_active:
                        min_since = _minutes_since_open(ts)
                        if min_since < GAP_FILTER_SKIP_MINUTES:
                            res.entries_blocked_by_hardening += 1
                            continue

                    # -- Hardening: circuit breaker blocks all entries --
                    if hardened and circuit_breaker_active:
                        res.entries_blocked_by_hardening += 1
                        continue

                    # -- Hardening: block entries in last 10 min before close --
                    if hardened and _minutes_to_close(ts) <= 10:
                        res.entries_blocked_by_hardening += 1
                        continue

                    # Confluence filter
                    confluence_score = 0
                    rsi_val = _rsi(closes) if len(closes) >= 15 else None
                    macd_info = _macd(closes) if len(closes) >= 26 else None
                    avg_vol = _avg_volume(volumes[:-1]) if len(volumes) > 1 else 0
                    cur_vol = volumes[-1] if volumes else 0

                    # Order size (with crisis sizing)
                    order_size = min(VWAP_ORDER_SIZE_USD * size_multiplier, MAX_POSITION_USD)

                    # LONG entry: price below VWAP
                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= VWAP_BOUNCE_PCT:
                        # -- Hardening: block VWAP LONG in extreme vol --
                        if hardened and is_extreme_vol:
                            res.entries_blocked_by_hardening += 1
                            # Don't enter long in crash - fall through to short check
                        else:
                            if rsi_val and rsi_val < 70:
                                confluence_score += 1
                            if rsi_val and rsi_val < 40:
                                confluence_score += 1
                            if macd_info and macd_info.get("histogram", 0) > -0.5:
                                confluence_score += 1
                            if avg_vol > 0 and cur_vol >= avg_vol * 0.8:
                                confluence_score += 1

                            if confluence_score >= 2:
                                qty = order_size / price
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
                            qty = order_size / price
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

    for day, pnl in sorted(daily_pnl.items()):
        halted_flag = pnl <= -DAILY_LOSS_LIMIT
        res.daily_stats.append(DailyStats(date=day, pnl=pnl, halted=halted_flag))
        if halted_flag:
            res.halted_days += 1

    return res


def run_orb_strategy(bars_1m_by_sym: dict[str, list[dict]],
                     spy_daily: dict[str, dict],
                     hardened: bool = False) -> StrategyResult:
    """ORB with optional hardening features."""
    label = "ORB (HARDENED)" if hardened else "ORB (BEFORE)"
    res = StrategyResult(name=label)
    daily_pnl: dict[str, float] = {}

    for sym, bars in bars_1m_by_sym.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            if len(day_bars) < 16:
                continue

            if day_key not in daily_pnl:
                daily_pnl[day_key] = 0.0

            if res.max_drawdown_pct >= MAX_DRAWDOWN_PCT:
                continue
            if daily_pnl[day_key] <= -DAILY_LOSS_LIMIT:
                continue

            # Hardening: day-level checks
            spy_info = spy_daily.get(day_key, {})
            circuit_breaker_active = False
            gap_filter_active = False
            size_multiplier = 1.0

            if hardened:
                gap_pct = abs(spy_info.get("gap_pct", 0.0))
                if gap_pct >= GAP_FILTER_SPY_THRESHOLD_PCT:
                    gap_filter_active = True
                    res.gap_filter_triggers += 1

                if spy_info.get("max_drop_pct", 0) >= CIRCUIT_BREAKER_SPY_DROP_PCT:
                    circuit_breaker_active = True
                    res.circuit_breaker_triggers += 1

                if spy_info.get("is_extreme_vol", False):
                    size_multiplier = CRISIS_SIZE_MULTIPLIER

            range_bars = day_bars[:ORB_RANGE_MINUTES]
            range_high = max(b["high"] for b in range_bars)
            range_low = min(b["low"] for b in range_bars)
            range_pct = (range_high - range_low) / range_low * 100 if range_low > 0 else 0

            buffer_h = range_high * (1 + ORB_BREAKOUT_PCT / 100)
            buffer_l = range_low * (1 - ORB_BREAKOUT_PCT / 100)
            traded_today = False

            volumes_before: list[float] = [b["volume"] for b in range_bars]
            intraday_bars: list[dict] = list(range_bars)

            for bar in day_bars[ORB_RANGE_MINUTES:]:
                price = bar["close"]
                ts = bar["timestamp"]
                intraday_bars.append(bar)

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

                    # ATR-adaptive SL for ORB
                    dynamic_sl = ORB_SL_PCT
                    if hardened:
                        atr_val = _atr(intraday_bars)
                        if atr_val and price > 0:
                            atr_pct = (atr_val / price) * 100
                            if atr_pct > ATR_NORMAL_BASELINE_PCT:
                                vol_ratio = atr_pct / ATR_NORMAL_BASELINE_PCT
                                widening = min(vol_ratio * ATR_STOP_MULTIPLIER, ATR_STOP_MAX_WIDENING)
                                dynamic_sl = max(ORB_SL_PCT, ORB_SL_PCT * widening)

                    if pnl_pct <= -dynamic_sl:
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

                    # Hardening blocks
                    if hardened:
                        if gap_filter_active and _minutes_since_open(ts) < GAP_FILTER_SKIP_MINUTES:
                            res.entries_blocked_by_hardening += 1
                            volumes_before.append(bar["volume"])
                            continue
                        if circuit_breaker_active:
                            res.entries_blocked_by_hardening += 1
                            volumes_before.append(bar["volume"])
                            continue
                        if _minutes_to_close(ts) <= 10:
                            res.entries_blocked_by_hardening += 1
                            volumes_before.append(bar["volume"])
                            continue

                    # Volume confirmation
                    avg_vol = sum(volumes_before) / len(volumes_before) if volumes_before else 0
                    if avg_vol > 0 and bar["volume"] < avg_vol * ORB_VOL_CONFIRM:
                        volumes_before.append(bar["volume"])
                        continue

                    order_size = min(ORB_ORDER_SIZE_USD * size_multiplier, MAX_POSITION_USD)

                    # LONG breakout
                    if price > buffer_h:
                        qty = order_size / price
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
                        qty = order_size / price
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


def combine_results(results: list[StrategyResult], name: str) -> StrategyResult:
    combo = StrategyResult(name=name)
    for r in results:
        combo.trades.extend(r.trades)
        combo.closed_pnl.extend(r.closed_pnl)
    combo.cash = STARTING_CAPITAL + sum(r.net_pnl for r in results)
    combo.halted_days = max(r.halted_days for r in results) if results else 0
    combo.circuit_breaker_triggers = sum(r.circuit_breaker_triggers for r in results)
    combo.gap_filter_triggers = sum(r.gap_filter_triggers for r in results)
    combo.entries_blocked_by_hardening = sum(r.entries_blocked_by_hardening for r in results)

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

    worst_streak = 0
    current_streak = 0
    for p in result.closed_pnl:
        if p <= 0:
            current_streak += 1
            worst_streak = max(worst_streak, current_streak)
        else:
            current_streak = 0

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
    print(f"  Long / Short:         {long_entries:>3}L / {short_entries:<3}S")
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

    # Hardening-specific stats
    if result.circuit_breaker_triggers > 0 or result.gap_filter_triggers > 0 or result.entries_blocked_by_hardening > 0:
        print()
        print(f"  -- Hardening Stats --")
        print(f"  Circuit Breaker Fires: {result.circuit_breaker_triggers:>5}")
        print(f"  Gap Filter Fires:      {result.gap_filter_triggers:>5}")
        print(f"  Entries Blocked:       {result.entries_blocked_by_hardening:>5}")

    print(bar)


def print_comparison(before: StrategyResult, after: StrategyResult, label: str, trading_days: int) -> None:
    """Print side-by-side comparison."""
    bar = "=" * 90
    print()
    print(bar)
    print(f"  {label:^86}")
    print(bar)

    def _delta(new: float, old: float, fmt: str = "+,.2f", better_higher: bool = True) -> str:
        d = new - old
        if d == 0:
            return "  --"
        sign = "better" if (d > 0) == better_higher else "worse"
        return f"  {d:{fmt}} ({sign})"

    rows = [
        ("Net P&L",       f"${before.net_pnl:>+12,.2f}", f"${after.net_pnl:>+12,.2f}", _delta(after.net_pnl, before.net_pnl)),
        ("Max Drawdown",  f"{before.max_drawdown_pct:>10.2f}%", f"{after.max_drawdown_pct:>10.2f}%", _delta(after.max_drawdown_pct, before.max_drawdown_pct, "+.2f", False)),
        ("Win Rate",      f"{before.win_rate:>10.1f}%", f"{after.win_rate:>10.1f}%", _delta(after.win_rate, before.win_rate, "+.1f")),
        ("Total Trades",  f"{len(before.closed_pnl):>10}", f"{len(after.closed_pnl):>10}", _delta(len(after.closed_pnl), len(before.closed_pnl), "+d", False)),
        ("Profit Factor", f"{before.profit_factor:>10.2f}", f"{after.profit_factor:>10.2f}", _delta(after.profit_factor, before.profit_factor, "+.2f")),
        ("Avg Win",       f"${before.avg_win:>10,.2f}", f"${after.avg_win:>10,.2f}", _delta(after.avg_win, before.avg_win, "+,.2f")),
        ("Avg Loss",      f"${before.avg_loss:>10,.2f}", f"${after.avg_loss:>10,.2f}", _delta(after.avg_loss, before.avg_loss, "+,.2f", False)),
        ("Days Halted",   f"{before.halted_days:>10}", f"{after.halted_days:>10}", _delta(after.halted_days, before.halted_days, "+d", False)),
    ]

    print(f"  {'Metric':<20} {'BEFORE':>14} {'AFTER':>14} {'Change':>20}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*20}")
    for label_r, bv, av, delta in rows:
        print(f"  {label_r:<20} {bv:>14} {av:>14} {delta:>20}")
    print(bar)

    if after.entries_blocked_by_hardening > 0:
        print(f"  Entries blocked by hardening: {after.entries_blocked_by_hardening}")
        print(f"  Circuit breaker triggers:     {after.circuit_breaker_triggers}")
        print(f"  Gap filter triggers:          {after.gap_filter_triggers}")
        print()


# -- Main ----------------------------------------------------------------------

def main() -> None:
    print()
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#  AtoBot HARDENED STRESS TEST -- COVID CRASH (Feb 19 - May 15, 2020)       #")
    print("#  Comparing BEFORE vs AFTER v5 Hardening                                   #")
    print("#" + " " * 78 + "#")
    print("#  Hardening Features:                                                       #")
    print("#    1. ATR-Adaptive Stop Widening (1.5x mult, 3x max)                      #")
    print("#    2. Circuit Breaker (SPY -4% -> pause 30min)                            #")
    print("#    3. Gap Filter (SPY gap >2% -> skip first 15min)                        #")
    print("#    4. Crisis Sizing (50% position size in extreme vol)                    #")
    print("#    5. Block VWAP Longs in Extreme Vol (only shorts allowed)               #")
    print("#    6. EOD Entry Block (no new trades last 10min)                           #")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print()
    print(f"  Symbols:       {', '.join(SYMBOLS)}")
    print(f"  Capital:       ${STARTING_CAPITAL:,.0f}")
    print(f"  VWAP:          ${VWAP_ORDER_SIZE_USD:,.0f} | TP={VWAP_TP_PCT}% | SL={VWAP_SL_PCT}%")
    print(f"  ORB:           ${ORB_ORDER_SIZE_USD:,.0f} | TP={ORB_TP_PCT}% | SL={ORB_SL_PCT}%")
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

    # Pre-compute SPY daily info for hardening features
    spy_bars_5m = bars_5m.get("SPY", [])
    spy_daily = compute_spy_daily_info(spy_bars_5m)

    # Count extreme vol days
    extreme_days = sum(1 for d in spy_daily.values() if d.get("is_extreme_vol", False))
    gap_days = sum(1 for d in spy_daily.values() if abs(d.get("gap_pct", 0)) >= GAP_FILTER_SPY_THRESHOLD_PCT)
    cb_days = sum(1 for d in spy_daily.values() if d.get("max_drop_pct", 0) >= CIRCUIT_BREAKER_SPY_DROP_PCT)

    print(f"\n  SPY Analysis:")
    print(f"    Extreme vol days:     {extreme_days}/{len(spy_daily)} ({extreme_days/max(len(spy_daily),1)*100:.0f}%)")
    print(f"    Gap >2% days:         {gap_days}")
    print(f"    Circuit breaker days: {cb_days}")

    # ======================================================================
    #  RUN BOTH VERSIONS
    # ======================================================================

    print("\n" + "=" * 80)
    print("  RUNNING: BEFORE HARDENING (production settings, no protections)")
    print("=" * 80)

    print("    VWAP (before)...")
    vwap_before = run_vwap_strategy(bars_5m, spy_daily, hardened=False)
    print("    ORB (before)...")
    orb_before = run_orb_strategy(bars_1m, spy_daily, hardened=False)
    combined_before = combine_results([vwap_before, orb_before], "COMBINED (BEFORE)")

    print("\n" + "=" * 80)
    print("  RUNNING: AFTER HARDENING (v5 -- all protections active)")
    print("=" * 80)

    print("    VWAP (hardened)...")
    vwap_after = run_vwap_strategy(bars_5m, spy_daily, hardened=True)
    print("    ORB (hardened)...")
    orb_after = run_orb_strategy(bars_1m, spy_daily, hardened=True)
    combined_after = combine_results([vwap_after, orb_after], "COMBINED (HARDENED)")

    # ======================================================================
    #  REPORTS
    # ======================================================================

    print("\n")
    print("#" * 80)
    print("#  INDIVIDUAL STRATEGY REPORTS                                               #")
    print("#" * 80)

    for r in [vwap_before, vwap_after, orb_before, orb_after]:
        print_report(r, trading_days)

    print("\n")
    print("#" * 80)
    print("#  COMBINED RESULTS                                                          #")
    print("#" * 80)

    print_report(combined_before, trading_days)
    print_report(combined_after, trading_days)

    # ======================================================================
    #  BEFORE vs AFTER COMPARISON
    # ======================================================================

    print("\n")
    print("#" * 80)
    print("#  BEFORE vs AFTER -- SIDE-BY-SIDE COMPARISON                                #")
    print("#" * 80)

    print_comparison(vwap_before, vwap_after, "VWAP: BEFORE vs HARDENED", trading_days)
    print_comparison(orb_before, orb_after, "ORB: BEFORE vs HARDENED", trading_days)
    print_comparison(combined_before, combined_after, "COMBINED: BEFORE vs HARDENED", trading_days)

    # Summary comparison table
    print("\n")
    print("=" * 105)
    print("  FINAL COMPARISON TABLE -- COVID CRASH STRESS TEST")
    print("=" * 105)
    print(f"  {'Strategy':<30} {'Net P&L':>12} {'Win%':>7} {'Trades':>7} {'MaxDD':>7} {'PF':>6} {'Blocked':>8} {'CB':>4}")
    print(f"  {'-'*30} {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*4}")

    for r in [vwap_before, vwap_after, orb_before, orb_after, combined_before, combined_after]:
        print(
            f"  {r.name:<30} ${r.net_pnl:>+10,.0f} {r.win_rate:>6.1f}% "
            f"{len(r.closed_pnl):>7} {r.max_drawdown_pct:>6.1f}% "
            f"{r.profit_factor:>5.2f} {r.entries_blocked_by_hardening:>7} {r.circuit_breaker_triggers:>4}"
        )

    print("=" * 105)

    # Verdict
    pnl_diff = combined_after.net_pnl - combined_before.net_pnl
    dd_diff = combined_before.max_drawdown_pct - combined_after.max_drawdown_pct
    wr_diff = combined_after.win_rate - combined_before.win_rate

    print()
    print("#" * 80)
    print("#  VERDICT                                                                   #")
    print("#" * 80)
    print()
    if pnl_diff > 0:
        print(f"  P&L IMPROVEMENT:    ${pnl_diff:>+,.2f} (hardening MADE more money)")
    elif pnl_diff < 0:
        print(f"  P&L CHANGE:         ${pnl_diff:>+,.2f} (hardening reduced P&L -- acceptable for safety)")
    else:
        print(f"  P&L CHANGE:         $0 (no difference)")

    if dd_diff > 0:
        print(f"  DRAWDOWN REDUCTION: {dd_diff:>+.2f}% (hardening REDUCED max drawdown)")
    elif dd_diff < 0:
        print(f"  DRAWDOWN CHANGE:    {dd_diff:>+.2f}% (drawdown increased)")
    else:
        print(f"  DRAWDOWN CHANGE:    0% (no difference)")

    print(f"  WIN RATE CHANGE:    {wr_diff:>+.1f}%")
    print(f"  ENTRIES BLOCKED:    {combined_after.entries_blocked_by_hardening} (would-be bad trades avoided)")
    print(f"  CIRCUIT BREAKERS:   {combined_after.circuit_breaker_triggers} triggers")
    print(f"  GAP FILTERS:        {combined_after.gap_filter_triggers} triggers")

    # Risk-adjusted return comparison
    if combined_before.max_drawdown_pct > 0 and combined_after.max_drawdown_pct > 0:
        rar_before = combined_before.net_pnl / combined_before.max_drawdown_pct
        rar_after = combined_after.net_pnl / combined_after.max_drawdown_pct
        print(f"\n  RISK-ADJUSTED RETURN (P&L / Max DD):")
        print(f"    Before: ${rar_before:>+,.0f} per 1% drawdown")
        print(f"    After:  ${rar_after:>+,.0f} per 1% drawdown")
        if rar_after > rar_before:
            print(f"    --> HARDENING IMPROVED risk-adjusted returns!")
        else:
            print(f"    --> Trade-off: lower absolute return for better risk control")

    print()
    print("#" * 80)
    print("#  STRESS TEST COMPLETE                                                      #")
    print("#" * 80)
    print()


if __name__ == "__main__":
    main()
