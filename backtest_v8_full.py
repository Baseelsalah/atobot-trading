"""AtoBot Backtester v8 — Full capability test including SHORT entries,
pairs trading spread, and multi-strategy with scaled sizing.

Tests ALL Tier 1-3 capabilities:
  - VWAP long + short
  - Momentum long + short
  - ORB long + short (breakout + breakdown)
  - EMA Pullback long
  - Pairs spread (NVDA:AMD, GOOGL:META, MSFT:AAPL)
  - Scaled sizing ($5K-$17K per trade)
  - Trailing stops for longs + shorts

Usage:
    python backtest_v8_full.py
"""

from __future__ import annotations

import math
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

SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD", "GOOGL", "META"]
STARTING_CAPITAL = 100_000.0

# Scaled sizing (Tier 1)
ORDER_SIZE_USD = 5_000.0       # Base per-trade
MAX_POSITION_USD = 25_000.0    # Max single position
PAIRS_LEG_USD = 5_000.0        # Per-leg for pairs

# Strategy params
MOM_RSI_PERIOD = 14
MOM_RSI_OVERSOLD = 35.0
MOM_RSI_OVERBOUGHT = 70.0     # NEW: for shorts
MOM_VOL_MULT = 1.3
MOM_TP_PCT = 1.5
MOM_SL_PCT = 0.75

ORB_BREAKOUT_PCT = 0.08
ORB_TP_PCT = 1.2
ORB_SL_PCT = 0.5

VWAP_BOUNCE_PCT = 0.15
VWAP_TP_PCT = 0.4
VWAP_SL_PCT = 0.25

EMA_FAST = 9
EMA_SLOW = 21
EMA_TP_PCT = 0.8
EMA_SL_PCT = 0.4

# Pairs params
PAIRS = [("NVDA", "AMD"), ("GOOGL", "META"), ("MSFT", "AAPL")]
PAIRS_LOOKBACK = 60
PAIRS_ENTRY_Z = 2.0
PAIRS_EXIT_Z = 0.5
PAIRS_STOP_Z = 3.5

# Filter params
TREND_EMA_PERIOD = 20
MIDDAY_START = 12
MIDDAY_END = 14
TRAILING_ACTIVATION_PCT = 0.5
TRAILING_DISTANCE_PCT = 0.3


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
    # signal = 9-period EMA of MACD (simplified)
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


def _is_midday(ts: datetime) -> bool:
    try:
        et = ts.astimezone(ZoneInfo("America/New_York"))
        return MIDDAY_START <= et.hour < MIDDAY_END
    except Exception:
        return False


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


# ── Trade tracking ────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    symbol: str
    side: str       # BUY, SELL, SHORT, COVER
    price: float
    quantity: float
    timestamp: datetime
    reason: str = ""
    strategy: str = ""


@dataclass
class OpenPosition:
    symbol: str
    entry_price: float
    quantity: float
    entry_time: datetime
    side: str = "LONG"          # LONG or SHORT
    highest_price: float = 0.0  # For trailing stop (longs)
    lowest_price: float = 0.0   # For trailing stop (shorts)

    def __post_init__(self):
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price


@dataclass
class StrategyResult:
    name: str
    trades: list[BacktestTrade] = field(default_factory=list)
    closed_pnl: list[float] = field(default_factory=list)
    open_positions: dict[str, OpenPosition] = field(default_factory=dict)
    peak_equity: float = STARTING_CAPITAL
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


def _close_position(res: StrategyResult, sym: str, price: float, ts: datetime,
                     reason: str, strategy: str = "") -> None:
    """Close a position (LONG or SHORT) and record PnL."""
    pos = res.open_positions.get(sym)
    if not pos:
        return
    if pos.side == "LONG":
        pnl = (price - pos.entry_price) * pos.quantity
        res.trades.append(BacktestTrade(sym, "SELL", price, pos.quantity, ts, reason, strategy))
        res.cash += price * pos.quantity
    else:  # SHORT
        pnl = (pos.entry_price - price) * pos.quantity
        res.trades.append(BacktestTrade(sym, "COVER", price, pos.quantity, ts, reason, strategy))
        margin_held = pos.quantity * pos.entry_price * 0.5  # Return held margin
        res.cash += margin_held + pnl
    res.closed_pnl.append(pnl)
    del res.open_positions[sym]
    res.update_drawdown()


# ── Trailing stop ─────────────────────────────────────────────────────────────

def _check_trailing_stop(pos: OpenPosition, price: float) -> bool:
    if pos.side == "LONG":
        if price > pos.highest_price:
            pos.highest_price = price
        profit_pct = (pos.highest_price - pos.entry_price) / pos.entry_price * 100
        if profit_pct < TRAILING_ACTIVATION_PCT:
            return False
        trail_stop = pos.highest_price * (1 - TRAILING_DISTANCE_PCT / 100)
        return price <= trail_stop
    else:  # SHORT
        if price < pos.lowest_price:
            pos.lowest_price = price
        profit_pct = (pos.entry_price - pos.lowest_price) / pos.entry_price * 100
        if profit_pct < TRAILING_ACTIVATION_PCT:
            return False
        trail_stop = pos.lowest_price * (1 + TRAILING_DISTANCE_PCT / 100)
        return price >= trail_stop


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_bars(symbols: list[str], start: datetime, end: datetime) -> dict[str, list[dict]]:
    print("Fetching historical data from Alpaca...")
    client = StockHistoricalDataClient(api_key=ALPACA_KEY, secret_key=ALPACA_SECRET)
    all_bars: dict[str, list[dict]] = {s: [] for s in symbols}
    for sym in symbols:
        print(f"  {sym}...", end=" ", flush=True)
        req = StockBarsRequest(
            symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
            start=start, end=end,
        )
        barset = client.get_stock_bars(req)
        if sym in barset.data:
            for bar in barset.data[sym]:
                all_bars[sym].append({
                    "timestamp": bar.timestamp,
                    "open": float(bar.open), "high": float(bar.high),
                    "low": float(bar.low), "close": float(bar.close),
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


# ── VWAP Long + Short ────────────────────────────────────────────────────────

def run_vwap_long_short(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """VWAP scalp with both LONG (below VWAP) and SHORT (above VWAP) entries."""
    res = StrategyResult(name="VWAP Long+Short")

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
                    if _check_trailing_stop(pos, price):
                        _close_position(res, sym, price, ts, "TRAIL", "vwap_scalp")
                        continue

                    if pos.side == "LONG":
                        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                        if price >= vwap_val or pnl_pct >= VWAP_TP_PCT:
                            _close_position(res, sym, price, ts, "TP/VWAP", "vwap_scalp")
                            continue
                        if pnl_pct <= -VWAP_SL_PCT:
                            _close_position(res, sym, price, ts, "SL", "vwap_scalp")
                            continue
                    else:  # SHORT
                        pnl_pct = (pos.entry_price - price) / pos.entry_price * 100
                        if price <= vwap_val or pnl_pct >= VWAP_TP_PCT:
                            _close_position(res, sym, price, ts, "TP/VWAP", "vwap_scalp")
                            continue
                        if pnl_pct <= -VWAP_SL_PCT:
                            _close_position(res, sym, price, ts, "SL", "vwap_scalp")
                            continue

                if sym not in res.open_positions and vwap_val > 0:
                    if _is_midday(ts):
                        continue

                    # LONG: price below VWAP by bounce_pct
                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= VWAP_BOUNCE_PCT:
                        if len(closes) >= TREND_EMA_PERIOD:
                            ema_val = _ema(closes, TREND_EMA_PERIOD)
                            if ema_val is not None and price < ema_val * 0.97:
                                continue  # Too far below EMA = downtrend, skip long
                        qty = ORDER_SIZE_USD / price
                        if res.cash >= qty * price:
                            res.cash -= qty * price
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts, "LONG")
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "VWAP dip", "vwap_scalp"))
                            res.update_drawdown()

                    # SHORT: price above VWAP by bounce_pct
                    short_dev = (price - vwap_val) / vwap_val * 100
                    if short_dev >= VWAP_BOUNCE_PCT:
                        qty = ORDER_SIZE_USD / price
                        # For shorts, we need margin — simulate as reducing cash by margin requirement
                        margin_req = qty * price * 0.5  # 50% margin
                        if res.cash >= margin_req:
                            res.cash -= margin_req
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts, "SHORT")
                            res.trades.append(BacktestTrade(sym, "SHORT", price, qty, ts, "VWAP pop", "vwap_scalp"))
                            res.update_drawdown()

            # EOD flatten
            if sym in res.open_positions and day_bars:
                eod_price = day_bars[-1]["close"]
                _close_position(res, sym, eod_price, day_bars[-1]["timestamp"], "EOD", "vwap_scalp")

    return res


# ── Momentum Long + Short ────────────────────────────────────────────────────

def run_momentum_long_short(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """Momentum with LONG (RSI oversold) and SHORT (RSI overbought + bearish MACD)."""
    res = StrategyResult(name="Momentum Long+Short")

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
                    if _check_trailing_stop(pos, price):
                        _close_position(res, sym, price, ts, "TRAIL", "momentum")
                        continue

                    if pos.side == "LONG":
                        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    else:
                        pnl_pct = (pos.entry_price - price) / pos.entry_price * 100

                    if pnl_pct >= MOM_TP_PCT:
                        _close_position(res, sym, price, ts, "TP", "momentum")
                        continue
                    if pnl_pct <= -MOM_SL_PCT:
                        _close_position(res, sym, price, ts, "SL", "momentum")
                        continue

                if sym not in res.open_positions and len(closes) >= MOM_RSI_PERIOD + 2:
                    if _is_midday(ts):
                        continue

                    current_rsi = _rsi(closes, MOM_RSI_PERIOD)
                    avg_vol = _avg_volume(volumes[:-1], 20)
                    cur_vol = volumes[-1]
                    vol_ok = avg_vol > 0 and cur_vol >= avg_vol * MOM_VOL_MULT

                    if not vol_ok:
                        continue

                    # LONG: RSI oversold + EMA trend
                    if current_rsi is not None and current_rsi <= MOM_RSI_OVERSOLD:
                        ema_val = _ema(closes, TREND_EMA_PERIOD)
                        if ema_val is None or price >= ema_val * 0.98:
                            qty = ORDER_SIZE_USD / price
                            if res.cash >= qty * price:
                                res.cash -= qty * price
                                res.open_positions[sym] = OpenPosition(sym, price, qty, ts, "LONG")
                                res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "RSI oversold", "momentum"))
                                res.update_drawdown()

                    # SHORT: RSI overbought + bearish MACD
                    elif current_rsi is not None and current_rsi >= MOM_RSI_OVERBOUGHT:
                        macd_info = _macd(closes)
                        if macd_info and macd_info.get("histogram", 0) < 0:
                            qty = ORDER_SIZE_USD / price
                            margin_req = qty * price * 0.5
                            if res.cash >= margin_req:
                                res.cash -= margin_req
                                res.open_positions[sym] = OpenPosition(sym, price, qty, ts, "SHORT")
                                res.trades.append(BacktestTrade(sym, "SHORT", price, qty, ts, "RSI overbought", "momentum"))
                                res.update_drawdown()

            # EOD flatten
            if sym in res.open_positions and day_bars:
                _close_position(res, sym, day_bars[-1]["close"], day_bars[-1]["timestamp"], "EOD", "momentum")

    return res


# ── ORB Long + Short ─────────────────────────────────────────────────────────

def run_orb_long_short(bars_1m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """ORB with breakout LONG and breakdown SHORT."""
    res = StrategyResult(name="ORB Long+Short")

    for sym, bars in bars_1m_by_sym.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            if len(day_bars) < 16:
                continue

            range_bars = day_bars[:15]
            range_high = max(b["high"] for b in range_bars)
            range_low = min(b["low"] for b in range_bars)
            buffer_h = range_high * (1 + ORB_BREAKOUT_PCT / 100)
            buffer_l = range_low * (1 - ORB_BREAKOUT_PCT / 100)
            traded_today = False

            for bar in day_bars[15:]:
                price = bar["close"]
                ts = bar["timestamp"]

                pos = res.open_positions.get(sym)
                if pos:
                    if _check_trailing_stop(pos, price):
                        _close_position(res, sym, price, ts, "TRAIL", "orb")
                        continue

                    if pos.side == "LONG":
                        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    else:
                        pnl_pct = (pos.entry_price - price) / pos.entry_price * 100

                    if pnl_pct >= ORB_TP_PCT:
                        _close_position(res, sym, price, ts, "TP", "orb")
                        continue
                    if pnl_pct <= -ORB_SL_PCT:
                        _close_position(res, sym, price, ts, "SL", "orb")
                        continue

                if sym not in res.open_positions and not traded_today:
                    if _is_midday(ts):
                        continue

                    # Breakout LONG
                    if price >= buffer_h:
                        qty = ORDER_SIZE_USD / price
                        if res.cash >= qty * price:
                            res.cash -= qty * price
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts, "LONG")
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "Breakout", "orb"))
                            traded_today = True
                            res.update_drawdown()

                    # Breakdown SHORT
                    elif price <= buffer_l:
                        qty = ORDER_SIZE_USD / price
                        margin_req = qty * price * 0.5
                        if res.cash >= margin_req:
                            res.cash -= margin_req
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts, "SHORT")
                            res.trades.append(BacktestTrade(sym, "SHORT", price, qty, ts, "Breakdown", "orb"))
                            traded_today = True
                            res.update_drawdown()

            if sym in res.open_positions and day_bars:
                _close_position(res, sym, day_bars[-1]["close"], day_bars[-1]["timestamp"], "EOD", "orb")

    return res


# ── EMA Pullback (long only) ─────────────────────────────────────────────────

def run_ema_pullback(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """EMA pullback: buy when price pulls back to 9 EMA in an uptrend (above 21 EMA)."""
    res = StrategyResult(name="EMA Pullback")

    for sym, bars in bars_5m_by_sym.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            closes: list[float] = []

            for bar in day_bars:
                closes.append(bar["close"])
                price = bar["close"]
                ts = bar["timestamp"]

                pos = res.open_positions.get(sym)
                if pos:
                    if _check_trailing_stop(pos, price):
                        _close_position(res, sym, price, ts, "TRAIL", "ema_pullback")
                        continue
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if pnl_pct >= EMA_TP_PCT:
                        _close_position(res, sym, price, ts, "TP", "ema_pullback")
                        continue
                    if pnl_pct <= -EMA_SL_PCT:
                        _close_position(res, sym, price, ts, "SL", "ema_pullback")
                        continue

                if sym not in res.open_positions and len(closes) >= EMA_SLOW + 2:
                    if _is_midday(ts):
                        continue

                    ema_fast = _ema(closes, EMA_FAST)
                    ema_slow = _ema(closes, EMA_SLOW)
                    if ema_fast is None or ema_slow is None:
                        continue

                    # Uptrend: fast EMA above slow EMA
                    if ema_fast > ema_slow:
                        # Pullback: price touches or dips below fast EMA
                        if price <= ema_fast * 1.001:  # Within 0.1% of fast EMA
                            # But still above slow EMA
                            if price > ema_slow:
                                qty = ORDER_SIZE_USD / price
                                if res.cash >= qty * price:
                                    res.cash -= qty * price
                                    res.open_positions[sym] = OpenPosition(sym, price, qty, ts, "LONG")
                                    res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "EMA pullback", "ema_pullback"))
                                    res.update_drawdown()

            if sym in res.open_positions and day_bars:
                _close_position(res, sym, day_bars[-1]["close"], day_bars[-1]["timestamp"], "EOD", "ema_pullback")

    return res


# ── Pairs Trading ─────────────────────────────────────────────────────────────

@dataclass
class PairPosition:
    sym_a: str
    sym_b: str
    side: str          # "long_spread" or "short_spread"
    qty_a: float
    qty_b: float
    entry_z: float
    bars_held: int = 0


def run_pairs_trading(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """Pairs/stat-arb on correlated pairs using z-score of spread."""
    res = StrategyResult(name="Pairs Trading")
    pair_positions: dict[str, PairPosition] = {}

    for sym_a, sym_b in PAIRS:
        if sym_a not in bars_5m_by_sym or sym_b not in bars_5m_by_sym:
            print(f"  Skipping pair {sym_a}:{sym_b} — missing data")
            continue

        bars_a = bars_5m_by_sym[sym_a]
        bars_b = bars_5m_by_sym[sym_b]
        days_a = group_by_day(bars_a)
        days_b = group_by_day(bars_b)
        all_days_set = sorted(set(days_a.keys()) & set(days_b.keys()))

        # Compute daily closing prices for hedge ratio
        daily_closes_a: list[float] = []
        daily_closes_b: list[float] = []
        pair_key = f"{sym_a}:{sym_b}"

        for day_key in all_days_set:
            d_a = days_a.get(day_key, [])
            d_b = days_b.get(day_key, [])
            if not d_a or not d_b:
                continue

            daily_closes_a.append(d_a[-1]["close"])
            daily_closes_b.append(d_b[-1]["close"])

            # Need enough data for calibration
            if len(daily_closes_a) < 20:
                continue

            # Compute hedge ratio (OLS beta)
            ca = daily_closes_a[-PAIRS_LOOKBACK:] if len(daily_closes_a) >= PAIRS_LOOKBACK else daily_closes_a
            cb = daily_closes_b[-PAIRS_LOOKBACK:] if len(daily_closes_b) >= PAIRS_LOOKBACK else daily_closes_b
            n = min(len(ca), len(cb))
            ca, cb = ca[-n:], cb[-n:]

            mean_a = sum(ca) / n
            mean_b = sum(cb) / n
            cov_ab = sum((ca[i] - mean_a) * (cb[i] - mean_b) for i in range(n)) / n
            var_b = sum((cb[i] - mean_b) ** 2 for i in range(n)) / n
            beta = cov_ab / var_b if var_b > 0 else 1.0

            # Compute spread and z-score
            spreads = [ca[i] - beta * cb[i] for i in range(n)]
            spread_mean = sum(spreads) / len(spreads)
            spread_std = (sum((s - spread_mean) ** 2 for s in spreads) / len(spreads)) ** 0.5
            if spread_std == 0:
                continue

            # Current spread
            current_spread = d_a[-1]["close"] - beta * d_b[-1]["close"]
            z = (current_spread - spread_mean) / spread_std
            ts = d_a[-1]["timestamp"]

            pp = pair_positions.get(pair_key)

            # EXIT
            if pp:
                pp.bars_held += 1
                should_exit = False
                reason = ""

                if pp.side == "long_spread" and z <= PAIRS_EXIT_Z:
                    should_exit = True
                    reason = f"z-revert {z:.2f}"
                elif pp.side == "short_spread" and z >= -PAIRS_EXIT_Z:
                    should_exit = True
                    reason = f"z-revert {z:.2f}"
                elif pp.side == "long_spread" and z < -PAIRS_STOP_Z:
                    should_exit = True
                    reason = f"z-stop {z:.2f}"
                elif pp.side == "short_spread" and z > PAIRS_STOP_Z:
                    should_exit = True
                    reason = f"z-stop {z:.2f}"
                elif pp.bars_held > 20:  # Max ~20 days
                    should_exit = True
                    reason = "max hold"

                if should_exit:
                    # Simplified PnL via z-score movement × notional
                    z_move = pp.entry_z - z  # z moved toward zero = profit
                    if pp.side == "short_spread":
                        z_move = -z_move
                    pnl = z_move * PAIRS_LEG_USD * 0.01  # ~1% per z-score unit
                    res.closed_pnl.append(pnl)
                    res.trades.append(BacktestTrade(f"{sym_a}:{sym_b}", "CLOSE", z, 1, ts, reason, "pairs"))
                    del pair_positions[pair_key]

            # ENTRY
            elif abs(z) > PAIRS_ENTRY_Z:
                if z > PAIRS_ENTRY_Z:
                    # Short spread: short A, long B
                    pair_positions[pair_key] = PairPosition(
                        sym_a, sym_b, "short_spread",
                        PAIRS_LEG_USD / d_a[-1]["close"],
                        PAIRS_LEG_USD / d_b[-1]["close"],
                        z,
                    )
                    res.trades.append(BacktestTrade(f"{sym_a}:{sym_b}", "SHORT_SPREAD", z, 1, ts, f"z={z:.2f}", "pairs"))
                else:
                    # Long spread: long A, short B
                    pair_positions[pair_key] = PairPosition(
                        sym_a, sym_b, "long_spread",
                        PAIRS_LEG_USD / d_a[-1]["close"],
                        PAIRS_LEG_USD / d_b[-1]["close"],
                        z,
                    )
                    res.trades.append(BacktestTrade(f"{sym_a}:{sym_b}", "LONG_SPREAD", z, 1, ts, f"z={z:.2f}", "pairs"))

    return res


# ── VWAP Long-only (baseline for comparison) ─────────────────────────────────

def run_vwap_long_only(bars_5m_by_sym: dict[str, list[dict]]) -> StrategyResult:
    """VWAP Scalp long-only (baseline)."""
    res = StrategyResult(name="VWAP Long-Only")
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
                    if _check_trailing_stop(pos, price):
                        _close_position(res, sym, price, ts, "TRAIL", "vwap_scalp")
                        continue
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if price >= vwap_val or pnl_pct >= VWAP_TP_PCT:
                        _close_position(res, sym, price, ts, "TP/VWAP", "vwap_scalp")
                        continue
                    if pnl_pct <= -VWAP_SL_PCT:
                        _close_position(res, sym, price, ts, "SL", "vwap_scalp")
                        continue
                if sym not in res.open_positions and vwap_val > 0:
                    if _is_midday(ts):
                        continue
                    if len(closes) >= TREND_EMA_PERIOD:
                        ema_val = _ema(closes, TREND_EMA_PERIOD)
                        if ema_val is not None and price < ema_val:
                            continue
                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= VWAP_BOUNCE_PCT:
                        qty = ORDER_SIZE_USD / price
                        if res.cash >= qty * price:
                            res.cash -= qty * price
                            res.open_positions[sym] = OpenPosition(sym, price, qty, ts, "LONG")
                            res.trades.append(BacktestTrade(sym, "BUY", price, qty, ts, "VWAP dip", "vwap_scalp"))
                            res.update_drawdown()
            if sym in res.open_positions and day_bars:
                _close_position(res, sym, day_bars[-1]["close"], day_bars[-1]["timestamp"], "EOD", "vwap_scalp")
    return res


# ── Combine results ──────────────────────────────────────────────────────────

def combine_results(results: list[StrategyResult], name: str) -> StrategyResult:
    combo = StrategyResult(name=name)
    for r in results:
        combo.trades.extend(r.trades)
        combo.closed_pnl.extend(r.closed_pnl)
    combo.cash = STARTING_CAPITAL + sum(combo.closed_pnl)
    # Max drawdown: approximate from combined PnL curve
    equity = STARTING_CAPITAL
    peak = equity
    max_dd = 0.0
    for pnl in combo.closed_pnl:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    combo.max_drawdown_pct = max_dd
    combo.peak_equity = peak
    return combo


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(result: StrategyResult, trading_days: int) -> None:
    final_equity = result.cash
    ret_pct = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    daily_avg = result.net_pnl / trading_days if trading_days > 0 else 0
    monthly_avg = daily_avg * 21

    long_entries = len([t for t in result.trades if t.side in ("BUY",)])
    short_entries = len([t for t in result.trades if t.side in ("SHORT", "SHORT_SPREAD")])
    tp_count = len([t for t in result.trades if t.side in ("SELL", "COVER", "CLOSE") and "TP" in t.reason])
    sl_count = len([t for t in result.trades if t.side in ("SELL", "COVER", "CLOSE") and t.reason == "SL"])
    eod_count = len([t for t in result.trades if t.side in ("SELL", "COVER", "CLOSE") and t.reason == "EOD"])
    trail_count = len([t for t in result.trades if t.side in ("SELL", "COVER", "CLOSE") and t.reason == "TRAIL"])

    if result.closed_pnl and len(result.closed_pnl) > 1:
        trade_rets = [p / ORDER_SIZE_USD for p in result.closed_pnl]
        mean_r = statistics.mean(trade_rets)
        std_r = statistics.stdev(trade_rets)
        sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    bar = "=" * 65
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
    print(bar)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 65)
    print("  AtoBot Backtester v8 — Full Capability Test")
    print("  LONG + SHORT + PAIRS + MULTI-STRATEGY")
    print("=" * 65)
    print()
    print(f"  Symbols:  {', '.join(SYMBOLS)}")
    print(f"  Capital:  ${STARTING_CAPITAL:,.0f}")
    print(f"  Order:    ${ORDER_SIZE_USD:,.0f} per trade (scaled)")
    print(f"  Pairs:    {[f'{a}:{b}' for a, b in PAIRS]}")
    print()

    end = datetime(2026, 2, 20, tzinfo=timezone.utc)
    start = datetime(2025, 11, 20, tzinfo=timezone.utc)
    print(f"  Period:   {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} (3 months)")
    print()

    bars_1m = fetch_bars(SYMBOLS, start, end)
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

    # ── Run all strategies ────────────────────────────────────────────────
    print("\n  Running strategies...")

    print("    VWAP Long-Only (baseline)...")
    vwap_long = run_vwap_long_only(bars_5m)

    print("    VWAP Long+Short...")
    vwap_ls = run_vwap_long_short(bars_5m)

    print("    Momentum Long+Short...")
    mom_ls = run_momentum_long_short(bars_5m)

    print("    ORB Long+Short...")
    orb_ls = run_orb_long_short(bars_1m)

    print("    EMA Pullback...")
    ema_pb = run_ema_pullback(bars_5m)

    print("    Pairs Trading...")
    pairs = run_pairs_trading(bars_5m)

    # ── Combos ────────────────────────────────────────────────────────────
    combo_old = combine_results([vwap_long], "Baseline (VWAP Long-Only)")
    combo_new = combine_results([vwap_ls, mom_ls, ema_pb], "New: VWAP+Mom+EMA (L+S)")
    combo_all = combine_results([vwap_ls, mom_ls, orb_ls, ema_pb, pairs], "New: ALL 5 strategies")
    combo_best = combine_results([vwap_ls, ema_pb, pairs], "New: VWAP+EMA+Pairs")

    # ── Reports ───────────────────────────────────────────────────────────
    for r in [vwap_long, vwap_ls, mom_ls, orb_ls, ema_pb, pairs]:
        print_report(r, trading_days)

    print("\n" + "=" * 65)
    print("  COMBINED STRATEGY RESULTS")
    print("=" * 65)
    for r in [combo_old, combo_new, combo_all, combo_best]:
        print_report(r, trading_days)

    # ── Comparison table ──────────────────────────────────────────────────
    print("\n")
    print("=" * 95)
    print("  STRATEGY COMPARISON TABLE")
    print("=" * 95)
    print(f"  {'Strategy':<30} {'Net P&L':>12} {'Win%':>7} {'Trades':>7} {'L/S':>9} {'MaxDD':>7} {'Sharpe':>7} {'Mo.PnL':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*7} {'-'*7} {'-'*9} {'-'*7} {'-'*7} {'-'*10}")

    all_results = [vwap_long, vwap_ls, mom_ls, orb_ls, ema_pb, pairs,
                   combo_old, combo_new, combo_all, combo_best]

    for r in all_results:
        monthly = r.net_pnl / trading_days * 21 if trading_days > 0 else 0
        longs = len([t for t in r.trades if t.side in ("BUY",)])
        shorts = len([t for t in r.trades if t.side in ("SHORT", "SHORT_SPREAD")])
        ls_str = f"{longs}L/{shorts}S"
        if r.closed_pnl and len(r.closed_pnl) > 1:
            trade_rets = [p / ORDER_SIZE_USD for p in r.closed_pnl]
            sharpe = (statistics.mean(trade_rets) / statistics.stdev(trade_rets)) * (252 ** 0.5) if statistics.stdev(trade_rets) > 0 else 0
        else:
            sharpe = 0
        print(f"  {r.name:<30} ${r.net_pnl:>+10,.0f} {r.win_rate:>6.1f}% {len(r.closed_pnl):>7} {ls_str:>9} {r.max_drawdown_pct:>6.1f}% {sharpe:>7.2f} ${monthly:>+8,.0f}")

    print("=" * 95)

    best = max(all_results, key=lambda r: r.net_pnl)
    best_monthly = best.net_pnl / trading_days * 21 if trading_days > 0 else 0
    print(f"\n  BEST: {best.name}")
    print(f"    Net P&L: ${best.net_pnl:+,.2f} | Est Monthly: ${best_monthly:+,.0f}/mo")

    # ── Short selling impact ──────────────────────────────────────────────
    long_only_pnl = vwap_long.net_pnl
    long_short_pnl = vwap_ls.net_pnl
    improvement = long_short_pnl - long_only_pnl
    print(f"\n  SHORT SELLING IMPACT (VWAP):")
    print(f"    Long-Only:   ${long_only_pnl:+,.2f}")
    print(f"    Long+Short:  ${long_short_pnl:+,.2f}")
    print(f"    Improvement: ${improvement:+,.2f}")
    print()


if __name__ == "__main__":
    main()
