#!/usr/bin/env python3
"""
AtoBot Ultra Backtester — Full-Stack Validation
=================================================
Runs Baseline → Improved → ULTRA side-by-side comparison.

Ultra features integrated:
  1. Confluence Score gating (indicators_advanced)
  2. ML Win-Probability gating (ml_features)
  3. Kelly Criterion position sizing (position_sizer)
  4. Bracket orders — partial TP1 / TP2 / trailing
  5. MACD death-cross early exit
  6. Strategy selector adaptive weights & cooldowns
  7. Portfolio heat tracking
  8. Trade journal analytics (Sharpe, Sortino, grade)

Usage:
  python backtest_ultra.py --period 1m     # last 1 month
  python backtest_ultra.py --period 3m     # last 3 months
  python backtest_ultra.py                 # runs both
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Alpaca SDK ────────────────────────────────────────────────────────────────
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ALPACA_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET", "")

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD",
          "META", "GOOGL", "AMZN", "AVGO", "NFLX",
          "SPY", "QQQ", "CRM", "UBER", "MU"]
STARTING_CAPITAL = 100_000.0
ORDER_SIZE_USD = 17_000.0          # Fixed size for baseline / improved
MAX_ORDER_FRACTION = 0.10          # 10% cap per trade (for Kelly)

# Strategy params (same as backtest_v2)
RSI_OVERSOLD = 35
VOL_MULTIPLIER = 1.3
MOM_TP_PCT = 1.5
MOM_SL_PCT = 0.75

ORB_BREAKOUT_PCT = 0.08
ORB_TP_PCT = 1.2
ORB_SL_PCT = 0.5

VWAP_BOUNCE_PCT = 0.12
VWAP_TP_PCT = 0.4
VWAP_SL_PCT = 0.25

# EMA Pullback strategy params
EMA_PB_FAST_PERIOD = 9           # Fast EMA (signal line)
EMA_PB_SLOW_PERIOD = 21          # Slow EMA (momentum filter)
EMA_PB_TREND_PERIOD = 50         # Trend EMA (direction)
EMA_PB_PULLBACK_PCT = 0.15      # Max % distance from fast EMA for entry
EMA_PB_RSI_LOW = 35              # RSI pullback zone lower bound
EMA_PB_RSI_HIGH = 55             # RSI pullback zone upper bound
EMA_PB_RSI_EXIT = 70             # RSI overbought exit threshold
EMA_PB_VOL_MULT = 1.2            # Volume confirmation multiplier
EMA_PB_TP_PCT = 1.5              # Take profit percent
EMA_PB_SL_PCT = 0.75             # Stop loss percent

# Improvement layer params
TREND_EMA_PERIOD = 20
MIDDAY_START = 12
MIDDAY_END = 14
TRAILING_ACTIVATION_PCT = 0.5
TRAILING_DISTANCE_PCT = 0.3

# Ultra layer params
CONFLUENCE_MIN = 40             # Minimum confluence score to enter
WIN_PROB_MIN = 0.40             # Minimum ML win-prob to enter
KELLY_FRACTION = 0.5            # Half-Kelly (conservative)
BRACKET_TP1_PCT = 1.5           # Partial TP #1 (close 50%)
BRACKET_TP2_PCT = 3.0           # Final TP #2
MACD_EXIT_ENABLED = True        # MACD death-cross early exit
MAX_PORTFOLIO_HEAT = 0.06       # 6% total risk
MAX_CONSECUTIVE_LOSSES = 5      # Cooldown trigger
COOLDOWN_BARS = 60              # ~5 hours of 5-min bars


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol: str
    side: str
    price: float
    qty: float
    ts: datetime
    reason: str
    strategy: str = ""


@dataclass
class Position:
    symbol: str
    entry_price: float
    quantity: float
    ts: datetime
    highest_price: float = 0.0
    partial_taken: bool = False   # For bracket TP1
    strategy: str = ""
    stop_loss: float = 0.0       # For Kelly risk calc

    def __post_init__(self):
        self.highest_price = self.entry_price


@dataclass
class Result:
    name: str
    cash: float = STARTING_CAPITAL
    closed_pnl: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    open_positions: dict = field(default_factory=dict)
    max_drawdown_pct: float = 0.0
    _peak_equity: float = STARTING_CAPITAL

    @property
    def net_pnl(self) -> float:
        return sum(self.closed_pnl)

    @property
    def wins(self) -> int:
        return sum(1 for p in self.closed_pnl if p > 0)

    @property
    def losses(self) -> int:
        return sum(1 for p in self.closed_pnl if p <= 0)

    @property
    def win_rate(self) -> float:
        t = len(self.closed_pnl)
        return (self.wins / t * 100) if t else 0

    @property
    def avg_win(self) -> float:
        w = [p for p in self.closed_pnl if p > 0]
        return statistics.mean(w) if w else 0

    @property
    def avg_loss(self) -> float:
        l_ = [p for p in self.closed_pnl if p <= 0]
        return statistics.mean(l_) if l_ else 0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(p for p in self.closed_pnl if p > 0)
        gross_loss = abs(sum(p for p in self.closed_pnl if p <= 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def update_drawdown(self):
        equity = self.cash + sum(
            pos.entry_price * pos.quantity
            for pos in self.open_positions.values()
        )
        if equity > self._peak_equity:
            self._peak_equity = equity
        dd = (self._peak_equity - equity) / self._peak_equity * 100
        self.max_drawdown_pct = max(self.max_drawdown_pct, dd)


# ── Indicator helpers (lightweight, inline) ───────────────────────────────────

def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    ema_val = sum(values[:period]) / period
    for v in values[period:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val


class VWAPTracker:
    """Incremental VWAP calculator — O(1) per bar instead of O(n)."""
    __slots__ = ('cum_pv', 'cum_v')
    def __init__(self):
        self.cum_pv = 0.0
        self.cum_v = 0.0
    def add(self, bar: dict) -> float | None:
        tp = (bar["high"] + bar["low"] + bar["close"]) / 3
        self.cum_pv += tp * bar["volume"]
        self.cum_v += bar["volume"]
        return self.cum_pv / self.cum_v if self.cum_v > 0 else None
    def reset(self):
        self.cum_pv = 0.0
        self.cum_v = 0.0


def _vwap(bars: list[dict]) -> float | None:
    """Legacy VWAP from full bar list (used by baseline only)."""
    if not bars:
        return None
    cum_pv, cum_v = 0.0, 0.0
    for b in bars:
        tp = (b["high"] + b["low"] + b["close"]) / 3
        cum_pv += tp * b["volume"]
        cum_v += b["volume"]
    return cum_pv / cum_v if cum_v > 0 else None


def _avg_volume(bars: list[dict], n: int = 20) -> float:
    vols = [b["volume"] for b in bars[-n:]]
    return sum(vols) / len(vols) if vols else 1.0


def _is_midday(ts: datetime) -> bool:
    h = ts.hour if ts.tzinfo else ts.hour
    return MIDDAY_START <= h < MIDDAY_END


def _macd(closes: list[float]) -> dict | None:
    """Compute MACD line, signal, histogram — single-pass O(n)."""
    if len(closes) < 35:
        return None
    k12 = 2.0 / 13
    k26 = 2.0 / 27
    # Init EMAs
    ema12 = sum(closes[:12]) / 12
    ema26 = sum(closes[:26]) / 26
    # Advance ema12 through indices 12..25
    for v in closes[12:26]:
        ema12 = v * k12 + ema12 * (1 - k12)
    # Single pass from index 26 onward, collect last 10 MACD values
    collect_start = max(26, len(closes) - 10)
    macd_series: list[float] = []
    for i in range(26, len(closes)):
        v = closes[i]
        ema12 = v * k12 + ema12 * (1 - k12)
        ema26 = v * k26 + ema26 * (1 - k26)
        if i >= collect_start:
            macd_series.append(ema12 - ema26)
    macd_line = ema12 - ema26
    if len(macd_series) < 2:
        return {"macd": macd_line, "signal": 0, "histogram": macd_line}
    sig_period = min(9, len(macd_series))
    signal = _ema(macd_series, sig_period) or (sum(macd_series) / len(macd_series))
    prev_signal = signal
    if len(macd_series) > 2:
        ps = _ema(macd_series[:-1], min(9, len(macd_series) - 1))
        if ps is not None:
            prev_signal = ps
    return {
        "macd": macd_line,
        "signal": signal,
        "histogram": macd_line - signal,
        "death_cross": macd_line < signal and macd_series[-2] >= prev_signal,
    }


def _check_trailing_stop(pos: Position, price: float) -> bool:
    if price > pos.highest_price:
        pos.highest_price = price
    gain_pct = (pos.highest_price - pos.entry_price) / pos.entry_price * 100
    if gain_pct >= TRAILING_ACTIVATION_PCT:
        drop = (pos.highest_price - price) / pos.highest_price * 100
        if drop >= TRAILING_DISTANCE_PCT:
            return True
    return False


# ── Confluence & ML scoring (simplified for backtest) ─────────────────────────

def _compute_confluence(closes: list[float], volumes: list[float],
                        highs: list[float], lows: list[float]) -> int:
    """Lightweight confluence score (0-100) for backtest use."""
    if len(closes) < 30:
        return 50  # neutral default

    score = 0
    max_score = 0

    # RSI (weight 15)
    rsi = _rsi(closes)
    max_score += 15
    if rsi is not None:
        if 30 < rsi < 70:
            score += 10
        elif rsi <= 30:
            score += 15  # Oversold = buy opportunity
        # Overbought = 0

    # EMA trend alignment (weight 15)
    max_score += 15
    ema9 = _ema(closes, 9)
    ema21 = _ema(closes, 21)
    if ema9 is not None and ema21 is not None:
        if closes[-1] > ema9 > ema21:
            score += 15  # Full trend alignment
        elif closes[-1] > ema21:
            score += 8

    # Volume surge (weight 10)
    max_score += 10
    avg_vol = sum(volumes[-20:]) / max(len(volumes[-20:]), 1)
    if avg_vol > 0:
        rvol = volumes[-1] / avg_vol
        if rvol >= 2.0:
            score += 10
        elif rvol >= 1.0:
            score += 5

    # MACD (weight 15)
    max_score += 15
    m = _macd(closes)
    if m:
        if m["histogram"] > 0:
            score += 10
        if m["macd"] > 0:
            score += 5

    # Price vs VWAP-like moving average (weight 10)
    max_score += 10
    sma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
    if closes[-1] > sma20:
        score += 10
    elif closes[-1] > sma20 * 0.995:
        score += 5

    # Candle pattern (weight 10)
    max_score += 10
    if len(closes) >= 3 and closes[-1] > closes[-2] > closes[-3]:
        score += 10  # Three rising bars
    elif len(closes) >= 2 and closes[-1] > closes[-2]:
        score += 5

    # Normalize to 0-100
    return min(100, int(score / max_score * 100)) if max_score > 0 else 50


def _compute_win_probability(closes: list[float], volumes: list[float]) -> float:
    """Lightweight win probability estimate for backtest (0.0 - 1.0)."""
    if len(closes) < 20:
        return 0.5

    score = 0.0
    total = 0

    # RSI sweet spot
    rsi = _rsi(closes)
    total += 15
    if rsi is not None:
        if 30 < rsi < 60:
            score += 15
        elif rsi < 25:
            score += 10
        elif rsi > 75:
            score -= 10

    # MACD momentum
    total += 20
    m = _macd(closes)
    if m:
        if m["histogram"] > 0:
            score += 12
        if m["macd"] > 0:
            score += 8

    # Trend alignment
    total += 15
    ema9 = _ema(closes, 9)
    ema21 = _ema(closes, 21)
    if ema9 is not None and ema21 is not None:
        if closes[-1] > ema9 and ema9 > ema21:
            score += 15

    # Volume
    total += 10
    avg_vol = sum(volumes[-20:]) / max(len(volumes[-20:]), 1)
    if avg_vol > 0:
        rvol = volumes[-1] / avg_vol
        if rvol >= 2.0:
            score += 10
        elif rvol >= 1.0:
            score += 5
        elif rvol < 0.5:
            score -= 5

    # Recent momentum
    total += 10
    ret5 = (closes[-1] / closes[-6] - 1) * 100 if len(closes) > 6 else 0
    if 0 < ret5 < 3:
        score += 10
    elif ret5 > 3:
        score += 5  # Might be extended

    prob = max(0.0, min(1.0, (score / total + 1) / 2))
    return round(prob, 4)


def _compute_atr(highs: list[float], lows: list[float],
                 closes: list[float], period: int = 14) -> float:
    """Compute ATR from price lists. Returns 0.0 if insufficient data."""
    if len(highs) < period + 1:
        return 0.0
    true_ranges: list[float] = []
    for i in range(-period, 0):
        h, l = highs[i], lows[i]
        prev_c = closes[i - 1]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        true_ranges.append(tr)
    return sum(true_ranges) / len(true_ranges)


def _kelly_size(win_rate: float, avg_win: float, avg_loss: float,
                account: float, entry_price: float,
                stop_loss_pct: float) -> float:
    """Kelly Criterion position sizing (half-Kelly).

    Returns dollar amount to risk.
    """
    if win_rate <= 0 or avg_win <= 0 or avg_loss >= 0:
        return ORDER_SIZE_USD  # Fallback to fixed

    b = avg_win / abs(avg_loss)  # Win/loss ratio
    p = win_rate / 100 if win_rate > 1 else win_rate
    q = 1 - p

    kelly_pct = (b * p - q) / b
    kelly_pct = max(0.0, kelly_pct)

    # Apply half-Kelly
    applied = kelly_pct * KELLY_FRACTION

    # Clamp to max position fraction
    applied = min(applied, MAX_ORDER_FRACTION)

    # Minimum 1% to stay active
    applied = max(0.01, applied)

    dollar_size = account * applied
    # Clamp to reasonable range
    dollar_size = max(1000, min(dollar_size, account * MAX_ORDER_FRACTION))
    return dollar_size


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_bars(symbols: list[str], start: datetime,
               end: datetime) -> dict[str, list[dict]]:
    """Fetch 1-minute bars from Alpaca."""
    client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    all_bars: dict[str, list[dict]] = {}

    for sym in symbols:
        print(f"    Fetching {sym}...", end=" ", flush=True)
        req = StockBarsRequest(
            symbol_or_symbols=sym,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )
        resp = client.get_stock_bars(req)
        bars = []
        if hasattr(resp, "data") and sym in resp.data:
            raw = resp.data[sym]
        elif hasattr(resp, "dict"):
            raw = resp.dict().get(sym, [])
        else:
            raw = resp.get(sym, []) if isinstance(resp, dict) else []

        for b in raw:
            if hasattr(b, "open"):
                bars.append({
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume),
                    "timestamp": b.timestamp,
                })
            elif isinstance(b, dict):
                bars.append({
                    "open": float(b["open"]),
                    "high": float(b["high"]),
                    "low": float(b["low"]),
                    "close": float(b["close"]),
                    "volume": float(b["volume"]),
                    "timestamp": b["timestamp"],
                })
        all_bars[sym] = bars
        print(f"{len(bars):,} bars")

    return all_bars


def bars_to_5min(bars_1m: list[dict]) -> list[dict]:
    """Aggregate 1-min bars into 5-min bars."""
    result = []
    for i in range(0, len(bars_1m) - 4, 5):
        chunk = bars_1m[i:i + 5]
        result.append({
            "open": chunk[0]["open"],
            "high": max(b["high"] for b in chunk),
            "low": min(b["low"] for b in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(b["volume"] for b in chunk),
            "timestamp": chunk[-1]["timestamp"],
        })
    return result


def group_by_day(bars: list[dict]) -> dict[str, list[dict]]:
    """Group bars by trading date."""
    days: dict[str, list[dict]] = {}
    for b in bars:
        key = b["timestamp"].strftime("%Y-%m-%d")
        days.setdefault(key, []).append(b)
    return days


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY RUNNERS
# ══════════════════════════════════════════════════════════════════════════════


# ── Momentum ──────────────────────────────────────────────────────────────────

def run_momentum_baseline(bars_5m: dict[str, list[dict]]) -> Result:
    """Momentum strategy — baseline (no filters, fixed exits)."""
    res = Result(name="Momentum (baseline)")
    for sym, bars in bars_5m.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            closes: list[float] = []
            for bar in day_bars:
                closes.append(bar["close"])
                price, ts = bar["close"], bar["timestamp"]
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if pnl_pct >= MOM_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP", "momentum"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                    if pnl_pct <= -MOM_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "SL", "momentum"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                if sym not in res.open_positions and len(closes) >= 15:
                    rsi = _rsi(closes)
                    avg_vol = _avg_volume(day_bars[:len(closes)])
                    if rsi is not None and rsi <= RSI_OVERSOLD:
                        if bar["volume"] >= avg_vol * VOL_MULTIPLIER:
                            qty = ORDER_SIZE_USD / price
                            if res.cash >= qty * price:
                                res.cash -= qty * price
                                res.open_positions[sym] = Position(sym, price, qty, ts, strategy="momentum")
                                res.trades.append(Trade(sym, "BUY", price, qty, ts, "RSI+Vol", "momentum"))
                                res.update_drawdown()
            # EOD flatten
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                ep = day_bars[-1]["close"]
                pnl = (ep - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += ep * pos.quantity
                res.trades.append(Trade(sym, "SELL", ep, pos.quantity, day_bars[-1]["timestamp"], "EOD", "momentum"))
                del res.open_positions[sym]
                res.update_drawdown()
    return res


def run_momentum_improved(bars_5m: dict[str, list[dict]]) -> Result:
    """Momentum + EMA filter + midday filter + trailing stop."""
    res = Result(name="Momentum (improved)")
    for sym, bars in bars_5m.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            closes: list[float] = []
            for bar in day_bars:
                closes.append(bar["close"])
                price, ts = bar["close"], bar["timestamp"]
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if _check_trailing_stop(pos, price):
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TRAIL", "momentum"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                    if pnl_pct >= MOM_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP", "momentum"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                    if pnl_pct <= -MOM_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "SL", "momentum"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                if sym not in res.open_positions and len(closes) >= 15:
                    rsi = _rsi(closes)
                    avg_vol = _avg_volume(day_bars[:len(closes)])
                    if rsi is not None and rsi <= RSI_OVERSOLD:
                        if bar["volume"] >= avg_vol * VOL_MULTIPLIER:
                            # EMA trend filter
                            if len(closes) >= TREND_EMA_PERIOD:
                                ema_val = _ema(closes, TREND_EMA_PERIOD)
                                if ema_val is not None and price < ema_val:
                                    continue
                            # Midday filter
                            if _is_midday(ts):
                                continue
                            qty = ORDER_SIZE_USD / price
                            if res.cash >= qty * price:
                                res.cash -= qty * price
                                res.open_positions[sym] = Position(sym, price, qty, ts, strategy="momentum")
                                res.trades.append(Trade(sym, "BUY", price, qty, ts, "RSI+Vol+Filt", "momentum"))
                                res.update_drawdown()
            # EOD flatten
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                ep = day_bars[-1]["close"]
                pnl = (ep - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += ep * pos.quantity
                res.trades.append(Trade(sym, "SELL", ep, pos.quantity, day_bars[-1]["timestamp"], "EOD", "momentum"))
                del res.open_positions[sym]
                res.update_drawdown()
    return res


def run_momentum_ultra(bars_5m: dict[str, list[dict]]) -> Result:
    """Momentum ULTRA v2 -- looser entry, bigger targets, better R:R.

    Key fixes from v1:
    - Looser RSI threshold (32 vs 35) for more opportunities
    - Wider TP (2.0% vs 1.5%) to let winners run
    - Wider SL (1.0% vs 0.75%) for more breathing room (R:R stays 2:1)
    - Lower confluence/win-prob gates for more trades
    - Bracket TP2 at 3.5% to let runners really run
    - Fixed buggy consecutive_losses reset
    """
    res = Result(name="Momentum (ULTRA)")
    # Track running stats for Kelly
    strategy_wins = 0
    strategy_losses = 0
    strategy_total_win = 0.0
    strategy_total_loss = 0.0
    cooldown_counter: dict[str, int] = {}
    consecutive_losses: dict[str, int] = {s: 0 for s in SYMBOLS}

    # V2 strategy-specific params
    rsi_oversold = 32      # Looser RSI (was 35)
    tp_pct = 2.0           # Bigger TP (was 1.5)
    sl_pct = 1.0           # More room (was 0.75), keeps 2:1 R:R
    bracket_tp1 = 1.5      # Partial close level
    bracket_tp2 = 3.5      # Let runners really run (was 3.0)
    conf_min = 30          # Lower confluence gate (was 40)
    wp_min = 0.35          # Lower win-prob gate (was 0.40)

    for sym, bars in bars_5m.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            closes: list[float] = []
            volumes: list[float] = []
            highs: list[float] = []
            lows: list[float] = []

            for bar in day_bars:
                closes.append(bar["close"])
                volumes.append(bar["volume"])
                highs.append(bar["high"])
                lows.append(bar["low"])
                price, ts = bar["close"], bar["timestamp"]

                # Cooldown management
                if sym in cooldown_counter:
                    cooldown_counter[sym] -= 1
                    if cooldown_counter[sym] <= 0:
                        del cooldown_counter[sym]

                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

                    # MACD death-cross early exit (OK on 5-min bars)
                    if MACD_EXIT_ENABLED and len(closes) >= 35:
                        m = _macd(closes)
                        if m and m.get("death_cross"):
                            pnl = (price - pos.entry_price) * pos.quantity
                            res.closed_pnl.append(pnl)
                            res.cash += price * pos.quantity
                            res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "MACD_EXIT", "momentum"))
                            del res.open_positions[sym]
                            _update_kelly_stats(pnl)
                            res.update_drawdown()
                            continue

                    # Trailing stop
                    if _check_trailing_stop(pos, price):
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TRAIL", "momentum"))
                        del res.open_positions[sym]
                        _update_kelly_stats(pnl)
                        res.update_drawdown()
                        continue

                    # Bracket TP1 — partial close
                    if not pos.partial_taken and pnl_pct >= bracket_tp1:
                        half_qty = pos.quantity * 0.5
                        pnl_partial = (price - pos.entry_price) * half_qty
                        res.closed_pnl.append(pnl_partial)
                        res.cash += price * half_qty
                        pos.quantity -= half_qty
                        pos.partial_taken = True
                        res.trades.append(Trade(sym, "SELL", price, half_qty, ts, "TP1(50%)", "momentum"))
                        _update_kelly_stats(pnl_partial)
                        res.update_drawdown()
                        continue

                    # Bracket TP2 — full close after partial
                    if pos.partial_taken and pnl_pct >= bracket_tp2:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP2", "momentum"))
                        del res.open_positions[sym]
                        _update_kelly_stats(pnl)
                        res.update_drawdown()
                        continue

                    # Standard TP (if no partial yet)
                    if not pos.partial_taken and pnl_pct >= tp_pct:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP", "momentum"))
                        del res.open_positions[sym]
                        _update_kelly_stats(pnl)
                        res.update_drawdown()
                        continue

                    # Stop loss (wider: 1.0%)
                    if pnl_pct <= -sl_pct:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "SL", "momentum"))
                        del res.open_positions[sym]
                        consecutive_losses[sym] = consecutive_losses.get(sym, 0) + 1
                        if consecutive_losses[sym] >= MAX_CONSECUTIVE_LOSSES:
                            cooldown_counter[sym] = COOLDOWN_BARS
                            consecutive_losses[sym] = 0
                        _update_kelly_stats(pnl)
                        res.update_drawdown()
                        continue

                # Entry logic with Ultra gates
                if sym not in res.open_positions and len(closes) >= 15:
                    # Skip if in cooldown
                    if sym in cooldown_counter:
                        continue

                    rsi = _rsi(closes)
                    avg_vol = _avg_volume(day_bars[:len(closes)])
                    if rsi is not None and rsi <= rsi_oversold:
                        if bar["volume"] >= avg_vol * VOL_MULTIPLIER:
                            # EMA trend filter
                            if len(closes) >= TREND_EMA_PERIOD:
                                ema_val = _ema(closes, TREND_EMA_PERIOD)
                                if ema_val is not None and price < ema_val:
                                    continue
                            # Midday filter
                            if _is_midday(ts):
                                continue

                            # ULTRA GATE 1: Confluence score
                            conf = _compute_confluence(closes, volumes, highs, lows)
                            if conf < conf_min:
                                continue

                            # ULTRA GATE 2: ML Win probability
                            win_prob = _compute_win_probability(closes, volumes)
                            if win_prob < wp_min:
                                continue

                            # ULTRA GATE 3: Portfolio heat check
                            current_heat = sum(
                                abs(p.entry_price * p.quantity - p.stop_loss * p.quantity)
                                for p in res.open_positions.values()
                            ) / max(res.cash, 1)
                            if current_heat >= MAX_PORTFOLIO_HEAT:
                                continue

                            # Kelly position sizing
                            wr = strategy_wins / max(strategy_wins + strategy_losses, 1)
                            aw = strategy_total_win / max(strategy_wins, 1)
                            al = strategy_total_loss / max(strategy_losses, 1)
                            order_usd = _kelly_size(wr, aw, al, res.cash, price, sl_pct)

                            # Progressive risk scaling (v5: reduce after losses)
                            cl = consecutive_losses.get(sym, 0)
                            if cl > 0:
                                prog_mult = max(0.25, 0.75 ** cl)
                                order_usd *= prog_mult

                            qty = order_usd / price
                            cost = qty * price

                            if res.cash >= cost:
                                res.cash -= cost
                                sl_price = price * (1 - sl_pct / 100)
                                res.open_positions[sym] = Position(
                                    sym, price, qty, ts, strategy="momentum",
                                    stop_loss=sl_price
                                )
                                res.trades.append(Trade(sym, "BUY", price, qty, ts,
                                                        f"ULTRA(c={conf},wp={win_prob:.0%})", "momentum"))
                                consecutive_losses[sym] = 0
                                res.update_drawdown()

            # EOD flatten
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                ep = day_bars[-1]["close"]
                pnl = (ep - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += ep * pos.quantity
                res.trades.append(Trade(sym, "SELL", ep, pos.quantity, day_bars[-1]["timestamp"], "EOD", "momentum"))
                del res.open_positions[sym]
                _update_kelly_stats(pnl)
                res.update_drawdown()

    return res


# ── VWAP Strategy ─────────────────────────────────────────────────────────────

def run_vwap_baseline(bars_5m: dict[str, list[dict]]) -> Result:
    res = Result(name="VWAP (baseline)")
    for sym, bars in bars_5m.items():
        days = group_by_day(bars)
        vwap_tracker = VWAPTracker()
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            vwap_tracker.reset()
            for bar in day_bars:
                price, ts = bar["close"], bar["timestamp"]
                vwap_val = vwap_tracker.add(bar)
                if vwap_val is None:
                    continue
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if price >= vwap_val or pnl_pct >= VWAP_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP/VWAP", "vwap"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                    if pnl_pct <= -VWAP_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "SL", "vwap"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                if sym not in res.open_positions and vwap_val > 0:
                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= VWAP_BOUNCE_PCT:
                        qty = ORDER_SIZE_USD / price
                        if res.cash >= qty * price:
                            res.cash -= qty * price
                            res.open_positions[sym] = Position(sym, price, qty, ts, strategy="vwap")
                            res.trades.append(Trade(sym, "BUY", price, qty, ts, "VWAP dip", "vwap"))
                            res.update_drawdown()
            # EOD
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                ep = day_bars[-1]["close"]
                pnl = (ep - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += ep * pos.quantity
                res.trades.append(Trade(sym, "SELL", ep, pos.quantity, day_bars[-1]["timestamp"], "EOD", "vwap"))
                del res.open_positions[sym]
                res.update_drawdown()
    return res


def run_vwap_improved(bars_5m: dict[str, list[dict]]) -> Result:
    res = Result(name="VWAP (improved)")
    for sym, bars in bars_5m.items():
        days = group_by_day(bars)
        vwap_tracker = VWAPTracker()
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            vwap_tracker.reset()
            closes: list[float] = []
            for bar in day_bars:
                closes.append(bar["close"])
                price, ts = bar["close"], bar["timestamp"]
                vwap_val = vwap_tracker.add(bar)
                if vwap_val is None:
                    continue
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if _check_trailing_stop(pos, price):
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TRAIL", "vwap"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                    if price >= vwap_val or pnl_pct >= VWAP_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP/VWAP", "vwap"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                    if pnl_pct <= -VWAP_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "SL", "vwap"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                if sym not in res.open_positions and vwap_val > 0:
                    # EMA filter
                    if len(closes) >= TREND_EMA_PERIOD:
                        ema_val = _ema(closes, TREND_EMA_PERIOD)
                        if ema_val is not None and price < ema_val:
                            continue
                    if _is_midday(ts):
                        continue
                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= VWAP_BOUNCE_PCT:
                        qty = ORDER_SIZE_USD / price
                        if res.cash >= qty * price:
                            res.cash -= qty * price
                            res.open_positions[sym] = Position(sym, price, qty, ts, strategy="vwap")
                            res.trades.append(Trade(sym, "BUY", price, qty, ts, "VWAP+Filt", "vwap"))
                            res.update_drawdown()
            # EOD
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                ep = day_bars[-1]["close"]
                pnl = (ep - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += ep * pos.quantity
                res.trades.append(Trade(sym, "SELL", ep, pos.quantity, day_bars[-1]["timestamp"], "EOD", "vwap"))
                del res.open_positions[sym]
                res.update_drawdown()
    return res


def run_vwap_ultra(bars_5m: dict[str, list[dict]]) -> Result:
    """VWAP ULTRA v2 -- optimized mean-reversion with smart sizing.

    Key fixes from v1:
    - Removed EMA/midday filters (they filtered 1725 profitable trades)
    - Disabled MACD exit (premature exits killed avg win)
    - Disabled brackets (VWAP moves too small for 1.5%/3% targets;
      baseline's full close at VWAP touch is superior)
    - Lowered confluence/win-prob gates (VWAP dips are inherently high-prob)
    - Kelly sizing with baseline floor (never smaller than $17k)
    - Tuned trailing stop for VWAP reversion moves
    """
    res = Result(name="VWAP (ULTRA)")
    strategy_wins, strategy_losses = 0, 0
    strategy_total_win, strategy_total_loss = 0.0, 0.0
    consecutive_losses: dict[str, int] = {s: 0 for s in SYMBOLS}
    cooldown_counter: dict[str, int] = {}

    # V2 strategy-specific params (tuned for mean-reversion)
    bounce_pct = 0.10      # Slightly easier entry (baseline: 0.12)
    tp_pct = 0.50          # Wider TP to let winners run (baseline: 0.4)
    sl_pct = 0.30          # Slightly wider SL (baseline: 0.25)
    conf_min = 20          # Very light confluence gate (was 40)
    wp_min = 0.25          # Very light win-prob gate (was 0.40)
    trail_act_pct = 0.35   # Trailing activation %
    trail_dist_pct = 0.20  # Trailing distance %

    for sym, bars in bars_5m.items():
        days = group_by_day(bars)
        vwap_tracker = VWAPTracker()
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            vwap_tracker.reset()
            closes: list[float] = []
            volumes: list[float] = []
            highs: list[float] = []
            lows: list[float] = []

            for bar in day_bars:
                closes.append(bar["close"])
                volumes.append(bar["volume"])
                highs.append(bar["high"])
                lows.append(bar["low"])
                price, ts = bar["close"], bar["timestamp"]
                vwap_val = vwap_tracker.add(bar)
                if vwap_val is None:
                    continue

                if sym in cooldown_counter:
                    cooldown_counter[sym] -= 1
                    if cooldown_counter[sym] <= 0:
                        del cooldown_counter[sym]

                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

                    # NO MACD exit -- premature exits destroyed avg win in v1

                    # Trailing stop (VWAP-tuned params)
                    if price > pos.highest_price:
                        pos.highest_price = price
                    gain_pct = (pos.highest_price - pos.entry_price) / pos.entry_price * 100
                    if gain_pct >= trail_act_pct:
                        drop = (pos.highest_price - price) / pos.highest_price * 100
                        if drop >= trail_dist_pct:
                            pnl = (price - pos.entry_price) * pos.quantity
                            res.closed_pnl.append(pnl)
                            res.cash += price * pos.quantity
                            res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TRAIL", "vwap"))
                            del res.open_positions[sym]
                            if pnl > 0:
                                strategy_wins += 1
                                strategy_total_win += pnl
                            else:
                                strategy_losses += 1
                                strategy_total_loss += abs(pnl)
                            res.update_drawdown()
                            continue

                    # NO brackets -- VWAP moves are 0.1-0.4%, brackets at 1.5%/3%
                    # never trigger meaningfully. Full close at VWAP touch is better.

                    # Take profit: full close at VWAP touch or tp_pct
                    if price >= vwap_val or pnl_pct >= tp_pct:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP/VWAP", "vwap"))
                        del res.open_positions[sym]
                        if pnl > 0:
                            strategy_wins += 1
                            strategy_total_win += pnl
                        else:
                            strategy_losses += 1
                            strategy_total_loss += abs(pnl)
                        consecutive_losses[sym] = 0
                        res.update_drawdown()
                        continue

                    # Stop loss (ATR-based: per-position dynamic stop price)
                    if price <= pos.stop_loss:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "SL", "vwap"))
                        del res.open_positions[sym]
                        strategy_losses += 1
                        strategy_total_loss += abs(pnl)
                        consecutive_losses[sym] = consecutive_losses.get(sym, 0) + 1
                        if consecutive_losses[sym] >= MAX_CONSECUTIVE_LOSSES:
                            cooldown_counter[sym] = COOLDOWN_BARS
                            consecutive_losses[sym] = 0
                        res.update_drawdown()
                        continue

                # Entry -- VWAP dip with light Ultra gates
                # NO EMA filter (filtered 1725 profitable trades in v1)
                # NO midday filter (mean-reversion works well in low-vol periods)
                if sym not in res.open_positions and vwap_val > 0:
                    if sym in cooldown_counter:
                        continue

                    deviation = (vwap_val - price) / vwap_val * 100
                    if deviation >= bounce_pct:
                        # Light confluence gate
                        conf = _compute_confluence(closes, volumes, highs, lows)
                        if conf < conf_min:
                            continue
                        # Light win-prob gate
                        win_prob = _compute_win_probability(closes, volumes)
                        if win_prob < wp_min:
                            continue
                        # Portfolio heat
                        current_heat = sum(
                            abs(p.entry_price * p.quantity - p.stop_loss * p.quantity)
                            for p in res.open_positions.values()
                        ) / max(res.cash, 1)
                        if current_heat >= MAX_PORTFOLIO_HEAT:
                            continue

                        # Kelly sizing with baseline floor
                        total_trades = strategy_wins + strategy_losses
                        wr = strategy_wins / max(total_trades, 1)
                        aw = strategy_total_win / max(strategy_wins, 1)
                        al = strategy_total_loss / max(strategy_losses, 1)

                        # ATR-based dynamic stop-loss (v7: never tighter than baseline)
                        atr_val = _compute_atr(highs, lows, closes)
                        if atr_val > 0:
                            atr_pct_local = (atr_val / price) * 100
                            dyn_sl = max(sl_pct, min(0.50, atr_pct_local * 1.5))
                        else:
                            dyn_sl = sl_pct  # Fallback to fixed

                        order_usd = _kelly_size(wr, aw, al, res.cash, price, dyn_sl)
                        order_usd = max(order_usd, ORDER_SIZE_USD)  # Never smaller than baseline

                        # Progressive risk scaling (v5: reduce after losses)
                        cl = consecutive_losses.get(sym, 0)
                        if cl > 0:
                            prog_mult = max(0.25, 0.75 ** cl)
                            order_usd *= prog_mult

                        qty = order_usd / price
                        cost = qty * price

                        if res.cash >= cost:
                            res.cash -= cost
                            sl_price = price * (1 - dyn_sl / 100)
                            res.open_positions[sym] = Position(
                                sym, price, qty, ts, strategy="vwap", stop_loss=sl_price
                            )
                            res.trades.append(Trade(sym, "BUY", price, qty, ts,
                                                    f"ULTRA(c={conf},wp={win_prob:.0%})", "vwap"))
                            consecutive_losses[sym] = 0
                            res.update_drawdown()

            # EOD
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                ep = day_bars[-1]["close"]
                pnl = (ep - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += ep * pos.quantity
                res.trades.append(Trade(sym, "SELL", ep, pos.quantity, day_bars[-1]["timestamp"], "EOD", "vwap"))
                del res.open_positions[sym]
                if pnl > 0:
                    strategy_wins += 1
                    strategy_total_win += pnl
                else:
                    strategy_losses += 1
                    strategy_total_loss += abs(pnl)
                res.update_drawdown()

    return res


# ── ORB Strategy ──────────────────────────────────────────────────────────────

def run_orb_baseline(bars_1m: dict[str, list[dict]]) -> Result:
    """Opening Range Breakout — baseline (15-min range)."""
    res = Result(name="ORB (baseline)")
    for sym, bars in bars_1m.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            if len(day_bars) < 15:
                continue
            opening = day_bars[:15]
            range_high = max(b["high"] for b in opening)
            range_low = min(b["low"] for b in opening)
            range_size = range_high - range_low
            if range_size <= 0:
                continue
            for bar in day_bars[15:]:
                price, ts = bar["close"], bar["timestamp"]
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if pnl_pct >= ORB_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP", "orb"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                    if pnl_pct <= -ORB_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "SL", "orb"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                if sym not in res.open_positions:
                    breakout_pct = (price - range_high) / range_high * 100
                    if breakout_pct >= ORB_BREAKOUT_PCT:
                        qty = ORDER_SIZE_USD / price
                        if res.cash >= qty * price:
                            res.cash -= qty * price
                            res.open_positions[sym] = Position(sym, price, qty, ts, strategy="orb")
                            res.trades.append(Trade(sym, "BUY", price, qty, ts, "ORB breakout", "orb"))
                            res.update_drawdown()
            # EOD
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                ep = day_bars[-1]["close"]
                pnl = (ep - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += ep * pos.quantity
                res.trades.append(Trade(sym, "SELL", ep, pos.quantity, day_bars[-1]["timestamp"], "EOD", "orb"))
                del res.open_positions[sym]
                res.update_drawdown()
    return res


def run_orb_improved(bars_1m: dict[str, list[dict]]) -> Result:
    """ORB + EMA filter + midday filter + trailing stop."""
    res = Result(name="ORB (improved)")
    for sym, bars in bars_1m.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            if len(day_bars) < 15:
                continue
            opening = day_bars[:15]
            range_high = max(b["high"] for b in opening)
            range_low = min(b["low"] for b in opening)
            range_size = range_high - range_low
            if range_size <= 0:
                continue
            closes: list[float] = [b["close"] for b in opening]
            for bar in day_bars[15:]:
                closes.append(bar["close"])
                price, ts = bar["close"], bar["timestamp"]
                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    if _check_trailing_stop(pos, price):
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TRAIL", "orb"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                    if pnl_pct >= ORB_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP", "orb"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                    if pnl_pct <= -ORB_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "SL", "orb"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                if sym not in res.open_positions:
                    # EMA
                    if len(closes) >= TREND_EMA_PERIOD:
                        ema_val = _ema(closes, TREND_EMA_PERIOD)
                        if ema_val is not None and price < ema_val:
                            continue
                    if _is_midday(ts):
                        continue
                    breakout_pct = (price - range_high) / range_high * 100
                    if breakout_pct >= ORB_BREAKOUT_PCT:
                        qty = ORDER_SIZE_USD / price
                        if res.cash >= qty * price:
                            res.cash -= qty * price
                            res.open_positions[sym] = Position(sym, price, qty, ts, strategy="orb")
                            res.trades.append(Trade(sym, "BUY", price, qty, ts, "ORB+Filt", "orb"))
                            res.update_drawdown()
            # EOD
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                ep = day_bars[-1]["close"]
                pnl = (ep - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += ep * pos.quantity
                res.trades.append(Trade(sym, "SELL", ep, pos.quantity, day_bars[-1]["timestamp"], "EOD", "orb"))
                del res.open_positions[sym]
                res.update_drawdown()
    return res


def run_orb_ultra(bars_1m: dict[str, list[dict]]) -> Result:
    """ORB ULTRA v3 -- single change from v2: remove brackets to fix R:R.

    V2 root cause: bracket TP1 at 1% cuts HALF position, then trailing
    captures only $34 on rest -> avg win $74 vs avg loss $122 = R:R 0.61.

    V3 fix: Remove brackets only. Full position rides to TP, trail, or SL.
    - Expected: avg win rises from $74 to $100+ (no half-cuts)
    - SL stays at 0.75% (proven -- 0.50% tested and caused 50.7% SL rate)
    - TP at 1.5% (was 1.8% with brackets -- less needed without half-cut)
    - Trailing stays at 0.5%/0.3% (matching global _check_trailing_stop)
    - EMA filter KEPT (removing it dropped WR from 60% to 53%)
    """
    res = Result(name="ORB (ULTRA)")
    strategy_wins, strategy_losses = 0, 0
    strategy_total_win, strategy_total_loss = 0.0, 0.0
    consecutive_losses: dict[str, int] = {s: 0 for s in SYMBOLS}
    cooldown_counter: dict[str, int] = {}

    # V3 params -- same as v2 except NO brackets
    breakout_thresh = 0.15   # Keep v2's proven filter
    tp_pct = 1.5             # Slightly lower than v2 (1.8) since no brackets
    sl_pct = 0.75            # KEEP v2's proven SL (0.50% tested = too tight)
    conf_min = 30            # Same as v2
    wp_min = 0.35            # Same as v2
    vol_confirm = 1.3        # Same as v2

    for sym, bars in bars_1m.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            if len(day_bars) < 15:
                continue
            opening = day_bars[:15]
            range_high = max(b["high"] for b in opening)
            range_low = min(b["low"] for b in opening)
            range_size = range_high - range_low
            if range_size <= 0:
                continue
            closes: list[float] = [b["close"] for b in opening]
            volumes: list[float] = [b["volume"] for b in opening]
            highs: list[float] = [b["high"] for b in opening]
            lows: list[float] = [b["low"] for b in opening]

            for bar in day_bars[15:]:
                closes.append(bar["close"])
                volumes.append(bar["volume"])
                highs.append(bar["high"])
                lows.append(bar["low"])
                price, ts = bar["close"], bar["timestamp"]

                if sym in cooldown_counter:
                    cooldown_counter[sym] -= 1
                    if cooldown_counter[sym] <= 0:
                        del cooldown_counter[sym]

                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

                    # Trailing stop (same as v2: 0.5% activation, 0.3% distance)
                    if _check_trailing_stop(pos, price):
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TRAIL", "orb"))
                        del res.open_positions[sym]
                        if pnl > 0:
                            strategy_wins += 1
                            strategy_total_win += pnl
                        else:
                            strategy_losses += 1
                            strategy_total_loss += abs(pnl)
                        res.update_drawdown()
                        continue

                    # NO BRACKETS -- v3 key change
                    # Single TP at 1.5% (full position, not half)
                    if pnl_pct >= tp_pct:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP", "orb"))
                        del res.open_positions[sym]
                        if pnl > 0:
                            strategy_wins += 1
                            strategy_total_win += pnl
                        else:
                            strategy_losses += 1
                            strategy_total_loss += abs(pnl)
                        consecutive_losses[sym] = 0
                        res.update_drawdown()
                        continue

                    # SL at 0.75% (same as v2)
                    if pnl_pct <= -sl_pct:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "SL", "orb"))
                        del res.open_positions[sym]
                        strategy_losses += 1
                        strategy_total_loss += abs(pnl)
                        consecutive_losses[sym] = consecutive_losses.get(sym, 0) + 1
                        if consecutive_losses[sym] >= MAX_CONSECUTIVE_LOSSES:
                            cooldown_counter[sym] = COOLDOWN_BARS
                            consecutive_losses[sym] = 0
                        res.update_drawdown()
                        continue

                # Entry -- same as v2 but NO brackets
                if sym not in res.open_positions:
                    if sym in cooldown_counter:
                        continue
                    # EMA filter KEPT (removing it dropped WR from 60% to 53%)
                    if len(closes) >= TREND_EMA_PERIOD:
                        ema_val = _ema(closes, TREND_EMA_PERIOD)
                        if ema_val is not None and price < ema_val:
                            continue
                    if _is_midday(ts):
                        continue
                    breakout_pct = (price - range_high) / range_high * 100
                    if breakout_pct >= breakout_thresh:
                        # Volume confirmation
                        avg_vol = sum(volumes[-20:]) / max(len(volumes[-20:]), 1)
                        if avg_vol > 0 and bar["volume"] < avg_vol * vol_confirm:
                            continue

                        # Ultra gates (same as v2)
                        conf = _compute_confluence(closes, volumes, highs, lows)
                        if conf < conf_min:
                            continue
                        win_prob = _compute_win_probability(closes, volumes)
                        if win_prob < wp_min:
                            continue
                        # Portfolio heat
                        current_heat = sum(
                            abs(p.entry_price * p.quantity - p.stop_loss * p.quantity)
                            for p in res.open_positions.values()
                        ) / max(res.cash, 1)
                        if current_heat >= MAX_PORTFOLIO_HEAT:
                            continue

                        # Kelly
                        total_t = strategy_wins + strategy_losses
                        wr = strategy_wins / max(total_t, 1)
                        aw = strategy_total_win / max(strategy_wins, 1)
                        al = strategy_total_loss / max(strategy_losses, 1)
                        order_usd = _kelly_size(wr, aw, al, res.cash, price, sl_pct)

                        # Progressive risk scaling (v5: reduce after losses)
                        cl = consecutive_losses.get(sym, 0)
                        if cl > 0:
                            prog_mult = max(0.25, 0.75 ** cl)
                            order_usd *= prog_mult

                        qty = order_usd / price
                        cost = qty * price

                        if res.cash >= cost:
                            res.cash -= cost
                            sl_price = price * (1 - sl_pct / 100)
                            res.open_positions[sym] = Position(
                                sym, price, qty, ts, strategy="orb", stop_loss=sl_price
                            )
                            res.trades.append(Trade(sym, "BUY", price, qty, ts,
                                                    f"ULTRA(c={conf},wp={win_prob:.0%})", "orb"))
                            consecutive_losses[sym] = 0
                            res.update_drawdown()

            # EOD
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                ep = day_bars[-1]["close"]
                pnl = (ep - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += ep * pos.quantity
                res.trades.append(Trade(sym, "SELL", ep, pos.quantity, day_bars[-1]["timestamp"], "EOD", "orb"))
                del res.open_positions[sym]
                if pnl > 0:
                    strategy_wins += 1
                    strategy_total_win += pnl
                else:
                    strategy_losses += 1
                    strategy_total_loss += abs(pnl)
                res.update_drawdown()

    return res


# ── EMA Pullback Strategy ─────────────────────────────────────────────────────

def run_ema_pullback_baseline(bars_5m: dict[str, list[dict]]) -> Result:
    """EMA Pullback baseline -- trend-following on 3-layer EMA stack.

    Entry: price > EMA50 (uptrend) + EMA9 > EMA21 (momentum aligned)
           + price within 0.15% of EMA9 (pullback zone)
           + RSI 35-55 (not overbought) + volume >= 1.2x avg
    Exit:  TP at 1.5%, SL at 0.75%
    """
    res = Result(name="EMA Pullback (baseline)")
    for sym, bars in bars_5m.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            closes: list[float] = []
            volumes: list[float] = []

            for bar in day_bars:
                closes.append(bar["close"])
                volumes.append(bar["volume"])
                price, ts = bar["close"], bar["timestamp"]

                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    # Take profit
                    if pnl_pct >= EMA_PB_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP", "ema_pullback"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue
                    # Stop loss
                    if pnl_pct <= -EMA_PB_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "SL", "ema_pullback"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue

                # Entry: 3-layer EMA pullback
                if sym not in res.open_positions and len(closes) >= EMA_PB_TREND_PERIOD:
                    ema_fast = _ema(closes, EMA_PB_FAST_PERIOD)
                    ema_slow = _ema(closes, EMA_PB_SLOW_PERIOD)
                    ema_trend = _ema(closes, EMA_PB_TREND_PERIOD)
                    rsi = _rsi(closes)

                    if ema_fast is None or ema_slow is None or ema_trend is None or rsi is None:
                        continue

                    # Trend confirmation: price above 50 EMA
                    if price <= ema_trend:
                        continue
                    # Momentum: fast EMA above slow EMA
                    if ema_fast <= ema_slow:
                        continue
                    # Pullback zone: price within 0.15% of fast EMA
                    distance_pct = abs(price - ema_fast) / ema_fast * 100
                    if distance_pct > EMA_PB_PULLBACK_PCT:
                        continue
                    # RSI in pullback zone (not overbought)
                    if rsi < EMA_PB_RSI_LOW or rsi > EMA_PB_RSI_HIGH:
                        continue
                    # Volume confirmation
                    avg_vol = _avg_volume(day_bars[:len(closes)])
                    if avg_vol > 0 and bar["volume"] < avg_vol * EMA_PB_VOL_MULT:
                        continue

                    qty = ORDER_SIZE_USD / price
                    cost = qty * price
                    if res.cash >= cost:
                        res.cash -= cost
                        res.open_positions[sym] = Position(sym, price, qty, ts, strategy="ema_pullback")
                        res.trades.append(Trade(sym, "BUY", price, qty, ts, "EMA pullback", "ema_pullback"))
                        res.update_drawdown()

            # EOD flatten
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                ep = day_bars[-1]["close"]
                pnl = (ep - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += ep * pos.quantity
                res.trades.append(Trade(sym, "SELL", ep, pos.quantity, day_bars[-1]["timestamp"], "EOD", "ema_pullback"))
                del res.open_positions[sym]
                res.update_drawdown()
    return res


def run_ema_pullback_improved(bars_5m: dict[str, list[dict]]) -> Result:
    """EMA Pullback improved -- adds trailing stop, midday filter, MACD confirm."""
    res = Result(name="EMA Pullback (improved)")
    for sym, bars in bars_5m.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            closes: list[float] = []
            volumes: list[float] = []

            for bar in day_bars:
                closes.append(bar["close"])
                volumes.append(bar["volume"])
                price, ts = bar["close"], bar["timestamp"]

                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

                    # Trailing stop
                    if _check_trailing_stop(pos, price):
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TRAIL", "ema_pullback"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue

                    # RSI overbought exit
                    if len(closes) >= 14:
                        rsi = _rsi(closes)
                        if rsi is not None and rsi >= EMA_PB_RSI_EXIT:
                            pnl = (price - pos.entry_price) * pos.quantity
                            res.closed_pnl.append(pnl)
                            res.cash += price * pos.quantity
                            res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "RSI_OB", "ema_pullback"))
                            del res.open_positions[sym]
                            res.update_drawdown()
                            continue

                    # Take profit
                    if pnl_pct >= EMA_PB_TP_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP", "ema_pullback"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue

                    # Stop loss
                    if pnl_pct <= -EMA_PB_SL_PCT:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "SL", "ema_pullback"))
                        del res.open_positions[sym]
                        res.update_drawdown()
                        continue

                # Entry with filters
                if sym not in res.open_positions and len(closes) >= EMA_PB_TREND_PERIOD:
                    # Midday filter
                    if _is_midday(ts):
                        continue

                    ema_fast = _ema(closes, EMA_PB_FAST_PERIOD)
                    ema_slow = _ema(closes, EMA_PB_SLOW_PERIOD)
                    ema_trend = _ema(closes, EMA_PB_TREND_PERIOD)
                    rsi = _rsi(closes)

                    if ema_fast is None or ema_slow is None or ema_trend is None or rsi is None:
                        continue

                    if price <= ema_trend:
                        continue
                    if ema_fast <= ema_slow:
                        continue
                    distance_pct = abs(price - ema_fast) / ema_fast * 100
                    if distance_pct > EMA_PB_PULLBACK_PCT:
                        continue
                    if rsi < EMA_PB_RSI_LOW or rsi > EMA_PB_RSI_HIGH:
                        continue
                    avg_vol = _avg_volume(day_bars[:len(closes)])
                    if avg_vol > 0 and bar["volume"] < avg_vol * EMA_PB_VOL_MULT:
                        continue

                    # MACD confirmation (golden cross = bullish)
                    if len(closes) >= 35:
                        m = _macd(closes)
                        if m and not m.get("golden_cross") and m.get("histogram", 0) < 0:
                            continue  # Skip if MACD bearish

                    qty = ORDER_SIZE_USD / price
                    cost = qty * price
                    if res.cash >= cost:
                        res.cash -= cost
                        res.open_positions[sym] = Position(sym, price, qty, ts, strategy="ema_pullback")
                        res.trades.append(Trade(sym, "BUY", price, qty, ts, "EMA+Filt", "ema_pullback"))
                        res.update_drawdown()

            # EOD flatten
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                ep = day_bars[-1]["close"]
                pnl = (ep - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += ep * pos.quantity
                res.trades.append(Trade(sym, "SELL", ep, pos.quantity, day_bars[-1]["timestamp"], "EOD", "ema_pullback"))
                del res.open_positions[sym]
                res.update_drawdown()
    return res


def run_ema_pullback_ultra(bars_5m: dict[str, list[dict]]) -> Result:
    """EMA Pullback ULTRA -- full gating with Kelly sizing.

    Trend-following complement to VWAP mean-reversion:
    - 3-layer EMA stack: 9/21/50 for precise pullback entries
    - Confluence + Win-Probability gates
    - Kelly position sizing with baseline floor
    - Portfolio heat tracking + cooldowns
    - Trailing stop primary exit, RSI overbought secondary
    - Bracket TP for partial profits on strong runners
    """
    res = Result(name="EMA Pullback (ULTRA)")
    strategy_wins, strategy_losses = 0, 0
    strategy_total_win, strategy_total_loss = 0.0, 0.0
    consecutive_losses: dict[str, int] = {s: 0 for s in SYMBOLS}
    cooldown_counter: dict[str, int] = {}

    # Ultra-tuned params for trend-following
    pullback_pct = 0.20    # Slightly wider pullback zone (more entries)
    rsi_low = 33           # Slightly wider RSI range
    rsi_high = 58          # Slightly wider RSI range
    rsi_exit = 72          # RSI overbought exit
    tp_pct = 2.0           # Wider TP to let trend winners run
    sl_pct = 0.85          # Wider SL for trend trades (R:R ~2.35:1)
    bracket_tp1 = 1.2      # Partial close at 1.2%
    bracket_tp2 = 3.0      # Let trend runners go to 3%
    conf_min = 25          # Light confluence gate
    wp_min = 0.30          # Light win-prob gate
    trail_act_pct = 0.6    # Activate trail at 0.6% gain
    trail_dist_pct = 0.35  # Trail distance 0.35%

    for sym, bars in bars_5m.items():
        days = group_by_day(bars)
        for day_key in sorted(days.keys()):
            day_bars = days[day_key]
            closes: list[float] = []
            volumes: list[float] = []
            highs: list[float] = []
            lows: list[float] = []

            for bar in day_bars:
                closes.append(bar["close"])
                volumes.append(bar["volume"])
                highs.append(bar["high"])
                lows.append(bar["low"])
                price, ts = bar["close"], bar["timestamp"]

                # Cooldown management
                if sym in cooldown_counter:
                    cooldown_counter[sym] -= 1
                    if cooldown_counter[sym] <= 0:
                        del cooldown_counter[sym]

                pos = res.open_positions.get(sym)
                if pos:
                    pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

                    # Trailing stop (ULTRA-tuned for trend)
                    if price > pos.highest_price:
                        pos.highest_price = price
                    gain_pct = (pos.highest_price - pos.entry_price) / pos.entry_price * 100
                    if gain_pct >= trail_act_pct:
                        drop = (pos.highest_price - price) / pos.highest_price * 100
                        if drop >= trail_dist_pct:
                            pnl = (price - pos.entry_price) * pos.quantity
                            res.closed_pnl.append(pnl)
                            res.cash += price * pos.quantity
                            res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TRAIL", "ema_pullback"))
                            del res.open_positions[sym]
                            if pnl > 0:
                                strategy_wins += 1
                                strategy_total_win += pnl
                            else:
                                strategy_losses += 1
                                strategy_total_loss += abs(pnl)
                            res.update_drawdown()
                            continue

                    # RSI overbought exit (trend exhaustion signal)
                    if len(closes) >= 14:
                        rsi = _rsi(closes)
                        if rsi is not None and rsi >= rsi_exit:
                            pnl = (price - pos.entry_price) * pos.quantity
                            res.closed_pnl.append(pnl)
                            res.cash += price * pos.quantity
                            res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "RSI_OB", "ema_pullback"))
                            del res.open_positions[sym]
                            if pnl > 0:
                                strategy_wins += 1
                                strategy_total_win += pnl
                            else:
                                strategy_losses += 1
                                strategy_total_loss += abs(pnl)
                            res.update_drawdown()
                            continue

                    # Bracket TP1 -- partial close (50%)
                    if not pos.partial_taken and pnl_pct >= bracket_tp1:
                        half_qty = pos.quantity * 0.5
                        pnl_partial = (price - pos.entry_price) * half_qty
                        res.closed_pnl.append(pnl_partial)
                        res.cash += price * half_qty
                        pos.quantity -= half_qty
                        pos.partial_taken = True
                        res.trades.append(Trade(sym, "SELL", price, half_qty, ts, "TP1(50%)", "ema_pullback"))
                        if pnl_partial > 0:
                            strategy_wins += 1
                            strategy_total_win += pnl_partial
                        else:
                            strategy_losses += 1
                            strategy_total_loss += abs(pnl_partial)
                        res.update_drawdown()
                        continue

                    # Bracket TP2 -- full close after partial
                    if pos.partial_taken and pnl_pct >= bracket_tp2:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP2", "ema_pullback"))
                        del res.open_positions[sym]
                        if pnl > 0:
                            strategy_wins += 1
                            strategy_total_win += pnl
                        else:
                            strategy_losses += 1
                            strategy_total_loss += abs(pnl)
                        consecutive_losses[sym] = 0
                        res.update_drawdown()
                        continue

                    # Standard TP (if no partial yet)
                    if not pos.partial_taken and pnl_pct >= tp_pct:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "TP", "ema_pullback"))
                        del res.open_positions[sym]
                        if pnl > 0:
                            strategy_wins += 1
                            strategy_total_win += pnl
                        else:
                            strategy_losses += 1
                            strategy_total_loss += abs(pnl)
                        consecutive_losses[sym] = 0
                        res.update_drawdown()
                        continue

                    # Stop loss
                    if pnl_pct <= -sl_pct:
                        pnl = (price - pos.entry_price) * pos.quantity
                        res.closed_pnl.append(pnl)
                        res.cash += price * pos.quantity
                        res.trades.append(Trade(sym, "SELL", price, pos.quantity, ts, "SL", "ema_pullback"))
                        del res.open_positions[sym]
                        strategy_losses += 1
                        strategy_total_loss += abs(pnl)
                        consecutive_losses[sym] = consecutive_losses.get(sym, 0) + 1
                        if consecutive_losses[sym] >= MAX_CONSECUTIVE_LOSSES:
                            cooldown_counter[sym] = COOLDOWN_BARS
                            consecutive_losses[sym] = 0
                        res.update_drawdown()
                        continue

                # Entry logic with Ultra gates
                if sym not in res.open_positions and len(closes) >= EMA_PB_TREND_PERIOD:
                    # Skip if in cooldown
                    if sym in cooldown_counter:
                        continue
                    # Midday filter (trend pullbacks work poorly midday)
                    if _is_midday(ts):
                        continue

                    ema_fast = _ema(closes, EMA_PB_FAST_PERIOD)
                    ema_slow = _ema(closes, EMA_PB_SLOW_PERIOD)
                    ema_trend = _ema(closes, EMA_PB_TREND_PERIOD)
                    rsi = _rsi(closes)

                    if ema_fast is None or ema_slow is None or ema_trend is None or rsi is None:
                        continue

                    # Trend: price above EMA50
                    if price <= ema_trend:
                        continue
                    # Momentum: EMA9 > EMA21
                    if ema_fast <= ema_slow:
                        continue
                    # Pullback zone: price near fast EMA
                    distance_pct = abs(price - ema_fast) / ema_fast * 100
                    if distance_pct > pullback_pct:
                        continue
                    # RSI in pullback zone
                    if rsi < rsi_low or rsi > rsi_high:
                        continue
                    # Volume confirmation
                    avg_vol = _avg_volume(day_bars[:len(closes)])
                    if avg_vol > 0 and bar["volume"] < avg_vol * EMA_PB_VOL_MULT:
                        continue

                    # ULTRA GATE 1: Confluence score
                    conf = _compute_confluence(closes, volumes, highs, lows)
                    if conf < conf_min:
                        continue

                    # ULTRA GATE 2: Win probability
                    win_prob = _compute_win_probability(closes, volumes)
                    if win_prob < wp_min:
                        continue

                    # ULTRA GATE 3: Portfolio heat check
                    current_heat = sum(
                        abs(p.entry_price * p.quantity - p.stop_loss * p.quantity)
                        for p in res.open_positions.values()
                    ) / max(res.cash, 1)
                    if current_heat >= MAX_PORTFOLIO_HEAT:
                        continue

                    # Kelly position sizing with baseline floor
                    total_t = strategy_wins + strategy_losses
                    wr = strategy_wins / max(total_t, 1)
                    aw = strategy_total_win / max(strategy_wins, 1)
                    al = strategy_total_loss / max(strategy_losses, 1)
                    order_usd = _kelly_size(wr, aw, al, res.cash, price, sl_pct)
                    order_usd = max(order_usd, ORDER_SIZE_USD)  # Never smaller than baseline

                    # Progressive risk scaling (v5: reduce after losses)
                    cl = consecutive_losses.get(sym, 0)
                    if cl > 0:
                        prog_mult = max(0.25, 0.75 ** cl)
                        order_usd *= prog_mult

                    qty = order_usd / price
                    cost = qty * price

                    if res.cash >= cost:
                        res.cash -= cost
                        sl_price = price * (1 - sl_pct / 100)
                        res.open_positions[sym] = Position(
                            sym, price, qty, ts, strategy="ema_pullback",
                            stop_loss=sl_price
                        )
                        res.trades.append(Trade(sym, "BUY", price, qty, ts,
                                                f"ULTRA(c={conf},wp={win_prob:.0%})", "ema_pullback"))
                        consecutive_losses[sym] = 0
                        res.update_drawdown()

            # EOD flatten
            if sym in res.open_positions and day_bars:
                pos = res.open_positions[sym]
                ep = day_bars[-1]["close"]
                pnl = (ep - pos.entry_price) * pos.quantity
                res.closed_pnl.append(pnl)
                res.cash += ep * pos.quantity
                res.trades.append(Trade(sym, "SELL", ep, pos.quantity, day_bars[-1]["timestamp"], "EOD", "ema_pullback"))
                del res.open_positions[sym]
                if pnl > 0:
                    strategy_wins += 1
                    strategy_total_win += pnl
                else:
                    strategy_losses += 1
                    strategy_total_loss += abs(pnl)
                res.update_drawdown()

    return res


# ── Kelly stats helper ────────────────────────────────────────────────────────

# Module-level accumulators for Kelly (reset per strategy run)
_kelly_wins = 0
_kelly_losses = 0
_kelly_win_total = 0.0
_kelly_loss_total = 0.0


def _update_kelly_stats(pnl: float):
    global _kelly_wins, _kelly_losses, _kelly_win_total, _kelly_loss_total
    if pnl > 0:
        _kelly_wins += 1
        _kelly_win_total += pnl
    else:
        _kelly_losses += 1
        _kelly_loss_total += abs(pnl)


def _reset_kelly_stats():
    global _kelly_wins, _kelly_losses, _kelly_win_total, _kelly_loss_total
    _kelly_wins = 0
    _kelly_losses = 0
    _kelly_win_total = 0.0
    _kelly_loss_total = 0.0


def _record_trade(pnl: float, wins: int, losses: int,
                   total_win: float, total_loss: float):
    """Record trade for local strategy stats (for Ultra strategies using locals)."""
    _update_kelly_stats(pnl)


# ── Combine results ──────────────────────────────────────────────────────────

def combine_results(results: list[Result], name: str) -> Result:
    combined = Result(name=name)
    combined.closed_pnl = []
    combined.trades = []
    for r in results:
        combined.closed_pnl.extend(r.closed_pnl)
        combined.trades.extend(r.trades)
    combined.cash = STARTING_CAPITAL + sum(combined.closed_pnl)
    combined.max_drawdown_pct = max(r.max_drawdown_pct for r in results)
    return combined


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(result: Result, trading_days: int) -> None:
    final_equity = result.cash
    ret_pct = (final_equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    daily_avg = result.net_pnl / trading_days if trading_days > 0 else 0
    monthly_avg = daily_avg * 21

    tp1_count = len([t for t in result.trades if t.side == "SELL" and "TP1" in t.reason])
    tp2_count = len([t for t in result.trades if t.side == "SELL" and "TP2" in t.reason])
    tp_count = len([t for t in result.trades if t.side == "SELL" and t.reason == "TP"])
    sl_count = len([t for t in result.trades if t.side == "SELL" and t.reason == "SL"])
    eod_count = len([t for t in result.trades if t.side == "SELL" and t.reason == "EOD"])
    trail_count = len([t for t in result.trades if t.side == "SELL" and t.reason == "TRAIL"])
    macd_count = len([t for t in result.trades if t.side == "SELL" and "MACD" in t.reason])

    if result.closed_pnl and len(result.closed_pnl) > 1:
        try:
            trade_rets = [p / ORDER_SIZE_USD for p in result.closed_pnl]
            mean_r = statistics.mean(trade_rets)
            std_r = statistics.stdev(trade_rets)
            sharpe = (mean_r / std_r) * (252 ** 0.5) if std_r > 0 else 0.0
            # Sortino
            downside = [r for r in trade_rets if r < 0]
            down_std = statistics.stdev(downside) if len(downside) > 1 else 0.001
            sortino = (mean_r / down_std) * (252 ** 0.5) if down_std > 0 else 0.0
        except Exception:
            sharpe, sortino = 0.0, 0.0
    else:
        sharpe, sortino = 0.0, 0.0

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
    print(f"  Sortino (approx):     {sortino:>12.2f}")
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
    print(f"    TP1 (partial 50%):  {tp1_count:>6}")
    print(f"    TP2 (full close):   {tp2_count:>6}")
    print(f"    Stop-Loss:          {sl_count:>6}")
    print(f"    Trailing Stop:      {trail_count:>6}")
    print(f"    MACD Death Cross:   {macd_count:>6}")
    print(f"    EOD Flatten:        {eod_count:>6}")
    print()
    print(f"  Avg Daily P&L:        ${daily_avg:>12,.2f}")
    print(f"  Est. Monthly P&L:     ${monthly_avg:>12,.2f}")
    print(f"  Trading Days:         {trading_days:>6}")
    print(bar)


def print_comparison(all_results: list[Result], trading_days: int, period_label: str) -> None:
    """Print side-by-side comparison table."""
    print("\n")
    print("=" * 100)
    print(f"  BASELINE vs IMPROVED vs ULTRA -- {period_label}")
    print("=" * 100)
    header = (f"  {'Strategy':<28} {'Net P&L':>12} {'Win Rate':>10} {'Trades':>8} "
              f"{'Max DD':>8} {'Sharpe':>8} {'Mo. P&L':>10}")
    print(header)
    print(f"  {'-'*28} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for r in all_results:
        pnl_str = f"${r.net_pnl:>+10,.2f}"
        monthly = r.net_pnl / trading_days * 21 if trading_days > 0 else 0
        mo_str = f"${monthly:>+8,.0f}"
        # Quick Sharpe
        if r.closed_pnl and len(r.closed_pnl) > 1:
            try:
                rets = [p / ORDER_SIZE_USD for p in r.closed_pnl]
                sharpe = (statistics.mean(rets) / statistics.stdev(rets)) * (252 ** 0.5)
            except Exception:
                sharpe = 0.0
        else:
            sharpe = 0.0
        print(f"  {r.name:<28} {pnl_str:>12} {r.win_rate:>9.1f}% "
              f"{len(r.closed_pnl):>8} {r.max_drawdown_pct:>7.2f}% "
              f"{sharpe:>8.2f} {mo_str:>10}")

    print("=" * 100)

    # Highlight best by: P&L, Win Rate, Sharpe, Lowest DD
    best_pnl = max(all_results, key=lambda r: r.net_pnl)
    best_wr = max(all_results, key=lambda r: r.win_rate)
    best_dd = min(all_results, key=lambda r: r.max_drawdown_pct)

    print(f"\n  BEST NET P&L:    {best_pnl.name} -> ${best_pnl.net_pnl:+,.2f}")
    print(f"  BEST WIN RATE:   {best_wr.name} -> {best_wr.win_rate:.1f}%")
    print(f"  LOWEST DRAWDOWN: {best_dd.name} -> {best_dd.max_drawdown_pct:.2f}%")


def print_ultra_analytics(ultra_results: list[Result], period_label: str) -> None:
    """Print Ultra-specific metrics (bracket stats, MACD exits, etc.)."""
    print("\n")
    print("-" * 80)
    print(f"  ULTRA ANALYTICS -- {period_label}")
    print("-" * 80)
    for r in ultra_results:
        total_exits = len([t for t in r.trades if t.side == "SELL"])
        tp1 = len([t for t in r.trades if "TP1" in t.reason])
        tp2 = len([t for t in r.trades if "TP2" in t.reason])
        macd = len([t for t in r.trades if "MACD" in t.reason])
        trail = len([t for t in r.trades if t.reason == "TRAIL"])
        sl_count = len([t for t in r.trades if t.reason == "SL"])
        eod = len([t for t in r.trades if t.reason == "EOD"])
        entries = len([t for t in r.trades if t.side == "BUY"])
        ultra_entries = len([t for t in r.trades if t.side == "BUY" and "ULTRA" in t.reason])

        print(f"\n  {r.name}:")
        print(f"    Entries:           {entries:>4}  (Ultra-gated: {ultra_entries})")
        print(f"    Bracket TP1 (50%): {tp1:>4}  ({tp1/max(total_exits,1)*100:.1f}% of exits)")
        print(f"    Bracket TP2 (full):{tp2:>4}  ({tp2/max(total_exits,1)*100:.1f}% of exits)")
        print(f"    MACD Death Cross:  {macd:>4}  ({macd/max(total_exits,1)*100:.1f}% of exits)")
        print(f"    Trailing Stop:     {trail:>4}  ({trail/max(total_exits,1)*100:.1f}% of exits)")
        print(f"    Stop Loss:         {sl_count:>4}  ({sl_count/max(total_exits,1)*100:.1f}% of exits)")
        print(f"    EOD Flatten:       {eod:>4}  ({eod/max(total_exits,1)*100:.1f}% of exits)")
    print("-" * 80)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(start: datetime, end: datetime, period_label: str) -> None:
    """Run full Baseline → Improved → ULTRA backtest for a time period."""
    _reset_kelly_stats()

    print()
    print("#" * 80)
    print(f"  AtoBot ULTRA Backtester -- {period_label}")
    print("#" * 80)
    print()
    print(f"  Symbols:   {', '.join(SYMBOLS)}")
    print(f"  Capital:   ${STARTING_CAPITAL:,.0f}")
    print(f"  Period:    {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print()
    print(f"  Ultra V2 Features (strategy-specific tuning):")
    print(f"    Momentum:     RSI<=32, TP=2.0%, SL=1.0%, conf>=30, wp>=0.35")
    print(f"    VWAP:         bounce>=0.10%, TP=0.50%, SL=0.30%, conf>=20, wp>=0.25")
    print(f"                  No EMA/midday/MACD/brackets (match baseline volume)")
    print(f"    EMA Pullback: 9/21/50 EMA stack, pullback<=0.20%, RSI 33-58")
    print(f"                  TP=2.0%, SL=0.85%, brackets 1.2%/3.0%, RSI OB exit")
    print(f"    ORB:          breakout>=0.15%, TP=1.8%, SL=0.75%, conf>=30, wp>=0.35")
    print(f"                  No MACD exit (was 82% over-triggering), vol confirm 1.3x")
    print(f"    Kelly sizing:        {KELLY_FRACTION:.0%} Kelly (all strategies)")
    print(f"    Portfolio heat cap:  {MAX_PORTFOLIO_HEAT:.0%}")
    print(f"    Max consec losses:   {MAX_CONSECUTIVE_LOSSES} -> {COOLDOWN_BARS} bar cooldown")
    print()

    # Fetch data
    print("  Fetching historical data...")
    bars_1m = fetch_bars(SYMBOLS, start, end)
    total_bars = sum(len(b) for b in bars_1m.values())
    if total_bars == 0:
        print("  ERROR: No data fetched. Check API keys / date range.")
        return

    all_days: set[str] = set()
    for sym_bars in bars_1m.values():
        for b in sym_bars:
            all_days.add(b["timestamp"].strftime("%Y-%m-%d"))
    trading_days = len(all_days)
    print(f"  Total bars: {total_bars:,}  |  Trading days: {trading_days}")

    bars_5m = {sym: bars_to_5min(b) for sym, b in bars_1m.items()}

    # ── Baseline ──────────────────────────────────────────────────────────
    print("\n  > Running BASELINE strategies...")
    mom_base = run_momentum_baseline(bars_5m)
    vwap_base = run_vwap_baseline(bars_5m)
    ema_pb_base = run_ema_pullback_baseline(bars_5m)
    orb_base = run_orb_baseline(bars_1m)

    # ── Improved ──────────────────────────────────────────────────────────
    print("  > Running IMPROVED strategies...")
    mom_imp = run_momentum_improved(bars_5m)
    vwap_imp = run_vwap_improved(bars_5m)
    ema_pb_imp = run_ema_pullback_improved(bars_5m)
    orb_imp = run_orb_improved(bars_1m)

    # ── Ultra ─────────────────────────────────────────────────────────────
    print("  > Running ULTRA strategies...")
    _reset_kelly_stats()
    mom_ultra = run_momentum_ultra(bars_5m)
    _reset_kelly_stats()
    vwap_ultra = run_vwap_ultra(bars_5m)
    _reset_kelly_stats()
    ema_pb_ultra = run_ema_pullback_ultra(bars_5m)
    _reset_kelly_stats()
    orb_ultra = run_orb_ultra(bars_1m)

    # ── Combos ────────────────────────────────────────────────────────────
    # VWAP + EMA Pullback (recommended combo: mean-reversion + trend-following)
    combo_ve_base = combine_results([vwap_base, ema_pb_base], "VWAP+EMA (baseline)")
    combo_ve_imp = combine_results([vwap_imp, ema_pb_imp], "VWAP+EMA (improved)")
    combo_ve_ultra = combine_results([vwap_ultra, ema_pb_ultra], "VWAP+EMA (ULTRA)")

    # Legacy: VWAP + ORB
    combo_vo_base = combine_results([vwap_base, orb_base], "VWAP+ORB (baseline)")
    combo_vo_imp = combine_results([vwap_imp, orb_imp], "VWAP+ORB (improved)")
    combo_vo_ultra = combine_results([vwap_ultra, orb_ultra], "VWAP+ORB (ULTRA)")

    # ALL 3 Core (VWAP + EMA + Momentum -- excludes ORB)
    all3_base = combine_results([mom_base, vwap_base, ema_pb_base], "ALL 3 Core (baseline)")
    all3_imp = combine_results([mom_imp, vwap_imp, ema_pb_imp], "ALL 3 Core (improved)")
    all3_ultra = combine_results([mom_ultra, vwap_ultra, ema_pb_ultra], "ALL 3 Core (ULTRA)")

    # ALL 4 (everything including ORB for reference)
    all4_base = combine_results([mom_base, vwap_base, ema_pb_base, orb_base], "ALL 4 (baseline)")
    all4_imp = combine_results([mom_imp, vwap_imp, ema_pb_imp, orb_imp], "ALL 4 (improved)")
    all4_ultra = combine_results([mom_ultra, vwap_ultra, ema_pb_ultra, orb_ultra], "ALL 4 (ULTRA)")

    # ── Reports ───────────────────────────────────────────────────────────
    # Individual strategy reports
    print("\n" + "-" * 60)
    print("  INDIVIDUAL STRATEGY REPORTS")
    print("-" * 60)

    for r in [mom_base, mom_imp, mom_ultra,
              vwap_base, vwap_imp, vwap_ultra,
              ema_pb_base, ema_pb_imp, ema_pb_ultra,
              orb_base, orb_imp, orb_ultra]:
        print_report(r, trading_days)

    # Combo reports
    for r in [combo_ve_base, combo_ve_imp, combo_ve_ultra,
              combo_vo_base, combo_vo_imp, combo_vo_ultra,
              all3_base, all3_imp, all3_ultra,
              all4_base, all4_imp, all4_ultra]:
        print_report(r, trading_days)

    # Comparison table
    all_results = [
        mom_base, mom_imp, mom_ultra,
        vwap_base, vwap_imp, vwap_ultra,
        ema_pb_base, ema_pb_imp, ema_pb_ultra,
        orb_base, orb_imp, orb_ultra,
        combo_ve_base, combo_ve_imp, combo_ve_ultra,
        combo_vo_base, combo_vo_imp, combo_vo_ultra,
        all3_base, all3_imp, all3_ultra,
        all4_base, all4_imp, all4_ultra,
    ]
    print_comparison(all_results, trading_days, period_label)

    # Ultra analytics
    print_ultra_analytics([mom_ultra, vwap_ultra, ema_pb_ultra, orb_ultra], period_label)

    # Final best
    best = max(all_results, key=lambda r: r.net_pnl)
    monthly = best.net_pnl / trading_days * 21 if trading_days > 0 else 0
    print(f"\n  >>> OVERALL BEST: {best.name}")
    print(f"    Net P&L: ${best.net_pnl:+,.2f}  |  Win Rate: {best.win_rate:.1f}%  |  "
          f"Max DD: {best.max_drawdown_pct:.2f}%  |  Est Monthly: ${monthly:+,.0f}/mo")

    # Recommended combo highlight
    ve_pnl = combo_ve_ultra.net_pnl
    ve_mo = ve_pnl / trading_days * 21 if trading_days > 0 else 0
    a3_pnl = all3_ultra.net_pnl
    a3_mo = a3_pnl / trading_days * 21 if trading_days > 0 else 0
    print(f"\n  >>> RECOMMENDED (VWAP+EMA ULTRA): ${ve_pnl:+,.2f}  |  Est Monthly: ${ve_mo:+,.0f}/mo")
    print(f"  >>> ALL 3 Core ULTRA:             ${a3_pnl:+,.2f}  |  Est Monthly: ${a3_mo:+,.0f}/mo")
    print()


def main() -> None:
    # Ensure UTF-8 output on Windows
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="AtoBot Ultra Backtester")
    parser.add_argument("--period", choices=["1m", "3m", "both"], default="both",
                        help="Backtest period: 1m (1 month), 3m (3 months), both")
    parser.add_argument("--output", type=str, default=None,
                        help="Save output to file")
    parser.add_argument("--start", type=str, default=None,
                        help="Custom start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                        help="Custom end date (YYYY-MM-DD)")
    parser.add_argument("--label", type=str, default=None,
                        help="Custom label for the backtest period")
    args = parser.parse_args()

    end = datetime(2026, 2, 21, tzinfo=timezone.utc)

    if args.output:
        import io
        output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = output

    try:
        # Custom date range mode
        if args.start and args.end:
            custom_start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            custom_end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            label = args.label or f"{args.start} to {args.end}"
            run_backtest(custom_start, custom_end, label)
        else:
            if args.period in ("1m", "both"):
                start_1m = end - timedelta(days=30)
                run_backtest(start_1m, end, "1 MONTH")

            if args.period in ("3m", "both"):
                start_3m = end - timedelta(days=90)
                run_backtest(start_3m, end, "3 MONTHS")

    finally:
        if args.output:
            sys.stdout = old_stdout
            text = output.getvalue()
            print(text)  # Also print to console
            Path(args.output).write_text(text)
            print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
