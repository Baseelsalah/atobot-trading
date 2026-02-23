"""Verify stress-test-optimized config vs old deployed config.

Compares:
  OLD:  VWAP(b=0.12,tp=0.4,sl=0.25) + ORB     [EMA+midday filters]
  NEW:  VWAP(b=0.05,tp=0.4,sl=0.50) ONLY       [midday filter only]
  BASE: VWAP baseline (no filters, old params)

Across 1, 3, 6 month lookbacks.
"""
from __future__ import annotations
import sys, statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
try:
    from alpaca.data.enums import DataFeed
except ImportError:
    DataFeed = None

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

SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD"]
STARTING_CAPITAL = 100_000.0
ORDER_SIZE_USD = 17_000.0

# ── Helpers ───────────────────────────────────────────────────────────────────
def _vwap(bars):
    tp_vol = vol = 0.0
    for b in bars:
        tp = (b["high"] + b["low"] + b["close"]) / 3.0
        v = b["volume"]
        if v > 0:
            tp_vol += tp * v; vol += v
    return tp_vol / vol if vol > 0 else None

def _ema(closes, period):
    if len(closes) < period: return None
    m = 2.0 / (period + 1)
    e = sum(closes[:period]) / period
    for p in closes[period:]: e = (p - e) * m + e
    return e

def _is_midday(ts):
    try:
        et = ts.astimezone(ZoneInfo("America/New_York"))
        return 12 <= et.hour < 14
    except: return False

# ── Trade tracking ────────────────────────────────────────────────────────────
@dataclass
class Pos:
    entry: float; qty: float; high: float = 0.0
    def __post_init__(self): self.high = self.entry

@dataclass
class Result:
    name: str
    pnl: list[float] = field(default_factory=list)
    cash: float = STARTING_CAPITAL
    peak: float = 0.0; max_dd: float = 0.0
    def net(self): return sum(self.pnl)
    def wr(self): return len([p for p in self.pnl if p > 0]) / len(self.pnl) * 100 if self.pnl else 0
    def trades(self): return len(self.pnl)
    def sharpe(self):
        if len(self.pnl) < 2: return 0.0
        m = statistics.mean(self.pnl); s = statistics.stdev(self.pnl)
        return m / s * (252*78)**0.5 if s > 0 else 0.0
    def update_dd(self):
        eq = self.cash
        if eq > self.peak: self.peak = eq
        if self.peak > 0:
            dd = (self.peak - eq) / self.peak * 100
            if dd > self.max_dd: self.max_dd = dd

# ── Data ──────────────────────────────────────────────────────────────────────
def fetch(symbols, start, end):
    client = StockHistoricalDataClient(api_key=ALPACA_KEY, secret_key=ALPACA_SECRET)
    data = {s: [] for s in symbols}
    for sym in symbols:
        print(f"  {sym}...", end=" ", flush=True)
        req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute, start=start, end=end,
                               feed=DataFeed.IEX if DataFeed else None)
        bs = client.get_stock_bars(req)
        if sym in bs.data:
            for b in bs.data[sym]:
                data[sym].append({"timestamp": b.timestamp, "open": float(b.open),
                    "high": float(b.high), "low": float(b.low),
                    "close": float(b.close), "volume": float(b.volume)})
        print(f"{len(data[sym])} bars")
    return data

def to_5min(bars):
    out = []; i = 0
    while i + 4 < len(bars):
        c = bars[i:i+5]
        out.append({"timestamp": c[0]["timestamp"], "open": c[0]["open"],
            "high": max(b["high"] for b in c), "low": min(b["low"] for b in c),
            "close": c[-1]["close"], "volume": sum(b["volume"] for b in c)})
        i += 5
    return out

def by_day(bars):
    d = {}
    for b in bars:
        k = b["timestamp"].strftime("%Y-%m-%d")
        d.setdefault(k, []).append(b)
    return d

# ── VWAP strategy (parameterized) ────────────────────────────────────────────
def run_vwap(bars_5m_by_sym, bounce_pct, tp_pct, sl_pct,
             use_ema=False, use_midday=False, ema_period=20, name="VWAP"):
    res = Result(name=name)
    positions = {}
    for sym, bars in bars_5m_by_sym.items():
        days = by_day(bars)
        for dk in sorted(days):
            db = days[dk]; intra = []; closes = []
            for bar in db:
                intra.append(bar); closes.append(bar["close"])
                p = bar["close"]; ts = bar["timestamp"]
                vw = _vwap(intra)
                if vw is None: continue
                pos = positions.get(sym)
                if pos:
                    pct = (p - pos.entry) / pos.entry * 100
                    if p >= vw or pct >= tp_pct:
                        pnl = (p - pos.entry) * pos.qty
                        res.pnl.append(pnl); res.cash += p * pos.qty
                        del positions[sym]; res.update_dd(); continue
                    if pct <= -sl_pct:
                        pnl = (p - pos.entry) * pos.qty
                        res.pnl.append(pnl); res.cash += p * pos.qty
                        del positions[sym]; res.update_dd(); continue
                if sym not in positions and vw > 0:
                    if use_midday and _is_midday(ts): continue
                    if use_ema and len(closes) >= ema_period:
                        ev = _ema(closes, ema_period)
                        if ev is not None and p < ev: continue
                    dev = (vw - p) / vw * 100
                    if dev >= bounce_pct:
                        qty = ORDER_SIZE_USD / p; cost = qty * p
                        if res.cash >= cost:
                            res.cash -= cost
                            positions[sym] = Pos(p, qty)
                            res.update_dd()
            if sym in positions and db:
                pos = positions[sym]; ep = db[-1]["close"]
                pnl = (ep - pos.entry) * pos.qty
                res.pnl.append(pnl); res.cash += ep * pos.qty
                del positions[sym]; res.update_dd()
    return res

# ── ORB strategy (parameterized) ─────────────────────────────────────────────
def run_orb(bars_1m_by_sym, bo_pct, tp_pct, sl_pct,
            use_ema=False, use_midday=False, use_trail=False,
            trail_act=0.5, trail_dist=0.3, ema_period=20, name="ORB"):
    res = Result(name=name)
    positions = {}
    bars_5m = {s: to_5min(b) for s, b in bars_1m_by_sym.items()}
    for sym, bars in bars_1m_by_sym.items():
        days = by_day(bars); d5 = by_day(bars_5m.get(sym, []))
        for dk in sorted(days):
            db = days[dk]
            if len(db) < 16: continue
            rh = max(b["high"] for b in db[:15])
            buf = rh * (1 + bo_pct / 100)
            traded = False
            c5 = [b["close"] for b in d5.get(dk, [])]
            for bar in db[15:]:
                p = bar["close"]; ts = bar["timestamp"]
                pos = positions.get(sym)
                if pos:
                    pct = (p - pos.entry) / pos.entry * 100
                    if use_trail:
                        if p > pos.high: pos.high = p
                        pp = (pos.high - pos.entry) / pos.entry * 100
                        if pp >= trail_act:
                            ts_price = pos.high * (1 - trail_dist / 100)
                            if p <= ts_price:
                                pnl = (p - pos.entry) * pos.qty
                                res.pnl.append(pnl); res.cash += p * pos.qty
                                del positions[sym]; res.update_dd(); continue
                    if pct >= tp_pct:
                        pnl = (p - pos.entry) * pos.qty
                        res.pnl.append(pnl); res.cash += p * pos.qty
                        del positions[sym]; res.update_dd(); continue
                    if pct <= -sl_pct:
                        pnl = (p - pos.entry) * pos.qty
                        res.pnl.append(pnl); res.cash += p * pos.qty
                        del positions[sym]; res.update_dd(); continue
                if sym not in positions and not traded:
                    if use_midday and _is_midday(ts): continue
                    if use_ema and len(c5) >= ema_period:
                        ev = _ema(c5, ema_period)
                        if ev is not None and p < ev: continue
                    if p >= buf:
                        qty = ORDER_SIZE_USD / p; cost = qty * p
                        if res.cash >= cost:
                            res.cash -= cost
                            positions[sym] = Pos(p, qty)
                            traded = True; res.update_dd()
            if sym in positions and db:
                pos = positions[sym]; ep = db[-1]["close"]
                pnl = (ep - pos.entry) * pos.qty
                res.pnl.append(pnl); res.cash += ep * pos.qty
                del positions[sym]; res.update_dd()
    return res

# ── Report ────────────────────────────────────────────────────────────────────
def print_row(r, months):
    mo = r.net() / months if months > 0 else 0
    print(f"  {r.name:42s} ${r.net():>+10,.0f}  {r.wr():5.1f}%  {r.trades():5d}  {r.max_dd:5.2f}%  {r.sharpe():6.2f}  ${mo:>+8,.0f}")

def banner(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now(timezone.utc)
    periods = [
        ("1 month",  now - timedelta(days=35),  now, 1),
        ("3 months", now - timedelta(days=95),  now, 3),
        ("6 months", now - timedelta(days=185), now, 6),
    ]

    all_1m = {}
    all_5m = {}

    for label, start, end, months in periods:
        banner(f"FETCHING DATA: {label}")
        bars_1m = fetch(SYMBOLS, start, end)
        bars_5m = {s: to_5min(b) for s, b in bars_1m.items()}
        total = sum(len(b) for b in bars_1m.values())
        days = len(set(b["timestamp"].strftime("%Y-%m-%d") for bl in bars_1m.values() for b in bl))
        print(f"  Bars: {total:,} | Trading days: {days}")
        all_1m[label] = bars_1m
        all_5m[label] = bars_5m

    banner("VERIFICATION: Old Deployed vs Stress-Test Optimized")
    print(f"  {'Config':42s} {'Net P&L':>10s}  {'WR%':>5s}  {'Trds':>5s}  {'MaxDD':>5s}  {'Sharpe':>6s}  {'Mo.P&L':>8s}")
    print(f"  {'-'*42} {'-'*10}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*8}")

    for label, start, end, months in periods:
        bars_1m = all_1m[label]
        bars_5m = all_5m[label]
        print(f"\n  -- {label} ({months*~30 if label != '1 month' else '~28'} days) --".replace("~30", "~30").replace("~28", "~28"))

        # Actually just print period heading
        print(f"\n  >> {label} <<")

        # 1) OLD DEPLOYED: VWAP(b=0.12,tp=0.4,sl=0.25) + ORB, with EMA+midday+trail
        old_vwap = run_vwap(bars_5m, 0.12, 0.4, 0.25,
                           use_ema=True, use_midday=True, name="OLD VWAP(b=.12,sl=.25)+EMA+mid")
        old_orb = run_orb(bars_1m, 0.1, 1.5, 0.75,
                         use_ema=True, use_midday=True, use_trail=True,
                         trail_act=0.5, trail_dist=0.3, name="OLD ORB+all filters")
        # Combined old
        old_combo = Result(name="OLD DEPLOYED (VWAP+ORB+filters)")
        old_combo.pnl = old_vwap.pnl + old_orb.pnl
        old_combo.cash = STARTING_CAPITAL + sum(old_combo.pnl)
        old_combo.max_dd = max(old_vwap.max_dd, old_orb.max_dd)
        print_row(old_combo, months)

        # 2) NEW OPTIMIZED: VWAP(b=0.05,tp=0.4,sl=0.50) only, midday only
        new_vwap = run_vwap(bars_5m, 0.05, 0.4, 0.50,
                           use_ema=False, use_midday=True,
                           name="NEW OPTIMIZED VWAP(b=.05,sl=.50)+mid")
        print_row(new_vwap, months)

        # 3) BASELINE: VWAP(b=0.12,tp=0.4,sl=0.25) no filters at all
        base_vwap = run_vwap(bars_5m, 0.12, 0.4, 0.25,
                            use_ema=False, use_midday=False,
                            name="BASELINE VWAP(b=.12,sl=.25) no filter")
        print_row(base_vwap, months)

        # 4) OPTIMIZED NO MIDDAY (to isolate midday effect)
        opt_no_mid = run_vwap(bars_5m, 0.05, 0.4, 0.50,
                             use_ema=False, use_midday=False,
                             name="OPTIMIZED VWAP no midday filter")
        print_row(opt_no_mid, months)

        # 5) VWAP+ORB baseline (for reference)
        base_orb = run_orb(bars_1m, 0.1, 1.5, 0.75,
                          use_ema=False, use_midday=False, use_trail=False,
                          name="BASELINE ORB (no filters)")
        combo_base = Result(name="BASELINE VWAP+ORB (no filters)")
        combo_base.pnl = base_vwap.pnl + base_orb.pnl
        combo_base.cash = STARTING_CAPITAL + sum(combo_base.pnl)
        combo_base.max_dd = max(base_vwap.max_dd, base_orb.max_dd)
        print_row(combo_base, months)

    # Per-symbol breakdown for the optimized config (6mo)
    bars_5m_6mo = all_5m["6 months"]
    banner("PER-SYMBOL BREAKDOWN: Optimized VWAP (6 months)")
    print(f"  {'Symbol':8s} {'Trades':>7s} {'Net P&L':>10s} {'WR%':>7s}")
    print(f"  {'-'*8} {'-'*7} {'-'*10} {'-'*7}")
    for sym in SYMBOLS:
        sym_5m = {sym: bars_5m_6mo[sym]}
        sr = run_vwap(sym_5m, 0.05, 0.4, 0.50, use_ema=False, use_midday=True, name=sym)
        print(f"  {sym:8s} {sr.trades():7d} ${sr.net():>+9,.0f} {sr.wr():6.1f}%")

    banner("SUMMARY")
    print("  If NEW > OLD across all periods --> deploy the optimized config")
    print("  Expected: ~+$1,000/mo vs ~-$1,100/mo = ~$2,100/mo improvement")

if __name__ == "__main__":
    main()
