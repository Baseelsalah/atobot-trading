"""AtoBot Crypto Backtester — CryptoSwingStrategy simulation.

Fetches historical 4H crypto bars from Alpaca and simulates the
CryptoSwingStrategy logic:
  - RSI oversold bounce, EMA proximity, volume surge, bullish candle,
    EMA stack, recovery from low → confluence scoring (need ≥2)
  - BTC trend gate for non-BTC pairs (20-EMA > 50-EMA on daily)
  - Stop loss 3%, take profit 5%, trailing stop +2.5%/1.5%
  - Time stop 7 days max hold
  - Position sizing: 4% risk / 3% stop = ~133% of equity capped at 50%
  - 25 bps taker fee per side (0.50% round-trip)

Usage:
    python backtest_crypto.py
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ── Config ────────────────────────────────────────────────────────────────────

SYMBOLS = ["BTC/USD", "ETH/USD"]
STARTING_CAPITAL = 500.0   # Small account growth scenario

# Strategy params (must mirror CryptoSwingStrategy defaults)
RSI_PERIOD = 14
RSI_OVERSOLD = 35.0
RSI_OVERBOUGHT = 75.0
VOLUME_SURGE_MULT = 1.5
MIN_CONFLUENCE = 2

TAKE_PROFIT_PCT = 5.0
STOP_LOSS_PCT = 3.0
TRAILING_ACTIVATION_PCT = 2.5
TRAILING_OFFSET_PCT = 1.5
MAX_HOLD_DAYS = 7
MAX_POSITIONS = 2
RISK_PER_TRADE_PCT = 4.0
FEE_BPS = 25.0          # 25 bps per side
EQUITY_CAP = 500.0       # Cap used equity for sizing

# BTC trend gate
BTC_TREND_GATE = True

# Test periods
PERIODS = [
    ("6-Month", datetime(2024, 8, 1, tzinfo=timezone.utc), datetime(2025, 2, 20, tzinfo=timezone.utc)),
    ("3-Month", datetime(2024, 11, 20, tzinfo=timezone.utc), datetime(2025, 2, 20, tzinfo=timezone.utc)),
    ("Bull Run", datetime(2024, 10, 1, tzinfo=timezone.utc), datetime(2024, 12, 31, tzinfo=timezone.utc)),
    ("Consolidation", datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 2, 20, tzinfo=timezone.utc)),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def rsi(closes: list[float], period: int = 14) -> float | None:
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


def ema(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    multiplier = 2.0 / (period + 1)
    ema_val = sum(closes[:period]) / period
    for price in closes[period:]:
        ema_val = (price - ema_val) * multiplier + ema_val
    return ema_val


def atr(bars: list[dict], period: int = 14) -> float | None:
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


def avg_volume(volumes: list[float], period: int = 20) -> float:
    if not volumes:
        return 0.0
    if len(volumes) < period:
        return sum(volumes) / len(volumes)
    return sum(volumes[-period:]) / period


def is_bullish_candle(bars: list[dict]) -> bool:
    """Check if the last candle is a hammer or bullish engulfing."""
    if len(bars) < 2:
        return False
    last = bars[-1]
    prev = bars[-2]

    body = last["close"] - last["open"]
    full_range = last["high"] - last["low"]
    if full_range <= 0:
        return False

    # Hammer
    lower_wick = min(last["open"], last["close"]) - last["low"]
    if body > 0 and lower_wick > body * 2 and lower_wick > full_range * 0.5:
        return True

    # Bullish engulfing
    if (prev["close"] < prev["open"] and
            last["close"] > last["open"] and
            last["close"] > prev["open"] and
            last["open"] < prev["close"]):
        return True

    return False


def compute_confluence(bars: list[dict], current_price: float) -> int:
    """Compute entry confluence score (0-6)."""
    confluence = 0
    closes = [b["close"] for b in bars]

    # 1. RSI oversold bounce
    rsi_val = rsi(closes, RSI_PERIOD)
    if rsi_val is not None and rsi_val < RSI_OVERSOLD:
        # Check if price is recovering (last change positive)
        if len(closes) >= 2 and closes[-1] > closes[-2]:
            confluence += 1

    # 2. Price near 20-EMA support
    ema20 = ema(closes, 20)
    if ema20 is not None:
        distance_pct = abs(current_price - ema20) / ema20 * 100
        atr_val = atr(bars, 14)
        atr_distance = (atr_val / current_price * 100) * 1.5 if atr_val else 3.0
        if distance_pct <= atr_distance and current_price >= ema20 * 0.98:
            confluence += 1

    # 3. Volume surge
    volumes = [b["volume"] for b in bars]
    avg_vol = avg_volume(volumes, 20)
    curr_vol = volumes[-1] if volumes else 0
    if avg_vol > 0 and curr_vol > avg_vol * VOLUME_SURGE_MULT:
        confluence += 1

    # 4. Bullish candle
    if is_bullish_candle(bars):
        confluence += 1

    # 5. EMA stack (20 > 50 = uptrend)
    ema50 = ema(closes, 50)
    if ema20 is not None and ema50 is not None and ema20 > ema50:
        confluence += 1

    # 6. Recovery from recent low
    if len(bars) >= 10:
        recent_lows = [b["low"] for b in bars[-10:]]
        recent_highs = [b["high"] for b in bars[-10:]]
        recent_low = min(recent_lows)
        recent_high = max(recent_highs)
        if recent_high > recent_low:
            position_in_range = (current_price - recent_low) / (recent_high - recent_low)
            if 0.4 <= position_in_range <= 0.7:
                confluence += 1

    return confluence


def check_btc_trend(btc_closes: list[float]) -> bool:
    """Check if BTC is in uptrend (20-EMA > 50-EMA)."""
    if len(btc_closes) < 50:
        return True  # Fail-open
    ema20 = ema(btc_closes, 20)
    ema50 = ema(btc_closes, 50)
    if ema20 is None or ema50 is None:
        return True
    return ema20 > ema50


# ── Trade / Position tracking ────────────────────────────────────────────────

@dataclass
class CryptoTrade:
    symbol: str
    side: str          # BUY, SELL
    price: float
    quantity: float
    timestamp: datetime
    reason: str = ""
    fee: float = 0.0


@dataclass
class CryptoPosition:
    symbol: str
    entry_price: float
    quantity: float
    entry_time: datetime
    highest_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_active: bool = False

    def __post_init__(self):
        self.highest_price = self.entry_price
        self.stop_loss = self.entry_price * (1 - STOP_LOSS_PCT / 100)
        self.take_profit = self.entry_price * (1 + TAKE_PROFIT_PCT / 100)


@dataclass
class BacktestResult:
    name: str
    period_label: str = ""
    trades: list[CryptoTrade] = field(default_factory=list)
    closed_pnl: list[float] = field(default_factory=list)
    open_positions: dict[str, CryptoPosition] = field(default_factory=dict)
    cash: float = STARTING_CAPITAL
    peak_equity: float = STARTING_CAPITAL
    max_drawdown_pct: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    total_fees: float = 0.0

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

    def update_equity(self, current_prices: dict[str, float]) -> None:
        """Update equity curve and drawdown."""
        open_value = sum(
            pos.quantity * current_prices.get(pos.symbol, pos.entry_price)
            for pos in self.open_positions.values()
        )
        equity = self.cash + open_value
        self.equity_curve.append(equity)
        if equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity * 100
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_4h_bars(symbols: list[str], start: datetime, end: datetime) -> dict[str, list[dict]]:
    """Fetch 4H bars from Alpaca crypto API, chunked by month."""
    print("Fetching historical 4H crypto bars from Alpaca...")
    client = CryptoHistoricalDataClient()
    tf = TimeFrame(4, TimeFrameUnit.Hour)
    all_bars: dict[str, list[dict]] = {s: [] for s in symbols}

    for sym in symbols:
        print(f"  {sym}...", end=" ", flush=True)
        # Fetch in monthly chunks to avoid timeouts
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=31), end)
            req = CryptoBarsRequest(
                symbol_or_symbols=sym,
                timeframe=tf,
                start=chunk_start,
                end=chunk_end,
            )
            try:
                barset = client.get_crypto_bars(req)
                data = barset.data.get(sym, [])
                for bar in data:
                    all_bars[sym].append({
                        "timestamp": bar.timestamp,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                    })
            except Exception as exc:
                print(f"\n    WARNING: Failed chunk {chunk_start.date()}-{chunk_end.date()}: {exc}")
            chunk_start = chunk_end

        # Deduplicate by timestamp
        seen = set()
        deduped = []
        for b in all_bars[sym]:
            ts_key = b["timestamp"].isoformat()
            if ts_key not in seen:
                seen.add(ts_key)
                deduped.append(b)
        all_bars[sym] = sorted(deduped, key=lambda x: x["timestamp"])
        print(f"{len(all_bars[sym])} bars")

    return all_bars


def fetch_daily_bars(symbol: str, start: datetime, end: datetime) -> list[dict]:
    """Fetch daily bars for BTC trend gate."""
    client = CryptoHistoricalDataClient()
    tf = TimeFrame(1, TimeFrameUnit.Day)
    all_bars: list[dict] = []

    chunk_start = start - timedelta(days=60)  # Extra lookback for EMAs
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=90), end)
        req = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=chunk_start,
            end=chunk_end,
        )
        try:
            barset = client.get_crypto_bars(req)
            data = barset.data.get(symbol, [])
            for bar in data:
                all_bars.append({
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                })
        except Exception as exc:
            print(f"    WARNING: BTC daily fetch error: {exc}")
        chunk_start = chunk_end

    # Deduplicate
    seen = set()
    deduped = []
    for b in all_bars:
        ts_key = b["timestamp"].isoformat()
        if ts_key not in seen:
            seen.add(ts_key)
            deduped.append(b)
    return sorted(deduped, key=lambda x: x["timestamp"])


# ── Simulation ────────────────────────────────────────────────────────────────

def calc_fee(price: float, quantity: float, fee_bps: float = FEE_BPS) -> float:
    """Calculate taker fee."""
    return price * quantity * (fee_bps / 10000)


def calc_position_size(equity: float, price: float, params: dict) -> float:
    """Risk-based position sizing with fee adjustment."""
    equity_cap = params.get("equity_cap", EQUITY_CAP)
    risk_pct = params.get("risk_per_trade", RISK_PER_TRADE_PCT)
    sl_pct = params.get("stop_loss", STOP_LOSS_PCT)
    fee_bps = params.get("fee_bps", FEE_BPS)

    eq = min(equity, equity_cap) if equity_cap > 0 else equity
    if eq <= 0:
        return 0.0

    risk_amount = eq * (risk_pct / 100)
    stop_distance = sl_pct / 100
    fee_pct = (fee_bps / 10000) * 2  # Round-trip
    effective_risk = stop_distance + fee_pct

    position_usd = risk_amount / effective_risk
    max_position = eq * 0.50
    position_usd = min(position_usd, max_position)
    position_usd = min(position_usd, equity * 0.95)  # Leave some cash

    qty = position_usd / price
    return round(qty, 6)


def close_position(
    res: BacktestResult, sym: str, price: float, ts: datetime, reason: str,
    fee_bps: float = FEE_BPS,
) -> None:
    """Close a crypto position and record PnL."""
    pos = res.open_positions.get(sym)
    if not pos:
        return

    # Exit fee
    exit_fee = calc_fee(price, pos.quantity, fee_bps)
    res.total_fees += exit_fee

    gross_pnl = (price - pos.entry_price) * pos.quantity
    net_pnl = gross_pnl - exit_fee  # Entry fee already deducted from cash

    res.trades.append(CryptoTrade(sym, "SELL", price, pos.quantity, ts, reason, exit_fee))
    res.cash += price * pos.quantity - exit_fee
    res.closed_pnl.append(net_pnl)
    del res.open_positions[sym]


def run_crypto_backtest(
    bars_4h: dict[str, list[dict]],
    btc_daily: list[dict],
    period_label: str,
    params: dict | None = None,
) -> BacktestResult:
    """Run the crypto swing strategy backtest with configurable params."""
    p = params or {}
    sl_pct = p.get("stop_loss", STOP_LOSS_PCT)
    tp_pct = p.get("take_profit", TAKE_PROFIT_PCT)
    trail_act = p.get("trailing_activation", TRAILING_ACTIVATION_PCT)
    trail_off = p.get("trailing_offset", TRAILING_OFFSET_PCT)
    max_hold = p.get("max_hold_days", MAX_HOLD_DAYS)
    max_pos = p.get("max_positions", MAX_POSITIONS)
    min_conf = p.get("min_confluence", MIN_CONFLUENCE)
    btc_gate = p.get("btc_trend_gate", BTC_TREND_GATE)
    fee_bps = p.get("fee_bps", FEE_BPS)
    cooldown_bars = p.get("cooldown_bars", 0)  # Bars to skip after a stop-loss
    config_name = p.get("name", "Crypto Swing")

    res = BacktestResult(name=config_name, period_label=period_label)

    # Build a unified timeline of all bar timestamps
    all_timestamps: set[datetime] = set()
    for sym_bars in bars_4h.values():
        for b in sym_bars:
            all_timestamps.add(b["timestamp"])
    timeline = sorted(all_timestamps)

    # Build bar-index lookups for O(1) access
    bar_indices: dict[str, dict[datetime, int]] = {}
    for sym, sym_bars in bars_4h.items():
        bar_indices[sym] = {b["timestamp"]: i for i, b in enumerate(sym_bars)}

    # BTC daily closes for trend gate
    btc_daily_closes = [b["close"] for b in btc_daily]
    btc_daily_dates = [b["timestamp"].date() if hasattr(b["timestamp"], "date") else b["timestamp"] for b in btc_daily]

    # Track last evaluation times per symbol (4H granularity)
    eval_done: dict[str, str] = {}
    # Cooldown tracking: symbol -> timestamp of last stop-loss exit
    cooldown_until: dict[str, datetime] = {}

    if not p.get("quiet", False):
        print(f"\n  Simulating {period_label}: {len(timeline)} time steps, {len(bars_4h)} symbols")

    for ts in timeline:
        current_prices: dict[str, float] = {}

        for sym in bars_4h:
            idx_map = bar_indices[sym]
            if ts not in idx_map:
                continue

            idx = idx_map[ts]
            sym_bars = bars_4h[sym]
            bar = sym_bars[idx]
            price = bar["close"]
            current_prices[sym] = price

            # ── Manage existing position ──────────────────────────────
            pos = res.open_positions.get(sym)
            if pos:
                # Update highest price
                if price > pos.highest_price:
                    pos.highest_price = price

                pnl_pct = (price - pos.entry_price) / pos.entry_price * 100

                # 1. Stop loss
                if price <= pos.stop_loss:
                    close_position(res, sym, price, ts, "STOP_LOSS", fee_bps)
                    if cooldown_bars > 0:
                        cooldown_until[sym] = ts + timedelta(hours=4 * cooldown_bars)
                    continue

                # 2. Take profit
                if price >= pos.take_profit:
                    close_position(res, sym, price, ts, "TAKE_PROFIT", fee_bps)
                    continue

                # 3. Trailing stop
                trailing_activation_price = pos.entry_price * (1 + trail_act / 100)
                if price >= trailing_activation_price and not pos.trailing_active:
                    pos.trailing_active = True

                if pos.trailing_active:
                    trail_stop = pos.highest_price * (1 - trail_off / 100)
                    # Only raise stop
                    if trail_stop > pos.stop_loss:
                        pos.stop_loss = trail_stop
                    if price <= trail_stop:
                        close_position(res, sym, price, ts, "TRAILING_STOP", fee_bps)
                        continue

                # 4. Time stop
                days_held = (ts - pos.entry_time).total_seconds() / 86400
                if days_held >= max_hold:
                    close_position(res, sym, price, ts, "TIME_STOP", fee_bps)
                    continue

                continue  # Position managed, skip entry scan

            # ── Entry scan ────────────────────────────────────────────
            # Cooldown check
            if sym in cooldown_until and ts < cooldown_until[sym]:
                continue

            # Only evaluate every 4 hours (dedup by 4H block)
            hour = ts.hour if hasattr(ts, "hour") else 0
            eval_hour = (hour // 4) * 4
            eval_key = f"{ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else ts}-{eval_hour:02d}"
            if eval_done.get(sym) == eval_key:
                continue

            # Max positions check
            if len(res.open_positions) >= max_pos:
                eval_done[sym] = eval_key
                continue

            # BTC trend gate for non-BTC
            if btc_gate and "BTC" not in sym:
                ts_date = ts.date() if hasattr(ts, "date") else ts
                # Use BTC daily closes up to this date
                btc_closes_up_to = []
                for i, d in enumerate(btc_daily_dates):
                    d_cmp = d.date() if hasattr(d, "date") else d
                    if d_cmp <= ts_date:
                        btc_closes_up_to.append(btc_daily_closes[i])
                if not check_btc_trend(btc_closes_up_to):
                    eval_done[sym] = eval_key
                    continue

            # Need enough bars for indicators
            lookback = min(idx + 1, 60)
            history = sym_bars[idx - lookback + 1: idx + 1]
            if len(history) < 30:
                eval_done[sym] = eval_key
                continue

            confluence = compute_confluence(history, price)

            if confluence >= min_conf:
                qty = calc_position_size(res.cash, price, p)
                if qty <= 0:
                    eval_done[sym] = eval_key
                    continue

                notional = qty * price
                if notional < 1.0:
                    eval_done[sym] = eval_key
                    continue

                # Entry fee
                entry_fee = calc_fee(price, qty, fee_bps)

                if res.cash >= notional + entry_fee:
                    res.cash -= notional + entry_fee
                    res.total_fees += entry_fee
                    pos_obj = CryptoPosition(sym, price, qty, ts)
                    # Override SL/TP with parameterized values
                    pos_obj.stop_loss = price * (1 - sl_pct / 100)
                    pos_obj.take_profit = price * (1 + tp_pct / 100)
                    res.open_positions[sym] = pos_obj
                    res.trades.append(
                        CryptoTrade(sym, "BUY", price, qty, ts, f"CONFLUENCE_{confluence}", entry_fee)
                    )

            eval_done[sym] = eval_key

        # Update equity at each timestep
        res.update_equity(current_prices)

    # Close any remaining open positions at last known prices
    for sym in list(res.open_positions.keys()):
        if bars_4h[sym]:
            last_price = bars_4h[sym][-1]["close"]
            last_ts = bars_4h[sym][-1]["timestamp"]
            close_position(res, sym, last_price, last_ts, "END_OF_BACKTEST", fee_bps)

    return res


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(result: BacktestResult) -> None:
    bar = "-" * 65
    print(f"\n{bar}")
    print(f"  {result.name} -- {result.period_label}")
    print(bar)

    n_trades = len(result.closed_pnl)
    if n_trades == 0:
        print("  No trades executed.")
        print(bar)
        return

    # Calculate trading days
    if result.trades:
        first_ts = result.trades[0].timestamp
        last_ts = result.trades[-1].timestamp
        span_days = max((last_ts - first_ts).days, 1)
    else:
        span_days = 1

    buys = [t for t in result.trades if t.side == "BUY"]
    sells = [t for t in result.trades if t.side == "SELL"]

    # Exit reason breakdown
    reasons: dict[str, int] = {}
    for t in sells:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1

    # Monthly estimate
    monthly_pnl = result.net_pnl / span_days * 30 if span_days > 0 else 0

    # Sharpe ratio (annualized from trade returns)
    if len(result.closed_pnl) > 1:
        avg_trade = statistics.mean(result.closed_pnl)
        std_trade = statistics.stdev(result.closed_pnl)
        if std_trade > 0:
            # Roughly 6 trades per month * 12 = 72 trades/year
            sharpe = (avg_trade / std_trade) * (72 ** 0.5)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    curr_wins = 0
    curr_losses = 0
    for pnl in result.closed_pnl:
        if pnl > 0:
            curr_wins += 1
            curr_losses = 0
            max_consec_wins = max(max_consec_wins, curr_wins)
        else:
            curr_losses += 1
            curr_wins = 0
            max_consec_losses = max(max_consec_losses, curr_losses)

    final_equity = STARTING_CAPITAL + result.net_pnl
    return_pct = (result.net_pnl / STARTING_CAPITAL) * 100

    print(f"  Starting Capital:     ${STARTING_CAPITAL:>10,.2f}")
    print(f"  Final Equity:         ${final_equity:>10,.2f}")
    print(f"  Net P&L:              ${result.net_pnl:>+10,.2f}  ({return_pct:+.1f}%)")
    print(f"  Total Fees:           ${result.total_fees:>10,.2f}")
    print()
    print(f"  Trades:               {n_trades:>6}")
    print(f"  Wins / Losses:        {result.wins:>3} / {result.losses}")
    print(f"  Win Rate:             {result.win_rate:>10.1f}%")
    print(f"  Avg Win:              ${result.avg_win:>+10,.2f}")
    print(f"  Avg Loss:             ${result.avg_loss:>+10,.2f}")
    print(f"  Profit Factor:        {result.profit_factor:>10.2f}")
    print(f"  Sharpe (ann.):        {sharpe:>10.2f}")
    print(f"  Max Drawdown:         {result.max_drawdown_pct:>10.1f}%")
    print()
    print(f"  Max Consec Wins:      {max_consec_wins:>6}")
    print(f"  Max Consec Losses:    {max_consec_losses:>6}")
    print()
    print(f"  Est. Monthly P&L:     ${monthly_pnl:>+10,.2f}")
    print(f"  Span:                 {span_days:>6} days")
    print()

    # Exit reason breakdown
    print("  Exit Reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<20} {count:>4}")

    # Per-symbol breakdown
    print()
    print("  Per-Symbol Breakdown:")
    sym_pnl: dict[str, list[float]] = {}
    for i, t in enumerate(sells):
        if t.symbol not in sym_pnl:
            sym_pnl[t.symbol] = []
        if i < len(result.closed_pnl):
            sym_pnl[t.symbol].append(result.closed_pnl[i])
    for sym in sorted(sym_pnl.keys()):
        pnls = sym_pnl[sym]
        total = sum(pnls)
        wins = len([p for p in pnls if p > 0])
        wr = wins / len(pnls) * 100 if pnls else 0
        print(f"    {sym:<10} {len(pnls):>3} trades  PnL=${total:>+8,.2f}  WR={wr:.0f}%")

    # Sample trades
    print()
    print("  Last 5 Trades:")
    for t in result.trades[-10:]:
        fee_str = f" fee=${t.fee:.2f}" if t.fee > 0 else ""
        print(f"    {t.timestamp.strftime('%Y-%m-%d %H:%M')} {t.side:>4} {t.symbol:<10} "
              f"qty={t.quantity:.6f} @ ${t.price:,.2f} [{t.reason}]{fee_str}")

    print(bar)


# ── Main ──────────────────────────────────────────────────────────────────────

# Parameter configurations to sweep
CONFIGS = [
    {
        "name": "A: Original",
        "stop_loss": 3.0, "take_profit": 5.0,
        "trailing_activation": 2.5, "trailing_offset": 1.5,
        "min_confluence": 2, "max_hold_days": 7, "cooldown_bars": 0,
    },
    {
        "name": "B: Wider Stops",
        "stop_loss": 5.0, "take_profit": 8.0,
        "trailing_activation": 4.0, "trailing_offset": 2.0,
        "min_confluence": 2, "max_hold_days": 10, "cooldown_bars": 0,
    },
    {
        "name": "C: Higher Bar",
        "stop_loss": 4.0, "take_profit": 7.0,
        "trailing_activation": 3.0, "trailing_offset": 2.0,
        "min_confluence": 3, "max_hold_days": 7, "cooldown_bars": 3,
    },
    {
        "name": "D: Conservative",
        "stop_loss": 5.0, "take_profit": 10.0,
        "trailing_activation": 5.0, "trailing_offset": 2.5,
        "min_confluence": 3, "max_hold_days": 14, "cooldown_bars": 6,
    },
    {
        "name": "E: Balanced",
        "stop_loss": 4.0, "take_profit": 8.0,
        "trailing_activation": 3.5, "trailing_offset": 2.0,
        "min_confluence": 2, "max_hold_days": 10, "cooldown_bars": 3,
    },
]


def main() -> None:
    print()
    print("=" * 65)
    print("  AtoBot Crypto Backtester -- CryptoSwingStrategy")
    print("  BTC/USD + ETH/USD | 4H Bars | $500 Capital")
    print("=" * 65)
    print()
    print(f"  Symbols:      {', '.join(SYMBOLS)}")
    print(f"  Capital:      ${STARTING_CAPITAL:,.0f}")
    print(f"  Configs:      {len(CONFIGS)} parameter sets to sweep")
    print()

    # Use the longest period to fetch all data at once
    test_period = ("6-Month", datetime(2024, 8, 1, tzinfo=timezone.utc), datetime(2025, 2, 20, tzinfo=timezone.utc))

    global_start = test_period[1]
    global_end = test_period[2]

    print(f"  Data Range:   {global_start.date()} to {global_end.date()}")
    print()

    # Fetch all data
    bars_4h = fetch_4h_bars(SYMBOLS, global_start, global_end)
    total_bars = sum(len(b) for b in bars_4h.values())
    if total_bars == 0:
        print("ERROR: No historical data received.")
        sys.exit(1)
    print(f"\n  Total 4H bars: {total_bars:,}")

    # Fetch BTC daily for trend gate
    btc_daily: list[dict] = []
    if BTC_TREND_GATE:
        print("  Fetching BTC/USD daily bars for trend gate...")
        btc_daily = fetch_daily_bars("BTC/USD", global_start, global_end)
        print(f"  BTC daily bars: {len(btc_daily)}")

    # Filter bars to period
    period_bars: dict[str, list[dict]] = {}
    for sym, sym_bars in bars_4h.items():
        period_bars[sym] = [
            b for b in sym_bars
            if global_start <= b["timestamp"] <= global_end
        ]

    period_btc = [
        b for b in btc_daily
        if global_start - timedelta(days=60) <= b["timestamp"] <= global_end
    ] if btc_daily else []

    # ── Run all configs ───────────────────────────────────────────────
    print("\n" + "=" * 95)
    print("  PARAMETER SWEEP -- 6-Month Period")
    print("=" * 95)

    all_results: list[BacktestResult] = []
    for cfg in CONFIGS:
        result = run_crypto_backtest(period_bars, period_btc, "6-Month", cfg)
        all_results.append(result)
        print_report(result)

    # ── Summary comparison table ──────────────────────────────────────
    print("\n")
    print("=" * 110)
    print("  PARAMETER SWEEP COMPARISON")
    print("=" * 110)
    print(f"  {'Config':<20} {'SL/TP':>8} {'Conflu':>6} {'Net PnL':>10} {'Return':>8} "
          f"{'Win%':>7} {'Trades':>7} {'MaxDD':>7} {'PF':>6} {'Fees':>8} {'Mo.PnL':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*10} {'-'*8} {'-'*7} {'-'*7} {'-'*7} "
          f"{'-'*6} {'-'*8} {'-'*10}")

    for i, r in enumerate(all_results):
        cfg = CONFIGS[i]
        return_pct = (r.net_pnl / STARTING_CAPITAL) * 100
        span_days = 1
        if r.trades:
            span_days = max((r.trades[-1].timestamp - r.trades[0].timestamp).days, 1)
        monthly = r.net_pnl / span_days * 30 if span_days > 0 else 0
        n = len(r.closed_pnl)
        sl_tp = f"{cfg['stop_loss']:.0f}/{cfg['take_profit']:.0f}%"
        conf = str(cfg['min_confluence'])
        print(f"  {r.name:<20} {sl_tp:>8} {conf:>6} ${r.net_pnl:>+8,.0f} {return_pct:>+7.1f}% "
              f"{r.win_rate:>6.1f}% {n:>7} {r.max_drawdown_pct:>6.1f}% "
              f"{r.profit_factor:>5.2f} ${r.total_fees:>6,.2f} ${monthly:>+8,.0f}")

    print("=" * 110)

    # Best config
    if all_results:
        best = max(all_results, key=lambda r: r.net_pnl)
        worst = min(all_results, key=lambda r: r.net_pnl)
        print(f"\n  BEST:  {best.name} -> ${best.net_pnl:+,.2f} ({best.win_rate:.1f}% WR, PF {best.profit_factor:.2f})")
        print(f"  WORST: {worst.name} -> ${worst.net_pnl:+,.2f}")

    # ── Now run best config across all time periods ───────────────────
    best_idx = all_results.index(best) if all_results else 0
    best_cfg = CONFIGS[best_idx]

    print(f"\n\n  Running BEST config ({best_cfg['name']}) across time periods...")
    print("=" * 95)

    period_results: list[BacktestResult] = []
    for label, start, end in PERIODS:
        p_bars: dict[str, list[dict]] = {}
        for sym, sym_bars in bars_4h.items():
            p_bars[sym] = [b for b in sym_bars if start <= b["timestamp"] <= end]
        p_btc = [
            b for b in btc_daily
            if start - timedelta(days=60) <= b["timestamp"] <= end
        ] if btc_daily else []
        r = run_crypto_backtest(p_bars, p_btc, label, best_cfg)
        period_results.append(r)

    print(f"\n  {'Period':<20} {'Net P&L':>10} {'Return':>8} {'Win%':>7} {'Trades':>7} "
          f"{'MaxDD':>7} {'PF':>6} {'Mo.PnL':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*10}")
    for r in period_results:
        return_pct = (r.net_pnl / STARTING_CAPITAL) * 100
        span = max((r.trades[-1].timestamp - r.trades[0].timestamp).days, 1) if r.trades else 1
        monthly = r.net_pnl / span * 30 if span > 0 else 0
        n = len(r.closed_pnl)
        print(f"  {r.period_label:<20} ${r.net_pnl:>+8,.0f} {return_pct:>+7.1f}% "
              f"{r.win_rate:>6.1f}% {n:>7} {r.max_drawdown_pct:>6.1f}% "
              f"{r.profit_factor:>5.2f} ${monthly:>+8,.0f}")

    print("=" * 95)

    profitable_count = sum(1 for r in period_results if r.net_pnl > 0)
    print(f"\n  Profitable periods: {profitable_count}/{len(period_results)}")
    if profitable_count >= len(period_results) * 0.5:
        print("  [OK] Strategy shows consistent profitability")
    else:
        print("  [!!] Strategy needs further tuning")
    print()


if __name__ == "__main__":
    main()
