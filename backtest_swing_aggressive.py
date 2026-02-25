"""
Aggressive Swing Strategy Backtest — $500 Account + $400/mo Deposits
=====================================================================
Tests the production swing strategy with:
  - $500 starting capital
  - $400 monthly deposits (compounding)
  - Aggressive 3% risk per trade
  - Max 2 concurrent positions
  - 1-5 day holds (no PDT)
  - Fractional shares (Alpaca)

Uses 6 months of real market data to project growth.
"""

import os
import sys
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from decimal import Decimal

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Load env
for env_path in [".env", "deploy/.env.production"]:
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def fetch_daily_bars(symbol, start, end, api_key, api_secret):
    """Fetch daily OHLCV bars from Alpaca."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key, api_secret)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(request)

    result = []
    for bar in bars[symbol]:
        result.append({
            "timestamp": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
        })
    return result


def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    recent = deltas[-period:]
    gains = [d for d in recent if d > 0]
    losses = [-d for d in recent if d < 0]
    avg_gain = sum(gains) / period if gains else 0.0001
    avg_loss = sum(losses) / period if losses else 0.0001
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(closes, period=20):
    if len(closes) < period:
        return closes[-1] if closes else 0
    multiplier = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calc_atr(bars, period=14):
    if len(bars) < period + 1:
        return 0
    true_ranges = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i]['high'], bars[i]['low'], bars[i - 1]['close']
        tr = max(h - l, abs(h - pc), abs(l - pc))
        true_ranges.append(tr)
    return sum(true_ranges[-period:]) / period


def avg_volume(bars, period=20):
    if len(bars) < period:
        return bars[-1].get('volume', 0) if bars else 0
    vols = [b.get('volume', 0) for b in bars[-period:]]
    return sum(vols) / len(vols) if vols else 0


def is_bullish_candle(bar):
    o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
    body = abs(c - o)
    full_range = h - l
    if full_range == 0:
        return False
    lower_wick = min(o, c) - l
    if c > o and lower_wick > body * 2 and body / full_range < 0.35:
        return True
    if c > o and body / full_range > 0.6:
        return True
    return False


def run_backtest():
    api_key = os.environ.get("ALPACA_API_KEY", "")
    api_secret = os.environ.get("ALPACA_API_SECRET", "") or os.environ.get("ALPACA_SECRET_KEY", "")

    if not api_key:
        print("ERROR: Set ALPACA_API_KEY / ALPACA_SECRET_KEY")
        return

    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "GOOGL", "AMZN"]

    # 6-month test period
    end_date = datetime(2026, 2, 21, tzinfo=timezone.utc)
    start_date = datetime(2025, 8, 21, tzinfo=timezone.utc)
    fetch_start = start_date - timedelta(days=80)

    # ── AGGRESSIVE PARAMS ──
    STARTING_CAPITAL = 500.0
    MONTHLY_DEPOSIT = 400.0
    RISK_PER_TRADE_PCT = 3.0      # 3% of equity per trade
    TAKE_PROFIT_PCT = 3.0         # Lower TP to actually hit (was 4%)
    STOP_LOSS_PCT = 1.5           # Tighter SL to cut losers fast (was 2.5%)
    TRAILING_ACT_PCT = 1.5        # Activate trailing earlier at +1.5% (was 2%)
    TRAILING_OFFSET_PCT = 0.75    # Tight trail to lock in gains (was 1.5%)
    MAX_HOLD_DAYS = 5
    MAX_POSITIONS = 3             # Allow 3 concurrent (was 2)
    RSI_OVERSOLD = 38.0           # Slightly less restrictive (was 35)
    VOLUME_SURGE = 1.3            # Lower threshold for more signals (was 1.5)
    MIN_CONFLUENCE = 2

    print("=" * 70)
    print("  AGGRESSIVE SWING BACKTEST — $500 + $400/mo DEPOSITS")
    print("=" * 70)
    print(f"  Period:       {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Symbols:      {', '.join(symbols)}")
    print(f"  Start Capital: ${STARTING_CAPITAL:,.0f}")
    print(f"  Monthly Add:   ${MONTHLY_DEPOSIT:,.0f}")
    print(f"  Risk/Trade:    {RISK_PER_TRADE_PCT}% of equity")
    print(f"  TP: {TAKE_PROFIT_PCT}%  SL: {STOP_LOSS_PCT}%  Trail: +{TRAILING_ACT_PCT}%/{TRAILING_OFFSET_PCT}%")
    print(f"  Max Hold: {MAX_HOLD_DAYS}d  Max Positions: {MAX_POSITIONS}")
    print()

    # Fetch data
    print("  Fetching daily bars...")
    all_bars = {}
    for sym in symbols:
        try:
            bars = fetch_daily_bars(sym, fetch_start, end_date, api_key, api_secret)
            all_bars[sym] = bars
            print(f"    {sym}: {len(bars)} bars")
        except Exception as e:
            print(f"    {sym}: ERROR - {e}")
    print()

    # Build trading calendar
    all_dates = set()
    for sym, bars in all_bars.items():
        for bar in bars:
            d = bar['timestamp'].date() if hasattr(bar['timestamp'], 'date') else bar['timestamp']
            all_dates.add(d)
    trading_days = sorted([d for d in all_dates if d >= start_date.date()])

    # ── SIMULATION ──
    equity = STARTING_CAPITAL
    cash = STARTING_CAPITAL
    total_deposited = STARTING_CAPITAL
    active_positions = {}
    completed_trades = []
    daily_equity = []
    max_equity = STARTING_CAPITAL
    max_drawdown = 0
    last_deposit_month = start_date.month

    print(f"  Simulating {len(trading_days)} trading days...")
    print()

    for day in trading_days:
        # Monthly deposit on 1st trading day of new month
        if day.month != last_deposit_month:
            cash += MONTHLY_DEPOSIT
            total_deposited += MONTHLY_DEPOSIT
            last_deposit_month = day.month
            # print(f"  💰 Monthly deposit +${MONTHLY_DEPOSIT:.0f} on {day} | Cash=${cash:.2f}")

        # ── CHECK EXITS ──
        symbols_to_close = []
        for sym, pos in active_positions.items():
            today_bar = None
            for bar in all_bars.get(sym, []):
                bd = bar['timestamp'].date() if hasattr(bar['timestamp'], 'date') else bar['timestamp']
                if bd == day:
                    today_bar = bar
                    break
            if today_bar is None:
                continue

            current_price = today_bar['close']
            pos['high'] = max(pos['high'], today_bar['high'])
            pos['low'] = min(pos['low'], today_bar['low'])

            signal = pos['signal']
            exit_reason = None
            exit_price = current_price

            # Stop loss (use intraday low)
            if today_bar['low'] <= signal['stop_loss']:
                exit_reason = "STOP_LOSS"
                exit_price = signal['stop_loss']
            # Take profit (use intraday high)
            elif today_bar['high'] >= signal['take_profit']:
                exit_reason = "TAKE_PROFIT"
                exit_price = signal['take_profit']
            # Trailing stop
            elif pos['high'] >= signal['trailing_activation']:
                trail_stop = pos['high'] * (1 - TRAILING_OFFSET_PCT / 100)
                if today_bar['low'] <= trail_stop:
                    exit_reason = "TRAILING_STOP"
                    exit_price = trail_stop
            # Time stop
            if exit_reason is None:
                days_held = (day - pos['entry_date']).days
                if days_held >= MAX_HOLD_DAYS:
                    exit_reason = "TIME_STOP"
                    exit_price = current_price

            if exit_reason:
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                cash += pos['shares'] * exit_price

                completed_trades.append({
                    'symbol': sym,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'pnl_pct': (pnl / (pos['entry_price'] * pos['shares'])) * 100,
                    'entry_date': str(pos['entry_date']),
                    'exit_date': str(day),
                    'days_held': (day - pos['entry_date']).days,
                    'exit_reason': exit_reason,
                    'equity_at_entry': pos['equity_at_entry'],
                })
                symbols_to_close.append(sym)

        for sym in symbols_to_close:
            del active_positions[sym]

        # ── CHECK ENTRIES ──
        for sym in symbols:
            if sym in active_positions:
                continue
            if len(active_positions) >= MAX_POSITIONS:
                break

            bars = all_bars.get(sym, [])
            history = []
            today_bar = None
            for bar in bars:
                bd = bar['timestamp'].date() if hasattr(bar['timestamp'], 'date') else bar['timestamp']
                if bd <= day:
                    history.append(bar)
                    if bd == day:
                        today_bar = bar

            if today_bar is None or len(history) < 55:
                continue

            closes = [b['close'] for b in history]
            price = today_bar['close']

            # Indicators
            rsi_now = calc_rsi(closes)
            rsi_prev = calc_rsi(closes[:-1]) if len(closes) > 15 else rsi_now
            ema_20 = calc_ema(closes, 20)
            ema_50 = calc_ema(closes, 50)
            atr = calc_atr(history)
            avg_vol = avg_volume(history)

            if atr <= 0 or avg_vol <= 0:
                continue

            # Score confluence
            reasons = []
            confidence = 0

            if rsi_prev < RSI_OVERSOLD and rsi_now >= RSI_OVERSOLD:
                reasons.append(f"RSI bounce {rsi_prev:.0f}->{rsi_now:.0f}")
                confidence += 25
            elif rsi_now < RSI_OVERSOLD + 5 and rsi_now > rsi_prev:
                reasons.append(f"RSI recovering {rsi_now:.0f}")
                confidence += 15

            dist_to_ema = (price - ema_20) / ema_20 * 100
            if -1.5 < dist_to_ema < 0.5 and ema_20 > ema_50:
                reasons.append(f"Near rising 20-EMA")
                confidence += 20

            if today_bar['volume'] > avg_vol * VOLUME_SURGE:
                reasons.append(f"Volume surge")
                confidence += 15

            if is_bullish_candle(today_bar):
                reasons.append("Bullish candle")
                confidence += 20

            if price > ema_20 and ema_20 > ema_50:
                reasons.append("Above 20/50 EMA")
                confidence += 15

            dist_to_50 = (price - ema_50) / ema_50 * 100
            if 0 < dist_to_50 < 1.0 and ema_20 > ema_50:
                reasons.append("50-EMA bounce")
                confidence += 15

            if len(reasons) < MIN_CONFLUENCE:
                continue

            # Calculate position size
            # Update equity estimate
            position_value = 0
            for s, p in active_positions.items():
                for bar in all_bars.get(s, []):
                    bd = bar['timestamp'].date() if hasattr(bar['timestamp'], 'date') else bar['timestamp']
                    if bd == day:
                        position_value += p['shares'] * bar['close']
                        break
            equity = cash + position_value

            atr_pct = (atr / price) * 100
            sl_pct = max(STOP_LOSS_PCT, atr_pct * 1.5)
            sl_pct = min(sl_pct, 5.0)
            tp_pct = max(TAKE_PROFIT_PCT, sl_pct * 2.5)

            sl_price = price * (1 - sl_pct / 100)
            tp_price = price * (1 + tp_pct / 100)
            trail_act = price * (1 + TRAILING_ACT_PCT / 100)

            # Risk-based sizing
            risk_dollars = equity * (RISK_PER_TRADE_PCT / 100)
            sl_distance = abs(price - sl_price)
            if sl_distance <= 0:
                continue

            shares = risk_dollars / sl_distance
            max_shares = equity * 2 * 0.50 / price  # 50% of buying power
            shares = min(shares, max_shares)
            shares = round(shares, 4)  # Fractional

            if shares <= 0:
                continue
            cost = shares * price
            if cost > cash:
                # Reduce to what we can afford
                shares = round(cash * 0.95 / price, 4)
                if shares <= 0:
                    continue
                cost = shares * price

            cash -= cost

            active_positions[sym] = {
                'signal': {
                    'stop_loss': sl_price,
                    'take_profit': tp_price,
                    'trailing_activation': trail_act,
                },
                'shares': shares,
                'entry_price': price,
                'entry_date': day,
                'high': today_bar['high'],
                'low': today_bar['low'],
                'equity_at_entry': equity,
            }

        # Daily equity calc
        position_value = 0
        for sym, pos in active_positions.items():
            for bar in all_bars.get(sym, []):
                bd = bar['timestamp'].date() if hasattr(bar['timestamp'], 'date') else bar['timestamp']
                if bd == day:
                    position_value += pos['shares'] * bar['close']
                    break
        equity = cash + position_value
        daily_equity.append({'date': str(day), 'equity': equity})

        max_equity = max(max_equity, equity)
        dd = (max_equity - equity) / max_equity * 100
        max_drawdown = max(max_drawdown, dd)

    # ── RESULTS ──
    wins = [t for t in completed_trades if t['pnl'] > 0]
    losses = [t for t in completed_trades if t['pnl'] <= 0]

    total_pnl = sum(t['pnl'] for t in completed_trades)
    win_rate = len(wins) / len(completed_trades) * 100 if completed_trades else 0
    avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
    gross_wins = sum(t['pnl'] for t in wins)
    gross_losses = abs(sum(t['pnl'] for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    avg_days = sum(t['days_held'] for t in completed_trades) / len(completed_trades) if completed_trades else 0

    trading_months = len(trading_days) / 21
    monthly_pnl_from_trading = total_pnl / trading_months if trading_months > 0 else 0
    total_return_on_deposits = (equity - total_deposited) / total_deposited * 100

    exit_reasons = defaultdict(int)
    for t in completed_trades:
        exit_reasons[t['exit_reason']] += 1

    print("=" * 70)
    print("  AGGRESSIVE SWING RESULTS — $500 + $400/mo")
    print("=" * 70)
    print(f"  Starting Capital:     $  {STARTING_CAPITAL:>10,.2f}")
    print(f"  Total Deposited:      $  {total_deposited:>10,.2f}")
    print(f"  Final Equity:         $  {equity:>10,.2f}")
    print(f"  Trading P&L:          $  {total_pnl:>10,.2f}  ({total_pnl/STARTING_CAPITAL*100:+.1f}% on initial)")
    print(f"  Return on Deposits:      {total_return_on_deposits:>8.1f}%")
    print(f"  Max Drawdown:                {max_drawdown:>5.1f}%")
    print()
    print(f"  Total Trades:            {len(completed_trades):>5}")
    print(f"  Wins / Losses:        {len(wins):>4} / {len(losses)}")
    print(f"  Win Rate:                {win_rate:>8.1f}%")
    print(f"  Avg Win:              $  {avg_win:>10,.2f}")
    print(f"  Avg Loss:             $  {avg_loss:>10,.2f}")
    print(f"  Profit Factor:           {profit_factor:>8.2f}")
    print(f"  Avg Hold (days):         {avg_days:>8.1f}")
    print()
    print(f"  Exit Reasons:")
    for reason, count in sorted(exit_reasons.items()):
        print(f"    {reason:<20s} {count:>4}")
    print()
    print(f"  Monthly Trading P&L:  $  {monthly_pnl_from_trading:>10,.2f}")
    print()

    # ── EQUITY CURVE BY MONTH ──
    print("-" * 70)
    print("  MONTHLY EQUITY GROWTH (actual, with deposits)")
    print("-" * 70)
    month_equity = {}
    for entry in daily_equity:
        month_key = entry['date'][:7]  # YYYY-MM
        month_equity[month_key] = entry['equity']

    prev_eq = STARTING_CAPITAL
    for month, eq in sorted(month_equity.items()):
        change = eq - prev_eq
        print(f"  {month}:  ${eq:>10,.2f}  ({change:>+8,.2f})")
        prev_eq = eq
    print()

    # ── GROWTH PROJECTIONS (forward-looking) ──
    print("-" * 70)
    print("  12-MONTH FORWARD PROJECTION (compound + $400/mo deposits)")
    print("-" * 70)

    if total_pnl > 0 and trading_months > 0:
        monthly_return_pct = monthly_pnl_from_trading / ((STARTING_CAPITAL + equity) / 2) * 100
        balance = equity  # Start from current end equity
        total_dep = total_deposited

        for month in range(1, 13):
            trading_gain = balance * (monthly_return_pct / 100)
            balance += trading_gain + MONTHLY_DEPOSIT
            total_dep += MONTHLY_DEPOSIT
            print(f"  Month {month:>2}:  ${balance:>10,.2f}  (gain: ${trading_gain:>8,.2f}, dep: +$400)")

        print()
        print(f"  Projected 12-month equity: ${balance:,.2f}")
        print(f"  Total deposited by then:   ${total_dep:,.2f}")
        print(f"  Trading profits:           ${balance - total_dep:,.2f}")
        print()

        # Milestones
        print("-" * 70)
        print("  MILESTONES")
        print("-" * 70)
        bal = equity
        month = 0
        milestones = [1000, 2500, 5000, 10000, 25000, 50000]
        hit = set()
        while bal < 50001 and month < 120:
            bal += bal * (monthly_return_pct / 100) + MONTHLY_DEPOSIT
            month += 1
            for m in milestones:
                if bal >= m and m not in hit:
                    hit.add(m)
                    years = month / 12
                    print(f"  ${m:>6,} reached in {month:>3} months ({years:.1f} years)")
    else:
        print("  Strategy was not profitable — no projections.")

    print()

    # ── TOP TRADES ──
    if completed_trades:
        print("-" * 70)
        print("  TOP 5 WINNERS")
        print("-" * 70)
        for t in sorted(completed_trades, key=lambda x: x['pnl'], reverse=True)[:5]:
            print(f"  {t['symbol']:>5} | ${t['entry_price']:.2f} -> ${t['exit_price']:.2f} | "
                  f"P&L: ${t['pnl']:>+8.2f} ({t['pnl_pct']:+.1f}%) | {t['days_held']}d | {t['exit_reason']}")

        print()
        print("-" * 70)
        print("  TOP 5 LOSERS")
        print("-" * 70)
        for t in sorted(completed_trades, key=lambda x: x['pnl'])[:5]:
            print(f"  {t['symbol']:>5} | ${t['entry_price']:.2f} -> ${t['exit_price']:.2f} | "
                  f"P&L: ${t['pnl']:>+8.2f} ({t['pnl_pct']:+.1f}%) | {t['days_held']}d | {t['exit_reason']}")

    print()
    print("=" * 70)

    # Save results
    results = {
        "starting_capital": STARTING_CAPITAL,
        "monthly_deposit": MONTHLY_DEPOSIT,
        "total_deposited": total_deposited,
        "final_equity": equity,
        "trading_pnl": total_pnl,
        "return_on_deposits_pct": total_return_on_deposits,
        "max_drawdown_pct": max_drawdown,
        "total_trades": len(completed_trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_hold_days": avg_days,
        "monthly_trading_pnl": monthly_pnl_from_trading,
        "daily_equity": daily_equity,
        "trades": completed_trades,
    }
    with open("backtest_swing_aggressive_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to backtest_swing_aggressive_results.json")


if __name__ == "__main__":
    run_backtest()
