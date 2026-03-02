"""AtoBot Backtester v2 — uses real strategy classes for accurate simulation.

>>> python backtest.py

Critical improvement over v1:
  v1 was an independent reimplementation of strategy logic; divergence between
  backtest and live code meant P&L numbers were measuring a different algorithm.

v2 fixes this by instantiating the ACTUAL VWAPScalpStrategy and
MomentumStrategy classes from the live codebase. Every entry/exit filter, warm-up
guard, ATR sizing, and confluence gate tested here is identical to what runs live.

Architecture:
  BacktestExchangeClient:
    - Implements BaseExchangeClient; stores all precomputed bar data
    - Uses a timestamp cursor + binary search (bisect) for O(log n) bar slicing
    - Aggregates 1m → 5m → 15m → 1D bars at init time (one-shot O(n))
    - Simulates fills immediately at current bar close + square-root slippage

  BacktestRunner:
    - Iterates all symbols chronologically through 5m bars
    - Calls strategy.on_tick(symbol, price) for each bar
    - Processes returned Order objects, simulates fills, calls on_order_filled
    - Tracks equity curve and P&L per trade

Slippage model (Fix #14):
  Square-root market impact: impact = half_spread + k * sqrt(order_size / ADV)
  where ADV = 20-bar average daily volume in USD, k = 0.10% empirical coefficient.
  This reduces bloated backtest P&L from the previous fixed-0% slippage model.
"""

from __future__ import annotations

import asyncio
import bisect
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal

# ── Alpaca SDK (data only) ─────────────────────────────────────────────────────
try:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

# ── Live codebase imports ──────────────────────────────────────────────────────
from src.config.settings import Settings
from src.exchange.base_client import BaseExchangeClient
from src.models.order import Order, OrderSide, OrderStatus
from src.models.position import Position
from src.risk.risk_manager import RiskManager
from src.strategies.vwap_strategy import VWAPScalpStrategy
from src.strategies.momentum_strategy import MomentumStrategy

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

# Runtime defaults — can be overridden via .env
SYMBOLS: list[str] = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD"]
STARTING_CAPITAL: float = 100_000.0


def _load_settings() -> Settings:
    """Load settings for backtest, with safe fallbacks for missing .env keys."""
    try:
        return Settings()
    except Exception:
        # Minimal settings when .env is absent
        return Settings(
            ALPACA_API_KEY="backtest_placeholder",
            ALPACA_API_SECRET="backtest_placeholder",
            ALPACA_PAPER=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Bar aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _agg_n_min(bars_1m: list[dict], n: int) -> list[dict]:
    """Aggregate 1-minute bars into n-minute bars aligned to the session open.

    Bars are aggregated in sequential groups of n, not clock-aligned.
    OHLCV rules: open=first, high=max, low=min, close=last, volume=sum.
    Complexity: O(len(bars_1m))
    """
    result: list[dict] = []
    for start in range(0, len(bars_1m) - n + 1, n):
        chunk = bars_1m[start : start + n]
        result.append({
            "timestamp": chunk[0]["timestamp"],
            "open":   chunk[0]["open"],
            "high":   max(b["high"]  for b in chunk),
            "low":    min(b["low"]   for b in chunk),
            "close":  chunk[-1]["close"],
            "volume": sum(b["volume"] for b in chunk),
        })
    return result


def _agg_daily(bars_1m: list[dict]) -> list[dict]:
    """Aggregate 1-minute bars into daily OHLCV bars. Complexity: O(n)."""
    days: dict[str, list[dict]] = {}
    for b in bars_1m:
        day = b["timestamp"].strftime("%Y-%m-%d")
        days.setdefault(day, []).append(b)

    result: list[dict] = []
    for day_key in sorted(days):
        chunk = days[day_key]
        result.append({
            "timestamp": chunk[0]["timestamp"].replace(
                hour=13, minute=30, second=0, microsecond=0
            ),  # 09:30 ET approximation
            "open":   chunk[0]["open"],
            "high":   max(b["high"]  for b in chunk),
            "low":    min(b["low"]   for b in chunk),
            "close":  chunk[-1]["close"],
            "volume": sum(b["volume"] for b in chunk),
        })
    return result


def _ts_list(bars: list[dict]) -> list[float]:
    """Extract timestamps as unix floats for binary search. O(n)."""
    return [b["timestamp"].timestamp() for b in bars]


def _slice_up_to(bars: list[dict], ts_idx: list[float], current_ts: float, limit: int) -> list[dict]:
    """Return up to `limit` bars whose timestamp <= current_ts. O(log n + limit)."""
    pos = bisect.bisect_right(ts_idx, current_ts)
    start = max(0, pos - limit)
    return bars[start:pos]


# ═══════════════════════════════════════════════════════════════════════════════
# Slippage model (Fix #14)
# ═══════════════════════════════════════════════════════════════════════════════

def _slippage_pct(order_usd: float, avg_daily_volume_usd: float) -> float:
    """Square-root market impact model.

    Total cost = half_spread + market_impact
      half_spread  ≈ 1bp  (typical for liquid large-cap US equities)
      market_impact = k * sqrt(order_size / ADV)  where k = 10bp

    This replaces the previous 0% slippage assumption. For our order sizes
    (~$5k-$17k) in large-cap stocks (ADV ~$500M), typical slippage is ~1-3bp
    per side, i.e. ~2-6bp round-trip.

    Math:
      participation = $17k / $500M = 0.0034%
      impact = 0.001 * sqrt(0.000034) ≈ 0.000006 = 0.0006%
      total ≈ 0.01% + 0.0006% ≈ 0.011%  per side

    Capped at 50bp to avoid extreme cases in illiquid conditions.
    Complexity: O(1)
    """
    if avg_daily_volume_usd <= 0.0:
        return 0.0002  # 2bp fallback

    half_spread = 0.0001                                         # 1bp
    participation = order_usd / avg_daily_volume_usd
    market_impact = 0.001 * (participation ** 0.5)               # 10bp * sqrt(p)
    return min(half_spread + market_impact, 0.005)               # cap 50bp


# ═══════════════════════════════════════════════════════════════════════════════
# BacktestExchangeClient
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestExchangeClient(BaseExchangeClient):
    """Mock exchange that replays pre-loaded historical bars for backtesting.

    Design goals:
    - Zero network calls during backtest (all data preloaded at __init__)
    - O(log n) bar lookups via binary search on precomputed timestamp lists
    - Realistic fill simulation: current bar close + slippage
    - Tracks cash balance and open notional for equity curve
    """

    def __init__(
        self,
        bars_1m: dict[str, list[dict]],
        starting_cash: float = STARTING_CAPITAL,
    ) -> None:
        # ── Pre-aggregate all timeframes (one-shot, O(n) per symbol) ────────
        self._bars: dict[str, dict[str, list[dict]]] = {}  # symbol -> tf -> bars
        self._ts:   dict[str, dict[str, list[float]]] = {}  # symbol -> tf -> timestamps

        for sym, bars in bars_1m.items():
            self._bars[sym] = {
                "1m":  bars,
                "5m":  _agg_n_min(bars, 5),
                "15m": _agg_n_min(bars, 15),
                "1D":  _agg_daily(bars),
            }
            self._ts[sym] = {
                tf: _ts_list(b) for tf, b in self._bars[sym].items()
            }

        # ── Precompute 20-day ADV in USD per symbol (for slippage model) ───
        self._adv_usd: dict[str, float] = {}
        for sym in bars_1m:
            daily = self._bars[sym]["1D"]
            if len(daily) >= 5:
                recent = daily[-20:]
                self._adv_usd[sym] = sum(
                    b["close"] * b["volume"] for b in recent
                ) / len(recent)
            else:
                self._adv_usd[sym] = 1_000_000.0  # safe default

        self._current_ts: float = 0.0
        self._cash: float = starting_cash
        self._filled_orders: dict[str, dict] = {}    # order_id -> fill_info
        self._order_seq: int = 0

    # ── Cursor management ────────────────────────────────────────────────────

    def set_current_time(self, ts: datetime) -> None:
        """Advance the time cursor. Must be called before on_tick."""
        self._current_ts = ts.timestamp()

    def get_cash(self) -> float:
        return self._cash

    def adjust_cash(self, delta: float) -> None:
        self._cash += delta

    # ── BaseExchangeClient implementation ───────────────────────────────────

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Return latest close price up to current timestamp. O(log n)."""
        bars = self._bars.get(symbol, {}).get("1m", [])
        ts_idx = self._ts.get(symbol, {}).get("1m", [])
        if not bars:
            return Decimal("0")
        pos = bisect.bisect_right(ts_idx, self._current_ts) - 1
        if pos < 0:
            return Decimal("0")
        return Decimal(str(bars[pos]["close"]))

    async def get_order_book(self, symbol: str, limit: int = 10) -> dict:
        price = float(await self.get_ticker_price(symbol))
        return {"bids": [[price * 0.9999, 100]], "asks": [[price * 1.0001, 100]]}

    async def get_klines(self, symbol: str, interval: str, limit: int = 100) -> list:
        """Return up to `limit` OHLCV bars ending at current timestamp.

        Interval normalization: "1m"/"1min" → 1m, "5m"/"5min" → 5m, etc.
        When 1D bars are requested (for trend filter) we return daily bars.
        If a symbol does not exist, return [].

        Complexity: O(log n + limit)
        """
        tf_map = {
            "1m": "1m", "1min": "1m", "1Min": "1m",
            "5m": "5m", "5min": "5m", "5Min": "5m",
            "15m": "15m", "15min": "15m", "15Min": "15m",
            "1D": "1D", "1d": "1D", "1day": "1D", "Day": "1D",
        }
        tf = tf_map.get(interval, "5m")
        sym_bars = self._bars.get(symbol, {})
        sym_ts   = self._ts.get(symbol, {})
        if tf not in sym_bars:
            # Fallback: if 15m not precomputed (shouldn't happen), skip
            return []
        return _slice_up_to(sym_bars[tf], sym_ts[tf], self._current_ts, limit)

    async def place_limit_order(
        self, symbol: str, side: str, price: Decimal, quantity: Decimal
    ) -> dict:
        # Treat limit orders as market fills for backtest simplicity
        return await self.place_market_order(symbol, side, quantity)

    async def place_market_order(
        self, symbol: str, side: str, quantity: Decimal
    ) -> dict:
        """Simulate immediate fill at current bar close + slippage. O(log n)."""
        bars_1m = self._bars.get(symbol, {}).get("1m", [])
        ts_idx  = self._ts.get(symbol, {}).get("1m", [])
        pos = bisect.bisect_right(ts_idx, self._current_ts) - 1
        if pos < 0 or not bars_1m:
            return {"status": "REJECTED", "error": "no_data"}

        current_close = bars_1m[pos]["close"]
        slip = _slippage_pct(
            float(quantity) * current_close,
            self._adv_usd.get(symbol, 1_000_000.0),
        )
        side_upper = str(side).upper()
        # Buys pay more (bought at ask), sells receive less (sold at bid)
        if side_upper in ("BUY", "COVER"):
            fill_price = current_close * (1.0 + slip)
        else:
            fill_price = current_close * (1.0 - slip)

        self._order_seq += 1
        order_id = f"BT-{self._order_seq:06d}"
        self._filled_orders[order_id] = {
            "status":         "FILLED",
            "executedQty":    str(quantity),
            "filledAvgPrice": str(round(fill_price, 5)),
            "commission":     "0",
        }
        return {"orderId": order_id, "status": "PENDING"}

    async def cancel_order(self, symbol: str, order_id: str) -> dict:
        self._filled_orders.pop(order_id, None)
        return {"status": "CANCELED"}

    async def get_order_status(self, symbol: str, order_id: str) -> dict:
        return self._filled_orders.get(order_id, {"status": "UNKNOWN"})

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        return []

    async def get_account_balance(self) -> dict[str, Decimal]:
        return {"USD": Decimal(str(round(self._cash, 2)))}

    async def get_exchange_info(self, symbol: str) -> dict:
        return {}

    async def get_symbol_filters(self, symbol: str) -> dict:
        """Return standard tick/step sizes for US equities."""
        return {
            "tick_size":    Decimal("0.01"),
            "step_size":    Decimal("0.001"),
            "min_notional": Decimal("1.0"),
            "min_qty":      Decimal("0.001"),
        }

    async def get_account_config(self) -> dict:
        return {"pdt_protection": False, "shorting_enabled": True}

    def is_streaming(self) -> bool:
        return False

    def is_trade_streaming(self) -> bool:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Trade accounting helper
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClosedTrade:
    symbol: str
    strategy: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str = ""


@dataclass
class BacktestStats:
    name: str
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    peak_equity: float = STARTING_CAPITAL
    max_drawdown_pct: float = 0.0

    @property
    def net_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades)

    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.pnl > 0)
        return wins / len(self.closed_trades) * 100

    @property
    def avg_win(self) -> float:
        w = [t.pnl for t in self.closed_trades if t.pnl > 0]
        return sum(w) / len(w) if w else 0.0

    @property
    def avg_loss(self) -> float:
        l = [t.pnl for t in self.closed_trades if t.pnl <= 0]
        return sum(l) / len(l) if l else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.closed_trades if t.pnl <= 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    def update_drawdown(self, current_equity: float) -> None:
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if self.peak_equity > 0:
            dd = (self.peak_equity - current_equity) / self.peak_equity * 100
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd

    def sharpe(self) -> float:
        """Trade-level Sharpe, annualised assuming 252 trading days."""
        if len(self.closed_trades) < 2:
            return 0.0
        import statistics
        pnls = [t.pnl for t in self.closed_trades]
        mean = statistics.mean(pnls)
        std = statistics.stdev(pnls)
        if std == 0:
            return 0.0
        # Normalised by order size to get return (approximate)
        order_usd = max(abs(t.entry_price * t.quantity) for t in self.closed_trades) or 1.0
        r = [p / order_usd for p in pnls]
        mr = statistics.mean(r)
        sr = statistics.stdev(r)
        return (mr / sr) * (252 ** 0.5) if sr > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# BacktestRunner — runs real strategy classes through historical data
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestRunner:
    """Drives real strategy instances through historical bars.

    Flow per 5m bar (per symbol):
      1. Advance exchange time cursor to bar.timestamp
      2. Call strategy.on_tick(symbol, close_price) → list[Order]
      3. For each returned order:
         a. Place via exchange.place_market_order (fills at close + slippage)
         b. Retrieve fill info from exchange.get_order_status
         c. Compute P&L for closing orders (SELL/COVER)
         d. Update order.exchange_response and call strategy.on_order_filled
         e. Update cash and equity
    """

    # Market hours (Eastern Time offset in winter +5h UTC, summer +4h UTC).
    # We approximate to UTC-5 (EST, no DST) for simplicity.
    _MARKET_OPEN_UTC_H  = 14  # 09:30 ET = 14:30 UTC (approx)
    _MARKET_OPEN_UTC_M  = 30
    _MARKET_CLOSE_UTC_H = 21  # 16:00 ET = 21:00 UTC

    def __init__(
        self,
        strategy_name: str,
        strategy: VWAPScalpStrategy | MomentumStrategy,
        exchange: BacktestExchangeClient,
        symbols: list[str],
        settings: Settings,
    ) -> None:
        self.strategy_name = strategy_name
        self.strategy = strategy
        self.exchange = exchange
        self.symbols = symbols
        self.settings = settings
        self.stats = BacktestStats(name=strategy_name)
        self._cash = exchange.get_cash()
        self._entry_times: dict[str, datetime] = {}

    async def run(self, bars_5m: dict[str, list[dict]]) -> BacktestStats:
        """Run full backtest. Returns populated BacktestStats."""
        # Initialize strategy for all symbols
        for sym in self.symbols:
            try:
                await self.strategy.initialize(sym)
            except Exception as exc:
                print(f"  [WARN] {self.strategy_name}: init failed for {sym}: {exc}")

        # Build global time-sorted event queue: (timestamp, symbol, bar_dict)
        events: list[tuple[datetime, str, dict]] = []
        for sym in self.symbols:
            for bar in bars_5m.get(sym, []):
                events.append((bar["timestamp"], sym, bar))
        events.sort(key=lambda e: (e[0], e[1]))  # stable sort by time then symbol

        for bar_ts, sym, bar in events:
            # Skip outside market hours
            if not self._is_market_hour(bar_ts):
                continue

            # Advance time cursor
            self.exchange.set_current_time(bar_ts)
            current_price = Decimal(str(bar["close"]))

            # Run strategy tick
            try:
                new_orders = await self.strategy.on_tick(sym, current_price)
            except Exception as exc:
                # Fail-open: log and continue
                print(f"  [ERR] {self.strategy_name} on_tick {sym} @ {bar_ts}: {exc}")
                continue

            # Process each returned order
            for order in new_orders:
                await self._process_order(order, sym, bar_ts, current_price)

            # Update equity for drawdown tracking
            pos = self.strategy.positions.get(sym)
            open_notional = float(current_price) * float(pos.quantity) if pos and not pos.is_closed else 0.0
            equity = self._cash + open_notional
            self.stats.update_drawdown(equity)

        # EOD: flatten any open positions at last bar's close
        for sym in self.symbols:
            pos = self.strategy.positions.get(sym)
            if pos and not pos.is_closed:
                last_bar = (bars_5m.get(sym) or [{}])[-1]
                eod_price = Decimal(str(last_bar.get("close", 0)))
                if eod_price > 0:
                    pnl = self._calc_pnl(pos, eod_price)
                    self._cash += float(eod_price) * float(pos.quantity)
                    last_ts = last_bar.get("timestamp", datetime.now(timezone.utc))
                    self.stats.closed_trades.append(ClosedTrade(
                        symbol=sym,
                        strategy=self.strategy_name,
                        entry_price=float(pos.entry_price),
                        exit_price=float(eod_price),
                        quantity=float(pos.quantity),
                        pnl=pnl,
                        entry_time=self._entry_times.get(sym, last_ts),
                        exit_time=last_ts,
                        exit_reason="EOD",
                    ))

        return self.stats

    async def _process_order(
        self,
        order: Order,
        sym: str,
        bar_ts: datetime,
        current_price: Decimal,
    ) -> None:
        """Place order via mock exchange, apply fill, call on_order_filled."""
        side_upper = str(order.side).upper()
        order_qty = order.quantity

        # Place order through mock exchange (fills immediately)
        try:
            if str(order.order_type).upper() == "LIMIT":
                resp = await self.exchange.place_limit_order(
                    sym, str(order.side), order.price, order_qty
                )
            else:
                resp = await self.exchange.place_market_order(
                    sym, str(order.side), order_qty
                )
        except Exception as exc:
            print(f"  [ERR] place_order {side_upper} {sym}: {exc}")
            return

        order_id = resp.get("orderId", "")
        if not order_id:
            return

        # Retrieve fill details
        fill_info = await self.exchange.get_order_status(sym, order_id)
        if fill_info.get("status") != "FILLED":
            return

        fill_price = Decimal(fill_info.get("filledAvgPrice", str(current_price)))
        fill_qty   = Decimal(fill_info.get("executedQty", str(order_qty)))

        # Update order state
        order.id = order_id
        order.mark_filled(fill_qty)
        order.exchange_response = fill_info

        # Compute realized P&L for closing orders
        pos = self.strategy.positions.get(sym)
        pnl: float = 0.0
        if side_upper in ("SELL", "COVER") and pos and not pos.is_closed:
            entry_p = float(pos.entry_price)
            exit_p  = float(fill_price)
            qty_f   = float(fill_qty)
            if side_upper == "SELL":          # Closing a long
                pnl = (exit_p - entry_p) * qty_f
                self._cash += exit_p * qty_f  # return proceeds
            else:                             # COVER = closing a short
                pnl = (entry_p - exit_p) * qty_f
                self._cash += 0.0             # margin returned (simplified)

            self.stats.closed_trades.append(ClosedTrade(
                symbol=sym,
                strategy=self.strategy_name,
                entry_price=entry_p,
                exit_price=exit_p,
                quantity=qty_f,
                pnl=pnl,
                entry_time=self._entry_times.pop(sym, bar_ts),
                exit_time=bar_ts,
                exit_reason="TP" if pnl > 0 else "SL",  # O(1) proxy: +PnL→TP, −PnL→SL
            ))
        elif side_upper in ("BUY", "SHORT"):
            # Opening a position — deduct cash
            self._cash -= float(fill_price) * float(fill_qty)
            self._entry_times[sym] = bar_ts

        # Notify strategy of fill
        try:
            follow_up = await self.strategy.on_order_filled(order)
            self.strategy.active_orders.extend(follow_up)
        except Exception as exc:
            print(f"  [ERR] on_order_filled {sym}: {exc}")

    @staticmethod
    def _calc_pnl(pos: Position, exit_price: Decimal) -> float:
        if pos.side == "SHORT":
            return (float(pos.entry_price) - float(exit_price)) * float(pos.quantity)
        return (float(exit_price) - float(pos.entry_price)) * float(pos.quantity)

    @staticmethod
    def _is_market_hour(ts: datetime) -> bool:
        """Return True if timestamp falls in regular US market hours (approx UTC)."""
        h, m = ts.hour, ts.minute
        after_open  = (h > 14) or (h == 14 and m >= 30)
        before_close = h < 21
        return after_open and before_close


# ═══════════════════════════════════════════════════════════════════════════════
# Data fetching
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_bars(
    symbols: list[str], start: datetime, end: datetime, api_key: str, api_secret: str
) -> dict[str, list[dict]]:
    """Fetch 1-minute bars for all symbols from Alpaca. Returns raw bar dicts."""
    if not _ALPACA_AVAILABLE:
        print("ERROR: alpaca-py not installed. Run: pip install alpaca-py")
        sys.exit(1)

    print("Fetching 1-minute historical bars from Alpaca...")
    client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)
    all_bars: dict[str, list[dict]] = {s: [] for s in symbols}

    for sym in symbols:
        print(f"  {sym}...", end=" ", flush=True)
        try:
            req = StockBarsRequest(
                symbol_or_symbols=sym,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
            barset = client.get_stock_bars(req)
            if sym in barset.data:
                for bar in barset.data[sym]:
                    all_bars[sym].append({
                        "timestamp": bar.timestamp,
                        "open":      float(bar.open),
                        "high":      float(bar.high),
                        "low":       float(bar.low),
                        "close":     float(bar.close),
                        "volume":    float(bar.volume),
                    })
            print(f"{len(all_bars[sym])} bars")
        except Exception as exc:
            print(f"ERROR ({exc})")

    return all_bars


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(stats: BacktestStats, trading_days: int) -> None:
    bar = "=" * 64
    final_equity = STARTING_CAPITAL + stats.net_pnl
    ret_pct = (stats.net_pnl / STARTING_CAPITAL) * 100
    daily_avg = stats.net_pnl / trading_days if trading_days > 0 else 0.0
    tp_count  = sum(1 for t in stats.closed_trades if "TP"  in t.exit_reason)
    sl_count  = sum(1 for t in stats.closed_trades if "SL"  in t.exit_reason)
    eod_count = sum(1 for t in stats.closed_trades if "EOD" in t.exit_reason)
    wins  = sum(1 for t in stats.closed_trades if t.pnl > 0)
    losses = len(stats.closed_trades) - wins

    print()
    print(bar)
    print(f"  {stats.name}  [v2 — real strategy classes]")
    print(bar)
    print(f"  Starting Capital : ${STARTING_CAPITAL:>12,.2f}")
    print(f"  Final Equity     : ${final_equity:>12,.2f}")
    print(f"  Net P&L          : ${stats.net_pnl:>+12,.2f}  ({ret_pct:+.2f}%)")
    print(f"  Max Drawdown     : {stats.max_drawdown_pct:>12.2f}%")
    print(f"  Sharpe (approx)  : {stats.sharpe():>12.2f}")
    print()
    print(f"  Round-trips      : {len(stats.closed_trades):>6}")
    print(f"  Wins / Losses    : {wins:>3} / {losses:<3}")
    print(f"  Win Rate         : {stats.win_rate:>12.1f}%")
    print(f"  Avg Win          : ${stats.avg_win:>12,.2f}")
    print(f"  Avg Loss         : ${stats.avg_loss:>12,.2f}")
    print(f"  Profit Factor    : {stats.profit_factor:>12.2f}")
    print()
    print(f"  Take-Profits     : {tp_count:>6}")
    print(f"  Stop-Losses      : {sl_count:>6}")
    print(f"  EOD Flattens     : {eod_count:>6}")
    print(f"  Avg Daily P&L    : ${daily_avg:>12,.2f}")
    print(f"  Trading Days     : {trading_days:>6}")
    print(bar)


def print_trade_log(stats: BacktestStats, last_n: int = 15) -> None:
    trades = stats.closed_trades
    if not trades:
        print("  No completed trades.")
        return
    shown = trades[-last_n:]
    print(f"\n  Last {len(shown)} trades:")
    print(f"  {'Date':<20} {'Sym':<6} {'Entry':>9} {'Exit':>9} {'P&L':>11} {'Reason'}")
    print(f"  {'-'*20} {'-'*6} {'-'*9} {'-'*9} {'-'*11} {'-'*10}")
    for t in shown:
        marker = "+" if t.pnl >= 0 else ""
        print(
            f"  {t.exit_time.strftime('%Y-%m-%d %H:%M'):<20} {t.symbol:<6} "
            f"${t.entry_price:>8.2f} ${t.exit_price:>8.2f} "
            f"{marker}${t.pnl:>9.2f}  {t.exit_reason}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def run_strategy_backtest(
    strategy_name: str,
    exchange: BacktestExchangeClient,
    settings: Settings,
    symbols: list[str],
    bars_5m: dict[str, list[dict]],
) -> BacktestStats:
    """Instantiate and run a single strategy. Each strategy gets its own exchange instance."""
    # Create fresh exchange clone per strategy (each tracks its own cash/positions)
    import copy
    # Exchange data is shared (read-only); we only clone the mutable state
    fresh_exchange = copy.copy(exchange)
    fresh_exchange._cash = STARTING_CAPITAL
    fresh_exchange._filled_orders = {}
    fresh_exchange._order_seq = 0

    risk = RiskManager(settings=settings)
    risk._account_balance = STARTING_CAPITAL

    if strategy_name == "vwap_scalp":
        strategy = VWAPScalpStrategy(fresh_exchange, risk, settings)
    elif strategy_name == "momentum":
        strategy = MomentumStrategy(fresh_exchange, risk, settings)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    runner = BacktestRunner(strategy_name, strategy, fresh_exchange, symbols, settings)
    return await runner.run(bars_5m)


def main() -> None:
    print()
    print("=" * 64)
    print("  AtoBot Backtester v2 — Real Strategy Classes")
    print("  Slippage: Square-Root Market Impact Model")
    print("=" * 64)
    print()

    settings = _load_settings()
    symbols = list(dict.fromkeys(SYMBOLS + list(settings.SYMBOLS)))[:8]  # dedupe, cap at 8

    end   = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=90)

    print(f"  Symbols : {', '.join(symbols)}")
    print(f"  Capital : ${STARTING_CAPITAL:,.0f}")
    print(f"  Period  : {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print()

    # ── Fetch data ──────────────────────────────────────────────────────────
    bars_1m = fetch_bars(
        symbols, start, end,
        api_key=settings.ALPACA_API_KEY,
        api_secret=settings.ALPACA_API_SECRET,
    )

    total_bars = sum(len(b) for b in bars_1m.values())
    if total_bars == 0:
        print("ERROR: No data received. Check Alpaca API keys and date range.")
        sys.exit(1)

    # Count trading days
    all_days: set[str] = set()
    for sym_bars in bars_1m.values():
        for b in sym_bars:
            all_days.add(b["timestamp"].strftime("%Y-%m-%d"))
    trading_days = len(all_days)
    print(f"\n  1-min bars : {total_bars:,}")
    print(f"  Trading days: {trading_days}")

    # ── Build shared exchange + precomputed bars ─────────────────────────────
    shared_exchange = BacktestExchangeClient(bars_1m, STARTING_CAPITAL)
    bars_5m = {sym: shared_exchange._bars[sym]["5m"] for sym in symbols if sym in shared_exchange._bars}
    total_5m = sum(len(b) for b in bars_5m.values())
    print(f"  5-min bars : {total_5m:,}")
    print()

    # ── Run strategies ───────────────────────────────────────────────────────
    strategies_to_run = ["vwap_scalp", "momentum"]
    results: dict[str, BacktestStats] = {}

    for strat_name in strategies_to_run:
        print(f"  Running {strat_name}...")
        try:
            stats = asyncio.run(
                run_strategy_backtest(strat_name, shared_exchange, settings, symbols, bars_5m)
            )
            results[strat_name] = stats
            print_report(stats, trading_days)
            print_trade_log(stats)
        except Exception as exc:
            print(f"  ERROR running {strat_name}: {exc}")
            import traceback; traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────────────
    if results:
        print("\n")
        print("=" * 64)
        print("  STRATEGY COMPARISON  (v2 — real classes, realistic slippage)")
        print("=" * 64)
        print(f"  {'Strategy':<20} {'Net P&L':>12} {'Win Rate':>10} {'Trades':>8} {'MaxDD':>8} {'Sharpe':>8}")
        print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        total_pnl = 0.0
        for name, r in results.items():
            pnl_str = f"${r.net_pnl:>+10,.2f}"
            total_pnl += r.net_pnl
            print(
                f"  {name:<20} {pnl_str:>12} {r.win_rate:>9.1f}% "
                f"{len(r.closed_trades):>8} {r.max_drawdown_pct:>7.2f}% {r.sharpe():>8.2f}"
            )
        print(f"  {'COMBINED':<20} ${total_pnl:>+10,.2f}")
        print("=" * 64)
        print()
        print("  NOTE: v2 results reflect ACTUAL live strategy logic including:")
        print("  warm-up guards, ATR stops, confluence gate, and Kelly sizing.")
        print("  v1 results (different algorithm) are preserved in backtest_v1.py")
        print()


if __name__ == "__main__":
    main()
