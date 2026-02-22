"""Backtesting runner for AtoBot Trading.

Usage:
    python scripts/backtest.py --strategy grid --symbol BTCUSDT \
        --start 2025-01-01 --end 2025-12-31

"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from binance import AsyncClient

from src.config.settings import Settings
from src.data import indicators as ind
from src.utils.helpers import calculate_pnl, format_usdt, round_price, round_quantity


# ── Mock exchange for backtesting ─────────────────────────────────────────────


class BacktestExchange:
    """Simulated exchange that replays historical price data."""

    def __init__(self, df: pd.DataFrame, tick_size: Decimal, step_size: Decimal) -> None:
        self.df = df
        self.tick_size = tick_size
        self.step_size = step_size
        self.balance_usdt = Decimal("1000")
        self.balance_asset = Decimal("0")
        self.open_orders: list[dict] = []
        self.trades: list[dict] = []
        self.current_idx = 0

    @property
    def current_price(self) -> Decimal:
        """Return close price at current index."""
        return Decimal(str(self.df.iloc[self.current_idx]["close"]))

    def place_limit_buy(self, price: Decimal, qty: Decimal) -> None:
        """Place a simulated limit buy order."""
        self.open_orders.append({"side": "BUY", "price": price, "qty": qty})

    def place_limit_sell(self, price: Decimal, qty: Decimal) -> None:
        """Place a simulated limit sell order."""
        self.open_orders.append({"side": "SELL", "price": price, "qty": qty})

    def check_fills(self, low: Decimal, high: Decimal) -> list[dict]:
        """Check if any open orders would have been filled in this candle."""
        filled: list[dict] = []
        remaining: list[dict] = []
        for order in self.open_orders:
            if order["side"] == "BUY" and low <= order["price"]:
                cost = order["price"] * order["qty"]
                if cost <= self.balance_usdt:
                    self.balance_usdt -= cost
                    self.balance_asset += order["qty"]
                    filled.append(order)
                    self.trades.append(
                        {
                            "side": "BUY",
                            "price": float(order["price"]),
                            "qty": float(order["qty"]),
                            "cost": float(cost),
                            "idx": self.current_idx,
                        }
                    )
                else:
                    remaining.append(order)
            elif order["side"] == "SELL" and high >= order["price"]:
                if order["qty"] <= self.balance_asset:
                    proceeds = order["price"] * order["qty"]
                    self.balance_usdt += proceeds
                    self.balance_asset -= order["qty"]
                    filled.append(order)
                    self.trades.append(
                        {
                            "side": "SELL",
                            "price": float(order["price"]),
                            "qty": float(order["qty"]),
                            "proceeds": float(proceeds),
                            "idx": self.current_idx,
                        }
                    )
                else:
                    remaining.append(order)
            else:
                remaining.append(order)
        self.open_orders = remaining
        return filled


# ── Backtester ────────────────────────────────────────────────────────────────


class Backtester:
    """Run a strategy against historical data and produce statistics."""

    def __init__(
        self,
        strategy: str,
        symbol: str,
        start: str,
        end: str,
        initial_balance: float = 1000.0,
        grid_levels: int = 5,
        grid_spacing_pct: float = 1.0,
        order_size_usdt: float = 10.0,
        dca_interval: int = 24,  # hours (in candles if 1h data)
        dca_drop_pct: float = 2.0,
        dca_tp_pct: float = 3.0,
    ) -> None:
        self.strategy = strategy
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_balance = Decimal(str(initial_balance))
        self.grid_levels = grid_levels
        self.grid_spacing_pct = Decimal(str(grid_spacing_pct)) / Decimal("100")
        self.order_size_usdt = Decimal(str(order_size_usdt))
        self.dca_interval = dca_interval
        self.dca_drop_pct = Decimal(str(dca_drop_pct)) / Decimal("100")
        self.dca_tp_pct = Decimal(str(dca_tp_pct)) / Decimal("100")

    async def fetch_data(self) -> pd.DataFrame:
        """Download historical klines from Binance."""
        client = await AsyncClient.create()
        try:
            klines = await client.get_historical_klines(
                self.symbol,
                AsyncClient.KLINE_INTERVAL_1HOUR,
                self.start,
                self.end,
            )
        finally:
            await client.close_connection()

        rows = []
        for k in klines:
            rows.append(
                {
                    "timestamp": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def run_grid(self, df: pd.DataFrame) -> dict:
        """Simulate grid strategy on historical data."""
        tick_size = Decimal("0.01")
        step_size = Decimal("0.00001")
        exchange = BacktestExchange(df, tick_size, step_size)
        exchange.balance_usdt = self.initial_balance

        equity_curve: list[float] = []
        peak = float(self.initial_balance)
        max_drawdown = 0.0

        for i in range(len(df)):
            exchange.current_idx = i
            row = df.iloc[i]
            price = Decimal(str(row["close"]))
            low = Decimal(str(row["low"]))
            high = Decimal(str(row["high"]))

            # Check fills
            exchange.check_fills(low, high)

            # Place grid orders if none exist
            if not exchange.open_orders:
                for lvl in range(1, self.grid_levels + 1):
                    buy_p = round_price(
                        price * (Decimal("1") - self.grid_spacing_pct * Decimal(str(lvl))),
                        tick_size,
                    )
                    sell_p = round_price(
                        price * (Decimal("1") + self.grid_spacing_pct * Decimal(str(lvl))),
                        tick_size,
                    )
                    qty = round_quantity(self.order_size_usdt / price, step_size)
                    if qty > Decimal("0"):
                        exchange.place_limit_buy(buy_p, qty)
                        exchange.place_limit_sell(sell_p, qty)

            # Track equity
            equity = float(exchange.balance_usdt + exchange.balance_asset * price)
            equity_curve.append(equity)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        final_equity = equity_curve[-1] if equity_curve else float(self.initial_balance)
        return self._compile_results(exchange, equity_curve, final_equity, max_drawdown)

    def run_dca(self, df: pd.DataFrame) -> dict:
        """Simulate DCA strategy on historical data."""
        tick_size = Decimal("0.01")
        step_size = Decimal("0.00001")
        exchange = BacktestExchange(df, tick_size, step_size)
        exchange.balance_usdt = self.initial_balance

        equity_curve: list[float] = []
        peak = float(self.initial_balance)
        max_drawdown = 0.0
        avg_entry = Decimal("0")
        total_qty = Decimal("0")
        last_buy_idx = -self.dca_interval  # Allow immediate first buy

        for i in range(len(df)):
            exchange.current_idx = i
            row = df.iloc[i]
            price = Decimal(str(row["close"]))
            low = Decimal(str(row["low"]))
            high = Decimal(str(row["high"]))

            exchange.check_fills(low, high)

            # Recalculate position from trades
            total_bought_qty = sum(
                Decimal(str(t["qty"])) for t in exchange.trades if t["side"] == "BUY"
            )
            total_sold_qty = sum(
                Decimal(str(t["qty"])) for t in exchange.trades if t["side"] == "SELL"
            )
            holding = total_bought_qty - total_sold_qty

            # Calculate avg entry
            if holding > Decimal("0"):
                total_cost = Decimal("0")
                running_qty = Decimal("0")
                for t in exchange.trades:
                    if t["side"] == "BUY":
                        running_qty += Decimal(str(t["qty"]))
                        total_cost += Decimal(str(t["price"])) * Decimal(str(t["qty"]))
                    elif t["side"] == "SELL":
                        running_qty -= Decimal(str(t["qty"]))
                if running_qty > Decimal("0"):
                    avg_entry = total_cost / running_qty if total_cost > 0 else price

            should_buy = False

            # Interval buy
            if i - last_buy_idx >= self.dca_interval:
                should_buy = True

            # Safety order on dip
            if holding > Decimal("0") and avg_entry > Decimal("0"):
                drop = (avg_entry - price) / avg_entry
                if drop >= self.dca_drop_pct:
                    should_buy = True

            # Take profit
            if holding > Decimal("0") and avg_entry > Decimal("0"):
                gain = (price - avg_entry) / avg_entry
                if gain >= self.dca_tp_pct:
                    qty = round_quantity(holding, step_size)
                    if qty > Decimal("0"):
                        exchange.place_limit_sell(price, qty)

            if should_buy and not exchange.open_orders:
                qty = round_quantity(self.order_size_usdt / price, step_size)
                if qty > Decimal("0"):
                    exchange.place_limit_buy(price, qty)
                    last_buy_idx = i

            equity = float(exchange.balance_usdt + exchange.balance_asset * price)
            equity_curve.append(equity)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        final_equity = equity_curve[-1] if equity_curve else float(self.initial_balance)
        return self._compile_results(exchange, equity_curve, final_equity, max_drawdown)

    def _compile_results(
        self,
        exchange: BacktestExchange,
        equity_curve: list[float],
        final_equity: float,
        max_drawdown: float,
    ) -> dict:
        """Compile backtest results into a summary dict."""
        total_trades = len(exchange.trades)
        buys = [t for t in exchange.trades if t["side"] == "BUY"]
        sells = [t for t in exchange.trades if t["side"] == "SELL"]
        pnl = final_equity - float(self.initial_balance)

        # Simple win rate: count sell trades where proceeds > cost of matched buy
        wins = 0
        for s in sells:
            # Find a matching buy with lower price
            for b in buys:
                if s["price"] > b["price"]:
                    wins += 1
                    break

        win_rate = (wins / len(sells) * 100) if sells else 0

        # Sharpe ratio approximation (annualised, assuming hourly data)
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * (8760**0.5)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "period": f"{self.start} → {self.end}",
            "initial_balance": float(self.initial_balance),
            "final_equity": round(final_equity, 2),
            "total_pnl": round(pnl, 2),
            "total_pnl_pct": round(pnl / float(self.initial_balance) * 100, 2),
            "total_trades": total_trades,
            "buys": len(buys),
            "sells": len(sells),
            "win_rate_pct": round(win_rate, 2),
            "max_drawdown_pct": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe, 4),
            "equity_curve": equity_curve,
        }

    async def run(self) -> dict:
        """Download data and run the backtest."""
        print(f"Fetching historical data for {self.symbol} ({self.start} → {self.end}) …")
        df = await self.fetch_data()
        print(f"Loaded {len(df)} candles.")

        if self.strategy == "grid":
            results = self.run_grid(df)
        elif self.strategy == "dca":
            results = self.run_dca(df)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return results


def print_results(results: dict) -> None:
    """Pretty-print backtest results."""
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Strategy:        {results['strategy']}")
    print(f"  Symbol:          {results['symbol']}")
    print(f"  Period:          {results['period']}")
    print(f"  Initial Balance: ${results['initial_balance']:,.2f}")
    print(f"  Final Equity:    ${results['final_equity']:,.2f}")
    print(f"  Total PnL:       ${results['total_pnl']:,.2f} ({results['total_pnl_pct']}%)")
    print(f"  Total Trades:    {results['total_trades']}")
    print(f"  Buys:            {results['buys']}")
    print(f"  Sells:           {results['sells']}")
    print(f"  Win Rate:        {results['win_rate_pct']}%")
    print(f"  Max Drawdown:    {results['max_drawdown_pct']}%")
    print(f"  Sharpe Ratio:    {results['sharpe_ratio']}")
    print("=" * 60)


def plot_equity(results: dict) -> None:
    """Plot the equity curve with matplotlib (optional)."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))
        plt.plot(results["equity_curve"], linewidth=1)
        plt.title(f"Equity Curve — {results['strategy'].upper()} on {results['symbol']}")
        plt.xlabel("Candle (1h)")
        plt.ylabel("Equity (USDT)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"data/backtest_{results['strategy']}_{results['symbol']}.png", dpi=150)
        print(f"\nEquity curve saved to data/backtest_{results['strategy']}_{results['symbol']}.png")
    except ImportError:
        print("\nInstall matplotlib to plot the equity curve.")


def main() -> None:
    """CLI entry point for backtesting."""
    parser = argparse.ArgumentParser(description="AtoBot Backtester")
    parser.add_argument("--strategy", choices=["grid", "dca"], default="grid")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--grid-levels", type=int, default=5)
    parser.add_argument("--grid-spacing", type=float, default=1.0)
    parser.add_argument("--order-size", type=float, default=10.0)
    parser.add_argument("--plot", action="store_true", help="Save equity curve plot")
    args = parser.parse_args()

    bt = Backtester(
        strategy=args.strategy,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        initial_balance=args.balance,
        grid_levels=args.grid_levels,
        grid_spacing_pct=args.grid_spacing,
        order_size_usdt=args.order_size,
    )

    results = asyncio.run(bt.run())
    print_results(results)
    if args.plot:
        plot_equity(results)


if __name__ == "__main__":
    main()
