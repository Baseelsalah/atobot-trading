# AtoBot Trading

**Autonomous algorithmic day-trading bot** built in async Python -- connects to **Alpaca Markets** (paper or live) to trade US equities during market hours with seven pluggable strategies, a self-healing Guardian agent, OpenAI-powered trade analysis, and a real-time Streamlit dashboard.

> **1-Year Backtest Result:** VWAP Scalp + ORB on 8 symbols at $35K order size -> **+$83,534 net P&L on a $100K account** using 2:1 margin (exceeds $75K/yr target by 11%).

---

## What It Does

AtoBot monitors a configurable watchlist of equities every 10 seconds during market hours. Each tick it:

1. **Reads the market regime** -- classifies trend, volatility, breadth, and sector rotation via SPY/QQQ/VIX/IWM.
2. **Runs signal generation** -- one or more active strategies vote on entry/exit signals for each symbol.
3. **Checks risk gates** -- position limits, daily loss cap, max drawdown, PDT, and stop-loss enforced before every order.
4. **Executes orders** -- async Alpaca REST client; fills persisted to SQLite and streamed to Telegram.
5. **Self-monitors via Guardian** -- separate process checks health every 60s, heals faults, auto-tunes parameters daily.

---

## Architecture

```
src/
|-- config/          Pydantic-Settings config loaded from .env
|-- models/          Order, Position, Trade -- typed Pydantic + Decimal
|-- utils/           Loguru logger, retry decorator, helpers
|-- exchange/        Abstract base + Alpaca async client
|-- data/            MarketDataProvider + indicators (RSI, EMA, VWAP, ATR, BBands)
|-- strategies/      7 strategies (see below) + strategy selector
|-- scanner/         MarketScanner, RegimeDetector, NewsIntel
|-- intelligence/    AITradeAdvisor (GPT-4o-mini), MLModel, MLFeatureEngine
|-- risk/            RiskManager + Kelly Criterion position sizer
|-- analytics/       TradeJournal, ProfitGoalTracker
|-- notifications/   TelegramNotifier
|-- persistence/     SQLAlchemy 2.0 async (SQLite)
|-- guardian/        HealthMonitor, SelfHealer, PerformanceAnalyzer, AutoTuner
|-- dashboard/       Streamlit real-time dashboard
`-- main.py          Entry point (asyncio + graceful SIGINT/SIGTERM)
backtest*.py         Backtesting suite (single-strategy, A/B, stress, crypto)
tests/               152 pytest-asyncio tests
```

---

## Key Features

| Feature | Detail |
|---------|--------|
| **7 trading strategies** | VWAP Scalp, ORB, Momentum, EMA Pullback, Swing, Pairs/StatArb, Crypto |
| **Multi-strategy** | Run several strategies simultaneously; signals weighted by regime |
| **Market regime detection** | Trend / volatility / breadth / sector-rotation classifier before every trade |
| **AI trade advisor** | GPT-4o-mini pre-trade sentiment check + daily market briefing (optional) |
| **Guardian agent** | Separate watchdog: health checks, self-healing, auto-parameter tuning |
| **Kelly Criterion sizing** | Mathematically optimal position sizing via f* = (bp - q) / b |
| **PDT protection** | Hard block on trades that would trigger the Pattern Day Trader rule |
| **End-of-day flatten** | All positions closed before market close; no overnight exposure |
| **Real-time dashboard** | Streamlit UI: live equity curve, open positions, trade log |
| **Telegram alerts** | Fill notifications, errors, daily P&L summaries, emergency shutdown |
| **Full test suite** | 152 pytest-asyncio tests: strategies, risk, exchange, persistence |
| **Docker** | docker-compose up -d starts bot + dashboard |

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url> && cd atobot-trading
python -m venv .venv
# Windows: .venv\Scriptsctivate  |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add Alpaca API keys -- ALPACA_PAPER=true and DRY_RUN=true by default (no real money)
```

### 3. Run the bot

```bash
python -m src.main
```

### 4. Launch the dashboard

```bash
streamlit run src/dashboard/app.py
```

### 5. Run the Guardian agent (recommended in production)

```bash
python -m src.guardian
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Strategies

All strategies inherit from `BaseStrategy` and are selected via `STRATEGIES` in `.env`.

### VWAP Scalp *(Best performer)*
Enters long when price bounces off VWAP from below. High-frequency; many small wins compound quickly.

| Setting | Default | Description |
|---------|---------|-------------|
| `VWAP_BOUNCE_PERCENT` | 0.15 | % from VWAP to trigger entry |
| `VWAP_TAKE_PROFIT_PERCENT` | 0.5 | % profit target |
| `VWAP_STOP_LOSS_PERCENT` | 0.3 | % stop-loss |
| `VWAP_ORDER_SIZE_USD` | 500 | Dollar amount per trade |

### ORB (Opening Range Breakout)
Defines the high/low of the first 15 minutes, then enters on a confirmed breakout above the range. Works best paired with VWAP.

| Setting | Default | Description |
|---------|---------|-------------|
| `ORB_RANGE_MINUTES` | 15 | Minutes to define opening range |
| `ORB_BREAKOUT_PERCENT` | 0.1 | % above range to confirm breakout |
| `ORB_TAKE_PROFIT_PERCENT` | 1.5 | % profit target |
| `ORB_STOP_LOSS_PERCENT` | 0.75 | % stop-loss |

### Momentum (RSI + Volume)
Buys when RSI is oversold **and** volume spikes above the 20-period average. Lower frequency, larger per-trade moves.

### EMA Pullback
3-layer EMA stack (9/21/50) on 5-minute bars. Enters on pullback to 9 or 21 EMA while price stays above the 50 EMA. RSI in the 35-55 zone confirms oversold-in-uptrend.

### Swing (Multi-Day)
Holds positions 1-5 days to capture 2-5% moves. Requires 2+ confluence signals (RSI bounce, EMA support, volume surge, candle pattern). Avoids PDT by holding overnight.

### Pairs / Statistical Arbitrage
Trades mean-reversion of the log-price spread between correlated assets (e.g. NVDA:AMD, GOOGL:META). Enters when rolling z-score exceeds threshold; hedge ratio via OLS regression.

### Crypto
VWAP + RSI adapted for 24/7 crypto markets via the Binance async client.

---

## Market Regime Detection

Before any strategy runs, `RegimeDetector` classifies five orthogonal dimensions from SPY/QQQ/VIX/IWM:

| Dimension | Classes |
|-----------|---------|
| **Trend** | strong_bull, bull, neutral, bear, strong_bear, choppy |
| **Volatility** | low (VIX <15), normal, elevated, extreme (VIX >30) |
| **Breadth** | Advance/decline ratio; narrow vs broad participation |
| **Momentum** | Risk-on (QQQ > SPY) vs risk-off (TLT, GLD, XLU flows) |
| **Time-of-day** | Open drive, mid-day chop, power hour |

The engine reduces position sizes in elevated-volatility regimes and skips new entries in choppy or extreme regimes.

---

## Entry Filters

| Filter | Default | Backtest Impact | Description |
|--------|---------|-----------------|-------------|
| **Midday Filter** | AVOID_MIDDAY=true | **+6.6%** improvement | Skips entries 12-2 PM ET (low-volume chop) |
| **EMA Trend Filter** | TREND_FILTER_ENABLED=false | -71% entry count, net negative | Only enter when price > 20-period EMA |
| **Trailing Stop** | TRAILING_STOP_ENABLED=false | Inactive at safe params | Trail stop behind highest price |

---

## Risk Management

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_POSITION_SIZE_USD` | 2000 | Max dollars in one symbol |
| `MAX_OPEN_ORDERS` | 10 | Max simultaneous open orders |
| `DAILY_LOSS_LIMIT_USD` | 200 | Max daily loss before halting |
| `MAX_DRAWDOWN_PERCENT` | 5 | Max portfolio drawdown % |
| `STOP_LOSS_PERCENT` | 2 | Per-position stop-loss % |
| `MAX_DAILY_TRADES` | 20 | Max trades per day |
| `PDT_PROTECTION` | true | Block trades that trigger the PDT rule |
| `FLATTEN_EOD` | true | Close all positions before market close |

The `RiskManager` is evaluated **before every order** and on each tick for stop-loss and drawdown. Three consecutive engine errors trigger an automatic emergency shutdown.

Position sizes use the **Kelly Criterion** (f* = (bp - q) / b) with a fractional-Kelly cap to limit variance.

---

## Guardian Agent

A separate long-running process (`python -m src.guardian`) acts as an autonomous watchdog:

| Module | Interval | Responsibility |
|--------|----------|----------------|
| `HealthMonitor` | 60 s | Checks process liveness, API connectivity, disk space, error rates |
| `SelfHealer` | on fault | Restarts dead workers, clears stale locks, resets error counters |
| `PerformanceAnalyzer` | 1 h | Computes Sharpe, MDD, win rate, profit factor from the trade journal |
| `AutoTuner` | 24 h | Adjusts stop sizes and take-profit targets within safe bounds |
| `WalkForwardValidator` | on tune | Out-of-sample validation before any parameter change is applied |

---

## AI Trade Advisor *(optional)*

Set `OPENAI_API_KEY` in `.env` to enable GPT-4o-mini integration:

- **Pre-trade sentiment check** -- evaluates price action, indicators, and market context before every entry.
- **Daily market briefing** -- summarises conditions before the open.
- **Trade review** -- analyses completed trades for patterns and improvement opportunities.
- Responses cached 5 minutes; hard cap of 200 API calls/day to control costs.
- Gracefully degrades to neutral/allow when the API key is absent.

---

## Backtesting

### 1-Year Results (best configuration)

| Configuration | Annual P&L | vs $75K Target |
|---------------|-----------|----------------|
| 5 symbols, $17K orders | +$5,854 | 8% |
| 5 symbols, $35K orders | +$42,408 | 57% |
| 8 symbols, $30K orders | +$46,321 | 62% |
| **8 symbols, $35K orders** | **+$83,534** | **111% (target exceeded)** |

Period: 2025-02-22 to 2026-02-22 | Starting capital: $100K | Margin: 2:1
Symbols: AAPL MSFT TSLA NVDA AMD META AMZN GOOG | Slippage: 1.0 bp/round-trip

### Run a backtest

```bash
python backtest.py              # quick single-strategy
python backtest_v2.py           # A/B comparison, all strategies
python backtest_1year.py        # 1-year scaling study
python backtest_stress_hardened.py  # COVID crash + high-vol stress test
```

---

## Docker

```bash
docker-compose up -d            # start bot + dashboard
docker-compose logs -f bot      # follow logs
# Dashboard: http://localhost:8501
```

---

## Notifications

Set `NOTIFICATIONS_ENABLED=true` plus `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` to receive:

- Trade fill alerts
- Error notifications
- Daily P&L summaries
- Emergency shutdown alerts

---

## Configuration Reference

All settings load from environment variables or a `.env` file. See `.env.example` for the full list.

| Category | Variable | Default |
|----------|----------|---------|
| Exchange | `EXCHANGE` | alpaca |
| Alpaca | `ALPACA_API_KEY` / `ALPACA_API_SECRET` | -- |
| Alpaca | `ALPACA_PAPER` | true |
| Trading | `SYMBOLS` | ["AAPL","TSLA",...] |
| Trading | `STRATEGIES` | ["vwap_scalp","orb"] |
| Trading | `BASE_ORDER_SIZE_USD` | 500 |
| Trading | `DRY_RUN` | true |
| Filters | `AVOID_MIDDAY` | true |
| AI | `OPENAI_API_KEY` | -- (optional) |
| Timing | `POLL_INTERVAL_SECONDS` | 10 |
| Database | `DATABASE_URL` | sqlite+aiosqlite:///data/atobot.db |

---

## Development

```bash
pip install -e .[dev]
ruff check src/ tests/          # lint
mypy src/                        # type check (strict)
pytest tests/ -v --tb=short     # 152 tests
```

**Tech stack:** Python 3.11 - asyncio - Pydantic v2 - SQLAlchemy 2.0 async - Alpaca-py - Streamlit - Loguru - pytest-asyncio - Docker

---

## License

MIT
