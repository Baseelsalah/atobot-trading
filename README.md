# AtoBot Trading

**Autonomous Stock Day-Trading Bot** — async Python bot using **Alpaca** paper/live accounts to trade equities during market hours with multi-strategy support, risk management, Telegram alerts, and backtesting.

> **Backtest winner:** VWAP Scalp + ORB with midday filter → **$3,194/mo** estimated on $100k account (3-month average).

---

## Architecture

```
src/
├── config/          Settings (pydantic-settings, .env)
├── models/          Order, Position, Trade (Pydantic, Decimal)
├── utils/           Logger, retry decorator, helpers
├── exchange/        Abstract base + Alpaca async client
├── risk/            RiskManager (position limits, drawdown, PDT, stop-loss)
├── data/            MarketDataProvider + technical indicators (RSI, EMA, VWAP)
├── strategies/      BaseStrategy → MomentumStrategy, ORBStrategy, VWAPScalpStrategy
├── notifications/   BaseNotifier → TelegramNotifier
├── persistence/     SQLAlchemy 2.0 async (SQLite)
├── core/            TradingEngine loop + AtoBot orchestrator
├── dashboard/       Streamlit real-time dashboard
└── main.py          Entry point
backtest.py          Quick single-strategy backtester
backtest_v2.py       Full A/B comparison engine (baseline vs improved filters)
tests/               152 pytest-asyncio tests
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url> && cd atobot-trading
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — add your Alpaca API keys
# ALPACA_PAPER=true and DRY_RUN=true are on by default (no real money)
```

### 3. Run the bot

```bash
python -m src.main
```

### 4. Launch the dashboard

```bash
streamlit run src/dashboard/app.py
```

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Strategies

The bot supports running **multiple strategies simultaneously** on the same symbols. Set `STRATEGIES=["vwap_scalp","orb"]` in `.env`.

### VWAP Scalp (Best performer)
Enters long when price bounces off VWAP from below. Exits at a fixed take-profit or stop-loss. High-frequency with many small wins.

| Setting | Default | Description |
|---------|---------|-------------|
| `VWAP_BOUNCE_PERCENT` | 0.15 | % from VWAP to trigger entry |
| `VWAP_TAKE_PROFIT_PERCENT` | 0.5 | % profit target |
| `VWAP_STOP_LOSS_PERCENT` | 0.3 | % stop-loss |
| `VWAP_ORDER_SIZE_USD` | 500 | Dollar amount per trade |

### ORB (Opening Range Breakout)
Defines the high/low of the first 15 minutes, then enters on a confirmed breakout above the range. Works well paired with VWAP.

| Setting | Default | Description |
|---------|---------|-------------|
| `ORB_RANGE_MINUTES` | 15 | Minutes to define opening range |
| `ORB_BREAKOUT_PERCENT` | 0.1 | % above range to confirm breakout |
| `ORB_TAKE_PROFIT_PERCENT` | 1.5 | % profit target |
| `ORB_STOP_LOSS_PERCENT` | 0.75 | % stop-loss |
| `ORB_ORDER_SIZE_USD` | 500 | Dollar amount per trade |

### Momentum (RSI + Volume)
Buys when RSI is oversold AND volume spikes above the moving average. Lower frequency, larger moves.

| Setting | Default | Description |
|---------|---------|-------------|
| `MOMENTUM_RSI_OVERSOLD` | 30 | RSI threshold for buy signal |
| `MOMENTUM_RSI_OVERBOUGHT` | 70 | RSI threshold for sell signal |
| `MOMENTUM_VOLUME_MULTIPLIER` | 1.5 | Min relative volume to entry |
| `MOMENTUM_TAKE_PROFIT_PERCENT` | 2.0 | % profit target |
| `MOMENTUM_STOP_LOSS_PERCENT` | 1.0 | % stop-loss |

---

## Entry Filters

Filters are applied before any strategy generates an entry signal. Tuned via backtesting.

| Filter | Default | Backtest Result | Description |
|--------|---------|-----------------|-------------|
| **Midday Filter** | `AVOID_MIDDAY=true` | **+6.6%** improvement | Skips entries 12–2 PM ET (low volume chop) |
| **EMA Trend Filter** | `TREND_FILTER_ENABLED=false` | **-71% entries**, net negative | Only enter when price > 20-period EMA |
| **Trailing Stop** | `TRAILING_STOP_ENABLED=false` | Inactive at safe params | Trail stop behind highest price after activation |

---

## Risk Management

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_POSITION_SIZE_USD` | 2000 | Max $ in one stock |
| `MAX_OPEN_ORDERS` | 10 | Max simultaneous open orders |
| `DAILY_LOSS_LIMIT_USD` | 200 | Max daily loss before halting |
| `MAX_DRAWDOWN_PERCENT` | 5 | Max portfolio drawdown % |
| `STOP_LOSS_PERCENT` | 2 | Per-position stop-loss % |
| `MAX_DAILY_TRADES` | 20 | Max trades per day |
| `PDT_PROTECTION` | true | Block trades that trigger PDT rule |
| `FLATTEN_EOD` | true | Close all positions before market close |

The risk manager is checked **before every order** and again at each tick for stop-loss and drawdown. Three consecutive engine errors trigger an automatic emergency shutdown.

---

## Backtesting

### Quick backtest (single strategy)
```bash
python backtest.py
```

### A/B comparison (baseline vs improved filters)
```bash
python backtest_v2.py
```

Compares all 3 strategies (baseline vs improved), isolated filter tests, and multi-strategy combos. Results are printed in a comparison table.

---

## Docker

```bash
# Build and run bot + dashboard
docker-compose up -d

# View logs
docker-compose logs -f bot

# Dashboard available at http://localhost:8501
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

All settings are loaded from environment variables (or `.env` file). See [.env.example](.env.example) for the full list with defaults.

| Category | Variable | Type | Default |
|----------|----------|------|---------|
| Exchange | `EXCHANGE` | str | `alpaca` |
| Alpaca | `ALPACA_API_KEY` | str | — |
| Alpaca | `ALPACA_API_SECRET` | str | — |
| Alpaca | `ALPACA_PAPER` | bool | `true` |
| Trading | `SYMBOLS` | JSON list | `["AAPL","MSFT","TSLA","NVDA","AMD"]` |
| Trading | `DEFAULT_STRATEGY` | str | `vwap_scalp` |
| Trading | `STRATEGIES` | JSON list | `["vwap_scalp","orb"]` |
| Trading | `BASE_ORDER_SIZE_USD` | float | `500` |
| Trading | `DRY_RUN` | bool | `true` |
| Filters | `AVOID_MIDDAY` | bool | `true` |
| Filters | `TREND_FILTER_ENABLED` | bool | `false` |
| Filters | `TRAILING_STOP_ENABLED` | bool | `false` |
| Timing | `POLL_INTERVAL_SECONDS` | int | `10` |
| Database | `DATABASE_URL` | str | `sqlite+aiosqlite:///data/atobot.db` |
| Logging | `LOG_LEVEL` | str | `INFO` |

---

## Development

```bash
# Install with dev extras
pip install -e ".[dev]"

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Test with coverage (152 tests)
pytest tests/ -v --tb=short
```

---

## License

MIT
