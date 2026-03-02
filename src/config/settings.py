"""Pydantic-based settings management for AtoBot Trading."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file.

    AtoBot is a **stock day-trading bot** that uses Alpaca paper/live
    accounts to trade equities during market hours.
    """

    # ── Exchange Selection ─────────────────────────────────────────────────────
    EXCHANGE: str = "alpaca"  # "alpaca" (primary) or "binance" (crypto, future)

    # ── Alpaca ─────────────────────────────────────────────────────────────────
    ALPACA_API_KEY: str = ""
    ALPACA_API_SECRET: str = ""
    ALPACA_PAPER: bool = True  # Default to paper trading for safety

    # ── Binance (reserved for future crypto support) ───────────────────────────
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    BINANCE_TESTNET: bool = True

    # ── Streaming (WebSocket) ─────────────────────────────────────────────────
    STREAMING_ENABLED: bool = True  # Real-time price stream via StockDataStream
    TRADE_STREAM_ENABLED: bool = True  # Instant order fill/reject via TradingStream
    NEWS_STREAM_ENABLED: bool = False  # Real-time news via NewsDataStream
    DATA_FEED: str = "sip"  # "sip" (paid SIP consolidated, best fills) or "iex" (free)

    # ── Trading ───────────────────────────────────────────────────────────────
    SYMBOLS: list[str] = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD",
                          "META", "GOOGL", "AMZN", "AVGO", "NFLX",
                          "SPY", "QQQ", "CRM", "UBER", "MU"]
    DEFAULT_STRATEGY: str = "vwap_scalp"  # VWAP is strongest strategy per backtest
    STRATEGIES: list[str] = ["vwap_scalp", "ema_pullback", "momentum"]  # Multi-strategy default
    BASE_ORDER_SIZE_USD: float = 5000.0  # Scaled from $500 for real returns
    MAX_OPEN_ORDERS: int = 15
    FRACTIONAL_SHARES: bool = True  # Alpaca supports fractional shares

    # ── Market Hours ──────────────────────────────────────────────────────────
    MARKET_HOURS_ONLY: bool = True  # Only trade during regular hours
    PRE_MARKET_TRADING: bool = False  # Allow pre-market (4AM-9:30AM ET)
    AFTER_HOURS_TRADING: bool = False  # Allow after-hours (4PM-8PM ET)
    FLATTEN_EOD: bool = True  # Close all positions before market close
    FLATTEN_MINUTES_BEFORE_CLOSE: int = 5  # Minutes before close to flatten

    # ── Momentum Strategy ─────────────────────────────────────────────────────
    MOMENTUM_LOOKBACK_BARS: int = 20  # Bars for momentum calculation
    MOMENTUM_RSI_PERIOD: int = 14
    MOMENTUM_RSI_OVERSOLD: float = 32.0  # Buy signal threshold (backtest v2: 32)
    MOMENTUM_RSI_OVERBOUGHT: float = 70.0  # Sell signal threshold
    MOMENTUM_VOLUME_MULTIPLIER: float = 1.5  # Min relative volume to trade
    MOMENTUM_TAKE_PROFIT_PERCENT: float = 2.0
    MOMENTUM_STOP_LOSS_PERCENT: float = 1.0

    # ── EMA Pullback Strategy ─────────────────────────────────────────────────
    EMA_PULLBACK_FAST_PERIOD: int = 9     # Fast EMA for signal (9-bar)
    EMA_PULLBACK_SLOW_PERIOD: int = 21    # Slow EMA for trend (21-bar)
    EMA_PULLBACK_TREND_PERIOD: int = 50   # Higher-TF trend EMA (50-bar on 5m)
    EMA_PULLBACK_RSI_OVERSOLD: float = 40.0  # RSI threshold for pullback zone
    EMA_PULLBACK_RSI_OVERBOUGHT: float = 70.0  # RSI exit threshold
    EMA_PULLBACK_VOLUME_MULTIPLIER: float = 1.2  # Min relative volume
    EMA_PULLBACK_TAKE_PROFIT_PERCENT: float = 1.5  # TP %
    EMA_PULLBACK_STOP_LOSS_PERCENT: float = 0.75  # SL %
    EMA_PULLBACK_ORDER_SIZE_USD: float = 5000.0

    # ── VWAP Scalp Strategy ───────────────────────────────────────────────────
    VWAP_BOUNCE_PERCENT: float = 0.10  # backtest v2: 0.10% = +$11K vs 0.05% over-trading
    VWAP_TAKE_PROFIT_PERCENT: float = 0.5
    VWAP_STOP_LOSS_PERCENT: float = 0.30  # backtest v2: 0.30% optimal (was 0.50)
    VWAP_ORDER_SIZE_USD: float = 5000.0

    # ── Trend Filter (EMA) ────────────────────────────────────────────────────
    TREND_FILTER_ENABLED: bool = False  # Backtest showed EMA filter hurts scalping
    TREND_FILTER_EMA_PERIOD: int = 20  # EMA period for trend filter
    TREND_FILTER_TIMEFRAME: str = "15m"  # Timeframe for trend EMA bars

    # ── Time-of-Day Filter ────────────────────────────────────────────────────
    AVOID_MIDDAY: bool = False  # v3: disabled for VWAP (needs all-session volume)
    MIDDAY_START_HOUR: int = 12  # Hour (ET) to start avoiding (noon)
    MIDDAY_END_HOUR: int = 14  # Hour (ET) to stop avoiding (2 PM)

    # ── Trailing Stop ─────────────────────────────────────────────────────────
    TRAILING_STOP_ENABLED: bool = True  # v3: enabled (backtest shows trailing helps)
    TRAILING_STOP_ACTIVATION_PCT: float = 0.5  # Activate after 0.5% profit
    TRAILING_STOP_DISTANCE_PCT: float = 0.3  # Trail distance from high

    # ── Market Scanner ─────────────────────────────────────────────────────────
    SCANNER_ENABLED: bool = True  # Pre-market & intraday scanner
    SCANNER_MIN_GAP_PCT: float = 2.0  # Minimum gap % for pre-market scan
    SCANNER_MIN_RVOL: float = 1.5  # Minimum relative volume
    SCANNER_MIN_EDGE_SCORE: float = 40.0  # Min edge score (0-100) to trade
    SCANNER_MAX_RESULTS: int = 20  # Max symbols to surface per scan
    SCANNER_INTERVAL_SECONDS: int = 60  # Intraday re-scan interval
    SCANNER_UNIVERSE: str = ""  # Comma-separated symbols or empty for default

    # ── News Intelligence ─────────────────────────────────────────────────────
    NEWS_INTEL_ENABLED: bool = True  # Classify incoming news in real-time

    # ── Market Regime Detection ───────────────────────────────────────────────
    REGIME_DETECTION_ENABLED: bool = True  # Detect trend/vol/breadth regime

    # ── Stale Order Cleanup ───────────────────────────────────────────────────
    STALE_ORDER_MAX_AGE_SECONDS: int = 1800  # Cancel limit orders older than 30 min

    # ── Circuit Breaker Detection ─────────────────────────────────────────────
    CIRCUIT_BREAKER_ENABLED: bool = True  # Pause entries if SPY drops sharply
    CIRCUIT_BREAKER_SPY_DROP_PCT: float = 4.0  # SPY intraday drop % to trigger
    CIRCUIT_BREAKER_PAUSE_MINUTES: int = 30  # Minutes to pause after trigger

    # ── Overnight Gap Filter ──────────────────────────────────────────────────
    GAP_FILTER_ENABLED: bool = True  # Skip first N min if SPY gaps big overnight
    GAP_FILTER_SPY_THRESHOLD_PCT: float = 2.0  # Abs gap % to trigger filter
    GAP_FILTER_SKIP_MINUTES: int = 15  # Minutes after open to skip (VWAP needs vol)

    # ── ATR-Adaptive Stop Widening ────────────────────────────────────────────
    ATR_ADAPTIVE_STOPS: bool = True  # Widen SL when vol is high (ATR > 2x normal)
    ATR_STOP_MULTIPLIER: float = 1.5  # SL = max(baseline, ATR * this multiplier)
    ATR_STOP_MAX_WIDENING: float = 3.0  # Max SL widening factor (e.g. 3x baseline)
    ATR_NORMAL_BASELINE_PCT: float = 0.3  # Normal ATR % (for detecting elevated vol)

    # ── Crisis Position Sizing ────────────────────────────────────────────────
    CRISIS_SIZING_ENABLED: bool = True  # Reduce size instead of shutting down
    CRISIS_SIZE_MULTIPLIER: float = 0.5  # Scale to 50% in extreme vol

    # ── Zero Overnight Risk ───────────────────────────────────────────────────
    FORCE_EOD_FLATTEN: bool = True  # Hard guarantee: no positions held overnight
    EOD_FLATTEN_FAILSAFE_MINUTES: int = 2  # Failsafe: flatten 2 min before close too
    EOD_BLOCK_ENTRIES_MINUTES: int = 10  # Block new entries N min before close

    # ── ATR-based Position Sizing ─────────────────────────────────────────────
    ATR_SIZING_ENABLED: bool = False  # Use volatility-adjusted order sizes
    ATR_SIZING_PERIOD: int = 14  # ATR look-back period
    ATR_SIZING_TIMEFRAME: str = "5m"  # Timeframe for ATR bars
    ATR_RISK_DOLLARS: float = 500.0  # Scaled: max $ risk per trade (size = risk / ATR)

    # ── Risk Management ───────────────────────────────────────────────────────
    MAX_DRAWDOWN_PERCENT: float = 8.0  # Scaled for larger positions
    STOP_LOSS_PERCENT: float = 2.0
    MAX_POSITION_SIZE_USD: float = 25000.0  # Scaled: max $ in one stock
    DAILY_LOSS_LIMIT_USD: float = 2000.0  # Scaled from $200
    MAX_DAILY_TRADES: int = 40  # Increased for multi-strategy
    PDT_PROTECTION: bool = True  # Block trades that would trigger PDT rule
    RISK_PER_TRADE_PERCENT: float = 1.0  # Max % of portfolio risked per trade
    MAX_PORTFOLIO_HEAT: float = 6.0  # Max total % of portfolio at risk

    # ── OpenAI / AI Advisor ───────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"  # Cost-effective for trade analysis
    AI_ADVISOR_ENABLED: bool = False  # Enable AI-powered trade evaluation
    AI_MIN_CONFIDENCE: float = 0.4  # Min AI confidence to allow trade (0-1)
    AI_MAX_DAILY_CALLS: int = 200  # Max OpenAI API calls per day

    # ── Ultra Bot: Smart Order Manager ────────────────────────────────────────
    BRACKET_ORDERS_ENABLED: bool = False    # v3: brackets hurt R:R (cut avg win)
    BRACKET_TP1_PCT: float = 1.5            # Take-profit 1 percent (50% of position)
    BRACKET_TP2_PCT: float = 3.0            # Take-profit 2 percent (remaining)
    BRACKET_TP1_SIZE: float = 0.5           # Fraction of qty for TP1
    BRACKET_SL_PCT: float = 2.0             # Stop-loss percent
    BRACKET_MAX_SCALE_INS: int = 2          # Max pyramid add-ons
    BRACKET_TRAILING_ACTIVATION: float = 1.0  # % gain to activate trailing
    BRACKET_TRAILING_DISTANCE: float = 0.5  # Trailing stop distance %

    # ── Ultra Bot: Position Sizing (Kelly) ────────────────────────────────────
    KELLY_SIZING_ENABLED: bool = True       # Use Kelly criterion sizing (v5: enabled)
    KELLY_FRACTION: float = 0.5             # Half-Kelly (conservative)
    MAX_RISK_PER_TRADE: float = 0.02        # 2% of account per trade
    MAX_PORTFOLIO_HEAT: float = 0.06        # 6% total portfolio risk
    MAX_POSITION_PCT: float = 0.10          # 10% max single position
    MAX_SECTOR_CONCENTRATION: float = 0.25  # 25% max in one sector
    MIN_KELLY_TRADES: int = 20              # Min trades for Kelly

    # ── Ultra Bot: Confluence Gate ────────────────────────────────────────────
    CONFLUENCE_GATE_ENABLED: bool = True    # Require min confluence score to enter
    CONFLUENCE_MIN_SCORE: int = 30          # Min 0-100 score (30 = filter weakest)
    CONFLUENCE_BARS_NEEDED: int = 60        # Bars needed for confluence calc

    # ── Ultra Bot: Progressive Risk Scaling ───────────────────────────────────
    PROGRESSIVE_RISK_ENABLED: bool = True   # Reduce size after consecutive losses
    PROGRESSIVE_LOSS_MULTIPLIER: float = 0.75  # Size multiplier per consec loss
    PROGRESSIVE_MIN_MULTIPLIER: float = 0.25   # Floor (never below 25% size)
    PROGRESSIVE_MAX_LOSSES: int = 5         # Max consec losses before cooldown

    # ── Ultra Bot: Strategy Selector ──────────────────────────────────────────
    STRATEGY_SELECTOR_ENABLED: bool = True  # Regime-aware strategy rotation
    STRATEGY_MAX_CONSECUTIVE_LOSSES: int = 5  # Auto-cooldown after N losses
    STRATEGY_COOLDOWN_MINUTES: int = 30     # Cooldown duration
    STRATEGY_MIN_WEIGHT: float = 0.2        # Min regime weight to trade

    # ── Profit Goals (Daily / Weekly / Monthly Targets) ────────────────────────
    PROFIT_GOALS_ENABLED: bool = True        # Track P&L against profit targets
    PROFIT_GOAL_DAILY: float = 500.0         # Daily profit target in USD
    PROFIT_GOAL_WEEKLY: float = 2500.0       # Weekly profit target in USD
    PROFIT_GOAL_MONTHLY: float = 10000.0     # Monthly profit target in USD
    PROFIT_GOAL_DAILY_LOSS_LIMIT: float = 0.0  # Hard daily loss stop (0 = use DAILY_LOSS_LIMIT_USD)
    PROFIT_GOAL_MET_RISK_SCALE: float = 0.25   # Scale risk to 25% after hitting daily goal
    PROFIT_GOAL_LOSING_RISK_SCALE: float = 0.50  # Scale risk to 50% when losing on the day

    # ── Ultra Bot: Trade Journal ──────────────────────────────────────────────
    TRADE_JOURNAL_ENABLED: bool = True      # Track all trades for analytics
    TRADE_JOURNAL_DIR: str = "data"         # Directory for journal files

    # ── Ultra Bot: ML Feature Engine ──────────────────────────────────────────
    ML_FEATURES_ENABLED: bool = True        # Compute ML features per trade
    ML_WIN_PROB_GATE: float = 0.0           # Min win probability to enter (0=disabled)

    # ── Short Selling ─────────────────────────────────────────────────────────
    SHORT_SELLING_ENABLED: bool = True       # Allow short entries on bearish signals
    SHORT_TREND_FILTER: bool = True          # Require bearish trend for shorts
    SHORT_MAX_POSITION_USD: float = 25000.0  # Max $ per short position
    SHORT_LOCATE_CHECK: bool = False         # Skip hard-to-borrow check (Alpaca handles)

    # ── Limit Order Entries ───────────────────────────────────────────────────
    LIMIT_ENTRY_ENABLED: bool = True         # Use limit orders for entries (better fills)
    LIMIT_OFFSET_PCT: float = 0.02           # Offset from signal price (e.g. 0.02% below for buys)
    LIMIT_ORDER_TIMEOUT_SECONDS: int = 120   # Cancel unfilled limit entries after N seconds

    # ── Correlation-Based Risk ────────────────────────────────────────────────
    CORRELATION_RISK_ENABLED: bool = True    # Check inter-position correlation
    MAX_CORRELATED_EXPOSURE: float = 0.40    # Max 40% of portfolio in correlated assets (r>0.7)
    CORRELATION_LOOKBACK_DAYS: int = 60      # Days of daily returns for correlation matrix
    CORRELATION_THRESHOLD: float = 0.70      # Pearson r threshold for "correlated" pair

    # ── Value-at-Risk (VaR) ───────────────────────────────────────────────────
    VAR_ENABLED: bool = True                 # Pre-trade VaR check
    VAR_CONFIDENCE: float = 0.95             # 95% confidence level
    VAR_MAX_PORTFOLIO_PCT: float = 0.03      # Max 3% daily VaR as % of portfolio
    VAR_LOOKBACK_DAYS: int = 30              # Days for historical VaR
    VAR_METHOD: str = "historical"           # "historical" or "parametric"

    # ── Walk-Forward Optimization ─────────────────────────────────────────────
    WALK_FORWARD_ENABLED: bool = True        # Periodic walk-forward re-optimization
    WALK_FORWARD_TRAIN_DAYS: int = 180       # Training window (6 months)
    WALK_FORWARD_TEST_DAYS: int = 30         # Test/validation window (1 month)
    WALK_FORWARD_INTERVAL_HOURS: int = 168   # Re-optimize every week (168h)

    # ── Pairs Trading Strategy ────────────────────────────────────────────────
    PAIRS_TRADING_ENABLED: bool = True       # Enable pairs/stat-arb strategy
    PAIRS: list[str] = ["NVDA:AMD", "GOOGL:META", "MSFT:AAPL"]  # Colon-separated pairs
    PAIRS_LOOKBACK_DAYS: int = 60            # Days for spread calculation
    PAIRS_ENTRY_ZSCORE: float = 2.0          # Enter when z-score exceeds this
    PAIRS_EXIT_ZSCORE: float = 0.5           # Exit when z-score reverts to this
    PAIRS_STOP_ZSCORE: float = 3.5           # Stop-loss z-score
    PAIRS_ORDER_SIZE_USD: float = 5000.0     # Per-leg notional
    PAIRS_MAX_HOLDING_BARS: int = 100        # Max holding period (5min bars)

    # ── Swing Strategy (Small Account Growth) ─────────────────────────────
    SWING_RSI_OVERSOLD: float = 38.0          # RSI entry threshold
    SWING_RSI_OVERBOUGHT: float = 70.0        # RSI exit threshold
    SWING_VOLUME_SURGE: float = 1.3           # Min rel volume for entry
    SWING_MIN_CONFLUENCE: int = 2             # Min signals needed for entry
    SWING_TAKE_PROFIT_PCT: float = 3.0        # Base take-profit %
    SWING_STOP_LOSS_PCT: float = 1.5          # Base stop-loss %
    SWING_TRAILING_ACTIVATION_PCT: float = 1.5  # Activate trailing at +1.5%
    SWING_TRAILING_OFFSET_PCT: float = 0.75   # Trail distance from high
    SWING_MAX_HOLD_DAYS: int = 5              # Max days before time stop
    SWING_MAX_POSITIONS: int = 3              # Max concurrent swing positions
    SWING_RISK_PER_TRADE_PCT: float = 3.0     # Risk 3% per trade (aggressive)
    SWING_ORDER_SIZE_USD: float = 250.0       # Fallback order size for $500 acct
    SWING_EQUITY_CAP: float = 0.0             # Cap equity for sizing (0 = use real)
    SWING_MAX_GAP_PCT: float = 5.0            # Max overnight gap % before exit
    SWING_SYMBOLS: str = "AAPL,MSFT,NVDA,TSLA,AMD,META,GOOGL,AMZN"  # Swing universe

    # ── Crypto Swing Strategy (24/7 Alpaca Crypto) ────────────────────────
    # Backtest-optimized (Config D: Conservative) — 6mo +22.9%, PF 1.53
    CRYPTO_ENABLED: bool = True              # Master switch for crypto trading
    CRYPTO_SYMBOLS: str = "BTC/USD,ETH/USD,SOL/USD,AVAX/USD,LINK/USD,DOGE/USD,DOT/USD,LTC/USD"
    CRYPTO_RSI_OVERSOLD: float = 35.0        # RSI entry threshold (crypto is volatile)
    CRYPTO_RSI_OVERBOUGHT: float = 75.0      # RSI exit threshold
    CRYPTO_VOLUME_SURGE: float = 1.5         # Min rel volume for entry
    CRYPTO_MIN_CONFLUENCE: int = 3           # Min signals needed (3 = more selective)
    CRYPTO_TAKE_PROFIT_PCT: float = 10.0     # Take-profit % (wide for crypto swings)
    CRYPTO_STOP_LOSS_PCT: float = 5.0        # Stop-loss % (crypto needs breathing room)
    CRYPTO_TRAILING_ACTIVATION_PCT: float = 5.0  # Activate trailing at +5%
    CRYPTO_TRAILING_OFFSET_PCT: float = 2.5  # Trail distance from high
    CRYPTO_MAX_HOLD_DAYS: int = 14           # Max days before time stop
    CRYPTO_MAX_POSITIONS: int = 4            # Max concurrent crypto positions
    CRYPTO_MAX_ALT_POSITIONS: int = 3        # Max altcoin positions (correlation limit)
    CRYPTO_RISK_PER_TRADE_PCT: float = 4.0   # Risk 4% per trade (aggressive)
    CRYPTO_ORDER_SIZE_USD: float = 200.0     # Fallback order size for $500 acct
    CRYPTO_EQUITY_CAP: float = 0.0           # Cap equity for sizing (0 = use real)
    CRYPTO_BTC_TREND_GATE: bool = True       # Only enter alts if BTC trend is up
    CRYPTO_BTC_PANIC_RSI: float = 30.0       # Block ALL alt longs when BTC RSI < this
    CRYPTO_FEE_BPS: float = 25.0             # Alpaca taker fee (basis points)

    # ── Crypto v2 Enhancements ────────────────────────────────────────────
    CRYPTO_ADX_FILTER_ENABLED: bool = True   # Skip entry if ADX < threshold (choppy)
    CRYPTO_ADX_MIN_TREND: float = 20.0       # ADX minimum for trending market
    CRYPTO_BB_FILTER_ENABLED: bool = True     # Bollinger Band lower-half filter
    CRYPTO_BB_PERIOD: int = 20               # BB lookback period
    CRYPTO_BB_STD: float = 2.0               # BB standard deviations
    CRYPTO_MACD_ENABLED: bool = True         # MACD histogram rising as confluence
    CRYPTO_DAILY_TREND_GATE: bool = True     # Daily EMA20>EMA50 + RSI>45 gate
    CRYPTO_MULTI_TP_ENABLED: bool = True     # Multi-level take-profit (3 stages)
    CRYPTO_TP1_PCT: float = 5.0              # TP1: sell 33% at +5%
    CRYPTO_TP2_PCT: float = 8.0              # TP2: sell 33% at +8%
    CRYPTO_TP3_PCT: float = 12.0             # TP3: sell 34% at +12%
    CRYPTO_DYNAMIC_STOPS: bool = True        # ATR-based dynamic stop-loss (3-7%)
    CRYPTO_FEAR_GREED_ENABLED: bool = True   # Fear & Greed Index sizing adjustment

    # ── MACD Entry Confirmation ───────────────────────────────────────────────
    MACD_CONFIRMATION_ENABLED: bool = False  # Disabled — only Momentum uses MACD
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9

    # ── Notifications ─────────────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    NOTIFICATIONS_ENABLED: bool = True

    # ── Dashboard ─────────────────────────────────────────────────────────────
    DASHBOARD_ENABLED: bool = False
    DASHBOARD_PORT: int = 8501

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///data/atobot.db"

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/atobot.log"

    # ── General ───────────────────────────────────────────────────────────────
    POLL_INTERVAL_SECONDS: int = 5
    DRY_RUN: bool = True  # Default to dry-run for safety

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("SYMBOLS", mode="before")
    @classmethod
    def parse_symbols(cls, v: Any) -> list[str]:
        """Accept JSON string or list for SYMBOLS."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [s.upper() for s in parsed]
            except json.JSONDecodeError:
                return [s.strip().upper() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [s.upper() for s in v]
        return v

    @field_validator("STRATEGIES", mode="before")
    @classmethod
    def parse_strategies(cls, v: Any) -> list[str]:
        """Accept JSON string, comma-separated string, or list for STRATEGIES."""
        if isinstance(v, str):
            if not v.strip():
                return []
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [s.lower() for s in parsed]
            except json.JSONDecodeError:
                return [s.strip().lower() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return [s.lower() for s in v]
        return v

    @field_validator("EXCHANGE")
    @classmethod
    def validate_exchange(cls, v: str) -> str:
        """Ensure exchange name is valid."""
        allowed = {"binance", "alpaca"}
        if v.lower() not in allowed:
            raise ValueError(f"EXCHANGE must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("DEFAULT_STRATEGY")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Ensure strategy name is valid."""
        allowed = {"momentum", "vwap_scalp", "ema_pullback", "pairs", "swing", "crypto_swing"}
        if v.lower() not in allowed:
            raise ValueError(f"DEFAULT_STRATEGY must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("PAIRS", mode="before")
    @classmethod
    def parse_pairs(cls, v: Any) -> list[str]:
        """Accept JSON string, comma-separated string, or list for PAIRS."""
        if isinstance(v, str):
            if not v.strip():
                return []
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, list):
            return v
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got '{v}'")
        return v.upper()

    @field_validator("DATA_FEED")
    @classmethod
    def validate_data_feed(cls, v: str) -> str:
        """Ensure data feed is valid."""
        allowed = {"iex", "sip"}
        if v.lower() not in allowed:
            raise ValueError(f"DATA_FEED must be one of {allowed}, got '{v}'")
        return v.lower()

    @model_validator(mode="after")
    def resolve_strategies_list(self) -> "Settings":
        """If STRATEGIES is empty, fall back to [DEFAULT_STRATEGY]."""
        if not self.STRATEGIES:
            self.STRATEGIES = [self.DEFAULT_STRATEGY]
        allowed = {"momentum", "vwap_scalp", "ema_pullback", "pairs", "swing", "crypto_swing"}
        for s in self.STRATEGIES:
            if s not in allowed:
                raise ValueError(f"Unknown strategy in STRATEGIES: '{s}'. Allowed: {allowed}")
        return self

    @model_validator(mode="after")
    def resolve_ai_advisor(self) -> "Settings":
        """Auto-enable AI advisor if OPENAI_API_KEY is set."""
        if self.OPENAI_API_KEY and not self.AI_ADVISOR_ENABLED:
            self.AI_ADVISOR_ENABLED = True
        return self

    @model_validator(mode="after")
    def validate_api_keys_for_live(self) -> "Settings":
        """API keys must be set when not using paper/testnet."""
        if self.EXCHANGE == "binance" and not self.BINANCE_TESTNET:
            if not self.BINANCE_API_KEY or not self.BINANCE_API_SECRET:
                raise ValueError(
                    "BINANCE_API_KEY and BINANCE_API_SECRET are required "
                    "when BINANCE_TESTNET is False"
                )
        if self.EXCHANGE == "alpaca" and not self.ALPACA_PAPER:
            if not self.ALPACA_API_KEY or not self.ALPACA_API_SECRET:
                raise ValueError(
                    "ALPACA_API_KEY and ALPACA_API_SECRET are required "
                    "when ALPACA_PAPER is False"
                )
        return self

    @model_validator(mode="after")
    def validate_telegram_config(self) -> "Settings":
        """Telegram credentials must be set when notifications are enabled."""
        if self.NOTIFICATIONS_ENABLED:
            if not self.TELEGRAM_BOT_TOKEN or not self.TELEGRAM_CHAT_ID:
                self.NOTIFICATIONS_ENABLED = False
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
