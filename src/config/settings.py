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
    DATA_FEED: str = "iex"  # "iex" (free) or "sip" (paid SIP consolidated)

    # ── Trading ───────────────────────────────────────────────────────────────
    SYMBOLS: list[str] = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD",
                          "META", "GOOGL", "AMZN", "AVGO", "NFLX",
                          "SPY", "QQQ", "CRM", "UBER", "MU"]
    DEFAULT_STRATEGY: str = "vwap_scalp"  # VWAP is strongest strategy per backtest
    STRATEGIES: list[str] = []  # Run multiple: ["vwap_scalp", "ema_pullback", "momentum"]
    BASE_ORDER_SIZE_USD: float = 500.0  # Base dollar amount per trade
    MAX_OPEN_ORDERS: int = 10
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

    # ── ORB (Opening Range Breakout) Strategy ─────────────────────────────────
    ORB_RANGE_MINUTES: int = 15  # First N minutes to define opening range
    ORB_BREAKOUT_PERCENT: float = 0.15  # % above range to confirm (backtest v3: 0.15)
    ORB_TAKE_PROFIT_PERCENT: float = 1.5  # Single TP (no brackets in v3)
    ORB_STOP_LOSS_PERCENT: float = 0.75  # Proven via backtest (0.50 too tight)
    ORB_ORDER_SIZE_USD: float = 500.0

    # ── EMA Pullback Strategy ─────────────────────────────────────────────────
    EMA_PULLBACK_FAST_PERIOD: int = 9     # Fast EMA for signal (9-bar)
    EMA_PULLBACK_SLOW_PERIOD: int = 21    # Slow EMA for trend (21-bar)
    EMA_PULLBACK_TREND_PERIOD: int = 50   # Higher-TF trend EMA (50-bar on 5m)
    EMA_PULLBACK_RSI_OVERSOLD: float = 40.0  # RSI threshold for pullback zone
    EMA_PULLBACK_RSI_OVERBOUGHT: float = 70.0  # RSI exit threshold
    EMA_PULLBACK_VOLUME_MULTIPLIER: float = 1.2  # Min relative volume
    EMA_PULLBACK_TAKE_PROFIT_PERCENT: float = 1.5  # TP %
    EMA_PULLBACK_STOP_LOSS_PERCENT: float = 0.75  # SL %
    EMA_PULLBACK_ORDER_SIZE_USD: float = 500.0

    # ── VWAP Scalp Strategy ───────────────────────────────────────────────────
    VWAP_BOUNCE_PERCENT: float = 0.10  # backtest v2: 0.10% = +$11K vs 0.05% over-trading
    VWAP_TAKE_PROFIT_PERCENT: float = 0.5
    VWAP_STOP_LOSS_PERCENT: float = 0.30  # backtest v2: 0.30% optimal (was 0.50)
    VWAP_ORDER_SIZE_USD: float = 500.0

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

    # ── ATR-based Position Sizing ─────────────────────────────────────────────
    ATR_SIZING_ENABLED: bool = False  # Use volatility-adjusted order sizes
    ATR_SIZING_PERIOD: int = 14  # ATR look-back period
    ATR_SIZING_TIMEFRAME: str = "5m"  # Timeframe for ATR bars
    ATR_RISK_DOLLARS: float = 50.0  # Max $ risk per trade (size = risk / ATR)

    # ── Risk Management ───────────────────────────────────────────────────────
    MAX_DRAWDOWN_PERCENT: float = 5.0  # Tighter for day trading
    STOP_LOSS_PERCENT: float = 2.0
    MAX_POSITION_SIZE_USD: float = 2000.0  # Max $ in one stock
    DAILY_LOSS_LIMIT_USD: float = 200.0
    MAX_DAILY_TRADES: int = 20  # PDT awareness
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

    # ── Ultra Bot: Trade Journal ──────────────────────────────────────────────
    TRADE_JOURNAL_ENABLED: bool = True      # Track all trades for analytics
    TRADE_JOURNAL_DIR: str = "data"         # Directory for journal files

    # ── Ultra Bot: ML Feature Engine ──────────────────────────────────────────
    ML_FEATURES_ENABLED: bool = False       # Compute ML features per trade
    ML_WIN_PROB_GATE: float = 0.0           # Min win probability to enter (0=disabled)

    # ── MACD Entry Confirmation ───────────────────────────────────────────────
    MACD_CONFIRMATION_ENABLED: bool = False  # v3: disabled for VWAP/ORB (only Momentum uses MACD)
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
        allowed = {"momentum", "orb", "vwap_scalp", "ema_pullback"}
        if v.lower() not in allowed:
            raise ValueError(f"DEFAULT_STRATEGY must be one of {allowed}, got '{v}'")
        return v.lower()

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
        allowed = {"momentum", "orb", "vwap_scalp", "ema_pullback"}
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
