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

    # ── Trading ───────────────────────────────────────────────────────────────
    SYMBOLS: list[str] = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD"]
    DEFAULT_STRATEGY: str = "momentum"  # "momentum", "orb", "vwap_scalp"
    STRATEGIES: list[str] = []  # Run multiple strategies: ["vwap_scalp", "orb"]
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
    MOMENTUM_RSI_OVERSOLD: float = 30.0  # Buy signal threshold
    MOMENTUM_RSI_OVERBOUGHT: float = 70.0  # Sell signal threshold
    MOMENTUM_VOLUME_MULTIPLIER: float = 1.5  # Min relative volume to trade
    MOMENTUM_TAKE_PROFIT_PERCENT: float = 2.0
    MOMENTUM_STOP_LOSS_PERCENT: float = 1.0

    # ── ORB (Opening Range Breakout) Strategy ─────────────────────────────────
    ORB_RANGE_MINUTES: int = 15  # First N minutes to define opening range
    ORB_BREAKOUT_PERCENT: float = 0.1  # % above/below range to confirm
    ORB_TAKE_PROFIT_PERCENT: float = 1.5
    ORB_STOP_LOSS_PERCENT: float = 0.75
    ORB_ORDER_SIZE_USD: float = 500.0

    # ── VWAP Scalp Strategy ───────────────────────────────────────────────────
    VWAP_BOUNCE_PERCENT: float = 0.05  # % from VWAP to enter (stress test optimal)
    VWAP_TAKE_PROFIT_PERCENT: float = 0.5
    VWAP_STOP_LOSS_PERCENT: float = 0.50  # Wider SL: 52%->62% WR, Sharpe 0.88
    VWAP_ORDER_SIZE_USD: float = 500.0

    # ── Trend Filter (EMA) ────────────────────────────────────────────────────
    TREND_FILTER_ENABLED: bool = False  # Backtest showed EMA filter hurts scalping
    TREND_FILTER_EMA_PERIOD: int = 20  # EMA period for trend filter
    TREND_FILTER_TIMEFRAME: str = "15m"  # Timeframe for trend EMA bars

    # ── Time-of-Day Filter ────────────────────────────────────────────────────
    AVOID_MIDDAY: bool = True  # Skip entries during lunch dead zone
    MIDDAY_START_HOUR: int = 12  # Hour (ET) to start avoiding (noon)
    MIDDAY_END_HOUR: int = 14  # Hour (ET) to stop avoiding (2 PM)

    # ── Trailing Stop ─────────────────────────────────────────────────────────
    TRAILING_STOP_ENABLED: bool = False  # Backtest: inactive at safe params, too aggressive when tight
    TRAILING_STOP_ACTIVATION_PCT: float = 0.5  # Activate after 0.5% profit
    TRAILING_STOP_DISTANCE_PCT: float = 0.3  # Trail distance from high

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
        allowed = {"momentum", "orb", "vwap_scalp"}
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

    @model_validator(mode="after")
    def resolve_strategies_list(self) -> "Settings":
        """If STRATEGIES is empty, fall back to [DEFAULT_STRATEGY]."""
        if not self.STRATEGIES:
            self.STRATEGIES = [self.DEFAULT_STRATEGY]
        allowed = {"momentum", "orb", "vwap_scalp"}
        for s in self.STRATEGIES:
            if s not in allowed:
                raise ValueError(f"Unknown strategy in STRATEGIES: '{s}'. Allowed: {allowed}")
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
