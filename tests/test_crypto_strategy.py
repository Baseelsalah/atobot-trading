"""Tests for CryptoSwingStrategy and engine crypto awareness."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.config.settings import Settings
from src.core.engine import TradingEngine
from src.data.market_data import MarketDataProvider
from src.models.order import Order, OrderSide, OrderType
from src.models.position import Position
from src.persistence.repository import TradingRepository
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy
from src.strategies.crypto_strategy import CryptoSwingStrategy


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def crypto_settings(mock_settings: Settings) -> Settings:
    """Settings configured for crypto testing."""
    s = mock_settings
    s.CRYPTO_ENABLED = True
    s.CRYPTO_SYMBOLS = "BTC/USD,ETH/USD"
    s.CRYPTO_RSI_OVERSOLD = 35.0
    s.CRYPTO_RSI_OVERBOUGHT = 75.0
    s.CRYPTO_VOLUME_SURGE = 1.5
    s.CRYPTO_MIN_CONFLUENCE = 2
    s.CRYPTO_TAKE_PROFIT_PCT = 5.0
    s.CRYPTO_STOP_LOSS_PCT = 3.0
    s.CRYPTO_TRAILING_ACTIVATION_PCT = 2.5
    s.CRYPTO_TRAILING_OFFSET_PCT = 1.5
    s.CRYPTO_MAX_HOLD_DAYS = 7
    s.CRYPTO_MAX_POSITIONS = 2
    s.CRYPTO_RISK_PER_TRADE_PCT = 4.0
    s.CRYPTO_ORDER_SIZE_USD = 200.0
    s.CRYPTO_EQUITY_CAP = 500.0
    s.CRYPTO_BTC_TREND_GATE = True
    s.CRYPTO_FEE_BPS = 25.0
    s.SYMBOLS = ["AAPL", "BTC/USD", "ETH/USD"]
    s.STRATEGIES = ["vwap_scalp", "crypto_swing"]
    return s


@pytest.fixture
def crypto_exchange(mock_exchange_client: AsyncMock) -> AsyncMock:
    """Exchange client with crypto-compatible responses."""
    client = mock_exchange_client

    # Build 100 bars of dummy 4H crypto data with realistic structure
    bars = []
    base_price = 50000.0
    for i in range(100):
        close = base_price + (i * 10) - 200
        bars.append({
            "timestamp": 1700000000000 + i * 14400000,
            "open": Decimal(str(close - 50)),
            "high": Decimal(str(close + 100)),
            "low": Decimal(str(close - 150)),
            "close": Decimal(str(close)),
            "volume": Decimal("500000"),
        })
    client.get_klines.return_value = bars

    # Crypto price
    client.get_ticker_price.return_value = Decimal("50800")

    # Account balance
    client.get_account_balance.return_value = {
        "USD": Decimal("500"),
        "EQUITY": Decimal("500"),
        "BUYING_POWER": Decimal("500"),
        "DAYTRADE_COUNT": Decimal("0"),
    }

    return client


@pytest.fixture
def crypto_strategy(
    crypto_exchange: AsyncMock,
    mock_risk_manager: RiskManager,
    crypto_settings: Settings,
) -> CryptoSwingStrategy:
    """A CryptoSwingStrategy instance for testing."""
    return CryptoSwingStrategy(crypto_exchange, mock_risk_manager, crypto_settings)


# ── Strategy Basics ───────────────────────────────────────────────────────

class TestCryptoSwingBasics:
    """Test basic properties and initialization."""

    def test_strategy_name(self, crypto_strategy: CryptoSwingStrategy) -> None:
        assert crypto_strategy.name == "crypto_swing"

    def test_exempt_eod_flatten(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Crypto positions must NOT be flattened at EOD."""
        assert crypto_strategy.exempt_eod_flatten is True

    def test_params_loaded(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Verify all crypto params are loaded from settings."""
        assert crypto_strategy._take_profit_pct == 5.0
        assert crypto_strategy._stop_loss_pct == 3.0
        assert crypto_strategy._trailing_act_pct == 2.5
        assert crypto_strategy._trailing_offset_pct == 1.5
        assert crypto_strategy._max_hold_days == 7
        assert crypto_strategy._max_positions == 2
        assert crypto_strategy._risk_per_trade == 4.0
        assert crypto_strategy._equity_cap == 500.0
        assert crypto_strategy._btc_trend_gate is True
        assert crypto_strategy._fee_bps == 25.0

    @pytest.mark.asyncio
    async def test_initialize(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Initialize should fetch bars for the symbol."""
        await crypto_strategy.initialize("BTC/USD")
        assert "BTC/USD" in crypto_strategy._initialized_symbols
        crypto_strategy.exchange.get_klines.assert_called()

    @pytest.mark.asyncio
    async def test_get_status(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """get_status returns well-formed dict."""
        status = await crypto_strategy.get_status()
        assert status["strategy"] == "crypto_swing"
        assert status["total_active"] == 0
        assert status["max_positions"] == 2


# ── Technical Indicators ─────────────────────────────────────────────────

class TestCryptoIndicators:
    """Test the static indicator helper methods."""

    def _make_df(self, closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
        n = len(closes)
        if volumes is None:
            volumes = [100000.0] * n
        return pd.DataFrame({
            "open": [c - 1 for c in closes],
            "high": [c + 2 for c in closes],
            "low": [c - 3 for c in closes],
            "close": closes,
            "volume": volumes,
        })

    def test_calc_rsi(self) -> None:
        # Steadily rising closes -> RSI should be high
        closes = [100 + i * 0.5 for i in range(30)]
        df = self._make_df(closes)
        rsi = CryptoSwingStrategy._calc_rsi(df, 14)
        assert rsi is not None
        assert rsi > 50.0

    def test_calc_rsi_insufficient_data(self) -> None:
        df = self._make_df([100, 101, 102])
        rsi = CryptoSwingStrategy._calc_rsi(df, 14)
        assert rsi is None

    def test_calc_ema(self) -> None:
        closes = [100 + i for i in range(30)]
        df = self._make_df(closes)
        ema = CryptoSwingStrategy._calc_ema(df, 20)
        assert ema is not None
        assert ema > 100

    def test_calc_atr(self) -> None:
        closes = [100 + i for i in range(30)]
        df = self._make_df(closes)
        atr = CryptoSwingStrategy._calc_atr(df, 14)
        assert atr is not None
        assert atr > 0

    def test_avg_volume(self) -> None:
        vols = [50000.0] * 25
        df = self._make_df([100] * 25, vols)
        avg = CryptoSwingStrategy._avg_volume(df, 20)
        assert avg == 50000.0

    def test_is_bullish_candle_hammer(self) -> None:
        # Hammer: small body, long lower wick
        df = pd.DataFrame({
            "open": [100, 100],
            "high": [101, 101],
            "low": [98, 95],    # long lower wick on last candle
            "close": [99, 100.5],
            "volume": [1000, 1000],
        })
        assert CryptoSwingStrategy._is_bullish_candle(df) is True

    def test_is_bullish_candle_engulfing(self) -> None:
        # Bullish engulfing: prev bearish, current bullish and engulfs prev
        df = pd.DataFrame({
            "open": [102, 98.5],   # prev bearish (open > close), curr bullish
            "high": [103, 104],
            "low": [98, 97],
            "close": [99, 103],  # curr close > prev open, curr open < prev close
            "volume": [1000, 1000],
        })
        assert CryptoSwingStrategy._is_bullish_candle(df) is True


# ── Position Management ──────────────────────────────────────────────────

class TestCryptoPositionManagement:
    """Test stop loss, take profit, trailing stop, and time stop."""

    @pytest.mark.asyncio
    async def test_stop_loss_triggered(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Price drop below stop triggers exit."""
        entry = Decimal("50000")
        pos = Position(
            symbol="BTC/USD", entry_price=entry, current_price=entry,
            quantity=Decimal("0.01"), side="LONG", strategy="crypto_swing",
        )
        crypto_strategy.positions["BTC/USD"] = pos
        crypto_strategy._entry_dates["BTC/USD"] = datetime.now(timezone.utc)
        crypto_strategy._swing_highs["BTC/USD"] = entry
        crypto_strategy._swing_stops["BTC/USD"] = entry * Decimal("0.97")  # 3% SL
        crypto_strategy._swing_targets["BTC/USD"] = entry * Decimal("1.05")
        crypto_strategy._trailing_active["BTC/USD"] = False

        # Price dropped below stop
        stop_price = entry * Decimal("0.96")  # 4% drop, below 3% stop
        orders = await crypto_strategy._manage_position("BTC/USD", pos, stop_price)
        assert len(orders) == 1
        assert str(orders[0].side).upper() == "SELL"

    @pytest.mark.asyncio
    async def test_take_profit_triggered(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Price rise above target triggers exit."""
        entry = Decimal("50000")
        pos = Position(
            symbol="BTC/USD", entry_price=entry, current_price=entry,
            quantity=Decimal("0.01"), side="LONG", strategy="crypto_swing",
        )
        crypto_strategy.positions["BTC/USD"] = pos
        crypto_strategy._entry_dates["BTC/USD"] = datetime.now(timezone.utc)
        crypto_strategy._swing_highs["BTC/USD"] = entry
        crypto_strategy._swing_stops["BTC/USD"] = entry * Decimal("0.97")
        crypto_strategy._swing_targets["BTC/USD"] = entry * Decimal("1.05")
        crypto_strategy._trailing_active["BTC/USD"] = False

        # Price above take profit
        orders = await crypto_strategy._manage_position(
            "BTC/USD", pos, Decimal("53000")
        )
        assert len(orders) == 1
        assert str(orders[0].side).upper() == "SELL"

    @pytest.mark.asyncio
    async def test_trailing_stop_activation(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Trailing stop activates at the configured percentage."""
        entry = Decimal("50000")
        pos = Position(
            symbol="BTC/USD", entry_price=entry, current_price=entry,
            quantity=Decimal("0.01"), side="LONG", strategy="crypto_swing",
        )
        crypto_strategy.positions["BTC/USD"] = pos
        crypto_strategy._entry_dates["BTC/USD"] = datetime.now(timezone.utc)
        crypto_strategy._swing_highs["BTC/USD"] = entry
        crypto_strategy._swing_stops["BTC/USD"] = entry * Decimal("0.97")
        crypto_strategy._swing_targets["BTC/USD"] = entry * Decimal("1.05")
        crypto_strategy._trailing_active["BTC/USD"] = False

        # +3% gain (above 2.5% trailing activation), but below TP
        orders = await crypto_strategy._manage_position(
            "BTC/USD", pos, Decimal("51300")  # +2.6%
        )
        assert crypto_strategy._trailing_active["BTC/USD"] is True
        # Should not exit yet — just activate trailing
        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_time_stop_triggered(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Position held beyond max days triggers exit."""
        entry = Decimal("50000")
        pos = Position(
            symbol="BTC/USD", entry_price=entry, current_price=entry,
            quantity=Decimal("0.01"), side="LONG", strategy="crypto_swing",
        )
        crypto_strategy.positions["BTC/USD"] = pos
        crypto_strategy._entry_dates["BTC/USD"] = datetime.now(timezone.utc) - timedelta(days=8)
        crypto_strategy._swing_highs["BTC/USD"] = entry
        crypto_strategy._swing_stops["BTC/USD"] = entry * Decimal("0.97")
        crypto_strategy._swing_targets["BTC/USD"] = entry * Decimal("1.05")
        crypto_strategy._trailing_active["BTC/USD"] = False

        # Price flat — no SL/TP, but time stop fires
        orders = await crypto_strategy._manage_position(
            "BTC/USD", pos, Decimal("50100")
        )
        assert len(orders) == 1
        assert str(orders[0].side).upper() == "SELL"


# ── Order Fill Handling ──────────────────────────────────────────────────

class TestCryptoOrderFills:
    """Test on_order_filled for creating/closing positions."""

    @pytest.mark.asyncio
    async def test_buy_fill_creates_position(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """A BUY fill should create a tracked position."""
        order = Order(
            symbol="BTC/USD", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=Decimal("0.01"), price=Decimal("50000"),
            strategy="crypto_swing",
        )
        order.mark_filled()

        await crypto_strategy.on_order_filled(order)

        assert "BTC/USD" in crypto_strategy.positions
        pos = crypto_strategy.positions["BTC/USD"]
        assert pos.entry_price == Decimal("50000")
        assert "BTC/USD" in crypto_strategy._swing_stops
        assert "BTC/USD" in crypto_strategy._swing_targets

    @pytest.mark.asyncio
    async def test_sell_fill_closes_position(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """A SELL fill should close the tracked position."""
        # First create a position
        buy = Order(
            symbol="ETH/USD", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=Decimal("0.1"), price=Decimal("3000"),
            strategy="crypto_swing",
        )
        buy.mark_filled()
        await crypto_strategy.on_order_filled(buy)
        assert "ETH/USD" in crypto_strategy.positions

        # Now sell fill
        sell = Order(
            symbol="ETH/USD", side=OrderSide.SELL, order_type=OrderType.MARKET,
            quantity=Decimal("0.1"), price=Decimal("3150"),
            strategy="crypto_swing",
        )
        sell.mark_filled()
        await crypto_strategy.on_order_filled(sell)

        pos = crypto_strategy.positions["ETH/USD"]
        assert pos.is_closed
        # Tracking state should be cleaned up
        assert "ETH/USD" not in crypto_strategy._swing_stops


# ── Position Sizing ──────────────────────────────────────────────────────

class TestCryptoSizing:
    """Test crypto-specific position sizing with fee adjustment."""

    @pytest.mark.asyncio
    async def test_size_respects_equity_cap(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Position size should use equity cap, not full account equity."""
        qty = await crypto_strategy._calc_crypto_size("BTC/USD", Decimal("50000"))
        assert qty > 0
        # With $500 cap, 4% risk = $20, 3% SL + 0.5% fees = 3.5%
        # Max position ~$571, so qty ~$571 / $50000 = ~0.01
        assert qty < 0.02  # Shouldn't be huge

    @pytest.mark.asyncio
    async def test_size_fallback_on_error(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """If balance fetch fails, uses fallback order size."""
        crypto_strategy.exchange.get_account_balance.side_effect = Exception("API error")
        qty = await crypto_strategy._calc_crypto_size("BTC/USD", Decimal("50000"))
        # Fallback: $200 / $50000 = 0.004
        assert qty == pytest.approx(0.004, abs=0.001)


# ── BTC Trend Gate ────────────────────────────────────────────────────────

class TestBTCTrendGate:

    @pytest.mark.asyncio
    async def test_btc_trend_bullish(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """BTC uptrend returns True."""
        # Default klines fixture has rising prices so 20-EMA > 50-EMA
        result = await crypto_strategy._check_btc_trend()
        assert result is True

    @pytest.mark.asyncio
    async def test_btc_trend_cached(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Second call within 4 hours uses cache, no API call."""
        await crypto_strategy._check_btc_trend()
        call_count = crypto_strategy.exchange.get_klines.call_count

        await crypto_strategy._check_btc_trend()
        assert crypto_strategy.exchange.get_klines.call_count == call_count  # No new call

    @pytest.mark.asyncio
    async def test_btc_trend_fail_open(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """On API error, trend gate fails open (returns True)."""
        crypto_strategy.exchange.get_klines.side_effect = Exception("API error")
        result = await crypto_strategy._check_btc_trend()
        assert result is True  # Fail-open


# ── Engine Crypto Awareness ──────────────────────────────────────────────

class TestEngineCryptoAwareness:
    """Test that the engine correctly routes and handles crypto symbols."""

    def test_is_crypto_symbol(self) -> None:
        """Engine._is_crypto_symbol correctly identifies crypto pairs."""
        assert TradingEngine._is_crypto_symbol("BTC/USD") is True
        assert TradingEngine._is_crypto_symbol("ETH/USD") is True
        assert TradingEngine._is_crypto_symbol("SOL/USD") is True
        assert TradingEngine._is_crypto_symbol("AAPL") is False
        assert TradingEngine._is_crypto_symbol("MSFT") is False
        assert TradingEngine._is_crypto_symbol("TSLA") is False

    def _make_engine(
        self,
        settings: Settings,
        exchange: AsyncMock,
        risk_manager: RiskManager,
        strategies: list | None = None,
    ) -> TradingEngine:
        """Create engine with mocked deps."""
        if strategies is None:
            strat = AsyncMock(spec=BaseStrategy)
            strat.name = "vwap_scalp"
            strat.active_orders = []
            strat.positions = {}
            strat.on_tick = AsyncMock(return_value=[])
            strat.exempt_eod_flatten = False
            strategies = [strat]

        market_data = AsyncMock(spec=MarketDataProvider)
        market_data.get_current_price = AsyncMock(return_value=Decimal("50000"))
        repo = AsyncMock(spec=TradingRepository)

        engine = TradingEngine(
            exchange=exchange,
            strategies=strategies,
            risk_manager=risk_manager,
            market_data=market_data,
            repository=repo,
            notifier=None,
            settings=settings,
        )
        return engine

    @pytest.mark.asyncio
    async def test_crypto_symbols_processed_when_market_closed(
        self,
        crypto_settings: Settings,
        crypto_exchange: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """Crypto symbols should still be processed when stock market is closed."""
        crypto_settings.MARKET_HOURS_ONLY = True

        # Create strategies
        stock_strat = AsyncMock(spec=BaseStrategy)
        stock_strat.name = "vwap_scalp"
        stock_strat.active_orders = []
        stock_strat.positions = {}
        stock_strat.on_tick = AsyncMock(return_value=[])
        stock_strat.exempt_eod_flatten = False

        crypto_strat = AsyncMock(spec=BaseStrategy)
        crypto_strat.name = "crypto_swing"
        crypto_strat.active_orders = []
        crypto_strat.positions = {}
        crypto_strat.on_tick = AsyncMock(return_value=[])
        crypto_strat.exempt_eod_flatten = True

        engine = self._make_engine(
            crypto_settings, crypto_exchange, mock_risk_manager,
            strategies=[stock_strat, crypto_strat],
        )

        # Market is CLOSED
        crypto_exchange.is_market_open = AsyncMock(return_value=False)
        crypto_exchange.get_positions = AsyncMock(return_value=[])

        # Run one tick  
        await engine._tick()

        # crypto_swing should have been called for BTC/USD and/or ETH/USD
        assert crypto_strat.on_tick.call_count > 0, "Crypto strategy should be called even when market closed"

    @pytest.mark.asyncio
    async def test_stock_strat_skips_crypto_symbols(
        self,
        crypto_settings: Settings,
        crypto_exchange: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """Stock strategies should NOT process crypto symbols."""
        crypto_settings.MARKET_HOURS_ONLY = False

        stock_strat = AsyncMock(spec=BaseStrategy)
        stock_strat.name = "vwap_scalp"
        stock_strat.active_orders = []
        stock_strat.positions = {}
        stock_strat.on_tick = AsyncMock(return_value=[])
        stock_strat.exempt_eod_flatten = False

        crypto_strat = AsyncMock(spec=BaseStrategy)
        crypto_strat.name = "crypto_swing"
        crypto_strat.active_orders = []
        crypto_strat.positions = {}
        crypto_strat.on_tick = AsyncMock(return_value=[])
        crypto_strat.exempt_eod_flatten = True

        engine = self._make_engine(
            crypto_settings, crypto_exchange, mock_risk_manager,
            strategies=[stock_strat, crypto_strat],
        )

        await engine._tick()

        # Check stock_strat was called only for AAPL (not BTC/USD, ETH/USD)
        stock_calls = [call.args[0] for call in stock_strat.on_tick.call_args_list]
        assert "AAPL" in stock_calls
        assert "BTC/USD" not in stock_calls
        assert "ETH/USD" not in stock_calls

        # Check crypto_strat was called only for BTC/USD and ETH/USD (not AAPL)
        crypto_calls = [call.args[0] for call in crypto_strat.on_tick.call_args_list]
        assert "BTC/USD" in crypto_calls
        assert "ETH/USD" in crypto_calls
        assert "AAPL" not in crypto_calls


# ── Bot Registration ─────────────────────────────────────────────────────

class TestBotCryptoRegistration:
    """Test that crypto_swing is properly registered in the bot."""

    def test_crypto_swing_in_strategy_map(
        self,
        crypto_settings: Settings,
        crypto_exchange: AsyncMock,
        mock_risk_manager: RiskManager,
    ) -> None:
        """CryptoSwingStrategy should be in the bot's strategy map."""
        from src.core.bot import AtoBot
        bot = AtoBot(crypto_settings)
        bot.exchange = crypto_exchange
        bot.risk_manager = mock_risk_manager
        strategies = bot._create_strategies()
        names = [s.name for s in strategies]
        assert "crypto_swing" in names
        crypto_strats = [s for s in strategies if s.name == "crypto_swing"]
        assert len(crypto_strats) == 1
        assert isinstance(crypto_strats[0], CryptoSwingStrategy)


# ── Settings Validation ──────────────────────────────────────────────────

class TestCryptoSettings:
    """Test that crypto_swing is accepted in settings validation."""

    def test_crypto_swing_in_allowed_strategies(self) -> None:
        """crypto_swing should be a valid strategy name."""
        s = Settings(
            EXCHANGE="alpaca",
            ALPACA_API_KEY="test",
            ALPACA_API_SECRET="test",
            ALPACA_PAPER=True,
            STRATEGIES=["crypto_swing"],
            SYMBOLS=["BTC/USD"],
            NOTIFICATIONS_ENABLED=False,
            DATABASE_URL="sqlite+aiosqlite:///data/test.db",
            DRY_RUN=True,
        )
        assert "crypto_swing" in s.STRATEGIES

    def test_combined_strategies_valid(self) -> None:
        """Combining stock + crypto strategies should work."""
        s = Settings(
            EXCHANGE="alpaca",
            ALPACA_API_KEY="test",
            ALPACA_API_SECRET="test",
            ALPACA_PAPER=True,
            STRATEGIES=["vwap_scalp", "momentum", "swing", "crypto_swing"],
            SYMBOLS=["AAPL"],
            NOTIFICATIONS_ENABLED=False,
            DATABASE_URL="sqlite+aiosqlite:///data/test.db",
            DRY_RUN=True,
        )
        assert len(s.STRATEGIES) == 4
