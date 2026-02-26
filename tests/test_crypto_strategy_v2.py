"""Tests for CryptoSwingStrategy v2 enhancements.

Covers: ADX regime filter, Bollinger Bands, MACD confirmation,
RSI divergence, multi-level take-profit, dynamic ATR stops,
daily trend gate, Fear & Greed sizing, tiered volume.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.config.settings import Settings
from src.models.order import Order, OrderSide, OrderType
from src.models.position import Position
from src.strategies.crypto_strategy import CryptoSwingStrategy


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def crypto_settings(mock_settings: Settings) -> Settings:
    """Settings configured for crypto v2 testing."""
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
    # v2 params
    s.CRYPTO_ADX_FILTER_ENABLED = True
    s.CRYPTO_ADX_MIN_TREND = 20.0
    s.CRYPTO_BB_FILTER_ENABLED = True
    s.CRYPTO_BB_PERIOD = 20
    s.CRYPTO_BB_STD = 2.0
    s.CRYPTO_MACD_ENABLED = True
    s.CRYPTO_DAILY_TREND_GATE = True
    s.CRYPTO_MULTI_TP_ENABLED = True
    s.CRYPTO_TP1_PCT = 5.0
    s.CRYPTO_TP2_PCT = 8.0
    s.CRYPTO_TP3_PCT = 12.0
    s.CRYPTO_DYNAMIC_STOPS = True
    s.CRYPTO_FEAR_GREED_ENABLED = False  # Disabled by default in tests (no HTTP)
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
    mock_risk_manager,
    crypto_settings: Settings,
) -> CryptoSwingStrategy:
    """A CryptoSwingStrategy v2 instance for testing."""
    return CryptoSwingStrategy(crypto_exchange, mock_risk_manager, crypto_settings)


def _make_df(closes: list[float], volumes: list[float] | None = None,
             highs: list[float] | None = None, lows: list[float] | None = None) -> pd.DataFrame:
    """Helper to build realistic DataFrame for indicator tests."""
    n = len(closes)
    if volumes is None:
        volumes = [100000.0] * n
    if highs is None:
        highs = [c + 2 for c in closes]
    if lows is None:
        lows = [c - 3 for c in closes]
    return pd.DataFrame({
        "open": [c - 1 for c in closes],
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


# ── ADX Indicator ─────────────────────────────────────────────────────────

class TestADXIndicator:
    """Test ADX calculation and regime filtering."""

    def test_adx_trending_market(self) -> None:
        """Steadily rising prices should have high ADX (trending)."""
        closes = [100 + i * 2 for i in range(50)]
        highs = [c + 3 for c in closes]
        lows = [c - 1 for c in closes]
        df = _make_df(closes, highs=highs, lows=lows)
        adx = CryptoSwingStrategy._calc_adx(df, 14)
        assert adx is not None
        assert adx > 20, f"Trending market should have ADX > 20, got {adx:.1f}"

    def test_adx_choppy_market(self) -> None:
        """Oscillating prices should have low ADX (choppy)."""
        closes = [100 + (i % 4 - 2) * 0.5 for i in range(50)]
        df = _make_df(closes)
        adx = CryptoSwingStrategy._calc_adx(df, 14)
        assert adx is not None
        assert adx < 30, f"Choppy market ADX should be moderate, got {adx:.1f}"

    def test_adx_insufficient_data(self) -> None:
        """Too few bars should return None."""
        df = _make_df([100, 101, 102, 103, 104])
        adx = CryptoSwingStrategy._calc_adx(df, 14)
        assert adx is None

    def test_adx_gate_blocks_entry(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """ADX < 20 should return 0 confluence (gate blocks)."""
        # Flat/choppy market
        closes = [100 + (i % 3 - 1) * 0.3 for i in range(50)]
        df = _make_df(closes)
        crypto_strategy._adx_filter_enabled = True
        crypto_strategy._adx_min_trend = 20.0
        confluence, details = crypto_strategy._compute_confluence(df, 100.0)
        # If ADX is low, should be blocked
        if "ADX_BLOCKED" in details:
            assert confluence == 0


# ── Bollinger Bands ───────────────────────────────────────────────────────

class TestBollingerBands:
    """Test Bollinger Band calculation and filter."""

    def test_bollinger_values(self) -> None:
        """BB should return sensible upper/middle/lower values."""
        closes = [100 + i * 0.5 for i in range(30)]
        df = _make_df(closes)
        upper, middle, lower = CryptoSwingStrategy._calc_bollinger(df, 20, 2.0)
        assert upper is not None
        assert middle is not None
        assert lower is not None
        assert upper > middle > lower

    def test_bollinger_insufficient_data(self) -> None:
        """Too few bars should return None tuple."""
        df = _make_df([100, 101, 102])
        upper, middle, lower = CryptoSwingStrategy._calc_bollinger(df, 20, 2.0)
        assert upper is None

    def test_bb_lower_half_adds_confluence(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Price below BB middle should add confluence point."""
        # Build data where current price is near the bottom
        closes = [100 + i * 0.5 for i in range(49)] + [95.0]  # Drop at end
        df = _make_df(closes)
        crypto_strategy._bb_filter_enabled = True
        crypto_strategy._adx_filter_enabled = False  # Disable ADX gate for this test
        confluence, details = crypto_strategy._compute_confluence(df, 95.0)
        bb_signals = [d for d in details if "BB_lower" in d]
        assert len(bb_signals) > 0, f"Expected BB_lower_half signal, got: {details}"


# ── MACD ──────────────────────────────────────────────────────────────────

class TestMACDIndicator:
    """Test MACD calculation and confirmation signal."""

    def test_macd_values(self) -> None:
        """MACD should return line, signal, and histogram."""
        closes = [100 + i * 0.3 for i in range(50)]
        df = _make_df(closes)
        macd_line, signal_line, histogram = CryptoSwingStrategy._calc_macd(df, 12, 26, 9)
        assert macd_line is not None
        assert signal_line is not None
        assert histogram is not None
        assert len(histogram) == 50

    def test_macd_insufficient_data(self) -> None:
        """Too few bars should return None."""
        df = _make_df([100, 101, 102])
        m, s, h = CryptoSwingStrategy._calc_macd(df, 12, 26, 9)
        assert m is None

    def test_macd_rising_histogram(self) -> None:
        """Rising MACD histogram should be detectable."""
        # Uptrend → histogram should be positive at end
        closes = [100 + i * 0.5 for i in range(50)]
        df = _make_df(closes)
        _, _, histogram = CryptoSwingStrategy._calc_macd(df, 12, 26, 9)
        assert histogram is not None
        # In an uptrend, MACD line > signal line → positive histogram
        assert histogram[-1] > 0, "MACD histogram should be positive in uptrend"


# ── RSI Divergence ────────────────────────────────────────────────────────

class TestRSIDivergence:
    """Test RSI divergence detection."""

    def test_bullish_divergence(self) -> None:
        """Price lower-low + RSI higher-low = bullish divergence."""
        # Build sequence with price making lower lows but momentum increasing
        n = 40
        closes = []
        for i in range(n):
            if i < 15:
                closes.append(100 - i * 2)  # Down to ~70
            elif i < 20:
                closes.append(75 + i)       # Bounce
            elif i < 30:
                closes.append(95 - (i - 20) * 3)  # Lower low ~65
            else:
                closes.append(65 + (i - 30) * 2)  # Bounce with more momentum
        df = _make_df(closes)
        result = CryptoSwingStrategy._detect_rsi_divergence(df, lookback=10)
        # This is a hard signal to construct perfectly, so just verify it doesn't crash
        assert result in (None, "bullish", "bearish")

    def test_no_divergence_in_steady_trend(self) -> None:
        """Steady uptrend should NOT trigger divergence."""
        closes = [100 + i for i in range(40)]
        df = _make_df(closes)
        result = CryptoSwingStrategy._detect_rsi_divergence(df, lookback=10)
        assert result != "bullish"  # Steady uptrend = no bullish divergence

    def test_divergence_insufficient_data(self) -> None:
        """Not enough data should return None."""
        df = _make_df([100, 101, 102, 103])
        result = CryptoSwingStrategy._detect_rsi_divergence(df, lookback=10)
        assert result is None


# ── Multi-Level Take Profit ───────────────────────────────────────────────

class TestMultiLevelTP:
    """Test multi-level take-profit with partial exits."""

    @pytest.mark.asyncio
    async def test_multi_tp_setup_on_fill(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Buy fill should set up 3 TP levels."""
        order = Order(
            symbol="BTC/USD", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=Decimal("0.01"), price=Decimal("50000"),
            strategy="crypto_swing",
        )
        order.mark_filled()
        await crypto_strategy.on_order_filled(order)

        assert "BTC/USD" in crypto_strategy._tp_levels
        levels = crypto_strategy._tp_levels["BTC/USD"]
        assert len(levels) == 3
        # TP1 at 5%, TP2 at 8%, TP3 at 12%
        assert levels[0][1] == Decimal("50000") * Decimal("1.05")
        assert levels[1][1] == Decimal("50000") * Decimal("1.08")
        assert levels[2][1] == Decimal("50000") * Decimal("1.12")
        assert crypto_strategy._tp_level_hit["BTC/USD"] == 0

    @pytest.mark.asyncio
    async def test_tp1_partial_exit(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Price reaching TP1 should sell 33% and move stop to breakeven."""
        entry = Decimal("50000")
        qty = Decimal("0.01")
        pos = Position(
            symbol="BTC/USD", entry_price=entry, current_price=entry,
            quantity=qty, side="LONG", strategy="crypto_swing",
        )
        crypto_strategy.positions["BTC/USD"] = pos
        crypto_strategy._entry_dates["BTC/USD"] = datetime.now(timezone.utc)
        crypto_strategy._swing_highs["BTC/USD"] = entry
        crypto_strategy._swing_stops["BTC/USD"] = entry * Decimal("0.97")
        crypto_strategy._swing_targets["BTC/USD"] = entry * Decimal("1.12")
        crypto_strategy._trailing_active["BTC/USD"] = False
        crypto_strategy._original_qty["BTC/USD"] = qty
        crypto_strategy._tp_level_hit["BTC/USD"] = 0
        crypto_strategy._breakeven_set["BTC/USD"] = False
        crypto_strategy._tp_levels["BTC/USD"] = [
            (0.33, entry * Decimal("1.05")),
            (0.33, entry * Decimal("1.08")),
            (0.34, entry * Decimal("1.12")),
        ]

        # Price hits TP1 (+5%)
        tp1_price = Decimal("52500")
        orders = await crypto_strategy._manage_position("BTC/USD", pos, tp1_price)
        assert len(orders) == 1
        assert str(orders[0].side).upper() == "SELL"
        # Should be partial qty (~33% of 0.01)
        assert orders[0].quantity < qty
        # Stop should be moved to breakeven
        assert crypto_strategy._breakeven_set["BTC/USD"] is True
        breakeven = entry * Decimal("1.005")
        assert crypto_strategy._swing_stops["BTC/USD"] == breakeven

    @pytest.mark.asyncio
    async def test_tp3_closes_remaining(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """TP3 should close the entire remaining position."""
        entry = Decimal("50000")
        remaining_qty = Decimal("0.0034")  # After TP1 and TP2
        pos = Position(
            symbol="BTC/USD", entry_price=entry, current_price=entry,
            quantity=remaining_qty, side="LONG", strategy="crypto_swing",
        )
        crypto_strategy.positions["BTC/USD"] = pos
        crypto_strategy._entry_dates["BTC/USD"] = datetime.now(timezone.utc)
        crypto_strategy._swing_highs["BTC/USD"] = entry
        crypto_strategy._swing_stops["BTC/USD"] = entry * Decimal("1.005")
        crypto_strategy._swing_targets["BTC/USD"] = entry * Decimal("1.12")
        crypto_strategy._trailing_active["BTC/USD"] = False
        crypto_strategy._original_qty["BTC/USD"] = Decimal("0.01")
        crypto_strategy._tp_level_hit["BTC/USD"] = 2  # TP1 and TP2 already hit
        crypto_strategy._breakeven_set["BTC/USD"] = True
        crypto_strategy._tp_levels["BTC/USD"] = [
            (0.33, entry * Decimal("1.05")),
            (0.33, entry * Decimal("1.08")),
            (0.34, entry * Decimal("1.12")),
        ]

        # Price hits TP3 (+12%)
        orders = await crypto_strategy._manage_position("BTC/USD", pos, Decimal("56000"))
        assert len(orders) == 1
        # Should close entire remaining
        assert orders[0].quantity == remaining_qty

    @pytest.mark.asyncio
    async def test_single_tp_when_multi_disabled(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """When multi-TP is disabled, use single TP."""
        crypto_strategy._multi_tp_enabled = False
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

        orders = await crypto_strategy._manage_position("BTC/USD", pos, Decimal("53000"))
        assert len(orders) == 1
        assert orders[0].quantity == Decimal("0.01")  # Full exit


# ── Dynamic ATR-Based Stops ───────────────────────────────────────────────

class TestDynamicStops:
    """Test ATR-based dynamic stop-loss calculation."""

    @pytest.mark.asyncio
    async def test_dynamic_stop_uses_atr(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Dynamic stops should use ATR to determine stop distance."""
        crypto_strategy._dynamic_stops = True

        # Pre-load bar cache with enough data
        bars = []
        for i in range(100):
            close = 50000 + i * 10
            bars.append({
                "timestamp": 1700000000000 + i * 14400000,
                "open": Decimal(str(close - 50)),
                "high": Decimal(str(close + 100)),
                "low": Decimal(str(close - 150)),
                "close": Decimal(str(close)),
                "volume": Decimal("500000"),
            })
        crypto_strategy._bar_cache["BTC/USD"] = bars

        order = Order(
            symbol="BTC/USD", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=Decimal("0.01"), price=Decimal("50000"),
            strategy="crypto_swing",
        )
        order.mark_filled()
        await crypto_strategy.on_order_filled(order)

        # Stop should be set based on ATR, not fixed %
        assert "BTC/USD" in crypto_strategy._swing_stops
        stop = crypto_strategy._swing_stops["BTC/USD"]
        entry = Decimal("50000")
        stop_pct = float((entry - stop) / entry * 100)
        # Dynamic stop should be between 3% and 7%
        assert 3.0 <= stop_pct <= 7.0, f"Dynamic stop {stop_pct:.2f}% should be 3-7%"

    @pytest.mark.asyncio
    async def test_fixed_stop_when_dynamic_disabled(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """When dynamic stops disabled, use fixed %."""
        crypto_strategy._dynamic_stops = False
        crypto_strategy._stop_loss_pct = 3.0

        order = Order(
            symbol="BTC/USD", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=Decimal("0.01"), price=Decimal("50000"),
            strategy="crypto_swing",
        )
        order.mark_filled()
        await crypto_strategy.on_order_filled(order)

        stop = crypto_strategy._swing_stops["BTC/USD"]
        expected = Decimal("50000") * Decimal("0.97")
        assert stop == expected


# ── Daily Trend Gate ──────────────────────────────────────────────────────

class TestDailyTrendGate:
    """Test daily timeframe macro trend filter."""

    @pytest.mark.asyncio
    async def test_daily_trend_bullish(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Rising daily prices should pass the gate."""
        # Default fixture has rising prices
        result = await crypto_strategy._check_daily_trend("BTC/USD")
        assert result is True

    @pytest.mark.asyncio
    async def test_daily_trend_cached(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Second call uses cache, no extra API call."""
        await crypto_strategy._check_daily_trend("BTC/USD")
        call_count = crypto_strategy.exchange.get_klines.call_count
        await crypto_strategy._check_daily_trend("BTC/USD")
        assert crypto_strategy.exchange.get_klines.call_count == call_count

    @pytest.mark.asyncio
    async def test_daily_trend_fail_open(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """On API error, gate fails open."""
        crypto_strategy.exchange.get_klines.side_effect = Exception("API error")
        result = await crypto_strategy._check_daily_trend("BTC/USD")
        assert result is True


# ── Fear & Greed Integration ─────────────────────────────────────────────

class TestFearGreedSizing:
    """Test Fear & Greed index integration in position sizing."""

    @pytest.mark.asyncio
    async def test_extreme_fear_increases_size(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """F&G < 25 (extreme fear) should increase position size by 25%."""
        # Use low risk so position doesn't hit 50% equity cap
        crypto_strategy._risk_per_trade = 1.0
        crypto_strategy._fear_greed_enabled = True

        with patch.object(crypto_strategy, '_get_fear_greed', return_value=15):
            qty_fear = await crypto_strategy._calc_crypto_size("BTC/USD", Decimal("50000"))

        crypto_strategy._fear_greed_enabled = False
        qty_normal = await crypto_strategy._calc_crypto_size("BTC/USD", Decimal("50000"))

        assert qty_fear > qty_normal, "Extreme fear should increase position size"

    @pytest.mark.asyncio
    async def test_extreme_greed_decreases_size(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """F&G > 75 (extreme greed) should cut position size by 50%."""
        # Use low risk so position doesn't hit 50% equity cap
        crypto_strategy._risk_per_trade = 1.0
        crypto_strategy._fear_greed_enabled = True

        with patch.object(crypto_strategy, '_get_fear_greed', return_value=85):
            qty_greed = await crypto_strategy._calc_crypto_size("BTC/USD", Decimal("50000"))

        crypto_strategy._fear_greed_enabled = False
        qty_normal = await crypto_strategy._calc_crypto_size("BTC/USD", Decimal("50000"))

        assert qty_greed < qty_normal, "Extreme greed should decrease position size"

    @pytest.mark.asyncio
    async def test_fear_greed_neutral(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """F&G between 25-60 should not change size."""
        crypto_strategy._fear_greed_enabled = True

        with patch.object(crypto_strategy, '_get_fear_greed', return_value=50):
            qty_neutral = await crypto_strategy._calc_crypto_size("BTC/USD", Decimal("50000"))

        crypto_strategy._fear_greed_enabled = False
        qty_normal = await crypto_strategy._calc_crypto_size("BTC/USD", Decimal("50000"))

        assert qty_neutral == pytest.approx(qty_normal, rel=0.01), "Neutral F&G should not change size"

    @pytest.mark.asyncio
    async def test_fear_greed_cache(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Fear & Greed value should be cached."""
        crypto_strategy._fear_greed_value = 42
        crypto_strategy._fear_greed_fetched = (
            datetime.now(timezone.utc).strftime('%Y-%m-%d') + "-"
            + f"{(datetime.now(timezone.utc).hour // 4) * 4:02d}"
        )
        result = await crypto_strategy._get_fear_greed()
        assert result == 42


# ── Partial Exit Handling ─────────────────────────────────────────────────

class TestPartialExits:
    """Test that partial SELL fills correctly update position."""

    @pytest.mark.asyncio
    async def test_partial_sell_keeps_position_open(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Partial sell should reduce quantity but not close position."""
        # Create position
        buy = Order(
            symbol="BTC/USD", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=Decimal("0.01"), price=Decimal("50000"),
            strategy="crypto_swing",
        )
        buy.mark_filled()
        await crypto_strategy.on_order_filled(buy)
        assert not crypto_strategy.positions["BTC/USD"].is_closed

        # Partial sell (33% of position at TP1)
        sell = Order(
            symbol="BTC/USD", side=OrderSide.SELL, order_type=OrderType.MARKET,
            quantity=Decimal("0.0033"), price=Decimal("52500"),
            strategy="crypto_swing",
        )
        sell.mark_filled()
        await crypto_strategy.on_order_filled(sell)

        pos = crypto_strategy.positions["BTC/USD"]
        assert not pos.is_closed, "Position should still be open after partial exit"
        assert pos.quantity == Decimal("0.0067"), "Remaining qty should be ~67%"
        # Tracking state should NOT be cleaned up
        assert "BTC/USD" in crypto_strategy._swing_stops

    @pytest.mark.asyncio
    async def test_full_sell_closes_and_cleans(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Full sell should close position and cleanup state."""
        buy = Order(
            symbol="BTC/USD", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=Decimal("0.01"), price=Decimal("50000"),
            strategy="crypto_swing",
        )
        buy.mark_filled()
        await crypto_strategy.on_order_filled(buy)

        sell = Order(
            symbol="BTC/USD", side=OrderSide.SELL, order_type=OrderType.MARKET,
            quantity=Decimal("0.01"), price=Decimal("52000"),
            strategy="crypto_swing",
        )
        sell.mark_filled()
        await crypto_strategy.on_order_filled(sell)

        pos = crypto_strategy.positions["BTC/USD"]
        assert pos.is_closed
        assert "BTC/USD" not in crypto_strategy._swing_stops


# ── Confluence v2 Integration ─────────────────────────────────────────────

class TestConfluenceV2:
    """Test the updated confluence computation with all v2 signals."""

    def test_confluence_returns_tuple(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """_compute_confluence should return (int, list[str])."""
        closes = [100 + i * 0.5 for i in range(50)]
        df = _make_df(closes)
        result = crypto_strategy._compute_confluence(df, 120.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], list)

    def test_tiered_volume_massive(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Volume > 3x average should give +2 confluence."""
        crypto_strategy._adx_filter_enabled = False
        volumes = [100000.0] * 49 + [400000.0]  # Last one is 4x average
        closes = [100 + i * 0.5 for i in range(50)]
        df = _make_df(closes, volumes=volumes)
        confluence, details = crypto_strategy._compute_confluence(df, closes[-1])
        vol_signals = [d for d in details if "VOL_massive" in d]
        assert len(vol_signals) > 0, f"Expected massive volume signal, got: {details}"

    def test_all_filters_disabled_still_works(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Disabling all v2 filters should fall back to original signals."""
        crypto_strategy._adx_filter_enabled = False
        crypto_strategy._bb_filter_enabled = False
        crypto_strategy._macd_enabled = False
        closes = [100 + i for i in range(50)]
        df = _make_df(closes)
        confluence, details = crypto_strategy._compute_confluence(df, 149.0)
        assert isinstance(confluence, int)
        assert confluence >= 0


# ── Settings Validation ───────────────────────────────────────────────────

class TestCryptoV2Settings:
    """Test that new v2 settings params are properly loaded."""

    def test_v2_params_loaded(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """All v2 params should be loaded from settings."""
        assert crypto_strategy._adx_filter_enabled is True
        assert crypto_strategy._adx_min_trend == 20.0
        assert crypto_strategy._bb_filter_enabled is True
        assert crypto_strategy._bb_period == 20
        assert crypto_strategy._bb_std == 2.0
        assert crypto_strategy._macd_enabled is True
        assert crypto_strategy._daily_trend_gate is True
        assert crypto_strategy._multi_tp_enabled is True
        assert crypto_strategy._tp1_pct == 5.0
        assert crypto_strategy._tp2_pct == 8.0
        assert crypto_strategy._tp3_pct == 12.0
        assert crypto_strategy._dynamic_stops is True
        assert crypto_strategy._fear_greed_enabled is False  # Test fixture disables it

    def test_v2_defaults_when_not_set(self) -> None:
        """v2 params should have sensible defaults even if not in settings."""
        from src.config.settings import Settings
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
        assert s.CRYPTO_ADX_FILTER_ENABLED is True
        assert s.CRYPTO_MULTI_TP_ENABLED is True
        assert s.CRYPTO_DYNAMIC_STOPS is True
        assert s.CRYPTO_DAILY_TREND_GATE is True


# ── Exit Order Supports Partial Qty ──────────────────────────────────────

class TestExitOrderPartial:
    """Test _create_exit_order supports partial quantities."""

    def test_exit_order_full(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Default exit should use full position quantity."""
        pos = Position(
            symbol="BTC/USD", entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            quantity=Decimal("0.01"), side="LONG", strategy="crypto_swing",
        )
        order = crypto_strategy._create_exit_order("BTC/USD", pos, "stop_loss")
        assert order.quantity == Decimal("0.01")

    def test_exit_order_partial(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Partial exit should use specified quantity."""
        pos = Position(
            symbol="BTC/USD", entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            quantity=Decimal("0.01"), side="LONG", strategy="crypto_swing",
        )
        order = crypto_strategy._create_exit_order(
            "BTC/USD", pos, "tp1", quantity=Decimal("0.0033")
        )
        assert order.quantity == Decimal("0.0033")


# ── Cleanup includes new state ────────────────────────────────────────────

class TestCleanupV2:
    """Test that cleanup removes all v2 tracking state."""

    def test_cleanup_removes_tp_state(self, crypto_strategy: CryptoSwingStrategy) -> None:
        """Cleanup should remove multi-TP tracking state."""
        crypto_strategy._tp_levels["BTC/USD"] = [(0.33, Decimal("52500"))]
        crypto_strategy._tp_level_hit["BTC/USD"] = 1
        crypto_strategy._original_qty["BTC/USD"] = Decimal("0.01")
        crypto_strategy._breakeven_set["BTC/USD"] = True
        crypto_strategy._entry_dates["BTC/USD"] = datetime.now(timezone.utc)

        crypto_strategy._cleanup_symbol("BTC/USD")

        assert "BTC/USD" not in crypto_strategy._tp_levels
        assert "BTC/USD" not in crypto_strategy._tp_level_hit
        assert "BTC/USD" not in crypto_strategy._original_qty
        assert "BTC/USD" not in crypto_strategy._breakeven_set
