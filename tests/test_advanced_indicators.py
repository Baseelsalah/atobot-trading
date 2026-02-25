"""Tests for the Advanced Indicators module (src/data/indicators_advanced.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.indicators_advanced import (
    adx,
    supertrend,
    parabolic_sar,
    hull_ma,
    ichimoku,
    stochastic,
    williams_r,
    cci,
    obv,
    mfi,
    cmf,
    vwap_bands,
    keltner_channels,
    donchian_channels,
    squeeze_momentum,
    pivot_points,
    fibonacci_levels,
    heikin_ashi,
    volume_profile,
    relative_volume,
    tape_reading_signals,
    confluence_score,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 100, trend: float = 0.001) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a slight uptrend."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5 + trend)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 100000, n).astype(float)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def df():
    return _make_ohlcv(100)


@pytest.fixture
def df_short():
    return _make_ohlcv(20)


# ═══════════════════════════════════════════════════════════════════════════════
# TREND INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════


class TestADX:
    def test_basic(self, df):
        result = adx(df)
        assert "adx" in result
        assert 0 <= result["adx"] <= 100
        assert isinstance(result["trending"], bool)
        assert isinstance(result["bullish_trend"], bool)
        assert "plus_di" in result
        assert "minus_di" in result

    def test_series_returned(self, df):
        result = adx(df)
        assert len(result["series"]) == len(df)

    def test_too_few_rows(self):
        df = _make_ohlcv(5)
        # Should still work with default period=14 if enough rows
        with pytest.raises(ValueError):
            adx(df, period=14)


class TestSuperTrend:
    def test_basic(self, df):
        result = supertrend(df)
        assert result["direction"] in (1, -1)
        assert isinstance(result["value"], (float, np.floating))
        assert result["changed"] in (True, False)
        assert result["bullish"] in (True, False)

    def test_series_length(self, df):
        result = supertrend(df)
        assert len(result["series"]) == len(df)


class TestParabolicSAR:
    def test_basic(self, df):
        result = parabolic_sar(df)
        assert isinstance(result["value"], (float, np.floating))
        assert result["bullish"] in (True, False)
        assert result["reversed"] in (True, False)

    def test_series(self, df):
        result = parabolic_sar(df)
        assert len(result["series"]) == len(df)


class TestHullMA:
    def test_basic(self, df):
        hma = hull_ma(df, period=9)
        assert isinstance(hma, pd.Series)
        assert len(hma) == len(df)
        # Some values may be NaN at the start
        assert not np.all(np.isnan(hma.values))


class TestIchimoku:
    def test_basic(self, df):
        result = ichimoku(df)
        assert "tenkan_sen" in result
        assert "kijun_sen" in result
        assert "senkou_a" in result
        assert "senkou_b" in result
        assert isinstance(result["price_above_cloud"], bool)
        assert isinstance(result["bullish_cross"], bool)
        assert isinstance(result["cloud_bullish"], bool)


# ═══════════════════════════════════════════════════════════════════════════════
# MOMENTUM / OSCILLATOR INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════


class TestStochastic:
    def test_basic(self, df):
        result = stochastic(df)
        assert 0 <= result["k"] <= 100
        assert 0 <= result["d"] <= 100
        assert isinstance(result["oversold"], bool)
        assert isinstance(result["overbought"], bool)
        assert isinstance(result["bullish_cross"], bool)

    def test_series(self, df):
        result = stochastic(df)
        assert len(result["series_k"]) == len(df)


class TestWilliamsR:
    def test_basic(self, df):
        result = williams_r(df)
        assert -100 <= result["value"] <= 0
        assert isinstance(result["oversold"], bool)
        assert isinstance(result["overbought"], bool)


class TestCCI:
    def test_basic(self, df):
        result = cci(df)
        assert isinstance(result["value"], float)
        assert isinstance(result["overbought"], bool)
        assert isinstance(result["oversold"], bool)


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════


class TestOBV:
    def test_basic(self, df):
        result = obv(df)
        assert isinstance(result["value"], float)
        assert result["trend"] in ("rising", "falling")
        assert isinstance(result["bullish_divergence"], bool)
        assert isinstance(result["bearish_divergence"], bool)


class TestMFI:
    def test_basic(self, df):
        result = mfi(df)
        assert 0 <= result["value"] <= 100
        assert isinstance(result["overbought"], bool)
        assert isinstance(result["oversold"], bool)


class TestCMF:
    def test_basic(self, df):
        result = cmf(df)
        assert -1 <= result["value"] <= 1
        assert isinstance(result["buying_pressure"], bool)
        assert isinstance(result["selling_pressure"], bool)


class TestVWAPBands:
    def test_basic(self, df):
        result = vwap_bands(df)
        assert "vwap" in result
        assert "upper_1.0" in result
        assert "lower_1.0" in result
        assert "upper_2.0" in result
        assert "lower_2.0" in result
        assert result["upper_2.0"] > result["upper_1.0"] > result["vwap"]
        assert result["lower_2.0"] < result["lower_1.0"] < result["vwap"]


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════


class TestKeltnerChannels:
    def test_basic(self, df):
        result = keltner_channels(df)
        assert result["upper"] > result["middle"] > result["lower"]
        assert result["width"] > 0
        assert isinstance(result["price_above_upper"], bool)
        assert isinstance(result["price_below_lower"], bool)


class TestDonchianChannels:
    def test_basic(self, df):
        result = donchian_channels(df)
        assert result["upper"] >= result["middle"] >= result["lower"]
        assert isinstance(result["breakout_high"], bool)
        assert isinstance(result["breakout_low"], bool)


class TestSqueezeMomentum:
    def test_basic(self, df):
        result = squeeze_momentum(df)
        assert isinstance(result["squeeze_on"], bool)
        assert isinstance(result["squeeze_off"], bool)
        assert isinstance(result["momentum"], float)
        assert isinstance(result["firing_long"], bool)
        assert isinstance(result["firing_short"], bool)


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE PATTERN INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPivotPoints:
    def test_basic(self, df):
        result = pivot_points(df)
        assert "pp" in result
        assert "r1" in result and "r2" in result and "r3" in result
        assert "s1" in result and "s2" in result and "s3" in result
        assert result["r3"] > result["r2"] > result["r1"] > result["pp"]
        assert result["s3"] < result["s2"] < result["s1"] < result["pp"]


class TestFibonacciLevels:
    def test_basic(self, df):
        result = fibonacci_levels(df)
        assert result["swing_high"] > result["swing_low"]
        assert "fib_236" in result
        assert "fib_382" in result
        assert "fib_500" in result
        assert "fib_618" in result
        assert "fib_786" in result
        assert "nearest_support" in result
        assert "nearest_resistance" in result


class TestHeikinAshi:
    def test_basic(self, df):
        ha = heikin_ashi(df)
        assert "ha_open" in ha.columns
        assert "ha_close" in ha.columns
        assert "ha_high" in ha.columns
        assert "ha_low" in ha.columns
        assert "trend" in ha.columns
        assert len(ha) == len(df)
        assert set(ha["trend"].unique()).issubset({1, -1})


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER FLOW / TAPE READING
# ═══════════════════════════════════════════════════════════════════════════════


class TestVolumeProfile:
    def test_basic(self, df):
        result = volume_profile(df)
        assert "poc" in result
        assert "vah" in result
        assert "val" in result
        assert result["vah"] >= result["val"]
        assert isinstance(result["above_poc"], bool)
        assert isinstance(result["in_value_area"], bool)


class TestRelativeVolume:
    def test_basic(self, df):
        result = relative_volume(df)
        assert "rvol" in result
        assert result["rvol"] >= 0
        assert isinstance(result["volume_surge"], bool)
        assert isinstance(result["above_average"], bool)
        assert isinstance(result["volume_dry"], bool)


class TestTapeReadingSignals:
    def test_basic(self, df):
        result = tape_reading_signals(df)
        assert isinstance(result["aggressive_buying"], bool)
        assert isinstance(result["aggressive_selling"], bool)
        assert isinstance(result["absorption"], bool)
        assert isinstance(result["hammer"], bool)
        assert isinstance(result["shooting_star"], bool)
        assert 0 <= result["close_position"] <= 1
        assert 0 <= result["body_ratio"] <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfluenceScore:
    def test_basic(self, df):
        result = confluence_score(df)
        assert 0 <= result["score"] <= 100
        assert "signals" in result
        assert isinstance(result["strong"], bool)
        assert isinstance(result["moderate"], bool)
        assert isinstance(result["weak"], bool)

    def test_score_categories_exclusive(self, df):
        result = confluence_score(df)
        # At most one of strong/moderate/weak should be True
        # (actually exactly one)
        assert sum([result["strong"], result["moderate"], result["weak"]]) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    def test_missing_column(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="high"):
            adx(df)

    def test_insufficient_rows(self):
        df = _make_ohlcv(3)
        with pytest.raises(ValueError):
            ichimoku(df)
