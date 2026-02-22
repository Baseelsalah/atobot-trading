"""Tests for the technical indicators module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
    volume_sma,
    vwap,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    close = pd.Series(np.cumsum(np.random.randn(n)) + 100)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.5,
            "high": close + abs(np.random.randn(n)),
            "low": close - abs(np.random.randn(n)),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )


class TestSMA:
    """Tests for Simple Moving Average."""

    def test_sma_length(self, sample_df: pd.DataFrame) -> None:
        """SMA result should have the same length as input."""
        result = sma(sample_df, period=20)
        assert len(result) == len(sample_df)

    def test_sma_values(self) -> None:
        """SMA should compute correct rolling mean."""
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
        result = sma(df, period=3)
        assert result.iloc[-1] == pytest.approx(4.0, abs=1e-6)

    def test_sma_insufficient_data(self) -> None:
        """SMA should raise ValueError with insufficient data."""
        df = pd.DataFrame({"close": [1, 2]})
        with pytest.raises(ValueError):
            sma(df, period=5)


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_length(self, sample_df: pd.DataFrame) -> None:
        """EMA result should have the same length as input."""
        result = ema(sample_df, period=20)
        assert len(result) == len(sample_df)

    def test_ema_responds_faster_than_sma(self, sample_df: pd.DataFrame) -> None:
        """EMA should be more responsive to recent prices than SMA."""
        sma_result = sma(sample_df, period=10)
        ema_result = ema(sample_df, period=10)
        # Compare only valid (non-NaN) indices that both series share
        valid_sma = sma_result.dropna()
        valid_ema = ema_result.loc[valid_sma.index]
        # They should be different (EMA weights recent data more)
        assert not np.allclose(valid_sma.values, valid_ema.values)


class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_range(self, sample_df: pd.DataFrame) -> None:
        """RSI should be between 0 and 100."""
        result = rsi(sample_df, period=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_insufficient_data(self) -> None:
        """RSI should raise ValueError with insufficient data."""
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError):
            rsi(df, period=14)


class TestMACD:
    """Tests for MACD indicator."""

    def test_macd_returns_three_series(self, sample_df: pd.DataFrame) -> None:
        """MACD should return macd_line, signal_line, and histogram."""
        macd_line, signal_line, histogram = macd(sample_df)
        assert len(macd_line) == len(sample_df)
        assert len(signal_line) == len(sample_df)
        assert len(histogram) == len(sample_df)

    def test_histogram_equals_diff(self, sample_df: pd.DataFrame) -> None:
        """Histogram should equal MACD line minus signal line."""
        macd_line, signal_line, histogram = macd(sample_df)
        valid_idx = macd_line.dropna().index.intersection(signal_line.dropna().index)
        expected = macd_line.loc[valid_idx] - signal_line.loc[valid_idx]
        np.testing.assert_allclose(
            histogram.loc[valid_idx].values, expected.values, atol=1e-10
        )


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bands_structure(self, sample_df: pd.DataFrame) -> None:
        """Should return upper, middle, lower bands."""
        upper, middle, lower = bollinger_bands(sample_df, period=20)
        assert len(upper) == len(sample_df)
        assert len(middle) == len(sample_df)
        assert len(lower) == len(sample_df)

    def test_upper_above_lower(self, sample_df: pd.DataFrame) -> None:
        """Upper band should always be above lower band."""
        upper, middle, lower = bollinger_bands(sample_df, period=20)
        valid = upper.dropna().index
        assert (upper.loc[valid] >= lower.loc[valid]).all()

    def test_middle_is_sma(self, sample_df: pd.DataFrame) -> None:
        """Middle band should equal SMA."""
        upper, middle, lower = bollinger_bands(sample_df, period=20)
        expected = sma(sample_df, period=20)
        np.testing.assert_allclose(
            middle.dropna().values, expected.dropna().values, atol=1e-10
        )


class TestATR:
    """Tests for Average True Range."""

    def test_atr_positive(self, sample_df: pd.DataFrame) -> None:
        """ATR should always be positive."""
        result = atr(sample_df, period=14)
        valid = result.dropna()
        assert (valid > 0).all()


class TestVolumeSMA:
    """Tests for Volume SMA."""

    def test_volume_sma(self, sample_df: pd.DataFrame) -> None:
        """Volume SMA should produce valid output."""
        result = volume_sma(sample_df, period=20)
        assert len(result) == len(sample_df)
        valid = result.dropna()
        assert (valid > 0).all()


class TestVWAP:
    """Tests for Volume-Weighted Average Price."""

    def test_vwap_length(self, sample_df: pd.DataFrame) -> None:
        """VWAP result should have the same length as input."""
        result = vwap(sample_df)
        assert len(result) == len(sample_df)

    def test_vwap_values(self) -> None:
        """VWAP should compute correct weighted average."""
        df = pd.DataFrame({
            "open": [100.0, 102.0],
            "high": [105.0, 108.0],
            "low": [95.0, 100.0],
            "close": [103.0, 106.0],
            "volume": [1000.0, 2000.0],
        })
        result = vwap(df)
        # Bar 0: typical = (105+95+103)/3 = 101.0, cum_tp_vol = 101000, cum_vol = 1000 → 101.0
        assert result.iloc[0] == pytest.approx(101.0, abs=0.1)
        # Bar 1: typical = (108+100+106)/3 = 104.667
        # cum_tp_vol = 101000 + 104666.67*2 = 101000+209333.33 = 310333.33
        # cum_vol = 3000 → ~103.44
        assert result.iloc[1] == pytest.approx(103.44, abs=0.1)

    def test_vwap_missing_column(self) -> None:
        """VWAP should raise ValueError with missing columns."""
        df = pd.DataFrame({"close": [100.0]})
        with pytest.raises(ValueError):
            vwap(df)
