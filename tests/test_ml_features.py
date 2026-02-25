"""Tests for ML Feature Engine (src/intelligence/ml_features.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.intelligence.ml_features import MLFeatureEngine, FeatureVector


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 60, trend: float = 0.001) -> pd.DataFrame:
    """Generate synthetic OHLCV data (needs >= 50 rows)."""
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
def engine():
    return MLFeatureEngine()


@pytest.fixture
def df():
    return _make_ohlcv(60)


@pytest.fixture
def df_with_time():
    df = _make_ohlcv(60)
    base = 1700000000000
    df["timestamp"] = [base + i * 300000 for i in range(len(df))]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeFeatures:
    def test_basic_features(self, engine, df):
        fv = engine.compute_features(df, symbol="AAPL")
        assert isinstance(fv, FeatureVector)
        assert fv.symbol == "AAPL"
        assert len(fv.features) > 30  # Should have 40+ features
        assert fv.label is None

    def test_price_action_features(self, engine, df):
        fv = engine.compute_features(df)
        f = fv.features
        assert "return_1bar" in f
        assert "return_5bar" in f
        assert "return_10bar" in f
        assert "return_20bar" in f
        assert "gap_pct" in f
        assert "range_vs_avg" in f
        assert "body_ratio" in f
        assert "close_position" in f
        assert 0 <= f["close_position"] <= 1
        assert 0 <= f["body_ratio"] <= 1

    def test_ma_features(self, engine, df):
        fv = engine.compute_features(df)
        f = fv.features
        assert "price_vs_ema9" in f
        assert "price_vs_ema21" in f
        assert "price_vs_sma50" in f
        assert "ema9_vs_ema21" in f
        assert "ema9_slope" in f

    def test_rsi_features(self, engine, df):
        fv = engine.compute_features(df)
        f = fv.features
        assert "rsi_14" in f
        assert 0 <= f["rsi_14"] <= 100
        assert "rsi_rate_of_change" in f
        assert f["rsi_oversold"] in (0.0, 1.0)
        assert f["rsi_overbought"] in (0.0, 1.0)

    def test_macd_features(self, engine, df):
        fv = engine.compute_features(df)
        f = fv.features
        assert "macd_histogram" in f
        assert f["macd_above_signal"] in (0.0, 1.0)
        assert f["macd_above_zero"] in (0.0, 1.0)
        assert "macd_hist_slope" in f

    def test_bollinger_features(self, engine, df):
        fv = engine.compute_features(df)
        f = fv.features
        assert "bb_position" in f
        assert "bb_width" in f
        assert f["bb_squeeze"] in (0.0, 1.0)

    def test_atr_features(self, engine, df):
        fv = engine.compute_features(df)
        f = fv.features
        assert "atr_pct" in f
        assert f["atr_expanding"] in (0.0, 1.0)

    def test_volume_features(self, engine, df):
        fv = engine.compute_features(df)
        f = fv.features
        assert "rvol" in f
        assert f["rvol"] >= 0
        assert "volume_trend" in f
        assert f["obv_rising"] in (0.0, 1.0)

    def test_candle_features(self, engine, df):
        fv = engine.compute_features(df)
        f = fv.features
        assert "upper_wick_ratio" in f
        assert "lower_wick_ratio" in f
        assert f["bullish_candle"] in (0.0, 1.0)
        assert f["three_bar_bullish"] in (0.0, 1.0)
        assert f["three_bar_bearish"] in (0.0, 1.0)

    def test_time_features_no_timestamp(self, engine, df):
        fv = engine.compute_features(df)
        f = fv.features
        assert f["hour_sin"] == 0.0
        assert f["hour_cos"] == 0.0
        assert f["day_of_week"] == 0.0

    def test_time_features_with_timestamp(self, engine, df_with_time):
        fv = engine.compute_features(df_with_time)
        f = fv.features
        # Should compute sin/cos values
        assert "hour_sin" in f
        assert "hour_cos" in f
        assert "day_of_week" in f

    def test_regime_features_default(self, engine, df):
        fv = engine.compute_features(df)
        f = fv.features
        assert f["regime_trend_score"] == 0.0
        assert f["regime_vol_score"] == 0.0
        assert f["regime_size_mult"] == 1.0

    def test_regime_features_provided(self, engine, df):
        regime = {"trend_score": 75, "vol_score": 30, "size_multiplier": 0.8}
        fv = engine.compute_features(df, regime=regime)
        f = fv.features
        assert f["regime_trend_score"] == 75
        assert f["regime_vol_score"] == 30
        assert f["regime_size_mult"] == 0.8

    def test_insufficient_rows(self, engine):
        df = _make_ohlcv(30)
        with pytest.raises(ValueError, match="at least 50"):
            engine.compute_features(df)

    def test_feature_vector_as_array(self, engine, df):
        fv = engine.compute_features(df)
        arr = fv.as_array
        assert isinstance(arr, np.ndarray)
        assert len(arr) == len(fv.features)

    def test_feature_names(self, engine, df):
        fv = engine.compute_features(df)
        names = fv.feature_names
        assert isinstance(names, list)
        assert len(names) == len(fv.features)


# ═══════════════════════════════════════════════════════════════════════════════
# WIN PROBABILITY
# ═══════════════════════════════════════════════════════════════════════════════


class TestWinProbability:
    def test_returns_probability(self, engine, df):
        fv = engine.compute_features(df)
        prob = engine.simple_win_probability(fv)
        assert 0 <= prob <= 1

    def test_favorable_features_higher_prob(self, engine):
        # Construct a feature vector with strong bullish signals
        bullish_fv = FeatureVector(
            symbol="AAPL", timestamp="2026-01-01",
            features={
                "rsi_14": 45.0,
                "macd_above_signal": 1.0,
                "macd_above_zero": 1.0,
                "macd_hist_slope": 0.5,
                "price_vs_ema9": 0.5,
                "ema9_vs_ema21": 0.3,
                "rvol": 2.5,
                "bb_position": 0.2,
                "obv_rising": 1.0,
                "atr_expanding": 1.0,
                "bullish_candle": 1.0,
                "three_bar_bullish": 1.0,
                "three_bar_bearish": 0.0,
                "regime_size_mult": 1.2,
                "bb_squeeze": 1.0,
            },
        )
        bearish_fv = FeatureVector(
            symbol="AAPL", timestamp="2026-01-01",
            features={
                "rsi_14": 80.0,
                "macd_above_signal": 0.0,
                "macd_above_zero": 0.0,
                "macd_hist_slope": -0.5,
                "price_vs_ema9": -1.0,
                "ema9_vs_ema21": -0.5,
                "rvol": 0.3,
                "bb_position": 0.95,
                "obv_rising": 0.0,
                "atr_expanding": 0.0,
                "bullish_candle": 0.0,
                "three_bar_bullish": 0.0,
                "three_bar_bearish": 1.0,
                "regime_size_mult": 0.3,
                "bb_squeeze": 0.0,
            },
        )
        bull_prob = engine.simple_win_probability(bullish_fv)
        bear_prob = engine.simple_win_probability(bearish_fv)
        assert bull_prob > bear_prob


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING SET
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainingSet:
    def test_empty_training_set(self, engine):
        X, y = engine.build_training_set()
        assert len(X) == 0
        assert len(y) == 0

    def test_build_training_set(self, engine, df):
        for i in range(10):
            fv = engine.compute_features(df, symbol="AAPL")
            engine.add_labeled_sample(fv, 1.0 if i % 2 == 0 else 0.0)

        X, y = engine.build_training_set()
        assert X.shape == (10, len(fv.features))
        assert y.shape == (10,)
        assert sum(y) == 5  # Half wins

    def test_unlabeled_excluded(self, engine, df):
        fv = engine.compute_features(df)
        engine._feature_history.append(fv)  # No label
        X, y = engine.build_training_set()
        assert len(X) == 0

    def test_history_cap(self, engine, df):
        for i in range(5100):
            fv = FeatureVector(symbol="X", timestamp="", features={"a": float(i)})
            engine.add_labeled_sample(fv, 1.0)
        assert len(engine._feature_history) == 5000


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureImportance:
    def test_insufficient_data(self, engine):
        result = engine.compute_feature_importance()
        assert result == {}

    def test_feature_importance(self, engine, df):
        for i in range(40):
            fv = engine.compute_features(df, symbol="AAPL")
            # Label based on a feature to create correlation
            label = 1.0 if fv.features.get("rsi_14", 50) > 50 else 0.0
            engine.add_labeled_sample(fv, label)

        importance = engine.compute_feature_importance()
        assert len(importance) > 0
        # All importance values should be 0-1
        for name, val in importance.items():
            assert 0 <= val <= 1

    def test_get_top_features(self, engine, df):
        for i in range(40):
            fv = engine.compute_features(df)
            engine.add_labeled_sample(fv, float(i % 2))

        top = engine.get_top_features(5)
        assert len(top) <= 5
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top)

    def test_feature_names_method(self, engine, df):
        assert engine.feature_names() == []
        fv = engine.compute_features(df)
        engine._feature_history.append(fv)
        names = engine.feature_names()
        assert len(names) > 30


# ═══════════════════════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════════════════════


class TestStats:
    def test_empty_stats(self, engine):
        stats = engine.get_stats()
        assert stats["total_samples"] == 0
        assert stats["labeled_samples"] == 0
        assert stats["feature_count"] == 0

    def test_stats_with_data(self, engine, df):
        fv = engine.compute_features(df)
        engine.add_labeled_sample(fv, 1.0)
        fv2 = engine.compute_features(df)
        engine.add_labeled_sample(fv2, 0.0)

        stats = engine.get_stats()
        assert stats["total_samples"] == 2
        assert stats["labeled_samples"] == 2
        assert stats["win_labels"] == 1
        assert stats["loss_labels"] == 1
        assert stats["feature_count"] > 30
