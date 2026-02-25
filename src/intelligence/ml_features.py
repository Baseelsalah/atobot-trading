"""ML Feature Engine — feature engineering pipeline for trade prediction.

Inspired by FinRL's gym environments and ML4Trading's alpha factor research.
Builds structured feature vectors from market data for:
- Entry quality prediction
- Win probability estimation
- Feature importance tracking
- Walk-forward validation

Works without heavy ML dependencies (numpy/pandas only) so the bot stays
light.  Provides scikit-learn compatible feature matrices if sklearn
is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger


@dataclass
class FeatureVector:
    """A single feature vector for one data point."""
    symbol: str
    timestamp: str
    features: dict[str, float]
    label: float | None = None  # 1.0=win, 0.0=loss, None=unknown

    @property
    def as_array(self) -> np.ndarray:
        return np.array(list(self.features.values()))

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.keys())


class MLFeatureEngine:
    """Builds feature vectors from OHLCV data for ML prediction.

    Features are organized into categories:
    1. Price action (returns, gaps, ranges)
    2. Technical indicators (RSI, MACD, BB, etc.)
    3. Volume profile (RVOL, OBV slope, etc.)
    4. Microstructure (spread, candle patterns)
    5. Time features (time of day, day of week)
    6. Regime context (trend, volatility)

    Usage:
        engine = MLFeatureEngine()
        features = engine.compute_features(df, symbol="AAPL")
        X, y = engine.build_training_set()
    """

    def __init__(self):
        self._feature_history: list[FeatureVector] = []
        self._feature_importance: dict[str, float] = {}

    # ── Compute Features ──────────────────────────────────────────────────────

    def compute_features(self, df: pd.DataFrame, symbol: str = "",
                         regime: dict | None = None) -> FeatureVector:
        """Compute full feature vector from OHLCV DataFrame.

        Requires at least 50 rows (for MA/indicator calculations).
        """
        if len(df) < 50:
            raise ValueError(f"Need at least 50 bars, got {len(df)}")

        features = {}

        # ── Price Action Features ─────────────────────────────────────────
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)

        # Returns
        features["return_1bar"] = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)
        features["return_5bar"] = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(df) > 6 else 0
        features["return_10bar"] = float((close.iloc[-1] / close.iloc[-11] - 1) * 100) if len(df) > 11 else 0
        features["return_20bar"] = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(df) > 21 else 0

        # Gap
        features["gap_pct"] = float((open_.iloc[-1] / close.iloc[-2] - 1) * 100)

        # Range
        bar_range = (high.iloc[-1] - low.iloc[-1])
        avg_range = (high - low).tail(20).mean()
        features["range_vs_avg"] = float(bar_range / avg_range) if avg_range > 0 else 1.0

        # Body ratio
        body = abs(close.iloc[-1] - open_.iloc[-1])
        features["body_ratio"] = float(body / bar_range) if bar_range > 0 else 0

        # Close position in range
        features["close_position"] = float(
            (close.iloc[-1] - low.iloc[-1]) / bar_range
        ) if bar_range > 0 else 0.5

        # ── Moving Average Features ───────────────────────────────────────
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        sma_50 = close.rolling(50).mean()

        features["price_vs_ema9"] = float((close.iloc[-1] / ema_9.iloc[-1] - 1) * 100)
        features["price_vs_ema21"] = float((close.iloc[-1] / ema_21.iloc[-1] - 1) * 100)
        features["price_vs_sma50"] = float((close.iloc[-1] / sma_50.iloc[-1] - 1) * 100) if not pd.isna(sma_50.iloc[-1]) else 0
        features["ema9_vs_ema21"] = float((ema_9.iloc[-1] / ema_21.iloc[-1] - 1) * 100)

        # EMA slope (momentum proxy)
        features["ema9_slope"] = float((ema_9.iloc[-1] / ema_9.iloc[-3] - 1) * 100) if len(df) > 3 else 0

        # ── RSI Features ──────────────────────────────────────────────────
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        features["rsi_14"] = float(rsi.iloc[-1])
        features["rsi_rate_of_change"] = float(rsi.iloc[-1] - rsi.iloc[-3]) if len(rsi) > 3 else 0
        features["rsi_oversold"] = 1.0 if float(rsi.iloc[-1]) < 30 else 0.0
        features["rsi_overbought"] = 1.0 if float(rsi.iloc[-1]) > 70 else 0.0

        # ── MACD Features ─────────────────────────────────────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        features["macd_histogram"] = float(histogram.iloc[-1])
        features["macd_above_signal"] = 1.0 if float(macd_line.iloc[-1]) > float(signal_line.iloc[-1]) else 0.0
        features["macd_above_zero"] = 1.0 if float(macd_line.iloc[-1]) > 0 else 0.0
        features["macd_hist_slope"] = float(histogram.iloc[-1] - histogram.iloc[-3]) if len(histogram) > 3 else 0

        # ── Bollinger Band Features ───────────────────────────────────────
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / bb_mid

        features["bb_position"] = float(
            (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        ) if float(bb_upper.iloc[-1] - bb_lower.iloc[-1]) > 0 else 0.5
        features["bb_width"] = float(bb_width.iloc[-1]) if not pd.isna(bb_width.iloc[-1]) else 0
        features["bb_squeeze"] = 1.0 if float(bb_width.iloc[-1]) < float(bb_width.tail(50).quantile(0.1)) else 0.0

        # ── ATR Features ──────────────────────────────────────────────────
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        features["atr_pct"] = float(atr_14.iloc[-1] / close.iloc[-1] * 100) if not pd.isna(atr_14.iloc[-1]) else 0
        features["atr_expanding"] = 1.0 if float(atr_14.iloc[-1]) > float(atr_14.iloc[-5]) else 0.0

        # ── Volume Features ───────────────────────────────────────────────
        vol_sma = volume.rolling(20).mean()
        features["rvol"] = float(volume.iloc[-1] / vol_sma.iloc[-1]) if float(vol_sma.iloc[-1]) > 0 else 1.0
        features["volume_trend"] = float(vol_sma.iloc[-1] / vol_sma.iloc[-10] - 1) if len(df) > 10 else 0

        # OBV direction
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_sma = obv.rolling(10).mean()
        features["obv_rising"] = 1.0 if float(obv.iloc[-1]) > float(obv_sma.iloc[-1]) else 0.0

        # ── Candle Pattern Features ───────────────────────────────────────
        upper_wick = high.iloc[-1] - max(close.iloc[-1], open_.iloc[-1])
        lower_wick = min(close.iloc[-1], open_.iloc[-1]) - low.iloc[-1]
        features["upper_wick_ratio"] = float(upper_wick / bar_range) if bar_range > 0 else 0
        features["lower_wick_ratio"] = float(lower_wick / bar_range) if bar_range > 0 else 0
        features["bullish_candle"] = 1.0 if close.iloc[-1] > open_.iloc[-1] else 0.0

        # 3-bar pattern
        features["three_bar_bullish"] = 1.0 if (
            close.iloc[-1] > close.iloc[-2] > close.iloc[-3]
        ) else 0.0
        features["three_bar_bearish"] = 1.0 if (
            close.iloc[-1] < close.iloc[-2] < close.iloc[-3]
        ) else 0.0

        # ── Time Features ─────────────────────────────────────────────────
        if "timestamp" in df.columns:
            try:
                ts = pd.to_datetime(df["timestamp"].iloc[-1], unit="ms")
                features["hour_sin"] = float(np.sin(2 * np.pi * ts.hour / 24))
                features["hour_cos"] = float(np.cos(2 * np.pi * ts.hour / 24))
                features["day_of_week"] = float(ts.dayofweek)
            except Exception:
                features["hour_sin"] = 0.0
                features["hour_cos"] = 0.0
                features["day_of_week"] = 0.0
        else:
            features["hour_sin"] = 0.0
            features["hour_cos"] = 0.0
            features["day_of_week"] = 0.0

        # ── Regime Features ────────────────────────────────────────────
        if regime:
            features["regime_trend_score"] = float(regime.get("trend_score", 0))
            features["regime_vol_score"] = float(regime.get("vol_score", 0))
            features["regime_size_mult"] = float(regime.get("size_multiplier", 1.0))
        else:
            features["regime_trend_score"] = 0.0
            features["regime_vol_score"] = 0.0
            features["regime_size_mult"] = 1.0

        fv = FeatureVector(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            features=features,
        )
        return fv

    # ── Training Set ──────────────────────────────────────────────────────────

    def add_labeled_sample(self, fv: FeatureVector, label: float) -> None:
        """Add a labeled sample (1.0=win, 0.0=loss, float=pnl)."""
        fv.label = label
        self._feature_history.append(fv)
        # Keep last 5000
        if len(self._feature_history) > 5000:
            self._feature_history = self._feature_history[-5000:]

    def build_training_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Build X, y arrays from labeled history.

        X: (n_samples, n_features)
        y: (n_samples,)
        """
        labeled = [fv for fv in self._feature_history if fv.label is not None]
        if not labeled:
            return np.array([]), np.array([])

        X = np.array([fv.as_array for fv in labeled])
        y = np.array([fv.label for fv in labeled])
        return X, y

    def feature_names(self) -> list[str]:
        """Return feature names from the latest vector."""
        if self._feature_history:
            return self._feature_history[-1].feature_names
        return []

    # ── Simple Prediction (no sklearn needed) ─────────────────────────────────

    def simple_win_probability(self, fv: FeatureVector) -> float:
        """Estimate win probability using simple heuristic scoring.

        This is a lightweight alternative to ML — scores features
        based on domain knowledge and returns a probability estimate.
        No external ML library required.
        """
        score = 0.0
        f = fv.features
        total_weight = 0

        # RSI in favorable zone
        rsi = f.get("rsi_14", 50)
        if 30 < rsi < 60:
            score += 15  # Sweet spot for entries
        elif rsi < 25:
            score += 10  # Oversold bounce
        elif rsi > 75:
            score -= 10  # Overbought risk
        total_weight += 15

        # MACD momentum
        if f.get("macd_above_signal", 0):
            score += 12
        if f.get("macd_above_zero", 0):
            score += 5
        if f.get("macd_hist_slope", 0) > 0:
            score += 8
        total_weight += 25

        # Trend alignment
        if f.get("price_vs_ema9", 0) > 0 and f.get("ema9_vs_ema21", 0) > 0:
            score += 15  # Price > EMA9 > EMA21
        total_weight += 15

        # Volume confirmation
        rvol = f.get("rvol", 1.0)
        if rvol >= 2.0:
            score += 12  # Strong volume
        elif rvol >= 1.0:
            score += 5   # Above average
        elif rvol < 0.5:
            score -= 5   # Dry volume
        total_weight += 12

        # Bollinger position (mean-reversion opportunity)
        bb_pos = f.get("bb_position", 0.5)
        if 0.1 < bb_pos < 0.3:
            score += 8  # Near lower band
        elif bb_pos > 0.9:
            score -= 5  # Near upper band
        total_weight += 8

        # OBV confirmation
        if f.get("obv_rising", 0):
            score += 5
        total_weight += 5

        # ATR expanding (volatility = opportunity)
        if f.get("atr_expanding", 0):
            score += 5
        total_weight += 5

        # Candle pattern
        if f.get("bullish_candle", 0):
            score += 3
        if f.get("three_bar_bullish", 0):
            score += 5
        if f.get("three_bar_bearish", 0):
            score -= 5
        total_weight += 8

        # Regime
        if f.get("regime_size_mult", 1.0) >= 1.0:
            score += 5
        elif f.get("regime_size_mult", 1.0) < 0.5:
            score -= 10
        total_weight += 10

        # Squeeze (opportunity setup)
        if f.get("bb_squeeze", 0):
            score += 7
        total_weight += 7

        # Normalize to 0-1 probability
        prob = max(0.0, min(1.0, (score / total_weight + 1) / 2))
        return round(prob, 4)

    # ── Feature Importance ────────────────────────────────────────────────────

    def compute_feature_importance(self) -> dict[str, float]:
        """Compute simple feature importance via correlation with labels.

        For each feature, compute the absolute correlation with the label.
        Higher correlation = more important feature.
        """
        X, y = self.build_training_set()
        if len(X) < 30:
            return {}

        names = self.feature_names()
        importance = {}

        for i, name in enumerate(names):
            col = X[:, i]
            # Avoid constant columns
            if np.std(col) < 1e-10:
                importance[name] = 0.0
                continue
            # Pearson correlation
            corr = np.corrcoef(col, y)[0, 1]
            importance[name] = round(abs(float(corr)), 4) if not np.isnan(corr) else 0.0

        # Sort by importance
        self._feature_importance = dict(
            sorted(importance.items(), key=lambda x: -x[1])
        )
        return self._feature_importance

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Return top-N most important features."""
        if not self._feature_importance:
            self.compute_feature_importance()
        return list(self._feature_importance.items())[:n]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return engine statistics."""
        labeled = [fv for fv in self._feature_history if fv.label is not None]
        return {
            "total_samples": len(self._feature_history),
            "labeled_samples": len(labeled),
            "feature_count": len(self.feature_names()) if self._feature_history else 0,
            "win_labels": sum(1 for fv in labeled if fv.label and fv.label > 0),
            "loss_labels": sum(1 for fv in labeled if fv.label is not None and fv.label <= 0),
        }
