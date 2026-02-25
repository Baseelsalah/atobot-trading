"""ML Model — LightGBM-based trade quality predictor for AtoBot.

Trains on labeled features from MLFeatureEngine and provides
real-time win-probability predictions as a gate for entry signals.

Features:
- Automatic training when sufficient labeled samples accumulate
- Walk-forward validation (train/test split by time)
- Model persistence (save/load to disk)
- Feature importance tracking
- Probability calibration
- Graceful fallback to heuristic scorer when model unavailable

Usage:
    model = MLModel()
    model.train(feature_engine)
    prob = model.predict(feature_vector)
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass
class ModelMetrics:
    """Training/validation metrics for the ML model."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: float = 0.0
    log_loss: float = 0.0
    train_samples: int = 0
    test_samples: int = 0
    feature_count: int = 0
    trained_at: str = ""
    walk_forward_window: int = 0


class MLModel:
    """LightGBM binary classifier for trade win/loss prediction.

    Designed to work with MLFeatureEngine's ~40 features.
    Falls back to heuristic scoring when:
    - Not enough labeled data (< MIN_SAMPLES)
    - LightGBM not installed
    - Model hasn't been trained yet
    """

    MIN_SAMPLES = 100          # Minimum labeled samples to train
    RETRAIN_INTERVAL = 500     # Retrain after this many new samples
    MODEL_DIR = "data/models"  # Persistence directory

    def __init__(self, model_path: str | None = None):
        self._model = None           # LightGBM Booster or sklearn-compatible
        self._feature_names: list[str] = []
        self._metrics: ModelMetrics | None = None
        self._sample_count_at_train: int = 0
        self._model_path = model_path or os.path.join(self.MODEL_DIR, "trade_predictor.pkl")
        self._is_available = False

        # Try to load existing model
        self._try_load()

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, feature_engine, test_ratio: float = 0.2) -> ModelMetrics | None:
        """Train the model using labeled samples from MLFeatureEngine.

        Uses time-ordered train/test split (walk-forward style).
        Returns metrics or None if training failed.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM not installed — ML model will use heuristic fallback")
            logger.warning("Install with: pip install lightgbm")
            return None

        X, y = feature_engine.build_training_set()
        if len(X) < self.MIN_SAMPLES:
            logger.info(
                "ML Model: Not enough samples ({}/{}) — skipping training",
                len(X), self.MIN_SAMPLES,
            )
            return None

        # Convert to binary labels (>0 = win)
        y_binary = (y > 0).astype(int)

        # Time-ordered split (don't shuffle — preserve temporal order)
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_binary[:split_idx], y_binary[split_idx:]

        if len(X_test) < 10:
            logger.warning("ML Model: Test set too small ({}) — skipping", len(X_test))
            return None

        self._feature_names = feature_engine.feature_names()

        # LightGBM parameters (tuned for small datasets + binary classification)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 10,
            "verbose": -1,
            "n_jobs": 1,
            "seed": 42,
        }

        train_data = lgb.Dataset(X_train, label=y_train,
                                  feature_name=self._feature_names)
        test_data = lgb.Dataset(X_test, label=y_test,
                                 feature_name=self._feature_names,
                                 reference=train_data)

        # Train with early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]

        try:
            self._model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[test_data],
                callbacks=callbacks,
            )
        except Exception as exc:
            logger.error("ML Model training failed: {}", exc)
            return None

        # Compute metrics
        y_pred_proba = self._model.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, log_loss,
        )

        self._metrics = ModelMetrics(
            accuracy=round(accuracy_score(y_test, y_pred), 4),
            precision=round(precision_score(y_test, y_pred, zero_division=0), 4),
            recall=round(recall_score(y_test, y_pred, zero_division=0), 4),
            f1=round(f1_score(y_test, y_pred, zero_division=0), 4),
            auc_roc=round(roc_auc_score(y_test, y_pred_proba), 4) if len(set(y_test)) > 1 else 0.0,
            log_loss=round(log_loss(y_test, y_pred_proba), 4),
            train_samples=len(X_train),
            test_samples=len(X_test),
            feature_count=X_train.shape[1],
            trained_at=datetime.now(timezone.utc).isoformat(),
            walk_forward_window=split_idx,
        )

        self._sample_count_at_train = len(X)
        self._is_available = True

        # Save model
        self._save()

        # Log feature importance
        importance = self._model.feature_importance(importance_type="gain")
        top_features = sorted(
            zip(self._feature_names, importance),
            key=lambda x: -x[1],
        )[:10]

        logger.info(
            "ML Model trained | acc={:.1%} prec={:.1%} AUC={:.3f} | "
            "train={} test={} | top features: {}",
            self._metrics.accuracy, self._metrics.precision,
            self._metrics.auc_roc,
            self._metrics.train_samples, self._metrics.test_samples,
            [(n, round(v, 1)) for n, v in top_features],
        )

        return self._metrics

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, feature_vector) -> float:
        """Predict win probability for a single FeatureVector.

        Returns probability 0.0-1.0. Falls back to heuristic if model
        is not available.
        """
        if not self._is_available or self._model is None:
            # Fallback to heuristic
            from src.intelligence.ml_features import MLFeatureEngine
            engine = MLFeatureEngine()
            return engine.simple_win_probability(feature_vector)

        try:
            X = feature_vector.as_array.reshape(1, -1)
            proba = self._model.predict(X)[0]
            return float(np.clip(proba, 0.0, 1.0))
        except Exception as exc:
            logger.warning("ML prediction error: {} — using heuristic", exc)
            from src.intelligence.ml_features import MLFeatureEngine
            engine = MLFeatureEngine()
            return engine.simple_win_probability(feature_vector)

    def predict_batch(self, feature_vectors: list) -> list[float]:
        """Predict win probabilities for multiple feature vectors."""
        if not self._is_available or self._model is None:
            from src.intelligence.ml_features import MLFeatureEngine
            engine = MLFeatureEngine()
            return [engine.simple_win_probability(fv) for fv in feature_vectors]

        try:
            X = np.array([fv.as_array for fv in feature_vectors])
            probas = self._model.predict(X)
            return [float(np.clip(p, 0.0, 1.0)) for p in probas]
        except Exception as exc:
            logger.warning("ML batch prediction error: {} — using heuristic", exc)
            from src.intelligence.ml_features import MLFeatureEngine
            engine = MLFeatureEngine()
            return [engine.simple_win_probability(fv) for fv in feature_vectors]

    # ── Auto-retrain check ────────────────────────────────────────────────────

    def should_retrain(self, current_sample_count: int) -> bool:
        """Check if model should be retrained based on new sample count."""
        if not self._is_available:
            return current_sample_count >= self.MIN_SAMPLES

        new_samples = current_sample_count - self._sample_count_at_train
        return new_samples >= self.RETRAIN_INTERVAL

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Save model to disk."""
        try:
            Path(self.MODEL_DIR).mkdir(parents=True, exist_ok=True)
            state = {
                "model": self._model,
                "feature_names": self._feature_names,
                "metrics": self._metrics,
                "sample_count": self._sample_count_at_train,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(self._model_path, "wb") as f:
                pickle.dump(state, f)
            logger.info("ML Model saved to {}", self._model_path)
        except Exception as exc:
            logger.warning("Failed to save ML model: {}", exc)

    def _try_load(self) -> None:
        """Try to load a previously saved model."""
        if not os.path.exists(self._model_path):
            return
        try:
            with open(self._model_path, "rb") as f:
                state = pickle.load(f)
            self._model = state["model"]
            self._feature_names = state["feature_names"]
            self._metrics = state.get("metrics")
            self._sample_count_at_train = state.get("sample_count", 0)
            self._is_available = True
            logger.info(
                "ML Model loaded from {} | {} features | trained on {} samples",
                self._model_path, len(self._feature_names),
                self._sample_count_at_train,
            )
        except Exception as exc:
            logger.warning("Failed to load ML model: {}", exc)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def metrics(self) -> ModelMetrics | None:
        return self._metrics

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""
        if not self._is_available or self._model is None:
            return {}
        try:
            importance = self._model.feature_importance(importance_type="gain")
            return dict(sorted(
                zip(self._feature_names, [round(float(v), 2) for v in importance]),
                key=lambda x: -x[1],
            ))
        except Exception:
            return {}

    def get_stats(self) -> dict:
        """Return model statistics."""
        return {
            "is_available": self._is_available,
            "model_path": self._model_path,
            "feature_count": len(self._feature_names),
            "sample_count_at_train": self._sample_count_at_train,
            "metrics": {
                "accuracy": self._metrics.accuracy if self._metrics else None,
                "auc_roc": self._metrics.auc_roc if self._metrics else None,
                "precision": self._metrics.precision if self._metrics else None,
                "f1": self._metrics.f1 if self._metrics else None,
                "trained_at": self._metrics.trained_at if self._metrics else None,
            },
        }
