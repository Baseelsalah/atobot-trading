"""Walk-Forward Optimizer — periodic strategy parameter re-optimization.

Splits trade history into rolling train/test windows and finds the
parameter set that maximises risk-adjusted returns on the test window.
Re-runs every WALK_FORWARD_INTERVAL_HOURS (default 168 = weekly).

Optimisable parameters per strategy:
  - VWAP:      bounce_pct, tp_pct, sl_pct
  - Momentum:  RSI thresholds, TP/SL
  - EMA:       fast/slow EMA, TP/SL

Uses grid search within safe bounds to keep complexity manageable.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from loguru import logger

from src.config.settings import Settings


# ── Parameter grids (bounds + step) ──────────────────────────────────────────

PARAM_GRIDS: dict[str, dict[str, tuple[float, float, float]]] = {
    "vwap_scalp": {
        "VWAP_BOUNCE_PERCENT": (0.05, 0.35, 0.05),
        "VWAP_TAKE_PROFIT_PERCENT": (0.20, 0.80, 0.10),
        "VWAP_STOP_LOSS_PERCENT": (0.10, 0.50, 0.05),
    },
    "momentum": {
        "RSI_OVERSOLD": (25.0, 35.0, 5.0),
        "RSI_OVERBOUGHT": (65.0, 80.0, 5.0),
        "MOMENTUM_TAKE_PROFIT_PERCENT": (0.50, 2.00, 0.25),
        "MOMENTUM_STOP_LOSS_PERCENT": (0.30, 1.00, 0.10),
    },
    "ema_pullback": {
        "EMA_FAST_PERIOD": (8.0, 13.0, 1.0),
        "EMA_SLOW_PERIOD": (20.0, 30.0, 5.0),
        "EMA_TAKE_PROFIT_PERCENT": (0.50, 1.50, 0.25),
        "EMA_STOP_LOSS_PERCENT": (0.20, 0.80, 0.10),
    },
}


@dataclass
class OptimizationResult:
    """Outcome of a walk-forward optimization cycle."""

    strategy: str
    best_params: dict[str, float]
    train_sharpe: float
    test_sharpe: float
    train_pf: float
    test_pf: float
    train_trades: int
    test_trades: int
    candidates_tested: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "best_params": self.best_params,
            "train_sharpe": round(self.train_sharpe, 4),
            "test_sharpe": round(self.test_sharpe, 4),
            "train_pf": round(self.train_pf, 4),
            "test_pf": round(self.test_pf, 4),
            "train_trades": self.train_trades,
            "test_trades": self.test_trades,
            "candidates_tested": self.candidates_tested,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TradeRow:
    """Minimal trade record for walk-forward evaluation."""

    timestamp: datetime
    strategy: str
    symbol: str
    pnl: float
    entry_price: float
    exit_price: float
    side: str = "LONG"


class WalkForwardOptimizer:
    """Periodic walk-forward re-optimisation of strategy parameters.

    Call `maybe_run()` from the Guardian's periodic loop.  It will only
    execute if enough time has elapsed since the last run.
    """

    def __init__(self, settings: Settings, env_path: str = ".env") -> None:
        self.settings = settings
        self.env_path = env_path
        self._last_run: float = 0.0
        self._history: list[OptimizationResult] = []
        self._load_history()

    # ── Public API ────────────────────────────────────────────────────────────

    async def maybe_run(
        self, trades: list[TradeRow]
    ) -> list[OptimizationResult]:
        """Run walk-forward if interval has elapsed and enough data."""
        if not getattr(self.settings, "WALK_FORWARD_ENABLED", False):
            return []

        interval_s = getattr(self.settings, "WALK_FORWARD_INTERVAL_HOURS", 168) * 3600
        if time.time() - self._last_run < interval_s:
            return []

        train_days = getattr(self.settings, "WALK_FORWARD_TRAIN_DAYS", 180)
        test_days = getattr(self.settings, "WALK_FORWARD_TEST_DAYS", 30)
        total_days_needed = train_days + test_days

        if not trades:
            return []

        # Sort by timestamp
        trades_sorted = sorted(trades, key=lambda t: t.timestamp)
        span_days = (trades_sorted[-1].timestamp - trades_sorted[0].timestamp).days
        if span_days < total_days_needed:
            logger.debug(
                "Walk-forward: only {} days of data (need {})",
                span_days, total_days_needed,
            )
            return []

        results: list[OptimizationResult] = []
        strategies_in_data = {t.strategy for t in trades_sorted}

        for strategy in strategies_in_data:
            if strategy not in PARAM_GRIDS:
                continue
            strat_trades = [t for t in trades_sorted if t.strategy == strategy]
            if len(strat_trades) < 30:
                continue

            result = self._optimise_strategy(
                strategy, strat_trades, train_days, test_days
            )
            if result:
                results.append(result)

        if results:
            self._apply_best_params(results)
            self._history.extend(results)
            self._persist_history()
            self._last_run = time.time()

        return results

    # ── Core optimisation ─────────────────────────────────────────────────────

    def _optimise_strategy(
        self,
        strategy: str,
        trades: list[TradeRow],
        train_days: int,
        test_days: int,
    ) -> OptimizationResult | None:
        """Grid-search over parameter space using train/test split."""
        # Split: last `test_days` = test, rest = train
        cutoff = trades[-1].timestamp.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        from datetime import timedelta

        test_start = cutoff - timedelta(days=test_days)
        train_start = test_start - timedelta(days=train_days)

        train_trades = [t for t in trades if train_start <= t.timestamp < test_start]
        test_trades = [t for t in trades if t.timestamp >= test_start]

        if len(train_trades) < 20 or len(test_trades) < 5:
            logger.debug(
                "Walk-forward {}: not enough trades (train={}, test={})",
                strategy, len(train_trades), len(test_trades),
            )
            return None

        grid = PARAM_GRIDS[strategy]
        param_names = list(grid.keys())
        param_ranges = []
        for pname in param_names:
            lo, hi, step = grid[pname]
            vals = []
            v = lo
            while v <= hi + 1e-9:
                vals.append(round(v, 4))
                v += step
            param_ranges.append(vals)

        best_score = -999.0
        best_combo: dict[str, float] = {}
        best_train_metrics: dict[str, float] = {}
        candidates = 0

        for combo in itertools.product(*param_ranges):
            params = dict(zip(param_names, combo))
            candidates += 1

            # Evaluate on TRAIN set
            train_metrics = self._evaluate_params(train_trades, params, strategy)
            if train_metrics["trades"] < 10:
                continue

            # Score = Sharpe × sqrt(trades) — penalise overfitting with few trades
            score = train_metrics["sharpe"] * (train_metrics["trades"] ** 0.5)
            if score > best_score:
                best_score = score
                best_combo = params
                best_train_metrics = train_metrics

        if not best_combo:
            return None

        # Evaluate best combo on TEST set (out-of-sample)
        test_metrics = self._evaluate_params(test_trades, best_combo, strategy)

        result = OptimizationResult(
            strategy=strategy,
            best_params=best_combo,
            train_sharpe=best_train_metrics.get("sharpe", 0),
            test_sharpe=test_metrics.get("sharpe", 0),
            train_pf=best_train_metrics.get("profit_factor", 0),
            test_pf=test_metrics.get("profit_factor", 0),
            train_trades=best_train_metrics.get("trades", 0),
            test_trades=test_metrics.get("trades", 0),
            candidates_tested=candidates,
        )

        logger.info(
            "Walk-forward {} | best params={} | "
            "train Sharpe={:.3f} PF={:.2f} | test Sharpe={:.3f} PF={:.2f} | "
            "{} combos tested",
            strategy, best_combo,
            result.train_sharpe, result.train_pf,
            result.test_sharpe, result.test_pf,
            candidates,
        )

        # Safety: only apply if test Sharpe is positive
        if result.test_sharpe <= 0:
            logger.warning(
                "Walk-forward {}: test Sharpe negative ({:.3f}) — NOT applying",
                strategy, result.test_sharpe,
            )
            return result  # Still recorded, but not applied

        return result

    def _evaluate_params(
        self,
        trades: list[TradeRow],
        params: dict[str, float],
        strategy: str,
    ) -> dict[str, float]:
        """Simulate trades with given parameter set and return metrics.

        This is a simplified evaluation: we apply parameter-based filters
        to the real trades (e.g. if TP is tighter, cap the win at that level).
        We don't run a full backtest — just adjust PnL heuristically.
        """
        import math

        # Get relevant TP/SL from params
        tp_key = next((k for k in params if "TAKE_PROFIT" in k), None)
        sl_key = next((k for k in params if "STOP_LOSS" in k), None)
        tp_pct = params.get(tp_key, 0.5) / 100 if tp_key else None
        sl_pct = params.get(sl_key, 0.3) / 100 if sl_key else None

        adjusted_pnls: list[float] = []
        for t in trades:
            pnl = t.pnl
            if tp_pct and t.entry_price > 0:
                max_win = t.entry_price * tp_pct * (1 if t.side == "LONG" else -1)
                if pnl > 0:
                    pnl = min(pnl, abs(max_win))
            if sl_pct and t.entry_price > 0:
                max_loss = t.entry_price * sl_pct
                if pnl < 0:
                    pnl = max(pnl, -max_loss)
            adjusted_pnls.append(pnl)

        if not adjusted_pnls:
            return {"sharpe": 0, "profit_factor": 0, "trades": 0}

        import numpy as np

        arr = np.array(adjusted_pnls)
        mean_pnl = float(np.mean(arr))
        std_pnl = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
        sharpe = (mean_pnl / std_pnl * math.sqrt(252)) if std_pnl > 0 else 0.0

        wins = arr[arr > 0]
        losses = arr[arr <= 0]
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999.0

        return {
            "sharpe": round(sharpe, 4),
            "profit_factor": round(profit_factor, 4),
            "trades": len(adjusted_pnls),
            "mean_pnl": round(mean_pnl, 2),
            "win_rate": round(len(wins) / len(arr), 4) if len(arr) > 0 else 0,
        }

    # ── Apply / Persist ───────────────────────────────────────────────────────

    def _apply_best_params(self, results: list[OptimizationResult]) -> None:
        """Write optimised params to .env and update Settings object."""
        env_path = Path(self.env_path)

        # Read existing .env if present
        existing_lines: list[str] = []
        if env_path.exists():
            existing_lines = env_path.read_text().splitlines()

        changes: dict[str, float] = {}
        for r in results:
            if r.test_sharpe <= 0:
                continue  # Don't apply negative-Sharpe results
            changes.update(r.best_params)

        if not changes:
            return

        # Merge into .env
        new_lines = []
        applied: set[str] = set()
        for line in existing_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in changes:
                    new_lines.append(f"{key}={changes[key]}")
                    applied.add(key)
                    continue
            new_lines.append(line)

        for key, val in changes.items():
            if key not in applied:
                new_lines.append(f"{key}={val}")

        env_path.write_text("\n".join(new_lines) + "\n")

        # Hot-reload into Settings
        for key, val in changes.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, val)

        logger.info(
            "Walk-forward applied {} param changes: {}", len(changes), changes
        )

    def _persist_history(self) -> None:
        """Save optimisation history to JSON."""
        path = Path("data/walk_forward_history.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        recent = self._history[-100:]
        try:
            path.write_text(json.dumps([r.to_dict() for r in recent], indent=2))
        except Exception as e:
            logger.warning("Failed to persist walk-forward history: {}", e)

    def _load_history(self) -> None:
        """Load previous history."""
        path = Path("data/walk_forward_history.json")
        if path.exists():
            try:
                data = json.loads(path.read_text())
                # Just keep the raw dicts — we don't need full objects
                self._history = [
                    OptimizationResult(
                        strategy=d.get("strategy", ""),
                        best_params=d.get("best_params", {}),
                        train_sharpe=d.get("train_sharpe", 0),
                        test_sharpe=d.get("test_sharpe", 0),
                        train_pf=d.get("train_pf", 0),
                        test_pf=d.get("test_pf", 0),
                        train_trades=d.get("train_trades", 0),
                        test_trades=d.get("test_trades", 0),
                        candidates_tested=d.get("candidates_tested", 0),
                    )
                    for d in data
                ]
            except Exception:
                pass

    def get_recent_results(self, n: int = 10) -> list[OptimizationResult]:
        """Return the N most recent optimisation results."""
        return self._history[-n:]
