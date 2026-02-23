"""Auto-Tuner — adjusts strategy parameters based on live performance.

Philosophy:
- Only tune within SAFE BOUNDS (never exceed backtested ranges).
- Small incremental changes (max 10-20% per adjustment).
- Minimum sample size before acting (≥20 trades per strategy).
- All changes logged with before/after + reason.
- Automatic rollback if performance worsens after a change.

Tunable parameters per strategy:
- VWAP: bounce %, take-profit %, stop-loss %, order size
- ORB: range minutes, breakout %, TP %, SL %, order size
- Midday filter hours
- Position sizing (order size USD)

Uses a simple "nudge toward what works" approach:
- If win rate > 50% and profit factor > 1.2 → widen TP slightly (capture more)
- If win rate < 40% → tighten SL (cut losers faster)
- If a strategy is deeply negative → reduce its order size
- If a strategy is highly profitable → increase its order size (slowly)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from loguru import logger

from src.config.settings import Settings
from src.guardian.performance_analyzer import PerformanceReport, StrategyMetrics


# ── Safe Bounds ───────────────────────────────────────────────────────────────
# These are absolute limits — the tuner will never go outside these.

BOUNDS = {
    "VWAP_BOUNCE_PERCENT": (0.05, 0.40),
    "VWAP_TAKE_PROFIT_PERCENT": (0.20, 1.00),
    "VWAP_STOP_LOSS_PERCENT": (0.10, 0.60),
    "VWAP_ORDER_SIZE_USD": (200.0, 1000.0),
    "ORB_RANGE_MINUTES": (10, 30),
    "ORB_BREAKOUT_PERCENT": (0.05, 0.30),
    "ORB_TAKE_PROFIT_PERCENT": (0.75, 3.00),
    "ORB_STOP_LOSS_PERCENT": (0.30, 1.50),
    "ORB_ORDER_SIZE_USD": (200.0, 1000.0),
    "MIDDAY_START_HOUR": (11, 13),
    "MIDDAY_END_HOUR": (13, 15),
}

# Minimum trades before the tuner will adjust a strategy
MIN_TRADES_FOR_TUNING = 20

# Max adjustment per cycle (as fraction of current value)
MAX_ADJUSTMENT_PCT = 0.15  # 15%


@dataclass
class TuneAction:
    """Record of a parameter adjustment."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    parameter: str = ""
    old_value: float = 0.0
    new_value: float = 0.0
    reason: str = ""
    strategy: str = ""
    metrics_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "parameter": self.parameter,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "strategy": self.strategy,
            "metrics": self.metrics_snapshot,
        }

    def __str__(self) -> str:
        return (
            f"[TUNE] {self.parameter}: {self.old_value} → {self.new_value} "
            f"({self.reason})"
        )


class AutoTuner:
    """Adjusts strategy parameters based on performance."""

    TUNE_COOLDOWN_S = 86400  # Only tune once per 24 hours

    def __init__(self, settings: Settings, env_path: str = ".env") -> None:
        self.settings = settings
        self.env_path = env_path
        self._history: list[TuneAction] = []
        self._last_tune_time: float = 0.0
        self._pending_rollbacks: list[TuneAction] = []
        self._load_history()

    async def evaluate_and_tune(
        self, report: PerformanceReport
    ) -> list[TuneAction]:
        """Evaluate performance and make adjustments if warranted."""
        import time

        actions: list[TuneAction] = []

        # Cooldown check
        if time.time() - self._last_tune_time < self.TUNE_COOLDOWN_S:
            return actions

        # Check if previous tune needs rollback
        rollback = self._check_rollback(report)
        if rollback:
            actions.extend(rollback)

        # Only tune if we have enough data
        if report.total_trades < MIN_TRADES_FOR_TUNING:
            logger.debug(
                "Skipping tune: only {} trades (need {})",
                report.total_trades, MIN_TRADES_FOR_TUNING,
            )
            return actions

        # Tune each strategy
        for name, metrics in report.strategy_metrics.items():
            if metrics.total_trades < MIN_TRADES_FOR_TUNING:
                continue

            strat_actions = self._tune_strategy(name, metrics, report)
            actions.extend(strat_actions)

        # Apply changes to .env
        if actions:
            self._apply_to_env(actions)
            self._last_tune_time = time.time()

            # Store for potential rollback
            self._pending_rollbacks = actions.copy()

        # Log and persist
        for a in actions:
            self._history.append(a)
            logger.info("{}", a)

        self._persist_history()
        return actions

    def _tune_strategy(
        self, name: str, metrics: StrategyMetrics, report: PerformanceReport
    ) -> list[TuneAction]:
        """Generate tune actions for a specific strategy."""
        actions = []

        if name == "vwap_scalp":
            actions.extend(self._tune_vwap(metrics))
        elif name == "orb":
            actions.extend(self._tune_orb(metrics))

        return actions

    def _tune_vwap(self, m: StrategyMetrics) -> list[TuneAction]:
        """Tune VWAP Scalp parameters."""
        actions = []
        snapshot = {
            "trades": m.total_trades, "wr": m.win_rate,
            "pf": m.profit_factor, "pnl": str(m.total_pnl),
        }

        # High win rate + good PF → widen take-profit to capture more
        if m.win_rate > 0.50 and m.profit_factor > 1.2:
            action = self._nudge_up(
                "VWAP_TAKE_PROFIT_PERCENT",
                reason=f"VWAP strong: WR={m.win_rate:.1%} PF={m.profit_factor:.2f} → widening TP",
                strategy="vwap_scalp",
                snapshot=snapshot,
            )
            if action:
                actions.append(action)

        # Low win rate → tighten stop-loss
        if m.win_rate < 0.40 and m.total_trades >= 30:
            action = self._nudge_down(
                "VWAP_STOP_LOSS_PERCENT",
                reason=f"VWAP low WR={m.win_rate:.1%} → tightening SL",
                strategy="vwap_scalp",
                snapshot=snapshot,
            )
            if action:
                actions.append(action)

        # Average loss > average win → tighten SL or widen TP
        if m.avg_loss != 0 and abs(m.avg_loss) > m.avg_win * Decimal("1.5"):
            action = self._nudge_down(
                "VWAP_STOP_LOSS_PERCENT",
                reason=f"VWAP avg loss (${m.avg_loss:.2f}) >> avg win (${m.avg_win:.2f})",
                strategy="vwap_scalp",
                snapshot=snapshot,
            )
            if action:
                actions.append(action)

        # Strategy deeply negative → reduce sizing
        if m.total_pnl < Decimal("-50") and m.total_trades >= 20:
            action = self._nudge_down(
                "VWAP_ORDER_SIZE_USD",
                reason=f"VWAP negative PnL ${m.total_pnl:.2f} → reducing size",
                strategy="vwap_scalp",
                snapshot=snapshot,
            )
            if action:
                actions.append(action)

        # Strategy very profitable → increase sizing slowly
        if m.total_pnl > Decimal("100") and m.profit_factor > 1.5:
            action = self._nudge_up(
                "VWAP_ORDER_SIZE_USD",
                reason=f"VWAP profitable ${m.total_pnl:.2f} PF={m.profit_factor:.2f} → sizing up",
                strategy="vwap_scalp",
                snapshot=snapshot,
            )
            if action:
                actions.append(action)

        return actions

    def _tune_orb(self, m: StrategyMetrics) -> list[TuneAction]:
        """Tune ORB parameters."""
        actions = []
        snapshot = {
            "trades": m.total_trades, "wr": m.win_rate,
            "pf": m.profit_factor, "pnl": str(m.total_pnl),
        }

        # High win rate → widen TP
        if m.win_rate > 0.50 and m.profit_factor > 1.2:
            action = self._nudge_up(
                "ORB_TAKE_PROFIT_PERCENT",
                reason=f"ORB strong: WR={m.win_rate:.1%} PF={m.profit_factor:.2f} → widening TP",
                strategy="orb",
                snapshot=snapshot,
            )
            if action:
                actions.append(action)

        # Low win rate → tighten SL
        if m.win_rate < 0.40 and m.total_trades >= 30:
            action = self._nudge_down(
                "ORB_STOP_LOSS_PERCENT",
                reason=f"ORB low WR={m.win_rate:.1%} → tightening SL",
                strategy="orb",
                snapshot=snapshot,
            )
            if action:
                actions.append(action)

        # Deeply negative → reduce sizing
        if m.total_pnl < Decimal("-50") and m.total_trades >= 20:
            action = self._nudge_down(
                "ORB_ORDER_SIZE_USD",
                reason=f"ORB negative PnL ${m.total_pnl:.2f} → reducing size",
                strategy="orb",
                snapshot=snapshot,
            )
            if action:
                actions.append(action)

        # Very profitable → increase sizing
        if m.total_pnl > Decimal("100") and m.profit_factor > 1.5:
            action = self._nudge_up(
                "ORB_ORDER_SIZE_USD",
                reason=f"ORB profitable ${m.total_pnl:.2f} PF={m.profit_factor:.2f} → sizing up",
                strategy="orb",
                snapshot=snapshot,
            )
            if action:
                actions.append(action)

        return actions

    # ── Nudge helpers ─────────────────────────────────────────────────────────

    def _nudge_up(
        self, param: str, reason: str, strategy: str, snapshot: dict
    ) -> TuneAction | None:
        """Increase a parameter by up to MAX_ADJUSTMENT_PCT within bounds."""
        current = getattr(self.settings, param, None)
        if current is None:
            return None

        current = float(current)
        lo, hi = BOUNDS.get(param, (current * 0.5, current * 2.0))
        delta = current * MAX_ADJUSTMENT_PCT
        new = min(current + delta, hi)

        if new == current:
            return None

        return TuneAction(
            parameter=param,
            old_value=current,
            new_value=round(new, 4),
            reason=reason,
            strategy=strategy,
            metrics_snapshot=snapshot,
        )

    def _nudge_down(
        self, param: str, reason: str, strategy: str, snapshot: dict
    ) -> TuneAction | None:
        """Decrease a parameter by up to MAX_ADJUSTMENT_PCT within bounds."""
        current = getattr(self.settings, param, None)
        if current is None:
            return None

        current = float(current)
        lo, hi = BOUNDS.get(param, (current * 0.5, current * 2.0))
        delta = current * MAX_ADJUSTMENT_PCT
        new = max(current - delta, lo)

        if new == current:
            return None

        return TuneAction(
            parameter=param,
            old_value=current,
            new_value=round(new, 4),
            reason=reason,
            strategy=strategy,
            metrics_snapshot=snapshot,
        )

    # ── Rollback ──────────────────────────────────────────────────────────────

    def _check_rollback(self, report: PerformanceReport) -> list[TuneAction]:
        """If performance worsened after last tune, roll back."""
        if not self._pending_rollbacks:
            return []

        # Only rollback if things are worse AND we have enough data
        if not report.is_degraded:
            self._pending_rollbacks = []
            return []

        logger.warning("Performance degraded after tuning — rolling back!")
        rollbacks = []
        for action in self._pending_rollbacks:
            rollbacks.append(TuneAction(
                parameter=action.parameter,
                old_value=action.new_value,
                new_value=action.old_value,
                reason=f"ROLLBACK: performance degraded after previous tune",
                strategy=action.strategy,
            ))

        if rollbacks:
            self._apply_to_env(rollbacks)

        self._pending_rollbacks = []
        return rollbacks

    # ── .env persistence ──────────────────────────────────────────────────────

    def _apply_to_env(self, actions: list[TuneAction]) -> None:
        """Write parameter changes to the .env file."""
        env_path = Path(self.env_path)
        if not env_path.exists():
            logger.error("Cannot apply tune: .env not found at {}", env_path)
            return

        lines = env_path.read_text().splitlines()
        changes = {a.parameter: a.new_value for a in actions}

        new_lines = []
        applied = set()
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in changes:
                    new_lines.append(f"{key}={changes[key]}")
                    applied.add(key)
                    continue
            new_lines.append(line)

        # Add any new keys that weren't in the file
        for key, val in changes.items():
            if key not in applied:
                new_lines.append(f"{key}={val}")

        env_path.write_text("\n".join(new_lines) + "\n")
        logger.info("Applied {} parameter changes to .env", len(changes))

    # ── History ───────────────────────────────────────────────────────────────

    def _persist_history(self) -> None:
        """Save tuning history to JSON."""
        path = Path("data/guardian_tune_history.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        recent = self._history[-200:]
        try:
            path.write_text(json.dumps([a.to_dict() for a in recent], indent=2))
        except Exception as e:
            logger.warning("Failed to persist tune history: {}", e)

    def _load_history(self) -> None:
        """Load previous tuning history."""
        path = Path("data/guardian_tune_history.json")
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self._history = [
                    TuneAction(
                        parameter=d.get("parameter", ""),
                        old_value=d.get("old_value", 0),
                        new_value=d.get("new_value", 0),
                        reason=d.get("reason", ""),
                        strategy=d.get("strategy", ""),
                    )
                    for d in data
                ]
            except Exception:
                pass

    def get_recent_tunes(self, n: int = 20) -> list[TuneAction]:
        """Return the N most recent tuning actions."""
        return self._history[-n:]
