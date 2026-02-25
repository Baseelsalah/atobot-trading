"""Adaptive Strategy Selector — regime-aware strategy rotation.

Wires the regime detector's strategy_weights and size_multiplier to the
engine's strategy loop.  Provides dynamic enable/disable and weight-based
filtering so AtoBot acts differently in trending vs choppy vs volatile
markets.

Features:
- Regime-based strategy weights (momentum gets +30% in trends, etc.)
- Size multiplier passthrough (reduce size in choppy/volatile regimes)
- Strategy enable/disable based on regime + performance
- Cooldown tracking for underperforming strategies
- Per-strategy performance feedback for auto-tuning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from loguru import logger


@dataclass
class StrategyState:
    """Runtime state for a single strategy."""
    name: str
    enabled: bool = True
    weight: float = 1.0           # Current regime weight (0.0-1.5)
    cooldown_until: datetime | None = None
    recent_wins: int = 0
    recent_losses: int = 0
    consecutive_losses: int = 0
    total_pnl: float = 0.0

    @property
    def is_active(self) -> bool:
        """Strategy is active if enabled, weight > 0, and not in cooldown."""
        if not self.enabled:
            return False
        if self.weight <= 0.1:
            return False
        if self.cooldown_until:
            if datetime.now(timezone.utc) < self.cooldown_until:
                return False
            self.cooldown_until = None  # Cooldown expired
        return True

    @property
    def win_rate(self) -> float:
        total = self.recent_wins + self.recent_losses
        return self.recent_wins / total if total > 0 else 0.5


class AdaptiveStrategySelector:
    """Dynamically adjusts strategy participation based on regime and performance.

    Usage:
        selector = AdaptiveStrategySelector()
        selector.register_strategy("momentum")
        selector.register_strategy("vwap")
        selector.register_strategy("orb")

        # On regime update
        selector.update_from_regime(regime_detector)

        # Before processing a symbol
        if selector.should_trade("momentum", symbol):
            # proceed
        qty = selector.apply_size_multiplier(base_qty)
    """

    def __init__(
        self,
        max_consecutive_losses: int = 5,
        cooldown_minutes: int = 30,
        min_weight_threshold: float = 0.2,
        performance_window: int = 20,
    ):
        self._strategies: dict[str, StrategyState] = {}
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_minutes = cooldown_minutes
        self.min_weight_threshold = min_weight_threshold
        self.performance_window = performance_window

        # Global regime state
        self._size_multiplier: float = 1.0
        self._direction_bias: str = "both"  # "long", "short", "both", "none"
        self._risk_on: bool = True
        self._regime_label: str = "unknown"

    # ── Registration ──────────────────────────────────────────────────────────

    def register_strategy(self, name: str, enabled: bool = True) -> None:
        """Register a strategy for tracking."""
        if name not in self._strategies:
            self._strategies[name] = StrategyState(name=name, enabled=enabled)
            logger.debug("Strategy registered: {}", name)

    def enable_strategy(self, name: str) -> None:
        if name in self._strategies:
            self._strategies[name].enabled = True

    def disable_strategy(self, name: str) -> None:
        if name in self._strategies:
            self._strategies[name].enabled = False

    # ── Regime Updates ────────────────────────────────────────────────────────

    def update_from_regime(self, regime_detector) -> None:
        """Pull strategy_weights, size_multiplier, direction from regime detector.

        Call this every time the regime updates (~60s interval in engine).
        """
        try:
            weights = regime_detector.get_strategy_weights()
            self._size_multiplier = regime_detector.get_size_multiplier()

            regime = regime_detector.get_current_regime()
            self._direction_bias = getattr(regime, "direction", "both")
            self._risk_on = getattr(regime, "risk_on", True)
            self._regime_label = str(getattr(regime, "trend", "unknown"))

            for name, weight in weights.items():
                if name in self._strategies:
                    self._strategies[name].weight = weight
                    logger.debug("Strategy {} weight→{:.2f} (regime: {})",
                                 name, weight, self._regime_label)

        except Exception as exc:
            logger.debug("Failed to update from regime: {}", exc)

    def update_weights_manual(self, weights: dict[str, float]) -> None:
        """Manually set strategy weights (for testing or override)."""
        for name, weight in weights.items():
            if name in self._strategies:
                self._strategies[name].weight = weight

    # ── Query ─────────────────────────────────────────────────────────────────

    def should_trade(self, strategy_name: str, side: str = "BUY") -> tuple[bool, str]:
        """Check if a strategy should be active right now.

        Returns (allowed, reason).
        """
        state = self._strategies.get(strategy_name)
        if not state:
            return True, ""  # Unknown strategy = allow

        if not state.is_active:
            reason = "disabled" if not state.enabled else \
                     f"weight too low ({state.weight:.2f})" if state.weight <= 0.1 else \
                     f"cooldown until {state.cooldown_until}"
            return False, f"Strategy {strategy_name}: {reason}"

        # Weight below threshold = skip (but not disable)
        if state.weight < self.min_weight_threshold:
            return False, f"Strategy {strategy_name}: weight {state.weight:.2f} < threshold {self.min_weight_threshold}"

        # Direction bias check
        if side.upper() == "BUY" and self._direction_bias == "short":
            return False, f"Regime bias is short-only, blocking BUY"
        if side.upper() == "SELL" and self._direction_bias == "long":
            return False, f"Regime bias is long-only, blocking SELL"
        if self._direction_bias == "none":
            return False, "Regime bias is none (choppy), holding off"

        # Risk-off check
        if not self._risk_on and state.weight < 0.8:
            return False, f"Risk-off regime, low-weight strategy ({state.weight:.2f})"

        return True, ""

    def get_size_multiplier(self) -> float:
        """Return regime-based size multiplier (0.1 - 1.5)."""
        return self._size_multiplier

    def get_strategy_weight(self, name: str) -> float:
        """Return current weight for a strategy."""
        state = self._strategies.get(name)
        return state.weight if state else 1.0

    def get_active_strategies(self) -> list[str]:
        """Return names of currently active strategies."""
        return [name for name, state in self._strategies.items() if state.is_active]

    # ── Performance Feedback ──────────────────────────────────────────────────

    def record_trade_result(self, strategy_name: str, pnl: float) -> None:
        """Record a trade outcome for adaptive cooldown logic.

        Auto-disables strategies that hit max consecutive losses.
        """
        state = self._strategies.get(strategy_name)
        if not state:
            return

        state.total_pnl += pnl
        if pnl > 0:
            state.recent_wins += 1
            state.consecutive_losses = 0
        else:
            state.recent_losses += 1
            state.consecutive_losses += 1

        # Auto-cooldown on streak
        if state.consecutive_losses >= self.max_consecutive_losses:
            state.cooldown_until = datetime.now(timezone.utc) + timedelta(
                minutes=self.cooldown_minutes
            )
            state.consecutive_losses = 0  # Reset after cooldown applied
            logger.warning(
                "Strategy {} entering {}min cooldown after {} consecutive losses",
                strategy_name, self.cooldown_minutes, self.max_consecutive_losses,
            )

        # Rolling window: keep last N trades
        total = state.recent_wins + state.recent_losses
        if total > self.performance_window * 2:
            # Halve the counters to create a rolling effect
            state.recent_wins = state.recent_wins // 2
            state.recent_losses = state.recent_losses // 2

    # ── Apply ─────────────────────────────────────────────────────────────────

    def apply_size_multiplier(self, base_quantity: float,
                               strategy_name: str = "") -> float:
        """Apply regime + strategy-specific sizing.

        Returns adjusted quantity.
        """
        qty = base_quantity * self._size_multiplier
        if strategy_name:
            weight = self.get_strategy_weight(strategy_name)
            qty *= weight
        return max(1.0, qty)

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return full selector status."""
        return {
            "regime": self._regime_label,
            "size_multiplier": self._size_multiplier,
            "direction_bias": self._direction_bias,
            "risk_on": self._risk_on,
            "strategies": {
                name: {
                    "active": state.is_active,
                    "enabled": state.enabled,
                    "weight": round(state.weight, 2),
                    "win_rate": round(state.win_rate, 2),
                    "consecutive_losses": state.consecutive_losses,
                    "cooldown": str(state.cooldown_until) if state.cooldown_until else None,
                    "total_pnl": round(state.total_pnl, 2),
                }
                for name, state in self._strategies.items()
            },
        }

    def reset_daily(self) -> None:
        """Reset daily performance counters."""
        for state in self._strategies.values():
            state.recent_wins = 0
            state.recent_losses = 0
            state.consecutive_losses = 0
            state.cooldown_until = None
