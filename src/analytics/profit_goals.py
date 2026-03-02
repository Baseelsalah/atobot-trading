"""Profit Goal Tracker — daily, weekly, and monthly P&L targets.

Gives the bot structured profit awareness:
- Tracks cumulative P&L against daily / weekly / monthly targets.
- Provides a risk multiplier that scales position sizing based on
  goal progress (reduce size when goal is met, cut risk when losing).
- Enforces a hard daily loss limit separate from RiskManager.
- Persists state to a JSON file so goals survive restarts.
- Auto-resets on period boundaries (new day / new week / new month).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


@dataclass
class GoalPeriod:
    """Tracks a single goal period (daily / weekly / monthly)."""

    target: float = 0.0      # Dollar target for the period
    current: float = 0.0     # Accumulated P&L this period
    trades: int = 0          # Number of trades this period
    wins: int = 0            # Winning trades this period
    losses: int = 0          # Losing trades this period
    peak: float = 0.0        # Peak P&L this period (for intra-period drawdown)
    reset_date: str = ""     # ISO date when this period was last reset

    @property
    def progress_pct(self) -> float:
        """Percentage of target achieved (can exceed 100%)."""
        if self.target <= 0:
            return 0.0
        return round((self.current / self.target) * 100, 1)

    @property
    def win_rate(self) -> float:
        """Win rate for this period."""
        if self.trades == 0:
            return 0.0
        return round((self.wins / self.trades) * 100, 1)

    @property
    def is_goal_met(self) -> bool:
        """True if the target has been reached."""
        return self.target > 0 and self.current >= self.target

    @property
    def drawdown_from_peak(self) -> float:
        """Current drawdown from the period's peak P&L."""
        if self.peak <= 0:
            return 0.0
        return round(self.peak - self.current, 2)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "current": round(self.current, 2),
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "peak": round(self.peak, 2),
            "reset_date": self.reset_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoalPeriod":
        return cls(
            target=d.get("target", 0.0),
            current=d.get("current", 0.0),
            trades=d.get("trades", 0),
            wins=d.get("wins", 0),
            losses=d.get("losses", 0),
            peak=d.get("peak", 0.0),
            reset_date=d.get("reset_date", ""),
        )


class ProfitGoalTracker:
    """Tracks P&L progress against daily / weekly / monthly profit goals.

    Usage::

        tracker = ProfitGoalTracker(
            daily_target=50.0,
            weekly_target=250.0,
            monthly_target=1000.0,
        )
        # After each trade closes:
        risk_mult = tracker.record_trade(pnl=12.50)
        # risk_mult is 0.0-1.0 — multiply position size by this

        # In heartbeat:
        status = tracker.get_status_summary()
    """

    def __init__(
        self,
        daily_target: float = 0.0,
        weekly_target: float = 0.0,
        monthly_target: float = 0.0,
        daily_loss_limit: float = 0.0,
        goal_met_risk_scale: float = 0.25,
        losing_day_risk_scale: float = 0.50,
        data_dir: str = "data",
    ) -> None:
        self.daily = GoalPeriod(target=daily_target)
        self.weekly = GoalPeriod(target=weekly_target)
        self.monthly = GoalPeriod(target=monthly_target)

        self._daily_loss_limit = daily_loss_limit  # Hard stop (0 = use settings)
        self._goal_met_risk_scale = goal_met_risk_scale  # Scale to 25% after goal
        self._losing_day_risk_scale = losing_day_risk_scale  # Scale to 50% when losing

        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._data_dir / "profit_goals.json"

        # Load persisted state
        self._load_state()

        # Ensure periods are current (reset stale ones)
        self._check_resets()

        logger.info(
            "ProfitGoalTracker initialised | daily=${:.0f} weekly=${:.0f} monthly=${:.0f}",
            self.daily.target, self.weekly.target, self.monthly.target,
        )

    # ── Trade Recording ───────────────────────────────────────────────────────

    def record_trade(self, pnl: float) -> float:
        """Record a completed trade's P&L and return risk multiplier (0-1).

        The risk multiplier tells the engine how much to scale the next
        position size:
        - 1.0 = normal size
        - 0.5 = 50% size (losing day, be cautious)
        - 0.25 = 25% size (daily goal met, lock in profits)
        - 0.0 = STOP trading (daily loss limit hit)
        """
        self._check_resets()

        is_win = pnl > 0
        for period in (self.daily, self.weekly, self.monthly):
            period.current += pnl
            period.trades += 1
            if is_win:
                period.wins += 1
            else:
                period.losses += 1
            # Track peak for intra-period drawdown
            if period.current > period.peak:
                period.peak = period.current

        self._save_state()

        risk_mult = self._compute_risk_multiplier()

        logger.info(
            "GoalTracker | trade PnL=${:+.2f} | daily=${:+.2f}/{:.0f} ({:.0f}%) "
            "| weekly=${:+.2f}/{:.0f} | monthly=${:+.2f}/{:.0f} | risk_mult={:.2f}",
            pnl,
            self.daily.current, self.daily.target, self.daily.progress_pct,
            self.weekly.current, self.weekly.target,
            self.monthly.current, self.monthly.target,
            risk_mult,
        )

        return risk_mult

    # ── Risk Multiplier ───────────────────────────────────────────────────────

    def _compute_risk_multiplier(self) -> float:
        """Calculate the position-size multiplier based on goal progress.

        Priority (first match wins):
        1. Daily loss limit hit → 0.0 (STOP)
        2. Daily goal met → goal_met_risk_scale (protect profits)
        3. Losing on the day → losing_day_risk_scale (reduce exposure)
        4. Otherwise → 1.0 (full size)
        """
        # 1. Hard daily loss limit
        if self._daily_loss_limit > 0 and self.daily.current <= -self._daily_loss_limit:
            return 0.0

        # 2. Daily goal met — scale down to lock in profits
        if self.daily.is_goal_met:
            return self._goal_met_risk_scale

        # 3. Weekly goal met — slightly conservative
        if self.weekly.is_goal_met:
            return max(self._goal_met_risk_scale, 0.5)

        # 4. Losing on the day — reduce risk
        if self.daily.current < 0:
            return self._losing_day_risk_scale

        # 5. Normal
        return 1.0

    def get_risk_multiplier(self) -> float:
        """Public accessor for current risk multiplier."""
        self._check_resets()
        return self._compute_risk_multiplier()

    def should_stop_trading(self) -> tuple[bool, str]:
        """Check if the bot should stop placing new trades entirely.

        Returns:
            (True, reason) if trading should stop.
            (False, "") if trading can continue.
        """
        self._check_resets()

        if self._daily_loss_limit > 0 and self.daily.current <= -self._daily_loss_limit:
            return True, (
                f"Daily loss limit hit: ${self.daily.current:+.2f} "
                f"(limit: -${self._daily_loss_limit:.2f})"
            )

        return False, ""

    # ── Status & Reporting ────────────────────────────────────────────────────

    def get_status_summary(self) -> str:
        """Human-readable status for heartbeat / notifications."""
        self._check_resets()
        lines = ["📊 Profit Goals:"]

        for label, period in [("Daily", self.daily), ("Weekly", self.weekly), ("Monthly", self.monthly)]:
            if period.target <= 0:
                continue
            emoji = "🟢" if period.current >= 0 else "🔴"
            goal_emoji = "🎯" if period.is_goal_met else ""
            bar = self._progress_bar(period.progress_pct)
            lines.append(
                f"  {emoji} {label}: ${period.current:+.2f} / ${period.target:.0f} "
                f"({period.progress_pct:.0f}%) {goal_emoji}\n    {bar} "
                f"| {period.trades} trades | WR {period.win_rate:.0f}%"
            )

        risk_mult = self._compute_risk_multiplier()
        if risk_mult < 1.0:
            if risk_mult == 0.0:
                lines.append("  🛑 TRADING STOPPED — daily loss limit hit")
            elif self.daily.is_goal_met:
                lines.append(f"  🎯 Daily goal met! Risk scaled to {risk_mult:.0%}")
            else:
                lines.append(f"  ⚠️ Risk scaled to {risk_mult:.0%} (losing day)")

        return "\n".join(lines)

    def get_dashboard_data(self) -> dict:
        """Structured data for the Streamlit dashboard."""
        return {
            "daily": self.daily.to_dict(),
            "weekly": self.weekly.to_dict(),
            "monthly": self.monthly.to_dict(),
            "risk_multiplier": self._compute_risk_multiplier(),
        }

    @staticmethod
    def _progress_bar(pct: float) -> str:
        """Visual progress bar (20 chars)."""
        pct = max(0, min(pct, 100))
        filled = int(pct / 5)
        empty = 20 - filled
        return f"[{'█' * filled}{'░' * empty}]"

    # ── Period Reset Logic ────────────────────────────────────────────────────

    def _check_resets(self) -> None:
        """Reset periods that have crossed their boundary."""
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        week = now.strftime("%Y-W%W")
        month = now.strftime("%Y-%m")

        if self.daily.reset_date != today:
            if self.daily.trades > 0:
                logger.info(
                    "Daily goal reset | yesterday: ${:+.2f} / ${:.0f} "
                    "({} trades, {:.0f}% WR)",
                    self.daily.current, self.daily.target,
                    self.daily.trades, self.daily.win_rate,
                )
            self.daily = GoalPeriod(
                target=self.daily.target,
                reset_date=today,
            )

        if self.weekly.reset_date != week:
            if self.weekly.trades > 0:
                logger.info(
                    "Weekly goal reset | last week: ${:+.2f} / ${:.0f} ({} trades)",
                    self.weekly.current, self.weekly.target, self.weekly.trades,
                )
            self.weekly = GoalPeriod(
                target=self.weekly.target,
                reset_date=week,
            )

        if self.monthly.reset_date != month:
            if self.monthly.trades > 0:
                logger.info(
                    "Monthly goal reset | last month: ${:+.2f} / ${:.0f} ({} trades)",
                    self.monthly.current, self.monthly.target, self.monthly.trades,
                )
            self.monthly = GoalPeriod(
                target=self.monthly.target,
                reset_date=month,
            )

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        """Persist current goal state to disk."""
        try:
            state = {
                "daily": self.daily.to_dict(),
                "weekly": self.weekly.to_dict(),
                "monthly": self.monthly.to_dict(),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            self._state_file.write_text(json.dumps(state, indent=2))
        except Exception as exc:
            logger.warning("Failed to save profit goals state: {}", exc)

    def _load_state(self) -> None:
        """Load persisted goal state from disk."""
        try:
            if self._state_file.exists():
                data = json.loads(self._state_file.read_text())
                # Restore each period but keep current targets from config
                for key, period_attr in [("daily", "daily"), ("weekly", "weekly"), ("monthly", "monthly")]:
                    if key in data:
                        loaded = GoalPeriod.from_dict(data[key])
                        # Preserve the target from config (in case it changed)
                        current_target = getattr(self, period_attr).target
                        loaded.target = current_target
                        setattr(self, period_attr, loaded)
                logger.info(
                    "Loaded profit goals state | daily=${:+.2f} weekly=${:+.2f} monthly=${:+.2f}",
                    self.daily.current, self.weekly.current, self.monthly.current,
                )
        except Exception as exc:
            logger.warning("Failed to load profit goals state: {}", exc)
