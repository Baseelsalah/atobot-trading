"""Guardian Agent — main loop that orchestrates all watchdog modules.

Runs as a separate process (Docker container) alongside the trading bot.
Every cycle:
  1. Health check → detect problems
  2. Self-heal → fix what's broken
  3. Analyze performance → compute metrics (hourly during market, daily overnight)
  4. Auto-tune → adjust parameters if warranted (max once/24h)
  5. Log summary

Cycle interval: 60 seconds (health), performance analysis every hour.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config.settings import Settings, get_settings
from src.guardian.auto_tuner import AutoTuner
from src.guardian.health_monitor import HealthMonitor, HealthReport
from src.guardian.performance_analyzer import PerformanceAnalyzer, PerformanceReport
from src.guardian.self_healer import SelfHealer


# ── Configuration ─────────────────────────────────────────────────────────────

HEALTH_CHECK_INTERVAL_S = 60        # Every minute
PERFORMANCE_INTERVAL_S = 3600       # Every hour
TUNE_INTERVAL_S = 86400             # Every 24 hours
REPORT_SUMMARY_INTERVAL_S = 21600   # Every 6 hours

# Guardian log file (separate from bot)
GUARDIAN_LOG = "logs/guardian.log"


def setup_guardian_logger() -> None:
    """Configure loguru for the guardian process."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>guardian</cyan> | {message}",
    )
    log_path = Path(GUARDIAN_LOG)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
    )


class GuardianAgent:
    """Main guardian agent that ties all modules together."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.health_monitor = HealthMonitor(settings)
        self.self_healer = SelfHealer(settings)
        self.performance_analyzer = PerformanceAnalyzer(
            db_path=settings.DATABASE_URL.replace("sqlite+aiosqlite:///", "")
        )
        self.auto_tuner = AutoTuner(settings, env_path=".env")

        self._shutdown = False
        self._last_health_check = 0.0
        self._last_performance_check = time.time()  # Don't run on first cycle
        self._last_tune_check = time.time()          # Don't run on first cycle
        self._last_summary = 0.0

        # Latest reports
        self.latest_health: HealthReport | None = None
        self.latest_performance: PerformanceReport | None = None

        # Stats
        self._cycles = 0
        self._heals_total = 0
        self._tunes_total = 0

    async def run(self) -> None:
        """Main guardian loop."""
        logger.info("=" * 60)
        logger.info("Guardian Agent starting")
        logger.info(
            "Health: every {}s | Performance: every {}s | Tune: every {}s",
            HEALTH_CHECK_INTERVAL_S, PERFORMANCE_INTERVAL_S, TUNE_INTERVAL_S,
        )
        logger.info("=" * 60)

        while not self._shutdown:
            try:
                now = time.time()
                self._cycles += 1

                # 1. Health check (every minute)
                if now - self._last_health_check >= HEALTH_CHECK_INTERVAL_S:
                    await self._run_health_cycle()
                    self._last_health_check = now

                # 2. Performance analysis (every hour)
                if now - self._last_performance_check >= PERFORMANCE_INTERVAL_S:
                    await self._run_performance_cycle()
                    self._last_performance_check = now

                # 3. Auto-tune (every 24 hours)
                if now - self._last_tune_check >= TUNE_INTERVAL_S:
                    await self._run_tune_cycle()
                    self._last_tune_check = now

                # 4. Periodic summary (every 6 hours)
                if now - self._last_summary >= REPORT_SUMMARY_INTERVAL_S:
                    self._log_summary()
                    self._last_summary = now

                # Sleep until next check
                await asyncio.sleep(HEALTH_CHECK_INTERVAL_S)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Guardian cycle error: {}", e)
                await asyncio.sleep(30)  # Back off on error

        logger.info("Guardian Agent stopped")

    async def _run_health_cycle(self) -> None:
        """Run health check and auto-heal."""
        try:
            report = await self.health_monitor.check()
            self.latest_health = report

            if report.is_healthy:
                logger.debug("Health OK | {}", report)
            else:
                logger.warning("Health ISSUE | {}", report)

                # Auto-heal
                actions = await self.self_healer.heal(report)
                self._heals_total += len(actions)
                for a in actions:
                    logger.info("Heal action: {}", a)

        except Exception as e:
            logger.error("Health cycle failed: {}", e)

    async def _run_performance_cycle(self) -> None:
        """Run performance analysis."""
        try:
            # Analyze last 7 days
            report = await self.performance_analyzer.analyze(period_days=7)
            self.latest_performance = report

            logger.info("Performance | {}", report)

            if report.alerts:
                for alert in report.alerts:
                    logger.warning("Performance alert: {}", alert)

            if report.is_degraded:
                logger.warning(
                    "PERFORMANCE DEGRADED — auto-tuner will evaluate"
                )

            # Save report to file
            self._save_performance_report(report)

        except Exception as e:
            logger.error("Performance cycle failed: {}", e)

    async def _run_tune_cycle(self) -> None:
        """Evaluate and potentially tune parameters."""
        if not self.latest_performance:
            logger.debug("Skipping tune: no performance data yet")
            return

        try:
            actions = await self.auto_tuner.evaluate_and_tune(
                self.latest_performance
            )
            self._tunes_total += len(actions)

            if actions:
                logger.info(
                    "Auto-tuner made {} adjustments:", len(actions)
                )
                for a in actions:
                    logger.info("  {}", a)

                # After tuning .env, restart the bot to pick up changes
                if any(a.parameter for a in actions):
                    logger.info("Restarting bot to apply tuned parameters...")
                    await self._restart_bot_for_tune()
            else:
                logger.info("Auto-tuner: no adjustments needed")

        except Exception as e:
            logger.error("Tune cycle failed: {}", e)

    async def _restart_bot_for_tune(self) -> None:
        """Restart the bot container to pick up .env changes."""
        try:
            import docker as docker_sdk

            client = docker_sdk.DockerClient(
                base_url="unix:///var/run/docker.sock", timeout=60
            )
            ctr = client.containers.get("atobot")
            ctr.restart(timeout=30)
            client.close()
            logger.info("Bot restarted successfully for parameter reload")
        except Exception as e:
            logger.error("Bot restart failed: {}", e)

    def _log_summary(self) -> None:
        """Log a periodic summary of guardian activity."""
        logger.info("=" * 50)
        logger.info("GUARDIAN SUMMARY")
        logger.info("  Cycles: {}", self._cycles)
        logger.info("  Heal actions: {}", self._heals_total)
        logger.info("  Tune actions: {}", self._tunes_total)
        if self.latest_health:
            logger.info("  Last health: {}", self.latest_health)
        if self.latest_performance:
            logger.info("  Last perf: {}", self.latest_performance)
        logger.info("=" * 50)

    def _save_performance_report(self, report: PerformanceReport) -> None:
        """Save performance report to JSON file."""
        path = Path("data/guardian_performance.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = {
                "timestamp": report.timestamp.isoformat(),
                "period_days": report.period_days,
                "total_trades": report.total_trades,
                "total_pnl": str(report.total_pnl),
                "win_rate": report.win_rate,
                "profit_factor": report.profit_factor,
                "sharpe_ratio": report.sharpe_ratio,
                "max_drawdown_pct": report.max_drawdown_pct,
                "consecutive_losses": report.consecutive_losses,
                "is_degraded": report.is_degraded,
                "alerts": report.alerts,
                "strategies": {
                    name: {
                        "trades": m.total_trades,
                        "wins": m.wins,
                        "losses": m.losses,
                        "pnl": str(m.total_pnl),
                        "win_rate": m.win_rate,
                        "profit_factor": m.profit_factor,
                    }
                    for name, m in report.strategy_metrics.items()
                },
            }
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save performance report: {}", e)

    def shutdown(self) -> None:
        """Signal the guardian to stop."""
        logger.info("Guardian shutdown requested")
        self._shutdown = True


# ── Entry point ───────────────────────────────────────────────────────────────


def run() -> None:
    """Run the guardian agent."""
    setup_guardian_logger()

    settings = get_settings()
    agent = GuardianAgent(settings)

    # Signal handlers for graceful shutdown
    loop = asyncio.new_event_loop()

    def handle_signal(sig: int, frame) -> None:
        logger.info("Received signal {}, shutting down...", sig)
        agent.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(agent.run())
    except KeyboardInterrupt:
        agent.shutdown()
    finally:
        loop.close()


if __name__ == "__main__":
    run()
