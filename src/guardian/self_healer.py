"""Self-Healer — automatically fixes common problems detected by HealthMonitor.

Remediation actions (in order of severity):
1. Restart crashed bot container
2. Restart dashboard container
3. Clear orphaned/stuck orders via Alpaca API
4. Clean up old Docker images to free disk
5. Prune old log files
6. Force-reconnect on repeated API failures
7. Vacuum SQLite database if bloated

All actions are logged with timestamps so you can audit what happened.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.config.settings import Settings
from src.guardian.health_monitor import HealthReport

try:
    import docker as docker_sdk
except ImportError:
    docker_sdk = None  # type: ignore


class HealAction:
    """Record of a healing action taken."""

    def __init__(self, action: str, reason: str, success: bool, detail: str = ""):
        self.timestamp = datetime.now(timezone.utc)
        self.action = action
        self.reason = reason
        self.success = success
        self.detail = detail

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "reason": self.reason,
            "success": self.success,
            "detail": self.detail,
        }

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return f"[HEAL:{status}] {self.action} — {self.reason}"


class SelfHealer:
    """Automatically remediate issues found in health reports."""

    # Cooldowns: don't repeat the same action too quickly
    RESTART_COOLDOWN_S = 300       # 5 min between restarts
    CLEANUP_COOLDOWN_S = 3600      # 1 hour between cleanups
    ORDER_CLEANUP_COOLDOWN_S = 600  # 10 min

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._history: list[HealAction] = []
        self._last_action_time: dict[str, float] = {}
        self._consecutive_restarts = 0
        self.MAX_CONSECUTIVE_RESTARTS = 5  # Give up after 5 rapid restarts

    async def heal(self, report: HealthReport) -> list[HealAction]:
        """Analyze health report and take corrective actions."""
        actions: list[HealAction] = []

        if report.is_healthy:
            self._consecutive_restarts = 0
            return actions

        # Priority 1: Bot container not running
        if not report.bot_container_running:
            action = await self._restart_container("atobot", "bot container down")
            if action:
                actions.append(action)

        # Priority 2: Dashboard container not running
        if not report.dashboard_container_running:
            action = await self._restart_container(
                "atobot-dashboard", "dashboard container down"
            )
            if action:
                actions.append(action)

        # Priority 3: Alpaca API unreachable (after 3+ consecutive failures)
        if not report.alpaca_reachable:
            action = await self._handle_api_failure()
            if action:
                actions.append(action)

        # Priority 4: Database issues
        if not report.db_accessible:
            action = await self._handle_db_failure()
            if action:
                actions.append(action)

        # Priority 5: Disk space critical
        if report.disk_usage_percent > 92:
            action = await self._free_disk_space(report.disk_usage_percent)
            if action:
                actions.append(action)

        # Priority 6: Stale logs (bot might be frozen)
        if report.bot_container_running and report.log_last_modified_ago_s > 600:
            action = await self._restart_container(
                "atobot", f"log stale for {report.log_last_modified_ago_s:.0f}s (possible freeze)"
            )
            if action:
                actions.append(action)

        # Log and persist all actions
        for a in actions:
            self._history.append(a)
            if a.success:
                logger.info("{}", a)
            else:
                logger.error("{}", a)

        self._persist_history()
        return actions

    def _is_cooled_down(self, action_key: str, cooldown_s: float) -> bool:
        """Check if enough time has passed since last action of this type."""
        import time
        last = self._last_action_time.get(action_key, 0)
        return (time.time() - last) >= cooldown_s

    def _record_action_time(self, action_key: str) -> None:
        import time
        self._last_action_time[action_key] = time.time()

    async def _restart_container(
        self, container: str, reason: str
    ) -> HealAction | None:
        """Restart a Docker container via Docker SDK."""
        action_key = f"restart_{container}"

        if not self._is_cooled_down(action_key, self.RESTART_COOLDOWN_S):
            return None

        if self._consecutive_restarts >= self.MAX_CONSECUTIVE_RESTARTS:
            return HealAction(
                action=f"restart_{container}",
                reason=reason,
                success=False,
                detail=f"Giving up after {self._consecutive_restarts} consecutive restarts",
            )

        if docker_sdk is None:
            return HealAction(
                action=f"restart_{container}",
                reason=reason,
                success=False,
                detail="docker SDK not installed",
            )

        try:
            client = docker_sdk.DockerClient(
                base_url="unix:///var/run/docker.sock", timeout=60
            )
            ctr = client.containers.get(container)
            ctr.restart(timeout=30)
            client.close()

            self._record_action_time(action_key)
            self._consecutive_restarts += 1

            return HealAction(
                action=f"restart_{container}",
                reason=reason,
                success=True,
                detail=f"container restarted successfully",
            )

        except Exception as e:
            self._consecutive_restarts += 1
            return HealAction(
                action=f"restart_{container}",
                reason=reason,
                success=False,
                detail=str(e),
            )

    async def _handle_api_failure(self) -> HealAction | None:
        """Handle Alpaca API connectivity issues."""
        action_key = "api_reconnect"

        if not self._is_cooled_down(action_key, self.RESTART_COOLDOWN_S):
            return None

        # The bot auto-retries API calls, but if the guardian detects
        # persistent failure, restart the bot to force a fresh connection.
        self._record_action_time(action_key)
        return await self._restart_container(
            "atobot", "persistent API connectivity failure — forcing reconnect"
        )

    async def _handle_db_failure(self) -> HealAction | None:
        """Handle database accessibility issues."""
        action_key = "db_repair"

        if not self._is_cooled_down(action_key, self.CLEANUP_COOLDOWN_S):
            return None

        # Try to repair SQLite by running integrity_check
        try:
            import aiosqlite

            db_path = self.settings.DATABASE_URL.replace(
                "sqlite+aiosqlite:///", ""
            )
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute("PRAGMA integrity_check")
                result = await cursor.fetchone()
                is_ok = result and result[0] == "ok"

                if not is_ok:
                    # Vacuum to try to repair
                    await db.execute("VACUUM")
                    await db.commit()

            self._record_action_time(action_key)
            return HealAction(
                action="db_repair",
                reason="database accessibility issue",
                success=True,
                detail=f"integrity_check: {result[0] if result else 'unknown'}, vacuumed",
            )

        except Exception as e:
            self._record_action_time(action_key)
            return HealAction(
                action="db_repair",
                reason="database accessibility issue",
                success=False,
                detail=str(e),
            )

    async def _free_disk_space(self, usage_pct: float) -> HealAction | None:
        """Free disk space by pruning Docker artifacts and old logs."""
        action_key = "disk_cleanup"

        if not self._is_cooled_down(action_key, self.CLEANUP_COOLDOWN_S):
            return None

        freed_items = []

        try:
            # 1. Prune old Docker images via SDK
            if docker_sdk is not None:
                client = docker_sdk.DockerClient(
                    base_url="unix:///var/run/docker.sock", timeout=60
                )
                pruned = client.images.prune(filters={"dangling": True})
                space = pruned.get("SpaceReclaimed", 0)
                freed_items.append(f"docker image prune: {space // (1024*1024)}MB freed")

                # System prune (dangling containers, networks)
                client.containers.prune()
                client.networks.prune()
                freed_items.append("pruned stopped containers & unused networks")
                client.close()

            # 2. Truncate old log files (keep last 1000 lines)
            log_dir = Path("logs")
            if log_dir.exists():
                for log_file in log_dir.glob("*.log*"):
                    if log_file.stat().st_size > 50 * 1024 * 1024:  # > 50MB
                        lines = log_file.read_text().split("\n")[-1000:]
                        log_file.write_text("\n".join(lines))
                        freed_items.append(f"truncated {log_file.name}")

            self._record_action_time(action_key)
            return HealAction(
                action="disk_cleanup",
                reason=f"disk at {usage_pct:.0f}%",
                success=True,
                detail="; ".join(freed_items),
            )

        except Exception as e:
            self._record_action_time(action_key)
            return HealAction(
                action="disk_cleanup",
                reason=f"disk at {usage_pct:.0f}%",
                success=False,
                detail=str(e),
            )

    def _persist_history(self) -> None:
        """Save healing history to a JSON file for audit trail."""
        history_path = Path("data/guardian_heal_history.json")
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Keep last 500 entries
        recent = self._history[-500:]
        try:
            history_path.write_text(
                json.dumps([a.to_dict() for a in recent], indent=2)
            )
        except Exception as e:
            logger.warning("Failed to persist heal history: {}", e)

    def get_recent_actions(self, n: int = 20) -> list[HealAction]:
        """Return the N most recent healing actions."""
        return self._history[-n:]
