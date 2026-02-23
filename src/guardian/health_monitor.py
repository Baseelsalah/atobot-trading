"""Health Monitor — checks that AtoBot & its dependencies are alive.

Probes:
- Docker container status (via Docker SDK & socket)
- Alpaca API connectivity (lightweight /v2/account call)
- Database accessibility (SELECT 1)
- Disk & memory usage
- Log file growth (stale = possible crash)
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from loguru import logger

from src.config.settings import Settings

try:
    import docker as docker_sdk
except ImportError:
    docker_sdk = None  # type: ignore


# ── Health Report ─────────────────────────────────────────────────────────────


@dataclass
class HealthReport:
    """Snapshot of system health at a point in time."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Container
    bot_container_running: bool = False
    dashboard_container_running: bool = False
    bot_restart_count: int = 0

    # API
    alpaca_reachable: bool = False
    alpaca_equity: Decimal = Decimal("0")
    alpaca_buying_power: Decimal = Decimal("0")
    api_latency_ms: float = 0.0

    # Database
    db_accessible: bool = False
    db_size_mb: float = 0.0

    # System
    disk_usage_percent: float = 0.0
    disk_free_mb: float = 0.0
    memory_usage_percent: float = 0.0
    swap_usage_percent: float = 0.0

    # Logs
    log_file_size_mb: float = 0.0
    log_last_modified_ago_s: float = 0.0  # seconds since last write

    # Overall
    is_healthy: bool = False
    issues: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "HEALTHY" if self.is_healthy else "UNHEALTHY"
        issues_str = "; ".join(self.issues) if self.issues else "none"
        return (
            f"[{status}] bot={self.bot_container_running} "
            f"api={self.alpaca_reachable} db={self.db_accessible} "
            f"disk={self.disk_usage_percent:.0f}% "
            f"equity=${self.alpaca_equity} | issues: {issues_str}"
        )


class HealthMonitor:
    """Probes system health and produces a HealthReport."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._consecutive_api_failures = 0
        self._consecutive_db_failures = 0

    async def check(self) -> HealthReport:
        """Run all health probes and return a report."""
        report = HealthReport()

        # 1. Container status (via Docker API / subprocess)
        await self._check_containers(report)

        # 2. Alpaca API
        await self._check_alpaca(report)

        # 3. Database
        await self._check_database(report)

        # 4. Disk & memory
        self._check_system_resources(report)

        # 5. Logs
        self._check_logs(report)

        # 6. Overall verdict — only critical checks affect health status
        report.is_healthy = (
            report.bot_container_running
            and report.alpaca_reachable
            and report.db_accessible
        )

        return report

    async def _check_containers(self, report: HealthReport) -> None:
        """Check Docker container status via Docker SDK (socket)."""
        if docker_sdk is None:
            report.issues.append("docker SDK not installed")
            return

        try:
            client = docker_sdk.DockerClient(
                base_url="unix:///var/run/docker.sock", timeout=10
            )

            # Bot container
            try:
                bot = client.containers.get("atobot")
                report.bot_container_running = bot.status == "running"
                report.bot_restart_count = bot.attrs.get(
                    "RestartCount", 0
                )
                if not report.bot_container_running:
                    report.issues.append("bot container not running")
            except docker_sdk.errors.NotFound:
                report.issues.append("bot container not found")
            except Exception as e:
                logger.warning("Bot container check failed: {}", e)
                report.issues.append(f"bot container check error: {e}")

            # Dashboard container (non-critical)
            try:
                dash = client.containers.get("atobot-dashboard")
                report.dashboard_container_running = dash.status == "running"
            except Exception:
                pass

            client.close()

        except Exception as e:
            logger.warning("Docker connection failed: {}", e)
            report.issues.append(f"Docker connection error: {e}")

    async def _check_alpaca(self, report: HealthReport) -> None:
        """Lightweight Alpaca account check."""
        try:
            from alpaca.trading.client import TradingClient

            start = time.monotonic()
            client = TradingClient(
                api_key=self.settings.ALPACA_API_KEY,
                secret_key=self.settings.ALPACA_API_SECRET,
                paper=self.settings.ALPACA_PAPER,
            )
            account = client.get_account()
            elapsed = (time.monotonic() - start) * 1000

            report.alpaca_reachable = True
            report.alpaca_equity = Decimal(str(account.equity))
            report.alpaca_buying_power = Decimal(str(account.buying_power))
            report.api_latency_ms = elapsed
            self._consecutive_api_failures = 0

            if elapsed > 5000:
                report.issues.append(f"API latency high: {elapsed:.0f}ms")

        except Exception as e:
            self._consecutive_api_failures += 1
            report.issues.append(f"Alpaca API unreachable: {e}")
            logger.warning(
                "Alpaca API check failed ({} consecutive): {}",
                self._consecutive_api_failures, e,
            )

    async def _check_database(self, report: HealthReport) -> None:
        """Verify database is accessible."""
        try:
            import aiosqlite

            db_path = self.settings.DATABASE_URL.replace(
                "sqlite+aiosqlite:///", ""
            )
            # Check file exists and size
            if os.path.exists(db_path):
                report.db_size_mb = os.path.getsize(db_path) / (1024 * 1024)

            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute("SELECT 1")
                await cursor.fetchone()

            report.db_accessible = True
            self._consecutive_db_failures = 0

        except Exception as e:
            self._consecutive_db_failures += 1
            report.issues.append(f"Database error: {e}")
            logger.warning(
                "DB check failed ({} consecutive): {}",
                self._consecutive_db_failures, e,
            )

    def _check_system_resources(self, report: HealthReport) -> None:
        """Check disk and memory usage."""
        # Disk
        try:
            usage = shutil.disk_usage("/")
            report.disk_usage_percent = (usage.used / usage.total) * 100
            report.disk_free_mb = usage.free / (1024 * 1024)
            if report.disk_usage_percent > 92:
                report.issues.append(
                    f"Disk usage high: {report.disk_usage_percent:.0f}%"
                )
        except Exception:
            pass

        # Memory (Linux only — read from /proc/meminfo)
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        meminfo[parts[0].rstrip(":")] = int(parts[1])

            total = meminfo.get("MemTotal", 1)
            available = meminfo.get("MemAvailable", total)
            report.memory_usage_percent = ((total - available) / total) * 100

            swap_total = meminfo.get("SwapTotal", 0)
            swap_free = meminfo.get("SwapFree", 0)
            if swap_total > 0:
                report.swap_usage_percent = ((swap_total - swap_free) / swap_total) * 100

            if report.memory_usage_percent > 90:
                report.issues.append(
                    f"Memory usage high: {report.memory_usage_percent:.0f}%"
                )
        except Exception:
            pass  # Non-Linux systems

    def _check_logs(self, report: HealthReport) -> None:
        """Check log file health."""
        log_path = Path(self.settings.LOG_FILE)
        if log_path.exists():
            stat = log_path.stat()
            report.log_file_size_mb = stat.st_size / (1024 * 1024)
            report.log_last_modified_ago_s = time.time() - stat.st_mtime

            # If log hasn't been written in 5 min during market hours, suspicious
            if report.log_last_modified_ago_s > 300:
                report.issues.append(
                    f"Log stale: last write {report.log_last_modified_ago_s:.0f}s ago"
                )
        else:
            logger.debug("Log file not found at {}", log_path)
