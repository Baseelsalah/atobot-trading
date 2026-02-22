"""AtoBot Trading entry point."""

from __future__ import annotations

import asyncio
import signal
import sys

from src.config.settings import get_settings
from src.core.bot import AtoBot


async def main() -> None:
    """Initialise settings, create the bot, and run it with graceful shutdown."""
    settings = get_settings()
    bot = AtoBot(settings)

    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        """Schedule graceful stop when SIGINT / SIGTERM is received."""
        loop.create_task(bot.stop())

    # Register signal handlers (Unix). On Windows we rely on KeyboardInterrupt.
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

    try:
        await bot.start()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()


def run() -> None:
    """Synchronous wrapper suitable for ``python -m`` or console_scripts."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAtoBot interrupted — shutting down.")
        sys.exit(0)


if __name__ == "__main__":
    run()
