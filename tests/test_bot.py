"""Tests for the main bot orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.config.settings import Settings
from src.core.bot import AtoBot


@pytest.mark.asyncio
async def test_bot_init(mock_settings: Settings) -> None:
    """Bot initialises with all components set to None."""
    bot = AtoBot(mock_settings)
    assert bot.exchange is None
    assert bot.strategies == []
    assert bot.engine is None
    assert bot.notifier is None


@pytest.mark.asyncio
async def test_bot_stop_without_start(mock_settings: Settings) -> None:
    """Stopping a bot that was never started should not raise."""
    bot = AtoBot(mock_settings)
    await bot.stop()  # Should complete without error


@pytest.mark.asyncio
async def test_bot_settings_stored(mock_settings: Settings) -> None:
    """Settings are stored on the bot instance."""
    bot = AtoBot(mock_settings)
    assert bot.settings.DRY_RUN is True
    assert bot.settings.ALPACA_PAPER is True
    assert bot.settings.DEFAULT_STRATEGY == "momentum"
