"""Abstract notifier interface for AtoBot Trading."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.trade import Trade


class BaseNotifier(ABC):
    """Every notification backend must implement this interface."""

    @abstractmethod
    async def send_message(self, message: str) -> bool:
        """Send a free-form text message.

        Returns:
            True if the message was sent successfully.
        """
        ...

    @abstractmethod
    async def send_trade_alert(self, trade: Trade) -> bool:
        """Send an alert when a trade is executed.

        Returns:
            True if the alert was sent successfully.
        """
        ...

    @abstractmethod
    async def send_error_alert(self, error: str) -> bool:
        """Send an alert for an error condition.

        Returns:
            True if the alert was sent successfully.
        """
        ...

    @abstractmethod
    async def send_daily_summary(self, summary: dict) -> bool:
        """Send an end-of-day summary.

        Args:
            summary: Dictionary with PnL, trade count, win rate, etc.

        Returns:
            True if the summary was sent successfully.
        """
        ...
