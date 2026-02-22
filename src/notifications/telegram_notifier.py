"""Telegram notification implementation for AtoBot Trading."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone

import aiohttp
from loguru import logger

from src.models.trade import Trade
from src.notifications.base_notifier import BaseNotifier
from src.utils.helpers import format_usd


class TelegramNotifier(BaseNotifier):
    """Send notifications via the Telegram Bot API.

    Messages are formatted with emojis and Markdown.  A simple rate-limiter
    queue prevents exceeding Telegram's ~30 msgs/sec limit.
    """

    TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
    MAX_MSGS_PER_SECOND = 25  # Conservative limit

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._url = self.TELEGRAM_API.format(token=bot_token)
        self._send_times: deque[float] = deque(maxlen=self.MAX_MSGS_PER_SECOND)

    # ── Public interface ──────────────────────────────────────────────────────

    async def send_message(self, message: str) -> bool:
        """Send a plain text message."""
        return await self._send(message)

    async def send_trade_alert(self, trade: Trade) -> bool:
        """Send a formatted trade alert.

        Format examples:
            🟢 BUY  | AAPL | 10 @ $185.50 | Momentum Strategy
            🔴 SELL | TSLA | 5 @ $250.00 | +$12.50 profit
        """
        emoji = "🟢" if trade.side == "BUY" else "🔴"
        pnl_str = ""
        if trade.pnl is not None:
            pnl_str = f" | PnL: {format_usd(trade.pnl)}"

        text = (
            f"{emoji} *{trade.side}* | `{trade.symbol}` | "
            f"{trade.quantity} @ {format_usd(trade.price)}"
            f"{pnl_str} | _{trade.strategy}_"
        )
        return await self._send(text)

    async def send_error_alert(self, error: str) -> bool:
        """Send an error alert."""
        text = f"⚠️ *ERROR* | {error}"
        return await self._send(text)

    async def send_daily_summary(self, summary: dict) -> bool:
        """Send an end-of-day summary.

        Expected keys: pnl, trades, win_rate, open_positions, balance
        """
        pnl = summary.get("pnl", "N/A")
        trades = summary.get("trades", 0)
        win_rate = summary.get("win_rate", 0)
        positions = summary.get("open_positions", 0)
        balance = summary.get("balance", "N/A")

        text = (
            "📊 *DAILY SUMMARY*\n"
            f"• PnL: {pnl}\n"
            f"• Trades: {trades}\n"
            f"• Win Rate: {win_rate}%\n"
            f"• Open Positions: {positions}\n"
            f"• Balance: {balance}"
        )
        return await self._send(text)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _send(self, text: str) -> bool:
        """Send a message to Telegram with rate-limiting."""
        await self._rate_limit()
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                }
                async with session.post(
                    self._url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        logger.debug("Telegram message sent successfully")
                        return True
                    body = await resp.text()
                    logger.warning(
                        "Telegram API returned {}: {}", resp.status, body
                    )
                    return False
        except asyncio.TimeoutError:
            logger.warning("Telegram send timed out")
            return False
        except Exception as exc:
            logger.error("Failed to send Telegram message: {}", exc)
            return False

    async def _rate_limit(self) -> None:
        """Simple sliding-window rate limiter."""
        now = asyncio.get_event_loop().time()
        if len(self._send_times) >= self.MAX_MSGS_PER_SECOND:
            oldest = self._send_times[0]
            elapsed = now - oldest
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
        self._send_times.append(asyncio.get_event_loop().time())
