"""OpenAI-powered market sentiment and trade analysis for AtoBot.

Provides:
1. **Pre-trade sentiment check** — Ask GPT to evaluate whether a trade is
   advisable given recent price action, indicators, and market context.
2. **Daily market briefing** — Summarise market conditions before trading starts.
3. **Trade review** — Analyse completed trades for patterns and improvements.

Usage requires OPENAI_API_KEY in .env. When missing, all methods gracefully
return neutral/allow signals so the bot still functions without AI.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from loguru import logger


class AITradeAdvisor:
    """OpenAI-powered trade advisor for smarter entry/exit decisions."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key
        self._model = model
        self._client: Any | None = None
        self._enabled = bool(api_key)
        self._cache: dict[str, tuple[datetime, dict]] = {}
        self._cache_ttl_seconds = 300  # 5 min cache to avoid excessive API calls
        self._daily_calls = 0
        self._max_daily_calls = 200  # Budget control

    async def initialize(self) -> None:
        """Create the OpenAI client."""
        if not self._enabled:
            logger.info("[AI] OpenAI API key not set — AI advisor disabled")
            return
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self._api_key)
            logger.info("[AI] OpenAI advisor initialized (model={})", self._model)
        except ImportError:
            logger.warning("[AI] openai package not installed — pip install openai")
            self._enabled = False
        except Exception as exc:
            logger.error("[AI] Failed to initialize OpenAI: {}", exc)
            self._enabled = False

    async def evaluate_entry(
        self,
        symbol: str,
        strategy: str,
        current_price: float,
        indicators_data: dict,
        recent_bars: list[dict] | None = None,
    ) -> dict:
        """Ask AI whether to enter a trade.

        Args:
            symbol: Stock ticker (e.g. "AAPL").
            strategy: Strategy name requesting the evaluation.
            current_price: Current stock price.
            indicators_data: Dict of indicator values (RSI, MACD, VWAP, etc.).
            recent_bars: Last 5-10 price bars for context.

        Returns:
            Dict with keys:
                - allow: bool (True = proceed with trade)
                - confidence: float (0-1)
                - reason: str (explanation)
                - suggested_adjustment: str | None
        """
        if not self._enabled or self._client is None:
            return {"allow": True, "confidence": 0.5, "reason": "AI disabled", "suggested_adjustment": None}

        # Check cache
        cache_key = f"entry_{symbol}_{strategy}"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).total_seconds() < self._cache_ttl_seconds:
                return cached_result

        # Budget check
        if self._daily_calls >= self._max_daily_calls:
            return {"allow": True, "confidence": 0.5, "reason": "AI daily budget reached", "suggested_adjustment": None}

        # Build context for GPT
        bars_summary = ""
        if recent_bars and len(recent_bars) >= 5:
            last_5 = recent_bars[-5:]
            bars_summary = "\n".join(
                f"  Bar {i+1}: O={b.get('open','?')} H={b.get('high','?')} L={b.get('low','?')} C={b.get('close','?')} V={b.get('volume','?')}"
                for i, b in enumerate(last_5)
            )

        prompt = f"""You are an expert stock day-trading analyst. Evaluate this potential trade:

**Stock:** {symbol}
**Strategy:** {strategy}
**Current Price:** ${current_price:.2f}
**Time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

**Indicators:**
{json.dumps(indicators_data, indent=2, default=str)}

**Recent 5-min Bars:**
{bars_summary if bars_summary else 'N/A'}

Based on the indicators and price action, should we enter this trade?
Consider:
1. Is the entry signal strong or marginal?
2. Are multiple indicators confirming the same direction?
3. Is there unusual risk (e.g., earnings, market-wide selloff)?
4. Is the risk/reward ratio favorable?

Reply ONLY with a JSON object (no markdown, no code blocks):
{{"allow": true/false, "confidence": 0.0-1.0, "reason": "brief explanation", "suggested_adjustment": "optional suggestion or null"}}"""

        try:
            self._daily_calls += 1
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": "You are a quantitative trading analyst. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=200,
                ),
                timeout=10.0,
            )
            content = response.choices[0].message.content.strip()

            # Parse response — handle potential markdown wrapping
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            result = json.loads(content)
            result.setdefault("allow", True)
            result.setdefault("confidence", 0.5)
            result.setdefault("reason", "No reason provided")
            result.setdefault("suggested_adjustment", None)

            # Cache the result
            self._cache[cache_key] = (datetime.now(timezone.utc), result)

            logger.info(
                "[AI] Entry eval {} [{}]: allow={} confidence={:.1%} | {}",
                symbol, strategy, result["allow"], result["confidence"], result["reason"],
            )
            return result

        except asyncio.TimeoutError:
            logger.warning("[AI] Timeout evaluating {} — allowing trade", symbol)
            return {"allow": True, "confidence": 0.5, "reason": "AI timeout", "suggested_adjustment": None}
        except json.JSONDecodeError as exc:
            logger.warning("[AI] JSON parse error for {}: {} — allowing trade", symbol, exc)
            return {"allow": True, "confidence": 0.5, "reason": "AI parse error", "suggested_adjustment": None}
        except Exception as exc:
            logger.warning("[AI] Error evaluating {}: {} — allowing trade", symbol, exc)
            return {"allow": True, "confidence": 0.5, "reason": f"AI error: {exc}", "suggested_adjustment": None}

    async def generate_market_briefing(self, symbols: list[str]) -> str:
        """Generate a pre-market briefing summarizing conditions.

        Args:
            symbols: List of symbols the bot trades.

        Returns:
            Markdown-formatted briefing string.
        """
        if not self._enabled or self._client is None:
            return "AI market briefing not available (API key not set)"

        if self._daily_calls >= self._max_daily_calls:
            return "AI daily budget reached"

        prompt = f"""Generate a brief pre-market day-trading briefing for these stocks: {', '.join(symbols)}.

Include for each stock:
- Key support/resistance levels
- Any known catalysts (earnings, FDA, etc.)
- Overall market sentiment

Keep it under 300 words. Focus on actionable insights for day traders.
Today's date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"""

        try:
            self._daily_calls += 1
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": "You are a concise pre-market analyst for day traders."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.5,
                    max_tokens=500,
                ),
                timeout=15.0,
            )
            briefing = response.choices[0].message.content.strip()
            logger.info("[AI] Market briefing generated ({} chars)", len(briefing))
            return briefing
        except Exception as exc:
            logger.warning("[AI] Market briefing failed: {}", exc)
            return f"AI briefing unavailable: {exc}"

    async def review_trade(
        self,
        symbol: str,
        strategy: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        duration_minutes: float,
        indicators_at_entry: dict | None = None,
    ) -> str:
        """Review a completed trade and provide improvement suggestions.

        Returns:
            Brief analysis string.
        """
        if not self._enabled or self._client is None:
            return ""

        if self._daily_calls >= self._max_daily_calls:
            return ""

        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        prompt = f"""Briefly review this completed day trade:
- Symbol: {symbol}, Strategy: {strategy}
- Entry: ${entry_price:.2f}, Exit: ${exit_price:.2f}
- PnL: ${pnl:.2f} ({pnl_pct:+.2f}%)
- Duration: {duration_minutes:.0f} minutes
- Indicators at entry: {json.dumps(indicators_at_entry, default=str) if indicators_at_entry else 'N/A'}

In 2-3 sentences: Was the entry/exit timing good? What could improve?"""

        try:
            self._daily_calls += 1
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": "You are a trading coach. Be direct and specific."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=150,
                ),
                timeout=10.0,
            )
            review = response.choices[0].message.content.strip()
            logger.info("[AI] Trade review for {} [{}]: {}", symbol, strategy, review[:100])
            return review
        except Exception as exc:
            logger.warning("[AI] Trade review failed: {}", exc)
            return ""

    def reset_daily_counters(self) -> None:
        """Reset daily API call counter. Call at start of each trading day."""
        self._daily_calls = 0
        self._cache.clear()
        logger.debug("[AI] Daily counters reset")
