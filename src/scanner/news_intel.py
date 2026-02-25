"""News Intelligence — real-time news classification & sentiment scoring.

A 7-10 year day trader instantly classifies news into:
- **Tier 1 (trade NOW)**: Earnings beat/miss, FDA approval/rejection, M&A,
  major analyst upgrade/downgrade, unexpected guidance change.
- **Tier 2 (watch closely)**: Sector news, competitor results, macro data
  (CPI, FOMC), commodity price spikes.
- **Tier 3 (background)**: General market commentary, opinion pieces, ESG
  news, non-material announcements.

The experienced trader also knows:
- Don't trade *into* earnings (but trade the reaction).
- FDA binary events = gambling, not trading.
- Analyst upgrades mid-day = often fade after initial pop.
- Conference call tone matters more than headline numbers.
- Macro releases at fixed times (8:30 AM, 2:00 PM ET) = volatility windows.

This module provides **instant keyword-based classification** (no API calls)
for every news event, and optionally uses the AI advisor for deeper analysis
on Tier 1 events only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any

from loguru import logger


class NewsTier(IntEnum):
    """Urgency tier for a news event."""

    TIER_1_CRITICAL = 1  # Trade immediately / adjust positions
    TIER_2_IMPORTANT = 2  # Monitor closely, may affect trades
    TIER_3_BACKGROUND = 3  # Informational, low impact


class NewsSentiment(str):
    """Sentiment classification."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class NewsEvent:
    """Classified news event with sentiment and impact assessment."""

    headline: str
    symbols: list[str]
    source: str = ""
    summary: str = ""
    url: str = ""
    tier: NewsTier = NewsTier.TIER_3_BACKGROUND
    sentiment: str = "neutral"
    impact_score: float = 0.0  # 0-100, how much this could move the stock
    category: str = ""  # "earnings", "fda", "analyst", "macro", etc.
    actionable: bool = False  # True = engine should act on this
    suggested_action: str = ""  # "reduce_position", "avoid_entry", "watch_for_dip", etc.
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[T{self.tier}|{self.sentiment}|{self.impact_score:.0f}] {self.headline[:80]}"


# ── Keyword patterns for instant classification ──────────────────────────────

# Tier 1: Critical — immediate market-moving events
_TIER1_BULLISH = [
    r"\bearnings\s+beat\b",
    r"\bblowout\s+quarter\b",
    r"\braises?\s+guidance\b",
    r"\braises?\s+outlook\b",
    r"\bupgrades?\s+to\s+buy\b",
    r"\bupgrades?\s+to\s+outperform\b",
    r"\bstrong\s+buy\b",
    r"\bfda\s+approv",
    r"\bfda\s+clears?\b",
    r"\bacquires?\b",
    r"\bacquisition\b",
    r"\bmerger\b",
    r"\bbuyout\b",
    r"\btakeover\b",
    r"\bstock\s+split\b",
    r"\bspecial\s+dividend\b",
    r"\bbeat\s+estimates\b",
    r"\brecord\s+revenue\b",
    r"\brecord\s+earnings\b",
    r"\bblockbuster\b",
    r"\bsurge[sd]?\b.*\bafter\s+hours\b",
    r"\bsoar",
    r"\bnew\s+all[- ]time\s+high\b",
    r"\bbig\s+contract\b",
    r"\bmajor\s+partnership\b",
    r"\bai\s+breakthrough\b",
]

_TIER1_BEARISH = [
    r"\bearnings\s+miss\b",
    r"\bmisses?\s+estimates\b",
    r"\blowers?\s+guidance\b",
    r"\bcuts?\s+guidance\b",
    r"\bcuts?\s+outlook\b",
    r"\bdowngrades?\s+to\s+sell\b",
    r"\bdowngrades?\s+to\s+underperform\b",
    r"\bdowngrades?\s+\w+\s+to\s+sell\b",
    r"\bdowngrades?\s+\w+\s+to\s+underperform\b",
    r"\bfda\s+reject",
    r"\bfda\s+refuse",
    r"\bcrl\b",  # Complete Response Letter (FDA rejection)
    r"\brecall\b",
    r"\bfraud\b",
    r"\binvestigation\b",
    r"\bsec\s+probe\b",
    r"\brestate",  # restatement
    r"\bdelisting\b",
    r"\bbankrupt",
    r"\bdefault\b",
    r"\blayoff",
    r"\bmassive\s+layoff",
    r"\bcrash",
    r"\bplunge[sd]?\b",
    r"\bsells?\s+off\b",
    r"\bwarning\b.*\bprofit\b",
    r"\bprofit\s+warning\b",
    r"\bshort\s+seller\b",
    r"\bceo\s+resign",
    r"\bceo\s+step",
    r"\baccounting\s+irregularit",
]

# Tier 2: Important — sector or macro impact
_TIER2_PATTERNS = [
    (r"\bfomc\b|federal\s+reserve|interest\s+rate", "macro", "mixed"),
    (r"\bcpi\b|inflation\s+data|consumer\s+price", "macro", "mixed"),
    (r"\bjobs?\s+report|nonfarm\s+payroll|unemployment", "macro", "mixed"),
    (r"\bgdp\b.*\bdata\b|economic\s+growth", "macro", "mixed"),
    (r"\boil\s+price|crude\s+oil|opec", "commodity", "mixed"),
    (r"\bchina\b.*\btariff|trade\s+war", "geopolitical", "bearish"),
    (r"\banalyst\b.*\bupgrade", "analyst", "bullish"),
    (r"\banalyst\b.*\bdowngrade", "analyst", "bearish"),
    (r"\bprice\s+target\s+raise|raises?\s+pt\b", "analyst", "bullish"),
    (r"\bprice\s+target\s+cut|cuts?\s+pt\b", "analyst", "bearish"),
    (r"\binitiate.*\bcoverage\b", "analyst", "neutral"),
    (r"\bcompetitor\b|rival\b", "competitive", "mixed"),
    (r"\bsector\s+rotation\b", "sector", "mixed"),
    (r"\binsider\s+buy", "insider", "bullish"),
    (r"\binsider\s+sell", "insider", "bearish"),
    (r"\bshort\s+interest\b", "short_interest", "mixed"),
    (r"\bsqueeze\b", "squeeze", "bullish"),
    (r"\bearnings\s+preview|reporting\s+after|before\s+the\s+bell", "earnings_preview", "mixed"),
    (r"\bipo\b|initial\s+public\s+offering", "ipo", "mixed"),
    (r"\bsecondary\s+offering|stock\s+offering|dilut", "dilution", "bearish"),
]

# Scheduled macro event times (ET)
MACRO_SCHEDULE = {
    "08:30": ["CPI", "PPI", "Jobs Report", "Retail Sales", "GDP"],
    "10:00": ["Consumer Sentiment", "ISM", "JOLTS", "New Home Sales"],
    "14:00": ["FOMC Decision", "FOMC Minutes", "Beige Book"],
    "14:30": ["FOMC Press Conference"],
    "16:00": ["Earnings After Hours"],
}


class NewsIntelligence:
    """Real-time news classification engine.

    Core loop:
    1. Receive raw news event (from WebSocket or polling).
    2. Instant keyword classification (< 1 ms, no API calls).
    3. If Tier 1 → flag for engine action + optionally deep-analyze via AI.
    4. If Tier 2 → log + adjust scanner weights.
    5. If Tier 3 → log only.

    The classification mimics an experienced day trader's mental model:
    years of pattern matching on headlines → instant "this matters / this doesn't".
    """

    def __init__(self, ai_advisor: Any | None = None) -> None:
        self._ai_advisor = ai_advisor
        self._event_history: list[NewsEvent] = []
        self._max_history = 200
        self._symbol_sentiment: dict[str, list[str]] = {}  # symbol → recent sentiments
        self._daily_catalyst_map: dict[str, str] = {}  # symbol → catalyst category

        # Pre-compile regex patterns for speed
        self._tier1_bullish = [re.compile(p, re.IGNORECASE) for p in _TIER1_BULLISH]
        self._tier1_bearish = [re.compile(p, re.IGNORECASE) for p in _TIER1_BEARISH]
        self._tier2_patterns = [
            (re.compile(p, re.IGNORECASE), cat, sent)
            for p, cat, sent in _TIER2_PATTERNS
        ]

    async def classify(self, raw_event: dict) -> NewsEvent:
        """Classify a raw news event dict from the stream.

        Args:
            raw_event: Dict with keys: headline, symbols, source, summary, url.

        Returns:
            Classified NewsEvent with tier, sentiment, impact, and action.
        """
        headline = raw_event.get("headline", "")
        summary = raw_event.get("summary", "")
        symbols = raw_event.get("symbols", [])
        source = raw_event.get("source", "")
        text = f"{headline} {summary}".strip()

        event = NewsEvent(
            headline=headline,
            symbols=symbols,
            source=source,
            summary=summary,
            url=raw_event.get("url", ""),
            raw=raw_event,
        )

        # ── Tier 1 classification (critical) ──────────────────────────
        for pattern in self._tier1_bullish:
            if pattern.search(text):
                event.tier = NewsTier.TIER_1_CRITICAL
                event.sentiment = "bullish"
                event.category = self._extract_category(text)
                event.impact_score = self._estimate_impact(event)
                event.actionable = True
                event.suggested_action = "watch_for_entry"
                break

        if event.tier != NewsTier.TIER_1_CRITICAL:
            for pattern in self._tier1_bearish:
                if pattern.search(text):
                    event.tier = NewsTier.TIER_1_CRITICAL
                    event.sentiment = "bearish"
                    event.category = self._extract_category(text)
                    event.impact_score = self._estimate_impact(event)
                    event.actionable = True
                    event.suggested_action = "reduce_or_exit"
                    break

        # ── Tier 2 classification (important) ─────────────────────────
        if event.tier == NewsTier.TIER_3_BACKGROUND:
            for pattern, category, sentiment in self._tier2_patterns:
                if pattern.search(text):
                    event.tier = NewsTier.TIER_2_IMPORTANT
                    event.sentiment = sentiment
                    event.category = category
                    event.impact_score = self._estimate_impact(event)
                    break

        # ── Update tracking ───────────────────────────────────────────
        for sym in symbols:
            self._symbol_sentiment.setdefault(sym, [])
            self._symbol_sentiment[sym].append(event.sentiment)
            # Keep only last 10 sentiments
            self._symbol_sentiment[sym] = self._symbol_sentiment[sym][-10:]

            if event.tier == NewsTier.TIER_1_CRITICAL:
                self._daily_catalyst_map[sym] = event.category

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Log based on tier
        if event.tier == NewsTier.TIER_1_CRITICAL:
            logger.warning("🚨 NEWS T1: {}", event)
        elif event.tier == NewsTier.TIER_2_IMPORTANT:
            logger.info("📰 NEWS T2: {}", event)
        else:
            logger.debug("📄 NEWS T3: {}", event)

        return event

    def get_symbol_sentiment(self, symbol: str) -> str:
        """Return the net sentiment for a symbol based on recent news.

        Logic: count bullish vs bearish mentions in recent history.
        """
        sentiments = self._symbol_sentiment.get(symbol, [])
        if not sentiments:
            return "neutral"

        bullish = sentiments.count("bullish")
        bearish = sentiments.count("bearish")

        if bullish > bearish:
            return "bullish"
        elif bearish > bullish:
            return "bearish"
        return "neutral"

    def get_symbol_catalyst(self, symbol: str) -> str:
        """Return the active catalyst category for a symbol, if any."""
        return self._daily_catalyst_map.get(symbol, "")

    def has_active_catalyst(self, symbol: str) -> bool:
        """Return True if symbol has a Tier 1 catalyst today."""
        return symbol in self._daily_catalyst_map

    def get_recent_events(
        self, symbol: str | None = None, tier: NewsTier | None = None, limit: int = 10
    ) -> list[NewsEvent]:
        """Return recent events, optionally filtered by symbol and tier."""
        events = self._event_history
        if symbol:
            events = [e for e in events if symbol in e.symbols]
        if tier:
            events = [e for e in events if e.tier == tier]
        return events[-limit:]

    def get_actionable_events(self) -> list[NewsEvent]:
        """Return all events that should trigger engine action."""
        return [e for e in self._event_history if e.actionable]

    def is_earnings_day(self, symbol: str) -> bool:
        """Return True if symbol has earnings catalyst today."""
        cat = self._daily_catalyst_map.get(symbol, "")
        return cat in ("earnings", "earnings_preview")

    def should_avoid(self, symbol: str) -> tuple[bool, str]:
        """Return (True, reason) if the symbol should be avoided.

        Experienced traders avoid:
        - Trading INTO earnings (before the report)
        - Stocks with active SEC investigations
        - Stocks with recent FDA rejections (knife-catching)
        """
        cat = self._daily_catalyst_map.get(symbol, "")

        if cat == "earnings_preview":
            return True, "earnings report pending — avoid pre-announcement trading"
        if cat in ("fraud", "investigation"):
            return True, "active investigation — headline risk"

        # Check recent bearish Tier 1 events
        recent_bearish = [
            e for e in self._event_history
            if symbol in e.symbols
            and e.tier == NewsTier.TIER_1_CRITICAL
            and e.sentiment == "bearish"
        ]
        if len(recent_bearish) >= 2:
            return True, f"multiple bearish catalysts ({len(recent_bearish)} events)"

        return False, ""

    def reset_daily(self) -> None:
        """Reset daily state. Call at start of each trading day."""
        self._daily_catalyst_map.clear()
        self._symbol_sentiment.clear()
        self._event_history.clear()
        logger.debug("[NewsIntel] Daily state reset")

    # ── Internal helpers ─────────────────────────────────────────────────

    def _extract_category(self, text: str) -> str:
        """Extract the category from headline text."""
        text_lower = text.lower()
        if any(w in text_lower for w in ("earning", "revenue", "eps", "quarter")):
            return "earnings"
        if any(w in text_lower for w in ("fda", "approval", "clinical", "trial")):
            return "fda"
        if any(w in text_lower for w in ("acqui", "merger", "buyout", "takeover")):
            return "mna"
        if any(w in text_lower for w in ("analyst", "upgrade", "downgrade", "price target")):
            return "analyst"
        if any(w in text_lower for w in ("ceo", "resign", "fired", "executive")):
            return "management"
        if any(w in text_lower for w in ("recall", "defect", "safety")):
            return "recall"
        if any(w in text_lower for w in ("fraud", "sec", "investigation", "lawsuit")):
            return "legal"
        if any(w in text_lower for w in ("split", "dividend")):
            return "corporate_action"
        if any(w in text_lower for w in ("contract", "partnership", "deal")):
            return "deal"
        return "general"

    def _estimate_impact(self, event: NewsEvent) -> float:
        """Estimate the potential price impact (0-100).

        Based on years of pattern matching:
        - Earnings surprise: 30-80 impact
        - FDA binary: 50-100 impact
        - Analyst: 10-30 impact
        - M&A: 40-90 impact
        - Management change: 15-40 impact
        """
        base_impact = {
            "earnings": 60,
            "fda": 75,
            "mna": 70,
            "analyst": 20,
            "management": 25,
            "recall": 35,
            "legal": 45,
            "corporate_action": 30,
            "deal": 25,
            "general": 10,
        }.get(event.category, 10)

        # Tier 1 events are inherently higher impact
        if event.tier == NewsTier.TIER_1_CRITICAL:
            base_impact = max(base_impact, 40)

        # Source reliability multiplier
        reliable_sources = {"reuters", "bloomberg", "wsj", "cnbc", "sec.gov", "fda.gov"}
        source_lower = event.source.lower()
        if any(s in source_lower for s in reliable_sources):
            base_impact *= 1.2

        return min(base_impact, 100)
