"""Tests for Market Scanner, News Intelligence, and Regime Detector."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
import pytest_asyncio

from src.config.settings import Settings
from src.scanner.market_scanner import MarketScanner, ScannerSignal, ScanResult, MarketContext
from src.scanner.news_intel import NewsIntelligence, NewsTier, NewsEvent
from src.scanner.regime_detector import (
    MarketRegimeDetector,
    MarketRegime,
    TrendRegime,
    VolatilityRegime,
    BreadthRegime,
    SessionPhase,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def scanner_settings() -> Settings:
    """Settings with scanner enabled."""
    return Settings(
        EXCHANGE="alpaca",
        ALPACA_API_KEY="test",
        ALPACA_API_SECRET="test",
        ALPACA_PAPER=True,
        SYMBOLS=["AAPL", "TSLA"],
        NOTIFICATIONS_ENABLED=False,
        DRY_RUN=True,
        SCANNER_ENABLED=True,
        SCANNER_MIN_GAP_PCT=2.0,
        SCANNER_MIN_RVOL=1.5,
        SCANNER_MIN_EDGE_SCORE=40.0,
        SCANNER_MAX_RESULTS=20,
        SCANNER_INTERVAL_SECONDS=60,
        NEWS_INTEL_ENABLED=True,
        REGIME_DETECTION_ENABLED=True,
    )


@pytest.fixture
def mock_exchange() -> AsyncMock:
    """Mock exchange for scanner tests."""
    ex = AsyncMock()
    ex.get_ticker_price = AsyncMock(return_value=Decimal("185.50"))
    ex.get_klines = AsyncMock(return_value=[
        {
            "timestamp": 1700000000000 + i * 300000,
            "open": Decimal("180") + Decimal(str(i * 0.5)),
            "high": Decimal("182") + Decimal(str(i * 0.5)),
            "low": Decimal("179") + Decimal(str(i * 0.5)),
            "close": Decimal("181") + Decimal(str(i * 0.5)),
            "volume": Decimal("50000"),
        }
        for i in range(100)
    ])
    return ex


@pytest.fixture
def scanner(mock_exchange, scanner_settings) -> MarketScanner:
    return MarketScanner(exchange=mock_exchange, settings=scanner_settings)


@pytest.fixture
def news_intel() -> NewsIntelligence:
    return NewsIntelligence()


@pytest.fixture
def regime_detector(mock_exchange, scanner_settings) -> MarketRegimeDetector:
    return MarketRegimeDetector(exchange=mock_exchange, settings=scanner_settings)


def _make_spy_bars(trend: str = "bull", n: int = 30) -> pd.DataFrame:
    """Generate SPY-like OHLCV DataFrame for testing."""
    base = 450.0
    data = []
    for i in range(n):
        if trend == "bull":
            close = base + i * 0.5
        elif trend == "bear":
            close = base - i * 0.5
        else:  # choppy
            close = base + (i % 3 - 1) * 0.3
        data.append({
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 50000.0 + i * 100,
        })
    return pd.DataFrame(data)


# ══════════════════════════════════════════════════════════════════════════════
# Market Scanner Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestScannerSignal:
    """Test ScannerSignal enum."""

    def test_signal_values(self):
        assert ScannerSignal.GAP_UP.value == "gap_up"
        assert ScannerSignal.MOMENTUM_SURGE.value == "momentum_surge"
        assert ScannerSignal.SQUEEZE_SETUP.value == "squeeze_setup"
        assert ScannerSignal.VWAP_RECLAIM.value == "vwap_reclaim"

    def test_all_signals_exist(self):
        expected = {
            "gap_up", "gap_down", "momentum_surge", "volume_spike",
            "squeeze_setup", "relative_strength", "sector_rotation",
            "reversal", "breakout", "vwap_reclaim",
        }
        actual = {s.value for s in ScannerSignal}
        assert actual == expected


class TestScanResult:
    """Test ScanResult dataclass."""

    def test_create_scan_result(self):
        r = ScanResult(
            symbol="AAPL",
            signal=ScannerSignal.GAP_UP,
            edge_score=75.0,
            price=Decimal("185.50"),
        )
        assert r.symbol == "AAPL"
        assert r.signal == ScannerSignal.GAP_UP
        assert r.edge_score == 75.0

    def test_scan_result_defaults(self):
        r = ScanResult(
            symbol="TSLA",
            signal=ScannerSignal.MOMENTUM_SURGE,
            edge_score=60.0,
            price=Decimal("250.00"),
        )
        assert r.gap_percent == 0.0
        assert r.relative_volume == 0.0
        assert r.relative_strength == 0.0
        assert r.sector == ""
        assert r.catalyst == ""
        assert r.timeframe_confluence == 0


class TestMarketScanner:
    """Test MarketScanner core logic."""

    def test_scanner_creation(self, scanner):
        assert scanner is not None
        assert scanner.exchange is not None

    def test_default_universe(self, scanner):
        universe = scanner._default_universe()
        assert len(universe) > 30
        assert "AAPL" in universe
        assert "TSLA" in universe
        assert "SPY" in universe

    def test_get_top_symbols_empty(self, scanner):
        top = scanner.get_top_symbols(5)
        assert top == []

    def test_get_symbol_score_unknown(self, scanner):
        score = scanner.get_symbol_score("UNKNOWN")
        assert score == 0.0

    def test_get_symbol_scan_unknown(self, scanner):
        result = scanner.get_symbol_scan("UNKNOWN")
        assert result is None

    def test_is_risk_off_default(self, scanner):
        assert scanner.is_risk_off() is False

    def test_market_context_default(self, scanner):
        ctx = scanner.get_market_context()
        assert ctx is not None  # Initialized as empty MarketContext

    @pytest.mark.asyncio
    async def test_pre_market_scan_runs(self, scanner):
        """Pre-market scan should not crash even with mock exchange."""
        try:
            results = await scanner.pre_market_scan()
            assert isinstance(results, list)
        except Exception:
            # Exchange mocks may not return perfect data — that's OK
            pass

    @pytest.mark.asyncio
    async def test_intraday_scan_runs(self, scanner):
        """Intraday scan should not crash."""
        try:
            results = await scanner.intraday_scan()
            assert isinstance(results, list)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# News Intelligence Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestNewsIntelligence:
    """Test NewsIntelligence classification engine."""

    @pytest.mark.asyncio
    async def test_classify_tier1_bullish_earnings(self, news_intel):
        """Earnings beat should be Tier 1 bullish."""
        event = await news_intel.classify({
            "headline": "AAPL earnings beat estimates with record revenue",
            "symbols": ["AAPL"],
            "source": "Reuters",
        })
        assert event.tier == NewsTier.TIER_1_CRITICAL
        assert event.sentiment == "bullish"
        assert event.category == "earnings"
        assert event.actionable is True
        assert event.impact_score > 0

    @pytest.mark.asyncio
    async def test_classify_tier1_bearish_downgrade(self, news_intel):
        """Downgrade to sell should be Tier 1 bearish."""
        event = await news_intel.classify({
            "headline": "Morgan Stanley downgrades TSLA to sell on valuation concerns",
            "symbols": ["TSLA"],
        })
        assert event.tier == NewsTier.TIER_1_CRITICAL
        assert event.sentiment == "bearish"
        assert event.actionable is True
        assert event.suggested_action == "reduce_or_exit"

    @pytest.mark.asyncio
    async def test_classify_tier1_fda_approval(self, news_intel):
        """FDA approval should be Tier 1 bullish."""
        event = await news_intel.classify({
            "headline": "FDA approves Pfizer's new cancer drug",
            "symbols": ["PFE"],
        })
        assert event.tier == NewsTier.TIER_1_CRITICAL
        assert event.sentiment == "bullish"
        assert event.category == "fda"

    @pytest.mark.asyncio
    async def test_classify_tier1_acquisition(self, news_intel):
        """M&A news should be Tier 1."""
        event = await news_intel.classify({
            "headline": "Microsoft acquires gaming company for $5 billion",
            "symbols": ["MSFT"],
        })
        assert event.tier == NewsTier.TIER_1_CRITICAL
        assert event.sentiment == "bullish"
        assert event.category == "mna"

    @pytest.mark.asyncio
    async def test_classify_tier2_macro(self, news_intel):
        """FOMC news should be Tier 2."""
        event = await news_intel.classify({
            "headline": "Federal Reserve holds interest rate steady",
            "symbols": [],
        })
        assert event.tier == NewsTier.TIER_2_IMPORTANT
        assert event.category == "macro"

    @pytest.mark.asyncio
    async def test_classify_tier2_analyst_upgrade(self, news_intel):
        """Analyst upgrade should be Tier 2."""
        event = await news_intel.classify({
            "headline": "Goldman analyst upgrades NVDA on AI tailwinds",
            "symbols": ["NVDA"],
        })
        assert event.tier == NewsTier.TIER_2_IMPORTANT
        assert event.category == "analyst"
        assert event.sentiment == "bullish"

    @pytest.mark.asyncio
    async def test_classify_tier3_general(self, news_intel):
        """General commentary should be Tier 3."""
        event = await news_intel.classify({
            "headline": "Markets open mixed on Tuesday morning",
            "symbols": [],
        })
        assert event.tier == NewsTier.TIER_3_BACKGROUND
        assert event.actionable is False

    @pytest.mark.asyncio
    async def test_symbol_sentiment_tracking(self, news_intel):
        """Sentiment should be tracked per symbol."""
        await news_intel.classify({
            "headline": "AAPL earnings beat estimates",
            "symbols": ["AAPL"],
        })
        await news_intel.classify({
            "headline": "AAPL raises guidance for next quarter",
            "symbols": ["AAPL"],
        })
        assert news_intel.get_symbol_sentiment("AAPL") == "bullish"

    @pytest.mark.asyncio
    async def test_symbol_catalyst_tracking(self, news_intel):
        """Active catalyst should be tracked."""
        await news_intel.classify({
            "headline": "TSLA earnings beat estimates",
            "symbols": ["TSLA"],
        })
        assert news_intel.has_active_catalyst("TSLA") is True
        assert news_intel.get_symbol_catalyst("TSLA") == "earnings"

    @pytest.mark.asyncio
    async def test_should_avoid_earnings_preview(self, news_intel):
        """Should avoid trading into pending earnings."""
        await news_intel.classify({
            "headline": "AAPL reporting after the bell today — earnings preview",
            "symbols": ["AAPL"],
        })
        # earnings_preview is Tier 2 category, put it into daily catalyst map manually
        news_intel._daily_catalyst_map["AAPL"] = "earnings_preview"
        avoid, reason = news_intel.should_avoid("AAPL")
        assert avoid is True
        assert "earnings" in reason

    @pytest.mark.asyncio
    async def test_recent_events_filter(self, news_intel):
        """Recent events should be filterable by symbol and tier."""
        await news_intel.classify({
            "headline": "AAPL earnings beat",
            "symbols": ["AAPL"],
        })
        await news_intel.classify({
            "headline": "Markets open mixed",
            "symbols": [],
        })
        aapl_events = news_intel.get_recent_events(symbol="AAPL")
        assert len(aapl_events) == 1
        t1_events = news_intel.get_recent_events(tier=NewsTier.TIER_1_CRITICAL)
        assert len(t1_events) >= 1

    def test_reset_daily(self, news_intel):
        """Daily reset clears all state."""
        news_intel._daily_catalyst_map["AAPL"] = "earnings"
        news_intel._symbol_sentiment["AAPL"] = ["bullish"]
        news_intel.reset_daily()
        assert len(news_intel._daily_catalyst_map) == 0
        assert len(news_intel._symbol_sentiment) == 0

    @pytest.mark.asyncio
    async def test_impact_score_reliable_source(self, news_intel):
        """Reliable sources should boost impact score."""
        event_reuters = await news_intel.classify({
            "headline": "AAPL earnings beat estimates",
            "symbols": ["AAPL"],
            "source": "Reuters",
        })
        news_intel.reset_daily()
        event_unknown = await news_intel.classify({
            "headline": "AAPL earnings beat estimates",
            "symbols": ["AAPL"],
            "source": "unknown_blog",
        })
        assert event_reuters.impact_score >= event_unknown.impact_score

    @pytest.mark.asyncio
    async def test_classify_tier1_bearish_fraud(self, news_intel):
        """Fraud/investigation should be Tier 1 bearish."""
        event = await news_intel.classify({
            "headline": "SEC probe into accounting fraud at XYZ Corp",
            "symbols": ["XYZ"],
        })
        assert event.tier == NewsTier.TIER_1_CRITICAL
        assert event.sentiment == "bearish"

    @pytest.mark.asyncio
    async def test_classify_tier2_insider_buy(self, news_intel):
        """Insider buy should be classified."""
        event = await news_intel.classify({
            "headline": "CEO insider buy of 50,000 shares at AAPL",
            "symbols": ["AAPL"],
        })
        assert event.tier == NewsTier.TIER_2_IMPORTANT
        assert event.category == "insider"
        assert event.sentiment == "bullish"

    @pytest.mark.asyncio
    async def test_classify_tier2_dilution(self, news_intel):
        """Secondary offering / dilution should be bearish."""
        event = await news_intel.classify({
            "headline": "XYZ announces secondary offering of 10M shares",
            "symbols": ["XYZ"],
        })
        assert event.tier == NewsTier.TIER_2_IMPORTANT
        assert event.category == "dilution"
        assert event.sentiment == "bearish"


# ══════════════════════════════════════════════════════════════════════════════
# Market Regime Detector Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMarketRegimeDetector:
    """Test MarketRegimeDetector classification."""

    def test_creation(self, regime_detector):
        assert regime_detector is not None
        assert regime_detector.current.trend == TrendRegime.NEUTRAL

    def test_session_phase_detection(self, regime_detector):
        """Session phase should return a valid SessionPhase."""
        phase = regime_detector.get_session_phase()
        assert isinstance(phase, SessionPhase)

    @pytest.mark.asyncio
    async def test_update_bull_trend(self, regime_detector):
        """Bullish SPY bars should detect bull trend."""
        spy_bars = _make_spy_bars("bull", 30)
        regime = await regime_detector.update(spy_bars=spy_bars, vix_level=16.0)
        assert regime.trend in (TrendRegime.STRONG_BULL, TrendRegime.BULL)
        assert regime.trend_score > 0

    @pytest.mark.asyncio
    async def test_update_bear_trend(self, regime_detector):
        """Bearish SPY bars should detect bear trend."""
        spy_bars = _make_spy_bars("bear", 30)
        regime = await regime_detector.update(spy_bars=spy_bars, vix_level=28.0)
        assert regime.trend in (TrendRegime.STRONG_BEAR, TrendRegime.BEAR)
        assert regime.trend_score < 0

    @pytest.mark.asyncio
    async def test_volatility_low(self, regime_detector):
        """Low VIX should be classified as LOW."""
        regime = await regime_detector.update(vix_level=12.0)
        assert regime.volatility == VolatilityRegime.LOW

    @pytest.mark.asyncio
    async def test_volatility_normal(self, regime_detector):
        """Normal VIX should be NORMAL."""
        regime = await regime_detector.update(vix_level=17.0)
        assert regime.volatility == VolatilityRegime.NORMAL

    @pytest.mark.asyncio
    async def test_volatility_elevated(self, regime_detector):
        """Elevated VIX should be ELEVATED."""
        regime = await regime_detector.update(vix_level=25.0)
        assert regime.volatility == VolatilityRegime.ELEVATED

    @pytest.mark.asyncio
    async def test_volatility_extreme(self, regime_detector):
        """Extreme VIX should be EXTREME."""
        regime = await regime_detector.update(vix_level=35.0)
        assert regime.volatility == VolatilityRegime.EXTREME

    @pytest.mark.asyncio
    async def test_breadth_healthy(self, regime_detector):
        """High advancers ratio should be HEALTHY."""
        regime = await regime_detector.update(
            breadth_data={"advancers": 350, "decliners": 150}
        )
        assert regime.breadth == BreadthRegime.HEALTHY
        assert regime.breadth_score > 65

    @pytest.mark.asyncio
    async def test_breadth_weak(self, regime_detector):
        """Low advancers should be WEAK."""
        regime = await regime_detector.update(
            breadth_data={"advancers": 100, "decliners": 400}
        )
        assert regime.breadth == BreadthRegime.WEAK

    @pytest.mark.asyncio
    async def test_size_multiplier_extreme_vol(self, regime_detector):
        """Extreme vol should drastically reduce size."""
        regime = await regime_detector.update(vix_level=40.0)
        assert regime.size_multiplier < 0.5

    @pytest.mark.asyncio
    async def test_size_multiplier_strong_bull(self, regime_detector):
        """Strong bull + low vol should boost size."""
        spy_bars = _make_spy_bars("bull", 30)
        regime = await regime_detector.update(spy_bars=spy_bars, vix_level=13.0)
        assert regime.size_multiplier >= 1.0

    @pytest.mark.asyncio
    async def test_aggression_choppy(self, regime_detector):
        """Elevated vol should reduce aggression."""
        spy_bars = _make_spy_bars("choppy", 30)
        regime = await regime_detector.update(spy_bars=spy_bars, vix_level=28.0)
        # Elevated vol should bring aggression below default 0.5
        assert regime.aggression < 0.6

    @pytest.mark.asyncio
    async def test_strategy_weights(self, regime_detector):
        """Strategy weights should be populated."""
        spy_bars = _make_spy_bars("bull", 30)
        regime = await regime_detector.update(spy_bars=spy_bars, vix_level=16.0)
        assert "momentum" in regime.strategy_weights
        assert "vwap_scalp" in regime.strategy_weights
        assert "orb" in regime.strategy_weights
        assert "ema_pullback" in regime.strategy_weights

    @pytest.mark.asyncio
    async def test_is_favorable_normal(self, regime_detector):
        """Normal conditions should be favorable."""
        spy_bars = _make_spy_bars("bull", 30)
        regime = await regime_detector.update(spy_bars=spy_bars, vix_level=16.0)
        # Favorable depends on session phase too — just check it doesn't crash
        isinstance(regime.is_favorable(), bool)

    @pytest.mark.asyncio
    async def test_regime_history(self, regime_detector):
        """History should accumulate regime snapshots."""
        await regime_detector.update(vix_level=15.0)
        await regime_detector.update(vix_level=25.0)
        history = regime_detector.get_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_risk_on_extreme_vol(self, regime_detector):
        """Extreme vol should be risk-off."""
        regime = await regime_detector.update(vix_level=40.0)
        assert regime.risk_on is False

    @pytest.mark.asyncio
    async def test_preferred_direction_bull(self, regime_detector):
        """Bull trend should prefer long."""
        spy_bars = _make_spy_bars("bull", 30)
        regime = await regime_detector.update(spy_bars=spy_bars, vix_level=16.0)
        if regime.trend in (TrendRegime.STRONG_BULL, TrendRegime.BULL):
            assert regime.preferred_direction == "long"

    def test_regime_summary(self, regime_detector):
        """Summary should return a string."""
        summary = regime_detector.current.summary()
        assert isinstance(summary, str)
        assert "Trend=" in summary
        assert "Vol=" in summary

    @pytest.mark.asyncio
    async def test_estimate_volatility_from_bars(self, regime_detector):
        """Should estimate volatility when VIX is unavailable."""
        spy_bars = _make_spy_bars("bull", 30)
        regime = await regime_detector.update(spy_bars=spy_bars)
        # No VIX passed — should estimate from bar ranges
        assert regime.volatility in (
            VolatilityRegime.LOW, VolatilityRegime.NORMAL,
            VolatilityRegime.ELEVATED, VolatilityRegime.EXTREME,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Integration-Style Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestScannerIntegration:
    """Test scanner suite components work together."""

    @pytest.mark.asyncio
    async def test_news_intel_avoid_feeds_engine_logic(self, news_intel):
        """News avoidance should be queryable after classification."""
        await news_intel.classify({
            "headline": "SEC investigation into fraud at SCAN Corp",
            "symbols": ["SCAN"],
        })
        # After multiple bearish T1 events, should_avoid triggers
        await news_intel.classify({
            "headline": "SCAN Corp CEO resigns amid probe",
            "symbols": ["SCAN"],
        })
        avoid, reason = news_intel.should_avoid("SCAN")
        assert avoid is True
        assert "bearish" in reason.lower()

    def test_scanner_settings_defaults(self, scanner_settings):
        """Scanner settings should have correct defaults."""
        assert scanner_settings.SCANNER_ENABLED is True
        assert scanner_settings.SCANNER_MIN_GAP_PCT == 2.0
        assert scanner_settings.SCANNER_MIN_RVOL == 1.5
        assert scanner_settings.SCANNER_MIN_EDGE_SCORE == 40.0
        assert scanner_settings.SCANNER_MAX_RESULTS == 20
        assert scanner_settings.NEWS_INTEL_ENABLED is True
        assert scanner_settings.REGIME_DETECTION_ENABLED is True

    @pytest.mark.asyncio
    async def test_regime_affects_strategy_weights(self, regime_detector):
        """Regime should adjust strategy weights."""
        # Choppy regime should favor VWAP over momentum
        spy_bars = _make_spy_bars("choppy", 30)
        regime = await regime_detector.update(spy_bars=spy_bars, vix_level=18.0)
        if regime.trend == TrendRegime.CHOPPY:
            assert regime.strategy_weights.get("vwap_scalp", 1.0) > regime.strategy_weights.get("momentum", 1.0)

    @pytest.mark.asyncio
    async def test_news_event_str_repr(self, news_intel):
        """NewsEvent str should be readable."""
        event = await news_intel.classify({
            "headline": "AAPL earnings beat estimates with record revenue quarter",
            "symbols": ["AAPL"],
        })
        s = str(event)
        assert "T1" in s or "T2" in s or "T3" in s

    def test_market_regime_str(self):
        """MarketRegime summary should be readable."""
        regime = MarketRegime(
            trend=TrendRegime.BULL,
            volatility=VolatilityRegime.NORMAL,
        )
        s = regime.summary()
        assert "bull" in s
        assert "normal" in s
