"""Market Regime Detector — classify market state like a veteran trader.

An experienced day trader never trades in a vacuum.  Before placing a single
order they know:

1. **Trend regime** — Is SPY making higher highs (bullish), lower lows
   (bearish), or chopping?  This decides *direction bias*.
2. **Volatility regime** — Is VIX elevated (>25) or compressed (<15)?
   High vol = wider stops, smaller size.  Low vol = tighter setups.
3. **Breadth regime** — Are most stocks participating in the move or is it
   narrow leadership?  Narrow breadth = fragile rally.
4. **Momentum regime** — Is money flowing into risk-on (QQQ > SPY) or
   risk-off (TLT, GLD, XLU)?
5. **Time-of-day regime** — Open drive (9:30-10:00), mid-day chop
   (11:30-14:00), power hour (15:00-16:00).

This module reads the market's pulse through SPY/QQQ/VIX/IWM and outputs
an actionable regime that the engine uses to:
- Skip trades in choppy regimes
- Reduce size in high-volatility regimes
- Increase aggression in trending regimes
- Switch strategy weights based on regime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd
from loguru import logger


class TrendRegime(str, Enum):
    STRONG_BULL = "strong_bull"  # Clear uptrend, buy dips
    BULL = "bull"  # Uptrend but slower
    NEUTRAL = "neutral"  # No clear direction
    BEAR = "bear"  # Downtrend, short bias
    STRONG_BEAR = "strong_bear"  # Crash / liquidation mode
    CHOPPY = "choppy"  # Range-bound, fade extremes


class VolatilityRegime(str, Enum):
    LOW = "low"  # VIX < 15 — tight ranges, breakouts
    NORMAL = "normal"  # VIX 15-20 — standard conditions
    ELEVATED = "elevated"  # VIX 20-30 — careful sizing
    EXTREME = "extreme"  # VIX > 30 — crisis mode


class BreadthRegime(str, Enum):
    HEALTHY = "healthy"  # Broad participation
    NARROW = "narrow"  # Few leaders carrying
    DETERIORATING = "deteriorating"  # Breadth declining
    WEAK = "weak"  # Most stocks red


class SessionPhase(str, Enum):
    PRE_MARKET = "pre_market"  # Before 9:30 ET
    OPEN_DRIVE = "open_drive"  # 9:30-10:00 — high vol, trend moves
    MORNING_MOMENTUM = "morning_momentum"  # 10:00-11:30 — follow-through
    MIDDAY_CHOP = "midday_chop"  # 11:30-14:00 — avoid
    AFTERNOON_SETUP = "afternoon_setup"  # 14:00-15:00 — setups form
    POWER_HOUR = "power_hour"  # 15:00-16:00 — institutional flow
    AFTER_HOURS = "after_hours"  # After 16:00


@dataclass
class MarketRegime:
    """Complete market regime snapshot."""

    trend: TrendRegime = TrendRegime.NEUTRAL
    volatility: VolatilityRegime = VolatilityRegime.NORMAL
    breadth: BreadthRegime = BreadthRegime.HEALTHY
    session: SessionPhase = SessionPhase.PRE_MARKET

    # Numeric scores
    trend_score: float = 0.0  # -100 (strong bear) to +100 (strong bull)
    volatility_score: float = 0.0  # 0-100 (higher = more volatile)
    breadth_score: float = 0.0  # 0-100 (higher = healthier)

    # Derived recommendations
    size_multiplier: float = 1.0  # 0.0-1.5x position size multiplier
    aggression: float = 0.5  # 0-1, how aggressively to trade
    preferred_direction: str = "long"  # "long", "short", "both", "none"
    strategy_weights: dict = field(default_factory=dict)

    # Supporting data
    spy_trend_ema_slope: float = 0.0
    vix_level: float = 0.0
    spy_above_vwap: bool = True
    qqq_above_vwap: bool = True
    risk_on: bool = True

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_favorable(self) -> bool:
        """Return True if regime is favorable for trading."""
        if self.trend in (TrendRegime.CHOPPY,):
            return False
        if self.volatility == VolatilityRegime.EXTREME:
            return False
        if self.session == SessionPhase.MIDDAY_CHOP:
            return False
        if self.aggression < 0.2:
            return False
        return True

    def summary(self) -> str:
        """One-line regime summary."""
        return (
            f"Trend={self.trend.value} Vol={self.volatility.value} "
            f"Breadth={self.breadth.value} Session={self.session.value} "
            f"Size={self.size_multiplier:.1f}x Aggr={self.aggression:.0%}"
        )


class MarketRegimeDetector:
    """Detect and classify the current market regime.

    Uses SPY for trend, VIX for volatility, IWM vs SPY for breadth,
    and clock for session phase.

    The detector is designed to be called frequently (every 1-5 minutes)
    with fresh data passed in, and it maintains a history of regime changes.
    """

    def __init__(self, exchange: Any | None = None, settings: Any | None = None) -> None:
        self._exchange = exchange
        self._settings = settings
        self._regime_history: list[MarketRegime] = []
        self._max_history = 50
        self._current: MarketRegime = MarketRegime()

        # Thresholds (tuned from years of observation)
        self.vix_low = 15.0
        self.vix_normal = 20.0
        self.vix_elevated = 30.0
        self.trend_ema_fast = 9
        self.trend_ema_slow = 21
        self.choppy_atr_mult = 1.5  # ATR range / price — above this = choppy

    @property
    def current(self) -> MarketRegime:
        """Latest regime snapshot."""
        return self._current

    async def update(
        self,
        spy_bars: pd.DataFrame | None = None,
        vix_level: float | None = None,
        breadth_data: dict | None = None,
    ) -> MarketRegime:
        """Update the regime classification with fresh data.

        Args:
            spy_bars: OHLCV DataFrame for SPY (at least 30 bars of 5m data).
            vix_level: Current VIX value.
            breadth_data: Optional dict with 'advancers', 'decliners' counts.

        Returns:
            Updated MarketRegime.
        """
        regime = MarketRegime()

        # ── Session phase (always available) ──────────────────────────
        regime.session = self._detect_session_phase()

        # ── Trend regime ──────────────────────────────────────────────
        if spy_bars is not None and len(spy_bars) >= 20:
            regime.trend, regime.trend_score, regime.spy_trend_ema_slope = (
                self._detect_trend(spy_bars)
            )
            regime.spy_above_vwap = self._is_above_vwap(spy_bars)

        # ── Volatility regime ─────────────────────────────────────────
        if vix_level is not None:
            regime.vix_level = vix_level
            regime.volatility, regime.volatility_score = self._detect_volatility(vix_level)
        elif spy_bars is not None and len(spy_bars) >= 14:
            # Estimate from SPY intra-day volatility
            regime.volatility, regime.volatility_score = (
                self._estimate_volatility_from_bars(spy_bars)
            )

        # ── Breadth regime ────────────────────────────────────────────
        if breadth_data:
            regime.breadth, regime.breadth_score = self._detect_breadth(breadth_data)

        # ── Derived recommendations ───────────────────────────────────
        regime.size_multiplier = self._compute_size_multiplier(regime)
        regime.aggression = self._compute_aggression(regime)
        regime.preferred_direction = self._compute_direction(regime)
        regime.strategy_weights = self._compute_strategy_weights(regime)
        regime.risk_on = self._is_risk_on(regime)

        # ── Store ─────────────────────────────────────────────────────
        self._current = regime
        self._regime_history.append(regime)
        if len(self._regime_history) > self._max_history:
            self._regime_history = self._regime_history[-self._max_history:]

        # Log regime changes
        if len(self._regime_history) >= 2:
            prev = self._regime_history[-2]
            if prev.trend != regime.trend or prev.volatility != regime.volatility:
                logger.info("🔄 REGIME CHANGE: {}", regime.summary())
        else:
            logger.info("📊 REGIME: {}", regime.summary())

        return regime

    def get_session_phase(self) -> SessionPhase:
        """Return the current session phase without needing market data."""
        return self._detect_session_phase()

    def is_favorable(self) -> bool:
        """Return True if the current regime is favorable for trading."""
        return self._current.is_favorable()

    def get_size_multiplier(self) -> float:
        """Return the position size multiplier for current regime."""
        return self._current.size_multiplier

    def get_strategy_weights(self) -> dict:
        """Return strategy weight adjustments for current regime."""
        return self._current.strategy_weights

    def get_history(self) -> list[MarketRegime]:
        """Return regime history."""
        return list(self._regime_history)

    # ── Internal classification methods ──────────────────────────────────

    def _detect_session_phase(self) -> SessionPhase:
        """Classify time-of-day into session phases (ET)."""
        from datetime import timezone as tz
        import pytz

        try:
            et = pytz.timezone("US/Eastern")
            now = datetime.now(et)
        except ImportError:
            # Fallback: UTC-5 approximation
            from datetime import timedelta
            now = datetime.now(timezone.utc) - timedelta(hours=5)

        hour, minute = now.hour, now.minute
        time_val = hour * 60 + minute  # Minutes since midnight

        if time_val < 9 * 60 + 30:
            return SessionPhase.PRE_MARKET
        elif time_val < 10 * 60:
            return SessionPhase.OPEN_DRIVE
        elif time_val < 11 * 60 + 30:
            return SessionPhase.MORNING_MOMENTUM
        elif time_val < 14 * 60:
            return SessionPhase.MIDDAY_CHOP
        elif time_val < 15 * 60:
            return SessionPhase.AFTERNOON_SETUP
        elif time_val < 16 * 60:
            return SessionPhase.POWER_HOUR
        else:
            return SessionPhase.AFTER_HOURS

    def _detect_trend(
        self, bars: pd.DataFrame
    ) -> tuple[TrendRegime, float, float]:
        """Detect trend regime from SPY bars.

        Uses:
        - 9/21 EMA cross and slope
        - Higher highs / lower lows pattern
        - Price position relative to 20-bar SMA
        """
        close = bars["close"].astype(float)

        # EMAs
        ema_fast = close.ewm(span=self.trend_ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.trend_ema_slow, adjust=False).mean()

        # EMA slope (normalized)
        slope = (ema_fast.iloc[-1] - ema_fast.iloc[-5]) / ema_fast.iloc[-5] * 100
        ema_spread = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1] * 100

        # Price vs SMA
        sma_20 = close.rolling(20).mean()
        above_sma = close.iloc[-1] > sma_20.iloc[-1]

        # Higher highs / lower lows (last 10 bars)
        highs = bars["high"].astype(float).tail(10)
        lows = bars["low"].astype(float).tail(10)
        hh = highs.iloc[-1] > highs.iloc[-5] if len(highs) >= 5 else False
        ll = lows.iloc[-1] < lows.iloc[-5] if len(lows) >= 5 else False

        # Compute trend score (-100 to +100)
        score = 0.0
        score += ema_spread * 20  # EMA spread contribution
        score += slope * 10  # Slope contribution
        score += 15 if above_sma else -15  # Price vs SMA
        score += 10 if hh else -10 if ll else 0
        score = max(-100.0, min(100.0, score))

        # Classify
        if score > 60:
            trend = TrendRegime.STRONG_BULL
        elif score > 25:
            trend = TrendRegime.BULL
        elif score > -25:
            # Check for choppiness
            if abs(ema_spread) < 0.05 and abs(slope) < 0.05:
                trend = TrendRegime.CHOPPY
            else:
                trend = TrendRegime.NEUTRAL
        elif score > -60:
            trend = TrendRegime.BEAR
        else:
            trend = TrendRegime.STRONG_BEAR

        return trend, score, slope

    def _is_above_vwap(self, bars: pd.DataFrame) -> bool:
        """Check if the latest price is above VWAP."""
        if "vwap" in bars.columns:
            return float(bars["close"].iloc[-1]) > float(bars["vwap"].iloc[-1])
        # Compute VWAP manually
        try:
            typical_price = (
                bars["high"].astype(float)
                + bars["low"].astype(float)
                + bars["close"].astype(float)
            ) / 3
            volume = bars["volume"].astype(float)
            cumulative_tp_vol = (typical_price * volume).cumsum()
            cumulative_vol = volume.cumsum()
            vwap = cumulative_tp_vol / cumulative_vol
            return float(bars["close"].iloc[-1]) > float(vwap.iloc[-1])
        except Exception:
            return True  # Default to True if we can't compute

    def _detect_volatility(
        self, vix: float
    ) -> tuple[VolatilityRegime, float]:
        """Classify volatility from VIX level."""
        # Normalize VIX to 0-100 score
        score = min(100.0, (vix / 50.0) * 100)

        if vix < self.vix_low:
            return VolatilityRegime.LOW, score
        elif vix < self.vix_normal:
            return VolatilityRegime.NORMAL, score
        elif vix < self.vix_elevated:
            return VolatilityRegime.ELEVATED, score
        else:
            return VolatilityRegime.EXTREME, score

    def _estimate_volatility_from_bars(
        self, bars: pd.DataFrame
    ) -> tuple[VolatilityRegime, float]:
        """Estimate volatility from bar ranges when VIX is unavailable."""
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)
        close = bars["close"].astype(float)

        # Average true range as % of price
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_pct = (tr.rolling(14).mean().iloc[-1] / close.iloc[-1]) * 100

        score = min(100.0, atr_pct * 50)

        if atr_pct < 0.3:
            return VolatilityRegime.LOW, score
        elif atr_pct < 0.6:
            return VolatilityRegime.NORMAL, score
        elif atr_pct < 1.2:
            return VolatilityRegime.ELEVATED, score
        else:
            return VolatilityRegime.EXTREME, score

    def _detect_breadth(
        self, breadth_data: dict
    ) -> tuple[BreadthRegime, float]:
        """Classify market breadth from advancer/decliner data."""
        advancers = breadth_data.get("advancers", 0)
        decliners = breadth_data.get("decliners", 0)
        total = advancers + decliners

        if total == 0:
            return BreadthRegime.HEALTHY, 50.0

        ratio = advancers / total
        score = ratio * 100

        if ratio > 0.65:
            return BreadthRegime.HEALTHY, score
        elif ratio > 0.50:
            return BreadthRegime.NARROW, score
        elif ratio > 0.35:
            return BreadthRegime.DETERIORATING, score
        else:
            return BreadthRegime.WEAK, score

    def _compute_size_multiplier(self, regime: MarketRegime) -> float:
        """Compute position size multiplier based on regime.

        Experienced logic:
        - Strong trend + low vol = full size or 1.25x
        - Choppy = 0.5x
        - High vol = reduce proportionally
        - Extreme vol = 0.25x or sit out
        - Midday chop session = 0.5x
        """
        mult = 1.0

        # Trend adjustment
        trend_mult = {
            TrendRegime.STRONG_BULL: 1.25,
            TrendRegime.BULL: 1.1,
            TrendRegime.NEUTRAL: 0.8,
            TrendRegime.CHOPPY: 0.5,
            TrendRegime.BEAR: 0.7,
            TrendRegime.STRONG_BEAR: 0.5,
        }
        mult *= trend_mult.get(regime.trend, 1.0)

        # Volatility adjustment
        vol_mult = {
            VolatilityRegime.LOW: 1.1,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.ELEVATED: 0.7,
            VolatilityRegime.EXTREME: 0.25,
        }
        mult *= vol_mult.get(regime.volatility, 1.0)

        # Session adjustment
        if regime.session == SessionPhase.MIDDAY_CHOP:
            mult *= 0.5
        elif regime.session == SessionPhase.OPEN_DRIVE:
            mult *= 1.1
        elif regime.session == SessionPhase.POWER_HOUR:
            mult *= 1.05

        # Breadth adjustment
        if regime.breadth == BreadthRegime.WEAK:
            mult *= 0.6
        elif regime.breadth == BreadthRegime.DETERIORATING:
            mult *= 0.8

        return round(max(0.1, min(1.5, mult)), 2)

    def _compute_aggression(self, regime: MarketRegime) -> float:
        """Compute aggression level (0-1).

        0 = don't trade at all
        0.5 = selective, high-quality setups only
        1.0 = take every valid setup
        """
        aggression = 0.5

        # Trend contribution
        trend_aggr = {
            TrendRegime.STRONG_BULL: 0.3,
            TrendRegime.BULL: 0.2,
            TrendRegime.NEUTRAL: 0.0,
            TrendRegime.CHOPPY: -0.3,
            TrendRegime.BEAR: -0.1,
            TrendRegime.STRONG_BEAR: -0.2,
        }
        aggression += trend_aggr.get(regime.trend, 0.0)

        # Volatility contribution
        vol_aggr = {
            VolatilityRegime.LOW: 0.1,
            VolatilityRegime.NORMAL: 0.05,
            VolatilityRegime.ELEVATED: -0.15,
            VolatilityRegime.EXTREME: -0.35,
        }
        aggression += vol_aggr.get(regime.volatility, 0.0)

        # Session contribution
        if regime.session == SessionPhase.OPEN_DRIVE:
            aggression += 0.1
        elif regime.session == SessionPhase.MIDDAY_CHOP:
            aggression -= 0.2

        return max(0.0, min(1.0, aggression))

    def _compute_direction(self, regime: MarketRegime) -> str:
        """Determine preferred direction bias."""
        if regime.trend in (TrendRegime.STRONG_BULL, TrendRegime.BULL):
            return "long"
        elif regime.trend in (TrendRegime.STRONG_BEAR, TrendRegime.BEAR):
            return "short"
        elif regime.trend == TrendRegime.CHOPPY:
            return "none"
        return "both"

    def _compute_strategy_weights(self, regime: MarketRegime) -> dict:
        """Suggest strategy weight adjustments based on regime.

        Returns dict like {"momentum": 1.2, "vwap_scalp": 0.8, "ema_pullback": 1.0}

        v4 update: ORB disabled (consistently negative). EMA Pullback added
        as trend-following complement to VWAP mean-reversion.
        """
        weights = {"momentum": 1.0, "vwap_scalp": 1.2, "orb": 0.3, "ema_pullback": 1.0}

        # Trending market favors momentum + EMA pullback
        if regime.trend in (TrendRegime.STRONG_BULL, TrendRegime.STRONG_BEAR):
            weights["momentum"] = 1.3
            weights["vwap_scalp"] = 0.9
            weights["orb"] = 0.3
            weights["ema_pullback"] = 1.4  # EMA pullback thrives in strong trends

        # Choppy market favors mean-reversion (VWAP)
        elif regime.trend == TrendRegime.CHOPPY:
            weights["momentum"] = 0.5
            weights["vwap_scalp"] = 1.4
            weights["orb"] = 0.2
            weights["ema_pullback"] = 0.6  # Pullbacks less reliable in chop

        # Bull trend favors EMA pullback
        elif regime.trend == TrendRegime.BULL:
            weights["ema_pullback"] = 1.2

        # Opening range breakout best at open (still lowered)
        if regime.session == SessionPhase.OPEN_DRIVE:
            weights["orb"] = min(weights["orb"] * 1.5, 0.6)
        elif regime.session == SessionPhase.MIDDAY_CHOP:
            weights["orb"] = 0.1

        # High vol favors momentum (bigger moves)
        if regime.volatility == VolatilityRegime.ELEVATED:
            weights["momentum"] = min(weights["momentum"] * 1.1, 1.5)
            weights["ema_pullback"] = min(weights["ema_pullback"] * 1.1, 1.5)

        return weights

    def _is_risk_on(self, regime: MarketRegime) -> bool:
        """Determine if the market is in risk-on mode."""
        if regime.trend in (TrendRegime.STRONG_BEAR,):
            return False
        if regime.volatility == VolatilityRegime.EXTREME:
            return False
        if regime.breadth == BreadthRegime.WEAK:
            return False
        return True
