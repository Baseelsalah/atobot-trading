"""Market Scanner — the eyes of an experienced day trader.

Replicates what a 7-10 year day trader does every morning and throughout
the session:

**Pre-Market (4:00 – 9:30 AM ET)**
1. Scan for gappers — stocks gapping up/down ≥ 2 % on volume.
2. Check relative volume — are shares trading 2× normal pre-market?
3. Note float / market-cap context (small-cap gapper vs mega-cap gap).
4. Identify catalysts — earnings, FDA, upgrades (via news module).
5. Build a ranked watchlist sorted by *edge score*.

**Intraday (9:30 AM – 4:00 PM ET)**
1. Continuously re-rank watchlist by momentum, relative strength vs SPY,
   volume-weighted move, and multi-TF confluence.
2. Detect *squeeze setups* (Bollinger squeeze + OBV divergence).
3. Track sector rotation in real-time (XLK, XLF, XLE, etc.).
4. Detect *risk-off* conditions (VIX, broad-market weakness).
5. Surface the top-N symbols to the engine for active trading.

**Everything is numeric** — no GPT calls needed for core scanning.
GPT is only used downstream (AI advisor) after the scanner surfaces a setup.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

import pandas as pd
from loguru import logger

from src.config.settings import Settings
from src.data import indicators
from src.exchange.base_client import BaseExchangeClient


# ── Data models ──────────────────────────────────────────────────────────────


class ScannerSignal(str, Enum):
    """Type of setup the scanner detected."""

    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    MOMENTUM_SURGE = "momentum_surge"
    VOLUME_SPIKE = "volume_spike"
    SQUEEZE_SETUP = "squeeze_setup"
    RELATIVE_STRENGTH = "relative_strength"
    SECTOR_ROTATION = "sector_rotation"
    REVERSAL = "reversal"
    BREAKOUT = "breakout"
    VWAP_RECLAIM = "vwap_reclaim"


@dataclass
class ScanResult:
    """One scanner hit — a symbol + the setup detected."""

    symbol: str
    signal: ScannerSignal
    edge_score: float  # 0-100, higher = stronger setup
    price: Decimal = Decimal("0")
    gap_percent: float = 0.0
    relative_volume: float = 0.0  # current vol / avg vol
    relative_strength: float = 0.0  # vs SPY
    atr: float = 0.0
    rsi: float = 50.0
    vwap_distance_pct: float = 0.0
    squeeze_width: float = 0.0  # Bollinger bandwidth %, lower = tighter
    sector: str = ""
    catalyst: str = ""
    timeframe_confluence: int = 0  # how many TFs agree (1-5)
    float_category: str = ""  # "low", "mid", "high", "mega"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_actionable(self) -> bool:
        """True if the edge score is above the minimum threshold."""
        return self.edge_score >= 40.0


@dataclass
class MarketContext:
    """Broad market conditions snapshot — updated every tick."""

    spy_price: Decimal = Decimal("0")
    spy_change_pct: float = 0.0
    qqq_change_pct: float = 0.0
    vix_level: float = 0.0
    market_trend: str = "neutral"  # "bullish", "bearish", "neutral", "choppy"
    breadth_ratio: float = 0.5  # advancers / (advancers + decliners)
    risk_off: bool = False
    leading_sector: str = ""
    lagging_sector: str = ""
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Sector ETFs ──────────────────────────────────────────────────────────────

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLP": "Staples",
    "XLY": "Discretionary",
    "XLU": "Utilities",
    "XLC": "Communication",
    "XLRE": "Real Estate",
    "XLB": "Materials",
}

# Map popular stocks to their sector ETF for correlation
STOCK_SECTORS: dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK",
    "GOOGL": "XLC", "GOOG": "XLC", "META": "XLC", "NFLX": "XLC",
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "NKE": "XLY",
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF",
    "XOM": "XLE", "CVX": "XLE", "SLB": "XLE", "OXY": "XLE",
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "ABBV": "XLV",
    "CAT": "XLI", "BA": "XLI", "HON": "XLI", "GE": "XLI",
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP",
}

FLOAT_THRESHOLDS = {
    "low": 20_000_000,        # < 20M shares
    "mid": 100_000_000,       # 20-100M
    "high": 500_000_000,      # 100-500M
    # above 500M = "mega"
}


# ── Main Scanner ─────────────────────────────────────────────────────────────


class MarketScanner:
    """Pre-market & intraday market scanner.

    Runs two modes:
    - **pre_market_scan()** — called once before market open
    - **intraday_scan()** — called every engine tick (lightweight)

    Both return ``list[ScanResult]`` sorted by ``edge_score`` descending.
    """

    def __init__(
        self,
        exchange: BaseExchangeClient,
        settings: Settings,
    ) -> None:
        self.exchange = exchange
        self.settings = settings

        # Configurable thresholds
        self.min_gap_pct: float = getattr(settings, "SCANNER_MIN_GAP_PCT", 2.0)
        self.min_relative_volume: float = getattr(settings, "SCANNER_MIN_RVOL", 1.5)
        self.min_edge_score: float = getattr(settings, "SCANNER_MIN_EDGE_SCORE", 40.0)
        self.max_results: int = getattr(settings, "SCANNER_MAX_RESULTS", 20)
        self.scan_universe: list[str] = getattr(
            settings, "SCANNER_UNIVERSE", []
        ) or self._default_universe()

        # State
        self._market_ctx = MarketContext()
        self._last_scan_results: list[ScanResult] = []
        self._prev_close_cache: dict[str, Decimal] = {}
        self._avg_volume_cache: dict[str, float] = {}
        self._sector_perf: dict[str, float] = {}
        self._last_full_scan: float = 0.0
        self._scan_interval: float = getattr(
            settings, "SCANNER_INTERVAL_SECONDS", 60.0
        )

    # ── Public API ────────────────────────────────────────────────────────

    async def pre_market_scan(self) -> list[ScanResult]:
        """Run a full pre-market scan: gappers, volume, relative strength.

        Call this once before market open (e.g. 9:00 AM ET).
        Returns top gappers sorted by edge score.
        """
        logger.info("🔍 Pre-market scan starting | universe={} symbols", len(self.scan_universe))
        results: list[ScanResult] = []

        # 1. Fetch previous-day closes and average volumes
        await self._load_baseline_data()

        # 2. Update broad market context (SPY, QQQ, VIX proxy)
        await self._update_market_context()

        # 3. Scan each symbol for gaps and volume
        tasks = [self._scan_symbol_premarket(sym) for sym in self.scan_universe]
        scan_outputs = await asyncio.gather(*tasks, return_exceptions=True)

        for output in scan_outputs:
            if isinstance(output, Exception):
                continue
            if output and isinstance(output, list):
                results.extend(output)
            elif output and isinstance(output, ScanResult):
                results.append(output)

        # 4. Score and rank
        results = self._rank_results(results)
        self._last_scan_results = results[:self.max_results]

        # 5. Update sector rotation
        self._compute_sector_rotation()

        logger.info(
            "🔍 Pre-market scan complete | {} hits (top: {})",
            len(results),
            [f"{r.symbol}({r.edge_score:.0f})" for r in results[:5]],
        )
        return self._last_scan_results

    async def intraday_scan(self) -> list[ScanResult]:
        """Lightweight intraday re-scan — called every engine tick.

        Performs a full re-scan only every ``_scan_interval`` seconds.
        Between scans, returns the cached results.
        """
        now = time.monotonic()
        if now - self._last_full_scan < self._scan_interval:
            return self._last_scan_results

        self._last_full_scan = now

        results: list[ScanResult] = []

        # Update market context
        await self._update_market_context()

        # Quick scan of our active symbols + top previous results
        scan_symbols = list(set(
            self.settings.SYMBOLS
            + [r.symbol for r in self._last_scan_results[:10]]
        ))

        tasks = [self._scan_symbol_intraday(sym) for sym in scan_symbols]
        scan_outputs = await asyncio.gather(*tasks, return_exceptions=True)

        for output in scan_outputs:
            if isinstance(output, Exception):
                continue
            if output and isinstance(output, list):
                results.extend(output)
            elif output and isinstance(output, ScanResult):
                results.append(output)

        results = self._rank_results(results)
        self._last_scan_results = results[:self.max_results]

        # Update sector rotation
        self._compute_sector_rotation()

        return self._last_scan_results

    def get_market_context(self) -> MarketContext:
        """Return the latest broad market snapshot."""
        return self._market_ctx

    def get_top_symbols(self, n: int = 5) -> list[str]:
        """Return the top-N symbols by edge score from the last scan."""
        return [r.symbol for r in self._last_scan_results[:n] if r.is_actionable]

    def get_symbol_score(self, symbol: str) -> float:
        """Return the edge score for a symbol, or 0 if not scanned."""
        for r in self._last_scan_results:
            if r.symbol == symbol:
                return r.edge_score
        return 0.0

    def get_symbol_scan(self, symbol: str) -> ScanResult | None:
        """Return the full scan result for a symbol."""
        for r in self._last_scan_results:
            if r.symbol == symbol:
                return r
        return None

    def is_risk_off(self) -> bool:
        """Return True if broad market conditions suggest caution."""
        return self._market_ctx.risk_off

    def get_sector_for_symbol(self, symbol: str) -> str:
        """Return sector ETF for a symbol, or empty string."""
        return STOCK_SECTORS.get(symbol, "")

    # ── Pre-market scanning ──────────────────────────────────────────────

    async def _load_baseline_data(self) -> None:
        """Load previous-day close prices and 20-day average volumes."""
        for symbol in self.scan_universe:
            try:
                bars = await self.exchange.get_klines(symbol, "1D", 21)
                if not bars or len(bars) < 2:
                    continue

                df = pd.DataFrame(bars)
                for col in ("open", "high", "low", "close", "volume"):
                    df[col] = df[col].astype(float)

                # Previous close = second-to-last daily bar
                self._prev_close_cache[symbol] = Decimal(str(df["close"].iloc[-2]))
                self._avg_volume_cache[symbol] = float(df["volume"].iloc[:-1].mean())

            except Exception as exc:
                logger.debug("Baseline data error for {}: {}", symbol, exc)

    async def _scan_symbol_premarket(self, symbol: str) -> list[ScanResult]:
        """Scan a single symbol in pre-market mode."""
        results: list[ScanResult] = []

        try:
            current_price = await self.exchange.get_ticker_price(symbol)
        except Exception:
            return results

        prev_close = self._prev_close_cache.get(symbol)
        if not prev_close or prev_close == 0:
            return results

        # Gap calculation
        gap_pct = float((current_price - prev_close) / prev_close * 100)

        # Get current session volume (approximate from latest bar)
        current_vol = 0.0
        try:
            bars = await self.exchange.get_klines(symbol, "5m", 10)
            if bars:
                df = pd.DataFrame(bars)
                df["volume"] = df["volume"].astype(float)
                current_vol = float(df["volume"].sum())
        except Exception:
            pass

        avg_vol = self._avg_volume_cache.get(symbol, 1.0)
        relative_vol = current_vol / max(avg_vol, 1.0)

        # Calculate edge score components
        gap_score = min(abs(gap_pct) * 5, 30)  # max 30 points from gap
        vol_score = min(relative_vol * 10, 25)  # max 25 points from rvol
        # Prefer gaps with volume confirmation
        confluence_bonus = 10 if abs(gap_pct) >= self.min_gap_pct and relative_vol >= self.min_relative_volume else 0

        edge_score = gap_score + vol_score + confluence_bonus

        # Determine signal type
        if abs(gap_pct) >= self.min_gap_pct:
            signal = ScannerSignal.GAP_UP if gap_pct > 0 else ScannerSignal.GAP_DOWN
        elif relative_vol >= self.min_relative_volume * 2:
            signal = ScannerSignal.VOLUME_SPIKE
        else:
            signal = ScannerSignal.MOMENTUM_SURGE

        result = ScanResult(
            symbol=symbol,
            signal=signal,
            edge_score=edge_score,
            price=current_price,
            gap_percent=gap_pct,
            relative_volume=relative_vol,
            sector=STOCK_SECTORS.get(symbol, ""),
        )

        if result.is_actionable:
            results.append(result)

        return results

    # ── Intraday scanning ────────────────────────────────────────────────

    async def _scan_symbol_intraday(self, symbol: str) -> list[ScanResult]:
        """Full intraday scan for a single symbol — multiple setups."""
        results: list[ScanResult] = []

        try:
            bars_5m = await self.exchange.get_klines(symbol, "5m", 50)
        except Exception:
            return results

        if not bars_5m or len(bars_5m) < 35:
            return results

        df = pd.DataFrame(bars_5m)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)

        current_price = Decimal(str(df["close"].iloc[-1]))
        prev_close = self._prev_close_cache.get(symbol)
        day_change_pct = 0.0
        if prev_close and prev_close > 0:
            day_change_pct = float((current_price - prev_close) / prev_close * 100)

        # ── Indicators ────────────────────────────────────────────────
        try:
            rsi_val = float(indicators.rsi(df, 14).iloc[-1])
        except Exception:
            rsi_val = 50.0

        try:
            macd_info = indicators.macd_signal(df)
        except Exception:
            macd_info = {"bullish": False, "golden_cross": False, "histogram": 0}

        try:
            atr_val = float(indicators.atr(df, 14).iloc[-1])
        except Exception:
            atr_val = 0.0

        try:
            vwap_val = float(indicators.vwap(df).iloc[-1])
            vwap_dist_pct = ((float(current_price) - vwap_val) / vwap_val * 100) if vwap_val else 0
        except Exception:
            vwap_val = 0.0
            vwap_dist_pct = 0.0

        # Relative volume
        avg_vol = self._avg_volume_cache.get(symbol, 1.0)
        recent_vol = float(df["volume"].iloc[-5:].sum())
        avg_5bar_vol = avg_vol / (78 / 5)  # ~78 five-min bars in a session, take 5-bar slice
        relative_vol = recent_vol / max(avg_5bar_vol, 1.0)

        # Bollinger squeeze detection
        try:
            upper, middle, lower = indicators.bollinger_bands(df, 20, 2.0)
            bb_width = float((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1] * 100)
        except Exception:
            bb_width = 999.0

        # Relative strength vs SPY
        rs_vs_spy = self._compute_relative_strength(symbol, day_change_pct)

        # Multi-timeframe confluence
        tf_confluence = self._compute_tf_confluence(df, macd_info, rsi_val)

        # ── Setup detection ───────────────────────────────────────────

        # 1. Momentum surge
        if abs(day_change_pct) >= 1.5 and relative_vol >= 1.5:
            edge = self._score_momentum(day_change_pct, relative_vol, rsi_val, macd_info, tf_confluence, rs_vs_spy)
            results.append(ScanResult(
                symbol=symbol,
                signal=ScannerSignal.MOMENTUM_SURGE,
                edge_score=edge,
                price=current_price,
                gap_percent=day_change_pct,
                relative_volume=relative_vol,
                relative_strength=rs_vs_spy,
                atr=atr_val,
                rsi=rsi_val,
                vwap_distance_pct=vwap_dist_pct,
                squeeze_width=bb_width,
                sector=STOCK_SECTORS.get(symbol, ""),
                timeframe_confluence=tf_confluence,
            ))

        # 2. Squeeze setup (Bollinger bandwidth contracting)
        if bb_width < 2.0 and relative_vol >= 1.2:
            edge = self._score_squeeze(bb_width, relative_vol, rsi_val, macd_info, tf_confluence)
            results.append(ScanResult(
                symbol=symbol,
                signal=ScannerSignal.SQUEEZE_SETUP,
                edge_score=edge,
                price=current_price,
                relative_volume=relative_vol,
                rsi=rsi_val,
                squeeze_width=bb_width,
                sector=STOCK_SECTORS.get(symbol, ""),
                timeframe_confluence=tf_confluence,
            ))

        # 3. VWAP reclaim (price crossing back above VWAP with volume)
        if (
            vwap_dist_pct > -0.1
            and vwap_dist_pct < 0.3
            and relative_vol >= 1.3
            and macd_info.get("bullish", False)
        ):
            edge = self._score_vwap_reclaim(vwap_dist_pct, relative_vol, rsi_val, macd_info, tf_confluence)
            results.append(ScanResult(
                symbol=symbol,
                signal=ScannerSignal.VWAP_RECLAIM,
                edge_score=edge,
                price=current_price,
                relative_volume=relative_vol,
                rsi=rsi_val,
                vwap_distance_pct=vwap_dist_pct,
                sector=STOCK_SECTORS.get(symbol, ""),
                timeframe_confluence=tf_confluence,
            ))

        # 4. Relative strength leader
        if rs_vs_spy > 1.0 and day_change_pct > 0.5 and relative_vol >= 1.0:
            edge = 30 + min(rs_vs_spy * 10, 20) + min(relative_vol * 5, 15) + tf_confluence * 5
            results.append(ScanResult(
                symbol=symbol,
                signal=ScannerSignal.RELATIVE_STRENGTH,
                edge_score=min(edge, 100),
                price=current_price,
                gap_percent=day_change_pct,
                relative_volume=relative_vol,
                relative_strength=rs_vs_spy,
                rsi=rsi_val,
                sector=STOCK_SECTORS.get(symbol, ""),
                timeframe_confluence=tf_confluence,
            ))

        # 5. Reversal (oversold bounce)
        if rsi_val < 30 and macd_info.get("golden_cross", False) and relative_vol >= 1.2:
            edge = 35 + (30 - rsi_val) * 0.5 + min(relative_vol * 5, 15) + tf_confluence * 5
            results.append(ScanResult(
                symbol=symbol,
                signal=ScannerSignal.REVERSAL,
                edge_score=min(edge, 100),
                price=current_price,
                rsi=rsi_val,
                relative_volume=relative_vol,
                sector=STOCK_SECTORS.get(symbol, ""),
                timeframe_confluence=tf_confluence,
            ))

        # 6. Breakout (new intraday high with volume)
        intraday_high = float(df["high"].max())
        if (
            float(current_price) >= intraday_high * 0.998  # within 0.2% of high
            and relative_vol >= 1.5
            and day_change_pct > 0.5
        ):
            edge = 30 + min(day_change_pct * 3, 15) + min(relative_vol * 5, 20) + tf_confluence * 5
            if macd_info.get("bullish"):
                edge += 10
            results.append(ScanResult(
                symbol=symbol,
                signal=ScannerSignal.BREAKOUT,
                edge_score=min(edge, 100),
                price=current_price,
                gap_percent=day_change_pct,
                relative_volume=relative_vol,
                rsi=rsi_val,
                sector=STOCK_SECTORS.get(symbol, ""),
                timeframe_confluence=tf_confluence,
            ))

        return results

    # ── Market context ───────────────────────────────────────────────────

    async def _update_market_context(self) -> None:
        """Fetch SPY, QQQ prices and compute broad market state."""
        ctx = self._market_ctx

        try:
            spy_price = await self.exchange.get_ticker_price("SPY")
            ctx.spy_price = spy_price
        except Exception:
            pass

        # SPY change
        spy_prev = self._prev_close_cache.get("SPY")
        if spy_prev and spy_prev > 0:
            ctx.spy_change_pct = float((ctx.spy_price - spy_prev) / spy_prev * 100)

        # QQQ change
        try:
            qqq_price = await self.exchange.get_ticker_price("QQQ")
            qqq_prev = self._prev_close_cache.get("QQQ")
            if qqq_prev and qqq_prev > 0:
                ctx.qqq_change_pct = float((qqq_price - qqq_prev) / qqq_prev * 100)
        except Exception:
            pass

        # VIX proxy — use UVXY or VXX if VIX not directly available
        try:
            for vix_proxy in ("UVXY", "VXX", "VIXY"):
                try:
                    vix_price = await self.exchange.get_ticker_price(vix_proxy)
                    vix_prev = self._prev_close_cache.get(vix_proxy)
                    if vix_prev and vix_prev > 0:
                        vix_change = float((vix_price - vix_prev) / vix_prev * 100)
                        ctx.vix_level = vix_change  # store as % change
                    break
                except Exception:
                    continue
        except Exception:
            pass

        # Determine market trend
        if ctx.spy_change_pct > 0.5 and ctx.qqq_change_pct > 0.5:
            ctx.market_trend = "bullish"
        elif ctx.spy_change_pct < -0.5 and ctx.qqq_change_pct < -0.5:
            ctx.market_trend = "bearish"
        elif abs(ctx.spy_change_pct) < 0.2 and abs(ctx.qqq_change_pct) < 0.2:
            ctx.market_trend = "choppy"
        else:
            ctx.market_trend = "neutral"

        # Risk-off detection: big down day or VIX spike
        ctx.risk_off = (
            ctx.spy_change_pct < -1.5
            or ctx.vix_level > 10  # VIX proxy up > 10%
            or (ctx.spy_change_pct < -0.8 and ctx.qqq_change_pct < -0.8)
        )

        ctx.updated_at = datetime.now(timezone.utc)

        logger.debug(
            "Market context: SPY={:.2f}% QQQ={:.2f}% trend={} risk_off={}",
            ctx.spy_change_pct, ctx.qqq_change_pct, ctx.market_trend, ctx.risk_off,
        )

    # ── Scoring functions (the "experience") ─────────────────────────────

    def _score_momentum(
        self,
        day_change_pct: float,
        relative_vol: float,
        rsi: float,
        macd_info: dict,
        tf_confluence: int,
        rs_vs_spy: float,
    ) -> float:
        """Score a momentum setup like a veteran trader.

        Experienced traders weight:
        - Strong move WITH volume (not a low-vol drift) = high conviction
        - MACD confirming direction = momentum is real
        - RSI not already exhausted (>85 = chasing)
        - Multi-TF agreement = institutional interest
        - Relative strength vs SPY = stock-specific catalyst, not just market
        """
        score = 0.0

        # Day change: 1.5-5% is sweet spot, above 5% = chasing risk
        if 1.5 <= abs(day_change_pct) <= 5.0:
            score += min(abs(day_change_pct) * 5, 20)
        elif abs(day_change_pct) > 5.0:
            score += 15  # still decent but penalize for chasing

        # Volume confirmation (most important factor for veterans)
        if relative_vol >= 3.0:
            score += 25  # massive volume = institutional
        elif relative_vol >= 2.0:
            score += 20
        elif relative_vol >= 1.5:
            score += 12
        else:
            score += 5

        # MACD direction
        if macd_info.get("bullish") and day_change_pct > 0:
            score += 10
        elif macd_info.get("golden_cross"):
            score += 15  # fresh crossover = strongest

        # RSI awareness: experienced traders avoid chasing overbought
        if 30 < rsi < 60:
            score += 10  # ideal entry zone
        elif 60 <= rsi < 75:
            score += 5
        elif rsi >= 75:
            score -= 5  # overextended — experienced trader waits for pullback

        # Multi-TF confluence (big edge for experienced traders)
        score += tf_confluence * 4

        # Relative strength vs SPY
        if rs_vs_spy > 1.5:
            score += 10  # significantly leading the market
        elif rs_vs_spy > 0.5:
            score += 5

        # Risk-off penalty: experienced trader reduces size / skips in risk-off
        if self._market_ctx.risk_off:
            score *= 0.6

        return min(max(score, 0), 100)

    def _score_squeeze(
        self,
        bb_width: float,
        relative_vol: float,
        rsi: float,
        macd_info: dict,
        tf_confluence: int,
    ) -> float:
        """Score a Bollinger squeeze setup.

        Veteran traders love squeezes because they predict explosive moves.
        Tighter squeeze + volume starting to pick up = imminent breakout.
        """
        score = 0.0

        # Tighter squeeze = more potential energy
        if bb_width < 1.0:
            score += 30  # very tight — big move coming
        elif bb_width < 1.5:
            score += 20
        elif bb_width < 2.0:
            score += 10

        # Volume picking up during squeeze = early sign of breakout
        if relative_vol >= 2.0:
            score += 20
        elif relative_vol >= 1.5:
            score += 12
        elif relative_vol >= 1.2:
            score += 5

        # MACD starting to turn = direction hint
        if macd_info.get("golden_cross"):
            score += 15
        elif macd_info.get("bullish"):
            score += 8

        # RSI neutral = best for squeeze (not overbought or oversold)
        if 40 < rsi < 60:
            score += 10

        score += tf_confluence * 3

        if self._market_ctx.risk_off:
            score *= 0.5

        return min(max(score, 0), 100)

    def _score_vwap_reclaim(
        self,
        vwap_dist_pct: float,
        relative_vol: float,
        rsi: float,
        macd_info: dict,
        tf_confluence: int,
    ) -> float:
        """Score a VWAP reclaim setup.

        VWAP reclaim is a bread-and-butter day trade setup:
        Stock falls below VWAP → reclaims with volume → continuation long.
        """
        score = 25.0  # base score for VWAP reclaim pattern

        # Closer to VWAP = cleaner setup
        if abs(vwap_dist_pct) < 0.1:
            score += 15
        elif abs(vwap_dist_pct) < 0.2:
            score += 10

        # Volume on reclaim = institutional buying
        if relative_vol >= 2.0:
            score += 15
        elif relative_vol >= 1.5:
            score += 10

        # MACD confirming
        if macd_info.get("golden_cross"):
            score += 10
        elif macd_info.get("bullish"):
            score += 5

        # RSI not exhausted
        if 35 < rsi < 65:
            score += 5

        score += tf_confluence * 3

        if self._market_ctx.risk_off:
            score *= 0.7

        return min(max(score, 0), 100)

    # ── Multi-timeframe confluence ───────────────────────────────────────

    def _compute_tf_confluence(self, df: pd.DataFrame, macd_info: dict, rsi: float) -> int:
        """Count how many signals agree across 'timeframes'.

        A 7-10 year day trader checks: 1m, 5m, 15m, daily alignment.
        We approximate from 5-min bars by using different look-backs.

        Returns 0-5 based on how many factors agree.
        """
        count = 0

        # 1. Short-term momentum (last 5 bars ~25 min) — close > open
        if len(df) >= 5:
            short_bullish = float(df["close"].iloc[-1]) > float(df["open"].iloc[-5])
            if short_bullish:
                count += 1

        # 2. Mid-term trend (last 20 bars ~100 min) — price above 20-bar EMA
        if len(df) >= 20:
            try:
                ema20 = float(indicators.ema(df, 20).iloc[-1])
                if float(df["close"].iloc[-1]) > ema20:
                    count += 1
            except Exception:
                pass

        # 3. MACD bullish
        if macd_info.get("bullish"):
            count += 1

        # 4. RSI in healthy range (30-70, not exhausted)
        if 30 < rsi < 70:
            count += 1

        # 5. Volume expansion (last bar > avg of previous 10)
        if len(df) >= 12:
            last_vol = float(df["volume"].iloc[-1])
            avg_vol = float(df["volume"].iloc[-11:-1].mean())
            if avg_vol > 0 and last_vol > avg_vol * 1.3:
                count += 1

        return count

    def _compute_relative_strength(self, symbol: str, symbol_change_pct: float) -> float:
        """Compute relative strength vs SPY.

        RS > 0 = outperforming SPY
        RS > 1 = significantly outperforming
        """
        spy_change = self._market_ctx.spy_change_pct
        if abs(spy_change) < 0.01:
            return symbol_change_pct  # SPY flat, just return absolute change
        return symbol_change_pct - spy_change

    # ── Sector rotation ──────────────────────────────────────────────────

    def _compute_sector_rotation(self) -> None:
        """Tag which sectors are leading/lagging."""
        if not self._last_scan_results:
            return

        sector_changes: dict[str, list[float]] = {}
        for r in self._last_scan_results:
            if r.sector:
                sector_changes.setdefault(r.sector, []).append(r.gap_percent)

        self._sector_perf = {
            sector: sum(changes) / len(changes)
            for sector, changes in sector_changes.items()
            if changes
        }

        if self._sector_perf:
            best = max(self._sector_perf, key=self._sector_perf.get)  # type: ignore[arg-type]
            worst = min(self._sector_perf, key=self._sector_perf.get)  # type: ignore[arg-type]
            self._market_ctx.leading_sector = SECTOR_ETFS.get(best, best)
            self._market_ctx.lagging_sector = SECTOR_ETFS.get(worst, worst)

    # ── Ranking ──────────────────────────────────────────────────────────

    def _rank_results(self, results: list[ScanResult]) -> list[ScanResult]:
        """Sort results by edge score, de-duplicate per symbol (keep best)."""
        # De-dup: keep highest edge per symbol
        best: dict[str, ScanResult] = {}
        for r in results:
            if r.symbol not in best or r.edge_score > best[r.symbol].edge_score:
                best[r.symbol] = r

        ranked = sorted(best.values(), key=lambda r: r.edge_score, reverse=True)
        return ranked

    # ── Default universe ─────────────────────────────────────────────────

    @staticmethod
    def _default_universe() -> list[str]:
        """Return a broad default scan universe of liquid US equities.

        A veteran day trader doesn't scan the entire market — they focus on
        liquid, mid-to-large-cap names with options and institutional flow.
        """
        return [
            # Mega-cap tech (most liquid, easiest to day trade)
            "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL", "META",
            "NFLX", "AVGO", "CRM", "ADBE", "INTC", "MU", "QCOM",
            # Financials
            "JPM", "BAC", "GS", "MS", "C", "WFC",
            # Energy
            "XOM", "CVX", "SLB", "OXY",
            # Healthcare / Biotech (volatile, catalyst-driven)
            "JNJ", "UNH", "PFE", "ABBV", "MRNA", "BNTX",
            # Consumer / Retail
            "WMT", "HD", "NKE", "SBUX", "MCD",
            # Industrials / Transport
            "BA", "CAT", "GE", "UPS", "FDX",
            # ETFs for market context (not traded, used for RS calc)
            "SPY", "QQQ", "IWM", "DIA",
            # Volatility proxies
            "UVXY", "VXX",
            # Sector ETFs for rotation
            "XLK", "XLF", "XLE", "XLV", "XLI",
        ]
