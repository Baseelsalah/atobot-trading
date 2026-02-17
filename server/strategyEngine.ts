/**
 * P5 Deterministic Strategy Engine
 * 
 * Generates trade signals from deterministic strategies (no LLM dependency).
 * All trades are tagged with strategyName for A/B evaluation in weekly scorecards.
 */

import * as alpaca from "./alpaca";
import { ema, rsi, atr, vwap, bollingerBands } from "./indicators";
import { getEasternTime } from "./timezone";
import { Tier1Indicators, Tier2Indicators, hasMacd } from "./indicatorPipeline";

// Strategy names (constants for tagging)
export const STRATEGY_NAMES = {
  VWAP_REVERSION: "VWAP_REVERSION",
  ORB: "ORB",
  EMA_CROSSOVER: "EMA_CROSSOVER",
  NO_SIGNAL: "NO_SIGNAL",
} as const;

export type StrategyName = typeof STRATEGY_NAMES[keyof typeof STRATEGY_NAMES];

// Signal side
export type SignalSide = "buy" | "sell" | "none";

// Strategy signal output
export interface StrategySignal {
  strategyName: StrategyName;
  symbol: string;
  side: SignalSide;
  confidence: number;           // 0-100
  entrySignalPrice: number;
  reason: string;
  invalidation: number | null;  // Price level where signal is invalid
  metadata?: {
    vwapDeviation?: number;
    rsi?: number;
    macdLine?: number;
    macdSignal?: number;
    macdConfirm?: boolean;
    volumeSurge?: boolean;
    orbHigh?: number;
    orbLow?: number;
    orbBreakoutDirection?: "up" | "down";
    emaCrossDirection?: "bullish" | "bearish";
    ema9?: number;
    ema20?: number;
    bollingerPosition?: string;
  };
}

// Strategy configuration
export interface StrategyConfig {
  enabled: boolean;
  weight: number;  // For A/B weighting (1.0 = normal, 0.5 = half weight)
}

// Strategy registry with enable/disable and weights
const strategyRegistry: Record<string, StrategyConfig> = {
  [STRATEGY_NAMES.VWAP_REVERSION]: { enabled: false, weight: 1.0 },  // Disabled: worst performer in 3-month backtest (PF 0.46)
  [STRATEGY_NAMES.ORB]: { enabled: false, weight: 1.0 },  // Disabled: net-negative over 6-month backtest (-$9K in chop). EMA_CROSSOVER is consistently profitable.
  [STRATEGY_NAMES.EMA_CROSSOVER]: { enabled: true, weight: 1.0 },
};

// Symbol data for strategy evaluation
export interface SymbolData {
  symbol: string;
  currentPrice: number;
  bid: number;
  ask: number;
  volume: number;
  indicators: Tier1Indicators;
  bars?: AlpacaBar[];  // Raw bars for VWAP/ORB calculation
}

interface AlpacaBar {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

// VWAP Reversion Strategy Configuration
const VWAP_CONFIG = {
  MIN_DEVIATION_PCT: 0.02,     // Minimum % deviation from VWAP to trigger
  MAX_DEVIATION_PCT: 4.0,      // Maximum % deviation (too extended = risky)
  RSI_OVERSOLD: 35,            // RSI threshold for buy signals (standard: below 35 = oversold)
  RSI_OVERBOUGHT: 65,          // RSI threshold for sell signals (standard: above 65 = overbought)
  MIN_VOLUME: 40000,           // Minimum volume requirement
  BASE_CONFIDENCE: 58,         // Base confidence score
  MAX_CONFIDENCE: 85,          // Maximum confidence score
};

// ORB Strategy Configuration
const ORB_CONFIG = {
  RANGE_MINUTES: 15,           // First 15 minutes define the opening range
  BREAKOUT_BUFFER_PCT: 0.005,  // Must break by this % to confirm
  TREND_ALIGNMENT_EMA: 20,     // Use EMA20 for trend alignment
  MIN_RANGE_PCT: 0.03,         // Minimum range size (% of price)
  MAX_RANGE_PCT: 3.5,          // Maximum range size (too wide = skip)
  BASE_CONFIDENCE: 55,         // Base confidence score (raised from 50 for better pass rate)
  MAX_CONFIDENCE: 85,          // Maximum confidence score
};

const ORB_TREND_TOLERANCE_PCT = 0.5; // Allow slight EMA misalignment for early-session breakouts

// EMA Crossover Strategy Configuration
const EMA_CROSS_CONFIG = {
  MIN_VOLUME: 50000,            // Minimum volume requirement
  RSI_MIN_BUY: 40,              // RSI must be above this for buy (not deeply oversold = weak)
  RSI_MAX_BUY: 65,              // RSI must be below this for buy (not already overbought)
  RSI_MIN_SELL: 35,             // RSI must be above this for sell (not already oversold)
  RSI_MAX_SELL: 60,             // RSI must be below this for sell (not deeply overbought = weak)
  MIN_EMA_SPREAD_PCT: 0.03,    // Minimum spread between EMA9 and EMA20 for valid cross
  BASE_CONFIDENCE: 55,          // Base confidence score
  MAX_CONFIDENCE: 85,           // Maximum confidence score
};

/**
 * Get strategy config (for A/B framework)
 */
export function getStrategyConfig(): Record<string, StrategyConfig> {
  return { ...strategyRegistry };
}

/**
 * Update strategy configuration (enable/disable, adjust weights)
 */
export function updateStrategyConfig(name: StrategyName, config: Partial<StrategyConfig>): void {
  if (strategyRegistry[name]) {
    strategyRegistry[name] = { ...strategyRegistry[name], ...config };
    console.log(`[StrategyEngine] Updated ${name}: enabled=${strategyRegistry[name].enabled}, weight=${strategyRegistry[name].weight}`);
  }
}

/**
 * Enable or disable a strategy
 */
export function setStrategyEnabled(name: StrategyName, enabled: boolean): void {
  updateStrategyConfig(name, { enabled });
}

/**
 * Get list of enabled strategy names
 */
export function getEnabledStrategies(): StrategyName[] {
  return Object.entries(strategyRegistry)
    .filter(([_, config]) => config.enabled)
    .map(([name, _]) => name as StrategyName);
}

/**
 * Calculate VWAP from bars
 */
function calculateVWAP(bars: AlpacaBar[]): number {
  if (!bars || bars.length === 0) return 0;
  
  const highs = bars.map(b => b.h);
  const lows = bars.map(b => b.l);
  const closes = bars.map(b => b.c);
  const volumes = bars.map(b => b.v);
  
  const vwapValues = vwap(highs, lows, closes, volumes);
  return vwapValues[vwapValues.length - 1] || 0;
}

/**
 * Calculate Opening Range (first N minutes of trading day)
 * Uses Intl.DateTimeFormat for correct ET conversion (handles EST/EDT automatically)
 */
function calculateOpeningRange(bars: AlpacaBar[], rangeMinutes: number = 15): { high: number; low: number; valid: boolean } {
  const marketOpenMinutes = 9 * 60 + 30; // 9:30 AM ET
  const rangeEndMinutes = marketOpenMinutes + rangeMinutes;

  // ET formatter for bar timestamps - handles EST/EDT automatically
  const etHourFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    hour: "numeric",
    hour12: false,
  });
  const etMinuteFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    minute: "numeric",
  });

  // Filter bars within the opening range (9:30 to 9:30 + rangeMinutes ET)
  const rangeBars = bars.filter(bar => {
    const barTime = new Date(bar.t);
    const barHourET = parseInt(etHourFormatter.format(barTime), 10);
    const barMinuteET = parseInt(etMinuteFormatter.format(barTime), 10);
    const barMinutesET = barHourET * 60 + barMinuteET;
    return barMinutesET >= marketOpenMinutes && barMinutesET < rangeEndMinutes;
  });

  if (rangeBars.length < 2) {
    return { high: 0, low: 0, valid: false };
  }

  const high = Math.max(...rangeBars.map(b => b.h));
  const low = Math.min(...rangeBars.map(b => b.l));

  return { high, low, valid: true };
}

/**
 * VWAP Reversion Strategy
 *
 * Mean reversion strategy that looks for:
 * - Price extended from VWAP by X%
 * - RSI confirming oversold/overbought
 * - MACD confirmation (Tier 2): histogram aligning with reversion direction
 * - Volume surge: recent bar volume above average signals conviction
 */
function evaluateVWAPReversion(data: SymbolData): StrategySignal | null {
  const config = strategyRegistry[STRATEGY_NAMES.VWAP_REVERSION];
  if (!config.enabled) return null;

  if (!data.bars || data.bars.length < 20) {
    return null;
  }

  const vwapValue = calculateVWAP(data.bars);
  if (vwapValue <= 0) return null;

  const currentPrice = data.currentPrice;
  const rsiVal = data.indicators.rsi14;
  const volume = data.volume;

  // Calculate deviation from VWAP
  const deviation = ((currentPrice - vwapValue) / vwapValue) * 100;
  const absDeviation = Math.abs(deviation);

  // MACD confirmation (available if Tier 2 indicators)
  let macdLine: number | undefined;
  let macdSignalVal: number | undefined;
  let macdConfirm = false;
  if (hasMacd(data.indicators)) {
    macdLine = data.indicators.macdLine;
    macdSignalVal = data.indicators.macdSignal;
  }

  // Volume surge: check if recent bars show above-average volume
  let volumeSurge = false;
  if (data.bars.length >= 10) {
    const recentBars = data.bars.slice(-5);
    const olderBars = data.bars.slice(-20, -5);
    const recentAvgVol = recentBars.reduce((s, b) => s + b.v, 0) / recentBars.length;
    const olderAvgVol = olderBars.length > 0 ? olderBars.reduce((s, b) => s + b.v, 0) / olderBars.length : recentAvgVol;
    volumeSurge = recentAvgVol > olderAvgVol * 1.3; // 30% above average
  }

  if (process.env.DEBUG_STRATEGY === "1" && data.symbol === "QQQ") {
    console.log(
      `[DEBUG][VWAP] ${data.symbol} price=${currentPrice.toFixed(2)} vwap=${vwapValue.toFixed(2)} ` +
      `dev=${deviation.toFixed(3)}% rsi=${rsiVal.toFixed(1)} vol=${Math.floor(volume)} ` +
      `macd=${macdLine?.toFixed(3) ?? 'N/A'} macdSig=${macdSignalVal?.toFixed(3) ?? 'N/A'} volSurge=${volumeSurge} ` +
      `minDev=${VWAP_CONFIG.MIN_DEVIATION_PCT}% maxDev=${VWAP_CONFIG.MAX_DEVIATION_PCT}% ` +
      `rsiOS=${VWAP_CONFIG.RSI_OVERSOLD} rsiOB=${VWAP_CONFIG.RSI_OVERBOUGHT}`
    );
  }

  // Check volume requirement
  if (volume < VWAP_CONFIG.MIN_VOLUME) {
    return null;
  }

  // Check deviation is within acceptable range
  if (absDeviation < VWAP_CONFIG.MIN_DEVIATION_PCT || absDeviation > VWAP_CONFIG.MAX_DEVIATION_PCT) {
    return null;
  }

  let side: SignalSide = "none";
  let reason = "";
  let invalidation: number | null = null;

  // BUY signal: Price below VWAP + RSI oversold
  if (deviation < -VWAP_CONFIG.MIN_DEVIATION_PCT && rsiVal < VWAP_CONFIG.RSI_OVERSOLD) {
    // MACD confirmation for buy: MACD histogram turning positive (momentum shifting up)
    if (macdLine !== undefined && macdSignalVal !== undefined) {
      macdConfirm = macdLine > macdSignalVal; // MACD crossing above signal = bullish
    }
    side = "buy";
    reason = `VWAP reversion: price ${Math.abs(deviation).toFixed(2)}% below VWAP, RSI=${rsiVal.toFixed(0)} oversold`;
    if (macdConfirm) reason += `, MACD confirmed`;
    if (volumeSurge) reason += `, volume surge`;
    invalidation = currentPrice * 0.99; // Invalidate if drops 1% more
  }
  // SELL signal: Price above VWAP + RSI overbought
  else if (deviation > VWAP_CONFIG.MIN_DEVIATION_PCT && rsiVal > VWAP_CONFIG.RSI_OVERBOUGHT) {
    // MACD confirmation for sell: MACD histogram turning negative (momentum shifting down)
    if (macdLine !== undefined && macdSignalVal !== undefined) {
      macdConfirm = macdLine < macdSignalVal; // MACD crossing below signal = bearish
    }
    side = "sell";
    reason = `VWAP reversion: price ${deviation.toFixed(2)}% above VWAP, RSI=${rsiVal.toFixed(0)} overbought`;
    if (macdConfirm) reason += `, MACD confirmed`;
    if (volumeSurge) reason += `, volume surge`;
    invalidation = currentPrice * 1.01; // Invalidate if rises 1% more
  }

  if (side === "none") return null;

  // Calculate confidence based on deviation, RSI extremity, MACD, and volume
  let confidence = VWAP_CONFIG.BASE_CONFIDENCE;

  // Boost confidence for larger deviations (up to a point)
  const deviationBoost = Math.min(10, absDeviation * 5);
  confidence += deviationBoost;

  // Boost for extreme RSI
  if (side === "buy" && rsiVal < 25) confidence += 10;
  if (side === "sell" && rsiVal > 75) confidence += 10;

  // MACD confirmation boost (+5)
  if (macdConfirm) confidence += 5;

  // Volume surge boost (+3)
  if (volumeSurge) confidence += 3;

  confidence = Math.min(VWAP_CONFIG.MAX_CONFIDENCE, confidence);

  // Apply strategy weight
  confidence = Math.round(confidence * config.weight);

  return {
    strategyName: STRATEGY_NAMES.VWAP_REVERSION,
    symbol: data.symbol,
    side,
    confidence,
    entrySignalPrice: currentPrice,
    reason,
    invalidation,
    metadata: {
      vwapDeviation: deviation,
      rsi: rsiVal,
      macdLine,
      macdSignal: macdSignalVal,
      macdConfirm,
      volumeSurge,
    },
  };
}

/**
 * ORB (Opening Range Breakout) Strategy
 *
 * Breakout strategy that:
 * - Defines opening range from first 15 minutes
 * - Signals on breakout with volume confirmation
 * - Requires trend alignment (EMA20)
 * - Volume surge at breakout confirms institutional participation
 */
function evaluateORB(data: SymbolData): StrategySignal | null {
  const config = strategyRegistry[STRATEGY_NAMES.ORB];
  if (!config.enabled) return null;

  // ORB only valid after opening range is established
  const et = getEasternTime();
  const currentMinutes = et.hour * 60 + et.minute;
  const marketOpenMinutes = 9 * 60 + 30;
  const rangeEndMinutes = marketOpenMinutes + ORB_CONFIG.RANGE_MINUTES;

  // Must be after opening range period
  if (currentMinutes < rangeEndMinutes) {
    return null;
  }

  // Valid until 11:30 AM (extended from 11:00 for more opportunities)
  if (currentMinutes > 11 * 60 + 30) {
    return null;
  }

  if (!data.bars || data.bars.length < 10) {
    if (process.env.DEBUG_STRATEGY === "1" && data.symbol === "QQQ") {
      console.log(`[DEBUG][ORB] ${data.symbol} insufficient bars: ${data.bars?.length || 0}`);
    }
    return null;
  }

  const range = calculateOpeningRange(data.bars, ORB_CONFIG.RANGE_MINUTES);
  if (!range.valid) {
    if (process.env.DEBUG_STRATEGY === "1" && data.symbol === "QQQ") {
      console.log(`[DEBUG][ORB] ${data.symbol} opening range invalid`);
    }
    return null;
  }

  const currentPrice = data.currentPrice;
  const rangeSize = range.high - range.low;
  const rangePct = (rangeSize / range.low) * 100;

  // Volume surge at breakout: compare recent volume to average
  let volumeSurge = false;
  if (data.bars.length >= 10) {
    const recentBars = data.bars.slice(-3);
    const olderBars = data.bars.slice(-15, -3);
    const recentAvgVol = recentBars.reduce((s, b) => s + b.v, 0) / recentBars.length;
    const olderAvgVol = olderBars.length > 0 ? olderBars.reduce((s, b) => s + b.v, 0) / olderBars.length : recentAvgVol;
    volumeSurge = recentAvgVol > olderAvgVol * 1.5; // 50% above average for breakout
  }

  if (process.env.DEBUG_STRATEGY === "1" && data.symbol === "QQQ") {
    console.log(
      `[DEBUG][ORB] ${data.symbol} price=${currentPrice.toFixed(2)} rangeHigh=${range.high.toFixed(2)} ` +
      `rangeLow=${range.low.toFixed(2)} rangePct=${rangePct.toFixed(3)}% ema20=${data.indicators.ema20.toFixed(2)} ` +
      `volSurge=${volumeSurge} buffer=${ORB_CONFIG.BREAKOUT_BUFFER_PCT}% ` +
      `minRange=${ORB_CONFIG.MIN_RANGE_PCT}% maxRange=${ORB_CONFIG.MAX_RANGE_PCT}%`
    );
  }

  // Range size validation
  if (rangePct < ORB_CONFIG.MIN_RANGE_PCT || rangePct > ORB_CONFIG.MAX_RANGE_PCT) {
    return null;
  }

  const ema20 = data.indicators.ema20;
  const breakoutBuffer = currentPrice * (ORB_CONFIG.BREAKOUT_BUFFER_PCT / 100);
  const emaBuyThreshold = ema20 * (1 - ORB_TREND_TOLERANCE_PCT / 100);
  const emaSellThreshold = ema20 * (1 + ORB_TREND_TOLERANCE_PCT / 100);

  // Require volume surge for ORB breakout — without it, breakouts are unreliable noise
  if (!volumeSurge) {
    return null;
  }

  let side: SignalSide = "none";
  let reason = "";
  let invalidation: number | null = null;
  let breakoutDirection: "up" | "down" | undefined;

  // Bullish breakout: price above ORB high + buffer + trend alignment
  if (currentPrice > range.high + breakoutBuffer && currentPrice > emaBuyThreshold) {
    side = "buy";
    reason = `ORB breakout: price ${currentPrice.toFixed(2)} above range high ${range.high.toFixed(2)}, trend aligned (EMA20=${ema20.toFixed(2)})`;
    if (volumeSurge) reason += `, volume surge`;
    invalidation = range.high - breakoutBuffer; // Invalidate if falls back into range
    breakoutDirection = "up";
  }
  // Bearish breakout: price below ORB low - buffer + trend alignment
  else if (currentPrice < range.low - breakoutBuffer && currentPrice < emaSellThreshold) {
    side = "sell";
    reason = `ORB breakdown: price ${currentPrice.toFixed(2)} below range low ${range.low.toFixed(2)}, trend aligned (EMA20=${ema20.toFixed(2)})`;
    if (volumeSurge) reason += `, volume surge`;
    invalidation = range.low + breakoutBuffer; // Invalidate if rises back into range
    breakoutDirection = "down";
  }

  if (side === "none") return null;

  // Calculate confidence (raised base from 50 to 55 for better pass rate)
  let confidence = ORB_CONFIG.BASE_CONFIDENCE;

  // Boost for clean breakout (further from range)
  const distanceFromRange = side === "buy"
    ? ((currentPrice - range.high) / range.high) * 100
    : ((range.low - currentPrice) / range.low) * 100;
  confidence += Math.min(10, distanceFromRange * 5);

  // Boost for volume (general)
  if (data.volume > 500000) confidence += 5;

  // Volume surge at breakout boost (+7) - strong institutional confirmation
  if (volumeSurge) confidence += 7;

  // RSI alignment boost: RSI > 50 for buy, < 50 for sell
  const rsiVal = data.indicators.rsi14;
  if ((side === "buy" && rsiVal > 55) || (side === "sell" && rsiVal < 45)) {
    confidence += 3;
  }

  confidence = Math.min(ORB_CONFIG.MAX_CONFIDENCE, confidence);

  // Apply strategy weight
  confidence = Math.round(confidence * config.weight);

  return {
    strategyName: STRATEGY_NAMES.ORB,
    symbol: data.symbol,
    side,
    confidence,
    entrySignalPrice: currentPrice,
    reason,
    invalidation,
    metadata: {
      orbHigh: range.high,
      orbLow: range.low,
      orbBreakoutDirection: breakoutDirection,
      rsi: rsiVal,
      volumeSurge,
    },
  };
}

/**
 * EMA Crossover Strategy
 *
 * Trend-following strategy that uses:
 * - EMA9/EMA20 crossover for direction (fast EMA crosses slow EMA)
 * - MACD confirmation (Tier 2): histogram aligning with crossover direction
 * - RSI filter: ensures momentum isn't exhausted
 * - Bollinger Band context: price near lower band for buy, upper for sell adds confidence
 * - Volume above minimum threshold
 */
function evaluateEMACrossover(data: SymbolData): StrategySignal | null {
  const config = strategyRegistry[STRATEGY_NAMES.EMA_CROSSOVER];
  if (!config.enabled) return null;

  if (!data.bars || data.bars.length < 20) {
    return null;
  }

  const currentPrice = data.currentPrice;
  const ema9 = data.indicators.ema9;
  const ema20 = data.indicators.ema20;
  const rsiVal = data.indicators.rsi14;
  const volume = data.volume;

  // Check volume requirement
  if (volume < EMA_CROSS_CONFIG.MIN_VOLUME) {
    return null;
  }

  // Calculate EMA spread
  const emaSpreadPct = Math.abs(ema9 - ema20) / ema20 * 100;
  if (emaSpreadPct < EMA_CROSS_CONFIG.MIN_EMA_SPREAD_PCT) {
    return null; // EMAs too close together, no clear crossover
  }

  // Determine crossover direction
  const isBullishCross = ema9 > ema20;
  const isBearishCross = ema9 < ema20;

  // MACD confirmation
  let macdLine: number | undefined;
  let macdSignalVal: number | undefined;
  let macdConfirm = false;
  if (hasMacd(data.indicators)) {
    macdLine = data.indicators.macdLine;
    macdSignalVal = data.indicators.macdSignal;
  }

  // Bollinger Band context (compute from bars closes)
  let bollingerPosition = "middle";
  const closes = data.bars.map(b => b.c);
  if (closes.length >= 20) {
    const bb = bollingerBands(closes, 20, 2);
    const bbUpper = bb.upper[bb.upper.length - 1];
    const bbLower = bb.lower[bb.lower.length - 1];
    const bbMiddle = bb.middle[bb.middle.length - 1];
    if (currentPrice <= bbLower) {
      bollingerPosition = "lower";
    } else if (currentPrice >= bbUpper) {
      bollingerPosition = "upper";
    } else if (currentPrice < bbMiddle) {
      bollingerPosition = "belowMiddle";
    } else {
      bollingerPosition = "aboveMiddle";
    }
  }

  // Volume surge detection
  let volumeSurge = false;
  if (data.bars.length >= 10) {
    const recentBars = data.bars.slice(-3);
    const olderBars = data.bars.slice(-15, -3);
    const recentAvgVol = recentBars.reduce((s, b) => s + b.v, 0) / recentBars.length;
    const olderAvgVol = olderBars.length > 0 ? olderBars.reduce((s, b) => s + b.v, 0) / olderBars.length : recentAvgVol;
    volumeSurge = recentAvgVol > olderAvgVol * 1.3;
  }

  if (process.env.DEBUG_STRATEGY === "1" && data.symbol === "QQQ") {
    console.log(
      `[DEBUG][EMA_CROSS] ${data.symbol} price=${currentPrice.toFixed(2)} ema9=${ema9.toFixed(2)} ema20=${ema20.toFixed(2)} ` +
      `spread=${emaSpreadPct.toFixed(3)}% rsi=${rsiVal.toFixed(1)} vol=${Math.floor(volume)} ` +
      `macd=${macdLine?.toFixed(3) ?? 'N/A'} bb=${bollingerPosition} volSurge=${volumeSurge}`
    );
  }

  let side: SignalSide = "none";
  let reason = "";
  let invalidation: number | null = null;
  let crossDirection: "bullish" | "bearish" | undefined;

  // BUY signal: EMA9 > EMA20 (bullish cross) + RSI in valid range
  if (isBullishCross && rsiVal > EMA_CROSS_CONFIG.RSI_MIN_BUY && rsiVal < EMA_CROSS_CONFIG.RSI_MAX_BUY) {
    // MACD confirmation for buy
    if (macdLine !== undefined && macdSignalVal !== undefined) {
      macdConfirm = macdLine > macdSignalVal;
    }
    // Price must be above EMA20 (confirms trend)
    if (currentPrice < ema20) return null;

    side = "buy";
    crossDirection = "bullish";
    reason = `EMA crossover: EMA9=${ema9.toFixed(2)} > EMA20=${ema20.toFixed(2)}, RSI=${rsiVal.toFixed(0)}`;
    if (macdConfirm) reason += `, MACD confirmed`;
    if (volumeSurge) reason += `, volume surge`;
    if (bollingerPosition === "lower" || bollingerPosition === "belowMiddle") reason += `, BB=${bollingerPosition}`;
    invalidation = ema20 * 0.995; // Invalidate if price drops below EMA20
  }
  // SELL signal: EMA9 < EMA20 (bearish cross) + RSI in valid range
  else if (isBearishCross && rsiVal > EMA_CROSS_CONFIG.RSI_MIN_SELL && rsiVal < EMA_CROSS_CONFIG.RSI_MAX_SELL) {
    // MACD confirmation for sell
    if (macdLine !== undefined && macdSignalVal !== undefined) {
      macdConfirm = macdLine < macdSignalVal;
    }
    // Price must be below EMA20 (confirms trend)
    if (currentPrice > ema20) return null;

    side = "sell";
    crossDirection = "bearish";
    reason = `EMA crossover: EMA9=${ema9.toFixed(2)} < EMA20=${ema20.toFixed(2)}, RSI=${rsiVal.toFixed(0)}`;
    if (macdConfirm) reason += `, MACD confirmed`;
    if (volumeSurge) reason += `, volume surge`;
    if (bollingerPosition === "upper" || bollingerPosition === "aboveMiddle") reason += `, BB=${bollingerPosition}`;
    invalidation = ema20 * 1.005; // Invalidate if price rises above EMA20
  }

  if (side === "none") return null;

  // Calculate confidence
  let confidence = EMA_CROSS_CONFIG.BASE_CONFIDENCE;

  // EMA spread boost: larger spread = stronger trend (up to +8)
  confidence += Math.min(8, emaSpreadPct * 10);

  // MACD confirmation boost (+5)
  if (macdConfirm) confidence += 5;

  // Volume surge boost (+5)
  if (volumeSurge) confidence += 5;

  // Bollinger Band context boost
  if (side === "buy" && (bollingerPosition === "lower" || bollingerPosition === "belowMiddle")) {
    confidence += 3; // Buying near lower band = good entry point
  }
  if (side === "sell" && (bollingerPosition === "upper" || bollingerPosition === "aboveMiddle")) {
    confidence += 3; // Selling near upper band = good entry point
  }

  confidence = Math.min(EMA_CROSS_CONFIG.MAX_CONFIDENCE, confidence);

  // Apply strategy weight
  confidence = Math.round(confidence * config.weight);

  return {
    strategyName: STRATEGY_NAMES.EMA_CROSSOVER,
    symbol: data.symbol,
    side,
    confidence,
    entrySignalPrice: currentPrice,
    reason,
    invalidation,
    metadata: {
      ema9,
      ema20,
      emaCrossDirection: crossDirection,
      rsi: rsiVal,
      macdLine,
      macdSignal: macdSignalVal,
      macdConfirm,
      volumeSurge,
      bollingerPosition,
    },
  };
}

/**
 * Evaluate all enabled strategies for a symbol
 * Returns array of signals (may be empty if no strategy triggers)
 */
export function evaluateStrategies(data: SymbolData): StrategySignal[] {
  const signals: StrategySignal[] = [];

  // Evaluate each strategy
  const vwapSignal = evaluateVWAPReversion(data);
  if (vwapSignal) {
    signals.push(vwapSignal);
  }

  const orbSignal = evaluateORB(data);
  if (orbSignal) {
    signals.push(orbSignal);
  }

  const emaCrossSignal = evaluateEMACrossover(data);
  if (emaCrossSignal) {
    signals.push(emaCrossSignal);
  }

  return signals;
}

/**
 * Decision policy: choose best signal from multiple strategies
 * Returns highest confidence signal, or null if none pass threshold
 * PRO: Raised to 68% to aggressively filter weak signals — backtest showed 55% generated 91% timeout exits
 */
export function selectBestSignal(signals: StrategySignal[], minConfidence: number = 68): StrategySignal | null {
  if (signals.length === 0) return null;
  
  // Filter by minimum confidence
  const validSignals = signals.filter(s => s.confidence >= minConfidence);
  if (validSignals.length === 0) return null;
  
  // Sort by confidence (descending)
  validSignals.sort((a, b) => b.confidence - a.confidence);
  
  const selected = validSignals[0];
  console.log(`[StrategyEngine] Selected: ${selected.strategyName} for ${selected.symbol} (confidence=${selected.confidence}, side=${selected.side})`);
  
  return selected;
}

/**
 * Main entry point: evaluate and select best signal for a symbol
 */
export async function generateSignal(data: SymbolData): Promise<StrategySignal | null> {
  const signals = evaluateStrategies(data);
  
  if (signals.length === 0) {
    console.log(`[StrategyEngine] ${data.symbol}: NO_SIGNAL - no strategy conditions met`);
    return null;
  }
  
  return selectBestSignal(signals);
}

/**
 * Fetch bars for strategy evaluation (with VWAP/ORB support)
 */
export async function fetchBarsForStrategy(symbol: string): Promise<AlpacaBar[] | null> {
  try {
    const result = await alpaca.getBarsSafe(symbol, "5Min", 130);
    if (result.ok && result.bars) {
      return result.bars;
    }
    return null;
  } catch (error) {
    console.error(`[StrategyEngine] Error fetching bars for ${symbol}:`, error);
    return null;
  }
}

/**
 * Log strategy engine status
 */
export function logStrategyEngineStatus(): void {
  const enabled = getEnabledStrategies();
  console.log(`[StrategyEngine] Status: ${enabled.length} strategies enabled: ${enabled.join(", ")}`);
  
  for (const [name, config] of Object.entries(strategyRegistry)) {
    console.log(`  ${name}: enabled=${config.enabled}, weight=${config.weight}`);
  }
}
