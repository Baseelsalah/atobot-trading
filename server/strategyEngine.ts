/**
 * P5 Deterministic Strategy Engine
 * 
 * Generates trade signals from deterministic strategies (no LLM dependency).
 * All trades are tagged with strategyName for A/B evaluation in weekly scorecards.
 */

import * as alpaca from "./alpaca";
import { ema, rsi, atr, vwap } from "./indicators";
import { getEasternTime } from "./timezone";
import { Tier1Indicators } from "./indicatorPipeline";

// Strategy names (constants for tagging)
export const STRATEGY_NAMES = {
  VWAP_REVERSION: "VWAP_REVERSION",
  ORB: "ORB",
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
    orbHigh?: number;
    orbLow?: number;
    orbBreakoutDirection?: "up" | "down";
  };
}

// Strategy configuration
export interface StrategyConfig {
  enabled: boolean;
  weight: number;  // For A/B weighting (1.0 = normal, 0.5 = half weight)
}

// Strategy registry with enable/disable and weights
const strategyRegistry: Record<string, StrategyConfig> = {
  [STRATEGY_NAMES.VWAP_REVERSION]: { enabled: true, weight: 1.0 },
  [STRATEGY_NAMES.ORB]: { enabled: true, weight: 1.0 },
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
  MIN_DEVIATION_PCT: 0.3,      // Minimum % deviation from VWAP to trigger
  MAX_DEVIATION_PCT: 2.0,      // Maximum % deviation (too extended = risky)
  RSI_OVERSOLD: 35,            // RSI threshold for buy signals
  RSI_OVERBOUGHT: 65,          // RSI threshold for sell signals
  MIN_VOLUME: 100000,          // Minimum volume requirement
  BASE_CONFIDENCE: 60,         // Base confidence score
  MAX_CONFIDENCE: 85,          // Maximum confidence score
};

// ORB Strategy Configuration
const ORB_CONFIG = {
  RANGE_MINUTES: 15,           // First 15 minutes define the opening range
  BREAKOUT_BUFFER_PCT: 0.05,   // Must break by this % to confirm
  TREND_ALIGNMENT_EMA: 20,     // Use EMA20 for trend alignment
  MIN_RANGE_PCT: 0.15,         // Minimum range size (% of price)
  MAX_RANGE_PCT: 1.5,          // Maximum range size (too wide = skip)
  BASE_CONFIDENCE: 65,         // Base confidence score
  MAX_CONFIDENCE: 85,          // Maximum confidence score
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
 */
function calculateOpeningRange(bars: AlpacaBar[], rangeMinutes: number = 15): { high: number; low: number; valid: boolean } {
  const et = getEasternTime();
  const marketOpenMinutes = 9 * 60 + 30; // 9:30 AM
  const rangeEndMinutes = marketOpenMinutes + rangeMinutes;
  
  // Filter bars within the opening range (9:30 to 9:30 + rangeMinutes)
  const rangeBars = bars.filter(bar => {
    const barTime = new Date(bar.t);
    const barMinutes = barTime.getHours() * 60 + barTime.getMinutes();
    // Adjust for ET if needed (bars might be in UTC)
    const utcOffset = -5; // EST is UTC-5 (simplified)
    const barMinutesET = barMinutes + utcOffset * 60;
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
 * - Signal: buy when oversold and reclaiming VWAP directionally
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
  const rsi = data.indicators.rsi14;
  const volume = data.volume;
  
  // Calculate deviation from VWAP
  const deviation = ((currentPrice - vwapValue) / vwapValue) * 100;
  const absDeviation = Math.abs(deviation);
  
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
  if (deviation < -VWAP_CONFIG.MIN_DEVIATION_PCT && rsi < VWAP_CONFIG.RSI_OVERSOLD) {
    side = "buy";
    reason = `VWAP reversion: price ${Math.abs(deviation).toFixed(2)}% below VWAP, RSI=${rsi.toFixed(0)} oversold`;
    invalidation = currentPrice * 0.99; // Invalidate if drops 1% more
  }
  // SELL signal: Price above VWAP + RSI overbought
  else if (deviation > VWAP_CONFIG.MIN_DEVIATION_PCT && rsi > VWAP_CONFIG.RSI_OVERBOUGHT) {
    side = "sell";
    reason = `VWAP reversion: price ${deviation.toFixed(2)}% above VWAP, RSI=${rsi.toFixed(0)} overbought`;
    invalidation = currentPrice * 1.01; // Invalidate if rises 1% more
  }
  
  if (side === "none") return null;
  
  // Calculate confidence based on deviation and RSI extremity
  let confidence = VWAP_CONFIG.BASE_CONFIDENCE;
  
  // Boost confidence for larger deviations (up to a point)
  const deviationBoost = Math.min(10, absDeviation * 5);
  confidence += deviationBoost;
  
  // Boost for extreme RSI
  if (side === "buy" && rsi < 25) confidence += 10;
  if (side === "sell" && rsi > 75) confidence += 10;
  
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
      rsi,
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
  
  // Only valid in morning session (until 11:00 AM)
  if (currentMinutes > 11 * 60) {
    return null;
  }
  
  if (!data.bars || data.bars.length < 10) {
    return null;
  }
  
  const range = calculateOpeningRange(data.bars, ORB_CONFIG.RANGE_MINUTES);
  if (!range.valid) {
    return null;
  }
  
  const currentPrice = data.currentPrice;
  const rangeSize = range.high - range.low;
  const rangePct = (rangeSize / range.low) * 100;
  
  // Range size validation
  if (rangePct < ORB_CONFIG.MIN_RANGE_PCT || rangePct > ORB_CONFIG.MAX_RANGE_PCT) {
    return null;
  }
  
  const ema20 = data.indicators.ema20;
  const breakoutBuffer = currentPrice * (ORB_CONFIG.BREAKOUT_BUFFER_PCT / 100);
  
  let side: SignalSide = "none";
  let reason = "";
  let invalidation: number | null = null;
  let breakoutDirection: "up" | "down" | undefined;
  
  // Bullish breakout: price above ORB high + buffer + trend alignment
  if (currentPrice > range.high + breakoutBuffer && currentPrice > ema20) {
    side = "buy";
    reason = `ORB breakout: price ${currentPrice.toFixed(2)} above range high ${range.high.toFixed(2)}, trend aligned (EMA20=${ema20.toFixed(2)})`;
    invalidation = range.high - breakoutBuffer; // Invalidate if falls back into range
    breakoutDirection = "up";
  }
  // Bearish breakout: price below ORB low - buffer + trend alignment
  else if (currentPrice < range.low - breakoutBuffer && currentPrice < ema20) {
    side = "sell";
    reason = `ORB breakdown: price ${currentPrice.toFixed(2)} below range low ${range.low.toFixed(2)}, trend aligned (EMA20=${ema20.toFixed(2)})`;
    invalidation = range.low + breakoutBuffer; // Invalidate if rises back into range
    breakoutDirection = "down";
  }
  
  if (side === "none") return null;
  
  // Calculate confidence
  let confidence = ORB_CONFIG.BASE_CONFIDENCE;
  
  // Boost for clean breakout (further from range)
  const distanceFromRange = side === "buy" 
    ? ((currentPrice - range.high) / range.high) * 100
    : ((range.low - currentPrice) / range.low) * 100;
  confidence += Math.min(10, distanceFromRange * 5);
  
  // Boost for volume
  if (data.volume > 500000) confidence += 5;
  
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
  
  return signals;
}

/**
 * Decision policy: choose best signal from multiple strategies
 * Returns highest confidence signal, or null if none pass threshold
 */
export function selectBestSignal(signals: StrategySignal[], minConfidence: number = 60): StrategySignal | null {
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
