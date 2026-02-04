/**
 * Indicator Pipeline - Safe bar fetching and indicator computation
 * 
 * Hardened pipeline that:
 * 1. Fetches bars safely with retry and validation (with warm-start)
 * 2. Determines tier based on bar availability (Tier 1: 130+, Tier 2: 200+)
 * 3. Computes indicators based on tier (Tier 1: limited, Tier 2: full)
 * 4. Validates all outputs for NaN/undefined
 * 5. Returns structured results for DATA_HEALTH tracking
 */

import * as alpaca from "./alpaca";
import { TIER_1_MIN_BARS, TIER_2_MIN_BARS } from "./alpaca";
import { ema, rsi, atr, macd } from "./indicators";
import { recordSkip, SkipReason } from "./skipCounters";

// Tier 1: Limited indicators (VWAP, RSI, short EMAs) - stable with 130+ bars
export interface Tier1Indicators {
  ema9: number;
  ema20: number;
  rsi14: number;
  atr14: number;
  latestClose: number;
}

// Tier 2: Full indicators (adds MACD which needs 26+9=35 periods warm-up)
export interface Tier2Indicators extends Tier1Indicators {
  macdLine: number;
  macdSignal: number;
}

// Legacy alias for backwards compatibility
export type IndicatorSet = Tier2Indicators;

export interface IndicatorPipelineResult {
  ok: boolean;
  symbol: string;
  tier: 0 | 1 | 2;  // 0 = insufficient, 1 = limited, 2 = full
  indicators?: Tier1Indicators | Tier2Indicators;
  reason?: SkipReason;
  barsLen: number;
}

/**
 * Check if indicators include MACD (Tier 2)
 */
export function hasMacd(indicators: Tier1Indicators | Tier2Indicators | undefined): indicators is Tier2Indicators {
  return !!indicators && 'macdLine' in indicators && 'macdSignal' in indicators;
}

/**
 * Compute and validate indicators for a symbol based on tier
 * - Tier 1 (130-199 bars): EMA9, EMA20, RSI14, ATR14 (no MACD)
 * - Tier 2 (200+ bars): Full stack including MACD
 */
export async function computeIndicatorsSafe(symbol: string): Promise<IndicatorPipelineResult> {
  const barResult = await alpaca.getBarsSafe(symbol, "5Min", TIER_1_MIN_BARS);
  
  // Check for bar fetch failure
  if (!barResult.ok) {
    const reason = barResult.reason as SkipReason;
    recordSkip(reason);
    return {
      ok: false,
      symbol,
      tier: 0,
      reason,
      barsLen: barResult.barsLen,
    };
  }
  
  const bars = barResult.bars;
  const tier = barResult.tier;
  const closes = bars.map(b => b.c);
  const highs = bars.map(b => b.h);
  const lows = bars.map(b => b.l);
  
  try {
    // Compute Tier 1 indicators (always computed if we have 130+ bars)
    const ema9Array = ema(closes, 9);
    const ema20Array = ema(closes, 20);
    const rsi14Array = rsi(closes, 14);
    const atr14Array = atr(highs, lows, closes, 14);
    
    const latest = bars.length - 1;
    const tier1Indicators: Tier1Indicators = {
      ema9: ema9Array[latest],
      ema20: ema20Array[latest],
      rsi14: rsi14Array[latest],
      atr14: atr14Array[latest],
      latestClose: closes[latest],
    };
    
    // Validate Tier 1 indicators
    const tier1Check = [
      { name: "ema9", val: tier1Indicators.ema9, check: (v: number) => Number.isFinite(v) },
      { name: "ema20", val: tier1Indicators.ema20, check: (v: number) => Number.isFinite(v) },
      { name: "rsi14", val: tier1Indicators.rsi14, check: (v: number) => Number.isFinite(v) && v >= 0 && v <= 100 },
      { name: "atr14", val: tier1Indicators.atr14, check: (v: number) => Number.isFinite(v) && v > 0 },
    ];
    
    const tier1Invalid = tier1Check.filter(item => !item.check(item.val));
    if (tier1Invalid.length > 0) {
      const invalidNames = tier1Invalid.map(i => `${i.name}=${i.val}`).join(", ");
      console.log(`[IndicatorPipeline] ${symbol}: NAN_INDICATORS tier=${tier} (${invalidNames})`);
      recordSkip("NAN_INDICATORS");
      return {
        ok: false,
        symbol,
        tier: 0,
        reason: "NAN_INDICATORS",
        barsLen: barResult.barsLen,
      };
    }
    
    // If Tier 2, also compute MACD
    if (tier === 2) {
      const macdResult = macd(closes, 12, 26, 9);
      const tier2Indicators: Tier2Indicators = {
        ...tier1Indicators,
        macdLine: macdResult.macd[latest],
        macdSignal: macdResult.signal[latest],
      };
      
      // Validate MACD
      if (!Number.isFinite(tier2Indicators.macdLine) || !Number.isFinite(tier2Indicators.macdSignal)) {
        // MACD failed but Tier 1 is valid - downgrade to Tier 1
        console.log(`[IndicatorPipeline] ${symbol}: MACD invalid, downgrading to tier=1 (${bars.length} bars)`);
        console.log(`[IndicatorPipeline] ${symbol}: OK tier=1 (EMA9=${tier1Indicators.ema9.toFixed(2)}, RSI=${tier1Indicators.rsi14.toFixed(1)}, ATR=${tier1Indicators.atr14.toFixed(2)}) bars=${bars.length}`);
        return {
          ok: true,
          symbol,
          tier: 1,
          indicators: tier1Indicators,
          barsLen: barResult.barsLen,
        };
      }
      
      console.log(`[IndicatorPipeline] ${symbol}: OK tier=2 (EMA9=${tier2Indicators.ema9.toFixed(2)}, RSI=${tier2Indicators.rsi14.toFixed(1)}, MACD=${tier2Indicators.macdLine.toFixed(2)}) bars=${bars.length}`);
      return {
        ok: true,
        symbol,
        tier: 2,
        indicators: tier2Indicators,
        barsLen: barResult.barsLen,
      };
    }
    
    // Tier 1 only
    console.log(`[IndicatorPipeline] ${symbol}: OK tier=1 (EMA9=${tier1Indicators.ema9.toFixed(2)}, RSI=${tier1Indicators.rsi14.toFixed(1)}, ATR=${tier1Indicators.atr14.toFixed(2)}) bars=${bars.length}`);
    return {
      ok: true,
      symbol,
      tier: 1,
      indicators: tier1Indicators,
      barsLen: barResult.barsLen,
    };
    
  } catch (error) {
    console.log(`[IndicatorPipeline] ${symbol}: Error computing indicators: ${error}`);
    recordSkip("NAN_INDICATORS");
    return {
      ok: false,
      symbol,
      tier: 0,
      reason: "NAN_INDICATORS",
      barsLen: barResult.barsLen,
    };
  }
}

/**
 * Fetch indicators for all symbols in universe
 */
export async function fetchIndicatorsForUniverse(symbols: string[]): Promise<{
  results: Map<string, IndicatorPipelineResult>;
  validCount: number;
  tier1Count: number;
  tier2Count: number;
  barsMin: number;
  tierMin: 0 | 1 | 2;
}> {
  const results = new Map<string, IndicatorPipelineResult>();
  let validCount = 0;
  let tier1Count = 0;
  let tier2Count = 0;
  let barsMin = Infinity;
  let tierMin: 0 | 1 | 2 = 2;
  
  for (const symbol of symbols) {
    const result = await computeIndicatorsSafe(symbol);
    results.set(symbol, result);
    
    if (result.ok) {
      validCount++;
      if (result.tier === 1) tier1Count++;
      if (result.tier === 2) tier2Count++;
    }
    
    if (result.barsLen > 0 && result.barsLen < barsMin) {
      barsMin = result.barsLen;
    }
    
    if (result.tier < tierMin) {
      tierMin = result.tier;
    }
  }
  
  return { results, validCount, tier1Count, tier2Count, barsMin, tierMin };
}

// TIER 1 RISK REDUCTION MULTIPLIER
export const TIER_1_SIZING_REDUCTION = 0.5;  // 50% of normal position size
