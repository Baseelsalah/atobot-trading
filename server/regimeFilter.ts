/**
 * Regime Filter v1 - Market-wide filter based on SPY trend
 * 
 * FAIL-CLOSED: If SPY data is missing, SKIP all trades.
 * 
 * Simple SPY trend filter:
 * - Compute EMA9 and EMA21 on SPY bars
 * - ok=true only if EMA9 >= EMA21 (bullish/baseline)
 * - REGIME-DEADBAND-1: If EMA9 < EMA21 but spread < REGIME_MIN_EMA_SPREAD_PCT,
 *   treat as neutral (deadband) and allow trading. This prevents blocking
 *   on trivial noise (e.g., 0.006% spread at market open).
 * - Otherwise SKIP all trades with reason: "regime:spyBearish"
 */

// Configurable deadband threshold: block only if bearish spread >= this value
// 0.10 = 0.10% spread required to block (prevents blocking on 0.006% noise)
export const REGIME_MIN_EMA_SPREAD_PCT = 0.10;

import * as alpaca from "./alpaca";
import { ema } from "./indicators";

export type RegimeLabel = "bull" | "bear" | "chop";

export interface RegimeResult {
  ok: boolean;
  reasons: string[];
  ema9?: number;
  ema21?: number;
  regimeLabel?: RegimeLabel;
  emaSpreadPct?: number;
}

export async function evaluateMarketRegime(): Promise<RegimeResult> {
  try {
    const spyBars = await alpaca.getBars("SPY", "5Min", 50);
    
    if (!spyBars || spyBars.length < 21) {
      console.log(`[RegimeFilter] SPY bars missing or insufficient (${spyBars?.length ?? 0} bars)`);
      return {
        ok: false,
        reasons: ["regime:missing"],
      };
    }
    
    const closes = spyBars.map(b => b.c);
    
    const ema9Array = ema(closes, 9);
    const ema21Array = ema(closes, 21);
    
    const latestEma9 = ema9Array[ema9Array.length - 1];
    const latestEma21 = ema21Array[ema21Array.length - 1];
    
    if (!Number.isFinite(latestEma9) || !Number.isFinite(latestEma21)) {
      console.log(`[RegimeFilter] Invalid EMA values: EMA9=${latestEma9}, EMA21=${latestEma21}`);
      return {
        ok: false,
        reasons: ["regime:invalidEMA"],
        ema9: latestEma9,
        ema21: latestEma21,
      };
    }
    
    const isBullish = latestEma9 >= latestEma21;
    const emaSpreadPct = Math.abs(latestEma9 - latestEma21) / latestEma21 * 100;
    
    // Determine regime label: bull, bear, or chop
    // Chop = EMA spread < 0.15% (very tight, no clear direction)
    const regimeLabel: RegimeLabel = determineRegimeLabel(latestEma9, latestEma21);
    
    if (!isBullish) {
      // REGIME-DEADBAND-1: Check if spread is below threshold (noise)
      if (emaSpreadPct < REGIME_MIN_EMA_SPREAD_PCT) {
        // Deadband: EMA9 < EMA21 but spread too small to be meaningful
        console.log(`[RegimeFilter] SPY deadband: EMA9=${latestEma9.toFixed(2)} EMA21=${latestEma21.toFixed(2)} spread=${emaSpreadPct.toFixed(3)}% < threshold=${REGIME_MIN_EMA_SPREAD_PCT}% reason=regime:deadband_bearish ALLOW_TRADING`);
        return {
          ok: true,
          reasons: ["regime:deadband_bearish"],
          ema9: latestEma9,
          ema21: latestEma21,
          regimeLabel: "chop", // Treat deadband as chop
          emaSpreadPct,
        };
      }
      
      // Spread >= threshold: meaningful bearish, block trading
      console.log(`[RegimeFilter] SPY bearish: EMA9=${latestEma9.toFixed(2)} EMA21=${latestEma21.toFixed(2)} spread=${emaSpreadPct.toFixed(3)}% >= threshold=${REGIME_MIN_EMA_SPREAD_PCT}% reason=regime:spyBearish BLOCK_TRADING`);
      return {
        ok: false,
        reasons: ["regime:spyBearish"],
        ema9: latestEma9,
        ema21: latestEma21,
        regimeLabel,
        emaSpreadPct,
      };
    }
    
    console.log(`[RegimeFilter] SPY bullish: EMA9=${latestEma9.toFixed(2)} EMA21=${latestEma21.toFixed(2)} spread=${emaSpreadPct.toFixed(3)}% reason=bullish ALLOW_TRADING`);
    return {
      ok: true,
      reasons: [],
      ema9: latestEma9,
      ema21: latestEma21,
      regimeLabel,
      emaSpreadPct,
    };
    
  } catch (error) {
    console.error(`[RegimeFilter] Error evaluating market regime:`, error);
    return {
      ok: false,
      reasons: ["regime:fetchError"],
    };
  }
}

export function logRegimeSkip(result: RegimeResult): void {
  if (!result.ok && result.reasons.length > 0) {
    console.log(`ACTION=SKIP symbol=ALL SKIP_REASONS=[${result.reasons.join(",")}]`);
  }
}

export function getRegimeStatus(result: RegimeResult): string {
  if (!result.ok) {
    return result.reasons[0] || "regime:unknown";
  }
  // Check for deadband bypass case (ok=true but reason is deadband)
  if (result.reasons.includes("regime:deadband_bearish")) {
    return "deadband_bearish (allowed)";
  }
  return "bullish";
}

/**
 * Determine regime label: bull, bear, or chop
 * - bull: EMA9 >= EMA21
 * - bear: EMA9 < EMA21
 * - chop: EMA spread < 0.15% (very tight, no clear direction)
 */
export function determineRegimeLabel(ema9: number, ema21: number): RegimeLabel {
  const emaSpreadPct = Math.abs(ema9 - ema21) / ema21 * 100;
  
  // Chop threshold: very tight spread indicates no clear trend
  const CHOP_THRESHOLD_PCT = 0.15;
  
  if (emaSpreadPct < CHOP_THRESHOLD_PCT) {
    return "chop";
  }
  
  return ema9 >= ema21 ? "bull" : "bear";
}

/**
 * Get current regime label from last evaluation
 * Returns "chop" if result not available
 */
export function getRegimeLabel(result: RegimeResult | null): RegimeLabel {
  if (!result || !result.regimeLabel) {
    return "chop"; // Default to chop if unknown
  }
  return result.regimeLabel;
}
