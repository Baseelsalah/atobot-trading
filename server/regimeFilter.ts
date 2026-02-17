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
// 0.30 = 0.30% spread required to block (relaxed for more trading opportunities)
// PRO DAY TRADING: Allow trading in most conditions, only block severe bearish trends
export const REGIME_MIN_EMA_SPREAD_PCT = 0.30;

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
    const barResult = await alpaca.getBarsSafe("SPY", "5Min", 21);

    if (!barResult.ok || !barResult.bars || barResult.bars.length < 21) {
      console.log(`[RegimeFilter] SPY bars missing or insufficient (${barResult.bars?.length ?? 0} bars, reason: ${barResult.reason ?? "unknown"})`);
      return {
        ok: false,
        reasons: ["regime:missing"],
      };
    }
    const spyBars = barResult.bars;
    
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
    // Chop = EMA spread < 0.08% (very tight, no clear direction)
    const regimeLabel: RegimeLabel = determineRegimeLabel(latestEma9, latestEma21);

    // Block chop regime — backtest showed 57% of trades in chop lost $9,433
    if (regimeLabel === "chop") {
      console.log(`[RegimeFilter] SPY chop: EMA9=${latestEma9.toFixed(2)} EMA21=${latestEma21.toFixed(2)} spread=${emaSpreadPct.toFixed(3)}% reason=regime:chop BLOCK_TRADING`);
      return {
        ok: false,
        reasons: ["regime:chop"],
        ema9: latestEma9,
        ema21: latestEma21,
        regimeLabel,
        emaSpreadPct,
      };
    }

    if (!isBullish) {
      // Block ALL bearish conditions (including deadband) — only trade in confirmed bull
      // Backtest showed bear deadband trades lost $513 on 85 trades
      console.log(`[RegimeFilter] SPY bearish: EMA9=${latestEma9.toFixed(2)} EMA21=${latestEma21.toFixed(2)} spread=${emaSpreadPct.toFixed(3)}% reason=regime:spyBearish BLOCK_TRADING`);
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
 * - chop: EMA spread < 0.08% (very tight, no clear direction)
 * PRO DAY TRADING: Reduced from 0.15% to 0.08% to classify less as chop
 */
export function determineRegimeLabel(ema9: number, ema21: number): RegimeLabel {
  const emaSpreadPct = Math.abs(ema9 - ema21) / ema21 * 100;

  // Chop threshold: very tight spread indicates no clear trend
  const CHOP_THRESHOLD_PCT = 0.08; // PRO: Lowered from 0.15% to allow more directional trades

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
