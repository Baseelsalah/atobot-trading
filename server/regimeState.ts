/**
 * REGIME-BLOCK-ENTRIES-ONLY-1: Centralized regime state tracking
 * 
 * Instead of early-returning when regime=avoid (which starves tradability gates),
 * we store the regime state and check it at precheck time before order submission.
 * This preserves receipts and learning data even when risk controls block trades.
 */

import { getEasternTime } from "./timezone";

type RegimeRecommendation = "aggressive" | "normal" | "cautious" | "avoid";

interface RegimeState {
  recommendation: RegimeRecommendation;
  qqq: string;
  spy: string;
  lastUpdateET: string;
  lastUpdateUTC: string;
  blockReason: string | null;
}

let currentRegimeState: RegimeState = {
  recommendation: "normal",
  qqq: "unknown",
  spy: "unknown",
  lastUpdateET: "",
  lastUpdateUTC: "",
  blockReason: null,
};

export function updateRegimeState(recommendation: RegimeRecommendation, qqq: string, spy: string): void {
  const now = new Date();
  const et = getEasternTime();
  
  currentRegimeState = {
    recommendation,
    qqq,
    spy,
    lastUpdateET: et.displayTime,
    lastUpdateUTC: now.toISOString(),
    blockReason: recommendation === "avoid" ? `regime:bothBearish (QQQ=${qqq}, SPY=${spy})` : null,
  };
  
  if (recommendation === "avoid") {
    console.log(`[REGIME_STATE] Updated: recommendation=avoid blockReason=${currentRegimeState.blockReason}`);
  }
}

export function getRegimeState(): RegimeState {
  return { ...currentRegimeState };
}

export function shouldBlockEntryDueToRegime(): { blocked: boolean; reason: string | null } {
  if (currentRegimeState.recommendation === "avoid") {
    return {
      blocked: true,
      reason: "REGIME_AVOID_BLOCK_ENTRY",
    };
  }
  return {
    blocked: false,
    reason: null,
  };
}

export function getRegimeBlockReason(): string | null {
  return currentRegimeState.blockReason;
}

export function resetRegimeState(): void {
  currentRegimeState = {
    recommendation: "normal",
    qqq: "unknown",
    spy: "unknown",
    lastUpdateET: "",
    lastUpdateUTC: "",
    blockReason: null,
  };
}
