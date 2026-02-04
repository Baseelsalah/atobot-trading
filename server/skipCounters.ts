/**
 * Skip Reason Counters - Track why trades are being skipped
 * Used for debugging and monitoring trading logic
 */

import { getEasternTime } from "./timezone";

// Skip reason types - aligned with trading reason precedence
// P1 Standard reason codes: NO_QUOTE, SPREAD_TOO_WIDE, VOLUME_TOO_LOW, ATR_TOO_LOW, ATR_TOO_HIGH, TIME_GUARD_BLOCKED, STRATEGY_NOT_ENABLED
export type SkipReason =
  | "MARKET_CLOSED"
  | "WEEKEND_HOLIDAY"
  | "BEFORE_ENTRY_WINDOW"
  | "OUTSIDE_ENTRY_WINDOW"
  | "AFTER_ENTRY_CUTOFF"
  | "FORCE_CLOSE_REQUIRED"
  | "CHOP_REGIME"
  | "SPREAD_TOO_WIDE"
  | "MAX_POSITIONS"
  | "MAX_ENTRIES_DAY"
  | "COOLDOWN_ACTIVE"
  | "PROFIT_LOCK"
  | "MAX_LOSS_LOCK"
  | "ATR_TOO_LOW"
  | "ATR_TOO_HIGH"
  | "SYMBOL_NOT_ALLOWED"
  | "KILL_THRESHOLD_HIT"
  | "RISK_CHECK_FAILED"
  | "NO_SIGNAL"
  | "INSUFFICIENT_BUYING_POWER"
  | "LOW_CONFIDENCE"
  | "NO_VALID_PRICE"
  | "EXECUTION_FAILED"
  // Data pipeline failure reasons
  | "NO_BARS_RETURNED"
  | "INSUFFICIENT_BARS"
  | "INVALID_CLOSE_VALUES"
  | "API_ERROR"
  | "NAN_INDICATORS"
  | "INDICATOR_PIPELINE_FAIL"
  // Tier-based strategy skip reasons
  | "MACD_REQUIRES_TIER_2"
  // P1 Tradability gate reasons (standard codes)
  | "NO_QUOTE"
  | "VOLUME_TOO_LOW"
  | "TIME_GUARD_BLOCKED"
  | "STRATEGY_NOT_ENABLED"
  | "LIQUIDITY_TOO_LOW"
  | "EXTREME_MOVE"
  | "TIME_GATE"
  | "TRADABILITY_GATE_FAIL"
  // P2 Idempotency guard
  | "IDEMPOTENCY"
  // P6 Execution quality reasons
  | "QUOTE_STALE_OR_MISSING"
  | "UNFILLED_TIMEOUT"
  | "ORDER_REJECTED"
  | "SPREAD_NEAR_MAX";

// In-memory counter map
const skipCounters: Map<string, number> = new Map();
let lastResetDate: string | null = null;
let lastPrintTime: number = 0;

// Check for TEST_PRINT_INTERVAL_MINUTES env override
const TEST_INTERVAL_MINUTES = parseInt(process.env.TEST_PRINT_INTERVAL_MINUTES || "0", 10);
const DEFAULT_PRINT_INTERVAL_MS = 15 * 60 * 1000; // 15 minutes
const PRINT_INTERVAL_MS = TEST_INTERVAL_MINUTES > 0 
  ? TEST_INTERVAL_MINUTES * 60 * 1000 
  : DEFAULT_PRINT_INTERVAL_MS;

if (TEST_INTERVAL_MINUTES > 0) {
  console.log(`[SKIP_STATS] TEST MODE: Print interval set to ${TEST_INTERVAL_MINUTES} minute(s)`);
}

/**
 * Record a skip reason
 */
export function recordSkip(reason: SkipReason | string): void {
  checkAndResetDaily();
  
  const currentCount = skipCounters.get(reason) || 0;
  skipCounters.set(reason, currentCount + 1);
  
  // Also track per-tick for activity ledger
  recordTickSkip(reason);
  
  // Check if we should print stats
  const now = Date.now();
  if (now - lastPrintTime >= PRINT_INTERVAL_MS) {
    printSkipStats();
    lastPrintTime = now;
  }
}

/**
 * Get all skip counts
 */
export function getSkipCounts(): Map<string, number> {
  return new Map(skipCounters);
}

// Per-tick counter for activity ledger (resets after each snapshot)
const tickSkipCounters: Map<string, number> = new Map();
let tickTotalSkips = 0;

/**
 * Record a skip for per-tick tracking (called alongside recordSkip)
 */
export function recordTickSkip(reason: SkipReason | string): void {
  const currentCount = tickSkipCounters.get(reason) || 0;
  tickSkipCounters.set(reason, currentCount + 1);
  tickTotalSkips++;
}

/**
 * Get per-tick skip snapshot and reset
 */
export function getAndResetTickSkips(): { totalSkips: number; reasonCounts: Record<string, number> } {
  const reasonCounts: Record<string, number> = {};
  tickSkipCounters.forEach((count, reason) => {
    reasonCounts[reason] = count;
  });
  
  const result = {
    totalSkips: tickTotalSkips,
    reasonCounts,
  };
  
  tickSkipCounters.clear();
  tickTotalSkips = 0;
  
  return result;
}

/**
 * Get top N skip reasons
 */
export function getTopSkipReasons(n: number = 5): Array<{ reason: string; count: number }> {
  const sorted = Array.from(skipCounters.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, n)
    .map(([reason, count]) => ({ reason, count }));
  
  return sorted;
}

/**
 * Print skip stats to console
 */
export function printSkipStats(): void {
  const top5 = getTopSkipReasons(5);
  const et = getEasternTime();
  
  if (top5.length === 0) {
    console.log(`[SKIP_STATS] ${et.displayTime} - No skips recorded in this period`);
    return;
  }
  
  const intervalMinutes = PRINT_INTERVAL_MS / 60000;
  console.log(`[SKIP_STATS] ${et.displayTime} - Top reasons (last ${intervalMinutes}m):`);
  top5.forEach((item, index) => {
    console.log(`  ${index + 1}) ${item.reason}: ${item.count}`);
  });
}

/**
 * Check if we should reset counters (new trading day)
 */
function checkAndResetDaily(): void {
  const et = getEasternTime();
  const todayDate = et.dateString;
  
  // Reset on new day (check just date, not minute)
  if (!lastResetDate || !lastResetDate.startsWith(todayDate)) {
    resetCounters();
    lastResetDate = todayDate;
    console.log(`[SKIP_STATS] Daily reset for ${todayDate}`);
  }
  
  // Also reset at market open (9:30 ET) - once per day
  if (et.hour === 9 && et.minute === 30) {
    const minuteKey = `${todayDate}-09:30`;
    if (lastResetDate !== minuteKey) {
      resetCounters();
      lastResetDate = minuteKey;
      console.log(`[SKIP_STATS] Market open reset at 9:30 ET`);
    }
  }
}

/**
 * Reset all counters
 */
export function resetCounters(): void {
  skipCounters.clear();
  lastPrintTime = Date.now();
}

/**
 * Get total skip count
 */
export function getTotalSkips(): number {
  let total = 0;
  skipCounters.forEach((count) => {
    total += count;
  });
  return total;
}

/**
 * Force print stats now (for testing)
 */
export function forcePrintStats(): void {
  printSkipStats();
  lastPrintTime = Date.now();
}

/**
 * Get count for a specific reason (for reporting)
 */
export function getSkipCount(reason: string): number {
  return skipCounters.get(reason) || 0;
}

/**
 * Get blocked count (all guardrail-based skips, excluding NO_SIGNAL)
 */
export function getBlockedCount(): number {
  let blocked = 0;
  skipCounters.forEach((count, reason) => {
    if (reason !== "NO_SIGNAL") {
      blocked += count;
    }
  });
  return blocked;
}

/**
 * Get no_signal count
 */
export function getNoSignalCount(): number {
  return skipCounters.get("NO_SIGNAL") || 0;
}
