/**
 * Signal Counters - Track raw signals vs executed trades
 * Used to diagnose "limbo" by measuring funnel conversion
 */

import { getEasternTime } from "./timezone";
import { getTopSkipReasons, getNoSignalCount, getBlockedCount } from "./skipCounters";

// Signal types
export type SignalType = 
  | "SIGNAL_LONG_SPY"
  | "SIGNAL_LONG_QQQ"
  | "SIGNAL_LONG_SH"
  | "SIGNAL_SHORT_SPY"
  | "SIGNAL_SHORT_QQQ"
  | "SIGNAL_SHORT_SH";

// In-memory counters
const signalCounters: Map<string, number> = new Map();
const executionCounters: Map<string, number> = new Map();
let candidatesTotal = 0;
let totalScansCounter = 0;
let lastResetDate: string | null = null;
let lastPrintTime: number = 0;

// Check for TEST_PRINT_INTERVAL_MINUTES env override
const TEST_INTERVAL_MINUTES = parseInt(process.env.TEST_PRINT_INTERVAL_MINUTES || "0", 10);
const DEFAULT_PRINT_INTERVAL_MS = 15 * 60 * 1000; // 15 minutes
const PRINT_INTERVAL_MS = TEST_INTERVAL_MINUTES > 0 
  ? TEST_INTERVAL_MINUTES * 60 * 1000 
  : DEFAULT_PRINT_INTERVAL_MS;

/**
 * Record a raw signal (BEFORE filters are applied)
 */
export function recordSignal(type: SignalType | string): void {
  checkAndResetDaily();
  const current = signalCounters.get(type) || 0;
  signalCounters.set(type, current + 1);
  checkPrintStats();
}

/**
 * Record a candidate being evaluated
 */
export function recordCandidate(): void {
  checkAndResetDaily();
  candidatesTotal++;
}

/**
 * Record a scan cycle (analysis tick)
 */
export function recordScan(): void {
  checkAndResetDaily();
  totalScansCounter++;
}

/**
 * Record an execution (actual or DRY_RUN)
 */
export function recordExecution(symbol: string, side: string): void {
  checkAndResetDaily();
  const key = `EXECUTED_${side.toUpperCase()}_${symbol.toUpperCase()}`;
  const current = executionCounters.get(key) || 0;
  executionCounters.set(key, current + 1);
  
  const totalKey = "EXECUTED_TOTAL";
  const totalCurrent = executionCounters.get(totalKey) || 0;
  executionCounters.set(totalKey, totalCurrent + 1);
  
  checkPrintStats();
}

/**
 * Get signal statistics
 */
export function getSignalStats(): {
  signals: Record<string, number>;
  executions: Record<string, number>;
  candidatesTotal: number;
  totalSignals: number;
  totalExecutions: number;
  totalScans: number;
} {
  const signals: Record<string, number> = {};
  let totalSignals = 0;
  signalCounters.forEach((count, key) => {
    signals[key] = count;
    totalSignals += count;
  });
  
  const executions: Record<string, number> = {};
  let totalExecutions = 0;
  executionCounters.forEach((count, key) => {
    executions[key] = count;
    if (key === "EXECUTED_TOTAL") {
      totalExecutions = count;
    }
  });
  
  return {
    signals,
    executions,
    candidatesTotal,
    totalSignals,
    totalExecutions,
    totalScans: totalScansCounter,
  };
}

/**
 * Check if we need to reset for new day
 */
function checkAndResetDaily(): void {
  const et = getEasternTime();
  const todayDate = et.dateString;
  
  if (!lastResetDate || !lastResetDate.startsWith(todayDate)) {
    resetSignalStats();
    lastResetDate = todayDate;
    console.log(`[SIGNAL_STATS] Daily reset for ${todayDate}`);
  }
  
  // Also reset at market open (9:30 ET)
  if (et.hour === 9 && et.minute === 30) {
    const minuteKey = `${todayDate}-09:30`;
    if (lastResetDate !== minuteKey) {
      resetSignalStats();
      lastResetDate = minuteKey;
      console.log(`[SIGNAL_STATS] Market open reset at 9:30 ET`);
    }
  }
}

/**
 * Reset all signal stats
 */
export function resetSignalStats(): void {
  signalCounters.clear();
  executionCounters.clear();
  candidatesTotal = 0;
  totalScansCounter = 0;
  lastPrintTime = Date.now();
}

/**
 * Check if we should print stats
 */
function checkPrintStats(): void {
  const now = Date.now();
  if (now - lastPrintTime >= PRINT_INTERVAL_MS) {
    printFunnelStats();
    lastPrintTime = now;
  }
}

/**
 * Print funnel stats with skip reasons
 */
export function printFunnelStats(): void {
  const stats = getSignalStats();
  const et = getEasternTime();
  
  // Extract signal counts per symbol
  const spySignals = (stats.signals["SIGNAL_LONG_SPY"] || 0) + (stats.signals["SIGNAL_SHORT_SPY"] || 0);
  const qqqSignals = (stats.signals["SIGNAL_LONG_QQQ"] || 0) + (stats.signals["SIGNAL_SHORT_QQQ"] || 0);
  const shSignals = (stats.signals["SIGNAL_LONG_SH"] || 0) + (stats.signals["SIGNAL_SHORT_SH"] || 0);
  
  // Extract execution counts per symbol
  const spyExec = (stats.executions["EXECUTED_BUY_SPY"] || 0) + (stats.executions["EXECUTED_SELL_SPY"] || 0);
  const qqqExec = (stats.executions["EXECUTED_BUY_QQQ"] || 0) + (stats.executions["EXECUTED_SELL_QQQ"] || 0);
  const shExec = (stats.executions["EXECUTED_BUY_SH"] || 0) + (stats.executions["EXECUTED_SELL_SH"] || 0);
  
  console.log(`[SIGNAL_STATS] ${et.displayTime} - Funnel:`);
  console.log(`  Signals: SPY=${spySignals}, QQQ=${qqqSignals}, SH=${shSignals} | TotalSignals=${stats.totalSignals}`);
  console.log(`  Executed: SPY=${spyExec}, QQQ=${qqqExec}, SH=${shExec} | TotalExec=${stats.totalExecutions}`);
  
  // Include top skip reasons
  const topSkips = getTopSkipReasons(3);
  if (topSkips.length > 0) {
    const skipStr = topSkips.map(s => `${s.reason}=${s.count}`).join(", ");
    console.log(`  Top Skips: ${skipStr}`);
  }
}

/**
 * Force print stats now (for testing)
 */
export function forcePrintFunnelStats(): void {
  printFunnelStats();
  lastPrintTime = Date.now();
}

/**
 * Print End-of-Day Summary block
 * Called at market close or end of trading session
 */
export function printEODSummary(additionalStats?: {
  totalScans?: number;
  symbolsEvaluated?: number;
  validPrices?: number;
  validIndicators?: number;
  positionsOpened?: number;
  positionsClosed?: number;
  errorsCount?: number;
}): void {
  const stats = getSignalStats();
  const et = getEasternTime();
  const topSkips = getTopSkipReasons(5);
  const noSignalCount = getNoSignalCount();
  const blockedCount = getBlockedCount();
  
  console.log(`\n[EOD_SUMMARY] ==========================================`);
  console.log(`[EOD_SUMMARY] End-of-Day Summary for ${et.dateString}`);
  console.log(`[EOD_SUMMARY] ==========================================`);
  console.log(`[EOD_SUMMARY] total_scans=${additionalStats?.totalScans ?? stats.totalScans}`);
  console.log(`[EOD_SUMMARY] symbols_evaluated=${additionalStats?.symbolsEvaluated ?? stats.candidatesTotal}`);
  console.log(`[EOD_SUMMARY] valid_prices=${additionalStats?.validPrices ?? 'N/A'}`);
  console.log(`[EOD_SUMMARY] valid_indicators=${additionalStats?.validIndicators ?? 'N/A'}`);
  console.log(`[EOD_SUMMARY] no_signal_count=${noSignalCount}`);
  console.log(`[EOD_SUMMARY] blocked_count=${blockedCount}`);
  console.log(`[EOD_SUMMARY] trades_placed=${stats.totalExecutions}`);
  console.log(`[EOD_SUMMARY] positions_opened=${additionalStats?.positionsOpened ?? stats.totalExecutions}`);
  console.log(`[EOD_SUMMARY] positions_closed=${additionalStats?.positionsClosed ?? 'N/A'}`);
  console.log(`[EOD_SUMMARY] errors_count=${additionalStats?.errorsCount ?? 0}`);
  
  if (topSkips.length > 0) {
    console.log(`[EOD_SUMMARY] top_skip_reasons:`);
    topSkips.forEach((item, index) => {
      console.log(`[EOD_SUMMARY]   ${index + 1}) ${item.reason}: ${item.count}`);
    });
  } else {
    console.log(`[EOD_SUMMARY] top_skip_reasons: none`);
  }
  
  console.log(`[EOD_SUMMARY] ==========================================\n`);
}
