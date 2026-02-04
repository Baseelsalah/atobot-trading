/**
 * Data Health Counters - Track pipeline health metrics
 * 
 * Distinguishes between:
 * A) Strategy truly saw no setups (rawSignals = 0 but data is valid)
 * B) Price/indicator pipeline failed (validPrices = 0 or validIndicators = 0)
 */

import { getEasternTime } from "./timezone";

interface DataHealthStats {
  symbolsEvaluatedTotal: number;
  validPricesTotal: number;
  validIndicatorsTotal: number;
  rawSignalsTotal: number;
  tickCount: number;
  barsMinTotal: number;      // Minimum bars fetched across all ticks
  barsReq: number;           // Required bars for Tier 2 (constant: 200)
  barsReqTier1: number;      // Required bars for Tier 1 (constant: 130)
  timeframe: string;         // Timeframe used (constant: 5Min)
  tierMinTotal: number;      // Minimum tier across all symbols (0/1/2)
  tier1Total: number;        // Count of tier 1 symbols across ticks
  tier2Total: number;        // Count of tier 2 symbols across ticks
}

let stats: DataHealthStats = {
  symbolsEvaluatedTotal: 0,
  validPricesTotal: 0,
  validIndicatorsTotal: 0,
  rawSignalsTotal: 0,
  tickCount: 0,
  barsMinTotal: Infinity,
  barsReq: 200,
  barsReqTier1: 130,
  timeframe: "5Min",
  tierMinTotal: 2,
  tier1Total: 0,
  tier2Total: 0,
};

let lastResetDate: string | null = null;
let analysisInProgress = false;

/**
 * Mark analysis start (for coordination with report generation)
 */
export function markAnalysisStart(): void {
  analysisInProgress = true;
}

/**
 * Record metrics for a single analysis tick (marks analysis complete)
 */
export function recordTickMetrics(metrics: {
  symbolsEvaluated: number;
  validPrices: number;
  validIndicators: number;
  rawSignals: number;
  barsMin?: number;
  tierMin?: number;
  tier1Count?: number;
  tier2Count?: number;
}): void {
  checkAndResetDaily();
  
  stats.symbolsEvaluatedTotal += metrics.symbolsEvaluated;
  stats.validPricesTotal += metrics.validPrices;
  stats.validIndicatorsTotal += metrics.validIndicators;
  stats.rawSignalsTotal += metrics.rawSignals;
  stats.tickCount++;
  
  // Track minimum bars across ticks
  if (metrics.barsMin !== undefined && metrics.barsMin < stats.barsMinTotal) {
    stats.barsMinTotal = metrics.barsMin;
  }
  
  // Track tier info
  if (metrics.tierMin !== undefined && metrics.tierMin < stats.tierMinTotal) {
    stats.tierMinTotal = metrics.tierMin;
  }
  if (metrics.tier1Count !== undefined) {
    stats.tier1Total += metrics.tier1Count;
  }
  if (metrics.tier2Count !== undefined) {
    stats.tier2Total += metrics.tier2Count;
  }
  
  analysisInProgress = false;
  
  const et = getEasternTime();
  const barsInfo = metrics.barsMin !== undefined 
    ? ` | barsReq=${stats.barsReq} | barsMin=${metrics.barsMin} | tf=${stats.timeframe}` 
    : "";
  const tierInfo = metrics.tierMin !== undefined 
    ? ` | tier=${metrics.tierMin} (t1=${metrics.tier1Count || 0}, t2=${metrics.tier2Count || 0})`
    : "";
  console.log(`[DATA] ${et.displayTime} | symbolsEvaluated=${metrics.symbolsEvaluated} | validPrices=${metrics.validPrices} | validIndicators=${metrics.validIndicators} | rawSignalsThisTick=${metrics.rawSignals}${barsInfo}${tierInfo}`);
}

/**
 * Check if first tick completed
 */
export function hasFirstTickCompleted(): boolean {
  return stats.tickCount > 0;
}

/**
 * Check if analysis is in progress
 */
export function isAnalysisInProgress(): boolean {
  return analysisInProgress;
}

/**
 * Get data health statistics
 */
export function getDataHealthStats(): DataHealthStats {
  return { ...stats };
}

/**
 * Print summary for debugging (called before report generation)
 */
export function printDataSummary(): void {
  const et = getEasternTime();
  console.log(`[DATA_SUMMARY] ${et.displayTime} | tickCount=${stats.tickCount} symbolsEvaluatedTotal=${stats.symbolsEvaluatedTotal} validPricesTotal=${stats.validPricesTotal} validIndicatorsTotal=${stats.validIndicatorsTotal} rawSignalsTotal=${stats.rawSignalsTotal}`);
}

/**
 * Get diagnosis based on data health
 */
export function getDataHealthDiagnosis(executedTotal: number): string[] {
  const lines: string[] = [];
  
  if (stats.validIndicatorsTotal === 0 && stats.tickCount > 0) {
    lines.push("DATA HEALTH FAIL: Indicators not computed. Check data fetch / indicator pipeline.");
  } else if (stats.validIndicatorsTotal > 0 && stats.rawSignalsTotal === 0) {
    lines.push("NO SIGNALS: Strategy conditions not met (logic too strict or market conditions).");
  } else if (stats.rawSignalsTotal > 0 && executedTotal === 0) {
    lines.push("FILTER BLOCKING: Signals exist but prevented from executing. Check SKIP_STATS.");
  } else if (stats.rawSignalsTotal > 0 && executedTotal > 0) {
    const convRate = ((executedTotal / stats.rawSignalsTotal) * 100).toFixed(1);
    lines.push(`SIGNAL CONVERSION: ${convRate}% of raw signals executed.`);
  }
  
  if (stats.validPricesTotal === 0 && stats.symbolsEvaluatedTotal > 0) {
    lines.push("WARNING: No valid prices obtained. Check Alpaca API connection.");
  }
  
  return lines;
}

/**
 * Check if we need to reset for new day
 */
function checkAndResetDaily(): void {
  const et = getEasternTime();
  const todayDate = et.dateString;
  
  if (!lastResetDate || !lastResetDate.startsWith(todayDate)) {
    resetDataHealthStats();
    lastResetDate = todayDate;
    console.log(`[DATA_HEALTH] Daily reset for ${todayDate}`);
  }
  
  if (et.hour === 9 && et.minute === 30) {
    const minuteKey = `${todayDate}-09:30`;
    if (lastResetDate !== minuteKey) {
      resetDataHealthStats();
      lastResetDate = minuteKey;
      console.log(`[DATA_HEALTH] Market open reset at 9:30 ET`);
    }
  }
}

/**
 * Reset all data health stats
 */
export function resetDataHealthStats(): void {
  stats = {
    symbolsEvaluatedTotal: 0,
    validPricesTotal: 0,
    validIndicatorsTotal: 0,
    rawSignalsTotal: 0,
    tickCount: 0,
    barsMinTotal: Infinity,
    barsReq: 200,
    barsReqTier1: 130,
    timeframe: "5Min",
    tierMinTotal: 2,
    tier1Total: 0,
    tier2Total: 0,
  };
}
