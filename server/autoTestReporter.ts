/**
 * Auto Test Reporter - Automated Monday Paper Test Reporter
 * 
 * Features:
 * - Auto-DRY-RUN window: 9:30-9:40 ET (first 10 minutes)
 * - Auto-generate reports at 10:30 ET and 12:05 ET (once per day each)
 * - Controlled by AUTO_TEST_MODE=1 env variable
 */

import * as fs from 'fs';
import * as path from 'path';
import { getEasternTime } from './timezone';
import { getSignalStats } from './signalCounters';
import { getTopSkipReasons, getSkipCounts } from './skipCounters';
import { getDataHealthStats, getDataHealthDiagnosis, printDataSummary, hasFirstTickCompleted, isAnalysisInProgress } from './dataHealthCounters';
import { DAY_TRADER_CONFIG } from './dayTraderConfig';
import * as tradeLifecycle from './tradeLifecycle';

// Track which reports have been generated today (by date and time slot)
const generatedReports: Set<string> = new Set();

// Report time slots (HH:MM format and file suffix)
const REPORT_SLOTS = [
  { hour: 10, minute: 30, suffix: '1030', label: '10:30 ET' },
  { hour: 12, minute: 5, suffix: '1205', label: '12:05 ET' },
];

/**
 * Check if AUTO_TEST_MODE is enabled
 */
export function isAutoTestModeEnabled(): boolean {
  return process.env.AUTO_TEST_MODE === "1";
}

/**
 * Determine if we should force DRY_RUN (9:30-9:40 ET on market open days)
 * This is checked internally and overrides the env DRY_RUN setting
 */
export function shouldForceDryRun(marketOpen: boolean): boolean {
  if (!isAutoTestModeEnabled()) {
    return false;
  }
  
  if (!marketOpen) {
    return false;
  }
  
  const et = getEasternTime();
  
  // Force DRY_RUN between 9:30 and 9:39 (inclusive)
  if (et.hour === 9 && et.minute >= 30 && et.minute < 40) {
    return true;
  }
  
  return false;
}

/**
 * Check if DRY_RUN is effectively active (env OR forced)
 */
export function isDryRunEffective(marketOpen: boolean): boolean {
  // Check env DRY_RUN first
  if (process.env.DRY_RUN === "1") {
    return true;
  }
  
  // Check auto-forced DRY_RUN
  if (shouldForceDryRun(marketOpen)) {
    return true;
  }
  
  return false;
}

/**
 * Get pending report slots that should be generated
 */
function getPendingReportSlots(marketOpen: boolean): typeof REPORT_SLOTS {
  if (!isAutoTestModeEnabled() || !marketOpen) {
    return [];
  }
  
  const et = getEasternTime();
  const pendingSlots: typeof REPORT_SLOTS = [];
  
  for (const slot of REPORT_SLOTS) {
    const reportKey = `${et.dateString}_${slot.suffix}`;
    
    // Skip if already generated
    if (generatedReports.has(reportKey)) {
      continue;
    }
    
    // Check if current time is at or past this slot's time
    if (et.hour > slot.hour || (et.hour === slot.hour && et.minute >= slot.minute)) {
      pendingSlots.push(slot);
    }
  }
  
  return pendingSlots;
}

/**
 * Check if any report should be generated
 */
export function shouldGenerateReport(marketOpen: boolean): boolean {
  return getPendingReportSlots(marketOpen).length > 0;
}

/**
 * Get auto-test status for logging
 */
export function getAutoTestStatus(marketOpen: boolean): {
  enabled: boolean;
  forceDryRun: boolean;
  dryRunEffective: boolean;
  reportPending: boolean;
} {
  const enabled = isAutoTestModeEnabled();
  const forceDryRun = shouldForceDryRun(marketOpen);
  const dryRunEffective = isDryRunEffective(marketOpen);
  const reportPending = getPendingReportSlots(marketOpen).length > 0;
  
  return {
    enabled,
    forceDryRun,
    dryRunEffective,
    reportPending,
  };
}

/**
 * Generate the auto test report content
 */
export function generateReport(
  clock: { is_open: boolean; next_open: string; next_close: string },
  timeGuardStatus: { canOpenNewTrades: boolean; canManagePositions: boolean; shouldForceClose: boolean; reason: string },
  marketOpen: boolean,
  slotLabel: string
): string {
  // Debug: Print data summary before generating report
  printDataSummary();
  
  const et = getEasternTime();
  const signalStats = getSignalStats();
  const topSkips = getTopSkipReasons(5);
  const dryRunEffective = isDryRunEffective(marketOpen);
  const signalsBlocked = signalStats.totalSignals - signalStats.totalExecutions;
  const dataHealth = getDataHealthStats();
  const dataHealthDiagnosis = getDataHealthDiagnosis(signalStats.totalExecutions);
  
  const lines: string[] = [];
  
  lines.push("================================================================================");
  lines.push("                    ATOBOT AUTOMATED PAPER TEST REPORT");
  lines.push("================================================================================");
  lines.push("");
  lines.push(`Report Date: ${et.dateString}`);
  lines.push(`Report Time: ${slotLabel}`);
  lines.push(`Generated (ET): ${et.dateString} ${et.displayTime}`);
  lines.push(`Generated (UTC): ${new Date().toISOString()}`);
  lines.push("");
  lines.push("--- AUTO_TEST STATUS ---");
  lines.push(`AUTO_TEST_MODE: ON`);
  lines.push(`Effective DRY_RUN: ${dryRunEffective ? 'ON' : 'OFF'}`);
  lines.push(`EXECUTED_TOTAL: ${signalStats.totalExecutions}`);
  lines.push(`TotalSignals: ${signalStats.totalSignals}`);
  lines.push(`SignalsBlocked: ${signalsBlocked}`);
  lines.push("");
  
  lines.push("--- MARKET STATUS ---");
  lines.push(`Alpaca Clock: is_open=${clock.is_open}`);
  lines.push(`Next Open: ${clock.next_open}`);
  lines.push(`Next Close: ${clock.next_close}`);
  lines.push("");
  
  lines.push("--- TIME GUARD STATUS ---");
  lines.push(`Can Open New Trades: ${timeGuardStatus.canOpenNewTrades}`);
  lines.push(`Can Manage Positions: ${timeGuardStatus.canManagePositions}`);
  lines.push(`Should Force Close: ${timeGuardStatus.shouldForceClose}`);
  lines.push(`Current Reason: ${timeGuardStatus.reason}`);
  lines.push("");
  
  lines.push("--- DATA HEALTH ---");
  lines.push(`tickCount: ${dataHealth.tickCount}`);
  lines.push(`symbolsEvaluatedTotal: ${dataHealth.symbolsEvaluatedTotal}`);
  lines.push(`validPricesTotal: ${dataHealth.validPricesTotal}`);
  lines.push(`validIndicatorsTotal: ${dataHealth.validIndicatorsTotal}`);
  lines.push(`rawSignalsTotal: ${dataHealth.rawSignalsTotal}`);
  if (dataHealth.barsMinTotal !== Infinity) {
    lines.push(`barsReq: ${dataHealth.barsReq}`);
    lines.push(`barsMin: ${dataHealth.barsMinTotal}`);
    lines.push(`timeframe: ${dataHealth.timeframe}`);
  }
  if (dataHealth.tickCount === 0) {
    lines.push("NOTE: No analysis ticks occurred before report time.");
  }
  lines.push("");
  
  const dataFailureReasons = ["NO_BARS_RETURNED", "INSUFFICIENT_BARS", "INVALID_CLOSE_VALUES", "API_ERROR", "NAN_INDICATORS"];
  const dataFailures: Array<{ reason: string; count: number }> = [];
  const allSkips = getSkipCounts();
  for (const reason of dataFailureReasons) {
    const count = allSkips.get(reason) || 0;
    if (count > 0) {
      dataFailures.push({ reason, count });
    }
  }
  
  if (dataFailures.length > 0) {
    lines.push("--- DATA FAILURE BREAKDOWN ---");
    for (const { reason, count } of dataFailures) {
      lines.push(`  ${reason}: ${count}`);
    }
    lines.push("");
  }
  
  lines.push("--- SIGNAL STATS (Funnel) ---");
  lines.push(`Total Signals Generated: ${signalStats.totalSignals}`);
  lines.push(`Total Executions: ${signalStats.totalExecutions}`);
  lines.push(`Candidates Evaluated: ${signalStats.candidatesTotal}`);
  lines.push("");
  
  if (Object.keys(signalStats.signals).length > 0) {
    lines.push("Signals by Type:");
    for (const [key, count] of Object.entries(signalStats.signals)) {
      lines.push(`  ${key}: ${count}`);
    }
    lines.push("");
  }
  
  if (Object.keys(signalStats.executions).length > 0) {
    lines.push("Executions by Type:");
    for (const [key, count] of Object.entries(signalStats.executions)) {
      lines.push(`  ${key}: ${count}`);
    }
    lines.push("");
  }
  
  lines.push("--- SKIP STATS (Top 5 Reasons) ---");
  if (topSkips.length === 0) {
    lines.push("No skips recorded.");
  } else {
    for (let i = 0; i < topSkips.length; i++) {
      lines.push(`  ${i + 1}) ${topSkips[i].reason}: ${topSkips[i].count}`);
    }
  }
  lines.push("");
  
  lines.push("--- EXECUTIONS SUMMARY ---");
  if (signalStats.totalExecutions === 0) {
    lines.push("Executions: 0 (paper). Use SKIP_STATS to tune filters.");
  } else {
    const execSymbols: string[] = [];
    for (const key of Object.keys(signalStats.executions)) {
      if (key !== "EXECUTED_TOTAL" && signalStats.executions[key] > 0) {
        execSymbols.push(key);
      }
    }
    lines.push(`Executions: ${signalStats.totalExecutions} (paper)`);
    lines.push(`Symbols: ${execSymbols.join(", ") || "N/A"}`);
  }
  lines.push("");
  
  // P2: Trade Lifecycle Stats (realized fills, slippage)
  const tradeSummary = tradeLifecycle.getDailyTradeSummary();
  lines.push("--- P2 TRADE LIFECYCLE (Realized Fills) ---");
  lines.push(`Total Trades: ${tradeSummary.totalTrades}`);
  lines.push(`Filled Entries: ${tradeSummary.filledTrades}`);
  lines.push(`Closed Trades: ${tradeSummary.closedTrades}`);
  lines.push(`Open Trades: ${tradeSummary.openTrades}`);
  lines.push(`Wins: ${tradeSummary.wins} | Losses: ${tradeSummary.losses}`);
  lines.push(`Realized P&L: $${tradeSummary.totalRealizedPnl.toFixed(2)}`);
  lines.push("");
  
  // P2: Slippage Stats
  const slippage = tradeSummary.slippage;
  if (slippage.sampleCount > 0) {
    lines.push("--- P2 SLIPPAGE STATS ---");
    lines.push(`Avg Slippage: ${slippage.avgSlippageBps.toFixed(1)} bps`);
    lines.push(`Median Slippage: ${slippage.medianSlippageBps.toFixed(1)} bps`);
    lines.push(`Worst Slippage: ${slippage.worstSlippageBps.toFixed(1)} bps`);
    lines.push(`Sample Count: ${slippage.sampleCount}`);
    lines.push("");
  }
  
  lines.push("--- CONFIGURATION ---");
  lines.push(`Scan Interval: ${DAY_TRADER_CONFIG.SCAN_INTERVAL_MINUTES} minutes`);
  lines.push(`Entry Window: 9:35 AM - 11:35 AM ET`);
  lines.push(`Force Close: 3:45 PM ET`);
  lines.push(`Universe: ${DAY_TRADER_CONFIG.ALLOWED_SYMBOLS.join(", ")}`);
  lines.push(`Max Positions: ${DAY_TRADER_CONFIG.MAX_OPEN_POSITIONS}`);
  lines.push(`Max Entries/Day: ${DAY_TRADER_CONFIG.MAX_NEW_ENTRIES_PER_DAY}`);
  lines.push(`Daily Kill Threshold: -$${Math.abs(DAY_TRADER_CONFIG.DAILY_MAX_LOSS)} / +$${DAY_TRADER_CONFIG.DAILY_MAX_PROFIT}`);
  lines.push("");
  
  lines.push("--- DIAGNOSIS ---");
  
  // Data health diagnosis (priority)
  if (dataHealthDiagnosis.length > 0) {
    for (const line of dataHealthDiagnosis) {
      lines.push(line);
    }
  }
  
  // Additional funnel diagnosis
  if (signalStats.totalSignals === 0 && dataHealth.validIndicatorsTotal > 0) {
    lines.push("DIAGNOSIS: NO SIGNALS generated (data valid).");
    lines.push("  - Strategy may be too strict (NO_SIGNAL dominant)");
    lines.push("  - Check if CHOP_REGIME is blocking (market filter)");
    lines.push("  - Verify market was open during entry window");
  } else if (signalStats.totalSignals === 0 && dataHealth.validIndicatorsTotal === 0) {
    lines.push("DIAGNOSIS: NO SIGNALS - check DATA_HEALTH above.");
  } else if (signalStats.totalExecutions === 0 && signalStats.totalSignals > 0) {
    lines.push("DIAGNOSIS: Signals exist but 0 executed.");
    lines.push("  - Filters are blocking all signals");
    lines.push("  - Check top SKIP reasons above");
    lines.push("  - Common blockers: CHOP_REGIME, MAX_POSITIONS, RISK_CHECK_FAILED");
  } else if (signalsBlocked > signalStats.totalExecutions * 2) {
    lines.push(`DIAGNOSIS: High block rate (${signalsBlocked} blocked vs ${signalStats.totalExecutions} executed).`);
    lines.push("  - Filters may be too strict");
    lines.push("  - Review top SKIP reasons to tune");
  } else if (signalStats.totalExecutions > 0) {
    const convRate = ((signalStats.totalExecutions / signalStats.totalSignals) * 100).toFixed(1);
    lines.push(`DIAGNOSIS: Funnel conversion rate = ${convRate}%`);
    lines.push("  - System is generating and executing trades");
    lines.push("  - Review skip reasons to optimize further");
    lines.push("  - If P&L flat, review exit logic tuning");
  }
  lines.push("");
  
  lines.push("================================================================================");
  lines.push("                           END OF REPORT");
  lines.push("================================================================================");
  
  return lines.join("\n");
}

/**
 * Save a single report for a specific time slot
 * Idempotent: checks if file exists before generating
 * Defers if analysis is in progress and no ticks have completed
 */
function saveReportForSlot(
  clock: { is_open: boolean; next_open: string; next_close: string },
  timeGuardStatus: { canOpenNewTrades: boolean; canManagePositions: boolean; shouldForceClose: boolean; reason: string },
  marketOpen: boolean,
  slot: typeof REPORT_SLOTS[0]
): string | null {
  const et = getEasternTime();
  
  // Defer if analysis is in progress and no ticks have completed yet
  if (isAnalysisInProgress() && !hasFirstTickCompleted()) {
    console.log(`[AUTO_TEST] Deferring report generation - analysis in progress, waiting for first tick...`);
    return null;
  }
  
  // Create reports directory if it doesn't exist
  const reportsDir = path.join(process.cwd(), 'reports');
  if (!fs.existsSync(reportsDir)) {
    fs.mkdirSync(reportsDir, { recursive: true });
  }
  
  // Build filename with time suffix
  const filename = `paper_test_${et.dateString}_${slot.suffix}.md`;
  const filepath = path.join(reportsDir, filename);
  const reportKey = `${et.dateString}_${slot.suffix}`;
  
  // Check if report already exists (idempotent - prevent duplicates after restart)
  if (fs.existsSync(filepath)) {
    console.log(`[AUTO_TEST] Report already exists: ${filename}, skipping.`);
    generatedReports.add(reportKey);
    return filepath;
  }
  
  // Generate report content
  const reportContent = generateReport(clock, timeGuardStatus, marketOpen, slot.label);
  
  try {
    // Write time-specific report
    fs.writeFileSync(filepath, reportContent, 'utf8');
    console.log(`[AUTO_TEST] Report written: ${filepath}`);
    
    // Update latest.md for easy retrieval
    const latestPath = path.join(reportsDir, 'latest.md');
    fs.writeFileSync(latestPath, reportContent, 'utf8');
    console.log(`[AUTO_TEST] Updated reports/latest.md -> ${filename}`);
    
    generatedReports.add(reportKey);
    return filepath;
  } catch (error) {
    console.error(`[AUTO_TEST] Failed to write report: ${error}`);
    return null;
  }
}

/**
 * Save all pending reports
 */
export function saveReport(
  clock: { is_open: boolean; next_open: string; next_close: string },
  timeGuardStatus: { canOpenNewTrades: boolean; canManagePositions: boolean; shouldForceClose: boolean; reason: string },
  marketOpen: boolean
): string | null {
  const pendingSlots = getPendingReportSlots(marketOpen);
  let lastPath: string | null = null;
  
  for (const slot of pendingSlots) {
    const result = saveReportForSlot(clock, timeGuardStatus, marketOpen, slot);
    if (result) {
      lastPath = result;
    }
  }
  
  return lastPath;
}

/**
 * Check and handle auto-test tasks (called from control loop)
 */
export function handleAutoTestTasks(
  marketOpen: boolean,
  clock: { is_open: boolean; next_open: string; next_close: string },
  timeGuardStatus: { canOpenNewTrades: boolean; canManagePositions: boolean; shouldForceClose: boolean; reason: string }
): void {
  if (!isAutoTestModeEnabled()) {
    return;
  }
  
  const pendingSlots = getPendingReportSlots(marketOpen);
  const et = getEasternTime();
  
  if (pendingSlots.length > 0) {
    console.log(`[AUTO_TEST] Report pending for slots: ${pendingSlots.map(s => s.label).join(', ')}`);
    saveReport(clock, timeGuardStatus, marketOpen);
  } else if (generatedReports.size === 0) {
    // No reports generated yet today, show pending status
    const nextSlot = REPORT_SLOTS.find(s => {
      return et.hour < s.hour || (et.hour === s.hour && et.minute < s.minute);
    });
    if (nextSlot) {
      console.log(`[AUTO_TEST] Next report at ${nextSlot.label}`);
    }
  }
}

/**
 * Reset for new day (for testing)
 */
export function resetForNewDay(): void {
  generatedReports.clear();
}
