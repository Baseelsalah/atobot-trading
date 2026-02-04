/**
 * Activity Ledger - Persistent per-tick scan summaries
 * 
 * Provides truthful reporting by recording actual bot activity.
 * Each scan cycle writes a tick record with evaluation counts.
 */

import * as fs from "fs";
import * as path from "path";
import { getEasternTime, toEasternDateString, getPtDateString } from "./timezone";
import * as tradeAccounting from "./tradeAccounting";
import * as reportStorage from "./reportStorage";

let notifyTickCallback: ((tickET: string) => void) | null = null;

export function registerTickCallback(callback: (tickET: string) => void): void {
  notifyTickCallback = callback;
}

export interface TickRecord {
  tickId: string;
  tsET: string;
  ptDate: string;
  etDate: string;
  symbolsEvaluated: number;
  validQuotes: number;
  validBars: number;
  noSignalCount: number;
  skipCount: number;
  skipReasonCounts: Record<string, number>;
  signalsGenerated: number;
  tradesProposed: number;
  tradesSubmitted: number;
  tradesFilled: number;
}

export interface ActivitySummary {
  botWasRunning: boolean;
  scanTicks: number;
  symbolsEvaluated: number;
  totalSkips: number;
  topSkipReasons: Array<{ reason: string; count: number; percent: number }>;
  noSignalCount: number;
  signalsGenerated: number;
  tradesProposed: number;
  tradesSubmitted: number;
  tradesFilled: number;
  firstTickET: string | null;
  lastTickET: string | null;
  // Trade accounting (authoritative Alpaca visibility)
  tradeAccounting?: {
    proposed: number;
    submitted: number;
    rejected: number;
    suppressed: number;
    canceled: number;
    lastRejectionReason: string | null;
    topSuppressReasons: Array<{ reason: string; count: number }>;
  };
}

const LEDGER_DIR = "reports/activity";

let tickCounter = 0;
const inMemoryTicks: TickRecord[] = [];
let lastPersistDate: string | null = null;
let counterInitialized = false;

function initializeTickCounter(): void {
  if (counterInitialized) return;
  
  try {
    const et = getEasternTime();
    const existingTicks = loadDayTicks(et.dateString);
    if (existingTicks.length > 0) {
      // Find the highest tick number from existing ticks
      let maxNum = 0;
      for (const tick of existingTicks) {
        const match = tick.tickId.match(/_(\d{6})$/);
        if (match) {
          const num = parseInt(match[1], 10);
          if (num > maxNum) maxNum = num;
        }
      }
      tickCounter = maxNum;
    }
    counterInitialized = true;
  } catch {
    counterInitialized = true;
  }
}

function ensureLedgerDir(): void {
  if (!fs.existsSync(LEDGER_DIR)) {
    fs.mkdirSync(LEDGER_DIR, { recursive: true });
  }
}

function getLedgerFilePath(etDate: string): string {
  return path.join(LEDGER_DIR, `activity_${etDate}.json`);
}

function loadDayTicks(etDate: string): TickRecord[] {
  const filePath = getLedgerFilePath(etDate);
  if (!fs.existsSync(filePath)) {
    return [];
  }
  try {
    const data = fs.readFileSync(filePath, "utf-8");
    return JSON.parse(data) as TickRecord[];
  } catch (error) {
    console.log(`[ActivityLedger] Error loading ${filePath}: ${error}`);
    return [];
  }
}

function saveDayTicks(etDate: string, ticks: TickRecord[]): void {
  ensureLedgerDir();
  const filePath = getLedgerFilePath(etDate);
  const content = JSON.stringify(ticks, null, 2);
  try {
    fs.writeFileSync(filePath, content);
  } catch (error) {
    console.log(`[ActivityLedger] Error saving ${filePath}: ${error}`);
  }
  reportStorage.putText("activity", `activity_${etDate}.json`, content).catch(() => {});
}

export function recordTick(data: {
  symbolsEvaluated: number;
  validQuotes: number;
  validBars: number;
  noSignalCount: number;
  skipCount: number;
  skipReasonCounts: Record<string, number>;
  signalsGenerated?: number;
  tradesProposed?: number;
  tradesSubmitted?: number;
  tradesFilled?: number;
}): void {
  const et = getEasternTime();
  const ptDate = getPtDateString(new Date());
  
  // Use timestamp-based ID to guarantee uniqueness across restarts
  const now = Date.now();
  const tickId = `tick_${et.dateString.replace(/-/g, "")}_${now}`;
  
  const tick: TickRecord = {
    tickId,
    tsET: et.displayTime,
    ptDate,
    etDate: et.dateString,
    symbolsEvaluated: data.symbolsEvaluated,
    validQuotes: data.validQuotes,
    validBars: data.validBars,
    noSignalCount: data.noSignalCount,
    skipCount: data.skipCount,
    skipReasonCounts: { ...data.skipReasonCounts },
    signalsGenerated: data.signalsGenerated || 0,
    tradesProposed: data.tradesProposed || 0,
    tradesSubmitted: data.tradesSubmitted || 0,
    tradesFilled: data.tradesFilled || 0,
  };
  
  inMemoryTicks.push(tick);
  
  if (lastPersistDate !== et.dateString) {
    lastPersistDate = et.dateString;
    tickCounter = inMemoryTicks.length;
  }
  
  if (inMemoryTicks.length % 5 === 0 || data.signalsGenerated) {
    persistCurrentDayTicks();
  }
  
  console.log(`[ActivityLedger] Tick ${tickId}: symbols=${data.symbolsEvaluated} quotes=${data.validQuotes} bars=${data.validBars} skips=${data.skipCount} noSignal=${data.noSignalCount}`);
  
  // Notify runtime monitor of tick
  if (notifyTickCallback) {
    notifyTickCallback(tick.tsET);
  }
}

function persistCurrentDayTicks(): void {
  const et = getEasternTime();
  const existingTicks = loadDayTicks(et.dateString);
  
  const existingIds = new Set(existingTicks.map(t => t.tickId));
  const newTicks = inMemoryTicks.filter(t => t.etDate === et.dateString && !existingIds.has(t.tickId));
  
  if (newTicks.length > 0) {
    const allTicks = [...existingTicks, ...newTicks];
    saveDayTicks(et.dateString, allTicks);
  }
}

export function flushToDisk(): void {
  persistCurrentDayTicks();
  console.log(`[ActivityLedger] Flushed ${inMemoryTicks.length} ticks to disk`);
}

// Register shutdown flush to prevent data loss in low-volume sessions
process.on("beforeExit", () => {
  if (inMemoryTicks.length > 0) {
    persistCurrentDayTicks();
    console.log(`[ActivityLedger] Shutdown flush: ${inMemoryTicks.length} ticks saved`);
  }
});

process.on("SIGTERM", () => {
  if (inMemoryTicks.length > 0) {
    persistCurrentDayTicks();
    console.log(`[ActivityLedger] SIGTERM flush: ${inMemoryTicks.length} ticks saved`);
  }
});

process.on("SIGINT", () => {
  if (inMemoryTicks.length > 0) {
    persistCurrentDayTicks();
    console.log(`[ActivityLedger] SIGINT flush: ${inMemoryTicks.length} ticks saved`);
  }
});

export function getActivitySummary(dateOrRange: string | { start: string; end: string }): ActivitySummary {
  let dates: string[];
  
  if (typeof dateOrRange === "string") {
    dates = [dateOrRange];
  } else {
    dates = [];
    // Parse dates with noon UTC to avoid timezone boundary issues
    const startDate = new Date(dateOrRange.start + "T12:00:00Z");
    const endDate = new Date(dateOrRange.end + "T12:00:00Z");
    for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
      // Use the date string directly from the Date object in UTC
      const year = d.getUTCFullYear();
      const month = String(d.getUTCMonth() + 1).padStart(2, "0");
      const day = String(d.getUTCDate()).padStart(2, "0");
      dates.push(`${year}-${month}-${day}`);
    }
  }
  
  const allTicks: TickRecord[] = [];
  for (const date of dates) {
    const dayTicks = loadDayTicks(date);
    allTicks.push(...dayTicks);
    
    const inMemoryForDay = inMemoryTicks.filter(t => t.etDate === date);
    const existingIds = new Set(dayTicks.map(t => t.tickId));
    for (const tick of inMemoryForDay) {
      if (!existingIds.has(tick.tickId)) {
        allTicks.push(tick);
      }
    }
  }
  
  if (allTicks.length === 0) {
    return {
      botWasRunning: false,
      scanTicks: 0,
      symbolsEvaluated: 0,
      totalSkips: 0,
      topSkipReasons: [],
      noSignalCount: 0,
      signalsGenerated: 0,
      tradesProposed: 0,
      tradesSubmitted: 0,
      tradesFilled: 0,
      firstTickET: null,
      lastTickET: null,
    };
  }
  
  const skipReasonTotals: Record<string, number> = {};
  let totalSymbols = 0;
  let totalSkips = 0;
  let totalNoSignal = 0;
  let totalSignalsGenerated = 0;
  let totalTradesProposed = 0;
  let totalTradesSubmitted = 0;
  let totalTradesFilled = 0;
  
  for (const tick of allTicks) {
    totalSymbols += tick.symbolsEvaluated;
    totalSkips += tick.skipCount;
    totalNoSignal += tick.noSignalCount;
    totalSignalsGenerated += tick.signalsGenerated || 0;
    totalTradesProposed += tick.tradesProposed || 0;
    totalTradesSubmitted += tick.tradesSubmitted || 0;
    totalTradesFilled += tick.tradesFilled || 0;
    
    for (const [reason, count] of Object.entries(tick.skipReasonCounts)) {
      skipReasonTotals[reason] = (skipReasonTotals[reason] || 0) + count;
    }
  }
  
  const topSkipReasons = Object.entries(skipReasonTotals)
    .map(([reason, count]) => ({
      reason,
      count,
      percent: totalSkips > 0 ? (count / totalSkips) * 100 : 0,
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);
  
  allTicks.sort((a, b) => a.tickId.localeCompare(b.tickId));
  
  // Get authoritative trade accounting data - aggregate from all days in range
  let totalProposed = 0;
  let totalSubmitted = 0;
  let totalRejected = 0;
  let totalSuppressed = 0;
  let totalCanceled = 0;
  let lastRejectionReason: string | null = null;
  const aggregatedSuppressReasons: Record<string, number> = {};
  
  for (const date of dates) {
    const dayAccounting = tradeAccounting.getAccountingForDate(date);
    if (dayAccounting) {
      totalProposed += dayAccounting.tradesProposed;
      totalSubmitted += dayAccounting.tradesSubmitted;
      totalRejected += dayAccounting.tradesRejected;
      totalSuppressed += dayAccounting.tradesSuppressed;
      totalCanceled += dayAccounting.tradesCanceled;
      if (dayAccounting.lastRejectionReason) {
        lastRejectionReason = dayAccounting.lastRejectionReason;
      }
      for (const [reason, count] of Object.entries(dayAccounting.suppressReasonCounts)) {
        aggregatedSuppressReasons[reason] = (aggregatedSuppressReasons[reason] || 0) + count;
      }
    }
  }
  
  const topSuppressReasons = Object.entries(aggregatedSuppressReasons)
    .map(([reason, count]) => ({ reason, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 5);
  
  return {
    botWasRunning: true,
    scanTicks: allTicks.length,
    symbolsEvaluated: totalSymbols,
    totalSkips,
    topSkipReasons,
    noSignalCount: totalNoSignal,
    signalsGenerated: totalSignalsGenerated,
    tradesProposed: totalTradesProposed,
    tradesSubmitted: totalTradesSubmitted,
    tradesFilled: totalTradesFilled,
    firstTickET: allTicks[0]?.tsET || null,
    lastTickET: allTicks[allTicks.length - 1]?.tsET || null,
    tradeAccounting: {
      proposed: totalProposed,
      submitted: totalSubmitted,
      rejected: totalRejected,
      suppressed: totalSuppressed,
      canceled: totalCanceled,
      lastRejectionReason,
      topSuppressReasons,
    },
  };
}

export function getTodaysSummary(): ActivitySummary {
  const et = getEasternTime();
  return getActivitySummary(et.dateString);
}

export function resetDailyLedger(): void {
  const et = getEasternTime();
  
  persistCurrentDayTicks();
  
  const previousDayTicks = inMemoryTicks.filter(t => t.etDate !== et.dateString);
  inMemoryTicks.length = 0;
  
  tickCounter = 0;
  lastPersistDate = et.dateString;
  
  console.log(`[ActivityLedger] Daily reset for ${et.dateString}, archived ${previousDayTicks.length} previous ticks`);
}

export interface FlaggedTicksResult {
  dateET: string;
  summary: ActivitySummary;
  flaggedTicks: TickRecord[];
  ticksFlaggedCount: number;
}

export function getFlaggedTicks(dateET: string): FlaggedTicksResult {
  const dayTicks = loadDayTicks(dateET);
  const inMemoryForDay = inMemoryTicks.filter(t => t.etDate === dateET);
  const existingIds = new Set(dayTicks.map(t => t.tickId));
  
  const allTicks: TickRecord[] = [...dayTicks];
  for (const tick of inMemoryForDay) {
    if (!existingIds.has(tick.tickId)) {
      allTicks.push(tick);
    }
  }
  
  const flaggedTicks = allTicks.filter(tick => {
    if (tick.tradesProposed > 0) return true;
    const skipReasons = Object.keys(tick.skipReasonCounts || {});
    if (skipReasons.includes("EXECUTION_FAILED") || skipReasons.includes("RISK_CHECK_FAILED")) {
      return true;
    }
    return false;
  });
  
  return {
    dateET,
    summary: getActivitySummary(dateET),
    flaggedTicks,
    ticksFlaggedCount: flaggedTicks.length,
  };
}
