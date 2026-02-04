/**
 * Trade Accounting - Authoritative trade attempt counters
 * 
 * Tracks the full lifecycle of trade attempts:
 * - Proposed: Signals that passed validation and would trade
 * - Submitted: Orders accepted by Alpaca (order_id received)
 * - Rejected: Orders rejected by Alpaca
 * - Suppressed: Blocked by idempotency/preflight gates
 * 
 * INVARIANT: proposed = submitted + rejected + suppressed
 * 
 * PERSISTENCE: Counters are saved to disk and survive restarts.
 */

import { getEasternTime } from "./timezone";
import * as fs from "fs";
import * as path from "path";
import * as reportStorage from "./reportStorage";

export type TradeAction = "PROPOSE" | "SUBMIT_OK" | "SUBMIT_FAIL" | "SUPPRESS" | "CANCEL";

export interface OrderEvent {
  timestampET: string;
  tradeId: string;
  symbol: string;
  strategyName: string;
  action: TradeAction;
  limitPrice: number | null;
  qty: number;
  alpacaOrderId: string | null;
  reason: string | null;
  errorCode: string | null;
  errorMessage: string | null;
}

export interface DailyAccounting {
  date: string;
  tradesProposed: number;
  tradesSubmitted: number;
  tradesRejected: number;
  tradesSuppressed: number;
  tradesCanceled: number;
  lastRejectionReason: string | null;
  suppressReasonCounts: Record<string, number>;
  topSuppressReasons: Array<{ reason: string; count: number; percent: number }>;
}

const MAX_ORDER_EVENTS = 10;
const ACCOUNTING_DIR = path.join(process.cwd(), "reports", "accounting");

let currentDate: string = "";
let tradesProposed = 0;
let tradesSubmitted = 0;
let tradesRejected = 0;
let tradesSuppressed = 0;
let tradesCanceled = 0;
let lastRejectionReason: string | null = null;
let suppressReasonCounts: Record<string, number> = {};
let orderEventsBuffer: OrderEvent[] = [];
let bootTimeET: string = "";
let bootPid: number = 0;
let bootVersion: string = "";
let stateInitialized: boolean = false;

// Persistence file paths
function getAccountingFilePath(dateStr: string): string {
  return path.join(ACCOUNTING_DIR, `accounting_${dateStr}.json`);
}

function getStateFilePath(dateStr: string): string {
  return path.join(ACCOUNTING_DIR, `state_${dateStr}.json`);
}

function getEventsFilePath(dateStr: string): string {
  return path.join(ACCOUNTING_DIR, `events_${dateStr}.jsonl`);
}

// Ensure accounting directory exists
function ensureAccountingDir(): void {
  if (!fs.existsSync(ACCOUNTING_DIR)) {
    fs.mkdirSync(ACCOUNTING_DIR, { recursive: true });
  }
}

// Save current counters to disk
function persistCounters(): void {
  ensureAccountingDir();
  const data = {
    date: currentDate,
    tradesProposed,
    tradesSubmitted,
    tradesRejected,
    tradesSuppressed,
    tradesCanceled,
    lastRejectionReason,
    // Clone to ensure serialization safety
    suppressReasonCounts: { ...suppressReasonCounts },
    savedAt: new Date().toISOString(),
  };
  const filePath = getAccountingFilePath(currentDate);
  const content = JSON.stringify(data, null, 2);
  fs.writeFileSync(filePath, content);
  reportStorage.putText("accounting", `accounting_${currentDate}.json`, content).catch(() => {});
}

// Save state file with boot metadata (for verification even on zero-proposal days)
function persistState(): void {
  if (!currentDate) return;
  ensureAccountingDir();
  
  const totalSuppressed = Object.values(suppressReasonCounts).reduce((a, b) => a + b, 0);
  const topSuppressReasons = Object.entries(suppressReasonCounts)
    .map(([reason, count]) => ({
      reason,
      count,
      percent: totalSuppressed > 0 ? (count / totalSuppressed) * 100 : 0,
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 5);
  
  const data = {
    dateET: currentDate,
    bootTimeET,
    version: bootVersion,
    pid: bootPid,
    counters: {
      proposed: tradesProposed,
      submitted: tradesSubmitted,
      rejected: tradesRejected,
      suppressed: tradesSuppressed,
      canceled: tradesCanceled,
    },
    lastRejectionReason,
    topSuppressReasons,
    suppressReasonCounts: { ...suppressReasonCounts },
    savedAt: new Date().toISOString(),
  };
  
  const filePath = getStateFilePath(currentDate);
  const content = JSON.stringify(data, null, 2);
  fs.writeFileSync(filePath, content);
  reportStorage.putText("accounting", `state_${currentDate}.json`, content).catch(() => {});
}

// Load state from disk
function loadStateFromDisk(dateStr: string): boolean {
  const filePath = getStateFilePath(dateStr);
  if (!fs.existsSync(filePath)) {
    return false;
  }
  
  try {
    const data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    tradesProposed = data.counters?.proposed || 0;
    tradesSubmitted = data.counters?.submitted || 0;
    tradesRejected = data.counters?.rejected || 0;
    tradesSuppressed = data.counters?.suppressed || 0;
    tradesCanceled = data.counters?.canceled || 0;
    lastRejectionReason = data.lastRejectionReason || null;
    suppressReasonCounts = { ...(data.suppressReasonCounts || {}) };
    console.log(`[TradeAccounting] Loaded state from disk: date=${dateStr} proposed=${tradesProposed} submitted=${tradesSubmitted} rejected=${tradesRejected} suppressed=${tradesSuppressed} canceled=${tradesCanceled}`);
    return true;
  } catch (err) {
    console.error(`[TradeAccounting] Error loading state from disk:`, err);
    return false;
  }
}

/**
 * Initialize trade accounting on boot - creates state file even with zeros
 */
export function initializeOnBoot(): void {
  const et = getEasternTime();
  currentDate = et.dateString;
  bootTimeET = et.displayTime;
  bootPid = process.pid;
  bootVersion = (process.env.REPL_ID || "dev").slice(0, 8);
  
  // Try to load existing state first (handles restarts within same day)
  const loadedState = loadStateFromDisk(et.dateString);
  const loadedCounters = !loadedState && loadCountersFromDisk(et.dateString);
  
  if (loadedState || loadedCounters) {
    loadOrderEventsFromDisk(et.dateString);
  } else {
    // Fresh day - counters are already zero
    console.log(`[TradeAccounting] Fresh boot for ${et.dateString} - counters at zero`);
  }
  
  // Always persist state on boot (creates file even with zeros)
  persistState();
  stateInitialized = true;
  console.log(`[TradeAccounting] State file created: state_${et.dateString}.json`);
}

/**
 * Persist state on heartbeat - call this periodically to ensure durability
 */
export function persistOnHeartbeat(): void {
  if (!stateInitialized) return;
  persistState();
  persistCounters();
}

/**
 * Persist state on shutdown
 */
export function persistOnShutdown(): void {
  if (!stateInitialized) return;
  persistState();
  persistCounters();
  console.log(`[TradeAccounting] State persisted on shutdown`);
}

// Append order event to JSONL file
function persistOrderEvent(event: OrderEvent): void {
  ensureAccountingDir();
  const filePath = getEventsFilePath(currentDate);
  const line = JSON.stringify(event) + "\n";
  fs.appendFileSync(filePath, line);
}

// Load counters from disk for a specific date
function loadCountersFromDisk(dateStr: string): boolean {
  const filePath = getAccountingFilePath(dateStr);
  if (!fs.existsSync(filePath)) {
    return false;
  }
  
  try {
    const data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    tradesProposed = data.tradesProposed || 0;
    tradesSubmitted = data.tradesSubmitted || 0;
    tradesRejected = data.tradesRejected || 0;
    tradesSuppressed = data.tradesSuppressed || 0;
    tradesCanceled = data.tradesCanceled || 0;
    lastRejectionReason = data.lastRejectionReason || null;
    // Clone suppressReasonCounts to prevent cross-day reference contamination
    suppressReasonCounts = { ...(data.suppressReasonCounts || {}) };
    console.log(`[TradeAccounting] Loaded counters from disk: date=${dateStr} proposed=${tradesProposed} submitted=${tradesSubmitted} rejected=${tradesRejected} suppressed=${tradesSuppressed}`);
    return true;
  } catch (err) {
    console.error(`[TradeAccounting] Error loading counters from disk:`, err);
    return false;
  }
}

// Load order events from disk
function loadOrderEventsFromDisk(dateStr: string): void {
  const filePath = getEventsFilePath(dateStr);
  if (!fs.existsSync(filePath)) {
    return;
  }
  
  try {
    const content = fs.readFileSync(filePath, "utf-8");
    const lines = content.trim().split("\n").filter(l => l.length > 0);
    
    // Load last N events into buffer
    const events: OrderEvent[] = [];
    for (const line of lines.slice(-MAX_ORDER_EVENTS)) {
      try {
        events.push(JSON.parse(line));
      } catch {
        // Skip malformed lines
      }
    }
    orderEventsBuffer = events;
    console.log(`[TradeAccounting] Loaded ${events.length} order events from disk`);
  } catch (err) {
    console.error(`[TradeAccounting] Error loading order events:`, err);
  }
}

function ensureCurrentDate(): void {
  const et = getEasternTime();
  if (currentDate !== et.dateString) {
    const isNewDay = currentDate !== "";
    currentDate = et.dateString;
    
    // Try to load existing counters from disk
    const loaded = loadCountersFromDisk(et.dateString);
    
    if (!loaded) {
      // Fresh day - reset counters
      tradesProposed = 0;
      tradesSubmitted = 0;
      tradesRejected = 0;
      tradesSuppressed = 0;
      tradesCanceled = 0;
      lastRejectionReason = null;
      suppressReasonCounts = {};
      orderEventsBuffer = [];
      console.log(`[TradeAccounting] Reset for new trading day: ${et.dateString}`);
    } else {
      // Also load order events buffer
      loadOrderEventsFromDisk(et.dateString);
    }
  }
}

function addOrderEvent(event: OrderEvent): void {
  orderEventsBuffer.push(event);
  if (orderEventsBuffer.length > MAX_ORDER_EVENTS) {
    orderEventsBuffer.shift();
  }
  // Persist event to disk
  persistOrderEvent(event);
}

function logOrderEvent(event: OrderEvent): void {
  const parts = [
    `ACTION=${event.action}`,
    `trade_id=${event.tradeId}`,
    `symbol=${event.symbol}`,
    `strategy=${event.strategyName}`,
    `qty=${event.qty}`,
  ];
  
  if (event.limitPrice !== null) {
    parts.push(`limit_price=${event.limitPrice.toFixed(2)}`);
  }
  
  if (event.alpacaOrderId) {
    parts.push(`order_id=${event.alpacaOrderId}`);
  }
  
  if (event.reason) {
    parts.push(`reason=${event.reason}`);
  }
  
  if (event.errorCode) {
    parts.push(`error_code=${event.errorCode}`);
  }
  
  if (event.errorMessage) {
    parts.push(`error_message=${event.errorMessage}`);
  }
  
  console.log(`[TradeAccounting] ${parts.join(" ")}`);
}

/**
 * Record a trade proposal - signal passed validation and will attempt submission
 */
export function recordProposal(data: {
  tradeId: string;
  symbol: string;
  strategyName: string;
  limitPrice: number | null;
  qty: number;
}): void {
  ensureCurrentDate();
  tradesProposed++;
  
  const et = getEasternTime();
  const event: OrderEvent = {
    timestampET: et.displayTime,
    tradeId: data.tradeId,
    symbol: data.symbol,
    strategyName: data.strategyName,
    action: "PROPOSE",
    limitPrice: data.limitPrice,
    qty: data.qty,
    alpacaOrderId: null,
    reason: null,
    errorCode: null,
    errorMessage: null,
  };
  
  addOrderEvent(event);
  logOrderEvent(event);
  persistCounters();
}

/**
 * Record successful order submission - Alpaca accepted the order
 */
export function recordSubmitOk(data: {
  tradeId: string;
  symbol: string;
  strategyName: string;
  limitPrice: number | null;
  qty: number;
  alpacaOrderId: string;
}): void {
  ensureCurrentDate();
  tradesSubmitted++;
  
  const et = getEasternTime();
  const event: OrderEvent = {
    timestampET: et.displayTime,
    tradeId: data.tradeId,
    symbol: data.symbol,
    strategyName: data.strategyName,
    action: "SUBMIT_OK",
    limitPrice: data.limitPrice,
    qty: data.qty,
    alpacaOrderId: data.alpacaOrderId,
    reason: null,
    errorCode: null,
    errorMessage: null,
  };
  
  addOrderEvent(event);
  logOrderEvent(event);
  persistCounters();
}

/**
 * Record order rejection - Alpaca rejected the order
 */
export function recordSubmitFail(data: {
  tradeId: string;
  symbol: string;
  strategyName: string;
  limitPrice: number | null;
  qty: number;
  errorCode: string;
  errorMessage: string;
}): void {
  ensureCurrentDate();
  tradesRejected++;
  lastRejectionReason = `${data.errorCode}: ${data.errorMessage}`;
  
  const et = getEasternTime();
  const event: OrderEvent = {
    timestampET: et.displayTime,
    tradeId: data.tradeId,
    symbol: data.symbol,
    strategyName: data.strategyName,
    action: "SUBMIT_FAIL",
    limitPrice: data.limitPrice,
    qty: data.qty,
    alpacaOrderId: null,
    reason: null,
    errorCode: data.errorCode,
    errorMessage: data.errorMessage,
  };
  
  addOrderEvent(event);
  logOrderEvent(event);
  persistCounters();
}

/**
 * Record trade suppression - blocked by idempotency or preflight gate
 */
export function recordSuppress(data: {
  tradeId: string;
  symbol: string;
  strategyName: string;
  limitPrice: number | null;
  qty: number;
  reason: string;
}): void {
  ensureCurrentDate();
  tradesSuppressed++;
  
  suppressReasonCounts[data.reason] = (suppressReasonCounts[data.reason] || 0) + 1;
  
  const et = getEasternTime();
  const event: OrderEvent = {
    timestampET: et.displayTime,
    tradeId: data.tradeId,
    symbol: data.symbol,
    strategyName: data.strategyName,
    action: "SUPPRESS",
    limitPrice: data.limitPrice,
    qty: data.qty,
    alpacaOrderId: null,
    reason: data.reason,
    errorCode: null,
    errorMessage: null,
  };
  
  addOrderEvent(event);
  logOrderEvent(event);
  persistCounters();
}

/**
 * Record order cancellation (timeout cancel)
 */
export function recordCancel(data: {
  tradeId: string;
  symbol: string;
  strategyName: string;
  alpacaOrderId: string;
  reason: string;
}): void {
  ensureCurrentDate();
  tradesCanceled++;
  
  const et = getEasternTime();
  const event: OrderEvent = {
    timestampET: et.displayTime,
    tradeId: data.tradeId,
    symbol: data.symbol,
    strategyName: data.strategyName,
    action: "CANCEL",
    limitPrice: null,
    qty: 0,
    alpacaOrderId: data.alpacaOrderId,
    reason: data.reason,
    errorCode: null,
    errorMessage: null,
  };
  
  addOrderEvent(event);
  logOrderEvent(event);
  persistCounters();
}

/**
 * Get today's accounting summary
 */
export function getTodayAccounting(): DailyAccounting {
  ensureCurrentDate();
  
  const totalSuppressed = Object.values(suppressReasonCounts).reduce((a, b) => a + b, 0);
  const topSuppressReasons = Object.entries(suppressReasonCounts)
    .map(([reason, count]) => ({
      reason,
      count,
      percent: totalSuppressed > 0 ? (count / totalSuppressed) * 100 : 0,
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 5);
  
  return {
    date: currentDate,
    tradesProposed,
    tradesSubmitted,
    tradesRejected,
    tradesSuppressed,
    tradesCanceled,
    lastRejectionReason,
    suppressReasonCounts: { ...suppressReasonCounts },
    topSuppressReasons,
  };
}

/**
 * Get the last N order events
 */
export function getRecentOrderEvents(): OrderEvent[] {
  ensureCurrentDate();
  return [...orderEventsBuffer];
}

/**
 * Verify accounting invariant: proposed = submitted + rejected + suppressed
 * Returns true if invariant holds
 */
export function verifyInvariant(): { valid: boolean; message: string } {
  ensureCurrentDate();
  
  const expectedTotal = tradesSubmitted + tradesRejected + tradesSuppressed;
  const valid = tradesProposed === expectedTotal;
  
  if (valid) {
    return {
      valid: true,
      message: `OK: proposed(${tradesProposed}) = submitted(${tradesSubmitted}) + rejected(${tradesRejected}) + suppressed(${tradesSuppressed})`,
    };
  } else {
    return {
      valid: false,
      message: `MISMATCH: proposed(${tradesProposed}) != submitted(${tradesSubmitted}) + rejected(${tradesRejected}) + suppressed(${tradesSuppressed}) = ${expectedTotal}`,
    };
  }
}

/**
 * Reset counters (for testing)
 */
export function resetForTesting(): void {
  currentDate = "";
  tradesProposed = 0;
  tradesSubmitted = 0;
  tradesRejected = 0;
  tradesSuppressed = 0;
  tradesCanceled = 0;
  lastRejectionReason = null;
  suppressReasonCounts = {};
  orderEventsBuffer = [];
}

/**
 * Get compact stats for activity summary integration
 */
export function getStats(): {
  proposed: number;
  submitted: number;
  rejected: number;
  suppressed: number;
  canceled: number;
  lastRejectionReason: string | null;
} {
  ensureCurrentDate();
  return {
    proposed: tradesProposed,
    submitted: tradesSubmitted,
    rejected: tradesRejected,
    suppressed: tradesSuppressed,
    canceled: tradesCanceled,
    lastRejectionReason,
  };
}

/**
 * Get suppress reason counts for top reasons analysis
 */
export function getSuppressReasonCounts(): Record<string, number> {
  ensureCurrentDate();
  return { ...suppressReasonCounts };
}

/**
 * Get accounting for a specific date (for rolling reports)
 * Loads from disk if not current day
 */
export function getAccountingForDate(dateStr: string): DailyAccounting | null {
  // If requesting today, return current counters
  const et = getEasternTime();
  if (dateStr === et.dateString) {
    ensureCurrentDate();
    return getTodayAccounting();
  }
  
  // Load from disk for historical dates
  const filePath = getAccountingFilePath(dateStr);
  if (!fs.existsSync(filePath)) {
    return null;
  }
  
  try {
    const data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    const suppressCounts = { ...(data.suppressReasonCounts || {}) };
    const totalSuppressed = Object.values(suppressCounts).reduce((a: number, b: unknown) => a + (b as number), 0);
    const topSuppressReasons = Object.entries(suppressCounts)
      .map(([reason, count]) => ({
        reason,
        count: count as number,
        percent: totalSuppressed > 0 ? ((count as number) / totalSuppressed) * 100 : 0,
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);
    
    return {
      date: data.date,
      tradesProposed: data.tradesProposed || 0,
      tradesSubmitted: data.tradesSubmitted || 0,
      tradesRejected: data.tradesRejected || 0,
      tradesSuppressed: data.tradesSuppressed || 0,
      tradesCanceled: data.tradesCanceled || 0,
      lastRejectionReason: data.lastRejectionReason || null,
      suppressReasonCounts: suppressCounts,
      topSuppressReasons,
    };
  } catch {
    return null;
  }
}
