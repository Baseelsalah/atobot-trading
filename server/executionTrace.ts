import { getEasternTime } from "./timezone";
import { nanoid } from "nanoid";
import * as reportStorage from "./reportStorage";

export type ExecTraceType = "EXEC_START" | "EXEC_FAIL" | "EXEC_OK";
export type ExecStage = "precheck" | "risk" | "order_build" | "alpaca_submit";

export interface DurableExecTrace {
  traceId: string;
  traceType: ExecTraceType;
  tsET: string;
  tsUTC: string;
  trade_id: string | null;
  symbol: string;
  strategy: string;
  tier: number | null;
  stage: ExecStage | null;
  errorMessage: string | null;
  stack: string | null;
  isLeader: boolean;
  entryAllowed: boolean;
  marketStatus: string;
}

const DURABLE_RING_SIZE = 50;
const durableTraceBuffer: DurableExecTrace[] = [];
let lastPersistTsUTC: string | null = null;
let lastStorageKey: string | null = null;
let persistPending = false;

export interface ExecutionAttempt {
  trade_id: string;
  tsET: string;
  tsUTC: string;
  symbol: string;
  strategy: string;
  side: "buy" | "sell";
  qty: number;
  price: number;
  stageReached: "proposed" | "precheck" | "alpaca_submit" | "alpaca_accepted" | "alpaca_rejected" | "filled";
  failureReason: string | null;
  failureStage: string | null;
  alpacaOrderId: string | null;
  clientOrderId: string | null;
  alpacaStatus: string | null;
  alpacaError: string | null;
  httpStatus: number | null;
  durationMs: number | null;
}

const RING_BUFFER_SIZE = 50;
const executionRingBuffer: ExecutionAttempt[] = [];

let dailyCounters = {
  signalsGenerated: 0,
  tradesProposed: 0,
  tradesSubmitted: 0,
  tradesFilled: 0,
  lastResetDate: "",
};

export function generateTradeId(): string {
  const et = getEasternTime();
  const datePrefix = et.dateString.replace(/-/g, "");
  return `trade_${datePrefix}_${nanoid(8)}`;
}

export function resetDailyCounters(): void {
  const et = getEasternTime();
  if (dailyCounters.lastResetDate !== et.dateString) {
    dailyCounters = {
      signalsGenerated: 0,
      tradesProposed: 0,
      tradesSubmitted: 0,
      tradesFilled: 0,
      lastResetDate: et.dateString,
    };
    console.log(`[EXEC_TRACE] Daily counters reset for ${et.dateString}`);
  }
}

export function incrementSignalsGenerated(): void {
  resetDailyCounters();
  dailyCounters.signalsGenerated++;
}

export function incrementTradesProposed(): void {
  resetDailyCounters();
  dailyCounters.tradesProposed++;
}

export function incrementTradesSubmitted(): void {
  resetDailyCounters();
  dailyCounters.tradesSubmitted++;
}

export function incrementTradesFilled(): void {
  resetDailyCounters();
  dailyCounters.tradesFilled++;
}

export function getExecutionCounters() {
  resetDailyCounters();
  return { ...dailyCounters };
}

export function logExecStart(trade_id: string, symbol: string, strategy: string, side: string, qty: number): void {
  console.log(`[EXEC_START] trade_id=${trade_id} symbol=${symbol} strategy=${strategy} side=${side} qty=${qty}`);
}

export function logExecPrecheckFail(trade_id: string, reason: string): void {
  console.log(`[EXEC_PRECHECK_FAIL] trade_id=${trade_id} reason=${reason}`);
}

export function logAlpacaSubmitAttempt(trade_id: string, clientOrderId: string, symbol: string, side: string, qty: number, type: string, limitPrice: number | null): void {
  const payloadSummary = `symbol=${symbol} side=${side} qty=${qty} type=${type} limitPrice=${limitPrice ?? "N/A"}`;
  console.log(`[ALPACA_SUBMIT_ATTEMPT] trade_id=${trade_id} client_order_id=${clientOrderId} ${payloadSummary}`);
}

export function logAlpacaSubmitOk(trade_id: string, alpacaOrderId: string, status: string): void {
  console.log(`[ALPACA_SUBMIT_OK] trade_id=${trade_id} alpacaOrderId=${alpacaOrderId} status=${status}`);
}

export function logAlpacaSubmitFail(trade_id: string, errorName: string, errorMessage: string, httpStatus: number | null, responseBody: string | null, stack: string | null): void {
  const httpPart = httpStatus ? `httpStatus=${httpStatus}` : "httpStatus=N/A";
  const bodyPart = responseBody ? `responseBody=${responseBody.substring(0, 200)}` : "";
  const stackPart = stack ? `stack=${stack.substring(0, 300)}` : "";
  console.log(`[ALPACA_SUBMIT_FAIL] trade_id=${trade_id} errorName=${errorName} errorMessage=${errorMessage} ${httpPart} ${bodyPart} ${stackPart}`);
}

export function recordExecutionAttempt(attempt: ExecutionAttempt): void {
  executionRingBuffer.push(attempt);
  if (executionRingBuffer.length > RING_BUFFER_SIZE) {
    executionRingBuffer.shift();
  }
}

export function createExecutionAttempt(
  trade_id: string,
  symbol: string,
  strategy: string,
  side: "buy" | "sell",
  qty: number,
  price: number
): ExecutionAttempt {
  const et = getEasternTime();
  return {
    trade_id,
    tsET: `${et.hour}:${String(et.minute).padStart(2, "0")} ET`,
    tsUTC: new Date().toISOString(),
    symbol,
    strategy,
    side,
    qty,
    price,
    stageReached: "proposed",
    failureReason: null,
    failureStage: null,
    alpacaOrderId: null,
    clientOrderId: null,
    alpacaStatus: null,
    alpacaError: null,
    httpStatus: null,
    durationMs: null,
  };
}

export function updateAttemptFailure(attempt: ExecutionAttempt, stage: string, reason: string): void {
  attempt.stageReached = stage as any;
  attempt.failureStage = stage;
  attempt.failureReason = reason;
}

export function updateAttemptAlpacaSuccess(
  attempt: ExecutionAttempt,
  alpacaOrderId: string,
  clientOrderId: string,
  status: string
): void {
  attempt.stageReached = status === "filled" ? "filled" : "alpaca_accepted";
  attempt.alpacaOrderId = alpacaOrderId;
  attempt.clientOrderId = clientOrderId;
  attempt.alpacaStatus = status;
}

export function updateAttemptAlpacaFailure(
  attempt: ExecutionAttempt,
  clientOrderId: string,
  errorName: string,
  errorMessage: string,
  httpStatus: number | null
): void {
  attempt.stageReached = "alpaca_rejected";
  attempt.clientOrderId = clientOrderId;
  attempt.alpacaError = `${errorName}: ${errorMessage}`;
  attempt.httpStatus = httpStatus;
  attempt.failureStage = "alpaca_submit";
  attempt.failureReason = errorMessage;
}

export function getRecentExecutions(): ExecutionAttempt[] {
  return [...executionRingBuffer];
}

export function getExecutionSummary() {
  const et = getEasternTime();
  const counters = getExecutionCounters();
  return {
    currentTimeET: `${et.hour}:${String(et.minute).padStart(2, "0")} ET`,
    counters,
    recentAttempts: executionRingBuffer.length,
    ringBufferCapacity: RING_BUFFER_SIZE,
    executions: getRecentExecutions(),
  };
}

function createTraceId(): string {
  return `trace_${Date.now()}_${nanoid(6)}`;
}

function formatET(): string {
  const et = getEasternTime();
  return `${et.hour}:${String(et.minute).padStart(2, "0")} ET`;
}

export interface ExecTraceContext {
  isLeader: boolean;
  entryAllowed: boolean;
  marketStatus: string;
}

export function recordExecStart(
  symbol: string,
  strategy: string,
  tier: number | null,
  trade_id: string | null,
  ctx: ExecTraceContext
): void {
  const trace: DurableExecTrace = {
    traceId: createTraceId(),
    traceType: "EXEC_START",
    tsET: formatET(),
    tsUTC: new Date().toISOString(),
    trade_id,
    symbol,
    strategy,
    tier,
    stage: null,
    errorMessage: null,
    stack: null,
    isLeader: ctx.isLeader,
    entryAllowed: ctx.entryAllowed,
    marketStatus: ctx.marketStatus,
  };
  
  addDurableTrace(trace);
  console.log(`[EXEC_START_DURABLE] traceId=${trace.traceId} symbol=${symbol} strategy=${strategy} tier=${tier} trade_id=${trade_id} isLeader=${ctx.isLeader} entryAllowed=${ctx.entryAllowed} marketStatus=${ctx.marketStatus}`);
}

export function recordExecFail(
  symbol: string,
  strategy: string,
  tier: number | null,
  trade_id: string | null,
  stage: ExecStage,
  errorMessage: string,
  stack: string | null,
  ctx: ExecTraceContext
): void {
  const trace: DurableExecTrace = {
    traceId: createTraceId(),
    traceType: "EXEC_FAIL",
    tsET: formatET(),
    tsUTC: new Date().toISOString(),
    trade_id,
    symbol,
    strategy,
    tier,
    stage,
    errorMessage,
    stack: stack ? stack.substring(0, 500) : null,
    isLeader: ctx.isLeader,
    entryAllowed: ctx.entryAllowed,
    marketStatus: ctx.marketStatus,
  };
  
  addDurableTrace(trace);
  console.log(`[EXEC_FAIL_DURABLE] traceId=${trace.traceId} symbol=${symbol} strategy=${strategy} tier=${tier} stage=${stage} errorMessage=${errorMessage} isLeader=${ctx.isLeader} entryAllowed=${ctx.entryAllowed} marketStatus=${ctx.marketStatus}`);
}

export function recordExecOk(
  symbol: string,
  strategy: string,
  tier: number | null,
  trade_id: string | null,
  ctx: ExecTraceContext
): void {
  const trace: DurableExecTrace = {
    traceId: createTraceId(),
    traceType: "EXEC_OK",
    tsET: formatET(),
    tsUTC: new Date().toISOString(),
    trade_id,
    symbol,
    strategy,
    tier,
    stage: null,
    errorMessage: null,
    stack: null,
    isLeader: ctx.isLeader,
    entryAllowed: ctx.entryAllowed,
    marketStatus: ctx.marketStatus,
  };
  
  addDurableTrace(trace);
  console.log(`[EXEC_OK_DURABLE] traceId=${trace.traceId} symbol=${symbol} strategy=${strategy} tier=${tier} trade_id=${trade_id}`);
}

function addDurableTrace(trace: DurableExecTrace): void {
  durableTraceBuffer.push(trace);
  if (durableTraceBuffer.length > DURABLE_RING_SIZE) {
    durableTraceBuffer.shift();
  }
  schedulePersist();
}

function schedulePersist(): void {
  if (persistPending) return;
  persistPending = true;
  
  setTimeout(async () => {
    persistPending = false;
    await persistDurableTraces();
  }, 2000);
}

async function persistDurableTraces(): Promise<void> {
  if (durableTraceBuffer.length === 0) return;
  if (!reportStorage.isStorageEnabled()) return;
  
  try {
    const et = getEasternTime();
    const dateStr = et.dateString;
    const filename = `execution_recent_${dateStr}.json`;
    
    const payload = {
      dateET: dateStr,
      lastUpdatedUTC: new Date().toISOString(),
      envScope: reportStorage.getStorageEnvScope(),
      traceCount: durableTraceBuffer.length,
      traces: [...durableTraceBuffer],
    };
    
    const success = await reportStorage.putText("execution", filename, JSON.stringify(payload, null, 2));
    if (success) {
      lastPersistTsUTC = new Date().toISOString();
      lastStorageKey = `atobot/${reportStorage.getStorageEnvScope()}/reports/execution/${filename}`;
    }
  } catch (err: any) {
    console.error(`[EXEC_TRACE] Persist failed: ${err?.message}`);
  }
}

export function getDurableTraces(): DurableExecTrace[] {
  return [...durableTraceBuffer];
}

export function getDurableTraceStatus() {
  const et = getEasternTime();
  return {
    currentTimeET: formatET(),
    dateET: et.dateString,
    counters: {
      signalsGenerated: dailyCounters.signalsGenerated,
      tradesProposed: dailyCounters.tradesProposed,
      tradesSubmitted: dailyCounters.tradesSubmitted,
      tradesFilled: dailyCounters.tradesFilled,
      source: "activityLedger",
    },
    ringBuffer: {
      capacity: DURABLE_RING_SIZE,
      count: durableTraceBuffer.length,
    },
    lastPersistTsUTC,
    lastStorageKey,
    envScope: reportStorage.getStorageEnvScope(),
    executions: getDurableTraces(),
  };
}
