/**
 * Runtime Monitor - Boot logging, heartbeat, and stall detection
 * 
 * Provides proof that the bot was actually running during market hours.
 * - Boot logging with market status
 * - Heartbeat every 60s when process is up
 * - Stall detector: alerts and restarts scan loop if market OPEN and no tick in 5 min
 * - Daily "did we run?" proof file at market close + 5 min
 */

import * as fs from "fs";
import * as path from "path";
import * as alpaca from "./alpaca";
import * as activityLedger from "./activityLedger";
import { getEasternTime, toEasternDateString } from "./timezone";
import * as timeGuard from "./tradingTimeGuard";
import * as tradeAccounting from "./tradeAccounting";
import * as reportStorage from "./reportStorage";
import * as leaderLock from "./leaderLock";
import * as controlLoopTrace from "./controlLoopTrace";

const RUNTIME_DIR = "reports/runtime";
const ALERTS_DIR = "reports/alerts";
const HEARTBEAT_FILE = "reports/runtime/heartbeat_latest.json";

// RECOVERY-MODE-1: Trading state machine
export type TradingState = "INITIALIZING" | "RECOVERY_MODE" | "ACTIVE_TRADING" | "MANAGE_ONLY" | "STOPPED";

interface RecoveryStatus {
  state: TradingState;
  bootRecoveryRan: boolean;
  recoveryResult: "PENDING" | "SUCCESS" | "FAILED" | "SKIPPED";
  recoveryFailReason: string | null;
  checks: {
    leaderLock: boolean | null;
    alpacaClock: boolean | null;
    alpacaAccount: boolean | null;
    positionsLoaded: boolean | null;
    ordersLoaded: boolean | null;
    riskStateOk: boolean | null;
  };
}

const recoveryStatus: RecoveryStatus = {
  state: "INITIALIZING",
  bootRecoveryRan: false,
  recoveryResult: "PENDING",
  recoveryFailReason: null,
  checks: {
    leaderLock: null,
    alpacaClock: null,
    alpacaAccount: null,
    positionsLoaded: null,
    ordersLoaded: null,
    riskStateOk: null,
  },
};

export function getRecoveryStatus(): RecoveryStatus {
  return { ...recoveryStatus, checks: { ...recoveryStatus.checks } };
}

export function getTradingState(): TradingState {
  return recoveryStatus.state;
}

export function setTradingState(state: TradingState): void {
  console.log(`[RUNTIME] TradingState changed: ${recoveryStatus.state} -> ${state}`);
  recoveryStatus.state = state;
}

interface BootShutdownEvent {
  tsET: string;
  tsUTC: string;
  pid: number;
  version: string;
  event: "BOOT" | "SHUTDOWN";
  signal?: string;
  exitCode?: number;
  reason?: string;
  memoryMB?: number;
  error?: string;
}

interface HeartbeatFile {
  tsET: string;
  tsUTC: string;
  pid: number;
}

function ensureAlertsDir(): void {
  if (!fs.existsSync(ALERTS_DIR)) {
    fs.mkdirSync(ALERTS_DIR, { recursive: true });
  }
}

function getBootsFilePath(dateET: string): string {
  return path.join(RUNTIME_DIR, `boots_${dateET}.jsonl`);
}

function appendBootEvent(event: BootShutdownEvent): void {
  ensureRuntimeDir();
  const dateET = toEasternDateString(new Date());
  const filePath = getBootsFilePath(dateET);
  const line = JSON.stringify(event) + "\n";
  fs.appendFileSync(filePath, line, "utf-8");
  console.log(`[RUNTIME] Logged ${event.event} to ${filePath}`);
  reportStorage.appendLine("runtime", `boots_${dateET}.jsonl`, JSON.stringify(event)).catch(() => {});
}

function writeHeartbeatFile(): void {
  ensureRuntimeDir();
  const et = getEasternTime();
  const data: HeartbeatFile = {
    tsET: et.displayTime,
    tsUTC: new Date().toISOString(),
    pid: process.pid,
  };
  const content = JSON.stringify(data, null, 2);
  fs.writeFileSync(HEARTBEAT_FILE, content, "utf-8");
  reportStorage.putText("runtime", "heartbeat_latest.json", content).catch(() => {});
}

function readLastHeartbeat(): HeartbeatFile | null {
  try {
    if (fs.existsSync(HEARTBEAT_FILE)) {
      const content = fs.readFileSync(HEARTBEAT_FILE, "utf-8");
      return JSON.parse(content) as HeartbeatFile;
    }
  } catch (err) {
    console.log("[RUNTIME] Could not read last heartbeat file");
  }
  return null;
}

async function checkForDowntime(): Promise<void> {
  const lastHeartbeat = readLastHeartbeat();
  if (!lastHeartbeat) return;
  
  const lastTime = new Date(lastHeartbeat.tsUTC).getTime();
  const now = Date.now();
  const downtimeMs = now - lastTime;
  const downtimeMinutes = Math.round(downtimeMs / 60000);
  
  if (downtimeMinutes < 2) return;
  
  let isMarketHours = false;
  try {
    const clock = await alpaca.getClock();
    isMarketHours = clock.is_open;
    
    if (!isMarketHours) {
      const nextOpen = new Date(clock.next_open).getTime();
      const lastClose = new Date(clock.next_close).getTime() - 6.5 * 60 * 60 * 1000;
      isMarketHours = lastTime > lastClose && lastTime < nextOpen && now > lastClose;
    }
  } catch (err) {
    console.log("[RUNTIME] Could not check market hours for downtime detection");
  }
  
  if (downtimeMinutes >= 2 && isMarketHours) {
    ensureAlertsDir();
    const et = getEasternTime();
    const dateET = toEasternDateString(new Date());
    const alertPath = path.join(ALERTS_DIR, "CRITICAL_downtime_detected.txt");
    const content = `CRITICAL DOWNTIME DETECTED
========================
lastHeartbeat: ${lastHeartbeat.tsET} (${lastHeartbeat.tsUTC})
currentTime: ${et.displayTime} (${new Date().toISOString()})
downtimeMinutes: ${downtimeMinutes}
lastPID: ${lastHeartbeat.pid}
currentPID: ${process.pid}
suspectedCause: Process died/restarted during market hours (no heartbeat for ${downtimeMinutes} min)
detectedAt: ${new Date().toISOString()}
`;
    fs.writeFileSync(alertPath, content, "utf-8");
    console.log(`[RUNTIME] CRITICAL: Downtime detected! ${downtimeMinutes} minutes. Alert written to ${alertPath}`);
    
    // OPS-DOWNTIME-PROOF-2: Persist to Object Storage for durable receipts
    reportStorage.putText("alerts", `CRITICAL_downtime_detected_${dateET}.txt`, content).catch((err) => {
      console.error(`[RUNTIME] Failed to persist downtime alert to Object Storage:`, err?.message);
    });
  }
}

function registerShutdownHandlers(): void {
  const version = process.env.REPL_ID?.slice(0, 8) || "unknown";
  
  // Track if we've already logged shutdown to prevent duplicates
  let shutdownLogged = false;
  
  const logShutdown = (reason: string, signal?: string, exitCode?: number, error?: Error) => {
    if (shutdownLogged) return;
    shutdownLogged = true;
    
    const et = getEasternTime();
    const memUsage = process.memoryUsage();
    const event: BootShutdownEvent = {
      tsET: et.displayTime,
      tsUTC: new Date().toISOString(),
      pid: process.pid,
      version,
      event: "SHUTDOWN",
      signal,
      exitCode,
      reason,
      memoryMB: Math.round(memUsage.heapUsed / 1024 / 1024),
      error: error ? `${error.name}: ${error.message}` : undefined,
    };
    appendBootEvent(event);
    console.log(`[RUNTIME] Shutdown logged: reason=${reason} signal=${signal || 'none'} exitCode=${exitCode ?? 'none'} memMB=${event.memoryMB}`);
  };
  
  // Graceful shutdown signals
  process.on("SIGTERM", () => {
    logShutdown("SIGTERM_RECEIVED", "SIGTERM");
  });
  
  process.on("SIGINT", () => {
    logShutdown("SIGINT_RECEIVED", "SIGINT");
  });
  
  // Normal exit
  process.on("exit", (code) => {
    logShutdown("PROCESS_EXIT", undefined, code);
  });
  
  // Uncaught exceptions - crashes
  process.on("uncaughtException", (error) => {
    console.error("[RUNTIME] UNCAUGHT EXCEPTION:", error);
    logShutdown("UNCAUGHT_EXCEPTION", undefined, 1, error);
    process.exit(1);
  });
  
  // Unhandled promise rejections - async crashes
  process.on("unhandledRejection", (reason, promise) => {
    const error = reason instanceof Error ? reason : new Error(String(reason));
    console.error("[RUNTIME] UNHANDLED REJECTION:", error);
    logShutdown("UNHANDLED_REJECTION", undefined, 1, error);
    process.exit(1);
  });
  
  // Before exit hook (for graceful shutdowns)
  process.on("beforeExit", (code) => {
    logShutdown("BEFORE_EXIT", undefined, code);
  });
  
  console.log("[RUNTIME] Shutdown handlers registered: SIGTERM, SIGINT, exit, uncaughtException, unhandledRejection, beforeExit");
}

interface RuntimeState {
  bootTime: Date;
  bootTimeET: string;
  bootTimeUTC: string; // ISO timestamp for unambiguous boot time
  bootId: string; // Unique identifier for this boot session
  version: string; // Git commit or REPL_ID
  heartbeatCount: number;
  lastHeartbeatET: string | null;
  lastTickET: string | null;
  stallAlertCount: number;
  didRunFileWritten: boolean;
  ticksSinceBoot: number; // OPS-METRICS-1: Clear counter for ticks since current process started
  memoryHistory: number[]; // Last N memory readings in MB (for trend detection)
  baselineMemoryMB: number; // Memory at boot time
}

interface DidRunReport {
  dateET: string;
  marketOpenET: string | null;
  marketCloseET: string | null;
  uptimeMinutes: number;
  heartbeatCount: number;
  scanTicks: number;
  symbolsEvaluated: number;
  signalsGenerated: number;
  tradesProposed: number;
  tradesSubmitted: number;
  tradesFilled: number;
  lastHeartbeatET: string | null;
  lastTickET: string | null;
  bootTimeET: string;
}

let runtimeState: RuntimeState | null = null;
let heartbeatInterval: NodeJS.Timeout | null = null;
let stallCheckInterval: NodeJS.Timeout | null = null;
let didRunInterval: NodeJS.Timeout | null = null;

let scanRestartCallback: (() => Promise<void>) | null = null;
let lastKnownTickTime: number = 0;

/**
 * RECOVERY-MODE-1: Run boot recovery sequence when starting during market hours
 */
async function runBootRecovery(marketStatus: string, et: { hour: number; minute: number; displayTime: string; dateString: string }): Promise<void> {
  const bootTimeMinutes = et.hour * 60 + et.minute;
  const marketOpenMinutes = 9 * 60 + 30; // 9:30 ET
  const marketCloseMinutes = 16 * 60; // 16:00 ET
  const entryStartMinutes = 9 * 60 + 35; // 9:35 ET
  const entryEndMinutes = 11 * 60 + 35; // 11:35 ET
  
  const isMarketHours = bootTimeMinutes >= marketOpenMinutes && bootTimeMinutes < marketCloseMinutes;
  const isWithinEntryWindow = bootTimeMinutes >= entryStartMinutes && bootTimeMinutes <= entryEndMinutes;
  
  // Skip recovery if market is closed
  if (marketStatus === "CLOSED" && !isMarketHours) {
    recoveryStatus.state = "STOPPED";
    recoveryStatus.recoveryResult = "SKIPPED";
    recoveryStatus.recoveryFailReason = "Market closed - recovery not needed";
    console.log("[RUNTIME] Recovery skipped - market is closed");
    return;
  }
  
  console.log(`[RUNTIME] ========== BOOT RECOVERY MODE ==========`);
  recoveryStatus.state = "RECOVERY_MODE";
  recoveryStatus.bootRecoveryRan = true;
  
  // Step 1: Confirm leader lock
  recoveryStatus.checks.leaderLock = leaderLock.isLeaderInstance();
  console.log(`[RUNTIME] Recovery check: leaderLock=${recoveryStatus.checks.leaderLock}`);
  
  if (!recoveryStatus.checks.leaderLock) {
    recoveryStatus.state = "MANAGE_ONLY";
    recoveryStatus.recoveryResult = "FAILED";
    recoveryStatus.recoveryFailReason = "NOT_LEADER - cannot enter trades";
    console.log(`[RUNTIME] Recovery FAILED: Not leader instance`);
    await persistRecoveryFailAlert(et.dateString, "NOT_LEADER");
    return;
  }
  
  // Step 2: Verify Alpaca clock reachable
  try {
    await alpaca.getClock();
    recoveryStatus.checks.alpacaClock = true;
    console.log(`[RUNTIME] Recovery check: alpacaClock=true`);
  } catch (err) {
    recoveryStatus.checks.alpacaClock = false;
    recoveryStatus.state = "MANAGE_ONLY";
    recoveryStatus.recoveryResult = "FAILED";
    recoveryStatus.recoveryFailReason = "ALPACA_CLOCK_UNREACHABLE";
    console.log(`[RUNTIME] Recovery FAILED: Alpaca clock unreachable`);
    await persistRecoveryFailAlert(et.dateString, "ALPACA_CLOCK_UNREACHABLE");
    return;
  }
  
  // Step 3: Verify Alpaca account reachable
  try {
    await alpaca.getAccount();
    recoveryStatus.checks.alpacaAccount = true;
    console.log(`[RUNTIME] Recovery check: alpacaAccount=true`);
  } catch (err) {
    recoveryStatus.checks.alpacaAccount = false;
    recoveryStatus.state = "MANAGE_ONLY";
    recoveryStatus.recoveryResult = "FAILED";
    recoveryStatus.recoveryFailReason = "ALPACA_ACCOUNT_UNREACHABLE";
    console.log(`[RUNTIME] Recovery FAILED: Alpaca account unreachable`);
    await persistRecoveryFailAlert(et.dateString, "ALPACA_ACCOUNT_UNREACHABLE");
    return;
  }
  
  // Step 4: Fetch open positions
  try {
    const positions = await alpaca.getPositions();
    recoveryStatus.checks.positionsLoaded = true;
    console.log(`[RUNTIME] Recovery check: positionsLoaded=true (${positions.length} positions)`);
  } catch (err) {
    recoveryStatus.checks.positionsLoaded = false;
    recoveryStatus.state = "MANAGE_ONLY";
    recoveryStatus.recoveryResult = "FAILED";
    recoveryStatus.recoveryFailReason = "POSITIONS_LOAD_FAILED";
    console.log(`[RUNTIME] Recovery FAILED: Could not load positions`);
    await persistRecoveryFailAlert(et.dateString, "POSITIONS_LOAD_FAILED");
    return;
  }
  
  // Step 5: Fetch open orders
  try {
    const orders = await alpaca.getOrders("open");
    recoveryStatus.checks.ordersLoaded = true;
    console.log(`[RUNTIME] Recovery check: ordersLoaded=true (${orders.length} open orders)`);
  } catch (err) {
    recoveryStatus.checks.ordersLoaded = false;
    recoveryStatus.state = "MANAGE_ONLY";
    recoveryStatus.recoveryResult = "FAILED";
    recoveryStatus.recoveryFailReason = "ORDERS_LOAD_FAILED";
    console.log(`[RUNTIME] Recovery FAILED: Could not load orders`);
    await persistRecoveryFailAlert(et.dateString, "ORDERS_LOAD_FAILED");
    return;
  }
  
  // Step 6: Check risk state (informational only - time guard/entry window are NOT recovery failures)
  const tgStatus = timeGuard.getTimeGuardStatus();
  recoveryStatus.checks.riskStateOk = tgStatus.canManagePositions;
  console.log(`[RUNTIME] Recovery check: riskStateOk=${recoveryStatus.checks.riskStateOk} canManagePositions=${tgStatus.canManagePositions} canOpenNewTrades=${tgStatus.canOpenNewTrades}`);
  
  // All infrastructure checks passed - recovery is successful
  // NOTE: Time window restrictions are NOT recovery failures, just trading mode restrictions
  recoveryStatus.recoveryResult = "SUCCESS";
  
  if (isWithinEntryWindow && tgStatus.canOpenNewTrades) {
    recoveryStatus.state = "ACTIVE_TRADING";
    console.log(`[RUNTIME] Recovery SUCCESS - state=ACTIVE_TRADING (within entry window, entry allowed)`);
  } else if (tgStatus.canManagePositions) {
    recoveryStatus.state = "MANAGE_ONLY";
    console.log(`[RUNTIME] Recovery SUCCESS - state=MANAGE_ONLY (can manage but cannot enter)`);
  } else {
    recoveryStatus.state = "MANAGE_ONLY";
    console.log(`[RUNTIME] Recovery SUCCESS - state=MANAGE_ONLY (market hours position management mode)`);
  }
  
  console.log(`[RUNTIME] ========== RECOVERY COMPLETE ==========`);
}

async function persistRecoveryFailAlert(dateET: string, reason: string): Promise<void> {
  const et = getEasternTime();
  const alertContent = `CRITICAL RECOVERY FAILED
=========================
dateET: ${dateET}
timeET: ${et.displayTime}
timeUTC: ${new Date().toISOString()}
pid: ${process.pid}
reason: ${reason}
checks: ${JSON.stringify(recoveryStatus.checks)}
`;
  
  ensureAlertsDir();
  const alertPath = path.join(ALERTS_DIR, `CRITICAL_recovery_failed_${dateET}.txt`);
  fs.writeFileSync(alertPath, alertContent, "utf-8");
  console.log(`[RUNTIME] CRITICAL: Recovery failed! Alert written to ${alertPath}`);
  
  // Persist to Object Storage
  reportStorage.putText("alerts", `CRITICAL_recovery_failed_${dateET}.txt`, alertContent).catch((err) => {
    console.error(`[RUNTIME] Failed to persist recovery-failed alert to Object Storage:`, err?.message);
  });
}

/**
 * Analyze memory trend from history
 * Returns: "STABLE" | "RISING" | "FALLING" | "UNKNOWN"
 */
function getMemoryTrend(history: number[]): string {
  if (history.length < 5) return "UNKNOWN";
  
  // Compare first half average to second half average
  const mid = Math.floor(history.length / 2);
  const firstHalf = history.slice(0, mid);
  const secondHalf = history.slice(mid);
  
  const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
  const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
  
  const delta = secondAvg - firstAvg;
  const threshold = 5; // 5MB threshold for significant change
  
  if (delta > threshold) return "RISING";
  if (delta < -threshold) return "FALLING";
  return "STABLE";
}

function ensureRuntimeDir(): void {
  if (!fs.existsSync(RUNTIME_DIR)) {
    fs.mkdirSync(RUNTIME_DIR, { recursive: true });
  }
}

function getDidRunFilePath(dateET: string): string {
  return path.join(RUNTIME_DIR, `did_run_${dateET}.json`);
}

/**
 * Initialize runtime monitor - call on process boot
 */
export async function initRuntimeMonitor(): Promise<void> {
  const et = getEasternTime();
  const bootTime = new Date();
  const version = process.env.REPL_ID?.slice(0, 8) || "unknown";
  
  // Log BOOT event to disk
  const bootEvent: BootShutdownEvent = {
    tsET: et.displayTime,
    tsUTC: bootTime.toISOString(),
    pid: process.pid,
    version,
    event: "BOOT",
  };
  appendBootEvent(bootEvent);
  
  // Check for downtime (if last heartbeat is stale during market hours)
  await checkForDowntime();
  
  // Register shutdown handlers
  registerShutdownHandlers();
  
  // Write initial heartbeat file
  writeHeartbeatFile();
  
  // Initialize lastKnownTickTime from today's activity summary (if any ticks exist)
  const todaySummary = activityLedger.getTodaysSummary();
  if (todaySummary.scanTicks > 0 && todaySummary.lastTickET) {
    // Set lastKnownTickTime to now so we don't immediately trigger stall detection
    // The actual tick times are in ET display format, not timestamps
    lastKnownTickTime = Date.now();
  }
  
  const memUsage = process.memoryUsage();
  const baselineMemoryMB = Math.round(memUsage.heapUsed / 1024 / 1024);
  const bootId = `boot_${Date.now()}_${process.pid}`;
  const bootTimeUTC = bootTime.toISOString();
  
  // Initialize leader lock and try to acquire
  leaderLock.initLeaderLock(bootId);
  const acquiredLock = await leaderLock.tryAcquireLock();
  if (!acquiredLock) {
    console.log("[RUNTIME] WARNING: Failed to acquire leader lock - this instance will NOT enter trades");
  }
  // OPS-PROD-LOCK-3: Start check loop even for non-leaders (to detect stale locks and take over)
  leaderLock.startLeaderCheckLoop();
  
  runtimeState = {
    bootTime,
    bootTimeET: et.displayTime,
    bootTimeUTC,
    bootId,
    version,
    heartbeatCount: 0,
    lastHeartbeatET: null,
    lastTickET: todaySummary.lastTickET || null,
    stallAlertCount: 0,
    didRunFileWritten: false,
    ticksSinceBoot: 0,
    memoryHistory: [baselineMemoryMB],
    baselineMemoryMB,
  };
  
  console.log(`[RUNTIME] bootId=${bootId} version=${version} baselineMemoryMB=${baselineMemoryMB}`);
  
  let marketStatus = "UNKNOWN";
  let nextOpen = "unknown";
  let nextClose = "unknown";
  
  try {
    const clock = await alpaca.getClock();
    marketStatus = clock.is_open ? "OPEN" : "CLOSED";
    nextOpen = clock.next_open;
    nextClose = clock.next_close;
    
    // If market is open and we have existing activity, set lastKnownTickTime
    // Otherwise if market is open and no activity, stall detector will alert after 5 min
    if (clock.is_open && todaySummary.scanTicks === 0) {
      // No ticks today but market is open - set to boot time so stall detector starts counting from now
      lastKnownTickTime = Date.now();
      console.log("[RUNTIME] Market OPEN but no ticks yet - stall detector will monitor from now");
    }
  } catch (err) {
    console.log("[RUNTIME] Unable to fetch Alpaca clock on boot");
  }
  
  const ptTime = bootTime.toLocaleString("en-US", { timeZone: "America/Los_Angeles", hour12: false });
  
  console.log(`[RUNTIME] ============================================`);
  console.log(`[RUNTIME] boot pt=${ptTime} et=${et.displayTime} marketStatus=${marketStatus} nextOpen=${nextOpen} nextClose=${nextClose}`);
  console.log(`[RUNTIME] ============================================`);
  
  // ALPACA-CONNECTIVITY-PROOF-1: Start Alpaca heartbeat for connectivity monitoring
  alpaca.startAlpacaHeartbeat();
  
  // RECOVERY-MODE-1: Run boot recovery if within market hours
  await runBootRecovery(marketStatus, et);
  
  // OPS-DOWNTIME-PROOF-2: Check if boot occurred during entry window (9:35-11:35 ET)
  const bootHour = et.hour;
  const bootMinute = et.minute;
  const bootTimeMinutes = bootHour * 60 + bootMinute;
  const entryStartMinutes = 9 * 60 + 35; // 9:35 ET
  const entryEndMinutes = 11 * 60 + 35; // 11:35 ET
  const isWithinEntryWindow = bootTimeMinutes >= entryStartMinutes && bootTimeMinutes <= entryEndMinutes;
  const isMarketOpen = marketStatus === "OPEN";
  
  if ((isMarketOpen || isWithinEntryWindow) && bootTimeMinutes >= entryStartMinutes) {
    const dateET = toEasternDateString(bootTime);
    const alertContent = `CRITICAL BOOT DURING ENTRY WINDOW
==================================
bootTimeET: ${et.displayTime}
bootTimeUTC: ${bootTimeUTC}
currentPID: ${process.pid}
bootId: ${bootId}
marketStatus: ${marketStatus}
nextOpen: ${nextOpen}
nextClose: ${nextClose}
suspectedCause: Process started/restarted during entry window (9:35-11:35 ET)
detectedAt: ${new Date().toISOString()}
`;
    
    ensureAlertsDir();
    const alertPath = path.join(ALERTS_DIR, `CRITICAL_boot_during_entry_${dateET}.txt`);
    fs.writeFileSync(alertPath, alertContent, "utf-8");
    console.log(`[RUNTIME] CRITICAL: Boot during entry window detected! Alert written to ${alertPath}`);
    
    // Persist to Object Storage for durable receipts
    reportStorage.putText("alerts", `CRITICAL_boot_during_entry_${dateET}.txt`, alertContent).catch((err) => {
      console.error(`[RUNTIME] Failed to persist boot-during-entry alert to Object Storage:`, err?.message);
    });
  }
  
  startHeartbeat();
  startStallDetector();
  startDidRunScheduler();
}

/**
 * Register callback to restart scan loop (for stall recovery)
 */
export function registerScanRestartCallback(callback: () => Promise<void>): void {
  scanRestartCallback = callback;
}

/**
 * Notify runtime monitor that a scan tick occurred
 */
export function notifyTick(tickET: string): void {
  lastKnownTickTime = Date.now();
  if (runtimeState) {
    runtimeState.lastTickET = tickET;
  }
}

/**
 * Heartbeat - runs every 60s to prove process is alive
 */
function startHeartbeat(): void {
  if (heartbeatInterval) {
    clearInterval(heartbeatInterval);
  }
  
  heartbeatInterval = setInterval(async () => {
    if (!runtimeState) return;
    
    const et = getEasternTime();
    runtimeState.heartbeatCount++;
    runtimeState.lastHeartbeatET = et.displayTime;
    
    let marketStatus = "UNKNOWN";
    let entryAllowed = false;
    
    const isLeader = leaderLock.isLeaderInstance();
    
    try {
      const clock = await alpaca.getClock();
      marketStatus = clock.is_open ? "OPEN" : "CLOSED";
      
      const tgStatus = timeGuard.getTimeGuardStatus();
      // Entry requires: market open + time guard allows + is leader instance
      entryAllowed = clock.is_open && tgStatus.canOpenNewTrades && isLeader;
    } catch (err) {
      console.log("[RUNTIME] Heartbeat: unable to fetch clock");
    }
    
    const summary = activityLedger.getTodaysSummary();
    const ticksSinceMidnightET = summary.scanTicks;
    const ticksSinceBoot = runtimeState.ticksSinceBoot;
    const lastTickET = summary.lastTickET || runtimeState.lastTickET || "none";
    
    // Track memory for trend detection
    const memUsage = process.memoryUsage();
    const currentMemoryMB = Math.round(memUsage.heapUsed / 1024 / 1024);
    runtimeState.memoryHistory.push(currentMemoryMB);
    // Keep last 60 readings (1 hour at 60s intervals)
    if (runtimeState.memoryHistory.length > 60) {
      runtimeState.memoryHistory.shift();
    }
    
    const memoryTrend = getMemoryTrend(runtimeState.memoryHistory);
    const uptimeMinutes = Math.round((Date.now() - runtimeState.bootTime.getTime()) / 60000);
    
    console.log(`[RUNTIME] heartbeat et=${et.displayTime} marketStatus=${marketStatus} entryAllowed=${entryAllowed} isLeader=${isLeader} ticksSinceMidnightET=${ticksSinceMidnightET} ticksSinceBoot=${ticksSinceBoot} lastTickET=${lastTickET} memMB=${currentMemoryMB} memTrend=${memoryTrend} uptime=${uptimeMinutes}m`);
    
    // Write heartbeat file for downtime detection
    writeHeartbeatFile();
    
    // Persist trade accounting state on every heartbeat
    tradeAccounting.persistOnHeartbeat();
    
  }, 60 * 1000); // Every 60 seconds
}

/**
 * Stall detector - if market OPEN and no scan tick in > 5 minutes, alert and attempt restart
 */
async function checkForStall(): Promise<void> {
  if (!runtimeState) return;
  
  const STALL_THRESHOLD_MS = 5 * 60 * 1000; // 5 minutes
  
  try {
    const clock = await alpaca.getClock();
    if (!clock.is_open) return; // Only check during market hours
    
    const tgStatus = timeGuard.getTimeGuardStatus();
    
    // Only check for stalls during entry window (9:35-11:35 ET)
    if (!tgStatus.canOpenNewTrades) return;
    
    const summary = activityLedger.getTodaysSummary();
    const now = Date.now();
    const timeSinceLastTick = lastKnownTickTime > 0 ? now - lastKnownTickTime : Infinity;
    
    // If we have ticks today but none in last 5 minutes, we may be stalled
    // OR if market is open with entry window active and no ticks at all after 5 min from boot
    if ((summary.scanTicks > 0 && timeSinceLastTick > STALL_THRESHOLD_MS) ||
        (summary.scanTicks === 0 && timeSinceLastTick > STALL_THRESHOLD_MS)) {
      const et = getEasternTime();
      const lastTickET = summary.lastTickET || runtimeState.lastTickET || "unknown";
      
      console.log(`[RUNTIME] ACTION=ALERT event=stalled_scans lastTickET=${lastTickET} minutesSinceLastTick=${Math.round(timeSinceLastTick / 60000)}`);
      
      runtimeState.stallAlertCount++;
      
      // Attempt to restart scan loop if callback registered
      if (scanRestartCallback) {
        console.log("[RUNTIME] Attempting idempotent scan loop restart...");
        try {
          await scanRestartCallback();
          console.log("[RUNTIME] Scan loop restart completed");
        } catch (err) {
          console.error("[RUNTIME] Scan loop restart failed:", err);
        }
      }
    }
  } catch (err) {
    console.log("[RUNTIME] Stall check error:", err);
  }
}

function startStallDetector(): void {
  if (stallCheckInterval) {
    clearInterval(stallCheckInterval);
  }
  
  // Run stall check immediately on startup
  setTimeout(() => {
    checkForStall();
  }, 1000); // Small delay to ensure everything is initialized
  
  stallCheckInterval = setInterval(async () => {
    await checkForStall();
  }, 60 * 1000); // Check every minute
}

/**
 * Schedule "did we run?" proof file at market close + 5 minutes
 */
function startDidRunScheduler(): void {
  if (didRunInterval) {
    clearInterval(didRunInterval);
  }
  
  didRunInterval = setInterval(async () => {
    if (!runtimeState) return;
    
    const et = getEasternTime();
    
    // Already wrote for today?
    if (runtimeState.didRunFileWritten) return;
    
    try {
      const clock = await alpaca.getClock();
      
      // Check if we're past market close + 5 minutes
      const nextClose = new Date(clock.next_close);
      const now = new Date();
      
      // If market is closed and we're within 10 minutes of close, write the file
      if (!clock.is_open) {
        // Get today's calendar to find actual close time
        const calendar = await alpaca.getCalendar(et.dateString, et.dateString);
        
        if (calendar.length > 0) {
          const todayCalendar = calendar[0];
          const closeTimeET = todayCalendar.close;
          
          // Parse close time (format: "16:00")
          const [closeHour, closeMin] = closeTimeET.split(":").map(Number);
          const closeDateTime = new Date(et.dateString + "T00:00:00");
          closeDateTime.setHours(closeHour, closeMin + 5, 0, 0); // Close + 5 min
          
          if (now >= closeDateTime) {
            await writeDidRunFile(et.dateString, todayCalendar.open, todayCalendar.close);
            runtimeState.didRunFileWritten = true;
          }
        }
      }
    } catch (err) {
      console.log("[RUNTIME] Did-run scheduler error:", err);
    }
  }, 60 * 1000); // Check every minute
}

/**
 * Write the daily "did we run?" proof file
 */
async function writeDidRunFile(dateET: string, marketOpenET: string | null, marketCloseET: string | null): Promise<void> {
  if (!runtimeState) return;
  
  ensureRuntimeDir();
  
  const summary = activityLedger.getActivitySummary(dateET);
  const uptimeMinutes = Math.round((Date.now() - runtimeState.bootTime.getTime()) / 60000);
  
  const report: DidRunReport = {
    dateET,
    marketOpenET,
    marketCloseET,
    uptimeMinutes,
    heartbeatCount: runtimeState.heartbeatCount,
    scanTicks: summary.scanTicks,
    symbolsEvaluated: summary.symbolsEvaluated,
    signalsGenerated: summary.signalsGenerated,
    tradesProposed: summary.tradesProposed,
    tradesSubmitted: summary.tradesSubmitted,
    tradesFilled: summary.tradesFilled,
    lastHeartbeatET: runtimeState.lastHeartbeatET,
    lastTickET: summary.lastTickET || runtimeState.lastTickET,
    bootTimeET: runtimeState.bootTimeET,
  };
  
  const filePath = getDidRunFilePath(dateET);
  
  try {
    fs.writeFileSync(filePath, JSON.stringify(report, null, 2));
    console.log(`[RUNTIME] Wrote did_run file: ${filePath}`);
  } catch (err) {
    console.error(`[RUNTIME] Failed to write did_run file: ${err}`);
  }
}

/**
 * Increment ticksSinceBoot counter - call whenever a tick is recorded
 * OPS-METRICS-1: Clear separation between "since boot" and "since midnight" counters
 */
export function incrementTicksSinceBoot(): void {
  if (runtimeState) {
    runtimeState.ticksSinceBoot++;
  }
}

/**
 * Get current runtime status (for /health endpoint)
 * OPS-METRICS-1: Clear metric definitions
 * - ticksSinceMidnightET: Total ticks in activity ledger for today (persists across boots)
 * - ticksSinceBoot: Ticks recorded since this process started
 * ALPACA-CONNECTIVITY-PROOF-1: Added tradingState, alpacaDegraded
 */
export async function getRuntimeStatus(): Promise<{
  status: string;
  version: string | null;
  marketStatus: string;
  entryAllowed: boolean;
  isLeader: boolean;
  leaderBlocking: boolean;
  tradeEntryAllowedByLeader: boolean;
  tradingState: TradingState;
  alpacaDegraded: boolean;
  ticksSinceMidnightET: number;
  ticksSinceBoot: number;
  lastTickET: string | null;
  symbolsEvaluatedToday: number;
  signalsGeneratedToday: number;
  tradesProposedToday: number;
  tradesSubmittedToday: number;
  tradesFilledToday: number;
  heartbeatCount: number;
  lastHeartbeatET: string | null;
  uptimeMinutes: number;
  bootTimeET: string | null;
  bootTimeUTC: string | null;
  bootId: string | null;
  memoryMB: number;
  memoryTrend: string;
  baselineMemoryMB: number;
  lastAnalysisRunET: string | null;
  lastSkipReason: string | null;
  tradingStateReason: string | null;
}> {
  let marketStatus = "UNKNOWN";
  let entryAllowed = false;
  
  try {
    const clock = await alpaca.getClock();
    marketStatus = clock.is_open ? "OPEN" : "CLOSED";
    
    const tgStatus = timeGuard.getTimeGuardStatus();
    // Entry requires: market open + time guard allows + is leader instance
    const isLeader = leaderLock.isLeaderInstance();
    entryAllowed = clock.is_open && tgStatus.canOpenNewTrades && isLeader;
  } catch (err) {
    console.log("[RUNTIME] getRuntimeStatus: unable to fetch clock");
  }
  
  const summary = activityLedger.getTodaysSummary();
  const memUsage = process.memoryUsage();
  const currentMemoryMB = Math.round(memUsage.heapUsed / 1024 / 1024);
  const memoryTrend = runtimeState ? getMemoryTrend(runtimeState.memoryHistory) : "UNKNOWN";
  
  const isLeaderVal = leaderLock.isLeaderInstance();
  const connectivity = alpaca.getAlpacaConnectivityState();
  
  return {
    status: "ok",
    version: runtimeState?.version || null,
    marketStatus,
    entryAllowed,
    isLeader: isLeaderVal,
    leaderBlocking: !isLeaderVal,
    tradeEntryAllowedByLeader: isLeaderVal && entryAllowed,
    tradingState: recoveryStatus.state,
    alpacaDegraded: connectivity.degraded,
    ticksSinceMidnightET: summary.scanTicks,
    ticksSinceBoot: runtimeState?.ticksSinceBoot || 0,
    lastTickET: summary.lastTickET || runtimeState?.lastTickET || null,
    symbolsEvaluatedToday: summary.symbolsEvaluated,
    signalsGeneratedToday: summary.signalsGenerated,
    tradesProposedToday: summary.tradesProposed,
    tradesSubmittedToday: summary.tradesSubmitted,
    tradesFilledToday: summary.tradesFilled,
    heartbeatCount: runtimeState?.heartbeatCount || 0,
    lastHeartbeatET: runtimeState?.lastHeartbeatET || null,
    uptimeMinutes: runtimeState ? Math.round((Date.now() - runtimeState.bootTime.getTime()) / 60000) : 0,
    bootTimeET: runtimeState?.bootTimeET || null,
    bootTimeUTC: runtimeState?.bootTimeUTC || null,
    bootId: runtimeState?.bootId || null,
    memoryMB: currentMemoryMB,
    memoryTrend,
    baselineMemoryMB: runtimeState?.baselineMemoryMB || 0,
    lastAnalysisRunET: controlLoopTrace.getTraceState().lastAnalysisRunET,
    lastSkipReason: controlLoopTrace.getTraceState().lastSkipReason,
    tradingStateReason: controlLoopTrace.getTraceState().tradingStateReason,
  };
}

/**
 * Get baseline snapshot for Tuesday runbook verification
 */
export function getBaseline(): {
  version: string | null;
  bootId: string | null;
  bootTimeET: string | null;
  bootTimeUTC: string | null;
  uptimeMinutes: number;
  baselineMemoryMB: number;
  currentMemoryMB: number;
  memoryTrend: string;
  memoryHistory: number[];
  ticksSinceBoot: number;
  heartbeatCount: number;
} {
  const memUsage = process.memoryUsage();
  const currentMemoryMB = Math.round(memUsage.heapUsed / 1024 / 1024);
  
  return {
    version: runtimeState?.version || null,
    bootId: runtimeState?.bootId || null,
    bootTimeET: runtimeState?.bootTimeET || null,
    bootTimeUTC: runtimeState?.bootTimeUTC || null,
    uptimeMinutes: runtimeState ? Math.round((Date.now() - runtimeState.bootTime.getTime()) / 60000) : 0,
    baselineMemoryMB: runtimeState?.baselineMemoryMB || 0,
    currentMemoryMB,
    memoryTrend: runtimeState ? getMemoryTrend(runtimeState.memoryHistory) : "UNKNOWN",
    memoryHistory: runtimeState?.memoryHistory || [],
    ticksSinceBoot: runtimeState?.ticksSinceBoot || 0,
    heartbeatCount: runtimeState?.heartbeatCount || 0,
  };
}

/**
 * Check if did_run file exists for a given date
 */
export function didRunFileExists(dateET: string): boolean {
  return fs.existsSync(getDidRunFilePath(dateET));
}

/**
 * Get did_run report for a given date
 */
export function getDidRunReport(dateET: string): DidRunReport | null {
  const filePath = getDidRunFilePath(dateET);
  if (!fs.existsSync(filePath)) return null;
  
  try {
    const data = fs.readFileSync(filePath, "utf-8");
    return JSON.parse(data) as DidRunReport;
  } catch (err) {
    console.log(`[RUNTIME] Error reading did_run file: ${err}`);
    return null;
  }
}

/**
 * Force write did_run file (for testing/manual trigger)
 */
export async function forceWriteDidRunFile(): Promise<void> {
  const et = getEasternTime();
  
  try {
    const calendar = await alpaca.getCalendar(et.dateString, et.dateString);
    const todayCalendar = calendar.length > 0 ? calendar[0] : null;
    
    await writeDidRunFile(
      et.dateString,
      todayCalendar?.open || null,
      todayCalendar?.close || null
    );
  } catch (err) {
    console.error("[RUNTIME] forceWriteDidRunFile error:", err);
  }
}

/**
 * Cleanup on shutdown
 */
export function shutdown(): void {
  if (heartbeatInterval) {
    clearInterval(heartbeatInterval);
    heartbeatInterval = null;
  }
  if (stallCheckInterval) {
    clearInterval(stallCheckInterval);
    stallCheckInterval = null;
  }
  if (didRunInterval) {
    clearInterval(didRunInterval);
    didRunInterval = null;
  }
  console.log("[RUNTIME] Shutdown complete");
}
