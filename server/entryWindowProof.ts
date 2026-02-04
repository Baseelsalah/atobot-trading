import fs from 'fs';
import path from 'path';
import { getEasternTime } from './timezone';
import * as reportStorage from './reportStorage';
import * as activityLedger from './activityLedger';

interface ProofSnapshot {
  capturedAt: string;
  capturedAtET: string;
  phase: 'before_entry' | 'during_entry' | 'after_entry';
  snapshot: any;
}

type FinalStatus = 'FINAL_OK' | 'FINAL_FAIL';
type ReasonCode = 'MISSED_ENTRY_WINDOW' | 'NO_GATES_RAN' | 'NO_SCANS' | 'BOT_NOT_RUNNING' | 'INSUFFICIENT_SNAPSHOTS';

interface ProofBundle {
  date: string;
  initializedAt: string;
  status: 'INIT' | 'ACTIVE' | 'FINAL';
  finalStatus?: FinalStatus;
  reasonCodes?: ReasonCode[];
  snapshotCount: number;
  gateLogCount: number;
  symbolsEvaluatedToday: number;
  snapshots: ProofSnapshot[];
  p1GateLogs: string[];
  p1TradabilityLogs: string[];
  passFailCounts: { pass: number; fail: number };
  topSkipReasons: { reason: string; count: number }[];
  events: string[];
}

const PROOF_DIR = 'reports/proof';
const RATE_LIMIT_MS = 60000;

let proofBundle: ProofBundle | null = null;
let capturedPhases: Set<string> = new Set();
let p1GateBuffer: string[] = [];
let p1TradabilityBuffer: string[] = [];
let passCount = 0;
let failCount = 0;
let lastSnapshotTime = 0;
let finalized = false;

// PROOF-SNAPSHOT-2: Track snapshot attempts for visibility
let lastSnapshotAttemptET: string | null = null;
let lastSnapshotSkipReason: string | null = null;

// GATE-TRUTH-1: Track tradability gate runs for transparency
let tradabilityGateRunsToday = 0;
let lastGateRunET: string | null = null;
let lastGateRunDate: string | null = null;

function getProofFilePath(dateET: string): string {
  return path.join(PROOF_DIR, `proof_${dateET}.json`);
}

function saveProofFile(): void {
  if (!proofBundle) return;
  
  if (!fs.existsSync(PROOF_DIR)) {
    fs.mkdirSync(PROOF_DIR, { recursive: true });
  }
  
  proofBundle.snapshotCount = proofBundle.snapshots.length;
  proofBundle.gateLogCount = p1GateBuffer.length;
  proofBundle.p1GateLogs = p1GateBuffer.slice(0, 100);
  proofBundle.p1TradabilityLogs = p1TradabilityBuffer.slice(0, 50);
  proofBundle.passFailCounts = { pass: passCount, fail: failCount };
  
  const filename = getProofFilePath(proofBundle.date);
  const content = JSON.stringify(proofBundle, null, 2);
  fs.writeFileSync(filename, content);
  reportStorage.putText("proof", `proof_${proofBundle.date}.json`, content).catch(() => {});
}

export function initProofBundle(dateET: string): void {
  const eastern = getEasternTime();
  const nowET = `${eastern.hour}:${String(eastern.minute).padStart(2, '0')} ET`;
  
  proofBundle = {
    date: dateET,
    initializedAt: new Date().toISOString(),
    status: 'INIT',
    snapshotCount: 0,
    gateLogCount: 0,
    symbolsEvaluatedToday: 0,
    snapshots: [],
    p1GateLogs: [],
    p1TradabilityLogs: [],
    passFailCounts: { pass: 0, fail: 0 },
    topSkipReasons: [],
    events: [`${nowET}: Proof bundle initialized`]
  };
  capturedPhases.clear();
  p1GateBuffer = [];
  p1TradabilityBuffer = [];
  passCount = 0;
  failCount = 0;
  lastSnapshotTime = 0;
  finalized = false;
  
  saveProofFile();
  console.log(`[ProofBundle] Initialized for ${dateET} - file created with status=INIT`);
}

export function recordP1GateLog(logLine: string, passed: boolean): void {
  p1GateBuffer.push(logLine);
  if (passed) {
    passCount++;
  } else {
    failCount++;
  }
  
  if (proofBundle && proofBundle.status === 'INIT') {
    proofBundle.status = 'ACTIVE';
    const eastern = getEasternTime();
    proofBundle.events.push(`${eastern.hour}:${String(eastern.minute).padStart(2, '0')} ET: First P1 gate log recorded`);
  }
}

export function recordP1TradabilityLog(logLine: string): void {
  p1TradabilityBuffer.push(logLine);
}

// GATE-TRUTH-1: Record that tradability gates ran this tick
export function recordTradabilityGateRun(symbolCount: number, passedCount: number, failedCount: number): void {
  const eastern = getEasternTime();
  const nowET = `${eastern.hour}:${String(eastern.minute).padStart(2, '0')} ET`;
  
  // Reset counter on date change
  if (lastGateRunDate !== eastern.dateString) {
    tradabilityGateRunsToday = 0;
    lastGateRunDate = eastern.dateString;
  }
  
  tradabilityGateRunsToday++;
  lastGateRunET = nowET;
  
  console.log(`[P1:TRADABILITY] ran=true symbols=${symbolCount} pass=${passedCount} fail=${failedCount} runsToday=${tradabilityGateRunsToday}`);
}

// GATE-TRUTH-1: Record when tradability gates were skipped
export function recordTradabilityGateSkip(reason: string): void {
  console.log(`[P1:TRADABILITY] ran=false reason=${reason} lastRunET=${lastGateRunET || 'never'}`);
}

// GATE-TRUTH-1: Get gate run stats
export function getTradabilityGateStats(): { tradabilityGateRunsToday: number; lastGateRunET: string | null; gateLogSource: string } {
  return {
    tradabilityGateRunsToday,
    lastGateRunET,
    gateLogSource: 'entryWindowProof.p1GateBuffer'
  };
}

export function captureSnapshot(phase: 'before_entry' | 'during_entry' | 'after_entry', snapshot: any): boolean {
  const eastern = getEasternTime();
  const nowET = `${eastern.hour}:${String(eastern.minute).padStart(2, '0')} ET`;
  
  // PROOF-SNAPSHOT-2: Track all attempts
  lastSnapshotAttemptET = nowET;
  
  if (!proofBundle) {
    lastSnapshotSkipReason = 'proof_bundle_not_initialized';
    console.log(`[ProofBundle] Snapshot attempt at ${nowET}: SKIPPED (${lastSnapshotSkipReason})`);
    return false;
  }
  
  const now = Date.now();
  if (now - lastSnapshotTime < RATE_LIMIT_MS) {
    lastSnapshotSkipReason = `rate_limited (${Math.round((RATE_LIMIT_MS - (now - lastSnapshotTime)) / 1000)}s remaining)`;
    console.log(`[ProofBundle] Snapshot attempt at ${nowET}: SKIPPED (${lastSnapshotSkipReason})`);
    return false;
  }
  
  const phaseKey = `${proofBundle.date}_${phase}`;
  if (capturedPhases.has(phaseKey)) {
    lastSnapshotSkipReason = `phase_already_captured (${phase})`;
    console.log(`[ProofBundle] Snapshot attempt at ${nowET}: SKIPPED (${lastSnapshotSkipReason})`);
    return false;
  }
  
  // Success - clear skip reason
  lastSnapshotSkipReason = null;
  
  proofBundle.snapshots.push({
    capturedAt: new Date().toISOString(),
    capturedAtET: nowET,
    phase,
    snapshot
  });
  capturedPhases.add(phaseKey);
  lastSnapshotTime = now;
  proofBundle.events.push(`${nowET}: Captured ${phase} snapshot`);
  
  saveProofFile();
  console.log(`[ProofBundle] Captured ${phase} snapshot at ${nowET}`);
  return true;
}

export function finalizeProofBundle(topSkipReasons: { reason: string; count: number }[], symbolsEvaluatedToday: number): boolean {
  if (!proofBundle) return false;
  if (finalized) return false;
  
  const eastern = getEasternTime();
  const nowET = `${eastern.hour}:${String(eastern.minute).padStart(2, '0')} ET`;
  
  proofBundle.status = 'FINAL';
  proofBundle.topSkipReasons = topSkipReasons;
  proofBundle.symbolsEvaluatedToday = symbolsEvaluatedToday;
  
  // OPS-ENTRY-PROOF-1: Determine FINAL_OK or FINAL_FAIL
  const snapshotCount = proofBundle.snapshots.length;
  const gateLogCount = p1GateBuffer.length;
  const hasActivity = gateLogCount > 0 || symbolsEvaluatedToday > 0;
  const hasSufficientSnapshots = snapshotCount >= 2;
  
  if (hasSufficientSnapshots && hasActivity) {
    proofBundle.finalStatus = 'FINAL_OK';
    proofBundle.events.push(`${nowET}: Proof bundle finalized with FINAL_OK`);
    console.log(`[ProofBundle] FINAL_OK: ${passCount} PASS, ${failCount} FAIL, ${snapshotCount} snapshots, ${symbolsEvaluatedToday} symbols evaluated`);
  } else {
    proofBundle.finalStatus = 'FINAL_FAIL';
    proofBundle.reasonCodes = determineReasonCodes(snapshotCount, gateLogCount, symbolsEvaluatedToday);
    proofBundle.events.push(`${nowET}: Proof bundle finalized with FINAL_FAIL - ${proofBundle.reasonCodes.join(', ')}`);
    console.log(`[ProofBundle] CRITICAL: FINAL_FAIL - reasons: ${proofBundle.reasonCodes.join(', ')}`);
    
    // Write CRITICAL alert file
    writeCriticalAlertFile(proofBundle.date, proofBundle.reasonCodes, {
      snapshotCount,
      gateLogCount,
      symbolsEvaluatedToday,
      passCount,
      failCount,
    });
  }
  
  saveProofFile();
  finalized = true;
  
  return true;
}

function determineReasonCodes(snapshotCount: number, gateLogCount: number, symbolsEvaluatedToday: number): ReasonCode[] {
  const reasons: ReasonCode[] = [];
  
  if (snapshotCount < 2) {
    reasons.push('INSUFFICIENT_SNAPSHOTS');
    if (snapshotCount === 0) {
      reasons.push('BOT_NOT_RUNNING');
    }
  }
  
  if (gateLogCount === 0 && symbolsEvaluatedToday === 0) {
    reasons.push('MISSED_ENTRY_WINDOW');
    reasons.push('NO_SCANS');
  } else if (gateLogCount === 0) {
    reasons.push('NO_GATES_RAN');
  }
  
  return reasons.length > 0 ? reasons : ['MISSED_ENTRY_WINDOW'];
}

function writeCriticalAlertFile(
  date: string, 
  reasonCodes: ReasonCode[], 
  stats: { snapshotCount: number; gateLogCount: number; symbolsEvaluatedToday: number; passCount: number; failCount: number }
): void {
  const eastern = getEasternTime();
  const nowET = `${eastern.hour}:${String(eastern.minute).padStart(2, '0')} ET`;
  
  const alertContent = `CRITICAL: Proof Bundle FINAL_FAIL
Date: ${date}
Finalized At: ${nowET}
Reason Codes: ${reasonCodes.join(', ')}

Statistics:
- Snapshots Captured: ${stats.snapshotCount} (required: >= 2)
- Gate Logs: ${stats.gateLogCount}
- Symbols Evaluated: ${stats.symbolsEvaluatedToday}
- Pass/Fail Counts: ${stats.passCount} PASS, ${stats.failCount} FAIL

This alert indicates the entry window did not execute as expected.
Review the proof bundle for more details: reports/proof/proof_${date}.json
`;
  
  // Write to local filesystem
  const alertDir = 'reports/alerts';
  if (!fs.existsSync(alertDir)) {
    fs.mkdirSync(alertDir, { recursive: true });
  }
  const alertFilename = path.join(alertDir, `CRITICAL_proof_fail_${date}.txt`);
  fs.writeFileSync(alertFilename, alertContent);
  
  // Write to Object Storage for durability
  reportStorage.putText("alerts", `CRITICAL_proof_fail_${date}.txt`, alertContent).catch((err) => {
    console.error(`[ProofBundle] Failed to write CRITICAL alert to storage: ${err}`);
  });
  
  console.log(`[ProofBundle] CRITICAL alert written to ${alertFilename} and Object Storage`);
}

export type CurrentPhase = 'pre_market' | 'before_entry' | 'entry_window' | 'after_entry' | 'management' | 'closed';

function getCurrentPhase(): CurrentPhase {
  const eastern = getEasternTime();
  const totalMinutes = eastern.hour * 60 + eastern.minute;
  
  if (totalMinutes < 9 * 60 + 30) return 'pre_market';
  if (totalMinutes < 9 * 60 + 35) return 'before_entry';
  if (totalMinutes < 11 * 60 + 35) return 'entry_window';
  if (totalMinutes < 11 * 60 + 45) return 'after_entry';
  if (totalMinutes < 15 * 60 + 45) return 'management';
  return 'closed';
}

export function getProofStatus(): { 
  initialized: boolean; 
  date: string | null; 
  status: string | null;
  finalStatus: string | null;
  reasonCodes: string[] | null;
  snapshotCount: number; 
  gateLogCount: number;
  symbolsEvaluatedToday: number;
  symbolsEvaluatedSource: string;
  finalized: boolean;
  expectedFinalizeET: string;
  currentPhase: CurrentPhase;
  passCriteria: {
    requiredSnapshots: number;
    hasSnapshots: boolean;
    hasActivity: boolean;
    wouldPass: boolean;
  };
  shouldCaptureBeforeEntry: boolean;
  shouldCaptureDuringEntry: boolean;
  shouldCaptureAfterEntry: boolean;
  lastSnapshotAttemptET: string | null;
  lastSnapshotSkipReason: string | null;
  tradabilityGateRunsToday: number;
  lastGateRunET: string | null;
  gateLogSource: string;
} {
  const snapshotCount = proofBundle?.snapshots.length || 0;
  const gateLogCount = p1GateBuffer.length;
  
  // PROOF-ALIGN-1: Read symbolsEvaluatedToday from activityLedger (same source as /health)
  const eastern = getEasternTime();
  const summary = activityLedger.getActivitySummary(eastern.dateString);
  const symbolsEvaluated = summary.symbolsEvaluated;
  const symbolsEvaluatedSource = 'activityLedger.getActivitySummary';
  
  // GATE-TRUTH-1: Get gate run stats
  const gateStats = getTradabilityGateStats();
  
  const hasActivity = gateLogCount > 0 || symbolsEvaluated > 0;
  const hasSufficientSnapshots = snapshotCount >= 2;
  
  return {
    initialized: proofBundle !== null,
    date: proofBundle?.date || null,
    status: proofBundle?.status || null,
    finalStatus: proofBundle?.finalStatus || null,
    reasonCodes: proofBundle?.reasonCodes || null,
    snapshotCount,
    gateLogCount,
    symbolsEvaluatedToday: symbolsEvaluated,
    symbolsEvaluatedSource,
    finalized,
    expectedFinalizeET: '11:45 ET',
    currentPhase: getCurrentPhase(),
    passCriteria: {
      requiredSnapshots: 2,
      hasSnapshots: hasSufficientSnapshots,
      hasActivity,
      wouldPass: hasSufficientSnapshots && hasActivity,
    },
    shouldCaptureBeforeEntry: shouldCaptureBeforeEntry(),
    shouldCaptureDuringEntry: shouldCaptureDuringEntry(),
    shouldCaptureAfterEntry: shouldCaptureAfterEntry(),
    lastSnapshotAttemptET,
    lastSnapshotSkipReason,
    tradabilityGateRunsToday: gateStats.tradabilityGateRunsToday,
    lastGateRunET: gateStats.lastGateRunET,
    gateLogSource: gateStats.gateLogSource,
  };
}

export function shouldCaptureBeforeEntry(): boolean {
  const eastern = getEasternTime();
  const totalMinutes = eastern.hour * 60 + eastern.minute;
  return totalMinutes >= 9 * 60 + 30 && totalMinutes <= 9 * 60 + 35;
}

export function shouldCaptureDuringEntry(): boolean {
  const eastern = getEasternTime();
  const totalMinutes = eastern.hour * 60 + eastern.minute;
  return totalMinutes >= 9 * 60 + 38 && totalMinutes <= 10 * 60 + 35;
}

export function shouldCaptureAfterEntry(): boolean {
  const eastern = getEasternTime();
  const totalMinutes = eastern.hour * 60 + eastern.minute;
  return totalMinutes >= 11 * 60 + 40 && totalMinutes <= 12 * 60;
}

export function isFinalized(): boolean {
  return finalized;
}

export function getTopSkipReasonsForProof(): { reason: string; count: number }[] {
  return proofBundle?.topSkipReasons || [];
}
