import * as reportStorage from "./reportStorage";
import * as controlLoopTrace from "./controlLoopTrace";
import { getEasternTime } from "./timezone";
import { getEnvScope, getLockKey, getBootId as getEnvBootId, isEnvScopeBlocked } from "./envScope";

// ENV-SCOPE-HARDEN-1: Use centralized envScope module
const ENV_SCOPE = getEnvScope();
const LOCK_KEY = getLockKey();
const LOCK_CATEGORY = "locks";
const LOCK_TTL_MS = 90_000; // Lock expires after 90s without refresh
const REFRESH_INTERVAL_MS = 60_000;

interface LeaderLock {
  bootId: string;
  acquiredAt: string;
  lastRefresh: string;
  pid: number;
}

let currentBootId: string | null = null;
let isLeader = false;
let lockCheckFailed = false;
let lastLockError: string | null = null;
let lastAcquireReason: string | null = null;  // OPS-PROD-LOCK-3: Track acquire reason
let refreshInterval: NodeJS.Timeout | null = null;

export function initLeaderLock(bootId: string): void {
  currentBootId = bootId;
  console.log(`[LeaderLock] Initialized with bootId: ${bootId} scope=${ENV_SCOPE} lockKey=${LOCK_KEY}`);
}

// OPS-PROD-LOCK-3: Auto takeover on stale lock with explicit logging
export async function tryAcquireLock(): Promise<boolean> {
  if (!currentBootId) {
    console.error("[LEADER_LOCK] NOT_INITIALIZED - call initLeaderLock first");
    return false;
  }

  try {
    const existingLock = await readLock();
    const now = Date.now();
    
    // Case 1: No lock exists - acquire immediately
    if (!existingLock) {
      const success = await writeLock();
      if (success) {
        const wasLeader = isLeader;
        isLeader = true;
        lockCheckFailed = false;
        lastLockError = null;
        lastAcquireReason = "missing";
        const nowUTC = new Date().toISOString();
        const expiresUTC = new Date(now + LOCK_TTL_MS).toISOString();
        console.log(`[LEADER_LOCK] ACQUIRE_PROOF nowUTC=${nowUTC} reason=missing holderBootId=null expiresUTC=${expiresUTC} ttlMs=${LOCK_TTL_MS}`);
        if (!wasLeader) {
          controlLoopTrace.recordLeaderTransition(false, true, "acquire_missing");
        }
        startRefreshLoop();
        return true;
      } else {
        isLeader = false;
        lockCheckFailed = true;
        lastLockError = "Failed to write lock";
        console.log(`[LEADER_LOCK] ACQUIRE_FAILED bootId=${currentBootId} reason=write_error`);
        return false;
      }
    }
    
    // Case 2: Lock exists - check if stale or held by another
    const lastRefreshTime = new Date(existingLock.lastRefresh).getTime();
    const lockAge = now - lastRefreshTime;
    const ttlRemaining = LOCK_TTL_MS - lockAge;
    const isStale = lockAge >= LOCK_TTL_MS;
    
    // If lock held by different instance AND not stale => blocked
    if (existingLock.bootId !== currentBootId && !isStale) {
      console.log(`[LEADER_LOCK] ACQUIRE_BLOCKED bootId=${currentBootId} holderBootId=${existingLock.bootId} ttlRemaining=${Math.round(ttlRemaining/1000)}s`);
      isLeader = false;
      lockCheckFailed = true;
      lastLockError = `Lock held by ${existingLock.bootId} (pid ${existingLock.pid})`;
      return false;
    }
    
    // Case 3: Lock is stale OR we already own it => acquire/refresh
    const acquireReason = isStale ? "stale" : "self";
    if (isStale) {
      console.log(`[LEADER_LOCK] STALE_DETECTED holderBootId=${existingLock.bootId} lockAge=${Math.round(lockAge/1000)}s ttlRemaining=${Math.round(ttlRemaining/1000)}s`);
    }
    
    const success = await writeLock();
    if (success) {
      const wasLeader = isLeader;
      isLeader = true;
      lockCheckFailed = false;
      lastLockError = null;
      lastAcquireReason = acquireReason;
      const nowUTC = new Date().toISOString();
      const expiresUTC = new Date(now + LOCK_TTL_MS).toISOString();
      console.log(`[LEADER_LOCK] ACQUIRE_PROOF nowUTC=${nowUTC} reason=${acquireReason} holderBootId=${existingLock.bootId} expiresUTC=${expiresUTC} ttlMs=${LOCK_TTL_MS}`);
      if (!wasLeader) {
        controlLoopTrace.recordLeaderTransition(false, true, `acquire_${acquireReason}`);
      }
      startRefreshLoop();
      return true;
    } else {
      isLeader = false;
      lockCheckFailed = true;
      lastLockError = "Failed to write lock";
      console.log(`[LEADER_LOCK] ACQUIRE_FAILED bootId=${currentBootId} reason=write_error`);
      return false;
    }
  } catch (err: any) {
    console.error(`[LEADER_LOCK] ACQUIRE_ERROR bootId=${currentBootId} error=${err?.message}`);
    isLeader = false;
    lockCheckFailed = true;
    lastLockError = err?.message || "Unknown error";
    return false;
  }
}

async function readLock(): Promise<LeaderLock | null> {
  const content = await reportStorage.getText(LOCK_CATEGORY, LOCK_KEY);
  if (!content) return null;
  try {
    return JSON.parse(content) as LeaderLock;
  } catch {
    return null;
  }
}

async function writeLock(): Promise<boolean> {
  if (!currentBootId) return false;
  
  const lock: LeaderLock = {
    bootId: currentBootId,
    acquiredAt: new Date().toISOString(),
    lastRefresh: new Date().toISOString(),
    pid: process.pid,
  };
  
  return await reportStorage.putText(LOCK_CATEGORY, LOCK_KEY, JSON.stringify(lock, null, 2));
}

async function refreshLock(): Promise<boolean> {
  if (!currentBootId || !isLeader) return false;
  
  try {
    const existingLock = await readLock();
    
    if (existingLock && existingLock.bootId !== currentBootId) {
      console.log(`[LeaderLock] CRITICAL: Lost leadership to ${existingLock.bootId}`);
      isLeader = false;
      lockCheckFailed = true;
      lastLockError = `Lost leadership to ${existingLock.bootId}`;
      stopRefreshLoop();
      return false;
    }
    
    const nowUTC = new Date().toISOString();
    const lock: LeaderLock = {
      bootId: currentBootId,
      acquiredAt: existingLock?.acquiredAt || nowUTC,
      lastRefresh: nowUTC,
      pid: process.pid,
    };
    
    const success = await reportStorage.putText(LOCK_CATEGORY, LOCK_KEY, JSON.stringify(lock, null, 2));
    if (!success) {
      console.error("[LeaderLock] Failed to refresh lock");
    } else {
      // REFRESH_PROOF: One log line per refresh cycle for verification
      const expiresUTC = new Date(new Date(nowUTC).getTime() + LOCK_TTL_MS).toISOString();
      console.log(`[LEADER_LOCK] REFRESH_PROOF nowUTC=${nowUTC} holderBootId=${currentBootId} lastRefreshUTC=${nowUTC} expiresUTC=${expiresUTC} ttlMs=${LOCK_TTL_MS} refreshIntervalMs=${REFRESH_INTERVAL_MS}`);
    }
    return success;
  } catch (err: any) {
    console.error(`[LeaderLock] Refresh error: ${err?.message}`);
    return false;
  }
}

function startRefreshLoop(): void {
  if (refreshInterval) return;
  
  refreshInterval = setInterval(async () => {
    if (isLeader) {
      await refreshLock();
    } else {
      // Non-leader: periodically check if lock became stale and try to acquire
      await tryAcquireLock();
    }
  }, REFRESH_INTERVAL_MS);
  
  console.log(`[LeaderLock] Started refresh loop (every ${REFRESH_INTERVAL_MS/1000}s)`);
}

// OPS-PROD-LOCK-3: Start check loop even for non-leaders (to detect stale locks)
export function startLeaderCheckLoop(): void {
  if (refreshInterval) return; // Already running
  startRefreshLoop();
}

function stopRefreshLoop(): void {
  if (refreshInterval) {
    clearInterval(refreshInterval);
    refreshInterval = null;
  }
}

export function isLeaderInstance(): boolean {
  return isLeader && !lockCheckFailed;
}

export function getLeaderStatus(): {
  isLeader: boolean;
  bootId: string | null;
  lockCheckFailed: boolean;
  lastError: string | null;
} {
  return {
    isLeader,
    bootId: currentBootId,
    lockCheckFailed,
    lastError: lastLockError,
  };
}

// OPS-PROD-LOCK-2 + OPS-PROD-LOCK-3 + OPS-PROD-LOCK-4: Enhanced status with lock freshness visibility
export interface LeaderLockDetailedStatus {
  isLeader: boolean;
  envScope: string;      // OPS-PROD-LOCK-4: prod or dev
  lockKey: string;       // OPS-PROD-LOCK-4: Full lock key path
  thisBootId: string | null;
  thisPid: number;
  lockHolderBootId: string | null;
  lockHolderPid: number | null;
  lockAcquiredAtUTC: string | null;
  lockLastRefreshAtUTC: string | null;
  lockExpiresAtUTC: string | null;
  ttlSecondsRemaining: number | null;
  lockAgeSeconds: number | null;
  stale: boolean;
  acquireReason: string | null;  // OPS-PROD-LOCK-3: Track how leadership was acquired
  lockTtlMs: number;
  refreshIntervalMs: number;
  parseError: string | null;
}

export async function getDetailedLeaderStatus(): Promise<LeaderLockDetailedStatus> {
  const now = Date.now();
  const result: LeaderLockDetailedStatus = {
    isLeader,
    envScope: ENV_SCOPE,      // OPS-PROD-LOCK-4
    lockKey: LOCK_KEY,        // OPS-PROD-LOCK-4
    thisBootId: currentBootId,
    thisPid: process.pid,
    lockHolderBootId: null,
    lockHolderPid: null,
    lockAcquiredAtUTC: null,
    lockLastRefreshAtUTC: null,
    lockExpiresAtUTC: null,
    ttlSecondsRemaining: null,
    lockAgeSeconds: null,
    stale: false,
    acquireReason: lastAcquireReason,  // OPS-PROD-LOCK-3
    lockTtlMs: LOCK_TTL_MS,
    refreshIntervalMs: REFRESH_INTERVAL_MS,
    parseError: null,
  };

  try {
    const lock = await readLock();
    
    if (!lock) {
      result.parseError = "No lock file found";
      result.stale = true;
      return result;
    }

    result.lockHolderBootId = lock.bootId;
    result.lockHolderPid = lock.pid;
    result.lockAcquiredAtUTC = lock.acquiredAt;
    result.lockLastRefreshAtUTC = lock.lastRefresh;

    const lastRefreshTime = new Date(lock.lastRefresh).getTime();
    const expiresAt = lastRefreshTime + LOCK_TTL_MS;
    result.lockExpiresAtUTC = new Date(expiresAt).toISOString();
    
    const ttlRemaining = expiresAt - now;
    result.ttlSecondsRemaining = Math.round(ttlRemaining / 1000);
    result.lockAgeSeconds = Math.round((now - lastRefreshTime) / 1000);
    
    // Stale if TTL expired OR if lastRefresh is older than TTL
    result.stale = ttlRemaining <= 0;

  } catch (err: any) {
    result.parseError = err?.message || "Unknown parse error";
    result.stale = true;
  }

  return result;
}

export function shouldBlockEntry(): boolean {
  // ENV-SCOPE-HARDEN-1: Block trading if env scope mismatch detected
  if (isEnvScopeBlocked()) {
    return true;
  }
  
  if (!reportStorage.isStorageEnabled()) {
    return false;
  }
  return !isLeader || lockCheckFailed;
}
