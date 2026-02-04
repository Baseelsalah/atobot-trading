import { objectStorageClient } from "./replit_integrations/object_storage";
import { getEnvScope as getCentralEnvScope, getReportsPrefix as getCentralReportsPrefix } from "./envScope";

const BUCKET_ID = process.env.DEFAULT_OBJECT_STORAGE_BUCKET_ID || "";

// ENV-SCOPE-HARDEN-1: Use centralized envScope module
const ENV_SCOPE = getCentralEnvScope();
const REPORTS_PREFIX = getCentralReportsPrefix();

export function getStorageEnvScope(): "prod" | "dev" {
  return ENV_SCOPE;
}

interface StorageState {
  enabled: boolean;
  lastWriteOk: boolean;
  lastWriteTsUTC: string | null;
  lastError: string | null;
  envScope: "prod" | "dev";
  reportsPrefix: string;
  keysPresentToday: {
    activity: boolean;
    runtime: boolean;
    proof: boolean;
    accounting: boolean;
    alerts: boolean;
    execution: boolean;
  };
}

const state: StorageState = {
  enabled: false,
  lastWriteOk: false,
  lastWriteTsUTC: null,
  lastError: null,
  envScope: ENV_SCOPE,
  reportsPrefix: REPORTS_PREFIX,
  keysPresentToday: {
    activity: false,
    runtime: false,
    proof: false,
    accounting: false,
    alerts: false,
    execution: false,
  },
};

export function isStorageEnabled(): boolean {
  return !!BUCKET_ID && state.enabled;
}

export function initReportStorage(): void {
  if (!BUCKET_ID) {
    console.log("[ReportStorage] No bucket configured, storage disabled");
    state.enabled = false;
    return;
  }
  state.enabled = true;
  console.log(`[ReportStorage] Initialized bucket=${BUCKET_ID.slice(0, 20)}... envScope=${ENV_SCOPE} prefix=${REPORTS_PREFIX}`);
}

function getBucket() {
  if (!BUCKET_ID) throw new Error("No bucket configured");
  return objectStorageClient.bucket(BUCKET_ID);
}

function getKey(category: string, filename: string): string {
  return `${REPORTS_PREFIX}/${category}/${filename}`;
}

export async function putText(category: string, filename: string, text: string): Promise<boolean> {
  if (!state.enabled) return false;
  
  try {
    const bucket = getBucket();
    const key = getKey(category, filename);
    const file = bucket.file(key);
    
    await file.save(text, {
      contentType: "application/json",
      resumable: false,
    });
    
    state.lastWriteOk = true;
    state.lastWriteTsUTC = new Date().toISOString();
    state.lastError = null;
    
    if (category === "activity") state.keysPresentToday.activity = true;
    if (category === "runtime") state.keysPresentToday.runtime = true;
    if (category === "proof") state.keysPresentToday.proof = true;
    if (category === "accounting") state.keysPresentToday.accounting = true;
    if (category === "alerts") state.keysPresentToday.alerts = true;
    if (category === "execution") state.keysPresentToday.execution = true;
    
    return true;
  } catch (err: any) {
    state.lastWriteOk = false;
    state.lastError = err?.message || "Unknown error";
    console.error(`[ReportStorage] Write failed for ${category}/${filename}:`, err?.message);
    return false;
  }
}

export async function getText(category: string, filename: string): Promise<string | null> {
  if (!state.enabled) return null;
  
  try {
    const bucket = getBucket();
    const key = getKey(category, filename);
    const file = bucket.file(key);
    
    const [exists] = await file.exists();
    if (!exists) return null;
    
    const [contents] = await file.download();
    return contents.toString("utf-8");
  } catch (err: any) {
    console.error(`[ReportStorage] Read failed for ${category}/${filename}:`, err?.message);
    return null;
  }
}

export async function exists(category: string, filename: string): Promise<boolean> {
  if (!state.enabled) return false;
  
  try {
    const bucket = getBucket();
    const key = getKey(category, filename);
    const file = bucket.file(key);
    const [fileExists] = await file.exists();
    return fileExists;
  } catch {
    return false;
  }
}

export async function listKeys(category: string, prefix?: string): Promise<string[]> {
  if (!state.enabled) return [];
  
  try {
    const bucket = getBucket();
    const fullPrefix = prefix 
      ? `${REPORTS_PREFIX}/${category}/${prefix}`
      : `${REPORTS_PREFIX}/${category}/`;
    
    const [files] = await bucket.getFiles({ prefix: fullPrefix });
    return files.map(f => f.name.replace(`${REPORTS_PREFIX}/${category}/`, ""));
  } catch (err: any) {
    console.error(`[ReportStorage] List failed for ${category}:`, err?.message);
    return [];
  }
}

export async function checkTodayKeys(dateStr: string): Promise<void> {
  state.keysPresentToday.activity = await exists("activity", `activity_${dateStr}.json`);
  // Check for boots file OR heartbeat (heartbeat updates every 60s, boots on boot)
  const hasBoots = await exists("runtime", `boots_${dateStr}.jsonl`);
  const hasHeartbeat = await exists("runtime", "heartbeat_latest.json");
  state.keysPresentToday.runtime = hasBoots || hasHeartbeat;
  state.keysPresentToday.proof = await exists("proof", `proof_${dateStr}.json`);
  state.keysPresentToday.accounting = await exists("accounting", `state_${dateStr}.json`);
  
  // OPS-DOWNTIME-PROOF-2 + RECOVERY-MODE-1: Check for ANY CRITICAL_* alert for today
  const hasProofFail = await exists("alerts", `CRITICAL_proof_fail_${dateStr}.txt`);
  const hasDowntime = await exists("alerts", `CRITICAL_downtime_detected_${dateStr}.txt`);
  const hasBootDuringEntry = await exists("alerts", `CRITICAL_boot_during_entry_${dateStr}.txt`);
  const hasRecoveryFailed = await exists("alerts", `CRITICAL_recovery_failed_${dateStr}.txt`);
  state.keysPresentToday.alerts = hasProofFail || hasDowntime || hasBootDuringEntry || hasRecoveryFailed;
  
  state.keysPresentToday.execution = await exists("execution", `execution_recent_${dateStr}.json`);
}

export function getStorageStatus(): StorageState {
  return { ...state };
}

export async function appendLine(category: string, filename: string, line: string): Promise<boolean> {
  if (!state.enabled) return false;
  
  try {
    const existing = await getText(category, filename);
    const newContent = existing ? existing + "\n" + line : line;
    return await putText(category, filename, newContent);
  } catch (err: any) {
    console.error(`[ReportStorage] Append failed for ${category}/${filename}:`, err?.message);
    return false;
  }
}
