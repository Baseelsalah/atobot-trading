/**
 * ENV-SCOPE-HARDEN-1: Single source of truth for environment scope
 * 
 * All modules MUST import envScope from here to ensure consistency.
 * This prevents the P0 bug where prod URL was reading from dev storage.
 */

import { nanoid } from "nanoid";

export type EnvScope = "prod" | "dev";

function determineEnvScope(): EnvScope {
  if (process.env.REPLIT_DEPLOYMENT === '1') {
    return 'prod';
  }
  if (process.env.NODE_ENV === 'production') {
    return 'prod';
  }
  return 'dev';
}

const BOOT_ID = `boot_${Date.now()}_${nanoid(6)}`;
const ENV_SCOPE = determineEnvScope();
const LOCK_KEY = `${ENV_SCOPE}/leader.json`;
const REPORTS_PREFIX = `atobot/${ENV_SCOPE}/reports`;
const VERSION = process.env.REPL_SLUG 
  ? `${process.env.REPL_SLUG?.slice(0, 8) || 'unknown'}`
  : (process.env.npm_package_version || 'dev');

let envScopeMismatchBlocked = false;
let envScopeMismatchReason: string | null = null;

export function getEnvScope(): EnvScope {
  return ENV_SCOPE;
}

export function getBootId(): string {
  return BOOT_ID;
}

export function getLockKey(): string {
  return LOCK_KEY;
}

export function getReportsPrefix(): string {
  return REPORTS_PREFIX;
}

export function getVersion(): string {
  return VERSION;
}

export function isEnvScopeBlocked(): boolean {
  return envScopeMismatchBlocked;
}

export function getEnvScopeMismatchReason(): string | null {
  return envScopeMismatchReason;
}

export interface EnvScopeStatus {
  envScope: EnvScope;
  lockKey: string;
  reportsPrefix: string;
  version: string;
  bootId: string;
  replitDeployment: string | undefined;
  nodeEnv: string | undefined;
  blocked: boolean;
  blockReason: string | null;
  determinedAt: string;
}

export function getEnvScopeStatus(): EnvScopeStatus {
  return {
    envScope: ENV_SCOPE,
    lockKey: LOCK_KEY,
    reportsPrefix: REPORTS_PREFIX,
    version: VERSION,
    bootId: BOOT_ID,
    replitDeployment: process.env.REPLIT_DEPLOYMENT,
    nodeEnv: process.env.NODE_ENV,
    blocked: envScopeMismatchBlocked,
    blockReason: envScopeMismatchReason,
    determinedAt: new Date().toISOString(),
  };
}

/**
 * ENV-SCOPE-HARDEN-1: Startup assertion
 * Call this at startup to verify prod context is using prod envScope.
 * If mismatch detected, logs CRITICAL and blocks trading.
 * 
 * Prod context signals: REPLIT_DEPLOYMENT=1 OR NODE_ENV=production
 */
export function assertEnvScopeConsistency(): { ok: boolean; reason: string | null } {
  const isProdDeployment = process.env.REPLIT_DEPLOYMENT === '1';
  const isProdNodeEnv = process.env.NODE_ENV === 'production';
  const isProdContext = isProdDeployment || isProdNodeEnv;
  
  if (isProdContext && ENV_SCOPE !== 'prod') {
    const reason = `CRITICAL: Prod context detected (REPLIT_DEPLOYMENT=${process.env.REPLIT_DEPLOYMENT}, NODE_ENV=${process.env.NODE_ENV}) ` +
      `but envScope=${ENV_SCOPE}. Expected envScope=prod. Trading blocked to prevent data corruption.`;
    
    console.error(`[ENV_SCOPE_ASSERT] ${reason}`);
    console.error(`[ENV_SCOPE_ASSERT] lockKey=${LOCK_KEY} reportsPrefix=${REPORTS_PREFIX}`);
    console.error(`[ENV_SCOPE_ASSERT] REPLIT_DEPLOYMENT=${process.env.REPLIT_DEPLOYMENT} NODE_ENV=${process.env.NODE_ENV}`);
    
    envScopeMismatchBlocked = true;
    envScopeMismatchReason = reason;
    
    return { ok: false, reason };
  }
  
  console.log(`[ENV_SCOPE_ASSERT] OK - envScope=${ENV_SCOPE} lockKey=${LOCK_KEY} reportsPrefix=${REPORTS_PREFIX}`);
  return { ok: true, reason: null };
}

console.log(`[ENV_SCOPE] Initialized: envScope=${ENV_SCOPE} lockKey=${LOCK_KEY} reportsPrefix=${REPORTS_PREFIX} bootId=${BOOT_ID}`);
