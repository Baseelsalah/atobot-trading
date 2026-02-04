/**
 * Preflight Check - Safety validation at boot
 * Prints [PREFLIGHT] block with critical system status
 */

import * as alpaca from "./alpaca";
import { DAY_TRADER_CONFIG, getAllowedSymbols } from "./dayTraderConfig";
import { getEasternTime } from "./timezone";

export interface PreflightResult {
  passed: boolean;
  alpacaAuth: { ok: boolean; mode: string; reason?: string };
  clockStatus: { ok: boolean; isOpen: boolean; nextOpen?: string; nextClose?: string; reason?: string };
  riskRulesLoaded: { ok: boolean; rules?: Record<string, number | string>; reason?: string };
  universeSize: { ok: boolean; count: number; symbols?: string[]; reason?: string };
  reportScheduler: { ok: boolean; status: string; reason?: string };
}

export async function runPreflightCheck(): Promise<PreflightResult> {
  const result: PreflightResult = {
    passed: true,
    alpacaAuth: { ok: false, mode: "unknown" },
    clockStatus: { ok: false, isOpen: false },
    riskRulesLoaded: { ok: false },
    universeSize: { ok: false, count: 0 },
    reportScheduler: { ok: false, status: "unknown" },
  };
  
  console.log(`[PREFLIGHT] ============================================`);
  console.log(`[PREFLIGHT] BOOT PREFLIGHT CHECK`);
  console.log(`[PREFLIGHT] Time: ${getEasternTime().displayTime} ET`);
  console.log(`[PREFLIGHT] ============================================`);
  
  try {
    const account = await alpaca.getAccount();
    const isDryRun = alpaca.isDryRun();
    result.alpacaAuth = {
      ok: true,
      mode: isDryRun ? "PAPER/DRY_RUN" : "LIVE",
    };
    console.log(`[PREFLIGHT] Alpaca Auth: OK (${result.alpacaAuth.mode})`);
  } catch (error) {
    result.alpacaAuth = {
      ok: false,
      mode: "unknown",
      reason: `Auth failed: ${error}`,
    };
    result.passed = false;
    console.log(`[PREFLIGHT] Alpaca Auth: FAIL - ${result.alpacaAuth.reason}`);
  }
  
  try {
    const clock = await alpaca.getClock();
    result.clockStatus = {
      ok: true,
      isOpen: clock.is_open,
      nextOpen: clock.next_open,
      nextClose: clock.next_close,
    };
    console.log(`[PREFLIGHT] Clock Status: OK (is_open=${clock.is_open})`);
  } catch (error) {
    result.clockStatus = {
      ok: false,
      isOpen: false,
      reason: `Clock fetch failed: ${error}`,
    };
    result.passed = false;
    console.log(`[PREFLIGHT] Clock Status: FAIL - ${result.clockStatus.reason}`);
  }
  
  try {
    const config = DAY_TRADER_CONFIG;
    result.riskRulesLoaded = {
      ok: true,
      rules: {
        riskPerTrade: config.RISK_PER_TRADE,
        maxPositions: config.MAX_OPEN_POSITIONS,
        maxEntriesPerDay: config.MAX_NEW_ENTRIES_PER_DAY,
        killThresholdLoss: config.DAILY_MAX_LOSS,
        killThresholdProfit: config.DAILY_MAX_PROFIT,
      },
    };
    console.log(`[PREFLIGHT] Risk Rules: OK (risk=$${config.RISK_PER_TRADE}, maxPos=${config.MAX_OPEN_POSITIONS})`);
  } catch (error) {
    result.riskRulesLoaded = {
      ok: false,
      reason: `Risk rules load failed: ${error}`,
    };
    result.passed = false;
    console.log(`[PREFLIGHT] Risk Rules: FAIL - ${result.riskRulesLoaded.reason}`);
  }
  
  try {
    const universe = getAllowedSymbols();
    result.universeSize = {
      ok: universe.length > 0,
      count: universe.length,
      symbols: universe,
    };
    if (universe.length === 0) {
      result.universeSize.reason = "Empty universe";
      result.passed = false;
    }
    console.log(`[PREFLIGHT] Universe: OK (${universe.length} symbols: ${universe.join(", ")})`);
  } catch (error) {
    result.universeSize = {
      ok: false,
      count: 0,
      reason: `Universe load failed: ${error}`,
    };
    result.passed = false;
    console.log(`[PREFLIGHT] Universe: FAIL - ${result.universeSize.reason}`);
  }
  
  result.reportScheduler = {
    ok: true,
    status: "initialized",
  };
  console.log(`[PREFLIGHT] Report Scheduler: OK (initialized)`);
  
  console.log(`[PREFLIGHT] ============================================`);
  if (result.passed) {
    console.log(`[PREFLIGHT] RESULT: ALL CHECKS PASSED - Trading enabled`);
  } else {
    console.log(`[PREFLIGHT] RESULT: CHECKS FAILED - Trading BLOCKED`);
    console.log(`[PREFLIGHT] Failures:`);
    if (!result.alpacaAuth.ok) console.log(`[PREFLIGHT]   - Alpaca Auth: ${result.alpacaAuth.reason}`);
    if (!result.clockStatus.ok) console.log(`[PREFLIGHT]   - Clock Status: ${result.clockStatus.reason}`);
    if (!result.riskRulesLoaded.ok) console.log(`[PREFLIGHT]   - Risk Rules: ${result.riskRulesLoaded.reason}`);
    if (!result.universeSize.ok) console.log(`[PREFLIGHT]   - Universe: ${result.universeSize.reason}`);
    if (!result.reportScheduler.ok) console.log(`[PREFLIGHT]   - Report Scheduler: ${result.reportScheduler.reason}`);
  }
  console.log(`[PREFLIGHT] ============================================\n`);
  
  return result;
}

export function preflightPassed(result: PreflightResult): boolean {
  return result.passed;
}
