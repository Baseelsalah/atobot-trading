import * as alpaca from "./alpaca";
import { storage } from "./storage";
import { getEasternTime, toMinutesSinceMidnight, logCurrentTime } from "./timezone";
import { printEODSummary } from "./signalCounters";
import * as positionManager from "./positionManager";
import { initializeRiskEngine } from "./riskEngine";

const TIME_GUARD_OVERRIDE = process.env.TIME_GUARD_OVERRIDE === "1";
if (TIME_GUARD_OVERRIDE) {
  console.log("[TIME GUARD] OVERRIDE ENABLED - time restrictions disabled (simulation mode)");
}

/**
 * ============================================================
 * FORT KNOX TRADING SCHEDULE - DYNAMIC EARLY CLOSE AWARE
 * ============================================================
 * 
 * ENTRY WINDOW (NEW POSITIONS ALLOWED):
 *   START: 9:35 AM ET (5 minutes after market opens)
 *   END:   Dynamic - min(2:00 PM ET, next_close - 5 minutes)
 * 
 * POSITION MANAGEMENT WINDOW:
 *   Entry cutoff - Force close: Manage existing positions only
 * 
 * FORCE CLOSE (DYNAMIC):
 *   min(3:45 PM ET, next_close - 2 minutes)
 *   On early close days (Christmas Eve, Black Friday, etc.),
 *   force close triggers 2 minutes before market close.
 * 
 * NORMAL DAY TIME CONVERSIONS:
 *   9:35 AM ET  = 6:35 AM PT (Entry Start)
 *   11:35 AM ET = 8:35 AM PT (Entry Cutoff - normal)
 *   3:45 PM ET  = 12:45 PM PT (Force Close - normal)
 * 
 * EARLY CLOSE (e.g., Christmas Eve 1:00 PM ET close):
 *   Entry Cutoff: 12:55 PM ET (10:55 AM PT) - next_close - 5 min
 *   Force Close:  12:58 PM ET (10:58 AM PT) - next_close - 2 min
 * ============================================================
 */

// ENTRY WINDOW (LEGACY - used as max bounds)
const ENTRY_START_HOUR_ET = 9;
const ENTRY_START_MINUTE_ET = 35;
const LEGACY_ENTRY_CUTOFF_HOUR_ET = 14; // Extended to 2:00 PM ET — proven in backtest for more EMA crossover opportunities
const LEGACY_ENTRY_CUTOFF_MINUTE_ET = 0;

// FORCE CLOSE (LEGACY - used as max bounds)
const LEGACY_FORCE_CLOSE_HOUR_ET = 15;
const LEGACY_FORCE_CLOSE_MINUTE_ET = 45;

// Dynamic cutoff offsets from next_close
const ENTRY_CUTOFF_BEFORE_CLOSE_MINUTES = 5;  // Stop entries 5 min before close
const FORCE_CLOSE_BEFORE_CLOSE_MINUTES = 2;   // Force close 2 min before close

// Market open reference (for display only)
const MARKET_OPEN_HOUR_ET = 9;
const MARKET_OPEN_MINUTE_ET = 30;

// Legacy aliases for backward compatibility
const ENTRY_CUTOFF_HOUR_ET = LEGACY_ENTRY_CUTOFF_HOUR_ET;
const ENTRY_CUTOFF_MINUTE_ET = LEGACY_ENTRY_CUTOFF_MINUTE_ET;
const FORCE_CLOSE_HOUR_ET = LEGACY_FORCE_CLOSE_HOUR_ET;
const FORCE_CLOSE_MINUTE_ET = LEGACY_FORCE_CLOSE_MINUTE_ET;

let endOfDayExecuted = false;
let lastCheckedDate: string | null = null;
let guardInterval: NodeJS.Timeout | null = null;

// Cached clock data for dynamic schedule
interface CachedClockData {
  nextClose: Date | null;
  nextCloseET: string;
  isEarlyClose: boolean;
  dynamicEntryCutoffMinutes: number;
  dynamicForceCloseMinutes: number;
  entryCutoffPT: string;
  forceClosePT: string;
  cachedAt: number;
}

let cachedClock: CachedClockData | null = null;
const CLOCK_CACHE_TTL_MS = 60 * 1000; // 1 minute cache

/**
 * Fetch and cache Alpaca clock data for dynamic schedule computation
 */
async function refreshClockCache(): Promise<CachedClockData> {
  try {
    const clock = await alpaca.getClock();
    const nextCloseStr = clock.next_close;
    const nextClose = new Date(nextCloseStr);
    
    // Convert next_close to ET minutes since midnight
    const nextCloseET = new Intl.DateTimeFormat("en-US", {
      timeZone: "America/New_York",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(nextClose);
    const [closeHour, closeMinute] = nextCloseET.split(":").map(Number);
    const nextCloseMinutesET = toMinutesSinceMidnight(closeHour, closeMinute);
    
    // Legacy cutoffs in minutes
    const legacyEntryCutoffMinutes = toMinutesSinceMidnight(LEGACY_ENTRY_CUTOFF_HOUR_ET, LEGACY_ENTRY_CUTOFF_MINUTE_ET);
    const legacyForceCloseMinutes = toMinutesSinceMidnight(LEGACY_FORCE_CLOSE_HOUR_ET, LEGACY_FORCE_CLOSE_MINUTE_ET);
    
    // Dynamic cutoffs based on next_close
    const dynamicEntryCutoff = nextCloseMinutesET - ENTRY_CUTOFF_BEFORE_CLOSE_MINUTES;
    const dynamicForceClose = nextCloseMinutesET - FORCE_CLOSE_BEFORE_CLOSE_MINUTES;
    
    // Use min of legacy and dynamic
    const effectiveEntryCutoff = Math.min(legacyEntryCutoffMinutes, dynamicEntryCutoff);
    const effectiveForceClose = Math.min(legacyForceCloseMinutes, dynamicForceClose);
    
    // Check if this is an early close day
    const isEarlyClose = nextCloseMinutesET < toMinutesSinceMidnight(16, 0); // Normal close is 4:00 PM ET
    
    // Format times for logging (PT = ET - 3 hours)
    const entryCutoffHour = Math.floor(effectiveEntryCutoff / 60);
    const entryCutoffMin = effectiveEntryCutoff % 60;
    const forceCloseHour = Math.floor(effectiveForceClose / 60);
    const forceCloseMin = effectiveForceClose % 60;
    
    // Convert to PT (subtract 3 hours)
    const entryCutoffPTHour = entryCutoffHour - 3;
    const forceClosePTHour = forceCloseHour - 3;
    
    const formatTime = (h: number, m: number): string => {
      const hour12 = h > 12 ? h - 12 : (h === 0 ? 12 : h);
      const ampm = h >= 12 ? "PM" : "AM";
      return `${hour12}:${m.toString().padStart(2, "0")} ${ampm}`;
    };
    
    const closeHourPT = closeHour - 3;
    const nextClosePT = formatTime(closeHourPT, closeMinute) + " PT";
    
    cachedClock = {
      nextClose,
      nextCloseET: `${closeHour}:${closeMinute.toString().padStart(2, "0")} ET`,
      isEarlyClose,
      dynamicEntryCutoffMinutes: effectiveEntryCutoff,
      dynamicForceCloseMinutes: effectiveForceClose,
      entryCutoffPT: formatTime(entryCutoffPTHour, entryCutoffMin) + " PT",
      forceClosePT: formatTime(forceClosePTHour, forceCloseMin) + " PT",
      cachedAt: Date.now(),
    };
    
    if (isEarlyClose) {
      console.log(`[TIME GUARD] EARLY CLOSE DETECTED: Market closes at ${cachedClock.nextCloseET} (${nextClosePT})`);
      console.log(`[TIME GUARD] Dynamic entry cutoff: ${formatTime(entryCutoffHour, entryCutoffMin)} ET (${cachedClock.entryCutoffPT})`);
      console.log(`[TIME GUARD] Dynamic force close: ${formatTime(forceCloseHour, forceCloseMin)} ET (${cachedClock.forceClosePT})`);
    }
    
    return cachedClock;
  } catch (error) {
    console.error("[TIME GUARD] Failed to fetch clock, using legacy schedule:", error);
    // Fallback to legacy schedule
    cachedClock = {
      nextClose: null,
      nextCloseET: "16:00 ET",
      isEarlyClose: false,
      dynamicEntryCutoffMinutes: toMinutesSinceMidnight(LEGACY_ENTRY_CUTOFF_HOUR_ET, LEGACY_ENTRY_CUTOFF_MINUTE_ET),
      dynamicForceCloseMinutes: toMinutesSinceMidnight(LEGACY_FORCE_CLOSE_HOUR_ET, LEGACY_FORCE_CLOSE_MINUTE_ET),
      entryCutoffPT: "8:35 AM PT",
      forceClosePT: "12:45 PM PT",
      cachedAt: Date.now(),
    };
    return cachedClock;
  }
}

/**
 * Get cached clock data, refreshing if stale
 */
async function getClockData(): Promise<CachedClockData> {
  if (!cachedClock || Date.now() - cachedClock.cachedAt > CLOCK_CACHE_TTL_MS) {
    return await refreshClockCache();
  }
  return cachedClock;
}

/**
 * Get dynamic cutoffs synchronously (uses cached data)
 * Returns legacy values if cache is empty
 */
function getDynamicCutoffs(): { entryCutoffMinutes: number; forceCloseMinutes: number; isEarlyClose: boolean } {
  if (cachedClock) {
    return {
      entryCutoffMinutes: cachedClock.dynamicEntryCutoffMinutes,
      forceCloseMinutes: cachedClock.dynamicForceCloseMinutes,
      isEarlyClose: cachedClock.isEarlyClose,
    };
  }
  // Fallback to legacy
  return {
    entryCutoffMinutes: toMinutesSinceMidnight(LEGACY_ENTRY_CUTOFF_HOUR_ET, LEGACY_ENTRY_CUTOFF_MINUTE_ET),
    forceCloseMinutes: toMinutesSinceMidnight(LEGACY_FORCE_CLOSE_HOUR_ET, LEGACY_FORCE_CLOSE_MINUTE_ET),
    isEarlyClose: false,
  };
}

/**
 * Log current time status (call every scan cycle)
 */
export function logTimeStatus(): void {
  const et = getEasternTime();
  const currentMinutes = toMinutesSinceMidnight(et.hour, et.minute);
  const ptHour = et.hour - 3;
  const nowPT = `${ptHour > 12 ? ptHour - 12 : ptHour}:${et.minute.toString().padStart(2, "0")} ${ptHour >= 12 ? "PM" : "AM"} PT`;
  
  const cutoffs = getDynamicCutoffs();
  const nextClosePT = cachedClock?.forceClosePT?.replace(" PT", "") || "12:47 PM";
  
  const entryCutoffHour = Math.floor(cutoffs.entryCutoffMinutes / 60);
  const entryCutoffMin = cutoffs.entryCutoffMinutes % 60;
  const forceCloseHour = Math.floor(cutoffs.forceCloseMinutes / 60);
  const forceCloseMin = cutoffs.forceCloseMinutes % 60;
  
  const formatPT = (h: number, m: number): string => {
    const ptH = h - 3;
    const hour12 = ptH > 12 ? ptH - 12 : (ptH === 0 ? 12 : ptH);
    const ampm = ptH >= 12 ? "PM" : "AM";
    return `${hour12}:${m.toString().padStart(2, "0")} ${ampm}`;
  };
  
  const entryCutoffPT = formatPT(entryCutoffHour, entryCutoffMin);
  const hardStopPT = formatPT(forceCloseHour, forceCloseMin);
  const nextCloseDisplay = cachedClock ? cachedClock.nextCloseET.replace(" ET", "") : "4:00 PM";
  const nextClosePTDisplay = formatPT(parseInt(nextCloseDisplay.split(":")[0]), parseInt(nextCloseDisplay.split(":")[1]));
  
  console.log(`[TIME] now_pt=${nowPT} next_close_pt=${nextClosePTDisplay} PT entry_cutoff_pt=${entryCutoffPT} PT hard_stop_close_pt=${hardStopPT} PT${cutoffs.isEarlyClose ? " [EARLY_CLOSE]" : ""}`);
}

function getEasternDateString(): string {
  return getEasternTime().dateString;
}

/**
 * ============================================================
 * STANDARDIZED TIME GUARD FLAGS (DYNAMIC - EARLY CLOSE AWARE)
 * ============================================================
 * canOpenNewTrades   - true only 9:35 AM ET until dynamic entry cutoff
 * canManagePositions - true 9:35 AM ET until dynamic force close
 * shouldForceClose   - true at/after dynamic force close
 * ============================================================
 */

/**
 * Check if we're within the ENTRY window (can open new positions)
 * Entry window: 9:35 AM ET - dynamic cutoff (min of 11:35 AM ET and next_close - 5 min)
 */
export function canOpenNewTrades(): boolean {
  if (TIME_GUARD_OVERRIDE) return true;
  const { hour, minute } = getEasternTime();
  const currentMinutes = toMinutesSinceMidnight(hour, minute);
  
  const entryStartMinutes = toMinutesSinceMidnight(ENTRY_START_HOUR_ET, ENTRY_START_MINUTE_ET);
  const cutoffs = getDynamicCutoffs();
  
  return currentMinutes >= entryStartMinutes && currentMinutes < cutoffs.entryCutoffMinutes;
}

/**
 * Check if we can manage positions (entry window + management window)
 * Management allowed: 9:35 AM ET - dynamic force close
 */
export function canManagePositions(): boolean {
  if (TIME_GUARD_OVERRIDE) return true;
  const { hour, minute } = getEasternTime();
  const currentMinutes = toMinutesSinceMidnight(hour, minute);
  
  const entryStartMinutes = toMinutesSinceMidnight(ENTRY_START_HOUR_ET, ENTRY_START_MINUTE_ET);
  const cutoffs = getDynamicCutoffs();
  
  return currentMinutes >= entryStartMinutes && currentMinutes < cutoffs.forceCloseMinutes;
}

/**
 * Check if force close is required (at or after dynamic force close)
 * Dynamic: min(3:45 PM ET, next_close - 2 minutes)
 */
export function shouldForceClose(): boolean {
  if (TIME_GUARD_OVERRIDE) return false;
  const { hour, minute } = getEasternTime();
  const currentMinutes = toMinutesSinceMidnight(hour, minute);
  
  const cutoffs = getDynamicCutoffs();
  
  return currentMinutes >= cutoffs.forceCloseMinutes;
}

/**
 * Check if we're before the entry window starts (before 9:35 AM ET)
 */
export function isBeforeEntryWindow(): boolean {
  if (TIME_GUARD_OVERRIDE) return false;
  const { hour, minute } = getEasternTime();
  const currentMinutes = toMinutesSinceMidnight(hour, minute);
  
  const entryStartMinutes = toMinutesSinceMidnight(ENTRY_START_HOUR_ET, ENTRY_START_MINUTE_ET);
  
  return currentMinutes < entryStartMinutes;
}

// Legacy aliases for backwards compatibility
export const isWithinEntryWindow = canOpenNewTrades;
export const isWithinManagementWindow = () => canManagePositions() && !canOpenNewTrades();
export const isWithinTradingHours = canManagePositions;
export const isPastForceClose = shouldForceClose;
export const isPastTradingCutoff = shouldForceClose;
export const isBeforeTradingStart = isBeforeEntryWindow;

export function isPastEntryCutoff(): boolean {
  const { hour, minute } = getEasternTime();
  const currentMinutes = toMinutesSinceMidnight(hour, minute);
  
  const cutoffs = getDynamicCutoffs();
  
  return currentMinutes >= cutoffs.entryCutoffMinutes;
}

/**
 * ============================================================
 * TRADING STATUS - Comprehensive status object
 * ============================================================
 */

export type TradingReason = 
  | "MARKET_CLOSED"
  | "BEFORE_ENTRY_WINDOW"
  | "ENTRY_WINDOW_ACTIVE"
  | "AFTER_ENTRY_CUTOFF_MANAGE_ONLY"
  | "FORCE_CLOSE_REQUIRED";

export interface TradingStatus {
  canOpenNewTrades: boolean;
  canManagePositions: boolean;
  shouldForceClose: boolean;
  reason: TradingReason;
  reasonDisplay: string;
  currentTimeET: string;
  entryStartET: string;
  entryCutoffET: string;
  forceCloseET: string;
  forceClosePT: string;
  // Legacy aliases
  canTrade: boolean;
  canEnterNewPositions: boolean;
}

/**
 * Get comprehensive trading status based ONLY on time guard (no market status)
 * This returns what WOULD be allowed if market were open
 * Now uses DYNAMIC cutoffs based on Alpaca next_close for early close days
 */
export function getTimeGuardStatus(): TradingStatus {
  const et = getEasternTime();
  const cutoffs = getDynamicCutoffs();
  
  const currentTimeET = et.displayTime;
  const entryStartET = `${ENTRY_START_HOUR_ET}:${ENTRY_START_MINUTE_ET.toString().padStart(2, "0")} ET`;
  
  // Use dynamic cutoffs for display
  const entryCutoffHour = Math.floor(cutoffs.entryCutoffMinutes / 60);
  const entryCutoffMin = cutoffs.entryCutoffMinutes % 60;
  const forceCloseHour = Math.floor(cutoffs.forceCloseMinutes / 60);
  const forceCloseMin = cutoffs.forceCloseMinutes % 60;
  
  const entryCutoffET = `${entryCutoffHour}:${entryCutoffMin.toString().padStart(2, "0")} ET`;
  const forceCloseET = `${forceCloseHour}:${forceCloseMin.toString().padStart(2, "0")} ET`;
  const forceClosePT = cachedClock?.forceClosePT || "12:45 PM PT";
  
  const _canOpenNew = canOpenNewTrades();
  const _canManage = canManagePositions();
  const _shouldForceClose = shouldForceClose();
  const _beforeEntry = isBeforeEntryWindow();
  
  const earlyCloseTag = cutoffs.isEarlyClose ? " [EARLY CLOSE]" : "";
  
  let reason: TradingReason;
  let reasonDisplay: string;
  
  if (_shouldForceClose) {
    reason = "FORCE_CLOSE_REQUIRED";
    reasonDisplay = `FORCE CLOSE: Past ${forceCloseET} (${forceClosePT}). All positions must be liquidated.${earlyCloseTag}`;
  } else if (_beforeEntry) {
    reason = "BEFORE_ENTRY_WINDOW";
    reasonDisplay = `PRE-MARKET: Entry window starts at ${entryStartET} (6:35 AM PT).`;
  } else if (_canOpenNew) {
    reason = "ENTRY_WINDOW_ACTIVE";
    reasonDisplay = `ENTRY WINDOW: ${entryStartET} - ${entryCutoffET}. New positions allowed.${earlyCloseTag}`;
  } else if (_canManage) {
    reason = "AFTER_ENTRY_CUTOFF_MANAGE_ONLY";
    reasonDisplay = `MANAGE ONLY: Past ${entryCutoffET} entry cutoff. Managing existing positions until ${forceCloseET}.${earlyCloseTag}`;
  } else {
    reason = "FORCE_CLOSE_REQUIRED";
    reasonDisplay = `FORCE CLOSE: Past ${forceCloseET} (${forceClosePT}). All positions must be liquidated.${earlyCloseTag}`;
  }
  
  return {
    canOpenNewTrades: _canOpenNew,
    canManagePositions: _canManage,
    shouldForceClose: _shouldForceClose,
    reason,
    reasonDisplay,
    currentTimeET,
    entryStartET,
    entryCutoffET,
    forceCloseET,
    forceClosePT,
    // Legacy aliases
    canTrade: _canManage,
    canEnterNewPositions: _canOpenNew,
  };
}

/**
 * Legacy function - use getTimeGuardStatus() instead
 */
export function getTradingStatus(): {
  canTrade: boolean;
  canEnterNewPositions: boolean;
  reason: string;
  currentTimeET: string;
  entryStartET: string;
  entryCutoffET: string;
  forceCloseET: string;
  forceClosePT: string;
} {
  const status = getTimeGuardStatus();
  return {
    canTrade: status.canManagePositions,
    canEnterNewPositions: status.canOpenNewTrades,
    reason: status.reasonDisplay,
    currentTimeET: status.currentTimeET,
    entryStartET: status.entryStartET,
    entryCutoffET: status.entryCutoffET,
    forceCloseET: status.forceCloseET,
    forceClosePT: status.forceClosePT,
  };
}

export async function closeAllPositionsNow(reason: string): Promise<{ closed: number; totalPL: number }> {
  console.log(`[TIME GUARD] CLOSING ALL POSITIONS: ${reason}`);
  
  let alpacaPositions: any[] = [];
  try {
    alpacaPositions = await alpaca.getPositions();
  } catch (error) {
    console.error("[TIME GUARD] Failed to fetch positions from Alpaca:", error);
    return { closed: 0, totalPL: 0 };
  }
  
  if (alpacaPositions.length === 0) {
    console.log("[TIME GUARD] No positions to close - already flat");
    return { closed: 0, totalPL: 0 };
  }
  
  console.log(`[TIME GUARD] Found ${alpacaPositions.length} positions to close`);
  
  let totalPL = 0;
  let closedCount = 0;
  
  for (const position of alpacaPositions) {
    const symbol = position.symbol;
    const unrealizedPL = parseFloat(position.unrealized_pl || "0");
    const qty = parseFloat(position.qty || "0");
    
    // Look up tradeId from positionManager for HIGH confidence pairing
    const managedPos = positionManager.getManagedPositions().find(p => p.symbol === symbol);
    const tradeId = (managedPos as any)?.tradeId || undefined;
    
    try {
      console.log(`ACTION=EXIT symbol=${symbol} side=sell reason=time_guard_force_close trade_id=${tradeId || 'UNKNOWN'}`);
      await alpaca.closePosition(symbol, "time_guard_force_close", tradeId);
      closedCount++;
      totalPL += unrealizedPL;
      
      console.log(`[TIME GUARD] Closed ${symbol} (${qty} shares): ${unrealizedPL >= 0 ? "+" : ""}$${unrealizedPL.toFixed(2)} trade_id=${tradeId || 'UNKNOWN'}`);
      
      await storage.createActivityLog({
        type: "trade",
        action: "Time Guard Close",
        description: `FORCED CLOSE: ${qty} ${symbol} at ${unrealizedPL >= 0 ? "+" : ""}$${unrealizedPL.toFixed(2)} - ${reason}`,
      });
    } catch (error) {
      console.error(`[TIME GUARD] FAILED to close ${symbol}:`, error);
      
      await storage.createAlert({
        type: "critical",
        title: `URGENT: Failed to Close ${symbol}`,
        message: `Time guard could not close position. MANUAL INTERVENTION REQUIRED.`,
        requiresApproval: false,
      });
    }
  }
  
  await storage.createAlert({
    type: "warning",
    title: "End of Day: All Positions Closed",
    message: `Closed ${closedCount}/${alpacaPositions.length} positions. Daily P/L: ${totalPL >= 0 ? "+" : ""}$${totalPL.toFixed(2)}. ${reason}`,
    requiresApproval: false,
  });
  
  console.log(`[TIME GUARD] Closed ${closedCount} positions, Total P/L: ${totalPL >= 0 ? "+" : ""}$${totalPL.toFixed(2)}`);
  
  // Print End-of-Day summary after closing all positions
  printEODSummary({
    positionsClosed: closedCount,
  });
  
  return { closed: closedCount, totalPL };
}

async function checkAndEnforceEndOfDay(): Promise<void> {
  const today = getEasternDateString();
  
  if (lastCheckedDate !== today) {
    endOfDayExecuted = false;
    lastCheckedDate = today;
    console.log(`[TIME GUARD] New trading day: ${today}. End-of-day reset.`);
    // Reset risk engine for new day (clears daily counters and rehydrates from today's trades)
    initializeRiskEngine();
    // Refresh clock cache for new day
    await refreshClockCache();
  }
  
  if (endOfDayExecuted) {
    return;
  }
  
  // Refresh clock cache periodically to detect early close
  await getClockData();
  
  if (shouldForceClose()) {
    const cutoffs = getDynamicCutoffs();
    const forceCloseTime = cachedClock?.forceClosePT || "12:45 PM PT";
    const earlyCloseMsg = cutoffs.isEarlyClose ? " [EARLY CLOSE DAY]" : "";
    console.log(`[TIME GUARD] *** FORT KNOX LOCKDOWN *** Force close triggered at ${forceCloseTime}${earlyCloseMsg} - CLOSING ALL POSITIONS`);
    endOfDayExecuted = true;
    await closeAllPositionsNow(`Force close at ${forceCloseTime}${earlyCloseMsg} - LIQUIDATING ALL POSITIONS`);
  }
}

export async function startTimeGuard(): Promise<void> {
  console.log("[TIME GUARD] ============================================");
  console.log("[TIME GUARD] FORT KNOX TRADING SCHEDULE ACTIVATED");
  console.log("[TIME GUARD] Dynamic early close detection enabled");
  console.log("[TIME GUARD] ============================================");
  
  // Initialize clock cache to detect early close days
  await refreshClockCache();
  const cutoffs = getDynamicCutoffs();
  
  const entryCutoffHour = Math.floor(cutoffs.entryCutoffMinutes / 60);
  const entryCutoffMin = cutoffs.entryCutoffMinutes % 60;
  const forceCloseHour = Math.floor(cutoffs.forceCloseMinutes / 60);
  const forceCloseMin = cutoffs.forceCloseMinutes % 60;
  
  console.log(`[TIME GUARD] Entry Window: ${ENTRY_START_HOUR_ET}:${ENTRY_START_MINUTE_ET.toString().padStart(2, "0")} ET - ${entryCutoffHour}:${entryCutoffMin.toString().padStart(2, "0")} ET`);
  console.log(`[TIME GUARD] Force Close:  ${forceCloseHour}:${forceCloseMin.toString().padStart(2, "0")} ET (${cachedClock?.forceClosePT || "12:45 PM PT"})`);
  if (cutoffs.isEarlyClose) {
    console.log(`[TIME GUARD] *** EARLY CLOSE DAY DETECTED *** Market closes at ${cachedClock?.nextCloseET}`);
  }
  console.log("[TIME GUARD] ============================================");
  
  const status = getTimeGuardStatus();
  console.log(`[TIME GUARD] Current: ${status.currentTimeET} | canOpenNewTrades: ${status.canOpenNewTrades} | canManagePositions: ${status.canManagePositions} | shouldForceClose: ${status.shouldForceClose}`);
  console.log(`[TIME GUARD] Reason: ${status.reason} - ${status.reasonDisplay}`);
  
  if (shouldForceClose()) {
    const forceCloseTime = cachedClock?.forceClosePT || "12:45 PM PT";
    console.log(`[TIME GUARD] *** STARTED AFTER FORCE CLOSE - EMERGENCY LIQUIDATION ***`);
    await closeAllPositionsNow(`Bot started after ${forceCloseTime} - EMERGENCY CLOSE`);
    endOfDayExecuted = true;
  }
  
  if (guardInterval) {
    clearInterval(guardInterval);
  }
  // Check every 10 seconds for tighter enforcement
  guardInterval = setInterval(checkAndEnforceEndOfDay, 10 * 1000);
  
  await checkAndEnforceEndOfDay();
  
  console.log("[TIME GUARD] Guard active - checking every 10 seconds");
}

export function stopTimeGuard(): void {
  if (guardInterval) {
    clearInterval(guardInterval);
    guardInterval = null;
  }
  console.log("[TIME GUARD] Time guard stopped");
}

export function resetEndOfDayFlag(): void {
  endOfDayExecuted = false;
  console.log("[TIME GUARD] End of day flag reset");
}
