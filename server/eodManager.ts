/**
 * EOD Flatten Manager - End of Day Position Management
 * 
 * Ensures all positions are flat before market close:
 * - T-12 min: Disable new entries (preclose lockout)
 * - T-10 min: Cancel open BUY orders
 * - T-7 min: Flatten all positions (verified with retry)
 * - T-2 min: Retry if positions still open
 * 
 * Also handles overnight watchdog:
 * - Startup check for overnight positions
 * - Auto-flatten at next market open + 1 minute
 * - Repeat every 60s for 10 minutes until flat
 */

import * as alpaca from "./alpaca";
import { getEasternTime } from "./timezone";
import * as fs from "fs";
import * as path from "path";

export interface EODStatus {
  entryAllowed: boolean;
  precloseTriggered: boolean;
  openBuysCancelTriggered: boolean;
  flattenTriggered: boolean;
  retryTriggered: boolean;
  positionsFlattenedCount: number;
  openBuysCancelledCount: number;
  overnightPositionsDetected: number;
  lastFlattenTs: string | null;
  criticalAlertWritten: boolean;
  entriesBlockedUntilFlat: boolean;
}

let eodStatus: EODStatus = {
  entryAllowed: true,
  precloseTriggered: false,
  openBuysCancelTriggered: false,
  flattenTriggered: false,
  retryTriggered: false,
  positionsFlattenedCount: 0,
  openBuysCancelledCount: 0,
  overnightPositionsDetected: 0,
  lastFlattenTs: null,
  criticalAlertWritten: false,
  entriesBlockedUntilFlat: false,
};

let eodCheckInterval: NodeJS.Timeout | null = null;
let overnightWatchdogInterval: NodeJS.Timeout | null = null;
let overnightCleanupInterval: NodeJS.Timeout | null = null;
let overnightCleanupAttempts = 0;

const EOD_CONFIG = {
  PRECLOSE_LOCKOUT_MINUTES: 12,
  CANCEL_OPEN_BUYS_MINUTES: 10,
  FLATTEN_POSITIONS_MINUTES: 7,
  RETRY_FLATTEN_MINUTES: 2,
  MAX_FLATTEN_RETRIES: 3,
  FLATTEN_VERIFY_DELAY_MS: 3000,
  OVERNIGHT_CLEANUP_INTERVAL_MS: 60000,
  OVERNIGHT_CLEANUP_MAX_ATTEMPTS: 10,
};

export function getEODStatus(): EODStatus {
  return { ...eodStatus };
}

export function isEntryAllowed(): boolean {
  return eodStatus.entryAllowed && !eodStatus.entriesBlockedUntilFlat;
}

export function resetDailyStatus(): void {
  eodStatus = {
    entryAllowed: true,
    precloseTriggered: false,
    openBuysCancelTriggered: false,
    flattenTriggered: false,
    retryTriggered: false,
    positionsFlattenedCount: 0,
    openBuysCancelledCount: 0,
    overnightPositionsDetected: 0,
    lastFlattenTs: null,
    criticalAlertWritten: false,
    entriesBlockedUntilFlat: false,
  };
  overnightCleanupAttempts = 0;
  console.log("[EODManager] Daily status reset");
}

async function getSessionClose(): Promise<Date | null> {
  try {
    const clock = await alpaca.getClock();
    if (!clock.is_open) {
      return null;
    }
    return new Date(clock.next_close);
  } catch (error) {
    console.log("[EODManager] Error getting session close:", error);
    return null;
  }
}

async function getMinutesUntilClose(): Promise<number | null> {
  const closeTime = await getSessionClose();
  if (!closeTime) return null;
  
  const now = new Date();
  const diffMs = closeTime.getTime() - now.getTime();
  return diffMs / (1000 * 60);
}

/**
 * Cancel all open orders for a specific symbol
 * CRITICAL: Must cancel bracket child orders before closing position
 */
async function cancelOpenOrdersForSymbol(symbol: string): Promise<number> {
  const et = getEasternTime();
  console.log(`[EOD] CANCEL_ORDERS symbol=${symbol} ts=${et.displayTime}`);
  
  let cancelCount = 0;
  
  try {
    const orders = await alpaca.getOrders("open", 100);
    const symbolOrders = orders.filter(o => o.symbol === symbol);
    
    for (const order of symbolOrders) {
      try {
        await alpaca.cancelOrder(order.id);
        cancelCount++;
        console.log(`[EOD] Cancelled order_id=${order.id} symbol=${symbol} side=${order.side} type=${order.type}`);
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        console.log(`[EOD] Cancel failed order_id=${order.id}: ${errMsg}`);
      }
    }
    
    console.log(`[EOD] CANCEL_ORDERS_COMPLETE symbol=${symbol} cancelled=${cancelCount}`);
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : String(error);
    console.log(`[EOD] ERROR fetching orders for ${symbol}: ${errMsg}`);
  }
  
  return cancelCount;
}

async function disableEntries(): Promise<void> {
  if (eodStatus.precloseTriggered) return;
  
  eodStatus.entryAllowed = false;
  eodStatus.precloseTriggered = true;
  
  const et = getEasternTime();
  console.log(`[EOD] PRECLOSE_LOCKOUT ts=${et.displayTime} entries_disabled=true`);
}

async function cancelOpenBuys(): Promise<void> {
  if (eodStatus.openBuysCancelTriggered) return;
  
  const et = getEasternTime();
  console.log(`[EOD] CANCEL_OPEN_BUYS_START ts=${et.displayTime}`);
  
  try {
    const orders = await alpaca.getOrders("open", 100);
    const openBuys = orders.filter(o => o.side === "buy");
    
    let cancelCount = 0;
    for (const order of openBuys) {
      try {
        await alpaca.cancelOrder(order.id);
        cancelCount++;
        console.log(`[EOD] Cancelled BUY order_id=${order.id} symbol=${order.symbol}`);
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        console.log(`[EOD] Cancel failed order_id=${order.id}: ${errMsg}`);
      }
    }
    
    eodStatus.openBuysCancelTriggered = true;
    eodStatus.openBuysCancelledCount = cancelCount;
    
    console.log(`[EOD] CANCEL_OPEN_BUYS_COMPLETE count=${cancelCount} ts=${et.displayTime}`);
  } catch (error) {
    console.log("[EOD] ERROR cancelling open buys:", error);
  }
}

/**
 * Verified flatten: cancel orders → close position → verify → retry
 * Returns true if successfully flat, false if position remains
 */
async function verifiedFlattenSymbol(symbol: string, qty: number, reason: string): Promise<boolean> {
  const et = getEasternTime();
  
  for (let attempt = 1; attempt <= EOD_CONFIG.MAX_FLATTEN_RETRIES; attempt++) {
    console.log(`[EOD] CLOSE_ATTEMPT symbol=${symbol} attempt=${attempt}/${EOD_CONFIG.MAX_FLATTEN_RETRIES} reason=${reason} ts=${et.displayTime}`);
    
    // Step 1: Cancel all open orders for this symbol first
    await cancelOpenOrdersForSymbol(symbol);
    
    // Small delay to let cancels process
    await new Promise(r => setTimeout(r, 500));
    
    // Step 2: Submit close position order
    try {
      const result = await alpaca.closePosition(symbol, reason);
      console.log(`[EOD] CLOSE_SUBMITTED symbol=${symbol} order_id=${result.id} ts=${et.displayTime}`);
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      console.log(`[EOD] CLOSE_SUBMIT_ERROR symbol=${symbol} error="${errMsg}" attempt=${attempt}`);
      
      // If position doesn't exist, it's already flat
      if (errMsg.includes("position does not exist") || errMsg.includes("40410000")) {
        console.log(`[EOD] CLOSE_ALREADY_FLAT symbol=${symbol}`);
        return true;
      }
      
      // Try aggressive limit as fallback
      try {
        const quote = await alpaca.getExtendedQuote(symbol);
        const aggressivePrice = quote.bid * 0.99; // Sell below bid
        const result = await alpaca.submitOrder(symbol, qty, "sell", "limit", aggressivePrice);
        console.log(`[EOD] CLOSE_LIMIT_FALLBACK symbol=${symbol} order_id=${result.id} price=${aggressivePrice} ts=${et.displayTime}`);
      } catch (limitErr) {
        const limitErrMsg = limitErr instanceof Error ? limitErr.message : String(limitErr);
        console.log(`[EOD] CLOSE_LIMIT_FALLBACK_ERROR symbol=${symbol} error="${limitErrMsg}"`);
      }
    }
    
    // Step 3: Wait for order to process
    await new Promise(r => setTimeout(r, EOD_CONFIG.FLATTEN_VERIFY_DELAY_MS));
    
    // Step 4: Verify position is closed
    try {
      const positions = await alpaca.getPositions();
      const stillOpen = positions.find(p => p.symbol === symbol);
      
      if (!stillOpen) {
        console.log(`[EOD] CLOSE_VERIFIED symbol=${symbol} attempt=${attempt}`);
        return true;
      }
      
      console.log(`[EOD] CLOSE_NOT_VERIFIED symbol=${symbol} still_qty=${stillOpen.qty} attempt=${attempt}`);
    } catch (posErr) {
      // If we can't get positions, assume still open
      console.log(`[EOD] POSITION_CHECK_ERROR: ${posErr}`);
    }
    
    // Wait before retry
    if (attempt < EOD_CONFIG.MAX_FLATTEN_RETRIES) {
      await new Promise(r => setTimeout(r, 2000));
    }
  }
  
  return false;
}

/**
 * Write CRITICAL alert file when positions remain after EOD flatten
 * Only writes once per session to prevent spam
 */
function writeCriticalAlert(remainingSymbols: string[]): void {
  // Guard: Only write alert once per session
  if (eodStatus.criticalAlertWritten) {
    console.log("[EOD] CRITICAL_ALERT_SKIPPED reason=already_written_this_session");
    return;
  }
  
  const et = getEasternTime();
  const alertDir = path.join(process.cwd(), "reports", "alerts");
  
  try {
    if (!fs.existsSync(alertDir)) {
      fs.mkdirSync(alertDir, { recursive: true });
    }
    
    const alertFile = path.join(alertDir, `overnight_risk_${et.dateString}.txt`);
    
    const content = [
      "========================================",
      "CRITICAL: OVERNIGHT POSITION RISK",
      "========================================",
      `Timestamp: ${et.displayTime} ET`,
      `Date: ${et.dateString}`,
      "",
      "Positions remaining after EOD flatten attempts:",
      ...remainingSymbols.map(s => `  - ${s}`),
      "",
      "ACTION REQUIRED:",
      "  1. Check Alpaca dashboard for open positions",
      "  2. Manually close positions if needed",
      "  3. Entries blocked until positions are flat",
      "",
      "This file must be reviewed. Bot entries blocked until flat.",
      "========================================",
    ].join("\n");
    
    fs.writeFileSync(alertFile, content);
    console.log(`[EOD] CRITICAL_ALERT_WRITTEN file=${alertFile}`);
    
    eodStatus.criticalAlertWritten = true;
  } catch (error) {
    console.log(`[EOD] ERROR writing critical alert: ${error}`);
  }
}

/**
 * Check for entry block file and clear if positions are flat
 */
async function checkAndClearEntryBlock(): Promise<void> {
  if (!eodStatus.entriesBlockedUntilFlat) return;
  
  try {
    const positions = await alpaca.getPositions();
    if (positions.length === 0) {
      eodStatus.entriesBlockedUntilFlat = false;
      eodStatus.entryAllowed = true;
      console.log("[EOD] Entry block cleared - positions are now flat");
    }
  } catch {
    // Keep blocked if we can't verify
  }
}

/**
 * Flatten all positions with verification
 * Returns number of positions that could NOT be closed
 */
async function flattenPositions(reason: string = "EOD_FLATTEN"): Promise<string[]> {
  const et = getEasternTime();
  console.log(`[EOD] FLATTEN_START reason=${reason} ts=${et.displayTime}`);
  
  try {
    const positions = await alpaca.getPositions();
    if (positions.length === 0) {
      console.log("[EOD] FLATTEN_OK no_positions=true");
      return [];
    }
    
    const remainingSymbols: string[] = [];
    
    for (const pos of positions) {
      const symbol = pos.symbol;
      const qty = Math.abs(parseInt(pos.qty));
      
      const closed = await verifiedFlattenSymbol(symbol, qty, reason);
      
      if (closed) {
        eodStatus.positionsFlattenedCount++;
      } else {
        remainingSymbols.push(symbol);
      }
    }
    
    eodStatus.lastFlattenTs = new Date().toISOString();
    
    if (remainingSymbols.length === 0) {
      console.log(`[EOD] FLATTEN_OK flattened=${eodStatus.positionsFlattenedCount} ts=${et.displayTime}`);
    } else {
      console.log(`[EOD] FLATTEN_CRITICAL positionsRemaining=[${remainingSymbols.join(",")}]`);
      writeCriticalAlert(remainingSymbols);
      eodStatus.entriesBlockedUntilFlat = true;
    }
    
    return remainingSymbols;
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : String(error);
    console.log(`[EOD] FLATTEN_ERROR error="${errMsg}"`);
    return [];
  }
}

async function eodCheck(): Promise<void> {
  const minutesUntilClose = await getMinutesUntilClose();
  if (minutesUntilClose === null) {
    // Market not open
    return;
  }
  
  // T-12: Preclose lockout
  if (minutesUntilClose <= EOD_CONFIG.PRECLOSE_LOCKOUT_MINUTES && !eodStatus.precloseTriggered) {
    await disableEntries();
  }
  
  // T-10: Cancel open buys
  if (minutesUntilClose <= EOD_CONFIG.CANCEL_OPEN_BUYS_MINUTES && !eodStatus.openBuysCancelTriggered) {
    await cancelOpenBuys();
  }
  
  // T-7: Flatten positions
  if (minutesUntilClose <= EOD_CONFIG.FLATTEN_POSITIONS_MINUTES && !eodStatus.flattenTriggered) {
    eodStatus.flattenTriggered = true;
    await flattenPositions("EOD_FLATTEN");
  }
  
  // T-2: Retry flatten if still have positions
  if (minutesUntilClose <= EOD_CONFIG.RETRY_FLATTEN_MINUTES && !eodStatus.retryTriggered) {
    const positions = await alpaca.getPositions();
    if (positions.length > 0) {
      eodStatus.retryTriggered = true;
      const symbols = positions.map(p => p.symbol);
      console.log(`[EOD] FLATTEN_RETRY_TRIGGERED positions=[${symbols.join(",")}]`);
      await flattenPositions("EOD_FLATTEN_RETRY");
    }
  }
}

/**
 * Overnight watchdog cleanup - runs every 60s for first 10 minutes after open
 */
async function overnightCleanupLoop(): Promise<void> {
  overnightCleanupAttempts++;
  const et = getEasternTime();
  
  console.log(`[OVERNIGHT] CLEANUP_ATTEMPT attempt=${overnightCleanupAttempts}/${EOD_CONFIG.OVERNIGHT_CLEANUP_MAX_ATTEMPTS} ts=${et.displayTime}`);
  
  try {
    const positions = await alpaca.getPositions();
    
    if (positions.length === 0) {
      console.log("[OVERNIGHT] CLEANUP_COMPLETE positions_flat=true");
      
      // Stop cleanup loop
      if (overnightCleanupInterval) {
        clearInterval(overnightCleanupInterval);
        overnightCleanupInterval = null;
      }
      
      // Re-enable entries
      eodStatus.overnightPositionsDetected = 0;
      eodStatus.entriesBlockedUntilFlat = false;
      eodStatus.entryAllowed = true;
      return;
    }
    
    const symbols = positions.map(p => p.symbol);
    console.log(`[OVERNIGHT] POSITIONS_REMAIN symbols=[${symbols.join(",")}]`);
    
    // Flatten each position
    for (const pos of positions) {
      const symbol = pos.symbol;
      const qty = Math.abs(parseInt(pos.qty));
      
      const closed = await verifiedFlattenSymbol(symbol, qty, "OVERNIGHT_FLATTEN");
      
      if (closed) {
        console.log(`[OVERNIGHT] FLATTEN_OK symbol=${symbol}`);
      } else {
        console.log(`[OVERNIGHT] FLATTEN_FAILED symbol=${symbol}`);
      }
    }
    
    // Check if we've exhausted attempts
    if (overnightCleanupAttempts >= EOD_CONFIG.OVERNIGHT_CLEANUP_MAX_ATTEMPTS) {
      console.log("[OVERNIGHT] CLEANUP_EXHAUSTED max_attempts_reached=true");
      
      if (overnightCleanupInterval) {
        clearInterval(overnightCleanupInterval);
        overnightCleanupInterval = null;
      }
      
      // Check final state
      const finalPositions = await alpaca.getPositions();
      if (finalPositions.length > 0) {
        const remainingSymbols = finalPositions.map(p => p.symbol);
        console.log(`[OVERNIGHT] CRITICAL positionsRemaining=[${remainingSymbols.join(",")}]`);
        writeCriticalAlert(remainingSymbols);
      }
    }
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : String(error);
    console.log(`[OVERNIGHT] CLEANUP_ERROR error="${errMsg}"`);
  }
}

async function overnightWatchdogCheck(): Promise<void> {
  try {
    const clock = await alpaca.getClock();
    const positions = await alpaca.getPositions();
    
    // Check if we have positions while market is closed
    if (!clock.is_open && positions.length > 0) {
      const symbols = positions.map(p => p.symbol);
      eodStatus.overnightPositionsDetected = positions.length;
      eodStatus.entryAllowed = false;
      eodStatus.entriesBlockedUntilFlat = true;
      
      console.log(`[OVERNIGHT] DETECTED positions=[${symbols.join(",")}] count=${positions.length}`);
    }
    
    // Check if market just opened and we had overnight positions
    if (clock.is_open && eodStatus.overnightPositionsDetected > 0 && !overnightCleanupInterval) {
      const now = new Date();
      const nextOpen = new Date(clock.next_open);
      
      // next_open is in the past when market is open (it's today's open time)
      // We need to check if we're within first minute of open
      const openTime = nextOpen.getTime() < now.getTime() 
        ? nextOpen.getTime() 
        : new Date(clock.next_close).getTime() - (6.5 * 60 * 60 * 1000); // Estimate open from close
      
      const minutesSinceOpen = (now.getTime() - openTime) / (1000 * 60);
      
      // Only start cleanup within first 11 minutes of open
      if (minutesSinceOpen >= 0 && minutesSinceOpen <= 11) {
        console.log(`[OVERNIGHT] CLEANUP_START minutesSinceOpen=${minutesSinceOpen.toFixed(1)}`);
        
        // Wait 1 minute after open, then start cleanup loop
        if (minutesSinceOpen >= 1) {
          overnightCleanupAttempts = 0;
          
          // Run first cleanup immediately
          await overnightCleanupLoop();
          
          // Then run every 60s
          overnightCleanupInterval = setInterval(() => {
            overnightCleanupLoop();
          }, EOD_CONFIG.OVERNIGHT_CLEANUP_INTERVAL_MS);
        }
      }
    }
    
    // Also check if we can clear entry block
    await checkAndClearEntryBlock();
    
  } catch (error) {
    console.log("[OVERNIGHT] WATCHDOG_ERROR:", error);
  }
}

export function startEODManager(): void {
  if (eodCheckInterval) return;
  
  console.log("[EODManager] Starting EOD manager (checks every 30s)");
  
  // Initial checks
  overnightWatchdogCheck();
  
  // EOD check every 30 seconds during market hours
  eodCheckInterval = setInterval(() => {
    eodCheck();
  }, 30000);
  
  // Overnight watchdog every 5 minutes
  overnightWatchdogInterval = setInterval(() => {
    overnightWatchdogCheck();
  }, 5 * 60 * 1000);
}

export function stopEODManager(): void {
  if (eodCheckInterval) {
    clearInterval(eodCheckInterval);
    eodCheckInterval = null;
  }
  if (overnightWatchdogInterval) {
    clearInterval(overnightWatchdogInterval);
    overnightWatchdogInterval = null;
  }
  if (overnightCleanupInterval) {
    clearInterval(overnightCleanupInterval);
    overnightCleanupInterval = null;
  }
  console.log("[EODManager] Stopped EOD manager");
}

// For validation: Force EOD check with simulated time
export async function forceEODCheck(): Promise<void> {
  await eodCheck();
}

// For validation: Force flatten all positions
export async function forceFlattenAll(reason: string = "FORCE_FLATTEN"): Promise<string[]> {
  return await flattenPositions(reason);
}

// For validation: Force overnight cleanup
export async function forceOvernightCleanup(): Promise<void> {
  await overnightCleanupLoop();
}

// Simulate session close time for testing
let simulatedCloseTime: Date | null = null;

export function setSimulatedCloseTime(closeTime: Date | null): void {
  simulatedCloseTime = closeTime;
}

export async function getMinutesUntilCloseForTest(): Promise<number | null> {
  if (simulatedCloseTime) {
    const now = new Date();
    const diffMs = simulatedCloseTime.getTime() - now.getTime();
    return diffMs / (1000 * 60);
  }
  return await getMinutesUntilClose();
}
