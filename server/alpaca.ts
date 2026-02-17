// Alpaca Trading API integration
// Connects to Alpaca for market data and trade execution
import * as autoTestReporter from "./autoTestReporter";
import { getEasternTime } from "./timezone";

// SIM_CLOCK_OPEN override for testing
const SIM_CLOCK_OPEN = process.env.SIM_CLOCK_OPEN;
if (SIM_CLOCK_OPEN !== undefined) {
  console.log(`[SIM] SIM_CLOCK_OPEN active: ${SIM_CLOCK_OPEN}`);
}

// SIM_CLOCK_FAIL: Simulate clock API failure for testing fail-closed behavior
const SIM_CLOCK_FAIL = process.env.SIM_CLOCK_FAIL === "1";
if (SIM_CLOCK_FAIL) {
  console.log("[SIM] SIM_CLOCK_FAIL active - clock API will simulate failure");
}

// DRY_RUN mode - no real orders placed
const DRY_RUN_ENV = process.env.DRY_RUN === "1";
if (DRY_RUN_ENV) {
  console.log("[SIM] DRY_RUN active - orders will be logged but not executed");
}

// Auto-test mode - force DRY_RUN during 9:30-9:40 ET
const AUTO_TEST_MODE = process.env.AUTO_TEST_MODE === "1";
if (AUTO_TEST_MODE) {
  console.log("[AUTO_TEST] AUTO_TEST_MODE active - auto-DRY_RUN during 9:30-9:40 ET");
}

// Track market open state for order functions
let cachedMarketOpen = false;
let lastClockCheck = 0;
const CLOCK_CACHE_TTL_MS = 5000; // 5 second cache

// Clock snapshot logging - once per hour
let lastClockSnapshotLog = 0;
const CLOCK_SNAPSHOT_INTERVAL_MS = 60 * 60 * 1000; // 1 hour

// Market-closed log cooldown - 60s per symbol+action
const guardLogCooldown: Map<string, number> = new Map();
const GUARD_LOG_COOLDOWN_MS = 60 * 1000; // 60 seconds

export function setMarketOpenState(isOpen: boolean): void {
  cachedMarketOpen = isOpen;
}

/**
 * Log clock snapshot once per session/hour
 */
function logClockSnapshot(clock: { timestamp: string; is_open: boolean; next_open: string; next_close: string }): void {
  const now = Date.now();
  if (now - lastClockSnapshotLog >= CLOCK_SNAPSHOT_INTERVAL_MS) {
    console.log(`[CLOCK] is_open=${clock.is_open} ts=${clock.timestamp} next_open=${clock.next_open} next_close=${clock.next_close}`);
    lastClockSnapshotLog = now;
  }
}

/**
 * Check if guard log should be suppressed (cooldown active)
 */
function shouldSuppressGuardLog(action: string, symbol: string): boolean {
  const key = `${action}:${symbol}`;
  const now = Date.now();
  const lastLogged = guardLogCooldown.get(key) || 0;
  
  if (now - lastLogged < GUARD_LOG_COOLDOWN_MS) {
    return true; // Suppress
  }
  
  guardLogCooldown.set(key, now);
  return false; // Allow full log
}

/**
 * BASELINE MODE: Trading window guard (09:35-11:35 ET)
 * Blocks orders outside the regular trading window for cleaner baseline data
 * ALIGNED with tradingTimeGuard.ts ENTRY_CUTOFF (11:35 ET)
 */
const TRADING_WINDOW_START_HOUR = 9;
const TRADING_WINDOW_START_MIN = 35;
const TRADING_WINDOW_END_HOUR = 11;
const TRADING_WINDOW_END_MIN = 35;

function isWithinTradingWindow(): { allowed: boolean; reason: string } {
  const et = getEasternTime();
  const currentMinutes = et.hour * 60 + et.minute;
  const windowStart = TRADING_WINDOW_START_HOUR * 60 + TRADING_WINDOW_START_MIN; // 9:35 = 575
  const windowEnd = TRADING_WINDOW_END_HOUR * 60 + TRADING_WINDOW_END_MIN; // 11:35 = 695
  
  if (currentMinutes < windowStart) {
    return { allowed: false, reason: "OUTSIDE_TRADING_WINDOW_BEFORE" };
  }
  if (currentMinutes >= windowEnd) {
    return { allowed: false, reason: "OUTSIDE_TRADING_WINDOW_AFTER" };
  }
  return { allowed: true, reason: "WITHIN_TRADING_WINDOW" };
}

/**
 * Market-open gate for order submission
 * Returns { allowed: boolean, reason: string }
 * This check runs IMMEDIATELY before any order submission
 */
// Cached clock data for UI display
let cachedClockData: { next_open: string | null; next_close: string | null } = {
  next_open: null,
  next_close: null,
};

export async function canSubmitOrderNow(): Promise<{ allowed: boolean; reason: string }> {
  // SIM_CLOCK_FAIL: Simulate clock API failure
  if (SIM_CLOCK_FAIL) {
    return { allowed: false, reason: "CLOCK_API_FAIL_ORDER_BLOCKED" };
  }
  
  try {
    const now = Date.now();
    
    // Use cached state if recent enough
    if (now - lastClockCheck < CLOCK_CACHE_TTL_MS) {
      if (!cachedMarketOpen) {
        return { allowed: false, reason: "MARKET_CLOSED_ORDER_BLOCKED" };
      }
      return { allowed: true, reason: "MARKET_OPEN" };
    }
    
    // Fetch fresh clock state
    const clock = await getClock();
    lastClockCheck = now;
    cachedMarketOpen = clock.is_open;
    cachedClockData = {
      next_open: clock.next_open || null,
      next_close: clock.next_close || null,
    };
    
    // Log clock snapshot once per hour
    logClockSnapshot(clock);
    
    if (!clock.is_open) {
      return { allowed: false, reason: "MARKET_CLOSED_ORDER_BLOCKED" };
    }
    
    return { allowed: true, reason: "MARKET_OPEN" };
  } catch (error) {
    // Fail closed - if we can't check clock, block orders
    console.error("[GUARD] Clock API failed - blocking orders for safety:", error);
    return { allowed: false, reason: "CLOCK_API_FAIL_ORDER_BLOCKED" };
  }
}

/**
 * Get current market status using the same cache as canSubmitOrderNow
 * Returns status consistent with order submission guard
 */
export async function getMarketStatusCached(): Promise<{
  is_open: boolean;
  timestamp: string;
  next_open: string | null;
  next_close: string | null;
  source: "alpaca_clock";
  simulated: boolean;
}> {
  const timestamp = new Date().toISOString();
  const simulated = !!(SIM_CLOCK_OPEN || SIM_CLOCK_FAIL);
  
  // SIM_CLOCK_FAIL: Report closed
  if (SIM_CLOCK_FAIL) {
    return {
      is_open: false,
      timestamp,
      next_open: null,
      next_close: null,
      source: "alpaca_clock",
      simulated: true,
    };
  }
  
  // Ensure cache is fresh by calling canSubmitOrderNow (which updates the cache)
  await canSubmitOrderNow();
  
  return {
    is_open: cachedMarketOpen,
    timestamp,
    next_open: cachedClockData.next_open,
    next_close: cachedClockData.next_close,
    source: "alpaca_clock",
    simulated,
  };
}

/**
 * Check if DRY_RUN is effectively active (uses autoTestReporter for consistency)
 */
function isDryRunEffective(): boolean {
  return autoTestReporter.isDryRunEffective(cachedMarketOpen);
}

interface AlpacaAccount {
  id: string;
  account_number: string;
  status: string;
  currency: string;
  buying_power: string;
  cash: string;
  portfolio_value: string;
  pattern_day_trader: boolean;
  trading_blocked: boolean;
  transfers_blocked: boolean;
  account_blocked: boolean;
  daytrade_count: number;
  equity: string;
  last_equity: string;
}

interface AlpacaPosition {
  asset_id: string;
  symbol: string;
  qty: string;
  avg_entry_price: string;
  market_value: string;
  current_price: string;
  unrealized_pl: string;
  unrealized_plpc: string;
  side: string;
}

interface AlpacaOrder {
  id: string;
  client_order_id: string;
  created_at: string;
  updated_at: string;
  submitted_at: string;
  filled_at: string | null;
  symbol: string;
  qty: string;
  filled_qty: string;
  side: string;
  type: string;
  status: string;
  filled_avg_price: string | null;
}

interface AlpacaClock {
  timestamp: string;
  is_open: boolean;
  next_open: string;
  next_close: string;
}

interface AlpacaBar {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

const ALPACA_API_KEY = process.env.ALPACA_API_KEY;
const ALPACA_API_SECRET = process.env.ALPACA_API_SECRET || process.env.ALPACA_SECRET_KEY;

// Multi-user credential switching
let activeCredentials: { key: string; secret: string } | null = null;

export function setActiveCredentials(key: string, secret: string): void {
  activeCredentials = { key, secret };
}

export function clearActiveCredentials(): void {
  activeCredentials = null;
}

function getEffectiveKey(): string {
  return activeCredentials?.key || ALPACA_API_KEY || "";
}

function getEffectiveSecret(): string {
  return activeCredentials?.secret || ALPACA_API_SECRET || "";
}

// Debug: Log if keys are present (not the actual values)
console.log(`[Alpaca] API Key present: ${!!ALPACA_API_KEY}, length: ${ALPACA_API_KEY?.length || 0}`);
console.log(`[Alpaca] API Secret present: ${!!ALPACA_API_SECRET}, length: ${ALPACA_API_SECRET?.length || 0}`);

// Use paper trading endpoint by default for safety
const ALPACA_BASE_URL = "https://paper-api.alpaca.markets";
const ALPACA_DATA_URL = "https://data.alpaca.markets";

// ALPACA-CONNECTIVITY-PROOF-1: Request timeout and retry configuration
const ALPACA_REQUEST_TIMEOUT_MS = 8000; // 8 second timeout
const ALPACA_MAX_RETRIES = 2;

// ALPACA-CONNECTIVITY-PROOF-1: Alpaca connectivity status tracking
interface AlpacaConnectivityState {
  lastOkUTC: string | null;
  consecutiveFailures: number;
  lastError: string | null;
  degraded: boolean;
  lastClockPingUTC: string | null;
  lastAccountPingUTC: string | null;
}

const alpacaConnectivityState: AlpacaConnectivityState = {
  lastOkUTC: null,
  consecutiveFailures: 0,
  lastError: null,
  degraded: false,
  lastClockPingUTC: null,
  lastAccountPingUTC: null,
};

export function getAlpacaConnectivityState(): AlpacaConnectivityState {
  return { ...alpacaConnectivityState };
}

function recordAlpacaSuccess(): void {
  alpacaConnectivityState.lastOkUTC = new Date().toISOString();
  alpacaConnectivityState.consecutiveFailures = 0;
  alpacaConnectivityState.lastError = null;
  alpacaConnectivityState.degraded = false;
}

function recordAlpacaFailure(error: string): void {
  alpacaConnectivityState.consecutiveFailures++;
  alpacaConnectivityState.lastError = error.substring(0, 100);
  if (alpacaConnectivityState.consecutiveFailures >= 3) {
    alpacaConnectivityState.degraded = true;
  }
}

function isRetryableError(status: number | null, isTimeout: boolean, isNetworkError: boolean): boolean {
  if (isTimeout || isNetworkError) return true;
  if (status === 429) return true; // Rate limit
  if (status && status >= 500 && status < 600) return true; // 5xx errors
  return false;
}

async function alpacaRequest<T>(
  endpoint: string,
  options: RequestInit = {},
  isDataApi = false
): Promise<T> {
  const baseUrl = isDataApi ? ALPACA_DATA_URL : ALPACA_BASE_URL;
  const url = `${baseUrl}${endpoint}`;
  
  // Debug: Log the URL for market data requests
  if (isDataApi && endpoint.includes('/bars?')) {
    console.log(`[Alpaca] Data API request: ${url.substring(0, 150)}...`);
  }

  let lastError: Error | null = null;
  
  for (let attempt = 0; attempt <= ALPACA_MAX_RETRIES; attempt++) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), ALPACA_REQUEST_TIMEOUT_MS);
    
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          "APCA-API-KEY-ID": getEffectiveKey(),
          "APCA-API-SECRET-KEY": getEffectiveSecret(),
          "Content-Type": "application/json",
          ...options.headers,
        },
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        const errorCode = `HTTP_${response.status}`;
        const shortEndpoint = endpoint.split("?")[0];
        console.log(`[Alpaca] FAIL code=${errorCode} endpoint=${shortEndpoint} status=${response.status}`);
        
        // Check if retryable
        if (isRetryableError(response.status, false, false) && attempt < ALPACA_MAX_RETRIES) {
          const backoffMs = Math.min(1000 * Math.pow(2, attempt), 4000);
          await new Promise(resolve => setTimeout(resolve, backoffMs));
          continue;
        }
        
        recordAlpacaFailure(`${errorCode}: ${errorText.substring(0, 50)}`);
        throw new Error(`Alpaca API error: ${response.status} - ${errorText}`);
      }

      recordAlpacaSuccess();
      return response.json();
      
    } catch (err: any) {
      clearTimeout(timeoutId);
      
      const isTimeout = err.name === "AbortError";
      const isNetworkError = err.name === "TypeError" || err.message?.includes("fetch");
      const shortEndpoint = endpoint.split("?")[0];
      
      if (isTimeout) {
        console.log(`[Alpaca] FAIL code=ALPACA_TIMEOUT endpoint=${shortEndpoint} attempt=${attempt + 1}/${ALPACA_MAX_RETRIES + 1}`);
        lastError = new Error(`ALPACA_TIMEOUT: Request to ${shortEndpoint} timed out after ${ALPACA_REQUEST_TIMEOUT_MS}ms`);
      } else if (isNetworkError) {
        console.log(`[Alpaca] FAIL code=NETWORK_ERROR endpoint=${shortEndpoint} attempt=${attempt + 1}/${ALPACA_MAX_RETRIES + 1}`);
        lastError = new Error(`NETWORK_ERROR: ${err.message}`);
      } else {
        console.log(`[Alpaca] FAIL code=UNKNOWN_ERROR endpoint=${shortEndpoint} error=${err.message?.substring(0, 50)}`);
        lastError = err;
      }
      
      // Retry for timeout/network errors
      if ((isTimeout || isNetworkError) && attempt < ALPACA_MAX_RETRIES) {
        const backoffMs = Math.min(1000 * Math.pow(2, attempt), 4000);
        await new Promise(resolve => setTimeout(resolve, backoffMs));
        continue;
      }
      
      recordAlpacaFailure(lastError?.message || "Unknown error");
      throw lastError || new Error("Alpaca request failed");
    }
  }
  
  // Should not reach here, but just in case
  recordAlpacaFailure(lastError?.message || "Unknown error after retries");
  throw lastError || new Error("Alpaca request failed after retries");
}

export async function getAccount(): Promise<AlpacaAccount> {
  return alpacaRequest<AlpacaAccount>("/v2/account");
}

export async function getPositions(): Promise<AlpacaPosition[]> {
  return alpacaRequest<AlpacaPosition[]>("/v2/positions");
}

export async function getPosition(symbol: string): Promise<AlpacaPosition | null> {
  try {
    return await alpacaRequest<AlpacaPosition>(`/v2/positions/${symbol}`);
  } catch {
    return null;
  }
}

// Calendar entry - represents a single trading day
export interface AlpacaCalendarDay {
  date: string;  // YYYY-MM-DD
  open: string;  // HH:MM (Eastern)
  close: string; // HH:MM (Eastern)
}

/**
 * Get market calendar for a date range
 * Returns only trading days (excludes weekends and holidays)
 */
export async function getCalendar(start: string, end: string): Promise<AlpacaCalendarDay[]> {
  return alpacaRequest<AlpacaCalendarDay[]>(`/v2/calendar?start=${start}&end=${end}`);
}

/**
 * Check if a specific date was a trading day
 * Returns the calendar entry if trading, null if market was closed
 */
export async function getCalendarDay(date: string): Promise<AlpacaCalendarDay | null> {
  const calendar = await getCalendar(date, date);
  return calendar.length > 0 ? calendar[0] : null;
}

export async function getClock(): Promise<AlpacaClock> {
  // SIM_CLOCK_OPEN override for testing
  if (SIM_CLOCK_OPEN === "1") {
    const now = new Date();
    const nextClose = new Date(now.getTime() + 6 * 60 * 60 * 1000); // 6 hours from now
    const nextOpen = new Date(now.getTime() + 24 * 60 * 60 * 1000); // tomorrow
    return {
      timestamp: now.toISOString(),
      is_open: true,
      next_open: nextOpen.toISOString(),
      next_close: nextClose.toISOString(),
    };
  }
  if (SIM_CLOCK_OPEN === "0") {
    const now = new Date();
    const nextOpen = new Date(now.getTime() + 12 * 60 * 60 * 1000); // 12 hours from now
    const nextClose = new Date(now.getTime() + 18 * 60 * 60 * 1000); // 18 hours from now
    return {
      timestamp: now.toISOString(),
      is_open: false,
      next_open: nextOpen.toISOString(),
      next_close: nextClose.toISOString(),
    };
  }
  return alpacaRequest<AlpacaClock>("/v2/clock");
}

export async function getOrders(status = "all", limit = 50): Promise<AlpacaOrder[]> {
  return alpacaRequest<AlpacaOrder[]>(`/v2/orders?status=${status}&limit=${limit}`);
}

export async function getOrder(orderId: string): Promise<AlpacaOrder> {
  return alpacaRequest<AlpacaOrder>(`/v2/orders/${orderId}`);
}

export async function cancelOrder(orderId: string): Promise<void> {
  await alpacaRequest<void>(`/v2/orders/${orderId}`, { method: "DELETE" });
}

export async function getOrdersWithDateRange(
  after: Date,
  until: Date,
  status = "all",
  limit = 500
): Promise<AlpacaOrder[]> {
  const afterISO = after.toISOString();
  const untilISO = until.toISOString();
  console.log(`[Alpaca] Fetching orders from ${afterISO} to ${untilISO}`);
  return alpacaRequest<AlpacaOrder[]>(
    `/v2/orders?status=${status}&limit=${limit}&after=${afterISO}&until=${untilISO}&direction=asc`
  );
}

export async function submitOrder(
  symbol: string,
  qty: number,
  side: "buy" | "sell",
  type: "market" | "limit" = "market",
  limitPrice?: number,
  reason?: string,
  clientOrderId?: string
): Promise<AlpacaOrder> {
  const tradeIdLog = clientOrderId ? ` trade_id=${clientOrderId}` : "";
  
  // MARKET-OPEN GATE: Block orders when market is closed
  const gate = await canSubmitOrderNow();
  if (!gate.allowed) {
    // Log with cooldown to prevent spam - only log first occurrence per 60s
    if (!shouldSuppressGuardLog("submit", symbol)) {
      console.log(`[GUARD] MARKET CLOSED — blocked order submit (reason=${gate.reason}, symbol=${symbol}, side=${side}, qty=${qty}${tradeIdLog})`);
    }
    throw new Error(`Order blocked: ${gate.reason}`);
  }
  
  // TRADING WINDOW GATE: Block ENTRY orders (buy) outside 09:35-11:35 ET
  // Exit orders (sell) are allowed at all times for position management and FORT KNOX force-close
  if (side === "buy") {
    const windowCheck = isWithinTradingWindow();
    if (!windowCheck.allowed) {
      if (!shouldSuppressGuardLog("window-submit", symbol)) {
        console.log(`[GUARD] OUTSIDE_TRADING_WINDOW — blocked order submit (reason=${windowCheck.reason}, symbol=${symbol}, side=${side}, qty=${qty}${tradeIdLog})`);
      }
      throw new Error(`Order blocked: ${windowCheck.reason}`);
    }
  }
  
  const body: Record<string, unknown> = {
    symbol,
    qty: qty.toString(),
    side,
    type,
    time_in_force: "day",
  };

  if (type === "limit" && limitPrice) {
    body.limit_price = limitPrice.toString();
  }

  if (clientOrderId) {
    body.client_order_id = clientOrderId;
  }

  // DRY_RUN mode - log intent but don't execute
  if (isDryRunEffective()) {
    console.log(`[DRY_RUN] WOULD PLACE ORDER: symbol=${symbol}, side=${side}, qty=${qty}, type=${type}, limit=${limitPrice || 'N/A'}${tradeIdLog}`);
    return {
      id: `dry-run-${Date.now()}`,
      client_order_id: clientOrderId || `dry-run-client-${Date.now()}`,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      submitted_at: new Date().toISOString(),
      filled_at: new Date().toISOString(),
      symbol,
      qty: qty.toString(),
      filled_qty: qty.toString(),
      side,
      type,
      status: "filled",
      filled_avg_price: limitPrice?.toString() || "0",
    };
  }

  const clientOrderIdLog = clientOrderId ? ` client_order_id=${clientOrderId}` : "";
  console.log(`[TRADE] ORDER_SUBMIT symbol=${symbol} side=${side} qty=${qty}${tradeIdLog}${clientOrderIdLog}`);

  const order = await alpacaRequest<AlpacaOrder>("/v2/orders", {
    method: "POST",
    body: JSON.stringify(body),
  });
  
  console.log(`[TRADE] ORDER_ACCEPTED order_id=${order.id} client_order_id=${order.client_order_id} symbol=${symbol} side=${side} qty=${qty}`);
  
  return order;
}

/**
 * Submit a bracket order with built-in stop loss and take profit (broker-native risk management)
 * This ensures every entry has an attached exit order - no orphaned positions
 */
export async function submitBracketOrder(
  symbol: string,
  qty: number,
  side: "buy" | "sell",
  stopLossPrice: number,
  takeProfitPrice: number,
  clientOrderId?: string
): Promise<AlpacaOrder> {
  const tradeIdLog = clientOrderId ? ` trade_id=${clientOrderId}` : "";
  
  // MARKET-OPEN GATE: Block orders when market is closed
  const gate = await canSubmitOrderNow();
  if (!gate.allowed) {
    if (!shouldSuppressGuardLog("bracket-submit", symbol)) {
      console.log(`[GUARD] MARKET CLOSED — blocked bracket order (reason=${gate.reason}, symbol=${symbol}, side=${side}, qty=${qty}${tradeIdLog})`);
    }
    throw new Error(`Bracket order blocked: ${gate.reason}`);
  }
  
  // TRADING WINDOW GATE: Block ENTRY orders outside 09:35-11:35 ET
  if (side === "buy") {
    const windowCheck = isWithinTradingWindow();
    if (!windowCheck.allowed) {
      if (!shouldSuppressGuardLog("window-bracket", symbol)) {
        console.log(`[GUARD] OUTSIDE_TRADING_WINDOW — blocked bracket order (reason=${windowCheck.reason}, symbol=${symbol}${tradeIdLog})`);
      }
      throw new Error(`Bracket order blocked: ${windowCheck.reason}`);
    }
  }
  
  // Validate stop loss and take profit prices
  if (side === "buy") {
    if (stopLossPrice >= takeProfitPrice) {
      throw new Error(`Invalid bracket: stop_loss (${stopLossPrice}) must be below take_profit (${takeProfitPrice}) for buy orders`);
    }
  } else {
    if (stopLossPrice <= takeProfitPrice) {
      throw new Error(`Invalid bracket: stop_loss (${stopLossPrice}) must be above take_profit (${takeProfitPrice}) for sell orders`);
    }
  }
  
  const body: Record<string, unknown> = {
    symbol,
    qty: qty.toString(),
    side,
    type: "market",
    time_in_force: "day",
    order_class: "bracket",
    stop_loss: {
      stop_price: stopLossPrice.toFixed(2),
    },
    take_profit: {
      limit_price: takeProfitPrice.toFixed(2),
    },
  };
  
  // Enforce standardized client_order_id format: ato:{tradeId}:ENTRY:{symbol}
  if (clientOrderId) {
    // If already in ato_ format, convert to leg format
    if (clientOrderId.startsWith("ato_")) {
      body.client_order_id = `${clientOrderId}:ENTRY:${symbol}`;
    } else {
      body.client_order_id = clientOrderId;
    }
  }
  
  // DRY_RUN mode - log intent but don't execute
  if (isDryRunEffective()) {
    console.log(`[DRY_RUN] WOULD PLACE BRACKET ORDER: symbol=${symbol}, side=${side}, qty=${qty}, stop_loss=${stopLossPrice}, take_profit=${takeProfitPrice}${tradeIdLog}`);
    return {
      id: `dry-run-bracket-${Date.now()}`,
      client_order_id: clientOrderId || `dry-run-bracket-${Date.now()}`,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      submitted_at: new Date().toISOString(),
      filled_at: new Date().toISOString(),
      symbol,
      qty: qty.toString(),
      filled_qty: qty.toString(),
      side,
      type: "market",
      status: "filled",
      filled_avg_price: "0",
    };
  }
  
  console.log(`[TRADE] BRACKET_ORDER_SUBMIT symbol=${symbol} side=${side} qty=${qty} stop_loss=${stopLossPrice} take_profit=${takeProfitPrice}${tradeIdLog}`);
  
  const order = await alpacaRequest<AlpacaOrder>("/v2/orders", {
    method: "POST",
    body: JSON.stringify(body),
  });
  
  console.log(`[TRADE] BRACKET_ORDER_ACCEPTED order_id=${order.id} symbol=${symbol} side=${side} qty=${qty} stop_loss=${stopLossPrice} take_profit=${takeProfitPrice}`);
  
  return order;
}

/**
 * P6: Submit a LIMIT bracket order for better fill quality
 * Uses limit price for entry instead of market to reduce slippage
 */
export async function submitLimitBracketOrder(
  symbol: string,
  qty: number,
  side: "buy" | "sell",
  limitPrice: number,
  stopLossPrice: number,
  takeProfitPrice: number,
  clientOrderId?: string
): Promise<AlpacaOrder> {
  const tradeIdLog = clientOrderId ? ` trade_id=${clientOrderId}` : "";
  
  // MARKET-OPEN GATE: Block orders when market is closed
  const gate = await canSubmitOrderNow();
  if (!gate.allowed) {
    if (!shouldSuppressGuardLog("limit-bracket-submit", symbol)) {
      console.log(`[GUARD] MARKET CLOSED — blocked limit bracket order (reason=${gate.reason}, symbol=${symbol}, side=${side}, qty=${qty}${tradeIdLog})`);
    }
    throw new Error(`Limit bracket order blocked: ${gate.reason}`);
  }
  
  // TRADING WINDOW GATE: Block ENTRY orders outside 09:35-11:35 ET
  if (side === "buy") {
    const windowCheck = isWithinTradingWindow();
    if (!windowCheck.allowed) {
      if (!shouldSuppressGuardLog("window-limit-bracket", symbol)) {
        console.log(`[GUARD] OUTSIDE_TRADING_WINDOW — blocked limit bracket order (reason=${windowCheck.reason}, symbol=${symbol}${tradeIdLog})`);
      }
      throw new Error(`Limit bracket order blocked: ${windowCheck.reason}`);
    }
  }
  
  // Validate stop loss and take profit prices
  if (side === "buy") {
    if (stopLossPrice >= limitPrice) {
      throw new Error(`Invalid bracket: stop_loss (${stopLossPrice}) must be below limit_price (${limitPrice}) for buy orders`);
    }
    if (limitPrice >= takeProfitPrice) {
      throw new Error(`Invalid bracket: limit_price (${limitPrice}) must be below take_profit (${takeProfitPrice}) for buy orders`);
    }
  }
  
  const body: Record<string, unknown> = {
    symbol,
    qty: qty.toString(),
    side,
    type: "limit",
    limit_price: limitPrice.toFixed(2),
    time_in_force: "day",
    order_class: "bracket",
    stop_loss: {
      stop_price: stopLossPrice.toFixed(2),
    },
    take_profit: {
      limit_price: takeProfitPrice.toFixed(2),
    },
  };
  
  // Enforce standardized client_order_id format: ato:{tradeId}:ENTRY:{symbol}
  if (clientOrderId) {
    // If already in ato_ format, convert to leg format
    if (clientOrderId.startsWith("ato_")) {
      body.client_order_id = `${clientOrderId}:ENTRY:${symbol}`;
    } else {
      body.client_order_id = clientOrderId;
    }
  }
  
  // DRY_RUN mode - log intent but don't execute
  if (isDryRunEffective()) {
    console.log(`[DRY_RUN] WOULD PLACE LIMIT BRACKET ORDER: symbol=${symbol}, side=${side}, qty=${qty}, limit=${limitPrice}, stop_loss=${stopLossPrice}, take_profit=${takeProfitPrice}${tradeIdLog}`);
    return {
      id: `dry-run-limit-bracket-${Date.now()}`,
      client_order_id: clientOrderId || `dry-run-limit-bracket-${Date.now()}`,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      submitted_at: new Date().toISOString(),
      filled_at: new Date().toISOString(),
      symbol,
      qty: qty.toString(),
      filled_qty: qty.toString(),
      side,
      type: "limit",
      status: "filled",
      filled_avg_price: limitPrice.toFixed(2),
    };
  }
  
  console.log(`[TRADE] LIMIT_BRACKET_ORDER_SUBMIT symbol=${symbol} side=${side} qty=${qty} limit=${limitPrice} stop_loss=${stopLossPrice} take_profit=${takeProfitPrice}${tradeIdLog}`);
  
  const order = await alpacaRequest<AlpacaOrder>("/v2/orders", {
    method: "POST",
    body: JSON.stringify(body),
  });
  
  console.log(`[TRADE] LIMIT_BRACKET_ORDER_ACCEPTED order_id=${order.id} symbol=${symbol} side=${side} qty=${qty} limit=${limitPrice} stop_loss=${stopLossPrice} take_profit=${takeProfitPrice}`);
  
  return order;
}

/**
 * VALIDATION ONLY: Submit a limit bracket order bypassing the trading window guard
 * This function is ONLY for the validation test suite to verify order pairing logic
 * It still checks market open status for safety
 */
export async function submitLimitBracketOrderForValidation(
  symbol: string,
  qty: number,
  side: "buy" | "sell",
  limitPrice: number,
  stopLossPrice: number,
  takeProfitPrice: number,
  clientOrderId?: string
): Promise<AlpacaOrder> {
  const tradeIdLog = clientOrderId ? ` trade_id=${clientOrderId}` : "";
  
  // MARKET-OPEN GATE: Still enforce - cannot place orders when market is closed
  const gate = await canSubmitOrderNow();
  if (!gate.allowed) {
    console.log(`[VALIDATION] MARKET CLOSED — blocked validation order (reason=${gate.reason}, symbol=${symbol}${tradeIdLog})`);
    throw new Error(`Validation order blocked: ${gate.reason}`);
  }
  
  // NOTE: SKIP trading window guard for validation purposes
  console.log(`[VALIDATION] Bypassing trading window guard for TEST B`);
  
  // Validate stop loss and take profit prices
  if (side === "buy") {
    if (stopLossPrice >= limitPrice) {
      throw new Error(`Invalid bracket: stop_loss (${stopLossPrice}) must be below limit_price (${limitPrice}) for buy orders`);
    }
    if (limitPrice >= takeProfitPrice) {
      throw new Error(`Invalid bracket: limit_price (${limitPrice}) must be below take_profit (${takeProfitPrice}) for buy orders`);
    }
  }
  
  const body: Record<string, unknown> = {
    symbol,
    qty: qty.toString(),
    side,
    type: "limit",
    limit_price: limitPrice.toFixed(2),
    time_in_force: "day",
    order_class: "bracket",
    stop_loss: {
      stop_price: stopLossPrice.toFixed(2),
    },
    take_profit: {
      limit_price: takeProfitPrice.toFixed(2),
    },
  };
  
  // Enforce standardized client_order_id format: ato:{tradeId}:ENTRY:{symbol}
  if (clientOrderId) {
    if (clientOrderId.startsWith("ato_")) {
      body.client_order_id = `${clientOrderId}:ENTRY:${symbol}`;
    } else {
      body.client_order_id = clientOrderId;
    }
  }
  
  console.log(`[VALIDATION] LIMIT_BRACKET_ORDER_SUBMIT symbol=${symbol} side=${side} qty=${qty} limit=${limitPrice} stop_loss=${stopLossPrice} take_profit=${takeProfitPrice}${tradeIdLog}`);
  
  const order = await alpacaRequest<AlpacaOrder>("/v2/orders", {
    method: "POST",
    body: JSON.stringify(body),
  });
  
  console.log(`[VALIDATION] LIMIT_BRACKET_ORDER_ACCEPTED order_id=${order.id} client_order_id=${order.client_order_id}`);
  
  return order;
}

export async function closePosition(symbol: string, exitReason?: string, tradeId?: string): Promise<AlpacaOrder> {
  const exitClientOrderId = tradeId ? `${tradeId}_EXIT` : undefined;
  const tradeIdLog = exitClientOrderId ? ` trade_id=${tradeId} exit_reason=${exitReason || 'close'}` : "";
  
  // MARKET-OPEN GATE: Block close orders when market is closed
  const gate = await canSubmitOrderNow();
  if (!gate.allowed) {
    // Log with cooldown to prevent spam - only log first occurrence per 60s
    if (!shouldSuppressGuardLog("close", symbol)) {
      console.log(`[GUARD] MARKET CLOSED — blocked close position (reason=${gate.reason}, symbol=${symbol}${tradeIdLog})`);
    }
    throw new Error(`Close blocked: ${gate.reason}`);
  }
  
  // NOTE: Trading window guard (09:35-11:35 ET) does NOT apply to exits
  // Exits must be allowed at all times for position management and FORT KNOX force-close at 15:45 ET
  
  // If we have a trade_id, use submitOrder to set client_order_id for HIGH confidence matching
  if (exitClientOrderId) {
    try {
      // Get current position quantity
      const positions = await getPositions();
      const pos = positions.find(p => p.symbol === symbol);
      if (pos) {
        const qty = Math.abs(parseInt(pos.qty));
        console.log(`ACTION=EXIT symbol=${symbol} side=sell qty=${qty} reason=${exitReason || 'close'} trade_id=${tradeId}`);
        
        // Use submitOrder which supports client_order_id
        // Skip the market-open gate check since we already did it
        const body: Record<string, unknown> = {
          symbol,
          qty: qty.toString(),
          side: "sell",
          type: "market",
          time_in_force: "day",
          client_order_id: exitClientOrderId,
        };
        
        if (isDryRunEffective()) {
          console.log(`[DRY_RUN] WOULD CLOSE POSITION: symbol=${symbol}${tradeIdLog}`);
          console.log(`ACTION=EXIT_FILL symbol=${symbol} side=sell qty=${qty} price=0.00 trade_id=${tradeId}`);
          return {
            id: `dry-run-close-${Date.now()}`,
            client_order_id: exitClientOrderId,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            submitted_at: new Date().toISOString(),
            filled_at: new Date().toISOString(),
            symbol,
            qty: qty.toString(),
            filled_qty: qty.toString(),
            side: "sell",
            type: "market",
            status: "filled",
            filled_avg_price: "0",
          };
        }
        
        console.log(`[TRADE] EXIT_SUBMIT symbol=${symbol} side=sell qty=${qty} client_order_id=${exitClientOrderId}`);
        
        const exitOrder = await alpacaRequest<AlpacaOrder>("/v2/orders", {
          method: "POST",
          body: JSON.stringify(body),
        });
        
        console.log(`[TRADE] EXIT_ACCEPTED order_id=${exitOrder.id} client_order_id=${exitOrder.client_order_id} symbol=${symbol} side=sell qty=${qty}`);
        
        const exitPrice = parseFloat(exitOrder.filled_avg_price || "0");
        console.log(`ACTION=EXIT_FILL symbol=${symbol} side=sell qty=${qty} price=${exitPrice.toFixed(2)} trade_id=${tradeId} client_order_id=${exitClientOrderId}`);
        
        return exitOrder;
      }
    } catch (error) {
      console.error(`[TRADE] Error closing with trade_id, falling back to DELETE:`, error);
    }
  }
  
  // DRY_RUN mode - log intent but don't execute
  if (isDryRunEffective()) {
    console.log(`[DRY_RUN] WOULD CLOSE POSITION: symbol=${symbol}${tradeIdLog}`);
    return {
      id: `dry-run-close-${Date.now()}`,
      client_order_id: exitClientOrderId || `dry-run-client-${Date.now()}`,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      submitted_at: new Date().toISOString(),
      filled_at: new Date().toISOString(),
      symbol,
      qty: "0",
      filled_qty: "0",
      side: "sell",
      type: "market",
      status: "filled",
      filled_avg_price: "0",
    };
  }
  
  console.log(`[TRADE] EXIT symbol=${symbol}${tradeIdLog}`);
  return alpacaRequest<AlpacaOrder>(`/v2/positions/${symbol}`, {
    method: "DELETE",
  });
}

export async function closeAllPositions(): Promise<AlpacaOrder[]> {
  // MARKET-OPEN GATE: Block close-all orders when market is closed
  const gate = await canSubmitOrderNow();
  if (!gate.allowed) {
    // Log with cooldown to prevent spam - only log first occurrence per 60s
    if (!shouldSuppressGuardLog("close-all", "ALL")) {
      console.log(`[GUARD] MARKET CLOSED — blocked close-all positions (reason=${gate.reason})`);
    }
    throw new Error(`Close-all blocked: ${gate.reason}`);
  }
  
  // NOTE: Trading window guard (09:35-11:35 ET) does NOT apply to exits
  // Exits must be allowed at all times for position management and FORT KNOX force-close at 15:45 ET
  
  // DRY_RUN mode - log intent but don't execute
  if (isDryRunEffective()) {
    console.log(`[DRY_RUN] WOULD CLOSE ALL POSITIONS`);
    return [];
  }
  return alpacaRequest<AlpacaOrder[]>("/v2/positions", {
    method: "DELETE",
  });
}

// Check if DRY_RUN mode is active (env only - legacy)
export function isDryRun(): boolean {
  return DRY_RUN_ENV;
}

// Check if DRY_RUN mode is effectively active (env OR auto-test forced)
export function isDryRunActive(): boolean {
  return isDryRunEffective();
}

export async function getLatestQuote(symbol: string): Promise<{ price: number }> {
  try {
    // Try the latest quote endpoint
    const response = await alpacaRequest<{
      quote?: { ap: number; bp: number };
      quotes?: Record<string, { ap: number; bp: number }>;
    }>(`/v2/stocks/${symbol}/quotes/latest`, {}, true);
    
    // Handle different response formats
    const quote = response.quote || response.quotes?.[symbol];
    if (quote && (quote.ap > 0 || quote.bp > 0)) {
      const price = quote.ap > 0 && quote.bp > 0 
        ? (quote.ap + quote.bp) / 2 
        : (quote.ap || quote.bp);
      console.log(`[Alpaca] Quote for ${symbol}: ask=${quote.ap}, bid=${quote.bp}, mid=${price}`);
      return { price };
    }
    
    // Fallback: try to get from latest trade
    const tradeResponse = await alpacaRequest<{
      trade?: { p?: number };
    }>(`/v2/stocks/${symbol}/trades/latest`, {}, true);
    
    const tradePrice = tradeResponse.trade?.p;
    if (tradePrice && tradePrice > 0) {
      console.log(`[Alpaca] Last trade for ${symbol}: ${tradePrice}`);
      return { price: tradePrice };
    }
    
    console.log(`[Alpaca] No price data for ${symbol}`);
    return { price: 0 };
  } catch (error) {
    console.log(`[Alpaca] Quote error for ${symbol}:`, error);
    return { price: 0 };
  }
}

// Extended quote with bid/ask/volume/timestamp for strategy engine and P6 execution quality
export interface ExtendedQuote {
  price: number;
  bid: number;
  ask: number;
  volume: number;
  timestamp: string | null;  // P6: For quote freshness gate
}

export async function getExtendedQuote(symbol: string): Promise<ExtendedQuote> {
  try {
    const response = await alpacaRequest<{
      quote?: { ap: number; bp: number; as?: number; bs?: number; t?: string };
      quotes?: Record<string, { ap: number; bp: number; as?: number; bs?: number; t?: string }>;
    }>(`/v2/stocks/${symbol}/quotes/latest`, {}, true);
    
    const quote = response.quote || response.quotes?.[symbol];
    if (quote && (quote.ap > 0 || quote.bp > 0)) {
      const price = quote.ap > 0 && quote.bp > 0 
        ? (quote.ap + quote.bp) / 2 
        : (quote.ap || quote.bp);
      return { 
        price, 
        bid: quote.bp || 0, 
        ask: quote.ap || 0,
        volume: 0,  // Quote doesn't include volume, will be from snapshot
        timestamp: quote.t || null,
      };
    }
    
    return { price: 0, bid: 0, ask: 0, volume: 0, timestamp: null };
  } catch (error) {
    return { price: 0, bid: 0, ask: 0, volume: 0, timestamp: null };
  }
}

export async function getBars(
  symbol: string,
  timeframe = "1Day",
  limit = 30
): Promise<AlpacaBar[]> {
  try {
    // Alpaca data API v2 response format
    const response = await alpacaRequest<{
      bars?: AlpacaBar[] | Record<string, AlpacaBar[]>;
      [key: string]: AlpacaBar[] | Record<string, AlpacaBar[]> | undefined;
    }>(`/v2/stocks/${symbol}/bars?timeframe=${timeframe}&limit=${limit}`, {}, true);
    
    // Handle different response formats from Alpaca
    // Format 1: { bars: [...] } - direct array
    if (Array.isArray(response.bars)) {
      console.log(`[Alpaca] Got ${response.bars.length} bars for ${symbol}`);
      return response.bars;
    }
    // Format 2: { bars: { SYMBOL: [...] } } - nested by symbol
    if (response.bars && typeof response.bars === 'object') {
      const symbolBars = (response.bars as Record<string, AlpacaBar[]>)[symbol];
      if (symbolBars) {
        console.log(`[Alpaca] Got ${symbolBars.length} bars for ${symbol} (nested format)`);
        return symbolBars;
      }
    }
    // Format 3: Direct symbol key in response
    if (response[symbol] && Array.isArray(response[symbol])) {
      const bars = response[symbol] as AlpacaBar[];
      console.log(`[Alpaca] Got ${bars.length} bars for ${symbol} (direct format)`);
      return bars;
    }
    
    console.log(`[Alpaca] No bars found for ${symbol}, response keys:`, Object.keys(response));
    return [];
  } catch (error) {
    console.error(`[Alpaca] Error fetching bars for ${symbol}:`, error);
    return [];
  }
}

// Structured result for safe bar fetching
export interface BarFetchResult {
  ok: boolean;
  bars: AlpacaBar[];
  reason?: "NO_BARS_RETURNED" | "INSUFFICIENT_BARS" | "INVALID_CLOSE_VALUES" | "API_ERROR";
  barsLen: number;
  tier: 1 | 2 | 0;  // 0 = insufficient, 1 = limited indicators, 2 = full indicators
}

// TIERED INDICATOR READINESS THRESHOLDS
export const TIER_1_MIN_BARS = 130;  // Limited indicators (VWAP, RSI, short EMAs)
export const TIER_2_MIN_BARS = 200;  // Full indicator stack (EMA/RSI/VWAP/MACD)
const BARS_FETCH_LIMIT = 300; // Request extra buffer for warm-start

/**
 * Determine tier based on bar count
 */
function determineTier(barsLen: number): 0 | 1 | 2 {
  if (barsLen >= TIER_2_MIN_BARS) return 2;
  if (barsLen >= TIER_1_MIN_BARS) return 1;
  return 0;
}

/**
 * Get historical bars with warm-start (includes previous trading day)
 * Uses start/end timestamps to span multiple sessions
 * Respects Alpaca free-tier constraints by:
 * - Using feed=iex parameter
 * - Ensuring end time is at least 15 minutes in the past to avoid SIP embargo
 */
async function getBarsWithWarmStart(
  symbol: string,
  timeframe: string,
  limit: number
): Promise<AlpacaBar[]> {
  try {
    // Calculate end time: 20 minutes in the past to avoid SIP embargo (15 min + 5 min buffer)
    const now = new Date();
    const end = new Date(now.getTime() - 20 * 60 * 1000);
    
    // Calculate start time: go back multiple trading days for warm-start
    const daysBack = timeframe === "1Day" ? 30 : 5; // 5 days for intraday, 30 for daily
    const start = new Date(end);
    start.setDate(start.getDate() - daysBack);
    
    // Format dates for Alpaca API (RFC-3339)
    const startStr = start.toISOString();
    const endStr = end.toISOString();
    
    // Request with time range for warm-start (use IEX feed for free Alpaca accounts)
    const response = await alpacaRequest<{
      bars?: AlpacaBar[] | Record<string, AlpacaBar[]>;
      [key: string]: AlpacaBar[] | Record<string, AlpacaBar[]> | undefined;
    }>(`/v2/stocks/${symbol}/bars?timeframe=${timeframe}&start=${startStr}&end=${endStr}&limit=${limit}&feed=iex`, {}, true);
    
    // Handle different response formats from Alpaca
    let bars: AlpacaBar[] = [];
    if (Array.isArray(response.bars)) {
      bars = response.bars;
    } else if (response.bars && typeof response.bars === 'object') {
      const symbolBars = (response.bars as Record<string, AlpacaBar[]>)[symbol];
      if (symbolBars) bars = symbolBars;
    } else if (response[symbol] && Array.isArray(response[symbol])) {
      bars = response[symbol] as AlpacaBar[];
    }
    
    if (bars.length > 0) {
      // Deduplicate by timestamp
      const seen = new Set<string>();
      const uniqueBars = bars.filter(b => {
        if (seen.has(b.t)) return false;
        seen.add(b.t);
        return true;
      });
      
      // Sort ascending by time
      uniqueBars.sort((a, b) => new Date(a.t).getTime() - new Date(b.t).getTime());
      
      // Cap at limit, keeping most recent
      const result = uniqueBars.length > limit 
        ? uniqueBars.slice(-limit) 
        : uniqueBars;
      
      console.log(`[Alpaca] WarmStart ${symbol}: ${result.length} bars (raw=${bars.length}, days_back=${daysBack})`);
      return result;
    }
    
    console.log(`[Alpaca] WarmStart ${symbol}: No bars returned`);
    return [];
    
  } catch (error) {
    console.log(`[Alpaca] WarmStart ${symbol}: Error - ${error}`);
    return [];
  }
}

/**
 * Fallback bar fetching without date range (simple limit-based)
 * Used when warm-start fails
 */
async function getBarsFallback(
  symbol: string,
  timeframe: string,
  limit: number
): Promise<AlpacaBar[]> {
  try {
    const response = await alpacaRequest<{
      bars?: AlpacaBar[] | Record<string, AlpacaBar[]>;
      [key: string]: AlpacaBar[] | Record<string, AlpacaBar[]> | undefined;
    }>(`/v2/stocks/${symbol}/bars?timeframe=${timeframe}&limit=${limit}&feed=iex`, {}, true);
    
    let bars: AlpacaBar[] = [];
    if (Array.isArray(response.bars)) {
      bars = response.bars;
    } else if (response.bars && typeof response.bars === 'object') {
      const symbolBars = (response.bars as Record<string, AlpacaBar[]>)[symbol];
      if (symbolBars) bars = symbolBars;
    } else if (response[symbol] && Array.isArray(response[symbol])) {
      bars = response[symbol] as AlpacaBar[];
    }
    
    if (bars.length > 0) {
      console.log(`[Alpaca] Fallback ${symbol}: ${bars.length} bars`);
      return bars;
    }
    
    return [];
  } catch (error) {
    console.log(`[Alpaca] Fallback ${symbol}: Error - ${error}`);
    return [];
  }
}

/**
 * Safe bar fetcher with retry, validation, tier determination, and warm-start
 * - Uses historical warm-start to bootstrap bars from previous trading day
 * - Falls back to simple limit-based fetch if warm-start fails
 * - Retries up to 2 times on failure
 * - Validates bar count and close values
 * - Returns tier based on bar availability (0=insufficient, 1=limited, 2=full)
 */
export async function getBarsSafe(
  symbol: string,
  timeframe: "1Min" | "5Min" | "15Min" | "1Hour" | "1Day" = "5Min",
  minBars: number = TIER_1_MIN_BARS
): Promise<BarFetchResult> {
  const maxRetries = 2;
  let lastBars: AlpacaBar[] = [];
  let usedFallback = false;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      if (attempt > 0) {
        await new Promise(resolve => setTimeout(resolve, 300)); // Short delay between retries
      }
      
      // Try warm-start first, then fallback if it fails
      let bars: AlpacaBar[];
      if (!usedFallback) {
        bars = await getBarsWithWarmStart(symbol, timeframe, BARS_FETCH_LIMIT);
        
        // If warm-start failed, try fallback
        if (!bars || bars.length === 0) {
          console.log(`[Alpaca] getBarsSafe ${symbol}: Warm-start failed, trying fallback...`);
          bars = await getBarsFallback(symbol, timeframe, BARS_FETCH_LIMIT);
          usedFallback = true;
        }
      } else {
        bars = await getBarsFallback(symbol, timeframe, BARS_FETCH_LIMIT);
      }
      
      lastBars = bars;
      
      // Validation 1: Check if bars returned
      if (!bars || bars.length === 0) {
        console.log(`[Alpaca] getBarsSafe ${symbol}: NO_BARS_RETURNED (attempt ${attempt + 1})`);
        if (attempt < maxRetries) continue;
        return { ok: false, bars: [], reason: "NO_BARS_RETURNED", barsLen: 0, tier: 0 };
      }
      
      // Validation 2: Check close values are finite
      const invalidCloses = bars.filter(b => !Number.isFinite(b.c) || b.c <= 0);
      if (invalidCloses.length > 0) {
        console.log(`[Alpaca] getBarsSafe ${symbol}: INVALID_CLOSE_VALUES (${invalidCloses.length} invalid) (attempt ${attempt + 1})`);
        if (attempt < maxRetries) continue;
        return { ok: false, bars, reason: "INVALID_CLOSE_VALUES", barsLen: bars.length, tier: 0 };
      }
      
      // Determine tier based on bar count
      const tier = determineTier(bars.length);
      
      // Validation 3: Check minimum bar count for Tier 1
      if (bars.length < TIER_1_MIN_BARS) {
        console.log(`[Alpaca] getBarsSafe ${symbol}: INSUFFICIENT_BARS (${bars.length}/${TIER_1_MIN_BARS}) tier=0 (attempt ${attempt + 1})`);
        if (attempt < maxRetries) continue;
        return { ok: false, bars, reason: "INSUFFICIENT_BARS", barsLen: bars.length, tier: 0 };
      }
      
      // Success - at least Tier 1 available
      const method = usedFallback ? "fallback" : "warm-start";
      console.log(`[Alpaca] getBarsSafe ${symbol}: OK (${bars.length} bars, tf=${timeframe}, tier=${tier}, method=${method})`);
      return { ok: true, bars, barsLen: bars.length, tier };
      
    } catch (error) {
      console.log(`[Alpaca] getBarsSafe ${symbol}: API_ERROR (attempt ${attempt + 1}): ${error}`);
      if (attempt < maxRetries) continue;
    }
  }
  
  return { ok: false, bars: lastBars, reason: "API_ERROR", barsLen: lastBars.length, tier: 0 };
}

export function isConfigured(): boolean {
  if (activeCredentials) return true;
  return !!(ALPACA_API_KEY && ALPACA_API_SECRET);
}

// Get candlestick (OHLCV) data for day trading analysis
export interface CandleData {
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  vwap?: number;
}

export async function getCandlesticks(
  symbol: string,
  timeframe: "1Min" | "5Min" | "15Min" | "1Hour" = "5Min",
  limit: number = 20
): Promise<CandleData[]> {
  try {
    const bars = await getBars(symbol, timeframe, limit);
    
    if (bars.length === 0) {
      console.log(`[Alpaca] No candlestick data for ${symbol}`);
      return [];
    }
    
    return bars.map(bar => ({
      timestamp: new Date(bar.t),
      open: bar.o,
      high: bar.h,
      low: bar.l,
      close: bar.c,
      volume: bar.v,
    }));
  } catch (error) {
    console.error(`[Alpaca] Error fetching candlesticks for ${symbol}:`, error);
    return [];
  }
}

// Get real-time snapshot for a symbol (best price data available)
export async function getSnapshot(symbol: string): Promise<{
  price: number;
  bid: number;
  ask: number;
  bidSize: number;
  askSize: number;
  lastTrade: number;
  volume: number;
} | null> {
  try {
    // Try to get the latest quote
    const quoteResponse = await alpacaRequest<{
      quote?: {
        ap: number;
        as: number;
        bp: number;
        bs: number;
      };
    }>(`/v2/stocks/${symbol}/quotes/latest`, {}, true);
    
    // Try to get latest trade
    const tradeResponse = await alpacaRequest<{
      trade?: {
        p: number;
        s: number;
      };
    }>(`/v2/stocks/${symbol}/trades/latest`, {}, true);
    
    const quote = quoteResponse.quote;
    const trade = tradeResponse.trade;
    
    if (!quote && !trade) {
      return null;
    }
    
    const bid = quote?.bp || 0;
    const ask = quote?.ap || 0;
    const lastTrade = trade?.p || 0;
    const price = bid > 0 && ask > 0 ? (bid + ask) / 2 : lastTrade;
    
    return {
      price,
      bid,
      ask,
      bidSize: quote?.bs || 0,
      askSize: quote?.as || 0,
      lastTrade,
      volume: trade?.s || 0,
    };
  } catch (error) {
    console.error(`[Alpaca] Snapshot error for ${symbol}:`, error);
    return null;
  }
}

interface PortfolioHistoryResponse {
  timestamp: number[];
  equity: number[];
  profit_loss: number[];
  profit_loss_pct: number[];
  base_value: number;
  timeframe: string;
}

export async function getPortfolioHistory(
  period: string = "1D",
  timeframe: string = "5Min"
): Promise<{ timestamp: number; equity: number; profitLoss: number }[]> {
  try {
    const response = await alpacaRequest<PortfolioHistoryResponse>(
      `/v2/account/portfolio/history?period=${period}&timeframe=${timeframe}`
    );
    
    if (!response.timestamp || !response.equity) {
      return [];
    }
    
    return response.timestamp.map((ts, i) => ({
      timestamp: ts * 1000, // Convert to milliseconds
      equity: response.equity[i] || 0,
      profitLoss: response.profit_loss?.[i] || 0,
    }));
  } catch (error) {
    console.error("[Alpaca] Portfolio history error:", error);
    return [];
  }
}

// ALPACA-CONNECTIVITY-PROOF-1: Heartbeat system
let alpacaHeartbeatInterval: NodeJS.Timeout | null = null;
let alpacaAccountPingInterval: NodeJS.Timeout | null = null;

async function pingAlpacaClock(): Promise<boolean> {
  try {
    await getClock();
    alpacaConnectivityState.lastClockPingUTC = new Date().toISOString();
    return true;
  } catch (err) {
    console.log("[Alpaca] Clock ping failed");
    return false;
  }
}

async function pingAlpacaAccount(): Promise<boolean> {
  try {
    await getAccount();
    alpacaConnectivityState.lastAccountPingUTC = new Date().toISOString();
    return true;
  } catch (err) {
    console.log("[Alpaca] Account ping failed");
    return false;
  }
}

export async function startAlpacaHeartbeat(): Promise<void> {
  console.log("[Alpaca] Starting connectivity heartbeat...");
  
  // Initial ping
  await pingAlpacaClock();
  await pingAlpacaAccount();
  
  // Clock ping every 45 seconds during market hours
  if (alpacaHeartbeatInterval) clearInterval(alpacaHeartbeatInterval);
  alpacaHeartbeatInterval = setInterval(async () => {
    try {
      const clock = await getClock();
      if (clock.is_open) {
        // Only log ping during market hours for debugging
        alpacaConnectivityState.lastClockPingUTC = new Date().toISOString();
      }
    } catch (err) {
      console.log("[Alpaca] Heartbeat clock ping failed");
    }
  }, 45 * 1000);
  
  // Account ping every 5 minutes
  if (alpacaAccountPingInterval) clearInterval(alpacaAccountPingInterval);
  alpacaAccountPingInterval = setInterval(async () => {
    await pingAlpacaAccount();
  }, 5 * 60 * 1000);
  
  console.log("[Alpaca] Connectivity heartbeat started (clock: 45s, account: 5min)");
}

export function stopAlpacaHeartbeat(): void {
  if (alpacaHeartbeatInterval) {
    clearInterval(alpacaHeartbeatInterval);
    alpacaHeartbeatInterval = null;
  }
  if (alpacaAccountPingInterval) {
    clearInterval(alpacaAccountPingInterval);
    alpacaAccountPingInterval = null;
  }
}
