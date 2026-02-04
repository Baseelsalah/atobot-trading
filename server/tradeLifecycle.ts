/**
 * Trade Lifecycle Manager - P2 Implementation
 * 
 * Handles:
 * 1. Trade ID generation and order linking (trade_id -> parent + child orders)
 * 2. Order reconciliation with Alpaca (sync fills, statuses)
 * 3. Idempotency guard (prevent duplicate entries)
 * 4. Slippage tracking (signal price vs fill price)
 */

import { v4 as uuidv4 } from "uuid";
import * as alpaca from "./alpaca";
import { getEasternTime } from "./timezone";

// Callback for P3/P4 risk engine to receive trade outcomes with regime context
// Set by riskEngine to avoid circular imports
type TradeCloseCallback = (strategy: string, pnl: number, isWin: boolean, regime: "bull" | "bear" | "chop") => void;
let onTradeCloseCallback: TradeCloseCallback | null = null;

export function setOnTradeCloseCallback(callback: TradeCloseCallback): void {
  onTradeCloseCallback = callback;
}

// Trade status lifecycle
export type TradeStatus = 
  | "pending"      // Trade created, order not yet submitted
  | "submitted"    // Order submitted to Alpaca
  | "accepted"     // Order accepted by Alpaca
  | "partial"      // Partially filled
  | "filled"       // Entry filled, waiting for exit
  | "closed"       // Exit filled (stop or take profit)
  | "canceled"     // Order canceled
  | "rejected";    // Order rejected

// Exit leg type
export type ExitLeg = "stop_loss" | "take_profit" | "manual" | "force_close" | "unknown";

// Exit reason for reporting
export type ExitReason = "TP" | "SL" | "TIME" | "MANUAL" | "RECONCILE" | "FORCE_CLOSE";

// Time window for session analysis
export type TimeWindow = "open" | "mid" | "close";

// P4 Trade Metadata for measurement + tuning
export interface TradeMetadata {
  regime: "bull" | "bear" | "chop";
  timeWindow: TimeWindow;
  atr: number | null;
  atrPct: number | null;
  stopDistance: number | null;
  rr: number | null;  // R:R ratio
  usedAtrFallback: boolean;
  gatesPassed: boolean;
  gateFailReasons: string[];
  exitReason: ExitReason | null;
}

// Slippage data
export interface SlippageData {
  signalPrice: number;       // Price at decision time
  filledPrice: number;       // Actual fill price
  slippageBps: number;       // Slippage in basis points (buy: filled-signal, sell: signal-filled)
  spreadAtSignal?: number;   // Spread at decision time if available
}

// Trade record with full lifecycle data
export interface TradeRecord {
  tradeId: string;
  symbol: string;
  side: "buy" | "sell";
  strategy: string;
  
  // Order linking
  parentOrderId: string | null;      // Bracket parent order ID
  stopLossOrderId: string | null;    // Child stop loss order ID
  takeProfitOrderId: string | null;  // Child take profit order ID
  exitOrderId: string | null;        // Manual/force close order ID
  
  // Entry data
  entrySignalPrice: number;          // Price at decision time
  entryFilledPrice: number | null;   // Actual fill price
  entryFilledQty: number | null;     // Filled quantity
  entryFilledAt: string | null;      // Fill timestamp
  entrySlippage: SlippageData | null;
  
  // Exit data
  exitFilledPrice: number | null;
  exitFilledQty: number | null;
  exitFilledAt: string | null;
  exitLeg: ExitLeg | null;           // Which leg closed the trade
  exitSlippage: SlippageData | null;
  
  // Bracket prices
  stopLossPrice: number;
  takeProfitPrice: number;
  
  // Status
  status: TradeStatus;
  createdAt: string;
  updatedAt: string;
  
  // P&L (realized, from fills)
  realizedPnl: number | null;
  realizedPnlPercent: number | null;
  
  // P4 Metadata for measurement + tuning
  metadata: TradeMetadata | null;
}

// In-memory trade store
const activeTrades: Map<string, TradeRecord> = new Map();
const completedTrades: TradeRecord[] = [];

// Reconciliation state
let lastReconcileTime = 0;
const RECONCILE_INTERVAL_MS = 30000; // 30 seconds
let reconcileIntervalId: NodeJS.Timeout | null = null;

/**
 * Generate a structured trade_id for Alpaca client_order_id
 * Format: ato_{uuid} - the leg/symbol suffix is added by submitBracketOrder
 */
export function generateTradeId(): string {
  return `ato_${uuidv4().replace(/-/g, "").substring(0, 12)}`;
}

/**
 * Parse client_order_id in new format: ato_{id}:ENTRY:{symbol}
 * Returns trade_id and leg type
 */
export function parseNewFormatClientOrderId(clientOrderId: string): { tradeId: string; leg: string; symbol: string } | null {
  if (!clientOrderId) return null;
  
  // New format: ato_{id}:ENTRY:{symbol} or ato_{id}:SL:{symbol} or ato_{id}:TP:{symbol}
  const match = clientOrderId.match(/^(ato_[a-zA-Z0-9]+):(ENTRY|SL|TP):([A-Z]+)$/);
  if (match) {
    return {
      tradeId: match[1],
      leg: match[2],
      symbol: match[3],
    };
  }
  
  // Legacy format: ato_xxx
  if (clientOrderId.startsWith("ato_") && !clientOrderId.includes(":")) {
    return {
      tradeId: clientOrderId,
      leg: "ENTRY",
      symbol: "",
    };
  }
  
  return null;
}

/**
 * Parse a client_order_id to extract trade_id and leg
 */
export function parseClientOrderId(clientOrderId: string): { tradeId: string; leg: string } | null {
  if (!clientOrderId) return null;
  
  // Handle ato_xxx format
  if (clientOrderId.startsWith("ato_")) {
    const parts = clientOrderId.split("_");
    if (parts.length >= 2) {
      const tradeId = `ato_${parts[1]}`;
      const leg = parts[2] || "entry";
      return { tradeId, leg };
    }
  }
  
  // Handle legacy format (SPY_strategy_buy_date_time_rand_T2)
  if (clientOrderId.includes("_EXIT") || clientOrderId.includes("_PARTIAL")) {
    const baseId = clientOrderId.replace(/_EXIT$/, "").replace(/_PARTIAL$/, "");
    return { tradeId: baseId, leg: clientOrderId.includes("_EXIT") ? "exit" : "partial" };
  }
  
  return null;
}

/**
 * Determine time window based on Eastern Time (per P4 spec)
 * - open: 9:30 - 10:00 AM ET (first 30 min)
 * - mid: 10:00 AM - 3:00 PM ET
 * - close: 3:00 - 4:00 PM ET (last 60 min)
 */
export function determineTimeWindow(): TimeWindow {
  const et = getEasternTime();
  const minutesSinceMidnight = et.hour * 60 + et.minute;
  
  const OPEN_START = 9 * 60 + 30;   // 9:30 AM
  const MID_START = 10 * 60;        // 10:00 AM (first 30 min = open)
  const CLOSE_START = 15 * 60;       // 3:00 PM (last 60 min = close)
  
  // Per P4 spec: open = 9:30-10:00, mid = 10:00-15:00, close = 15:00-16:00
  // Pre-market (before 9:30) defaults to "mid" since no trades should occur then
  if (minutesSinceMidnight >= OPEN_START && minutesSinceMidnight < MID_START) {
    return "open";  // 9:30-10:00 AM = first 30 min
  } else if (minutesSinceMidnight >= MID_START && minutesSinceMidnight < CLOSE_START) {
    return "mid";   // 10:00 AM - 3:00 PM
  } else if (minutesSinceMidnight >= CLOSE_START) {
    return "close"; // 3:00 PM - 4:00 PM
  } else {
    return "mid";   // Pre-market defaults to mid (shouldn't happen during trading)
  }
}

/**
 * Create a new trade record when entry is decided
 * Call this BEFORE submitting the order
 */
export function createTrade(
  symbol: string,
  side: "buy" | "sell",
  strategy: string,
  signalPrice: number,
  stopLossPrice: number,
  takeProfitPrice: number,
  spreadAtSignal?: number,
  metadata?: Partial<TradeMetadata>
): TradeRecord {
  const now = new Date().toISOString();
  const tradeId = generateTradeId();
  
  // Build full metadata with defaults
  const timeWindow = determineTimeWindow();
  const fullMetadata: TradeMetadata = {
    regime: metadata?.regime || "chop",
    timeWindow,
    atr: metadata?.atr ?? null,
    atrPct: metadata?.atrPct ?? null,
    stopDistance: metadata?.stopDistance ?? null,
    rr: metadata?.rr ?? null,
    usedAtrFallback: metadata?.usedAtrFallback ?? false,
    gatesPassed: metadata?.gatesPassed ?? true,
    gateFailReasons: metadata?.gateFailReasons || [],
    exitReason: null,
  };
  
  const trade: TradeRecord = {
    tradeId,
    symbol,
    side,
    strategy,
    parentOrderId: null,
    stopLossOrderId: null,
    takeProfitOrderId: null,
    exitOrderId: null,
    entrySignalPrice: signalPrice,
    entryFilledPrice: null,
    entryFilledQty: null,
    entryFilledAt: null,
    entrySlippage: spreadAtSignal !== undefined ? {
      signalPrice,
      filledPrice: 0,
      slippageBps: 0,
      spreadAtSignal,
    } : null,
    exitFilledPrice: null,
    exitFilledQty: null,
    exitFilledAt: null,
    exitLeg: null,
    exitSlippage: null,
    stopLossPrice,
    takeProfitPrice,
    status: "pending",
    createdAt: now,
    updatedAt: now,
    realizedPnl: null,
    realizedPnlPercent: null,
    metadata: fullMetadata,
  };
  
  activeTrades.set(tradeId, trade);
  console.log(`[TradeLifecycle] Created trade_id=${tradeId} symbol=${symbol} side=${side} signal_price=${signalPrice.toFixed(2)} regime=${fullMetadata.regime} window=${fullMetadata.timeWindow}`);
  
  return trade;
}

/**
 * Link parent order ID to trade after submission
 */
export function linkParentOrder(tradeId: string, parentOrderId: string): void {
  const trade = activeTrades.get(tradeId);
  if (!trade) {
    console.warn(`[TradeLifecycle] Trade not found: ${tradeId}`);
    return;
  }
  
  trade.parentOrderId = parentOrderId;
  trade.status = "submitted";
  trade.updatedAt = new Date().toISOString();
  
  console.log(`[TradeLifecycle] Linked parent_order_id=${parentOrderId} to trade_id=${tradeId}`);
}

/**
 * Get client_order_id for entry order
 */
export function getEntryClientOrderId(tradeId: string): string {
  return `${tradeId}_entry`;
}

/**
 * Get client_order_id for exit order
 */
export function getExitClientOrderId(tradeId: string, reason: string): string {
  return `${tradeId}_exit_${reason}`;
}

/**
 * IDEMPOTENCY GUARD: Check if entry is allowed for symbol
 * Returns { allowed: boolean, reason: string }
 */
export async function canEnterPosition(symbol: string): Promise<{ allowed: boolean; reason: string }> {
  // Check 1: Open position in Alpaca
  try {
    const position = await alpaca.getPosition(symbol);
    if (position && parseInt(position.qty) !== 0) {
      return { allowed: false, reason: "idempotency:existingPosition" };
    }
  } catch {
    // No position is fine
  }
  
  // Check 2: Open order in Alpaca for this symbol (created in last 5 minutes)
  try {
    const orders = await alpaca.getOrders("open", 100);
    const symbolOrders = orders.filter(o => 
      o.symbol === symbol && 
      o.side === "buy" &&
      o.status !== "canceled" &&
      o.status !== "rejected"
    );
    
    if (symbolOrders.length > 0) {
      const recentOrder = symbolOrders.find(o => {
        const createdAt = new Date(o.created_at).getTime();
        const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
        return createdAt > fiveMinutesAgo;
      });
      
      if (recentOrder) {
        return { allowed: false, reason: "idempotency:existingOrder" };
      }
    }
  } catch (error) {
    console.warn(`[TradeLifecycle] Error checking orders for idempotency:`, error);
  }
  
  // Check 3: Active local trade for symbol
  for (const trade of Array.from(activeTrades.values())) {
    if (trade.symbol === symbol && 
        trade.side === "buy" && 
        (trade.status === "pending" || trade.status === "submitted" || 
         trade.status === "accepted" || trade.status === "filled")) {
      return { allowed: false, reason: "idempotency:activeTrade" };
    }
  }
  
  return { allowed: true, reason: "ok" };
}

/**
 * Update trade with entry fill data
 */
export function recordEntryFill(
  tradeId: string,
  filledPrice: number,
  filledQty: number,
  filledAt: string
): void {
  const trade = activeTrades.get(tradeId);
  if (!trade) {
    console.warn(`[TradeLifecycle] Trade not found for entry fill: ${tradeId}`);
    return;
  }
  
  trade.entryFilledPrice = filledPrice;
  trade.entryFilledQty = filledQty;
  trade.entryFilledAt = filledAt;
  trade.status = "filled";
  trade.updatedAt = new Date().toISOString();
  
  // Calculate entry slippage
  const slippageBps = trade.side === "buy"
    ? ((filledPrice - trade.entrySignalPrice) / trade.entrySignalPrice) * 10000
    : ((trade.entrySignalPrice - filledPrice) / trade.entrySignalPrice) * 10000;
  
  trade.entrySlippage = {
    signalPrice: trade.entrySignalPrice,
    filledPrice,
    slippageBps,
    spreadAtSignal: trade.entrySlippage?.spreadAtSignal,
  };
  
  console.log(`[TradeLifecycle] Entry fill: trade_id=${tradeId} price=${filledPrice.toFixed(2)} qty=${filledQty} slippage=${slippageBps.toFixed(1)}bps`);
}

/**
 * Update trade with exit fill data
 */
export function recordExitFill(
  tradeId: string,
  filledPrice: number,
  filledQty: number,
  filledAt: string,
  exitLeg: ExitLeg
): void {
  const trade = activeTrades.get(tradeId);
  if (!trade) {
    console.warn(`[TradeLifecycle] Trade not found for exit fill: ${tradeId}`);
    return;
  }
  
  trade.exitFilledPrice = filledPrice;
  trade.exitFilledQty = filledQty;
  trade.exitFilledAt = filledAt;
  trade.exitLeg = exitLeg;
  trade.status = "closed";
  trade.updatedAt = new Date().toISOString();
  
  // Calculate exit slippage (for sells: positive = got less than expected = bad)
  // Expected exit price depends on the leg:
  // - stop_loss: expected = stopLossPrice
  // - take_profit: expected = takeProfitPrice
  // - manual/force_close: use signal price (entry price as reference)
  let expectedExitPrice = trade.entrySignalPrice; // default
  if (exitLeg === "stop_loss") {
    expectedExitPrice = trade.stopLossPrice;
  } else if (exitLeg === "take_profit") {
    expectedExitPrice = trade.takeProfitPrice;
  }
  
  // For exit slippage: positive = worse than expected
  // Sell exit: if filled < expected, that's bad (positive slippage)
  const exitSlippageBps = ((expectedExitPrice - filledPrice) / expectedExitPrice) * 10000;
  
  trade.exitSlippage = {
    signalPrice: expectedExitPrice,
    filledPrice,
    slippageBps: exitSlippageBps,
  };
  
  // Calculate realized P&L
  if (trade.entryFilledPrice !== null && trade.entryFilledQty !== null) {
    const entryValue = trade.entryFilledPrice * trade.entryFilledQty;
    const exitValue = filledPrice * filledQty;
    
    if (trade.side === "buy") {
      trade.realizedPnl = exitValue - entryValue;
    } else {
      trade.realizedPnl = entryValue - exitValue;
    }
    
    trade.realizedPnlPercent = (trade.realizedPnl / entryValue) * 100;
  }
  
  // P4: Set exit reason in metadata for measurement
  if (trade.metadata) {
    trade.metadata.exitReason = exitLegToExitReason(exitLeg);
  }
  
  // Move to completed trades
  activeTrades.delete(tradeId);
  completedTrades.push(trade);
  
  const pnlStr = trade.realizedPnl !== null ? `$${trade.realizedPnl.toFixed(2)}` : "N/A";
  const regime = trade.metadata?.regime || "chop";
  console.log(`[TradeLifecycle] Exit fill: trade_id=${tradeId} price=${filledPrice.toFixed(2)} leg=${exitLeg} pnl=${pnlStr} regime=${regime}`);
  
  // P3/P4: Notify risk engine of trade outcome with REAL regime for kill-switch tracking
  if (onTradeCloseCallback && trade.realizedPnl !== null) {
    const isWin = trade.realizedPnl >= 0;
    onTradeCloseCallback(trade.strategy || "unknown", trade.realizedPnl, isWin, regime);
  }
}

/**
 * Convert ExitLeg to ExitReason for reporting
 */
function exitLegToExitReason(leg: ExitLeg): ExitReason {
  switch (leg) {
    case "take_profit": return "TP";
    case "stop_loss": return "SL";
    case "manual": return "MANUAL";
    case "force_close": return "FORCE_CLOSE";
    case "unknown": return "RECONCILE";
    default: return "RECONCILE";
  }
}

/**
 * Link child order IDs (stop loss and take profit) to trade
 */
export function linkChildOrders(
  tradeId: string,
  stopLossOrderId: string | null,
  takeProfitOrderId: string | null
): void {
  const trade = activeTrades.get(tradeId);
  if (!trade) {
    console.warn(`[TradeLifecycle] Trade not found for child linking: ${tradeId}`);
    return;
  }
  
  trade.stopLossOrderId = stopLossOrderId;
  trade.takeProfitOrderId = takeProfitOrderId;
  trade.updatedAt = new Date().toISOString();
  
  console.log(`[TradeLifecycle] Linked children to trade_id=${tradeId} sl_order=${stopLossOrderId} tp_order=${takeProfitOrderId}`);
}

/**
 * Get active trade by symbol
 */
export function getActiveTradeBySymbol(symbol: string): TradeRecord | null {
  for (const trade of Array.from(activeTrades.values())) {
    if (trade.symbol === symbol && trade.status !== "closed" && trade.status !== "canceled") {
      return trade;
    }
  }
  return null;
}

/**
 * Get all active trades
 */
export function getActiveTrades(): TradeRecord[] {
  return Array.from(activeTrades.values());
}

/**
 * Get completed trades for today
 */
export function getTodayCompletedTrades(): TradeRecord[] {
  const today = getEasternTime().dateString;
  return completedTrades.filter(t => t.createdAt.startsWith(today));
}

/**
 * Get all completed trades (for weekly scorecard)
 */
export function getCompletedTrades(): TradeRecord[] {
  return [...completedTrades];
}

/**
 * Get pairing metrics for daily/rolling reports
 * Returns counts of paired vs unpaired trades from lifecycle data
 */
export function getPairingMetrics(dateOrRange: string | { start: string; end: string }): {
  pairedTrades: number;
  unpairedEntries: number;
  unpairedExits: number;
  legacyOrders: number;
} {
  let dates: string[];
  if (typeof dateOrRange === "string") {
    dates = [dateOrRange];
  } else {
    dates = [];
    const startDate = new Date(dateOrRange.start);
    const endDate = new Date(dateOrRange.end);
    for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
      dates.push(d.toISOString().split("T")[0]);
    }
  }
  
  // Get all trades (active + completed) for the date range
  const allTrades = [...Array.from(activeTrades.values()), ...completedTrades];
  const tradesInRange = allTrades.filter(t => {
    const tradeDate = t.createdAt.split("T")[0];
    return dates.includes(tradeDate);
  });
  
  // Paired: has both entry AND exit filled
  const pairedTrades = tradesInRange.filter(t => 
    t.entryFilledPrice !== null && t.exitFilledPrice !== null
  ).length;
  
  // Unpaired entries: has entry but no exit (still open)
  const unpairedEntries = tradesInRange.filter(t =>
    t.entryFilledPrice !== null && t.exitFilledPrice === null && t.status !== "closed"
  ).length;
  
  // Unpaired exits: exit happened but no matching entry (shouldn't happen with proper tracking)
  // These would be orders matched by time in performanceReport but not in lifecycle
  const unpairedExits = 0; // Lifecycle doesn't track orphan exits
  
  // Legacy: trades without proper ato_ format trade_id
  const legacyOrders = tradesInRange.filter(t => {
    // Check if trade_id follows the new ato_ format
    const isNewFormat = t.tradeId.startsWith("ato_");
    return !isNewFormat;
  }).length;
  
  return {
    pairedTrades,
    unpairedEntries,
    unpairedExits,
    legacyOrders,
  };
}

/**
 * ORDER RECONCILIATION: Sync with Alpaca orders
 * Fetches orders and updates local trade records
 */
export async function reconcileOrders(): Promise<void> {
  const now = Date.now();
  
  try {
    // Get all orders from today
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const orders = await alpaca.getOrders("all", 200);
    
    // Build order map by ID
    const orderById = new Map<string, typeof orders[0]>();
    for (const order of orders) {
      orderById.set(order.id, order);
    }
    
    // Process each active trade
    for (const trade of Array.from(activeTrades.values())) {
      // Check parent order status
      if (trade.parentOrderId) {
        const parentOrder = orderById.get(trade.parentOrderId);
        if (parentOrder) {
          // Update entry fill if filled
          if (parentOrder.status === "filled" && !trade.entryFilledPrice) {
            recordEntryFill(
              trade.tradeId,
              parseFloat(parentOrder.filled_avg_price || "0"),
              parseInt(parentOrder.filled_qty),
              parentOrder.filled_at || new Date().toISOString()
            );
          }
          
          // Discover and link child orders from bracket legs - check individually
          // Also try to get legs from parent order's legs array (Alpaca may include them)
          const legs = (parentOrder as any).legs;
          if (Array.isArray(legs)) {
            for (const leg of legs) {
              const legId = leg.id || leg.order_id;
              if (!legId) continue;
              const legType = leg.type || leg.order_type;
              const hasStopPrice = leg.stop_price !== undefined && leg.stop_price !== null;
              const hasLimitPrice = leg.limit_price !== undefined && leg.limit_price !== null;
              
              if (!trade.stopLossOrderId && (legType === "stop" || hasStopPrice)) {
                trade.stopLossOrderId = legId;
              }
              if (!trade.takeProfitOrderId && (legType === "limit" || hasLimitPrice)) {
                trade.takeProfitOrderId = legId;
              }
            }
          }
          
          // Also look for child orders with this parent order as their parent_order_id
          for (const order of orders) {
            if ((order as any).parent_order_id === trade.parentOrderId) {
              const orderType = (order as any).order_type || order.type;
              const hasStopPrice = (order as any).stop_price !== undefined;
              const hasLimitPrice = (order as any).limit_price !== undefined;
              
              if (!trade.stopLossOrderId && (orderType === "stop" || hasStopPrice)) {
                trade.stopLossOrderId = order.id;
              }
              if (!trade.takeProfitOrderId && (orderType === "limit" || hasLimitPrice)) {
                trade.takeProfitOrderId = order.id;
              }
            }
          }
          
          if (trade.stopLossOrderId || trade.takeProfitOrderId) {
            trade.updatedAt = new Date().toISOString();
          }
          
          // Update status from order
          if (parentOrder.status === "accepted" && trade.status === "submitted") {
            trade.status = "accepted";
            trade.updatedAt = new Date().toISOString();
          }
          if (parentOrder.status === "canceled" && trade.status !== "closed") {
            trade.status = "canceled";
            trade.updatedAt = new Date().toISOString();
            activeTrades.delete(trade.tradeId);
            completedTrades.push(trade);
          }
          if (parentOrder.status === "rejected") {
            trade.status = "rejected";
            trade.updatedAt = new Date().toISOString();
            activeTrades.delete(trade.tradeId);
            completedTrades.push(trade);
          }
        }
      }
      
      // Check child orders for exit
      if (trade.status === "filled") {
        // Check stop loss
        if (trade.stopLossOrderId) {
          const slOrder = orderById.get(trade.stopLossOrderId);
          if (slOrder && slOrder.status === "filled" && !trade.exitFilledPrice) {
            recordExitFill(
              trade.tradeId,
              parseFloat(slOrder.filled_avg_price || "0"),
              parseInt(slOrder.filled_qty),
              slOrder.filled_at || new Date().toISOString(),
              "stop_loss"
            );
          }
        }
        
        // Check take profit
        if (trade.takeProfitOrderId) {
          const tpOrder = orderById.get(trade.takeProfitOrderId);
          if (tpOrder && tpOrder.status === "filled" && !trade.exitFilledPrice) {
            recordExitFill(
              trade.tradeId,
              parseFloat(tpOrder.filled_avg_price || "0"),
              parseInt(tpOrder.filled_qty),
              tpOrder.filled_at || new Date().toISOString(),
              "take_profit"
            );
          }
        }
        
        // Also check for manual/force close orders not tracked as child orders
        // Look for sell orders matching this symbol after entry
        // Only attribute to this trade if:
        // 1. Order was placed AFTER entry fill
        // 2. Order is not already attributed to another trade
        // 3. Quantity matches (or is close to) entry quantity
        if (!trade.exitFilledPrice && trade.entryFilledAt && trade.entryFilledQty) {
          for (const order of orders) {
            // Skip known child orders
            if (order.id === trade.stopLossOrderId || order.id === trade.takeProfitOrderId) {
              continue;
            }
            
            // Check if already attributed to another trade
            let alreadyUsed = false;
            for (const other of Array.from(activeTrades.values())) {
              if (other.tradeId !== trade.tradeId && other.exitOrderId === order.id) {
                alreadyUsed = true;
                break;
              }
            }
            if (alreadyUsed) continue;
            
            if (order.symbol === trade.symbol && 
                order.side === "sell" && 
                order.status === "filled") {
              const fillTime = new Date(order.filled_at || "").getTime();
              const entryTime = new Date(trade.entryFilledAt).getTime();
              const orderQty = parseInt(order.filled_qty);
              
              // Must be after entry and quantity should match within tolerance
              if (fillTime > entryTime && 
                  Math.abs(orderQty - trade.entryFilledQty) <= trade.entryFilledQty * 0.1) {
                trade.exitOrderId = order.id; // Track to prevent reuse
                recordExitFill(
                  trade.tradeId,
                  parseFloat(order.filled_avg_price || "0"),
                  orderQty,
                  order.filled_at || new Date().toISOString(),
                  "force_close"
                );
                break;
              }
            }
          }
        }
      }
    }
    
    // Detect orphan orders (orders with our prefix but no local trade)
    const orphanOrders: string[] = [];
    for (const order of orders) {
      const parsed = parseClientOrderId(order.client_order_id);
      if (parsed && parsed.tradeId.startsWith("ato_")) {
        const trade = activeTrades.get(parsed.tradeId) || 
                      completedTrades.find(t => t.tradeId === parsed.tradeId);
        if (!trade) {
          orphanOrders.push(order.id);
        }
      }
    }
    
    if (orphanOrders.length > 0) {
      console.log(`ACTION=RECONCILE ORPHAN_ORDERS=[${orphanOrders.join(",")}]`);
    }
    
    lastReconcileTime = now;
    
  } catch (error) {
    console.error(`[TradeLifecycle] Reconciliation error:`, error);
  }
}

/**
 * Start the reconciliation loop
 */
export function startReconciliationLoop(): void {
  if (reconcileIntervalId) {
    clearInterval(reconcileIntervalId);
  }
  
  reconcileIntervalId = setInterval(async () => {
    await reconcileOrders();
  }, RECONCILE_INTERVAL_MS);
  
  console.log(`[TradeLifecycle] Reconciliation loop started (interval: ${RECONCILE_INTERVAL_MS / 1000}s)`);
}

/**
 * Stop the reconciliation loop
 */
export function stopReconciliationLoop(): void {
  if (reconcileIntervalId) {
    clearInterval(reconcileIntervalId);
    reconcileIntervalId = null;
    console.log(`[TradeLifecycle] Reconciliation loop stopped`);
  }
}

/**
 * Get slippage statistics for completed trades
 */
export function getSlippageStats(): {
  avgSlippageBps: number;
  medianSlippageBps: number;
  worstSlippageBps: number;
  sampleCount: number;
} {
  const slippages: number[] = [];
  
  for (const trade of completedTrades) {
    if (trade.entrySlippage) {
      slippages.push(trade.entrySlippage.slippageBps);
    }
  }
  
  if (slippages.length === 0) {
    return {
      avgSlippageBps: 0,
      medianSlippageBps: 0,
      worstSlippageBps: 0,
      sampleCount: 0,
    };
  }
  
  slippages.sort((a, b) => a - b);
  
  const sum = slippages.reduce((a, b) => a + b, 0);
  const avg = sum / slippages.length;
  const median = slippages.length % 2 === 0
    ? (slippages[slippages.length / 2 - 1] + slippages[slippages.length / 2]) / 2
    : slippages[Math.floor(slippages.length / 2)];
  const worst = Math.max(...slippages);
  
  return {
    avgSlippageBps: avg,
    medianSlippageBps: median,
    worstSlippageBps: worst,
    sampleCount: slippages.length,
  };
}

/**
 * Get trade summary for daily report
 */
export function getDailyTradeSummary(): {
  totalTrades: number;
  filledTrades: number;
  closedTrades: number;
  openTrades: number;
  wins: number;
  losses: number;
  totalRealizedPnl: number;
  slippage: ReturnType<typeof getSlippageStats>;
} {
  const today = getEasternTime().dateString;
  const todayTrades = completedTrades.filter(t => t.createdAt.startsWith(today));
  const todayActive = Array.from(activeTrades.values()).filter(t => t.createdAt.startsWith(today));
  
  let wins = 0;
  let losses = 0;
  let totalPnl = 0;
  
  for (const trade of todayTrades) {
    if (trade.status === "closed" && trade.realizedPnl !== null) {
      totalPnl += trade.realizedPnl;
      if (trade.realizedPnl >= 0) {
        wins++;
      } else {
        losses++;
      }
    }
  }
  
  return {
    totalTrades: todayTrades.length + todayActive.length,
    filledTrades: todayTrades.filter(t => t.entryFilledPrice !== null).length + 
                  todayActive.filter(t => t.entryFilledPrice !== null).length,
    closedTrades: todayTrades.filter(t => t.status === "closed").length,
    openTrades: todayActive.filter(t => t.status === "filled").length,
    wins,
    losses,
    totalRealizedPnl: totalPnl,
    slippage: getSlippageStats(),
  };
}

/**
 * Clear daily state (call at start of new trading day)
 */
export function resetDailyState(): void {
  // Archive any stale active trades
  for (const trade of Array.from(activeTrades.values())) {
    trade.status = "canceled";
    completedTrades.push(trade);
  }
  activeTrades.clear();
  
  console.log(`[TradeLifecycle] Daily state reset`);
}
