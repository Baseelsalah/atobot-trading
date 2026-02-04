/**
 * P6 Execution Quality v1 - Limit entries + slippage controls
 * 
 * Reduces slippage and improves fill quality:
 * - Limit entries instead of market orders
 * - Quote staleness validation
 * - Spread-tightening near threshold
 * - Execution metrics tracking
 */

import * as alpaca from "./alpaca";

// Execution configuration
export const EXECUTION_CONFIG = {
  // Limit order settings
  ENTRY_OFFSET_SPREAD_PCT: 0.10,    // 10% of spread added to entry limit
  ENTRY_OFFSET_MIN_CENTS: 1,         // Minimum offset in cents
  ENTRY_OFFSET_MAX_CENTS: 5,         // Maximum offset in cents
  LOW_PRICE_THRESHOLD: 20,           // Sub-$20 uses fixed offset
  LOW_PRICE_FIXED_OFFSET: 0.01,      // $0.01 for sub-$20 stocks
  
  // Timeout settings
  FILL_TIMEOUT_MS: 45000,            // 45 seconds to fill
  POLL_INTERVAL_MS: 1000,            // Check every 1 second
  
  // Quote freshness
  QUOTE_MAX_AGE_MS: 5000,            // 5 seconds max quote age
  
  // Spread near-max gate
  SPREAD_NEAR_MAX_PCT: 0.90,         // Block if spread > 90% of max
};

// Execution metrics for reporting
export interface ExecutionMetrics {
  ordersSubmitted: number;
  ordersFilled: number;
  ordersCancelled: number;
  ordersTimedOut: number;
  totalTimeToFillMs: number;
  fillCount: number;  // For calculating average
  slippageByStrategy: Map<string, number[]>;  // Strategy -> slippage bps array
  slippageByTimeWindow: Map<string, number[]>; // TimeWindow -> slippage bps array
}

// Daily metrics storage
let dailyMetrics: ExecutionMetrics = createEmptyMetrics();

function createEmptyMetrics(): ExecutionMetrics {
  return {
    ordersSubmitted: 0,
    ordersFilled: 0,
    ordersCancelled: 0,
    ordersTimedOut: 0,
    totalTimeToFillMs: 0,
    fillCount: 0,
    slippageByStrategy: new Map(),
    slippageByTimeWindow: new Map(),
  };
}

export function resetDailyMetrics(): void {
  dailyMetrics = createEmptyMetrics();
  console.log("[ExecutionQuality] Daily metrics reset");
}

export function recordOrderSubmitted(): void {
  dailyMetrics.ordersSubmitted++;
}

export function recordOrderFilled(timeToFillMs: number): void {
  dailyMetrics.ordersFilled++;
  dailyMetrics.totalTimeToFillMs += timeToFillMs;
  dailyMetrics.fillCount++;
}

export function recordOrderCancelled(): void {
  dailyMetrics.ordersCancelled++;
}

export function recordOrderTimedOut(): void {
  dailyMetrics.ordersTimedOut++;
}

export function recordSlippage(
  strategyName: string,
  timeWindow: string,
  slippageBps: number
): void {
  // By strategy
  if (!dailyMetrics.slippageByStrategy.has(strategyName)) {
    dailyMetrics.slippageByStrategy.set(strategyName, []);
  }
  dailyMetrics.slippageByStrategy.get(strategyName)!.push(slippageBps);
  
  // By time window
  if (!dailyMetrics.slippageByTimeWindow.has(timeWindow)) {
    dailyMetrics.slippageByTimeWindow.set(timeWindow, []);
  }
  dailyMetrics.slippageByTimeWindow.get(timeWindow)!.push(slippageBps);
}

export function getExecutionMetrics(): ExecutionMetrics {
  return { ...dailyMetrics };
}

export interface ExecutionReport {
  fillRate: number;            // % of orders filled (vs cancelled/timed out)
  avgTimeToFillMs: number;
  cancelRate: number;          // % of orders cancelled
  timeoutRate: number;         // % of orders timed out
  slippageByStrategy: Record<string, { avg: number; median: number; worst: number }>;
  slippageByTimeWindow: Record<string, { avg: number; median: number; worst: number }>;
}

function calculateSlippageStats(values: number[]): { avg: number; median: number; worst: number } {
  if (values.length === 0) {
    return { avg: 0, median: 0, worst: 0 };
  }
  
  const sorted = [...values].sort((a, b) => a - b);
  const avg = values.reduce((sum, v) => sum + v, 0) / values.length;
  const median = sorted[Math.floor(sorted.length / 2)];
  const worst = sorted[sorted.length - 1];  // Highest slippage (worst)
  
  return { avg: Math.round(avg * 100) / 100, median, worst };
}

export function getExecutionReport(): ExecutionReport {
  const total = dailyMetrics.ordersSubmitted;
  
  const fillRate = total > 0 ? (dailyMetrics.ordersFilled / total) * 100 : 0;
  const cancelRate = total > 0 ? (dailyMetrics.ordersCancelled / total) * 100 : 0;
  const timeoutRate = total > 0 ? (dailyMetrics.ordersTimedOut / total) * 100 : 0;
  const avgTimeToFillMs = dailyMetrics.fillCount > 0 
    ? dailyMetrics.totalTimeToFillMs / dailyMetrics.fillCount 
    : 0;
  
  // Aggregate slippage stats
  const slippageByStrategy: Record<string, { avg: number; median: number; worst: number }> = {};
  dailyMetrics.slippageByStrategy.forEach((values, strategy) => {
    slippageByStrategy[strategy] = calculateSlippageStats(values);
  });
  
  const slippageByTimeWindow: Record<string, { avg: number; median: number; worst: number }> = {};
  dailyMetrics.slippageByTimeWindow.forEach((values, window) => {
    slippageByTimeWindow[window] = calculateSlippageStats(values);
  });
  
  return {
    fillRate: Math.round(fillRate * 100) / 100,
    avgTimeToFillMs: Math.round(avgTimeToFillMs),
    cancelRate: Math.round(cancelRate * 100) / 100,
    timeoutRate: Math.round(timeoutRate * 100) / 100,
    slippageByStrategy,
    slippageByTimeWindow,
  };
}

/**
 * Calculate entry limit price based on current quote
 * For BUY: min(ask, lastPrice) + offset
 */
export function calculateEntryLimit(
  bid: number,
  ask: number,
  lastPrice: number
): { limitPrice: number; offset: number; reasoning: string } {
  const spread = ask - bid;
  
  // For sub-$20 stocks, use fixed offset
  if (lastPrice < EXECUTION_CONFIG.LOW_PRICE_THRESHOLD) {
    const offset = EXECUTION_CONFIG.LOW_PRICE_FIXED_OFFSET;
    const basePrice = Math.min(ask, lastPrice);
    const limitPrice = basePrice + offset;
    return {
      limitPrice: Math.round(limitPrice * 100) / 100,
      offset,
      reasoning: `sub-$20 fixed offset $${offset.toFixed(2)}`,
    };
  }
  
  // Calculate offset as 10% of spread, bounded
  let offset = spread * EXECUTION_CONFIG.ENTRY_OFFSET_SPREAD_PCT;
  offset = Math.max(offset, EXECUTION_CONFIG.ENTRY_OFFSET_MIN_CENTS / 100);
  offset = Math.min(offset, EXECUTION_CONFIG.ENTRY_OFFSET_MAX_CENTS / 100);
  
  const basePrice = Math.min(ask, lastPrice);
  const limitPrice = basePrice + offset;
  
  return {
    limitPrice: Math.round(limitPrice * 100) / 100,
    offset: Math.round(offset * 100) / 100,
    reasoning: `10% of spread=$${spread.toFixed(3)}, offset=$${offset.toFixed(3)}`,
  };
}

/**
 * Check if quote is fresh enough to trade
 * Returns false if quote timestamp is older than MAX_AGE_MS
 */
export function isQuoteFresh(quoteTimestamp: string | null | undefined): {
  fresh: boolean;
  ageMs: number | null;
  reason?: string;
} {
  if (!quoteTimestamp) {
    return { fresh: false, ageMs: null, reason: "quote:missing" };
  }
  
  try {
    const quoteTime = new Date(quoteTimestamp).getTime();
    const now = Date.now();
    const ageMs = now - quoteTime;
    
    if (ageMs > EXECUTION_CONFIG.QUOTE_MAX_AGE_MS) {
      return {
        fresh: false,
        ageMs,
        reason: `quote:stale_age=${Math.round(ageMs / 1000)}s`,
      };
    }
    
    return { fresh: true, ageMs };
  } catch {
    return { fresh: false, ageMs: null, reason: "quote:invalid_timestamp" };
  }
}

/**
 * Check if spread is near the maximum allowed threshold
 * Blocks trades if spread > 90% of max
 */
export function checkSpreadNearMax(
  spread: number,
  maxSpread: number
): { passed: boolean; reason?: string } {
  const thresholdSpread = maxSpread * EXECUTION_CONFIG.SPREAD_NEAR_MAX_PCT;
  
  if (spread > thresholdSpread) {
    const pctOfMax = (spread / maxSpread) * 100;
    return {
      passed: false,
      reason: `spread:nearMax_${pctOfMax.toFixed(0)}%`,
    };
  }
  
  return { passed: true };
}

/**
 * Wait for order to fill with timeout
 * Returns order status after fill or timeout
 */
interface OrderResult {
  id: string;
  status: string;
  filled_avg_price?: string | null;
  filled_qty?: string | null;
}

export async function waitForFill(
  orderId: string,
  timeoutMs: number = EXECUTION_CONFIG.FILL_TIMEOUT_MS
): Promise<{ filled: boolean; order: OrderResult | null; timedOut: boolean; timeToFillMs: number }> {
  const startTime = Date.now();
  const pollInterval = EXECUTION_CONFIG.POLL_INTERVAL_MS;
  const maxAttempts = Math.ceil(timeoutMs / pollInterval);
  
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const order = await alpaca.getOrder(orderId);
      
      if (order.status === "filled") {
        const timeToFillMs = Date.now() - startTime;
        console.log(`[ExecutionQuality] Order ${orderId} FILLED in ${timeToFillMs}ms`);
        return { filled: true, order, timedOut: false, timeToFillMs };
      }
      
      if (order.status === "cancelled" || order.status === "rejected" || order.status === "expired") {
        console.log(`[ExecutionQuality] Order ${orderId} ${order.status}`);
        return { filled: false, order, timedOut: false, timeToFillMs: 0 };
      }
      
      // Still pending, wait and poll again
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    } catch (error) {
      console.log(`[ExecutionQuality] Error checking order ${orderId}: ${error}`);
      // Continue polling on error
      await new Promise(resolve => setTimeout(resolve, pollInterval));
    }
  }
  
  // Timeout reached
  const timeToFillMs = Date.now() - startTime;
  console.log(`[ExecutionQuality] Order ${orderId} TIMED OUT after ${timeToFillMs}ms`);
  return { filled: false, order: null, timedOut: true, timeToFillMs };
}

/**
 * Cancel unfilled order
 */
export async function cancelUnfilledOrder(orderId: string): Promise<boolean> {
  try {
    await alpaca.cancelOrder(orderId);
    console.log(`[ExecutionQuality] Cancelled unfilled order ${orderId}`);
    recordOrderCancelled();
    return true;
  } catch (error) {
    console.log(`[ExecutionQuality] Failed to cancel order ${orderId}: ${error}`);
    return false;
  }
}

/**
 * Log execution quality summary
 */
export function logExecutionSummary(): void {
  const report = getExecutionReport();
  console.log("[ExecutionQuality] ============================================");
  console.log("[ExecutionQuality] EXECUTION QUALITY SUMMARY");
  console.log("[ExecutionQuality] ============================================");
  console.log(`[ExecutionQuality] Fill Rate: ${report.fillRate}%`);
  console.log(`[ExecutionQuality] Cancel Rate: ${report.cancelRate}%`);
  console.log(`[ExecutionQuality] Timeout Rate: ${report.timeoutRate}%`);
  console.log(`[ExecutionQuality] Avg Time-to-Fill: ${report.avgTimeToFillMs}ms`);
  
  if (Object.keys(report.slippageByStrategy).length > 0) {
    console.log("[ExecutionQuality] Slippage by Strategy:");
    for (const [strategy, stats] of Object.entries(report.slippageByStrategy)) {
      console.log(`  ${strategy}: avg=${stats.avg}bps, median=${stats.median}bps, worst=${stats.worst}bps`);
    }
  }
  
  console.log("[ExecutionQuality] ============================================");
}
