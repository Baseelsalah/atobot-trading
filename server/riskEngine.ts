/**
 * P3 Risk Engine v1 - ATR-based position sizing + adaptive stops
 * 
 * Features:
 * 1. ATR-based position sizing (risk per trade, not fixed %)
 * 2. Adaptive bracket stops/TP based on ATR + time-of-day
 * 3. Strategy/regime kill-switch using P2 realized stats
 * 4. Daily safety controls (max trades, max consecutive losses)
 */

import { getEasternTime } from "./timezone";
import * as tradeLifecycle from "./tradeLifecycle";
import { recordSkip } from "./skipCounters";

// P3: Flag to track if callback is registered
let callbackRegistered = false;

// P3 Risk Engine Configuration
export const RISK_ENGINE_CONFIG = {
  // Position Sizing
  RISK_PER_TRADE_PCT: 0.0025,    // 0.25% of equity risked per trade (paper safe)
  MAX_POSITION_PCT: 0.075,       // 7.5% max position size cap
  MIN_NOTIONAL: 100,             // Minimum notional value ($100)
  
  // ATR-based stops
  ATR_STOP_MULTIPLIER: 1.2,      // Stop distance = k * ATR (default 1.2)
  REWARD_RISK_RATIO: 2.0,        // R:R for take profit
  MIN_STOP_PCT: 0.004,           // Hard floor: 0.4% minimum stop
  MAX_STOP_PCT: 0.04,            // Hard cap: 4% maximum stop
  
  // Time-of-day modifiers
  EARLY_WINDOW_MINUTES: 30,      // First 30 min after 9:30 AM
  EARLY_STOP_MULTIPLIER: 1.25,   // Widen stops in early volatility
  
  // Kill-switch thresholds
  KILLSWITCH_MIN_TRADES: 10,     // Min trades to evaluate bucket
  KILLSWITCH_MIN_EXPECTANCY: 0,  // Negative expectancy triggers
  KILLSWITCH_MIN_PROFIT_FACTOR: 0.9,
  KILLSWITCH_ROLLING_WINDOW: 20, // Last N closed trades
  KILLSWITCH_COOLDOWN_HOURS: 4,  // Disabled for this many hours
  
  // Daily safety controls
  MAX_TRADES_PER_DAY: 6,
  MAX_CONSECUTIVE_LOSSES: 3,
};

// Track kill-switch state
interface KillSwitchBucket {
  name: string;
  trades: number;
  wins: number;
  losses: number;
  totalPnl: number;
  disabledUntil: Date | null;
}

const killSwitchBuckets: Map<string, KillSwitchBucket> = new Map();
let consecutiveLosses = 0;
let tradesExecutedToday = 0;
let dailySafetyTriggered = false;
let dailySafetyReason = "";

/**
 * Calculate ATR-based position size
 * Returns { qty, stopDistance, stopPrice, takeProfitPrice, notional, reasoning }
 */
export function calculateATRPositionSize(
  symbol: string,
  entryPrice: number,
  atr: number,
  equity: number,
  side: "buy" | "sell" = "buy"
): {
  qty: number;
  stopDistance: number;
  stopPrice: number;
  takeProfitPrice: number;
  notional: number;
  riskAmount: number;
  reasoning: string;
} {
  const config = RISK_ENGINE_CONFIG;
  const et = getEasternTime();
  
  // Calculate base stop distance from ATR
  let stopMultiplier = config.ATR_STOP_MULTIPLIER;
  
  // Time-of-day adjustment: widen stops in first 30 minutes
  const minutesSinceOpen = (et.hour - 9) * 60 + et.minute - 30;
  if (minutesSinceOpen >= 0 && minutesSinceOpen < config.EARLY_WINDOW_MINUTES) {
    stopMultiplier *= config.EARLY_STOP_MULTIPLIER;
  }
  
  let stopDistance = atr * stopMultiplier;
  
  // Enforce hard caps on stop distance as percentage of price
  const stopPct = stopDistance / entryPrice;
  if (stopPct < config.MIN_STOP_PCT) {
    stopDistance = entryPrice * config.MIN_STOP_PCT;
  } else if (stopPct > config.MAX_STOP_PCT) {
    stopDistance = entryPrice * config.MAX_STOP_PCT;
  }
  
  // Calculate risk amount
  const riskAmount = equity * config.RISK_PER_TRADE_PCT;
  
  // Calculate position size: qty = riskAmount / stopDistance
  let qty = Math.floor(riskAmount / stopDistance);
  qty = Math.max(1, qty); // At least 1 share
  
  // Calculate notional and enforce max position cap
  let notional = qty * entryPrice;
  const maxNotional = equity * config.MAX_POSITION_PCT;
  
  if (notional > maxNotional) {
    qty = Math.floor(maxNotional / entryPrice);
    qty = Math.max(1, qty);
    notional = qty * entryPrice;
  }
  
  // Enforce minimum notional
  if (notional < config.MIN_NOTIONAL && qty === 1) {
    // Keep 1 share even if below min notional (can't buy fractional)
  }
  
  // Calculate stop and take profit prices
  let stopPrice: number;
  let takeProfitPrice: number;
  
  if (side === "buy") {
    stopPrice = entryPrice - stopDistance;
    takeProfitPrice = entryPrice + (stopDistance * config.REWARD_RISK_RATIO);
  } else {
    stopPrice = entryPrice + stopDistance;
    takeProfitPrice = entryPrice - (stopDistance * config.REWARD_RISK_RATIO);
  }
  
  const actualRisk = qty * stopDistance;
  const reasoning = `ATR=${atr.toFixed(3)} stop_mult=${stopMultiplier.toFixed(2)} ` +
    `stop_dist=$${stopDistance.toFixed(2)} (${(stopDistance/entryPrice*100).toFixed(2)}%) ` +
    `risk$=${riskAmount.toFixed(2)} qty=${qty} notional=$${notional.toFixed(2)} ` +
    `actual_risk$=${actualRisk.toFixed(2)}`;
  
  console.log(`[RISK_ENGINE] ${symbol}: ${reasoning}`);
  
  return {
    qty,
    stopDistance,
    stopPrice,
    takeProfitPrice,
    notional,
    riskAmount: actualRisk,
    reasoning,
  };
}

/**
 * Check if trading is allowed based on daily safety controls
 * Returns { allowed: boolean, reason: string }
 */
export function checkDailySafetyControls(): { allowed: boolean; reason: string } {
  if (dailySafetyTriggered) {
    return { allowed: false, reason: dailySafetyReason };
  }
  
  if (tradesExecutedToday >= RISK_ENGINE_CONFIG.MAX_TRADES_PER_DAY) {
    dailySafetyTriggered = true;
    dailySafetyReason = `MAX_TRADES_PER_DAY (${RISK_ENGINE_CONFIG.MAX_TRADES_PER_DAY}) reached`;
    console.log(`[RISK_ENGINE] DAILY SAFETY TRIGGERED: ${dailySafetyReason}`);
    return { allowed: false, reason: dailySafetyReason };
  }
  
  if (consecutiveLosses >= RISK_ENGINE_CONFIG.MAX_CONSECUTIVE_LOSSES) {
    dailySafetyTriggered = true;
    dailySafetyReason = `MAX_CONSECUTIVE_LOSSES (${RISK_ENGINE_CONFIG.MAX_CONSECUTIVE_LOSSES}) reached`;
    console.log(`[RISK_ENGINE] DAILY SAFETY TRIGGERED: ${dailySafetyReason}`);
    return { allowed: false, reason: dailySafetyReason };
  }
  
  return { allowed: true, reason: "ok" };
}

/**
 * Record a trade result for kill-switch tracking
 */
export function recordTradeResult(
  strategy: string,
  regime: "bullish" | "bearish" | "chop",
  pnl: number,
  isWin: boolean
): void {
  tradesExecutedToday++;
  
  if (isWin) {
    consecutiveLosses = 0;
  } else {
    consecutiveLosses++;
  }
  
  // Update strategy bucket
  updateKillSwitchBucket(`strategy:${strategy}`, pnl, isWin);
  
  // Update regime bucket
  updateKillSwitchBucket(`regime:${regime}`, pnl, isWin);
}

function updateKillSwitchBucket(name: string, pnl: number, isWin: boolean): void {
  let bucket = killSwitchBuckets.get(name);
  if (!bucket) {
    bucket = {
      name,
      trades: 0,
      wins: 0,
      losses: 0,
      totalPnl: 0,
      disabledUntil: null,
    };
    killSwitchBuckets.set(name, bucket);
  }
  
  bucket.trades++;
  if (isWin) {
    bucket.wins++;
  } else {
    bucket.losses++;
  }
  bucket.totalPnl += pnl;
  
  // Check kill-switch conditions
  if (bucket.trades >= RISK_ENGINE_CONFIG.KILLSWITCH_MIN_TRADES) {
    const expectancy = bucket.totalPnl / bucket.trades;
    const profitFactor = bucket.wins > 0 && bucket.losses > 0 
      ? (bucket.wins / bucket.losses)
      : bucket.wins > 0 ? 999 : 0;
    
    if (expectancy < RISK_ENGINE_CONFIG.KILLSWITCH_MIN_EXPECTANCY && 
        profitFactor < RISK_ENGINE_CONFIG.KILLSWITCH_MIN_PROFIT_FACTOR) {
      const cooldownMs = RISK_ENGINE_CONFIG.KILLSWITCH_COOLDOWN_HOURS * 60 * 60 * 1000;
      bucket.disabledUntil = new Date(Date.now() + cooldownMs);
      console.log(`[KILL_SWITCH] ${name} DISABLED until ${bucket.disabledUntil.toISOString()} ` +
        `(trades=${bucket.trades} expectancy=$${expectancy.toFixed(2)} pf=${profitFactor.toFixed(2)})`);
    }
  }
}

/**
 * Check if a strategy/regime bucket is currently disabled
 */
export function isKillSwitchTriggered(
  strategy: string | null,
  regime: "bullish" | "bearish" | "chop" | null
): { blocked: boolean; reason: string } {
  const now = new Date();
  
  // Check strategy bucket
  if (strategy) {
    const stratBucket = killSwitchBuckets.get(`strategy:${strategy}`);
    if (stratBucket?.disabledUntil && now < stratBucket.disabledUntil) {
      return { blocked: true, reason: `killswitch:strategy_${strategy}` };
    }
  }
  
  // Check regime bucket
  if (regime) {
    const regimeBucket = killSwitchBuckets.get(`regime:${regime}`);
    if (regimeBucket?.disabledUntil && now < regimeBucket.disabledUntil) {
      return { blocked: true, reason: `killswitch:regime_${regime}` };
    }
  }
  
  return { blocked: false, reason: "ok" };
}

/**
 * Pre-trade check combining all P3 safety checks
 * Call before executing any trade
 */
export function preTradeCheck(
  strategy: string | null,
  regime: "bullish" | "bearish" | "chop" | null
): { allowed: boolean; skipReason: string | null } {
  // Check daily safety controls
  const dailyCheck = checkDailySafetyControls();
  if (!dailyCheck.allowed) {
    recordSkip("KILL_THRESHOLD_HIT" as any);
    return { allowed: false, skipReason: `DAILY_SAFETY:${dailyCheck.reason}` };
  }
  
  // Check kill-switch
  const killCheck = isKillSwitchTriggered(strategy, regime);
  if (killCheck.blocked) {
    recordSkip("KILL_THRESHOLD_HIT" as any);
    return { allowed: false, skipReason: killCheck.reason };
  }
  
  return { allowed: true, skipReason: null };
}

/**
 * Initialize P3 risk engine and register callback with tradeLifecycle
 * Called once at bot startup
 */
export function initializeRiskEngine(): void {
  // Reset all daily counters first (critical for day rollover)
  resetDaily();
  
  if (!callbackRegistered) {
    // Register callback to receive trade outcomes as they close
    // P4: Now receives real regime from trade metadata
    tradeLifecycle.setOnTradeCloseCallback((strategy, pnl, isWin, regime) => {
      // Map regime label to riskEngine regime format
      const engineRegime = regime === "bull" ? "bullish" : regime === "bear" ? "bearish" : "chop";
      recordTradeResult(strategy, engineRegime, pnl, isWin);
    });
    callbackRegistered = true;
    console.log("[RISK_ENGINE] Callback registered with trade lifecycle (P4: real regime)");
  }
  
  // Rehydrate from today's completed trades (rebuilds counters from actual data)
  rehydrateFromCompletedTrades();
}

/**
 * Rehydrate kill-switch stats from today's completed trades
 * Called after resetDaily() to rebuild counters from actual data
 */
function rehydrateFromCompletedTrades(): void {
  const completedTrades = tradeLifecycle.getTodayCompletedTrades();
  
  if (completedTrades.length === 0) {
    return; // Nothing to rehydrate
  }
  
  // Sort by close time to track consecutive losses correctly
  const sortedTrades = [...completedTrades].sort((a, b) => {
    const aTime = a.exitFilledAt ? new Date(a.exitFilledAt).getTime() : 0;
    const bTime = b.exitFilledAt ? new Date(b.exitFilledAt).getTime() : 0;
    return aTime - bTime;
  });
  
  for (const trade of sortedTrades) {
    if (trade.status !== "closed" || trade.realizedPnl === null) continue;
    
    const isWin = trade.realizedPnl >= 0;
    const strategy = trade.strategy || "unknown";
    
    // Silently accumulate stats (don't use recordTradeResult to avoid duplicate logs)
    tradesExecutedToday++;
    
    if (isWin) {
      consecutiveLosses = 0;
    } else {
      consecutiveLosses++;
    }
    
    // P4: Use real regime from trade metadata if available
    const tradeRegime = trade.metadata?.regime || "chop";
    const engineRegime = tradeRegime === "bull" ? "bullish" : tradeRegime === "bear" ? "bearish" : "chop";
    
    updateKillSwitchBucket(`strategy:${strategy}`, trade.realizedPnl, isWin);
    updateKillSwitchBucket(`regime:${engineRegime}`, trade.realizedPnl, isWin);
  }
  
  console.log(`[RISK_ENGINE] Rehydrated: ${tradesExecutedToday} trades, ${consecutiveLosses} consecutive losses`);
}

/**
 * Reset daily state (call at start of trading day)
 */
export function resetDaily(): void {
  killSwitchBuckets.clear();
  consecutiveLosses = 0;
  tradesExecutedToday = 0;
  dailySafetyTriggered = false;
  dailySafetyReason = "";
  console.log("[RISK_ENGINE] Daily reset complete");
}

/**
 * Get current risk engine status for API/dashboard
 */
export function getRiskEngineStatus() {
  const buckets: Array<{
    name: string;
    trades: number;
    wins: number;
    losses: number;
    pnl: number;
    disabled: boolean;
    disabledUntil: string | null;
  }> = [];
  
  for (const [name, bucket] of Array.from(killSwitchBuckets.entries())) {
    buckets.push({
      name,
      trades: bucket.trades,
      wins: bucket.wins,
      losses: bucket.losses,
      pnl: bucket.totalPnl,
      disabled: bucket.disabledUntil ? new Date() < bucket.disabledUntil : false,
      disabledUntil: bucket.disabledUntil?.toISOString() || null,
    });
  }
  
  return {
    config: {
      riskPerTradePct: RISK_ENGINE_CONFIG.RISK_PER_TRADE_PCT * 100,
      maxPositionPct: RISK_ENGINE_CONFIG.MAX_POSITION_PCT * 100,
      atrStopMultiplier: RISK_ENGINE_CONFIG.ATR_STOP_MULTIPLIER,
      rewardRiskRatio: RISK_ENGINE_CONFIG.REWARD_RISK_RATIO,
      maxTradesPerDay: RISK_ENGINE_CONFIG.MAX_TRADES_PER_DAY,
      maxConsecutiveLosses: RISK_ENGINE_CONFIG.MAX_CONSECUTIVE_LOSSES,
    },
    dailyState: {
      tradesExecutedToday,
      consecutiveLosses,
      dailySafetyTriggered,
      dailySafetyReason,
    },
    killSwitchBuckets: buckets,
  };
}
