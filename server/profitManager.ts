import * as alpaca from "./alpaca";
import { storage } from "./storage";
import { getEasternTime, toMinutesSinceMidnight } from "./timezone";
import {
  loadDailyState,
  updatePersistentPnL,
  markResetCompleted,
} from "./persistentState.js";

// REMOVED: Daily profit goal forcing logic - don't size up or trade more because we're "behind goal"
// Keep only hard safety limits (max daily loss)
const MAX_DAILY_LOSS = 50000; // Aligned with storage.ts maxDailyLoss — raised for 700% leverage (avg loss ~$4K)

// DISPLAY ONLY: Daily profit goal for UI display (not used for forcing trades)
const DAILY_PROFIT_GOAL = 3000; // Display target only - no behavioral impact

interface ProfitGoalState {
  dailyGoal: number;
  currentProfit: number;
  realizedProfit: number;
  unrealizedProfit: number;
  progressPercent: number;
  goalMet: boolean;
  tradesNeeded: number;
  avgProfitPerTrade: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  expectancy: number;
  profitFactor: number;
}

interface PositionTracker {
  symbol: string;
  entryPrice: number;
  quantity: number;
  entryTime: Date;
  peakPrice: number;
  peakProfit: number;
  breakEvenStop: number | null;
  trailingStop: number | null;
  partialExitDone: boolean;
  tier1Exit: boolean;
  tier2Exit: boolean;
  tradeId: string | null;
}

interface TradePerformance {
  totalTrades: number;
  wins: number;
  losses: number;
  totalProfit: number;
  totalLoss: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  expectancy: number;
  profitFactor: number;
  largestWin: number;
  largestLoss: number;
  consecutiveWins: number;
  consecutiveLosses: number;
}

const positionTrackers: Map<string, PositionTracker> = new Map();
let dailyRealizedPL = 0;
let todayPerformance: TradePerformance = createEmptyPerformance();
let lastTradingDate: string | null = null;

function getTradingDateET(): string {
  return getEasternTime().dateString;
}

export function checkAndResetDaily(): boolean {
  const currentDate = getTradingDateET();

  if (lastTradingDate !== currentDate) {
    console.log(`[ProfitManager] New trading day detected: ${currentDate} (previous: ${lastTradingDate || 'none'})`);

    if (lastTradingDate !== null) {
      console.log(`[ProfitManager] Previous day summary: Realized P/L: $${dailyRealizedPL.toFixed(2)}, Trades: ${todayPerformance.totalTrades}, Win Rate: ${todayPerformance.winRate.toFixed(1)}%`);
    }

    // Try to load persisted state for today
    const persistedState = loadDailyState();

    if (persistedState) {
      // Restore state from persistent storage
      dailyRealizedPL = persistedState.dailyRealizedPL;
      console.log(`[ProfitManager] Restored from persistent state: Realized P/L: $${dailyRealizedPL.toFixed(2)}`);
    } else {
      // New day - reset everything
      dailyRealizedPL = 0;
      markResetCompleted();
    }

    todayPerformance = createEmptyPerformance();
    positionTrackers.clear();
    lastTradingDate = currentDate;

    console.log("[ProfitManager] Daily tracking reset for new trading day");
    return true;
  }

  return false;
}

function createEmptyPerformance(): TradePerformance {
  return {
    totalTrades: 0,
    wins: 0,
    losses: 0,
    totalProfit: 0,
    totalLoss: 0,
    winRate: 0,
    avgWin: 0,
    avgLoss: 0,
    expectancy: 0,
    profitFactor: 0,
    largestWin: 0,
    largestLoss: 0,
    consecutiveWins: 0,
    consecutiveLosses: 0,
  };
}

export function getDailyGoal(): number {
  return DAILY_PROFIT_GOAL;
}

export interface HungerState {
  hungerLevel: "starving" | "hungry" | "fed" | "satisfied" | "full";
  urgency: number;
  aggressiveness: number;
  positionSizeMultiplier: number;
  thresholdReduction: number;
  message: string;
  profitNeeded: number;
  timeRemainingHours: number;
  profitPerHourNeeded: number;
}

export interface WarriorState {
  mode: "hunt" | "attack" | "defend" | "celebrate" | "regroup";
  killCount: number;
  conquestStreak: number;
  biggestKill: number;
  missionActive: boolean;
  currentMission: string | null;
  warCry: string;
  momentumScore: number;
  battleReadiness: number;
  failuresBeforeCooldown: number;
  coolingOff: boolean;
}

let warriorState: WarriorState = {
  mode: "hunt",
  killCount: 0,
  conquestStreak: 0,
  biggestKill: 0,
  missionActive: false,
  currentMission: null,
  warCry: "THE HUNT BEGINS!",
  momentumScore: 50,
  battleReadiness: 100,
  failuresBeforeCooldown: 0,
  coolingOff: false,
};

export function getWarriorState(): WarriorState {
  return { ...warriorState };
}

export function recordConquest(symbol: string, profit: number): void {
  warriorState.killCount++;
  warriorState.conquestStreak++;
  warriorState.momentumScore = Math.min(100, warriorState.momentumScore + 15);
  warriorState.battleReadiness = Math.min(100, warriorState.battleReadiness + 10);
  warriorState.failuresBeforeCooldown = 0;
  warriorState.coolingOff = false;
  
  if (profit > warriorState.biggestKill) {
    warriorState.biggestKill = profit;
    warriorState.warCry = `MASSIVE KILL! $${profit.toFixed(0)} CONQUERED!`;
  } else {
    warriorState.warCry = `KILL #${warriorState.killCount}! +$${profit.toFixed(0)} | Streak: ${warriorState.conquestStreak}!`;
  }
  
  if (warriorState.conquestStreak >= 3) {
    warriorState.mode = "attack";
    warriorState.warCry = `ON FIRE! ${warriorState.conquestStreak} KILLS IN A ROW! DOMINATE!`;
  } else {
    warriorState.mode = "hunt";
  }
  
  if (warriorState.missionActive) {
    completeMission(true);
  }
  
  console.log(`[WARRIOR] ${warriorState.warCry}`);
}

export function recordDefeat(symbol: string, loss: number): void {
  warriorState.conquestStreak = 0;
  warriorState.failuresBeforeCooldown++;
  warriorState.momentumScore = Math.max(20, warriorState.momentumScore - 20);
  warriorState.battleReadiness = Math.max(50, warriorState.battleReadiness - 15);
  
  if (warriorState.failuresBeforeCooldown >= 2) {
    warriorState.mode = "regroup";
    warriorState.coolingOff = true;
    warriorState.warCry = "REGROUP! Two defeats. Recalibrating for next attack...";
    if (warriorState.missionActive) {
      completeMission(false);
    }
  } else {
    warriorState.mode = "defend";
    warriorState.warCry = `LOSS! -$${Math.abs(loss).toFixed(0)}. Warriors recover. Next trade is the comeback!`;
  }
  
  console.log(`[WARRIOR] ${warriorState.warCry}`);
}

export function startMission(mission: string): void {
  warriorState.missionActive = true;
  warriorState.currentMission = mission;
  warriorState.mode = "attack";
  warriorState.warCry = `MISSION ACTIVATED: ${mission}`;
  console.log(`[WARRIOR] ${warriorState.warCry}`);
}

export function completeMission(success: boolean): void {
  if (success) {
    warriorState.warCry = `MISSION COMPLETE: ${warriorState.currentMission}! VICTORY!`;
    warriorState.mode = "celebrate";
  } else {
    warriorState.warCry = `Mission failed. Regrouping for next assault.`;
    warriorState.mode = "regroup";
  }
  warriorState.missionActive = false;
  warriorState.currentMission = null;
  console.log(`[WARRIOR] ${warriorState.warCry}`);
}

export function resetWarriorDaily(): void {
  warriorState = {
    mode: "hunt",
    killCount: 0,
    conquestStreak: 0,
    biggestKill: 0,
    missionActive: false,
    currentMission: null,
    warCry: "NEW DAY! THE HUNT FOR $3,000 BEGINS!",
    momentumScore: 50,
    battleReadiness: 100,
    failuresBeforeCooldown: 0,
    coolingOff: false,
  };
  console.log(`[WARRIOR] ${warriorState.warCry}`);
}

/**
 * SIMPLIFIED: No more "hunger" forcing behavior
 * Returns neutral state - position sizing and thresholds are NOT modified based on P&L
 * Only hard safety limits apply (max daily loss)
 */
export async function getHungerState(): Promise<HungerState> {
  checkAndResetDaily();
  
  const positions = await storage.getPositions();
  const unrealizedProfit = positions.reduce((sum, p) => sum + p.unrealizedPL, 0);
  const currentProfit = dailyRealizedPL + unrealizedProfit;
  
  const et = getEasternTime();
  const currentMinutes = toMinutesSinceMidnight(et.hour, et.minute);
  
  // FORT KNOX SCHEDULE: 9:35 AM - 2:45 PM ET
  const tradingStartMinutes = toMinutesSinceMidnight(9, 35);
  const tradingEndMinutes = toMinutesSinceMidnight(14, 45);
  
  const totalTradingMinutes = tradingEndMinutes - tradingStartMinutes;
  const elapsedMinutes = Math.max(0, currentMinutes - tradingStartMinutes);
  const remainingMinutes = Math.max(0, totalTradingMinutes - elapsedMinutes);
  const timeRemainingHours = remainingMinutes / 60;
  
  // Check hard safety limit - max daily loss
  const dailyLoss = currentProfit < 0 ? Math.abs(currentProfit) : 0;
  const atDailyLossLimit = dailyLoss >= MAX_DAILY_LOSS;
  
  // PRO: SMART POSITION SIZING - Reduce after losses, increase after wins
  // This is the OPPOSITE of hunger-driven sizing (which was dangerous)
  let smartMultiplier = 1.0;

  // After 2+ consecutive losses: REDUCE size by 30%
  if (warriorState.conquestStreak <= -2) {
    smartMultiplier = 0.7;
  }
  // After 1 loss: REDUCE size by 15%
  else if (warriorState.conquestStreak === -1) {
    smartMultiplier = 0.85;
  }
  // After 3+ consecutive wins: INCREASE size by 20%
  else if (warriorState.conquestStreak >= 3) {
    smartMultiplier = 1.2;
  }
  // After 2 consecutive wins: INCREASE size by 10%
  else if (warriorState.conquestStreak === 2) {
    smartMultiplier = 1.1;
  }

  // NEUTRAL STATE: No forcing based on profit goals
  // Position sizing adjusts based on win/loss streaks (smart risk management)
  return {
    hungerLevel: atDailyLossLimit ? "full" : "fed", // "full" means stop trading
    urgency: 0,
    aggressiveness: 0.5, // Neutral - no aggressive sizing
    positionSizeMultiplier: smartMultiplier, // PRO: Smart sizing based on streaks
    thresholdReduction: 0, // NO THRESHOLD REDUCTION - maintain standards
    message: atDailyLossLimit
      ? `HARD STOP: Daily loss limit reached ($${dailyLoss.toFixed(0)}/$${MAX_DAILY_LOSS}). Trading paused.`
      : `P&L: $${currentProfit.toFixed(0)} | Smart sizing: ${(smartMultiplier * 100).toFixed(0)}% (streak: ${warriorState.conquestStreak})`,
    profitNeeded: 0, // No goal forcing
    timeRemainingHours,
    profitPerHourNeeded: 0,
  };
}

export async function getProfitGoalState(): Promise<ProfitGoalState> {
  checkAndResetDaily();
  
  const positions = await storage.getPositions();
  const unrealizedProfit = positions.reduce((sum, p) => sum + p.unrealizedPL, 0);
  const currentProfit = dailyRealizedPL + unrealizedProfit;
  const progressPercent = Math.min(100, (currentProfit / DAILY_PROFIT_GOAL) * 100);
  
  const tradesRemaining = Math.max(0, 8 - todayPerformance.totalTrades);
  const profitNeeded = DAILY_PROFIT_GOAL - currentProfit;
  const avgProfitPerTrade = tradesRemaining > 0 ? profitNeeded / tradesRemaining : 0;
  
  return {
    dailyGoal: DAILY_PROFIT_GOAL,
    currentProfit,
    realizedProfit: dailyRealizedPL,
    unrealizedProfit,
    progressPercent: Math.max(0, progressPercent),
    goalMet: currentProfit >= DAILY_PROFIT_GOAL,
    tradesNeeded: tradesRemaining,
    avgProfitPerTrade,
    winRate: todayPerformance.winRate,
    avgWin: todayPerformance.avgWin,
    avgLoss: todayPerformance.avgLoss,
    expectancy: todayPerformance.expectancy,
    profitFactor: todayPerformance.profitFactor,
  };
}

export function getPerformance(): TradePerformance {
  return { ...todayPerformance };
}

export function recordTradeResult(symbol: string, profit: number, isWin: boolean): void {
  dailyRealizedPL += profit;

  // Persist to disk after every trade
  updatePersistentPnL(0, dailyRealizedPL); // dailyPnL will be synced from dayTraderConfig

  todayPerformance.totalTrades++;
  
  if (isWin) {
    todayPerformance.wins++;
    todayPerformance.totalProfit += profit;
    todayPerformance.consecutiveWins++;
    todayPerformance.consecutiveLosses = 0;
    if (profit > todayPerformance.largestWin) {
      todayPerformance.largestWin = profit;
    }
  } else {
    todayPerformance.losses++;
    todayPerformance.totalLoss += Math.abs(profit);
    todayPerformance.consecutiveLosses++;
    todayPerformance.consecutiveWins = 0;
    if (Math.abs(profit) > todayPerformance.largestLoss) {
      todayPerformance.largestLoss = Math.abs(profit);
    }
  }
  
  todayPerformance.winRate = todayPerformance.wins / todayPerformance.totalTrades * 100;
  todayPerformance.avgWin = todayPerformance.wins > 0 ? todayPerformance.totalProfit / todayPerformance.wins : 0;
  todayPerformance.avgLoss = todayPerformance.losses > 0 ? todayPerformance.totalLoss / todayPerformance.losses : 0;
  
  if (todayPerformance.avgLoss > 0) {
    const winRateDecimal = todayPerformance.winRate / 100;
    todayPerformance.expectancy = (winRateDecimal * todayPerformance.avgWin) - ((1 - winRateDecimal) * todayPerformance.avgLoss);
  }
  
  todayPerformance.profitFactor = todayPerformance.totalLoss > 0 
    ? todayPerformance.totalProfit / todayPerformance.totalLoss 
    : todayPerformance.totalProfit > 0 ? Infinity : 0;
  
  console.log(`[ProfitManager] Trade recorded: ${symbol} ${isWin ? "WIN" : "LOSS"} $${profit.toFixed(2)}`);
  console.log(`[ProfitManager] Daily P/L: $${dailyRealizedPL.toFixed(2)} | Win Rate: ${todayPerformance.winRate.toFixed(1)}% | Expectancy: $${todayPerformance.expectancy.toFixed(2)}`);
}

export function trackPosition(symbol: string, entryPrice: number, quantity: number, tradeId?: string): void {
  const tradeIdLog = tradeId ? ` trade_id=${tradeId}` : "";
  positionTrackers.set(symbol, {
    symbol,
    entryPrice,
    quantity,
    entryTime: new Date(),
    peakPrice: entryPrice,
    peakProfit: 0,
    breakEvenStop: null,
    trailingStop: null,
    partialExitDone: false,
    tier1Exit: false,
    tier2Exit: false,
    tradeId: tradeId || null,
  });
  console.log(`[ProfitManager] Tracking position: ${symbol} @ $${entryPrice}${tradeIdLog}`);
}

export function getTradeId(symbol: string): string | null {
  const tracker = positionTrackers.get(symbol);
  return tracker?.tradeId || null;
}

export function updatePositionTracker(symbol: string, currentPrice: number): PositionTracker | null {
  const tracker = positionTrackers.get(symbol);
  if (!tracker) return null;
  
  const profitPercent = ((currentPrice - tracker.entryPrice) / tracker.entryPrice) * 100;
  const profitDollars = (currentPrice - tracker.entryPrice) * tracker.quantity;
  
  if (currentPrice > tracker.peakPrice) {
    tracker.peakPrice = currentPrice;
    tracker.peakProfit = profitDollars;
  }
  
  // PRO: Activate breakeven stop at 0.3% gain (was 1.0%) - faster profit protection
  if (profitPercent >= 0.3 && !tracker.breakEvenStop) {
    tracker.breakEvenStop = tracker.entryPrice * 1.0005; // Set stop at entry + 0.05%
    console.log(`[ProfitManager] ${symbol}: Breakeven stop activated @ $${tracker.breakEvenStop.toFixed(2)} (PRO: early protection)`);
  }
  
  if (profitPercent >= 1.5 && !tracker.trailingStop) {
    tracker.trailingStop = currentPrice * 0.995;
  } else if (tracker.trailingStop && currentPrice > tracker.peakPrice * 0.995) {
    tracker.trailingStop = currentPrice * 0.995;
  }
  
  return tracker;
}

export interface ExitSignal {
  shouldExit: boolean;
  exitType: "none" | "stop_loss" | "breakeven" | "trailing" | "take_profit" | "tier1" | "tier2" | "tier3";
  exitQuantity: number;
  reason: string;
  profitPercent: number;
}

export function checkExitSignals(
  symbol: string, 
  currentPrice: number, 
  quantity: number,
  stopLossPercent: number = 1.0,
  takeProfitPercent: number = 3.0
): ExitSignal {
  const tracker = positionTrackers.get(symbol);
  
  if (!tracker) {
    trackPosition(symbol, currentPrice, quantity);
    return { shouldExit: false, exitType: "none", exitQuantity: 0, reason: "Position tracking started", profitPercent: 0 };
  }
  
  updatePositionTracker(symbol, currentPrice);
  
  const profitPercent = ((currentPrice - tracker.entryPrice) / tracker.entryPrice) * 100;
  const profitDollars = (currentPrice - tracker.entryPrice) * tracker.quantity;
  
  if (profitPercent <= -stopLossPercent) {
    return {
      shouldExit: true,
      exitType: "stop_loss",
      exitQuantity: quantity,
      reason: `Stop loss hit at ${profitPercent.toFixed(2)}%`,
      profitPercent,
    };
  }
  
  if (tracker.breakEvenStop && currentPrice <= tracker.breakEvenStop && profitPercent < 0.5) {
    return {
      shouldExit: true,
      exitType: "breakeven",
      exitQuantity: quantity,
      reason: `Breakeven stop triggered - protecting capital`,
      profitPercent,
    };
  }
  
  if (tracker.trailingStop && currentPrice <= tracker.trailingStop) {
    return {
      shouldExit: true,
      exitType: "trailing",
      exitQuantity: quantity,
      reason: `Trailing stop hit at ${profitPercent.toFixed(2)}% (peak was ${((tracker.peakPrice - tracker.entryPrice) / tracker.entryPrice * 100).toFixed(2)}%)`,
      profitPercent,
    };
  }
  
  if (profitPercent >= 1.0 && !tracker.tier1Exit && quantity >= 2) {
    tracker.tier1Exit = true;
    const exitQty = Math.max(1, Math.floor(quantity * 0.33));
    return {
      shouldExit: true,
      exitType: "tier1",
      exitQuantity: exitQty,
      reason: `Tier 1 exit: Taking 33% profit at +${profitPercent.toFixed(2)}%`,
      profitPercent,
    };
  }
  
  if (profitPercent >= 2.0 && !tracker.tier2Exit && quantity >= 2) {
    tracker.tier2Exit = true;
    const exitQty = Math.max(1, Math.floor(quantity * 0.33));
    return {
      shouldExit: true,
      exitType: "tier2",
      exitQuantity: exitQty,
      reason: `Tier 2 exit: Taking 33% profit at +${profitPercent.toFixed(2)}%`,
      profitPercent,
    };
  }
  
  if (profitPercent >= takeProfitPercent) {
    return {
      shouldExit: true,
      exitType: "tier3",
      exitQuantity: quantity,
      reason: `Full take profit at +${profitPercent.toFixed(2)}%`,
      profitPercent,
    };
  }
  
  return { shouldExit: false, exitType: "none", exitQuantity: 0, reason: "Holding position", profitPercent };
}

/**
 * SIMPLIFIED: Fixed position sizing - no goal-based scaling
 * Uses consistent position sizing based on portfolio value and risk limits
 */
export function calculateGoalBasedPositionSize(
  currentPrice: number,
  portfolioValue: number,
  existingPositions: number,
  avgWinPercent: number = 2.0
): { shares: number; value: number; expectedProfit: number; reasoning: string } {
  // FIXED position sizing - no goal-based scaling
  // Use 5% of portfolio per position, max 4 positions (20% total exposure)
  const positionPercent = 0.05; // 5% per position
  const maxPositionValue = 25000; // Hard cap
  
  const basePositionValue = portfolioValue * positionPercent;
  const maxFromConcentration = portfolioValue / Math.max(1, existingPositions + 1);
  
  const positionValue = Math.min(basePositionValue, maxFromConcentration, maxPositionValue);
  const finalPositionValue = Math.max(1000, positionValue); // Minimum $1000
  
  const shares = Math.max(1, Math.floor(finalPositionValue / currentPrice));
  const actualValue = shares * currentPrice;
  const expectedProfit = actualValue * (avgWinPercent / 100) * 0.55; // Assume 55% win rate
  
  const reasoning = `FIXED SIZING: ${(positionPercent * 100).toFixed(0)}% of portfolio | ` +
    `Position: $${actualValue.toFixed(0)} (${shares} shares @ $${currentPrice.toFixed(2)})`;
  
  console.log(`[ProfitManager] Position sizing: ${reasoning}`);
  
  return { shares, value: actualValue, expectedProfit, reasoning };
}

/**
 * Check if trading should continue - only enforces hard safety limits
 * REMOVED: goal-met stopping (we don't force trades, so no goal to "meet")
 */
export async function shouldContinueTrading(): Promise<{ allowed: boolean; reason: string }> {
  checkAndResetDaily();
  const goalState = await getProfitGoalState();
  
  // HARD STOP: Daily loss limit reached
  if (goalState.currentProfit <= -MAX_DAILY_LOSS) {
    return { allowed: false, reason: `HARD STOP: Daily loss limit ($${MAX_DAILY_LOSS}) reached: $${goalState.currentProfit.toFixed(2)}` };
  }
  
  // SOFT LIMIT: Max daily trades - aligned with riskEngine MAX_TRADES_PER_DAY
  if (todayPerformance.totalTrades >= 20) {
    return { allowed: false, reason: `Max daily trades (20) reached` };
  }
  
  // SOFT LIMIT: Consecutive losses - aligned with riskEngine MAX_CONSECUTIVE_LOSSES
  if (todayPerformance.consecutiveLosses >= 4) {
    return { allowed: false, reason: `4 consecutive losses - take a break` };
  }
  
  // SOFT LIMIT: Negative expectancy 
  if (todayPerformance.expectancy < -50 && todayPerformance.totalTrades >= 3) {
    return { allowed: false, reason: `Negative expectancy ($${todayPerformance.expectancy.toFixed(2)}) - strategy not working today` };
  }
  
  return { allowed: true, reason: "Trading allowed" };
}

export function clearPositionTracker(symbol: string): void {
  positionTrackers.delete(symbol);
}

export function updateTrackerQuantity(symbol: string, soldQuantity: number): void {
  const tracker = positionTrackers.get(symbol);
  if (tracker) {
    tracker.quantity = Math.max(0, tracker.quantity - soldQuantity);
    console.log(`[ProfitManager] ${symbol}: Updated tracker quantity to ${tracker.quantity} after selling ${soldQuantity}`);
    
    if (tracker.quantity <= 0) {
      positionTrackers.delete(symbol);
      console.log(`[ProfitManager] ${symbol}: Position fully closed, tracker removed`);
    }
  }
}

export function resetDailyTracking(): void {
  dailyRealizedPL = 0;
  todayPerformance = createEmptyPerformance();
  positionTrackers.clear();
  resetWarriorDaily();
  console.log("[ProfitManager] Daily tracking reset for new trading day");
}

export function getDailyRealizedPL(): number {
  return dailyRealizedPL;
}

export function getPositionTracker(symbol: string): PositionTracker | undefined {
  return positionTrackers.get(symbol);
}
