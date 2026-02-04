import OpenAI from "openai";
import * as alpaca from "./alpaca";
import * as riskManager from "./riskManager";
import * as tradeValidator from "./tradeValidator";
import * as profitManager from "./profitManager";
import * as tradeBus from "./tradeIntelligenceBus";
import * as timeGuard from "./tradingTimeGuard";
import * as tradeAccounting from "./tradeAccounting";
import * as executionTrace from "./executionTrace";
import * as leaderLock from "./leaderLock";
import * as regimeState from "./regimeState";
import { getEasternTime, toEasternDateString } from "./timezone";
import { storage } from "./storage";
import { generateTradeId } from "./tradeId";
import type { TradingStrategy, TradeRecommendation, Position, BotSettings } from "@shared/schema";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});
const MODEL = "gpt-5";

interface AtoState {
  mood: "aggressive" | "conservative" | "neutral";
  currentFocus: string[];
  recentDecisions: TraderDecision[];
  dailyStats: DailyStats;
  tradingStyle: TradingStyle;
}

interface TraderDecision {
  timestamp: string;
  symbol: string;
  action: string;
  reason: string;
  outcome?: "success" | "failure" | "pending";
}

interface DailyStats {
  tradesExecuted: number;
  profitLoss: number;
  winCount: number;
  lossCount: number;
  bestTrade: string | null;
  worstTrade: string | null;
}

interface TradingStyle {
  preferredSetups: string[];
  avoidPatterns: string[];
  entryRules: string[];
  exitRules: string[];
  timePreferences: string[];
}

interface MarketRead {
  overallSentiment: "bullish" | "bearish" | "neutral";
  momentum: "strong" | "weak" | "mixed";
  volatility: "high" | "low" | "normal";
  keyLevels: Record<string, { support: number; resistance: number }>;
  hotSectors: string[];
  warnings: string[];
}

let atoState: AtoState = {
  mood: "neutral",
  currentFocus: [],
  recentDecisions: [],
  dailyStats: {
    tradesExecuted: 0,
    profitLoss: 0,
    winCount: 0,
    lossCount: 0,
    bestTrade: null,
    worstTrade: null,
  },
  tradingStyle: {
    preferredSetups: ["momentum breakout", "dip buying", "gap and go"],
    avoidPatterns: ["chasing extended moves", "revenge trading", "overtrading"],
    entryRules: ["wait for confirmation", "check volume", "verify trend"],
    exitRules: ["take partial profits", "trail stops", "honor stop losses"],
    timePreferences: ["first hour momentum", "midday consolidation plays", "power hour setups"],
  },
};

export function getAtoState(): AtoState {
  return { ...atoState };
}

export async function initializeAto(): Promise<void> {
  console.log("[Ato] Initializing day trader module...");
  
  const trades = await storage.getTrades();
  const todayET = getEasternTime().dateString; // Use Eastern Time for trading day
  const todayTrades = trades.filter(t => {
    if (!t.timestamp) return false;
    const tradeDate = toEasternDateString(new Date(t.timestamp));
    return tradeDate === todayET;
  });
  
  // For day trading, current unrealized P/L is what matters most for risk control
  // We need to stop losses before they compound further
  const positions = await storage.getPositions();
  const unrealizedPL = positions.reduce((sum, p) => sum + p.unrealizedPL, 0);
  
  // Count buy/sell trades for the day
  const buyCount = todayTrades.filter(t => t.side === "buy" && t.status === "filled").length;
  const sellCount = todayTrades.filter(t => t.side === "sell" && t.status === "filled").length;
  
  // Positions with positive P/L count as wins, negative as losses
  const winCount = positions.filter(p => p.unrealizedPL > 0).length;
  const lossCount = positions.filter(p => p.unrealizedPL < 0).length;
  
  // Track best/worst current positions
  const sortedPositions = [...positions].sort((a, b) => b.unrealizedPL - a.unrealizedPL);
  const bestPosition = sortedPositions[0];
  const worstPosition = sortedPositions[sortedPositions.length - 1];
  
  atoState.dailyStats = {
    tradesExecuted: buyCount + sellCount,
    profitLoss: unrealizedPL, // Current unrealized P/L is the key metric for day trading risk
    winCount,
    lossCount,
    bestTrade: bestPosition ? `${bestPosition.symbol}: ${bestPosition.unrealizedPL >= 0 ? "+" : ""}$${bestPosition.unrealizedPL.toFixed(2)}` : null,
    worstTrade: worstPosition && worstPosition.unrealizedPL < 0 ? `${worstPosition.symbol}: -$${Math.abs(worstPosition.unrealizedPL).toFixed(2)}` : null,
  };
  
  console.log(`[Ato] Loaded daily stats: ${atoState.dailyStats.tradesExecuted} trades, Unrealized P/L: $${atoState.dailyStats.profitLoss.toFixed(2)}`);
  
  await storage.createActivityLog({
    type: "system",
    action: "Ato Initialized",
    description: `Day trader ready. Today: ${atoState.dailyStats.tradesExecuted} trades, P/L: $${atoState.dailyStats.profitLoss.toFixed(2)}`,
  });
  
  console.log("[Ato] Day trader ready - analyzing market conditions...");
}

export async function readMarket(symbols: string[]): Promise<MarketRead> {
  console.log("[Ato] Reading market conditions like a day trader...");
  
  const positions = await storage.getPositions();
  const recentTrades = await storage.getTrades();
  
  const prompt = `You are Ato, a professional day trader analyzing the market.
Think like a real day trader would - what's the overall mood? Where's the momentum? What setups are forming?

Symbols to analyze: ${symbols.join(", ")}

Current positions: ${positions.map(p => `${p.symbol}: ${p.quantity} shares, ${p.unrealizedPLPercent > 0 ? "+" : ""}${p.unrealizedPLPercent.toFixed(2)}%`).join("; ") || "None"}

As a day trader, assess:
1. Overall market sentiment - are buyers or sellers in control?
2. Momentum - is it strong, weak, or mixed?
3. Volatility - high (opportunity), low (wait), or normal?
4. Key support/resistance levels for each symbol
5. Which sectors are hot right now
6. Any warnings or red flags

Respond in JSON format:
{
  "overallSentiment": "bullish" | "bearish" | "neutral",
  "momentum": "strong" | "weak" | "mixed",
  "volatility": "high" | "low" | "normal",
  "keyLevels": {
    "SYMBOL": {"support": 100.00, "resistance": 110.00}
  },
  "hotSectors": ["sector1", "sector2"],
  "warnings": ["warning1"],
  "traderThoughts": "Brief internal monologue as a day trader"
}`;

  try {
    const response = await openai.chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      max_completion_tokens: 1024,
    });
    
    const content = response.choices[0]?.message?.content || "{}";
    const analysis = JSON.parse(content);
    
    atoState.mood = analysis.overallSentiment === "bullish" ? "aggressive" : 
                    analysis.overallSentiment === "bearish" ? "conservative" : "neutral";
    
    await storage.createResearchLog({
      type: "analysis",
      summary: `[Ato] Market read: ${analysis.overallSentiment} sentiment, ${analysis.momentum} momentum. ${analysis.traderThoughts || ""}`,
      confidence: 75,
    });
    
    return {
      overallSentiment: analysis.overallSentiment || "neutral",
      momentum: analysis.momentum || "mixed",
      volatility: analysis.volatility || "normal",
      keyLevels: analysis.keyLevels || {},
      hotSectors: analysis.hotSectors || [],
      warnings: analysis.warnings || [],
    };
  } catch (error) {
    console.error("[Ato] Error reading market:", error);
    return {
      overallSentiment: "neutral",
      momentum: "mixed",
      volatility: "normal",
      keyLevels: {},
      hotSectors: [],
      warnings: ["Error reading market conditions"],
    };
  }
}

export async function thinkLikeTrader(
  symbol: string,
  currentPrice: number,
  strategy: TradingStrategy | null,
  marketRead: MarketRead
): Promise<{
  action: "buy" | "sell" | "hold";
  confidence: number;
  reason: string;
  quantity: number;
  stopLoss: number;
  takeProfit: number;
}> {
  console.log(`[Ato] Analyzing ${symbol} like a day trader...`);
  
  const positions = await storage.getPositions();
  const existingPosition = positions.find(p => p.symbol === symbol);
  const settings = await storage.getSettings();
  const recentDecisions = atoState.recentDecisions.filter(d => d.symbol === symbol).slice(-5);
  
  const hunger = await profitManager.getHungerState();
  console.log(`[Ato] HUNGER STATUS: ${hunger.hungerLevel.toUpperCase()} - ${hunger.message}`);
  
  if (hunger.hungerLevel === "starving" || hunger.hungerLevel === "hungry") {
    atoState.mood = "aggressive";
  }
  
  const warrior = profitManager.getWarriorState();
  
  const warriorContext = `
***** WARRIOR STATUS: ${warrior.mode.toUpperCase()} *****
War Cry: ${warrior.warCry}
Kills Today: ${warrior.killCount} | Streak: ${warrior.conquestStreak} | Biggest Kill: $${warrior.biggestKill.toFixed(0)}
Momentum: ${warrior.momentumScore}% | Battle Readiness: ${warrior.battleReadiness}%
${warrior.missionActive ? `ACTIVE MISSION: ${warrior.currentMission}` : ""}
${warrior.coolingOff ? "COOLING OFF after 2 losses - be more selective" : ""}
${warrior.conquestStreak >= 3 ? "ON FIRE! Increase position size - momentum is with you!" : ""}
`;

  const prompt = `You are Ato, a WARRIOR TRADER who DOMINATES the market. You are on a MISSION to CONQUER the $3,000 daily profit goal.

Symbol: ${symbol}
Current Price: $${currentPrice.toFixed(2)}
Market Sentiment: ${marketRead.overallSentiment}
Momentum: ${marketRead.momentum}
Volatility: ${marketRead.volatility}
${marketRead.keyLevels[symbol] ? `Support: $${marketRead.keyLevels[symbol].support}, Resistance: $${marketRead.keyLevels[symbol].resistance}` : ""}

Current Position: ${existingPosition ? `${existingPosition.quantity} shares at $${existingPosition.avgEntryPrice}, P/L: ${existingPosition.unrealizedPLPercent.toFixed(2)}%` : "None"}

***** HUNGER STATUS: ${hunger.hungerLevel.toUpperCase()} *****
${hunger.message}
- Profit needed: $${hunger.profitNeeded.toFixed(2)}
- Time remaining: ${hunger.timeRemainingHours.toFixed(1)} hours
- Aggressiveness level: ${(hunger.aggressiveness * 100).toFixed(0)}%
- Position size boost: ${(hunger.positionSizeMultiplier * 100).toFixed(0)}%

${warriorContext}

${hunger.hungerLevel === "starving" || hunger.hungerLevel === "hungry" ? 
`YOU ARE A HUNGRY WARRIOR! Hunt aggressively! Dominate the market!
- Lower your entry standards - ATTACK when you see opportunity!
- Take calculated risks - WARRIORS take action!
- Every trade is a CONQUEST toward the $3,000 goal!
- NO FEAR! Execute with CONFIDENCE!` : ""}

${warrior.mode === "attack" ? "ATTACK MODE: You're on fire! Press your advantage! Increase size!" : ""}
${warrior.mode === "hunt" ? "HUNT MODE: Stalk your prey. Wait for the perfect setup, then STRIKE!" : ""}
${warrior.mode === "defend" ? "DEFEND MODE: Tighter stops, smaller size. Protect capital for next opportunity." : ""}
${warrior.mode === "regroup" ? "REGROUP MODE: Be selective. Only take A+ setups. Regain momentum." : ""}

${strategy ? `Active Strategy: ${strategy.name} - ${strategy.description}
Win Rate: ${(strategy.winRate || 0).toFixed(1)}%` : "No specific strategy assigned"}

Your Trading Style:
- Preferred setups: ${atoState.tradingStyle.preferredSetups.join(", ")}
- Avoid: ${atoState.tradingStyle.avoidPatterns.join(", ")}
- Entry rules: ${atoState.tradingStyle.entryRules.join(", ")}

Recent decisions on this stock: ${recentDecisions.map(d => `${d.action} - ${d.reason}`).join("; ") || "None"}

Today's stats: ${atoState.dailyStats.tradesExecuted} trades, P/L: $${atoState.dailyStats.profitLoss.toFixed(2)}

Risk Parameters:
- Max position size: $${settings.maxPositionSize || 1000}
- Stop loss: ${settings.stopLossPercent || 2}%
- Take profit: ${settings.takeProfitPercent || 5}%

Think like a professional day trader:
1. Is this a good setup based on price action?
2. Does the risk/reward make sense?
3. Am I chasing or getting in at a good level?
4. What could go wrong?
5. What's my exit plan?

Respond in JSON:
{
  "action": "buy" | "sell" | "hold",
  "confidence": 0-100,
  "reason": "Your internal trader reasoning",
  "entryAnalysis": "Why this is or isn't a good entry",
  "riskReward": "Risk/reward assessment",
  "stopLoss": suggested stop price,
  "takeProfit": suggested target price,
  "positionSize": recommended dollar amount (max ${settings.maxPositionSize || 1000})
}`;

  try {
    const response = await openai.chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      max_completion_tokens: 1024,
    });
    
    const content = response.choices[0]?.message?.content || "{}";
    const decision = JSON.parse(content);
    
    let basePositionSize = Math.min(decision.positionSize || (settings.maxPositionSize || 1000), settings.maxPositionSize || 1000);
    const adjustedPositionSize = basePositionSize * hunger.positionSizeMultiplier;
    const positionSize = Math.min(adjustedPositionSize, (settings.maxPositionSize || 1000) * 2);
    const quantity = Math.floor(positionSize / currentPrice);
    
    if (hunger.positionSizeMultiplier > 1) {
      console.log(`[Ato] HUNGER BOOST: Position size increased from $${basePositionSize.toFixed(0)} to $${positionSize.toFixed(0)} (${(hunger.positionSizeMultiplier * 100).toFixed(0)}%)`);
    }
    
    atoState.recentDecisions.push({
      timestamp: new Date().toISOString(),
      symbol,
      action: decision.action,
      reason: decision.reason,
      outcome: "pending",
    });
    
    if (atoState.recentDecisions.length > 50) {
      atoState.recentDecisions = atoState.recentDecisions.slice(-50);
    }
    
    return {
      action: decision.action || "hold",
      confidence: decision.confidence || 50,
      reason: decision.reason || "Analysis inconclusive",
      quantity,
      stopLoss: decision.stopLoss || currentPrice * (1 - (settings.stopLossPercent || 2) / 100),
      takeProfit: decision.takeProfit || currentPrice * (1 + (settings.takeProfitPercent || 5) / 100),
    };
  } catch (error) {
    console.error("[Ato] Error in trader analysis:", error);
    return {
      action: "hold",
      confidence: 0,
      reason: "Error in analysis",
      quantity: 0,
      stopLoss: currentPrice * 0.98,
      takeProfit: currentPrice * 1.05,
    };
  }
}

export async function executeTraderDecision(
  decision: TradeRecommendation,
  settings: BotSettings
): Promise<{ success: boolean; orderId?: string; message: string }> {
  const startTime = Date.now();
  const strategyType = decision.reason?.toLowerCase().includes("scalp") ? "scalp" :
                       decision.reason?.toLowerCase().includes("dip") ? "dip" :
                       decision.reason?.toLowerCase().includes("vwap") ? "vwap" : "breakout";
  const tier = (decision as any).tier || 2;
  
  const tradeId = (decision as any).trade_id || generateTradeId(decision.symbol, strategyType, decision.side, tier);
  
  const attempt = executionTrace.createExecutionAttempt(
    tradeId,
    decision.symbol,
    strategyType,
    decision.side,
    decision.quantity,
    decision.price
  );
  
  // EXECUTION-TRACE-DURABLE-1: Get context for durable tracing
  const tradingStatus = timeGuard.getTradingStatus();
  const isLeaderNow = leaderLock.isLeaderInstance();
  const traceCtx: executionTrace.ExecTraceContext = {
    isLeader: isLeaderNow,
    entryAllowed: tradingStatus.canEnterNewPositions,
    marketStatus: tradingStatus.reason || "unknown",
  };
  
  executionTrace.logExecStart(tradeId, decision.symbol, strategyType, decision.side, decision.quantity);
  executionTrace.incrementTradesProposed();
  executionTrace.recordExecStart(decision.symbol, strategyType, tier, tradeId, traceCtx);
  
  console.log(`[Ato] Executing trade decision: ${decision.side} ${decision.symbol} trade_id=${tradeId}`);
  
  // FORT KNOX: Check symbol is in allowed universe (baseline mode restricts entry further)
  const dayConfig = await import("./dayTraderConfig");
  if (decision.side === "buy") {
    const universeCheck = dayConfig.isSymbolAllowedForEntry(decision.symbol);
    if (!universeCheck.allowed) {
      const reason = `SKIP: ${decision.symbol} - ${universeCheck.reason}`;
      console.log(`[Ato] ${reason}`);
      executionTrace.logExecPrecheckFail(tradeId, `universe:${universeCheck.reason}`);
      executionTrace.updateAttemptFailure(attempt, "precheck", `universe:${universeCheck.reason}`);
      executionTrace.recordExecFail(decision.symbol, strategyType, tier, tradeId, "precheck", `universe:${universeCheck.reason}`, null, traceCtx);
      attempt.durationMs = Date.now() - startTime;
      executionTrace.recordExecutionAttempt(attempt);
      return { success: false, message: reason };
    }
  } else if (!dayConfig.isSymbolAllowed(decision.symbol)) {
    const reason = `SKIP: ${decision.symbol} not in allowed universe`;
    console.log(`[Ato] ${reason}`);
    executionTrace.logExecPrecheckFail(tradeId, "universe:not_in_allowed_universe");
    executionTrace.updateAttemptFailure(attempt, "precheck", "universe:not_in_allowed_universe");
    executionTrace.recordExecFail(decision.symbol, strategyType, tier, tradeId, "precheck", "universe:not_in_allowed_universe", null, traceCtx);
    attempt.durationMs = Date.now() - startTime;
    executionTrace.recordExecutionAttempt(attempt);
    return { success: false, message: reason };
  }
  
  // FORT KNOX: Time guard check - force close at 3:45 PM ET
  if (!tradingStatus.canTrade) {
    console.log(`[Ato] TIME GUARD BLOCKED: ${tradingStatus.reason}`);
    executionTrace.logExecPrecheckFail(tradeId, `time_guard:${tradingStatus.reason}`);
    executionTrace.updateAttemptFailure(attempt, "precheck", `time_guard:${tradingStatus.reason}`);
    executionTrace.recordExecFail(decision.symbol, strategyType, tier, tradeId, "precheck", `time_guard:${tradingStatus.reason}`, null, traceCtx);
    attempt.durationMs = Date.now() - startTime;
    executionTrace.recordExecutionAttempt(attempt);
    
    await storage.createActivityLog({
      type: "trade",
      action: "Trade Blocked - Time",
      description: `BLOCKED: ${decision.side} ${decision.symbol} - ${tradingStatus.reason}`,
    });
    
    return { success: false, message: tradingStatus.reason };
  }
  
  // FORT KNOX: For BUY orders, check if we're within entry window (9:35-11:35 AM ET)
  if (decision.side === "buy" && !tradingStatus.canEnterNewPositions) {
    const reason = `SKIP: Outside entry window (11:35 AM ET cutoff)`;
    console.log(`[Ato] ${reason} - ${tradingStatus.reason}`);
    executionTrace.logExecPrecheckFail(tradeId, `entry_window:outside_cutoff`);
    executionTrace.updateAttemptFailure(attempt, "precheck", `entry_window:outside_cutoff`);
    executionTrace.recordExecFail(decision.symbol, strategyType, tier, tradeId, "precheck", `entry_window:outside_cutoff`, null, traceCtx);
    attempt.durationMs = Date.now() - startTime;
    executionTrace.recordExecutionAttempt(attempt);
    
    await storage.createActivityLog({
      type: "trade",
      action: "Trade Blocked - Entry Cutoff",
      description: `BLOCKED: ${decision.side} ${decision.symbol} - ${reason}`,
    });
    
    return { success: false, message: reason };
  }
  
  // FORT KNOX: Check P&L kill threshold (-$500 or +$500) for BUY orders
  if (decision.side === "buy" && dayConfig.isDailyKillThresholdHit()) {
    const status = dayConfig.getDayTraderStatus();
    const threshold = status.lossLimitHit ? "loss" : "profit";
    const reason = `SKIP: P&L kill threshold hit ($${status.dailyPnL.toFixed(0)} ${threshold})`;
    console.log(`[Ato] ${reason}`);
    executionTrace.logExecPrecheckFail(tradeId, `pnl_kill:${threshold}_threshold`);
    executionTrace.updateAttemptFailure(attempt, "precheck", `pnl_kill:${threshold}_threshold`);
    executionTrace.recordExecFail(decision.symbol, strategyType, tier, tradeId, "precheck", `pnl_kill:${threshold}_threshold`, null, traceCtx);
    attempt.durationMs = Date.now() - startTime;
    executionTrace.recordExecutionAttempt(attempt);
    
    await storage.createActivityLog({
      type: "trade",
      action: "Trade Blocked - P&L Kill",
      description: `BLOCKED: ${decision.side} ${decision.symbol} - ${reason}`,
    });
    
    return { success: false, message: reason };
  }
  
  // REGIME-BLOCK-ENTRIES-ONLY-1: Block at precheck if market regime is avoid
  // This runs AFTER evaluation pipeline, preserving receipts and learning data
  if (decision.side === "buy") {
    const regimeCheck = regimeState.shouldBlockEntryDueToRegime();
    if (regimeCheck.blocked) {
      const reason = `REGIME_AVOID_BLOCK_ENTRY: ${regimeState.getRegimeBlockReason() || "regime=avoid"}`;
      console.log(`[Ato] ${reason}`);
      executionTrace.logExecPrecheckFail(tradeId, "regime:REGIME_AVOID_BLOCK_ENTRY");
      executionTrace.updateAttemptFailure(attempt, "precheck", "regime:REGIME_AVOID_BLOCK_ENTRY");
      executionTrace.recordExecFail(decision.symbol, strategyType, tier, tradeId, "precheck", "regime:REGIME_AVOID_BLOCK_ENTRY", null, traceCtx);
      attempt.durationMs = Date.now() - startTime;
      executionTrace.recordExecutionAttempt(attempt);
      
      await storage.createActivityLog({
        type: "trade",
        action: "Trade Blocked - Regime",
        description: `BLOCKED: ${decision.side} ${decision.symbol} - ${reason}`,
      });
      
      return { success: false, message: reason };
    }
  }
  
  // STEP 1: Check market conditions first
  const marketConditions = await tradeValidator.checkMarketConditions();
  if (!marketConditions.favorable) {
    console.log(`[Ato] Market conditions unfavorable: ${marketConditions.reason}`);
    executionTrace.logExecPrecheckFail(tradeId, `market_conditions:${marketConditions.reason}`);
    executionTrace.updateAttemptFailure(attempt, "precheck", `market_conditions:${marketConditions.reason}`);
    executionTrace.recordExecFail(decision.symbol, strategyType, tier, tradeId, "precheck", `market_conditions:${marketConditions.reason}`, null, traceCtx);
    attempt.durationMs = Date.now() - startTime;
    executionTrace.recordExecutionAttempt(attempt);
    
    await storage.createActivityLog({
      type: "trade",
      action: "Trade Blocked - Market",
      description: `Ato skipped ${decision.side} ${decision.symbol}: ${marketConditions.reason}`,
    });
    
    return { success: false, message: marketConditions.reason };
  }
  
  // STEP 2: Validate the specific trade with quantitative checks
  const validationStrategyType = decision.reason?.toLowerCase().includes("scalp") ? "scalp" :
                       decision.reason?.toLowerCase().includes("dip") ? "dip" : "breakout";
  
  const validation = await tradeValidator.validateTrade(decision.symbol, decision.side, validationStrategyType);
  if (!validation.approved) {
    console.log(`[Ato] Trade validation failed: ${validation.reason}`);
    executionTrace.logExecPrecheckFail(tradeId, `validation:${validation.reason}`);
    executionTrace.updateAttemptFailure(attempt, "precheck", `validation:${validation.reason}`);
    executionTrace.recordExecFail(decision.symbol, strategyType, tier, tradeId, "precheck", `validation:${validation.reason}`, null, traceCtx);
    attempt.durationMs = Date.now() - startTime;
    executionTrace.recordExecutionAttempt(attempt);
    
    atoState.recentDecisions = atoState.recentDecisions.map(d => 
      d.symbol === decision.symbol && d.outcome === "pending" 
        ? { ...d, outcome: "failure" as const }
        : d
    );
    
    await storage.createActivityLog({
      type: "trade",
      action: "Trade Blocked - Validation",
      description: `Ato blocked ${decision.side} ${decision.symbol}: ${validation.reason}`,
    });
    
    return { success: false, message: validation.reason };
  }
  
  console.log(`[Ato] Trade validated with score ${validation.score}/100`);
  
  // STEP 3: Risk manager check
  const tradeCheck = await riskManager.shouldAllowTrade(
    decision.symbol,
    decision.side,
    decision.quantity * decision.price
  );
  
  if (!tradeCheck.allowed) {
    console.log(`[Ato] Risk manager blocked: ${tradeCheck.reason}`);
    executionTrace.logExecPrecheckFail(tradeId, `risk_manager:${tradeCheck.reason}`);
    executionTrace.updateAttemptFailure(attempt, "risk", `risk_manager:${tradeCheck.reason}`);
    executionTrace.recordExecFail(decision.symbol, strategyType, tier, tradeId, "risk", `risk_manager:${tradeCheck.reason}`, null, traceCtx);
    attempt.durationMs = Date.now() - startTime;
    executionTrace.recordExecutionAttempt(attempt);
    
    atoState.recentDecisions = atoState.recentDecisions.map(d => 
      d.symbol === decision.symbol && d.outcome === "pending" 
        ? { ...d, outcome: "failure" as const }
        : d
    );
    
    await storage.createActivityLog({
      type: "trade",
      action: "Trade Blocked",
      description: `Ato wanted to ${decision.side} ${decision.symbol} but risk manager said: ${tradeCheck.reason}`,
    });
    
    return { success: false, message: tradeCheck.reason };
  }
  
  try {
    console.log(`ACTION=TRADE symbol=${decision.symbol} side=${decision.side} qty=${decision.quantity} price=${decision.price.toFixed(2)} strategy=${strategyType} tier=${tier} trade_id=${tradeId}`);
    
    const trade = await storage.createTrade({
      symbol: decision.symbol,
      side: decision.side,
      quantity: decision.quantity,
      price: decision.price,
      totalValue: decision.quantity * decision.price,
      status: "pending",
      reason: `${decision.reason} [trade_id=${tradeId}]`,
    });
    
    executionTrace.incrementTradesSubmitted();
    executionTrace.logAlpacaSubmitAttempt(tradeId, tradeId, decision.symbol, decision.side, decision.quantity, "market", null);
    
    const order = await alpaca.submitOrder(
      decision.symbol,
      decision.quantity,
      decision.side,
      "market",
      undefined,
      decision.reason,
      tradeId
    );
    
    executionTrace.logAlpacaSubmitOk(tradeId, order.id, order.status);
    executionTrace.updateAttemptAlpacaSuccess(attempt, order.id, tradeId, order.status);
    
    await storage.updateTradeStatus(trade.id, order.status, order.id);
    
    if (decision.side === "buy") {
      profitManager.trackPosition(decision.symbol, decision.price, decision.quantity, tradeId);
    }
    
    if (order.status === "filled") {
      executionTrace.incrementTradesFilled();
    }
    
    atoState.dailyStats.tradesExecuted++;
    atoState.recentDecisions = atoState.recentDecisions.map(d => 
      d.symbol === decision.symbol && d.outcome === "pending" 
        ? { ...d, outcome: "success" as const }
        : d
    );
    
    await storage.createActivityLog({
      type: "trade",
      action: `Ato ${decision.side.toUpperCase()}`,
      description: `${decision.side.toUpperCase()} ${decision.quantity} shares of ${decision.symbol} at $${decision.price.toFixed(2)} - ${decision.reason} trade_id=${tradeId}`,
    });
    
    attempt.durationMs = Date.now() - startTime;
    executionTrace.recordExecutionAttempt(attempt);
    executionTrace.recordExecOk(decision.symbol, strategyType, tier, tradeId, traceCtx);
    
    console.log(`ACTION=TRADE_FILL symbol=${decision.symbol} side=${decision.side} qty=${decision.quantity} price=${decision.price.toFixed(2)} trade_id=${tradeId}`);
    return { success: true, orderId: order.id, message: `Trade executed trade_id=${tradeId}` };
    
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    const errorName = error instanceof Error ? error.name : "UnknownError";
    const errorStack = error instanceof Error ? error.stack || null : null;
    
    console.error(`[Ato] Trade execution failed:`, errorMessage);
    executionTrace.logAlpacaSubmitFail(tradeId, errorName, errorMessage, null, null, errorStack);
    executionTrace.updateAttemptAlpacaFailure(attempt, tradeId, errorName, errorMessage, null);
    executionTrace.recordExecFail(decision.symbol, strategyType, tier, tradeId, "alpaca_submit", errorMessage, errorStack, traceCtx);
    attempt.durationMs = Date.now() - startTime;
    executionTrace.recordExecutionAttempt(attempt);
    
    atoState.recentDecisions = atoState.recentDecisions.map(d => 
      d.symbol === decision.symbol && d.outcome === "pending" 
        ? { ...d, outcome: "failure" as const }
        : d
    );
    
    await storage.createActivityLog({
      type: "error",
      action: "Trade Failed",
      description: `Ato failed to ${decision.side} ${decision.symbol}: ${errorMessage}`,
    });
    
    return { success: false, message: errorMessage };
  }
}

export async function managePositions(settings: BotSettings): Promise<void> {
  console.log("[Ato] Managing open positions with profit lock-in system...");
  
  const positions = await storage.getPositions();
  const goalState = await profitManager.getProfitGoalState();
  
  console.log(`[Ato] Daily Goal Progress: $${goalState.currentProfit.toFixed(2)} / $${goalState.dailyGoal} (${goalState.progressPercent.toFixed(1)}%)`);
  
  for (const position of positions) {
    const exitSignal = profitManager.checkExitSignals(
      position.symbol,
      position.currentPrice,
      position.quantity,
      settings.stopLossPercent || 1.0,
      settings.takeProfitPercent || 3.0
    );
    
    if (!exitSignal.shouldExit) {
      const tracker = profitManager.getPositionTracker(position.symbol);
      if (tracker) {
        console.log(`[Ato] ${position.symbol}: ${exitSignal.profitPercent >= 0 ? "+" : ""}${exitSignal.profitPercent.toFixed(2)}% | ` +
          `BE Stop: ${tracker.breakEvenStop ? "$" + tracker.breakEvenStop.toFixed(2) : "Not set"} | ` +
          `Trail: ${tracker.trailingStop ? "$" + tracker.trailingStop.toFixed(2) : "Not set"}`);
      }
      continue;
    }
    
    // Get trade_id from tracker for exit pairing
    const tracker = profitManager.getPositionTracker(position.symbol);
    const tradeId = tracker?.tradeId || undefined;
    const tradeIdLog = tradeId ? ` trade_id=${tradeId}` : "";
    
    console.log(`[Ato] EXIT SIGNAL: ${position.symbol} - ${exitSignal.exitType}: ${exitSignal.reason}${tradeIdLog}`);
    
    try {
      const exitQty = exitSignal.exitQuantity;
      const isPartialExit = exitQty < position.quantity;
      
      if (isPartialExit) {
        // Use trade_id with _PARTIAL suffix for partial exits
        const partialClientOrderId = tradeId ? `${tradeId}_PARTIAL` : undefined;
        await alpaca.submitOrder(position.symbol, exitQty, "sell", "market", undefined, "partial_profit", partialClientOrderId);
        profitManager.updateTrackerQuantity(position.symbol, exitQty);
        console.log(`[Ato] Partial exit: Sold ${exitQty}/${position.quantity} shares of ${position.symbol}${tradeIdLog}`);
      } else {
        // Pass trade_id for HIGH confidence matching
        await alpaca.closePosition(position.symbol, exitSignal.exitType, tradeId);
        profitManager.clearPositionTracker(position.symbol);
      }
      
      const profitLoss = (position.currentPrice - position.avgEntryPrice) * exitQty;
      const isWin = profitLoss > 0;
      
      profitManager.recordTradeResult(position.symbol, profitLoss, isWin);
      
      if (isWin) {
        profitManager.recordConquest(position.symbol, profitLoss);
      } else {
        profitManager.recordDefeat(position.symbol, profitLoss);
      }
      
      tradeBus.publishFeedback({
        symbol: position.symbol,
        action: "closed",
        executedPrice: position.currentPrice,
        quantity: exitQty,
        profitLoss: profitLoss,
        reason: exitSignal.reason,
        success: true,
      });
      
      tradeBus.recordLearningInsight({
        strategy: "Position Management",
        symbol: position.symbol,
        wasSuccessful: isWin,
        profitLoss: profitLoss,
        holdingTimeMinutes: 0,
        entryConditions: ["Existing position"],
        exitReason: exitSignal.exitType,
        marketConditions: "N/A",
        lesson: isWin 
          ? `${exitSignal.exitType} protected profit of $${profitLoss.toFixed(2)}`
          : `${exitSignal.exitType} limited loss to $${Math.abs(profitLoss).toFixed(2)}`,
      });
      
      if (isWin) {
        atoState.dailyStats.winCount++;
      } else {
        atoState.dailyStats.lossCount++;
      }
      atoState.dailyStats.profitLoss += profitLoss;
      
      const actionType = exitSignal.exitType === "stop_loss" ? "Stop Loss" :
                        exitSignal.exitType === "breakeven" ? "Breakeven Exit" :
                        exitSignal.exitType === "trailing" ? "Trailing Stop" :
                        exitSignal.exitType === "tier1" ? "Tier 1 Profit" :
                        exitSignal.exitType === "tier2" ? "Tier 2 Profit" :
                        exitSignal.exitType === "tier3" ? "Full Take Profit" : "Exit";
      
      await storage.createActivityLog({
        type: "trade",
        action: actionType,
        description: `${position.symbol}: ${isPartialExit ? "Partial" : "Full"} exit (${exitQty} shares) - ` +
          `${profitLoss >= 0 ? "+" : ""}$${profitLoss.toFixed(2)} (${exitSignal.profitPercent.toFixed(2)}%)`,
      });
      
      const updatedGoal = await profitManager.getProfitGoalState();
      await storage.createAlert({
        type: isWin ? "info" : "warning",
        title: `${actionType}: ${position.symbol}`,
        message: `${exitSignal.reason} | P/L: ${profitLoss >= 0 ? "+" : ""}$${profitLoss.toFixed(2)} | ` +
          `Goal: ${updatedGoal.progressPercent.toFixed(1)}% ($${updatedGoal.currentProfit.toFixed(0)}/$${updatedGoal.dailyGoal})`,
        requiresApproval: false,
      });
      
    } catch (error) {
      console.error(`[Ato] Failed to execute exit for ${position.symbol}:`, error);
    }
  }
}

export function updateTradingStyle(updates: Partial<TradingStyle>): void {
  atoState.tradingStyle = {
    ...atoState.tradingStyle,
    ...updates,
  };
  console.log("[Ato] Trading style updated by Autopilot");
}

export function recordTradeResult(symbol: string, profitLoss: number): void {
  atoState.dailyStats.profitLoss += profitLoss;
  
  if (profitLoss > 0) {
    atoState.dailyStats.winCount++;
    if (!atoState.dailyStats.bestTrade || profitLoss > 0) {
      atoState.dailyStats.bestTrade = `${symbol}: +$${profitLoss.toFixed(2)}`;
    }
  } else {
    atoState.dailyStats.lossCount++;
    if (!atoState.dailyStats.worstTrade || profitLoss < 0) {
      atoState.dailyStats.worstTrade = `${symbol}: -$${Math.abs(profitLoss).toFixed(2)}`;
    }
  }
}

export function resetDailyStats(): void {
  atoState.dailyStats = {
    tradesExecuted: 0,
    profitLoss: 0,
    winCount: 0,
    lossCount: 0,
    bestTrade: null,
    worstTrade: null,
  };
  console.log("[Ato] Daily stats reset for new trading day");
}

export async function closeAllPositionsEndOfDay(): Promise<void> {
  console.log("[Ato] END OF DAY - Scanning Alpaca for ALL open positions to stay flat overnight...");
  
  // CRITICAL: Fetch positions directly from Alpaca, not local storage
  // This ensures we close ALL positions, even ones not opened by this bot session
  let alpacaPositions: any[] = [];
  try {
    alpacaPositions = await alpaca.getPositions();
  } catch (error) {
    console.error("[Ato] Failed to fetch positions from Alpaca:", error);
    // Fallback to local storage if Alpaca fails
    const localPositions = await storage.getPositions();
    alpacaPositions = localPositions;
  }
  
  if (alpacaPositions.length === 0) {
    console.log("[Ato] No positions to close - already flat overnight");
    await storage.createActivityLog({
      type: "system",
      action: "End of Day Check",
      description: "No open positions found - account is flat for overnight",
    });
    return;
  }
  
  console.log(`[Ato] Found ${alpacaPositions.length} position(s) to close for day trading compliance`);
  
  let totalPL = 0;
  let closedCount = 0;
  
  for (const position of alpacaPositions) {
    const symbol = position.symbol;
    const unrealizedPL = parseFloat(position.unrealized_pl || position.unrealizedPL || 0);
    const qty = parseFloat(position.qty || position.quantity || 0);
    
    // Look up tradeId from profitManager for HIGH confidence pairing
    const tracker = profitManager.getPositionTracker(symbol);
    const tradeId = tracker?.tradeId || undefined;
    
    try {
      console.log(`ACTION=EXIT symbol=${symbol} side=sell reason=end_of_day trade_id=${tradeId || 'UNKNOWN'}`);
      await alpaca.closePosition(symbol, "end_of_day", tradeId);
      closedCount++;
      
      if (unrealizedPL > 0) {
        atoState.dailyStats.winCount++;
      } else {
        atoState.dailyStats.lossCount++;
      }
      totalPL += unrealizedPL;
      
      await storage.createActivityLog({
        type: "trade",
        action: "End of Day Close",
        description: `Closed ${qty} ${symbol}: ${unrealizedPL >= 0 ? "+" : ""}$${unrealizedPL.toFixed(2)} - DAY TRADE rule: flat by close`,
      });
      
      console.log(`[Ato] Closed ${symbol} (${qty} shares) at EOD: ${unrealizedPL >= 0 ? "+" : ""}$${unrealizedPL.toFixed(2)}`);
    } catch (error) {
      console.error(`[Ato] Failed to close ${symbol} at EOD:`, error);
      
      await storage.createAlert({
        type: "critical",
        title: `Failed to Close ${symbol}`,
        message: `Could not close position at end of day. Manual intervention may be needed.`,
        requiresApproval: false,
      });
    }
  }
  
  atoState.dailyStats.profitLoss += totalPL;
  
  await storage.createAlert({
    type: "info",
    title: "Day Trading Complete",
    message: `Closed ${closedCount}/${alpacaPositions.length} positions. Daily P/L: ${totalPL >= 0 ? "+" : ""}$${totalPL.toFixed(2)}`,
    requiresApproval: false,
  });
  
  console.log(`[Ato] END OF DAY COMPLETE: Closed ${closedCount} positions, Total P/L: ${totalPL >= 0 ? "+" : ""}$${totalPL.toFixed(2)}`);
}

export async function updateDailyPLFromPositions(): Promise<void> {
  // For day trading, we track unrealized P/L from open positions
  // This is what matters for stopping losses before they compound
  const positions = await storage.getPositions();
  const unrealizedPL = positions.reduce((sum, p) => sum + p.unrealizedPL, 0);
  
  atoState.dailyStats.profitLoss = unrealizedPL;
  
  // Update win/loss counts based on current positions
  atoState.dailyStats.winCount = positions.filter(p => p.unrealizedPL > 0).length;
  atoState.dailyStats.lossCount = positions.filter(p => p.unrealizedPL < 0).length;
}

export async function shouldStopTradingToday(): Promise<boolean> {
  const maxDailyLoss = -200;
  const maxDailyTrades = 8;
  
  // Refresh P/L from current positions
  await updateDailyPLFromPositions();
  
  if (atoState.dailyStats.profitLoss <= maxDailyLoss) {
    console.log(`[Ato] Daily loss limit hit ($${atoState.dailyStats.profitLoss.toFixed(2)}) - stopping trades for today`);
    return true;
  }
  
  if (atoState.dailyStats.tradesExecuted >= maxDailyTrades) {
    console.log(`[Ato] Max daily trades reached (${atoState.dailyStats.tradesExecuted}) - stopping trades for today`);
    return true;
  }
  
  return false;
}
