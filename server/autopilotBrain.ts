import OpenAI from "openai";
import { storage } from "./storage";
import * as alpaca from "./alpaca";
import * as tradeValidator from "./tradeValidator";
import * as tradeBus from "./tradeIntelligenceBus";
import * as profitManager from "./profitManager";
import type { Trade, TradingStrategy, InsertTradingStrategy } from "@shared/schema";

const openai = new OpenAI({
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
});

const MODEL = "gpt-5";

interface BrainState {
  strategies: TradingStrategy[];
  marketInsights: MarketInsight[];
  stockPreferences: Record<string, number>;
  learningCycles: number;
  overallWinRate: number;
}

interface MarketInsight {
  pattern: string;
  reliability: number;
  lastSeen: string;
  successRate: number;
}

interface StrategyRules {
  entryConditions: string[];
  exitConditions: string[];
  positionSizing: string;
  riskManagement: string;
  preferredTimeframes: string[];
}

let brainState: BrainState = {
  strategies: [],
  marketInsights: [],
  stockPreferences: {},
  learningCycles: 0,
  overallWinRate: 0,
};

export async function initializeBrain(): Promise<void> {
  console.log("[Autopilot Brain] Initializing intelligent trading brain...");
  
  const strategies = await storage.getStrategies();
  brainState.strategies = strategies;
  
  const brain = await storage.getAutopilotBrain();
  if (brain) {
    brainState.marketInsights = brain.marketInsights ? JSON.parse(brain.marketInsights) : [];
    brainState.stockPreferences = brain.stockPreferences ? JSON.parse(brain.stockPreferences) : {};
    brainState.learningCycles = brain.totalLearningCycles || 0;
    brainState.overallWinRate = brain.overallWinRate || 0;
    
    // Sync preferences to trade validator for blacklist enforcement on startup
    tradeValidator.setStockPreferences(brainState.stockPreferences);
  }
  
  if (brainState.strategies.length === 0) {
    console.log("[Autopilot Brain] No strategies found. Creating initial strategies...");
    await createInitialStrategies();
  }
  
  console.log("[Autopilot Brain] Brain initialized with", brainState.strategies.length, "strategies");
}

async function createInitialStrategies(): Promise<void> {
  const defaultStrategies: InsertTradingStrategy[] = [
    {
      name: "Scalp Momentum",
      description: "DAY TRADE: Quick 1-2% profits on momentum bursts, tight stops",
      type: "momentum",
      rules: JSON.stringify({
        entryConditions: ["Strong volume spike", "Breaking intraday high", "RSI momentum above 50"],
        exitConditions: ["Take profit at 1.5-2%", "Stop loss at 1%", "MUST exit by market close"],
        positionSizing: "Risk 1% of portfolio per trade",
        riskManagement: "Hard stop at 1% - NO exceptions",
        preferredTimeframes: ["intraday-only"],
        dayTradingRules: ["Flat by close", "Max 5 trades per day", "No overnight holds"],
      }),
      symbols: JSON.stringify(["NVDA", "TSLA", "AMD", "AAPL"]),
      confidence: 65,
      isActive: true,
    },
    {
      name: "Intraday Dip Buy",
      description: "DAY TRADE: Buy quick dips for bounce plays, exit same day",
      type: "mean_reversion",
      rules: JSON.stringify({
        entryConditions: ["Sharp intraday pullback of 2%+", "Still above VWAP", "Volume expanding on dip"],
        exitConditions: ["Quick bounce profit 1-1.5%", "Stop loss at 1%", "MUST exit by 3:30 PM"],
        positionSizing: "Risk 1% of portfolio per trade",
        riskManagement: "Tight stop at 1% - cut losses fast",
        preferredTimeframes: ["intraday-only"],
        dayTradingRules: ["Flat by close", "Only trade first 2 hours or power hour", "No overnight holds"],
      }),
      symbols: JSON.stringify(["GOOGL", "MSFT", "AMZN", "META"]),
      confidence: 60,
      isActive: true,
    },
    {
      name: "Opening Range Breakout",
      description: "DAY TRADE: Trade breakouts from first 15-min range",
      type: "breakout",
      rules: JSON.stringify({
        entryConditions: ["Price breaks above/below first 15-min range", "Volume confirms breakout", "Market direction aligned"],
        exitConditions: ["Target 2% profit", "Stop at 1% or back inside range", "MUST exit by market close"],
        positionSizing: "Risk 1% of portfolio per trade",
        riskManagement: "Stop inside opening range - max 1% loss",
        preferredTimeframes: ["intraday-only"],
        dayTradingRules: ["Flat by close", "Only first hour trades", "No overnight holds"],
      }),
      symbols: JSON.stringify(["QQQ", "SPY", "TSLA", "NVDA"]),
      confidence: 60,
      isActive: true,
    },
  ];
  
  for (const strategy of defaultStrategies) {
    const created = await storage.createStrategy(strategy);
    brainState.strategies.push(created);
    console.log("[Autopilot Brain] Created strategy:", strategy.name);
  }
}

export async function conductResearch(): Promise<string> {
  console.log("[Autopilot Brain] Conducting market research...");
  
  const trades = await storage.getTrades();
  const positions = await storage.getPositions();
  const settings = await storage.getSettings();
  const hunger = await profitManager.getHungerState();
  
  console.log(`[Autopilot Brain] HUNGER STATUS: ${hunger.hungerLevel.toUpperCase()} - Need $${hunger.profitNeeded.toFixed(0)}`);
  
  const tradeHistory = trades.slice(-20).map(t => 
    `${t.side.toUpperCase()} ${t.symbol} at $${t.price} - ${t.status}`
  ).join("\n");
  
  const hungerContext = hunger.hungerLevel === "starving" || hunger.hungerLevel === "hungry" 
    ? `

***** CRITICAL: YOU ARE ${hunger.hungerLevel.toUpperCase()}! *****
${hunger.message}
- Need $${hunger.profitNeeded.toFixed(0)} more profit to hit $3,000 daily goal
- Time remaining: ${hunger.timeRemainingHours.toFixed(1)} hours
- Must earn $${hunger.profitPerHourNeeded.toFixed(0)} per hour!

BE AGGRESSIVE! Find opportunities NOW! Lower your standards slightly - we NEED trades!
Prioritize high-probability setups that can generate $200-500 per trade.
This is not the time to be conservative - the goal MUST be met!
`
    : `\nProgress toward $3,000 goal: $${(3000 - hunger.profitNeeded).toFixed(0)} earned, $${hunger.profitNeeded.toFixed(0)} to go.`;

  const warrior = profitManager.getWarriorState();
  
  const warriorHunterContext = `
***** WARRIOR COMMAND CENTER *****
Mode: ${warrior.mode.toUpperCase()} | War Cry: ${warrior.warCry}
Kills Today: ${warrior.killCount} | Conquest Streak: ${warrior.conquestStreak}
Biggest Kill: $${warrior.biggestKill.toFixed(0)} | Momentum: ${warrior.momentumScore}%
${warrior.missionActive ? `ACTIVE MISSION: ${warrior.currentMission}` : "No active mission - ASSIGN ONE!"}

${warrior.mode === "attack" ? "ATO IS ON FIRE! Find MORE opportunities to capitalize on momentum!" : ""}
${warrior.mode === "hunt" ? "HUNT aggressively! Scan for the best setups to STRIKE!" : ""}
${warrior.mode === "regroup" ? "Ato is regrouping. Find ONLY A+ setups with high probability." : ""}
`;

  const prompt = `You are Autopilot, the WARRIOR COMMANDER behind AtoBot. You are a HUNTER who finds and delivers KILL OPPORTUNITIES to Ato.
Your MISSION: Lead the CONQUEST of the $3,000 daily profit goal! Hunt relentlessly! Find targets! Command victory!
${warriorHunterContext}
${hungerContext}

Current Portfolio Value: ~$100,000
Current Positions: ${positions.length > 0 ? positions.map(p => `${p.symbol}: ${p.quantity} shares, P/L: $${p.unrealizedPL.toFixed(2)}`).join(", ") : "None"}
Recent Trades:
${tradeHistory || "No recent trades"}

Our Active Strategies:
${brainState.strategies.filter(s => s.isActive).map(s => `- ${s.name}: ${s.description} (Win Rate: ${(s.winRate || 0).toFixed(1)}%)`).join("\n")}

Current Market Conditions (consider typical market behaviors):
- Hunt aggressively for momentum opportunities
- Look for mean reversion setups in oversold stocks
- Find breakout patterns ready to explode
- Identify high-volatility stocks for bigger moves

YOU ARE A WARRIOR COMMANDER! Hunt for KILLS! Find CONQUESTS! Lead Ato to VICTORY!

Respond in JSON format:
{
  "marketOutlook": "Brief market analysis - where are the kills?",
  "warCry": "An aggressive battle cry to motivate the trading assault",
  "missionAssignment": "A specific mission for Ato (e.g., 'HUNT NVDA breakout above $150')",
  "targets": [{"symbol": "TICKER", "attackPlan": "how to conquer this stock", "killPotential": estimated_profit, "strategy": "which strategy", "priority": "primary|secondary|scout"}],
  "riskFactors": ["threats to avoid"],
  "strategyImprovements": [{"strategyName": "name", "suggestion": "improvement"}],
  "newStrategyIdeas": [{"name": "name", "description": "what it does", "type": "momentum|mean_reversion|breakout|custom"}],
  "commandMessage": "Direct order to Ato - aggressive and commanding"
}`;

  try {
    const response = await openai.chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      max_completion_tokens: 2048,
    });
    
    const content = response.choices[0]?.message?.content || "{}";
    const research = JSON.parse(content);
    
    await storage.createResearchLog({
      type: "analysis",
      summary: `[WARRIOR COMMAND] ${research.warCry || research.marketOutlook || "Hunting for kills..."}`,
      details: JSON.stringify(research),
      confidence: 75,
    });
    
    if (research.missionAssignment) {
      profitManager.startMission(research.missionAssignment);
    }
    
    if (research.commandMessage) {
      tradeBus.notifyAto(`[COMMANDER] ${research.commandMessage}`);
    } else if (research.warCry) {
      tradeBus.notifyAto(`[WAR CRY] ${research.warCry}`);
    }
    
    if (research.targets && research.targets.length > 0) {
      for (const target of research.targets.slice(0, 4)) {
        const confidenceLevel = target.priority === "primary" ? 85 : 
                               target.priority === "secondary" ? 70 : 55;
        tradeBus.publishSignal({
          symbol: target.symbol,
          action: "buy",
          confidence: confidenceLevel,
          strategy: target.strategy || "Warrior Hunt",
          reasoning: `[${target.priority?.toUpperCase() || "TARGET"}] ${target.attackPlan} (Kill potential: $${target.killPotential || "N/A"})`,
        });
      }
      console.log(`[WARRIOR COMMAND] Deployed ${research.targets.length} targets for Ato to hunt!`);
    }
    
    if (research.newStrategyIdeas && research.newStrategyIdeas.length > 0) {
      for (const idea of research.newStrategyIdeas) {
        await createNewStrategy(idea.name, idea.description, idea.type);
      }
    }
    
    await storage.updateAutopilotBrain({
      lastResearch: new Date(),
      totalLearningCycles: brainState.learningCycles + 1,
    });
    
    brainState.learningCycles++;
    
    console.log("[Autopilot Brain] Research complete. Outlook:", research.marketOutlook);
    return research.marketOutlook || "Research completed";
    
  } catch (error) {
    console.error("[Autopilot Brain] Research failed:", error);
    return "Research unavailable";
  }
}

export async function createNewStrategy(
  name: string,
  description: string,
  type: string
): Promise<TradingStrategy | null> {
  console.log("[Autopilot Brain] Creating new strategy:", name);
  
  const prompt = `Create a detailed trading strategy for:
Name: ${name}
Description: ${description}
Type: ${type}

Generate specific rules in JSON format:
{
  "entryConditions": ["condition1", "condition2"],
  "exitConditions": ["condition1", "condition2"],
  "positionSizing": "how to size positions",
  "riskManagement": "stop loss and risk rules",
  "preferredTimeframes": ["intraday", "swing", "position"]
}`;

  try {
    const response = await openai.chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      max_completion_tokens: 1024,
    });
    
    const rules = response.choices[0]?.message?.content || "{}";
    
    const strategy = await storage.createStrategy({
      name,
      description,
      type,
      rules,
      confidence: 50,
      isActive: true,
    });
    
    brainState.strategies.push(strategy);
    
    await storage.createActivityLog({
      type: "system",
      action: "Strategy Created",
      description: `Autopilot Brain created new strategy: ${name}`,
    });
    
    console.log("[Autopilot Brain] Strategy created successfully");
    return strategy;
    
  } catch (error) {
    console.error("[Autopilot Brain] Failed to create strategy:", error);
    return null;
  }
}

export async function learnFromTrade(trade: Trade, profitLoss: number): Promise<void> {
  console.log("[Autopilot Brain] Learning from trade:", trade.symbol, "P/L:", profitLoss);
  
  const wasSuccessful = profitLoss > 0;
  
  const matchingStrategy = brainState.strategies.find(s => {
    const symbols = s.symbols ? JSON.parse(s.symbols) : [];
    return symbols.includes(trade.symbol);
  });
  
  if (matchingStrategy) {
    const newTotalTrades = (matchingStrategy.totalTrades || 0) + 1;
    const newTotalProfit = (matchingStrategy.totalProfit || 0) + profitLoss;
    const winCount = wasSuccessful 
      ? Math.round((matchingStrategy.winRate || 0) / 100 * (matchingStrategy.totalTrades || 0)) + 1
      : Math.round((matchingStrategy.winRate || 0) / 100 * (matchingStrategy.totalTrades || 0));
    const newWinRate = (winCount / newTotalTrades) * 100;
    
    await storage.updateStrategy(matchingStrategy.id, {
      totalTrades: newTotalTrades,
      totalProfit: newTotalProfit,
      winRate: newWinRate,
      confidence: Math.max(30, Math.min(90, (matchingStrategy.confidence || 50) + (wasSuccessful ? 2 : -1))),
    });
    
    await storage.recordStrategyPerformance({
      strategyId: matchingStrategy.id,
      tradeId: trade.id,
      symbol: trade.symbol,
      profitLoss,
      wasSuccessful,
      entryReason: trade.reason,
    });
    
    console.log("[Autopilot Brain] Updated strategy", matchingStrategy.name, "- New win rate:", newWinRate.toFixed(1), "%");
  }
  
  const currentPref = brainState.stockPreferences[trade.symbol] || 50;
  brainState.stockPreferences[trade.symbol] = Math.max(10, Math.min(100, currentPref + (wasSuccessful ? 5 : -3)));
  
  await storage.updateAutopilotBrain({
    stockPreferences: JSON.stringify(brainState.stockPreferences),
  });
}

export async function selectBestStrategy(symbol: string): Promise<TradingStrategy | null> {
  const activeStrategies = brainState.strategies.filter(s => s.isActive);
  
  if (activeStrategies.length === 0) {
    return null;
  }
  
  const strategiesForSymbol = activeStrategies.filter(s => {
    const symbols = s.symbols ? JSON.parse(s.symbols) : [];
    return symbols.includes(symbol) || symbols.length === 0;
  });
  
  if (strategiesForSymbol.length === 0) {
    return activeStrategies.sort((a, b) => (b.confidence || 0) - (a.confidence || 0))[0];
  }
  
  return strategiesForSymbol.sort((a, b) => {
    const scoreA = ((a.confidence || 0) * 0.4) + ((a.winRate || 0) * 0.6);
    const scoreB = ((b.confidence || 0) * 0.4) + ((b.winRate || 0) * 0.6);
    return scoreB - scoreA;
  })[0];
}

export async function getSmartRecommendation(
  watchlist: string[],
  portfolioValue: number,
  currentPositions: { symbol: string; qty: number; unrealizedPL: number }[]
): Promise<{
  symbol: string;
  action: "buy" | "sell" | "hold";
  reason: string;
  confidence: number;
  strategy: string;
}> {
  const rankedSymbols = watchlist
    .map(symbol => ({
      symbol,
      preference: brainState.stockPreferences[symbol] || 50,
    }))
    .sort((a, b) => b.preference - a.preference);
  
  const existingSymbols = currentPositions.map(p => p.symbol);
  const availableSymbols = rankedSymbols.filter(s => !existingSymbols.includes(s.symbol));
  
  if (availableSymbols.length === 0) {
    const worstPosition = currentPositions.sort((a, b) => a.unrealizedPL - b.unrealizedPL)[0];
    if (worstPosition && worstPosition.unrealizedPL < 0) {
      return {
        symbol: worstPosition.symbol,
        action: "sell",
        reason: "Cutting losses on underperforming position",
        confidence: 60,
        strategy: "Risk Management",
      };
    }
    
    return {
      symbol: currentPositions[0]?.symbol || watchlist[0],
      action: "hold",
      reason: "Maintaining current positions",
      confidence: 50,
      strategy: "Position Management",
    };
  }
  
  const selectedSymbol = availableSymbols[0].symbol;
  const strategy = await selectBestStrategy(selectedSymbol);
  
  return {
    symbol: selectedSymbol,
    action: "buy",
    reason: `${strategy?.name || "Default"} strategy - High preference score based on past performance`,
    confidence: Math.min(75, (availableSymbols[0].preference + (strategy?.confidence || 50)) / 2),
    strategy: strategy?.name || "Default",
  };
}

export async function improveStrategies(): Promise<void> {
  console.log("[Autopilot Brain] Analyzing strategies for improvement...");
  
  const strategies = brainState.strategies.filter(s => (s.totalTrades || 0) >= 5);
  
  for (const strategy of strategies) {
    if ((strategy.winRate || 0) < 40 && (strategy.totalTrades || 0) >= 10) {
      console.log("[Autopilot Brain] Deactivating underperforming strategy:", strategy.name);
      await storage.updateStrategy(strategy.id, { isActive: false });
      
      await storage.createActivityLog({
        type: "system",
        action: "Strategy Deactivated",
        description: `Autopilot deactivated ${strategy.name} due to ${(strategy.winRate || 0).toFixed(1)}% win rate`,
      });
    }
    
    if ((strategy.winRate || 0) > 60 && (strategy.confidence || 0) < 80) {
      console.log("[Autopilot Brain] Boosting confidence for successful strategy:", strategy.name);
      await storage.updateStrategy(strategy.id, { 
        confidence: Math.min(85, (strategy.confidence || 50) + 10) 
      });
    }
  }
}

export function getBrainStatus(): {
  strategies: number;
  activeStrategies: number;
  learningCycles: number;
  topStrategy: string | null;
  overallConfidence: number;
} {
  const activeStrategies = brainState.strategies.filter(s => s.isActive);
  const topStrategy = activeStrategies.sort((a, b) => 
    ((b.winRate || 0) + (b.confidence || 0)) - ((a.winRate || 0) + (a.confidence || 0))
  )[0];
  
  const avgConfidence = activeStrategies.reduce((sum, s) => sum + (s.confidence || 50), 0) / 
    (activeStrategies.length || 1);
  
  return {
    strategies: brainState.strategies.length,
    activeStrategies: activeStrategies.length,
    learningCycles: brainState.learningCycles,
    topStrategy: topStrategy?.name || null,
    overallConfidence: Math.round(avgConfidence),
  };
}

export async function runBrainCycle(): Promise<void> {
  console.log("[Autopilot Brain] Running brain cycle...");
  
  await conductResearch();
  
  await improveStrategies();
  
  await learnFromAtoFeedback();
  
  const status = getBrainStatus();
  console.log("[Autopilot Brain] Cycle complete.", status);
}

async function learnFromAtoFeedback(): Promise<void> {
  console.log("[Autopilot Brain] Learning from Ato's trade feedback...");
  
  const feedback = tradeBus.getRecentFeedback(20);
  const closedTrades = feedback.filter(f => f.action === "closed" && f.profitLoss !== undefined);
  
  if (closedTrades.length === 0) {
    return;
  }
  
  const insights = tradeBus.getLearningInsights();
  const recentInsights = insights.slice(-10);
  
  const winningPatterns = recentInsights.filter(i => i.wasSuccessful);
  const losingPatterns = recentInsights.filter(i => !i.wasSuccessful);
  
  if (winningPatterns.length > losingPatterns.length) {
    tradeBus.notifyAto("Positive trend detected - maintaining current strategy");
  } else if (losingPatterns.length > 2) {
    tradeBus.notifyAto("Multiple losses detected - consider tighter stops");
  }
  
  for (const trade of closedTrades) {
    const symbol = trade.symbol;
    const profitLoss = trade.profitLoss || 0;
    const isWin = profitLoss > 0;
    
    const currentPref = brainState.stockPreferences[symbol] || 50;
    const adjustment = isWin ? 5 : -10;
    brainState.stockPreferences[symbol] = Math.max(0, Math.min(100, currentPref + adjustment));
    
    tradeValidator.setStockPreferences(brainState.stockPreferences);
  }
  
  const commSummary = tradeBus.getCommunicationSummary();
  if (commSummary.topStrategies.length > 0) {
    const topStrategy = commSummary.topStrategies[0];
    if (topStrategy.successRate > 60 && topStrategy.trades >= 5) {
      await storage.createResearchLog({
        type: "learning",
        summary: `[Autopilot] Top performing strategy: ${topStrategy.name} (${topStrategy.successRate.toFixed(1)}% win rate)`,
        details: JSON.stringify(topStrategy),
        confidence: 85,
      });
    }
  }
}

// Record trade outcome and update strategy performance
export async function recordTradeOutcome(
  symbol: string,
  action: "buy" | "sell",
  entryPrice: number,
  exitPrice: number,
  quantity: number,
  strategyName?: string
): Promise<void> {
  console.log(`[Autopilot Brain] Recording trade outcome for ${symbol}`);
  
  const profit = action === "buy" 
    ? (exitPrice - entryPrice) * quantity 
    : (entryPrice - exitPrice) * quantity;
  const profitPercent = action === "buy"
    ? ((exitPrice - entryPrice) / entryPrice) * 100
    : ((entryPrice - exitPrice) / entryPrice) * 100;
  const isWin = profit > 0;
  
  // Find matching strategy
  let strategy = brainState.strategies.find(s => 
    s.name === strategyName || 
    (s.symbols && s.symbols.includes(symbol))
  );
  
  if (!strategy && brainState.strategies.length > 0) {
    // Use default strategy if no match
    strategy = brainState.strategies[0];
  }
  
  if (strategy) {
    // Record performance
    await storage.recordStrategyPerformance({
      strategyId: strategy.id,
      tradeId: `trade-${Date.now()}`,
      symbol,
      profitLoss: profit,
      wasSuccessful: isWin,
      entryReason: strategyName || "Auto-trade",
      exitReason: isWin ? "Take profit" : "Stop loss",
      marketConditions: null,
    });
    
    // Update strategy metrics
    const newTotalTrades = (strategy.totalTrades || 0) + 1;
    const currentWins = Math.round(((strategy.winRate || 0) / 100) * (strategy.totalTrades || 0));
    const newWins = currentWins + (isWin ? 1 : 0);
    const newWinRate = (newWins / newTotalTrades) * 100;
    const newTotalProfit = (strategy.totalProfit || 0) + profit;
    
    await storage.updateStrategy(strategy.id, {
      totalTrades: newTotalTrades,
      winRate: newWinRate,
      totalProfit: newTotalProfit,
    });
    
    // Update local state
    const idx = brainState.strategies.findIndex(s => s.id === strategy!.id);
    if (idx >= 0) {
      brainState.strategies[idx] = {
        ...brainState.strategies[idx],
        totalTrades: newTotalTrades,
        winRate: newWinRate,
        totalProfit: newTotalProfit,
      };
    }
    
    // Check if strategy should be deactivated
    if (newTotalTrades >= 10 && newWinRate < 35) {
      console.log(`[Autopilot Brain] Deactivating low-performing strategy: ${strategy.name}`);
      await storage.updateStrategy(strategy.id, { isActive: false });
      
      await storage.createActivityLog({
        type: "system",
        action: "Strategy Deactivated",
        description: `Brain deactivated ${strategy.name} - Win rate ${newWinRate.toFixed(1)}% after ${newTotalTrades} trades`,
      });
    }
    
    // Boost successful strategies
    if (newTotalTrades >= 5 && newWinRate > 65) {
      const newConfidence = Math.min(90, (strategy.confidence || 50) + 5);
      await storage.updateStrategy(strategy.id, { confidence: newConfidence });
      console.log(`[Autopilot Brain] Boosted ${strategy.name} confidence to ${newConfidence}`);
    }
    
    console.log(`[Autopilot Brain] Updated ${strategy.name}: Win rate ${newWinRate.toFixed(1)}%, Total profit $${newTotalProfit.toFixed(2)}`);
  }
  
  // Update stock preference based on outcome - AGGRESSIVE PENALTY for losses
  const currentPref = brainState.stockPreferences[symbol] || 50;
  // Win: +5 points, Loss: -10 points (asymmetric to protect capital)
  const prefChange = isWin ? 5 : -10;
  const newPref = Math.max(0, Math.min(100, currentPref + prefChange));
  brainState.stockPreferences[symbol] = newPref;
  
  // Sync preferences to trade validator for blacklist enforcement
  tradeValidator.setStockPreferences(brainState.stockPreferences);
  
  // If preference drops below 20, blacklist the symbol temporarily
  if (newPref < 20) {
    console.log(`[Autopilot Brain] WARNING: ${symbol} preference dropped to ${newPref} - avoiding this stock`);
    await storage.createAlert({
      type: "warning",
      title: `Stock Blacklisted: ${symbol}`,
      message: `${symbol} has been temporarily blacklisted due to poor performance. Preference: ${newPref}/100`,
      requiresApproval: false,
    });
  }
  
  // IMMEDIATE confidence adjustment on strategy based on outcome
  if (strategy) {
    const confidenceChange = isWin ? 2 : -5; // Larger penalty for losses
    const newConfidence = Math.max(20, Math.min(90, (strategy.confidence || 50) + confidenceChange));
    
    if (newConfidence !== strategy.confidence) {
      await storage.updateStrategy(strategy.id, { confidence: newConfidence });
      console.log(`[Autopilot Brain] Adjusted ${strategy.name} confidence: ${strategy.confidence} -> ${newConfidence}`);
    }
  }
  
  // Save brain state
  await storage.updateAutopilotBrain({
    stockPreferences: JSON.stringify(brainState.stockPreferences),
    marketInsights: JSON.stringify(brainState.marketInsights),
    totalLearningCycles: brainState.learningCycles,
    overallWinRate: brainState.overallWinRate,
  });
}

// Guide Ato's trading style based on market conditions and performance
export async function guideAtoStyle(): Promise<{
  preferredSetups: string[];
  avoidPatterns: string[];
  entryRules: string[];
  exitRules: string[];
}> {
  console.log("[Autopilot Brain] Analyzing optimal trading style for Ato...");
  
  const trades = await storage.getTrades();
  const recentTrades = trades.slice(-30);
  const winningTrades = recentTrades.filter(t => t.status === "filled");
  
  const strategies = brainState.strategies.filter(s => s.isActive);
  const topStrategies = strategies.sort((a, b) => (b.winRate || 0) - (a.winRate || 0)).slice(0, 3);
  
  const prompt = `You are Autopilot, the intelligent brain guiding Ato the DAY TRADER.
CRITICAL: Ato is a day trader - ALL positions must be FLAT by market close. No overnight holds.

Key Day Trading Principles to enforce:
- Quick profits (1-2%) are better than waiting for bigger moves
- Tight stop losses (1% max) to prevent losses from compounding
- Exit all positions by 2:45 PM ET (11:45 AM PT) - FORT KNOX SCHEDULE
- Focus on first hour (9:35-10:35 AM ET) after our 5-min delayed start
- Maximum 5 trades per day to avoid overtrading

Active Strategies and Performance:
${topStrategies.map(s => `- ${s.name} (${s.type}): ${(s.winRate || 0).toFixed(1)}% win rate, ${s.totalTrades || 0} trades`).join("\n")}

Stock Preferences (higher = better historical performance):
${Object.entries(brainState.stockPreferences).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([s, v]) => `${s}: ${v}`).join(", ")}

Create optimal DAY TRADING guidelines for Ato. Respond in JSON:
{
  "preferredSetups": ["intraday setup1", "intraday setup2"],
  "avoidPatterns": ["pattern to avoid1", "pattern to avoid2"],
  "entryRules": ["quick entry rule1", "rule2"],
  "exitRules": ["quick exit rule1", "flat by close"],
  "focusSymbols": ["SYM1", "SYM2"],
  "maxDailyLoss": "dollar amount to stop trading for the day",
  "profitTarget": "quick profit percentage target"
}`;

  try {
    const response = await openai.chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      max_completion_tokens: 1024,
    });
    
    const content = response.choices[0]?.message?.content || "{}";
    const guidance = JSON.parse(content);
    
    await storage.createResearchLog({
      type: "recommendation",
      summary: `[Autopilot] Updated Ato's trading style: Focus on ${guidance.preferredSetups?.join(", ") || "momentum"}`,
      confidence: 80,
    });
    
    return {
      preferredSetups: guidance.preferredSetups || ["momentum breakout", "dip buying"],
      avoidPatterns: guidance.avoidPatterns || ["chasing", "revenge trading"],
      entryRules: guidance.entryRules || ["wait for confirmation", "check volume"],
      exitRules: guidance.exitRules || ["take partial profits", "trail stops"],
    };
  } catch (error) {
    console.error("[Autopilot Brain] Error guiding Ato:", error);
    return {
      preferredSetups: ["momentum breakout", "dip buying"],
      avoidPatterns: ["chasing extended moves", "overtrading"],
      entryRules: ["wait for confirmation", "verify trend"],
      exitRules: ["take partial profits", "honor stop losses"],
    };
  }
}

// Evaluate and potentially upgrade Ato's capabilities
export async function evaluateAndUpgrade(): Promise<string[]> {
  console.log("[Autopilot Brain] Evaluating Ato's performance for upgrades...");
  
  const upgrades: string[] = [];
  const trades = await storage.getTrades();
  const positions = await storage.getPositions();
  
  // Analyze recent performance
  const last20Trades = trades.slice(-20);
  const wins = last20Trades.filter(t => t.status === "filled").length;
  const winRate = trades.length > 0 ? (wins / last20Trades.length) * 100 : 0;
  
  // Check for improvement opportunities
  if (winRate < 40 && trades.length >= 10) {
    upgrades.push("Implement stricter entry criteria - win rate below target");
    await storage.createActivityLog({
      type: "system",
      action: "Ato Upgrade",
      description: "Autopilot recommends stricter entry criteria due to low win rate",
    });
  }
  
  if (positions.length > 3) {
    upgrades.push("Consider position concentration - too many open positions");
  }
  
  // Check strategy diversity
  const activeStrategies = brainState.strategies.filter(s => s.isActive);
  if (activeStrategies.length < 3) {
    upgrades.push("Add more strategy diversity for different market conditions");
    await createNewStrategy(
      "Volatility Play",
      "Capitalize on high volatility periods with quick scalps",
      "custom"
    );
  }
  
  console.log("[Autopilot Brain] Identified", upgrades.length, "potential upgrades");
  return upgrades;
}

// Filter recommendations using brain insights
export function filterRecommendations(
  recommendations: Array<{ symbol: string; action: string; confidence: number; reason: string }>
): Array<{ symbol: string; action: string; confidence: number; reason: string }> {
  const activeStrategies = brainState.strategies.filter(s => s.isActive);
  
  return recommendations
    .filter(rec => {
      // BLACKLIST CHECK: Filter out stocks with preference < 20
      const stockPref = brainState.stockPreferences[rec.symbol] || 50;
      if (stockPref < 20 && rec.action === "buy") {
        console.log(`[Autopilot Brain] Blocked ${rec.symbol} - blacklisted (preference: ${stockPref})`);
        return false;
      }
      return true;
    })
    .map(rec => {
      // Boost confidence for symbols with good history
      const stockPref = brainState.stockPreferences[rec.symbol] || 50;
      const prefBoost = (stockPref - 50) / 10; // -5 to +5 boost
      
      // Check if any active strategy targets this symbol
      const hasStrategy = activeStrategies.some(s => 
        s.symbols && s.symbols.includes(rec.symbol)
      );
      const strategyBoost = hasStrategy ? 5 : 0;
      
      const adjustedConfidence = Math.min(95, Math.max(10, 
        rec.confidence + prefBoost + strategyBoost
      ));
      
      return {
        ...rec,
        confidence: Math.round(adjustedConfidence),
        reason: `${rec.reason}${hasStrategy ? " [Strategy aligned]" : ""}`,
      };
    }).sort((a, b) => b.confidence - a.confidence);
}
