import { storage } from "./storage";
import type { Trade, ResearchLog } from "@shared/schema";

export interface StrategySignal {
  id: string;
  timestamp: Date;
  source: "autopilot";
  symbol: string;
  action: "buy" | "sell" | "hold";
  confidence: number;
  strategy: string;
  reasoning: string;
  targetEntry?: number;
  stopLoss?: number;
  takeProfit?: number;
  positionSize?: number;
}

export interface ExecutionFeedback {
  id: string;
  timestamp: Date;
  source: "ato";
  signalId?: string;
  symbol: string;
  action: "executed" | "skipped" | "failed" | "closed";
  executedPrice?: number;
  quantity?: number;
  profitLoss?: number;
  reason: string;
  success: boolean;
}

export interface LearningInsight {
  id: string;
  timestamp: Date;
  strategy: string;
  symbol: string;
  wasSuccessful: boolean;
  profitLoss: number;
  holdingTimeMinutes: number;
  entryConditions: string[];
  exitReason: string;
  marketConditions: string;
  lesson: string;
}

interface BusState {
  pendingSignals: StrategySignal[];
  recentFeedback: ExecutionFeedback[];
  learningInsights: LearningInsight[];
  strategyPerformance: Map<string, StrategyPerformance>;
  lastAutopilotUpdate: Date | null;
  lastAtoUpdate: Date | null;
}

interface StrategyPerformance {
  name: string;
  totalTrades: number;
  winCount: number;
  lossCount: number;
  totalProfitLoss: number;
  avgHoldingTime: number;
  successRate: number;
  lastUpdated: Date;
}

const state: BusState = {
  pendingSignals: [],
  recentFeedback: [],
  learningInsights: [],
  strategyPerformance: new Map(),
  lastAutopilotUpdate: null,
  lastAtoUpdate: null,
};

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

export function publishSignal(signal: Omit<StrategySignal, "id" | "timestamp" | "source">): StrategySignal {
  const fullSignal: StrategySignal = {
    ...signal,
    id: generateId(),
    timestamp: new Date(),
    source: "autopilot",
  };
  
  state.pendingSignals.push(fullSignal);
  state.lastAutopilotUpdate = new Date();
  
  if (state.pendingSignals.length > 50) {
    state.pendingSignals = state.pendingSignals.slice(-50);
  }
  
  console.log(`[TradeBus] Autopilot signal: ${signal.action.toUpperCase()} ${signal.symbol} (${signal.strategy}, ${signal.confidence}% confidence)`);
  
  storage.createResearchLog({
    type: "signal",
    summary: `[Signal] ${signal.action.toUpperCase()} ${signal.symbol}`,
    details: JSON.stringify(fullSignal),
    confidence: signal.confidence,
  });
  
  return fullSignal;
}

export function getSignalsForSymbol(symbol: string): StrategySignal[] {
  return state.pendingSignals.filter(s => s.symbol === symbol);
}

export function getPendingSignals(): StrategySignal[] {
  return [...state.pendingSignals];
}

export function consumeSignal(signalId: string): StrategySignal | null {
  const index = state.pendingSignals.findIndex(s => s.id === signalId);
  if (index >= 0) {
    const [signal] = state.pendingSignals.splice(index, 1);
    return signal;
  }
  return null;
}

export function publishFeedback(feedback: Omit<ExecutionFeedback, "id" | "timestamp" | "source">): ExecutionFeedback {
  const fullFeedback: ExecutionFeedback = {
    ...feedback,
    id: generateId(),
    timestamp: new Date(),
    source: "ato",
  };
  
  state.recentFeedback.push(fullFeedback);
  state.lastAtoUpdate = new Date();
  
  if (state.recentFeedback.length > 100) {
    state.recentFeedback = state.recentFeedback.slice(-100);
  }
  
  console.log(`[TradeBus] Ato feedback: ${feedback.action} ${feedback.symbol} - ${feedback.reason}`);
  
  if (fullFeedback.action === "closed" && fullFeedback.profitLoss !== undefined) {
    updateStrategyPerformance(fullFeedback);
  }
  
  storage.createActivityLog({
    type: feedback.success ? "trade" : "error",
    action: `Trade ${feedback.action}`,
    description: `${feedback.symbol}: ${feedback.reason}${feedback.profitLoss !== undefined ? ` (P/L: $${feedback.profitLoss.toFixed(2)})` : ""}`,
  });
  
  return fullFeedback;
}

export function getRecentFeedback(limit: number = 20): ExecutionFeedback[] {
  return state.recentFeedback.slice(-limit);
}

function updateStrategyPerformance(feedback: ExecutionFeedback): void {
  const signal = state.pendingSignals.find(s => s.id === feedback.signalId) || 
    state.recentFeedback.find(f => f.signalId === feedback.signalId);
  
  const strategyName = "General";
  
  let perf = state.strategyPerformance.get(strategyName);
  if (!perf) {
    perf = {
      name: strategyName,
      totalTrades: 0,
      winCount: 0,
      lossCount: 0,
      totalProfitLoss: 0,
      avgHoldingTime: 0,
      successRate: 0,
      lastUpdated: new Date(),
    };
    state.strategyPerformance.set(strategyName, perf);
  }
  
  perf.totalTrades++;
  perf.totalProfitLoss += feedback.profitLoss || 0;
  
  if ((feedback.profitLoss || 0) > 0) {
    perf.winCount++;
  } else {
    perf.lossCount++;
  }
  
  perf.successRate = perf.totalTrades > 0 ? (perf.winCount / perf.totalTrades) * 100 : 0;
  perf.lastUpdated = new Date();
}

export function recordLearningInsight(insight: Omit<LearningInsight, "id" | "timestamp">): void {
  const fullInsight: LearningInsight = {
    ...insight,
    id: generateId(),
    timestamp: new Date(),
  };
  
  state.learningInsights.push(fullInsight);
  
  if (state.learningInsights.length > 200) {
    state.learningInsights = state.learningInsights.slice(-200);
  }
  
  console.log(`[TradeBus] Learning: ${insight.strategy} on ${insight.symbol} - ${insight.lesson}`);
  
  storage.createResearchLog({
    type: "learning",
    summary: `[Learning] ${insight.strategy}: ${insight.lesson}`,
    details: JSON.stringify(fullInsight),
    confidence: insight.wasSuccessful ? 80 : 40,
  });
}

export function getLearningInsights(strategy?: string): LearningInsight[] {
  if (strategy) {
    return state.learningInsights.filter(i => i.strategy === strategy);
  }
  return [...state.learningInsights];
}

export function getStrategyPerformance(): Map<string, StrategyPerformance> {
  return new Map(state.strategyPerformance);
}

export function getAllStrategyPerformance(): StrategyPerformance[] {
  return Array.from(state.strategyPerformance.values());
}

export interface CommunicationSummary {
  autopilotActive: boolean;
  atoActive: boolean;
  pendingSignals: number;
  recentFeedbackCount: number;
  learningInsightsCount: number;
  strategiesTracked: number;
  lastAutopilotUpdate: Date | null;
  lastAtoUpdate: Date | null;
  topStrategies: { name: string; successRate: number; trades: number }[];
}

export function getCommunicationSummary(): CommunicationSummary {
  const now = new Date();
  const fiveMinutesAgo = new Date(now.getTime() - 5 * 60 * 1000);
  
  const topStrategies = getAllStrategyPerformance()
    .sort((a, b) => b.successRate - a.successRate)
    .slice(0, 5)
    .map(s => ({ name: s.name, successRate: s.successRate, trades: s.totalTrades }));
  
  return {
    autopilotActive: state.lastAutopilotUpdate ? state.lastAutopilotUpdate > fiveMinutesAgo : false,
    atoActive: state.lastAtoUpdate ? state.lastAtoUpdate > fiveMinutesAgo : false,
    pendingSignals: state.pendingSignals.length,
    recentFeedbackCount: state.recentFeedback.length,
    learningInsightsCount: state.learningInsights.length,
    strategiesTracked: state.strategyPerformance.size,
    lastAutopilotUpdate: state.lastAutopilotUpdate,
    lastAtoUpdate: state.lastAtoUpdate,
    topStrategies,
  };
}

export async function aggregateTradeAnalytics(): Promise<{
  closedTrades: number;
  totalProfitLoss: number;
  winRate: number;
  profitFactor: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  tradingDays: number;
}> {
  const trades = await storage.getTrades();
  const closedTrades = trades.filter(t => t.status === "filled");
  
  const buyTrades = closedTrades.filter(t => t.side === "buy");
  const sellTrades = closedTrades.filter(t => t.side === "sell");
  
  let wins = 0;
  let losses = 0;
  let totalProfit = 0;
  let totalLoss = 0;
  let largestWin = 0;
  let largestLoss = 0;
  
  for (const sell of sellTrades) {
    const matchingBuy = buyTrades.find(b => b.symbol === sell.symbol);
    if (matchingBuy) {
      const profitLoss = (sell.price - matchingBuy.price) * sell.quantity;
      if (profitLoss > 0) {
        wins++;
        totalProfit += profitLoss;
        largestWin = Math.max(largestWin, profitLoss);
      } else {
        losses++;
        totalLoss += Math.abs(profitLoss);
        largestLoss = Math.max(largestLoss, Math.abs(profitLoss));
      }
    }
  }
  
  const totalClosedSells = sellTrades.length;
  
  return {
    closedTrades: totalClosedSells,
    totalProfitLoss: totalProfit - totalLoss,
    winRate: totalClosedSells > 0 ? (wins / totalClosedSells) * 100 : 0,
    profitFactor: totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? Infinity : 0,
    avgWin: wins > 0 ? totalProfit / wins : 0,
    avgLoss: losses > 0 ? totalLoss / losses : 0,
    largestWin,
    largestLoss,
    tradingDays: new Set(closedTrades.map(t => t.timestamp?.toDateString()).filter(Boolean)).size,
  };
}

export function notifyAutopilot(message: string, data?: any): void {
  console.log(`[TradeBus] -> Autopilot: ${message}`);
  state.lastAtoUpdate = new Date();
}

export function notifyAto(message: string, data?: any): void {
  console.log(`[TradeBus] -> Ato: ${message}`);
  state.lastAutopilotUpdate = new Date();
}
