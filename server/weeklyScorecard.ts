/**
 * P4 Weekly Scorecard Generator
 * Analyzes completed trades and generates performance metrics for tuning
 */

import * as fs from "fs";
import * as path from "path";
import * as tradeLifecycle from "./tradeLifecycle";
import type { TradeRecord, TradeMetadata, TimeWindow, ExitReason } from "./tradeLifecycle";
import * as skipCounters from "./skipCounters";

// Reports directory
const REPORTS_DIR = "./reports";

export interface RegimeStats {
  regime: "bull" | "bear" | "chop";
  trades: number;
  wins: number;
  losses: number;
  winRate: number;
  totalPnl: number;
  avgPnl: number;
  expectancy: number;
  profitFactor: number;
}

export interface TimeWindowStats {
  timeWindow: TimeWindow;
  trades: number;
  wins: number;
  losses: number;
  winRate: number;
  totalPnl: number;
  avgPnl: number;
}

export interface StrategyStats {
  strategy: string;
  trades: number;
  wins: number;
  losses: number;
  winRate: number;
  totalPnl: number;
  avgPnl: number;
  avgSlippageBps: number;
}

export interface ExitReasonStats {
  exitReason: ExitReason;
  count: number;
  pctOfTotal: number;
  avgPnl: number;
  totalPnl: number;
}

export interface ATRAnalysis {
  avgAtr: number;
  avgAtrPct: number;
  avgStopDistance: number;
  avgRR: number;
  fallbackUsageRate: number;
}

export interface SlippageAnalysis {
  avgEntrySlippageBps: number;
  avgExitSlippageBps: number;
  medianEntrySlippageBps: number;
  worstEntrySlippageBps: number;
}

export interface SkipReasonStats {
  reason: string;
  count: number;
  pctOfTotal: number;
}

export interface WeeklyScorecard {
  periodStart: string;
  periodEnd: string;
  generatedAt: string;
  
  summary: {
    totalTrades: number;
    wins: number;
    losses: number;
    winRate: number;
    totalPnl: number;
    avgPnl: number;
    expectancy: number;
    profitFactor: number;
    sharpeEstimate: number;
  };
  
  byRegime: RegimeStats[];
  byTimeWindow: TimeWindowStats[];
  byStrategy: StrategyStats[];
  byExitReason: ExitReasonStats[];
  
  atrAnalysis: ATRAnalysis;
  slippageAnalysis: SlippageAnalysis;
  skipReasons: SkipReasonStats[];
  
  recommendations: string[];
}

function calculateExpectancy(wins: number, losses: number, avgWin: number, avgLoss: number): number {
  const totalTrades = wins + losses;
  if (totalTrades === 0) return 0;
  
  const winRate = wins / totalTrades;
  const lossRate = losses / totalTrades;
  
  return (winRate * avgWin) - (lossRate * Math.abs(avgLoss));
}

function calculateProfitFactor(grossProfit: number, grossLoss: number): number {
  if (grossLoss === 0) return grossProfit > 0 ? Infinity : 0;
  return Math.abs(grossProfit / grossLoss);
}

function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Get trades from a specific period
 */
function getTradesInPeriod(trades: TradeRecord[], startDate: Date, endDate: Date): TradeRecord[] {
  return trades.filter(trade => {
    if (!trade.exitFilledAt) return false;
    const exitDate = new Date(trade.exitFilledAt);
    return exitDate >= startDate && exitDate <= endDate;
  });
}

/**
 * Generate weekly scorecard from completed trades
 */
export function generateWeeklyScorecard(
  startDate: Date = getLastWeekStart(),
  endDate: Date = new Date()
): WeeklyScorecard {
  const allTrades = tradeLifecycle.getCompletedTrades();
  const periodTrades = getTradesInPeriod(allTrades, startDate, endDate);
  
  const closedTrades = periodTrades.filter(t => 
    t.status === "closed" && t.realizedPnl !== null
  );
  
  const recommendations: string[] = [];
  
  // Summary stats
  const wins = closedTrades.filter(t => t.realizedPnl! >= 0);
  const losses = closedTrades.filter(t => t.realizedPnl! < 0);
  const totalPnl = closedTrades.reduce((sum, t) => sum + (t.realizedPnl || 0), 0);
  const grossProfit = wins.reduce((sum, t) => sum + (t.realizedPnl || 0), 0);
  const grossLoss = losses.reduce((sum, t) => sum + (t.realizedPnl || 0), 0);
  
  const avgWin = wins.length > 0 ? grossProfit / wins.length : 0;
  const avgLoss = losses.length > 0 ? grossLoss / losses.length : 0;
  const winRate = closedTrades.length > 0 ? wins.length / closedTrades.length : 0;
  
  const pnlArray = closedTrades.map(t => t.realizedPnl || 0);
  const avgPnl = pnlArray.length > 0 ? totalPnl / pnlArray.length : 0;
  const stdDev = pnlArray.length > 1 
    ? Math.sqrt(pnlArray.reduce((sum, p) => sum + Math.pow(p - avgPnl, 2), 0) / (pnlArray.length - 1))
    : 0;
  const sharpeEstimate = stdDev > 0 ? (avgPnl / stdDev) * Math.sqrt(252) : 0;
  
  // By Regime (handle legacy trades without metadata as "chop")
  const regimeStats: RegimeStats[] = (["bull", "bear", "chop"] as const).map(regime => {
    const regimeTrades = closedTrades.filter(t => {
      const tradeRegime = t.metadata?.regime || "chop"; // Legacy trades default to chop
      return tradeRegime === regime;
    });
    const regimeWins = regimeTrades.filter(t => t.realizedPnl! >= 0);
    const regimeLosses = regimeTrades.filter(t => t.realizedPnl! < 0);
    const regimeTotalPnl = regimeTrades.reduce((sum, t) => sum + (t.realizedPnl || 0), 0);
    const regimeGrossProfit = regimeWins.reduce((sum, t) => sum + (t.realizedPnl || 0), 0);
    const regimeGrossLoss = regimeLosses.reduce((sum, t) => sum + (t.realizedPnl || 0), 0);
    
    const regimeAvgWin = regimeWins.length > 0 ? regimeGrossProfit / regimeWins.length : 0;
    const regimeAvgLoss = regimeLosses.length > 0 ? regimeGrossLoss / regimeLosses.length : 0;
    
    return {
      regime,
      trades: regimeTrades.length,
      wins: regimeWins.length,
      losses: regimeLosses.length,
      winRate: regimeTrades.length > 0 ? regimeWins.length / regimeTrades.length : 0,
      totalPnl: regimeTotalPnl,
      avgPnl: regimeTrades.length > 0 ? regimeTotalPnl / regimeTrades.length : 0,
      expectancy: calculateExpectancy(regimeWins.length, regimeLosses.length, regimeAvgWin, regimeAvgLoss),
      profitFactor: calculateProfitFactor(regimeGrossProfit, regimeGrossLoss),
    };
  });
  
  // Regime recommendations
  for (const rs of regimeStats) {
    if (rs.trades >= 5 && rs.expectancy < 0) {
      recommendations.push(`Consider disabling trading in ${rs.regime} regime (expectancy: $${rs.expectancy.toFixed(2)})`);
    }
  }
  
  // By Time Window (handle legacy trades without metadata as "mid")
  const timeWindowStats: TimeWindowStats[] = (["open", "mid", "close"] as const).map(window => {
    const windowTrades = closedTrades.filter(t => {
      const tradeWindow = t.metadata?.timeWindow || "mid"; // Legacy trades default to mid
      return tradeWindow === window;
    });
    const windowWins = windowTrades.filter(t => t.realizedPnl! >= 0);
    const windowLosses = windowTrades.filter(t => t.realizedPnl! < 0);
    const windowTotalPnl = windowTrades.reduce((sum, t) => sum + (t.realizedPnl || 0), 0);
    
    return {
      timeWindow: window,
      trades: windowTrades.length,
      wins: windowWins.length,
      losses: windowLosses.length,
      winRate: windowTrades.length > 0 ? windowWins.length / windowTrades.length : 0,
      totalPnl: windowTotalPnl,
      avgPnl: windowTrades.length > 0 ? windowTotalPnl / windowTrades.length : 0,
    };
  });
  
  // Time window recommendations
  const openWindow = timeWindowStats.find(t => t.timeWindow === "open");
  const closeWindow = timeWindowStats.find(t => t.timeWindow === "close");
  if (openWindow && openWindow.trades >= 5 && openWindow.avgPnl < 0) {
    recommendations.push(`Opening window (9:30-10:30 AM) shows negative performance. Consider delaying entries.`);
  }
  if (closeWindow && closeWindow.trades >= 3 && closeWindow.avgPnl < 0) {
    recommendations.push(`Close window (3:00-4:00 PM) shows negative performance. Consider earlier force close.`);
  }
  
  // By Strategy
  const strategies = Array.from(new Set(closedTrades.map(t => t.strategy || "unknown")));
  const strategyStats: StrategyStats[] = strategies.map(strategy => {
    const strategyTrades = closedTrades.filter(t => t.strategy === strategy);
    const strategyWins = strategyTrades.filter(t => t.realizedPnl! >= 0);
    const strategyLosses = strategyTrades.filter(t => t.realizedPnl! < 0);
    const strategyTotalPnl = strategyTrades.reduce((sum, t) => sum + (t.realizedPnl || 0), 0);
    
    const slippages = strategyTrades
      .filter(t => t.entrySlippage?.slippageBps !== undefined)
      .map(t => t.entrySlippage!.slippageBps);
    const avgSlippageBps = slippages.length > 0 
      ? slippages.reduce((sum, s) => sum + s, 0) / slippages.length 
      : 0;
    
    return {
      strategy,
      trades: strategyTrades.length,
      wins: strategyWins.length,
      losses: strategyLosses.length,
      winRate: strategyTrades.length > 0 ? strategyWins.length / strategyTrades.length : 0,
      totalPnl: strategyTotalPnl,
      avgPnl: strategyTrades.length > 0 ? strategyTotalPnl / strategyTrades.length : 0,
      avgSlippageBps,
    };
  });
  
  // By Exit Reason (handle legacy trades without exitReason)
  const exitReasons: ExitReason[] = ["TP", "SL", "TIME", "MANUAL", "RECONCILE", "FORCE_CLOSE"];
  const exitReasonStats: ExitReasonStats[] = exitReasons.map(reason => {
    const reasonTrades = closedTrades.filter(t => {
      const tradeExitReason = t.metadata?.exitReason || "RECONCILE"; // Legacy trades default to RECONCILE
      return tradeExitReason === reason;
    });
    const reasonTotalPnl = reasonTrades.reduce((sum, t) => sum + (t.realizedPnl || 0), 0);
    
    return {
      exitReason: reason,
      count: reasonTrades.length,
      pctOfTotal: closedTrades.length > 0 ? reasonTrades.length / closedTrades.length : 0,
      avgPnl: reasonTrades.length > 0 ? reasonTotalPnl / reasonTrades.length : 0,
      totalPnl: reasonTotalPnl,
    };
  }).filter(s => s.count > 0);
  
  // ATR Analysis (only from trades with valid metadata)
  const tradesWithMetadata = closedTrades.filter(t => t.metadata !== null && t.metadata !== undefined);
  const atrValues = tradesWithMetadata
    .filter(t => t.metadata!.atr !== null && t.metadata!.atr !== undefined && !isNaN(t.metadata!.atr))
    .map(t => t.metadata!.atr!);
  const atrPctValues = tradesWithMetadata
    .filter(t => t.metadata!.atrPct !== null && t.metadata!.atrPct !== undefined && !isNaN(t.metadata!.atrPct))
    .map(t => t.metadata!.atrPct!);
  const stopDistances = tradesWithMetadata
    .filter(t => t.metadata!.stopDistance !== null && t.metadata!.stopDistance !== undefined && !isNaN(t.metadata!.stopDistance))
    .map(t => t.metadata!.stopDistance!);
  const rrValues = tradesWithMetadata
    .filter(t => t.metadata!.rr !== null && t.metadata!.rr !== undefined && !isNaN(t.metadata!.rr))
    .map(t => t.metadata!.rr!);
  const fallbackCount = tradesWithMetadata.filter(t => t.metadata!.usedAtrFallback === true).length;
  
  const atrAnalysis: ATRAnalysis = {
    avgAtr: atrValues.length > 0 ? atrValues.reduce((a, b) => a + b, 0) / atrValues.length : 0,
    avgAtrPct: atrPctValues.length > 0 ? atrPctValues.reduce((a, b) => a + b, 0) / atrPctValues.length : 0,
    avgStopDistance: stopDistances.length > 0 ? stopDistances.reduce((a, b) => a + b, 0) / stopDistances.length : 0,
    avgRR: rrValues.length > 0 ? rrValues.reduce((a, b) => a + b, 0) / rrValues.length : 0,
    fallbackUsageRate: tradesWithMetadata.length > 0 ? fallbackCount / tradesWithMetadata.length : 0,
  };
  
  if (atrAnalysis.fallbackUsageRate > 0.2) {
    recommendations.push(`High ATR fallback rate (${(atrAnalysis.fallbackUsageRate * 100).toFixed(0)}%). Check indicator pipeline.`);
  }
  
  // Slippage Analysis
  const entrySlippages = closedTrades
    .filter(t => t.entrySlippage?.slippageBps !== undefined)
    .map(t => t.entrySlippage!.slippageBps);
  const exitSlippages = closedTrades
    .filter(t => t.exitSlippage?.slippageBps !== undefined)
    .map(t => t.exitSlippage!.slippageBps);
  
  const slippageAnalysis: SlippageAnalysis = {
    avgEntrySlippageBps: entrySlippages.length > 0 
      ? entrySlippages.reduce((a, b) => a + b, 0) / entrySlippages.length 
      : 0,
    avgExitSlippageBps: exitSlippages.length > 0 
      ? exitSlippages.reduce((a, b) => a + b, 0) / exitSlippages.length 
      : 0,
    medianEntrySlippageBps: median(entrySlippages),
    worstEntrySlippageBps: entrySlippages.length > 0 ? Math.max(...entrySlippages) : 0,
  };
  
  if (slippageAnalysis.avgEntrySlippageBps > 5) {
    recommendations.push(`High entry slippage (${slippageAnalysis.avgEntrySlippageBps.toFixed(1)}bps). Consider tighter spread gates.`);
  }
  
  // Skip Reasons from counters
  const skipCounts = skipCounters.getSkipCounts();
  const totalSkips = skipCounters.getTotalSkips();
  const skipReasons: SkipReasonStats[] = [];
  skipCounts.forEach((count, reason) => {
    skipReasons.push({
      reason,
      count,
      pctOfTotal: totalSkips > 0 ? count / totalSkips : 0,
    });
  });
  skipReasons.sort((a, b) => b.count - a.count);
  
  return {
    periodStart: startDate.toISOString(),
    periodEnd: endDate.toISOString(),
    generatedAt: new Date().toISOString(),
    
    summary: {
      totalTrades: closedTrades.length,
      wins: wins.length,
      losses: losses.length,
      winRate,
      totalPnl,
      avgPnl,
      expectancy: calculateExpectancy(wins.length, losses.length, avgWin, avgLoss),
      profitFactor: calculateProfitFactor(grossProfit, grossLoss),
      sharpeEstimate,
    },
    
    byRegime: regimeStats,
    byTimeWindow: timeWindowStats,
    byStrategy: strategyStats,
    byExitReason: exitReasonStats,
    
    atrAnalysis,
    slippageAnalysis,
    skipReasons,
    
    recommendations,
  };
}

/**
 * Get start of last week (Monday 00:00)
 */
function getLastWeekStart(): Date {
  const now = new Date();
  const dayOfWeek = now.getDay();
  const daysToMonday = dayOfWeek === 0 ? 6 : dayOfWeek - 1;
  const lastMonday = new Date(now);
  lastMonday.setDate(now.getDate() - daysToMonday - 7);
  lastMonday.setHours(0, 0, 0, 0);
  return lastMonday;
}

/**
 * Print scorecard to console in readable format
 */
export function printScorecard(scorecard: WeeklyScorecard): void {
  console.log("\n=== WEEKLY SCORECARD ===");
  console.log(`Period: ${scorecard.periodStart.split('T')[0]} to ${scorecard.periodEnd.split('T')[0]}`);
  console.log(`Generated: ${scorecard.generatedAt}`);
  
  console.log("\n--- SUMMARY ---");
  const s = scorecard.summary;
  console.log(`Trades: ${s.totalTrades} (${s.wins}W / ${s.losses}L)`);
  console.log(`Win Rate: ${(s.winRate * 100).toFixed(1)}%`);
  console.log(`Total P&L: $${s.totalPnl.toFixed(2)}`);
  console.log(`Avg P&L: $${s.avgPnl.toFixed(2)}`);
  console.log(`Expectancy: $${s.expectancy.toFixed(2)}`);
  console.log(`Profit Factor: ${s.profitFactor === Infinity ? "∞" : s.profitFactor.toFixed(2)}`);
  console.log(`Sharpe Est: ${s.sharpeEstimate.toFixed(2)}`);
  
  console.log("\n--- BY REGIME ---");
  for (const r of scorecard.byRegime) {
    if (r.trades === 0) continue;
    console.log(`${r.regime.toUpperCase()}: ${r.trades} trades, ${(r.winRate * 100).toFixed(0)}% WR, $${r.totalPnl.toFixed(2)} P&L, E[$${r.expectancy.toFixed(2)}]`);
  }
  
  console.log("\n--- BY TIME WINDOW ---");
  for (const t of scorecard.byTimeWindow) {
    if (t.trades === 0) continue;
    console.log(`${t.timeWindow.toUpperCase()}: ${t.trades} trades, ${(t.winRate * 100).toFixed(0)}% WR, $${t.totalPnl.toFixed(2)} P&L`);
  }
  
  console.log("\n--- BY STRATEGY ---");
  for (const st of scorecard.byStrategy) {
    console.log(`${st.strategy}: ${st.trades} trades, ${(st.winRate * 100).toFixed(0)}% WR, $${st.totalPnl.toFixed(2)} P&L, ${st.avgSlippageBps.toFixed(1)}bps slip`);
  }
  
  console.log("\n--- EXIT REASONS ---");
  for (const e of scorecard.byExitReason) {
    console.log(`${e.exitReason}: ${e.count} (${(e.pctOfTotal * 100).toFixed(0)}%), avg $${e.avgPnl.toFixed(2)}`);
  }
  
  console.log("\n--- ATR ANALYSIS ---");
  const a = scorecard.atrAnalysis;
  console.log(`Avg ATR: $${a.avgAtr.toFixed(2)} (${a.avgAtrPct.toFixed(2)}%)`);
  console.log(`Avg Stop Distance: $${a.avgStopDistance.toFixed(2)}`);
  console.log(`Avg R:R: ${a.avgRR.toFixed(2)}`);
  console.log(`ATR Fallback Rate: ${(a.fallbackUsageRate * 100).toFixed(0)}%`);
  
  console.log("\n--- SLIPPAGE ---");
  const sl = scorecard.slippageAnalysis;
  console.log(`Entry: avg ${sl.avgEntrySlippageBps.toFixed(1)}bps, median ${sl.medianEntrySlippageBps.toFixed(1)}bps, worst ${sl.worstEntrySlippageBps.toFixed(1)}bps`);
  console.log(`Exit: avg ${sl.avgExitSlippageBps.toFixed(1)}bps`);
  
  if (scorecard.skipReasons.length > 0) {
    console.log("\n--- TOP SKIP REASONS ---");
    const top5 = scorecard.skipReasons.slice(0, 5);
    for (const sk of top5) {
      console.log(`${sk.reason}: ${sk.count} (${(sk.pctOfTotal * 100).toFixed(0)}%)`);
    }
  }
  
  if (scorecard.recommendations.length > 0) {
    console.log("\n--- RECOMMENDATIONS ---");
    for (const rec of scorecard.recommendations) {
      console.log(`* ${rec}`);
    }
  }
  
  console.log("\n=== END SCORECARD ===\n");
}

/**
 * Generate and print weekly scorecard
 */
export function runWeeklyReport(): WeeklyScorecard {
  const scorecard = generateWeeklyScorecard();
  printScorecard(scorecard);
  return scorecard;
}

/**
 * Format date as YYYY-MM-DD
 */
function formatDate(date: Date): string {
  return date.toISOString().split('T')[0];
}

/**
 * Generate text summary for file output
 */
function generateSummaryText(scorecard: WeeklyScorecard): string {
  const lines: string[] = [];
  lines.push("=== WEEKLY SCORECARD ===");
  lines.push(`Period: ${scorecard.periodStart.split('T')[0]} to ${scorecard.periodEnd.split('T')[0]}`);
  lines.push(`Generated: ${scorecard.generatedAt}`);
  lines.push("");
  
  lines.push("--- SUMMARY ---");
  const s = scorecard.summary;
  lines.push(`Trades: ${s.totalTrades} (${s.wins}W / ${s.losses}L)`);
  lines.push(`Win Rate: ${(s.winRate * 100).toFixed(1)}%`);
  lines.push(`Total P&L: $${s.totalPnl.toFixed(2)}`);
  lines.push(`Avg P&L: $${s.avgPnl.toFixed(2)}`);
  lines.push(`Expectancy: $${s.expectancy.toFixed(2)}`);
  lines.push(`Profit Factor: ${s.profitFactor === Infinity ? "∞" : s.profitFactor.toFixed(2)}`);
  lines.push(`Sharpe Est: ${s.sharpeEstimate.toFixed(2)}`);
  lines.push("");
  
  lines.push("--- BY REGIME ---");
  for (const r of scorecard.byRegime) {
    if (r.trades === 0) continue;
    lines.push(`${r.regime.toUpperCase()}: ${r.trades} trades, ${(r.winRate * 100).toFixed(0)}% WR, $${r.totalPnl.toFixed(2)} P&L, E[$${r.expectancy.toFixed(2)}]`);
  }
  lines.push("");
  
  lines.push("--- BY TIME WINDOW ---");
  for (const t of scorecard.byTimeWindow) {
    if (t.trades === 0) continue;
    lines.push(`${t.timeWindow.toUpperCase()}: ${t.trades} trades, ${(t.winRate * 100).toFixed(0)}% WR, $${t.totalPnl.toFixed(2)} P&L`);
  }
  lines.push("");
  
  lines.push("--- BY STRATEGY ---");
  for (const st of scorecard.byStrategy) {
    lines.push(`${st.strategy}: ${st.trades} trades, ${(st.winRate * 100).toFixed(0)}% WR, $${st.totalPnl.toFixed(2)} P&L, ${st.avgSlippageBps.toFixed(1)}bps slip`);
  }
  lines.push("");
  
  lines.push("--- EXIT REASONS ---");
  for (const e of scorecard.byExitReason) {
    lines.push(`${e.exitReason}: ${e.count} (${(e.pctOfTotal * 100).toFixed(0)}%), avg $${e.avgPnl.toFixed(2)}`);
  }
  lines.push("");
  
  lines.push("--- ATR ANALYSIS ---");
  const a = scorecard.atrAnalysis;
  lines.push(`Avg ATR: $${a.avgAtr.toFixed(2)} (${a.avgAtrPct.toFixed(2)}%)`);
  lines.push(`Avg Stop Distance: $${a.avgStopDistance.toFixed(2)}`);
  lines.push(`Avg R:R: ${a.avgRR.toFixed(2)}`);
  lines.push(`ATR Fallback Rate: ${(a.fallbackUsageRate * 100).toFixed(0)}%`);
  lines.push("");
  
  lines.push("--- SLIPPAGE ---");
  const sl = scorecard.slippageAnalysis;
  lines.push(`Entry: avg ${sl.avgEntrySlippageBps.toFixed(1)}bps, median ${sl.medianEntrySlippageBps.toFixed(1)}bps, worst ${sl.worstEntrySlippageBps.toFixed(1)}bps`);
  lines.push(`Exit: avg ${sl.avgExitSlippageBps.toFixed(1)}bps`);
  lines.push("");
  
  if (scorecard.skipReasons.length > 0) {
    lines.push("--- TOP SKIP REASONS ---");
    const top5 = scorecard.skipReasons.slice(0, 5);
    for (const sk of top5) {
      lines.push(`${sk.reason}: ${sk.count} (${(sk.pctOfTotal * 100).toFixed(0)}%)`);
    }
    lines.push("");
  }
  
  if (scorecard.recommendations.length > 0) {
    lines.push("--- RECOMMENDATIONS ---");
    for (const rec of scorecard.recommendations) {
      lines.push(`* ${rec}`);
    }
    lines.push("");
  }
  
  lines.push("=== END SCORECARD ===");
  return lines.join("\n");
}

/**
 * Save scorecard to files (JSON + summary text)
 */
export function saveWeeklyScorecardToFiles(scorecard: WeeklyScorecard): { jsonPath: string; txtPath: string } {
  if (!fs.existsSync(REPORTS_DIR)) {
    fs.mkdirSync(REPORTS_DIR, { recursive: true });
  }
  
  const dateStr = formatDate(new Date(scorecard.periodEnd));
  const jsonPath = path.join(REPORTS_DIR, `weekly_${dateStr}.json`);
  const txtPath = path.join(REPORTS_DIR, `weekly_${dateStr}_summary.txt`);
  
  fs.writeFileSync(jsonPath, JSON.stringify(scorecard, null, 2));
  fs.writeFileSync(txtPath, generateSummaryText(scorecard));
  
  console.log(`[SCORECARD] Saved: ${jsonPath}`);
  console.log(`[SCORECARD] Saved: ${txtPath}`);
  
  return { jsonPath, txtPath };
}

/**
 * Generate, print, and save weekly scorecard
 */
export function generateAndSaveWeeklyScorecard(): WeeklyScorecard {
  const scorecard = generateWeeklyScorecard();
  printScorecard(scorecard);
  saveWeeklyScorecardToFiles(scorecard);
  return scorecard;
}
