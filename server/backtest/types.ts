/**
 * Backtest Framework Type Definitions
 */

import type { StrategyName } from "../strategyEngine";
import type { RegimeLabel } from "../regimeFilter";

// ─── Bar Types ───

export interface AlpacaBar {
  t: string;   // ISO timestamp
  o: number;   // open
  h: number;   // high
  l: number;   // low
  c: number;   // close
  v: number;   // volume
}

/** Cached data for a single symbol across the backtest period */
export interface SymbolBarData {
  symbol: string;
  bars5min: AlpacaBar[];
  bars1min: AlpacaBar[];
  fetchedAt: string;
}

// ─── Configuration ───

export interface BacktestConfig {
  startDate: string;           // YYYY-MM-DD
  endDate: string;             // YYYY-MM-DD
  symbols: string[];
  strategies: StrategyName[];
  initialEquity: number;

  // Risk parameters
  riskPerTradePct: number;
  maxPositionPct: number;
  maxOpenPositions: number;
  maxEntriesPerDay: number;
  cooldownMinutes: number;
  minConfidence: number;
  maxConsecutiveLosses: number;

  // Position management
  partialProfitPct: number;
  partialProfitRMultiple: number;  // R-multiple at which to take partial profit
  tradeTimeoutMinutes: number;
  breakevenTriggerPct: number;
  breakevenOffsetPct: number;

  // ATR sizing
  atrStopMultiplier: number;
  rewardRiskRatio: number;
  minStopPct: number;
  maxStopPct: number;

  // Slippage model
  slippageMode: "none" | "fixed" | "pctOfSpread";
  slippageFixedBps: number;
  slippagePctOfSpread: number;

  // Commission
  commissionPerShare: number;

  // Trading window (minutes since midnight ET)
  entryWindowStartMinutes: number;   // 575 = 9:35 AM
  entryWindowEndMinutes: number;     // 695 = 11:35 AM
  forceCloseMinutes: number;         // 945 = 3:45 PM

  // Daily P&L limits
  dailyMaxLoss: number;
  dailyMaxProfit: number;

  // Data
  warmupBars: number;
  cachePath: string;

  // Output
  outputPath: string;
  outputFormat: ("json" | "csv")[];

  // Flags
  verbose: boolean;
  noCache: boolean;
}

export const DEFAULT_CONFIG: BacktestConfig = {
  startDate: "",
  endDate: "",
  symbols: [
    // 18-symbol universe: removed COIN (-$56K) and AMD (-$62K) — net destroyers
    "SLV", "NVDA", "QQQ", "TSLA", "AAPL", "MSFT", "SPY", "META",
    "AMZN", "GOOG", "NFLX", "BA", "JPM", "GS",
    "UBER", "PLTR", "XOM", "DIS"
  ],
  strategies: ["EMA_CROSSOVER"] as StrategyName[],
  initialEquity: 100_000,

  riskPerTradePct: 1.00,           // 100% — cap is always the binding constraint
  maxPositionPct: 7.00,            // 700% — moderate leverage for $5K/month target
  maxOpenPositions: 5,            // 5 concurrent positions for more active trading
  maxEntriesPerDay: 12,           // Proven value
  cooldownMinutes: 2,             // Shorter cooldown for frequent entries
  minConfidence: 68,              // Proven optimal — DO NOT lower
  maxConsecutiveLosses: 2,        // Proven safe — DO NOT increase (3 was net negative)

  partialProfitPct: 0.33,        // Proven R2 value — balanced profit capture
  partialProfitRMultiple: 0.5,   // Take partial at +0.5R to activate trailing stop
  tradeTimeoutMinutes: 30,       // Proven R2 value — 60 min timeouts were losers
  breakevenTriggerPct: 0.3,      // Proven R2 value — protects remaining position after partial
  breakevenOffsetPct: 0.05,

  atrStopMultiplier: 1.2,          // Wider stops survive noise better (0.9 caused 45% stop-outs)
  rewardRiskRatio: 2.0,
  minStopPct: 0.004,
  maxStopPct: 0.04,

  slippageMode: "fixed",
  slippageFixedBps: 5,
  slippagePctOfSpread: 0.5,

  commissionPerShare: 0,

  entryWindowStartMinutes: 9 * 60 + 35,   // 9:35 AM
  entryWindowEndMinutes: 14 * 60,         // 2:00 PM — extended for more EMA crossover opportunities
  forceCloseMinutes: 15 * 60 + 45,         // 3:45 PM

  dailyMaxLoss: -100000,            // Uncapped for aggressive paper trading
  dailyMaxProfit: 200000,           // Uncapped for aggressive paper trading

  warmupBars: 130,
  cachePath: "data/backtest_cache",

  outputPath: "reports/backtest",
  outputFormat: ["json", "csv"],

  verbose: false,
  noCache: false,
};

// ─── Trade Tracking ───

export type TradeExitReason =
  | "stop_loss"
  | "take_profit"
  | "trailing_stop"
  | "breakeven_stop"
  | "timeout"
  | "force_close_eod"
  | "daily_loss_limit"
  | "daily_profit_limit";

export interface TradeExit {
  symbol: string;
  price: number;
  quantity: number;
  time: string;
  reason: TradeExitReason;
  pnl: number;
}

export interface BacktestTrade {
  id: number;
  symbol: string;
  strategyName: string;
  side: "buy" | "sell";

  // Entry
  signalPrice: number;
  entryPrice: number;
  entryTime: string;
  quantity: number;
  confidence: number;

  // Bracket levels
  initialStopPrice: number;
  targetPrice: number;

  // Exit
  exitPrice: number | null;
  exitTime: string | null;
  exitReason: TradeExitReason | null;

  // Multi-exit records
  exits: TradeExit[];

  // P&L
  realizedPnl: number | null;
  realizedPnlPct: number | null;
  holdingTimeMinutes: number | null;
  rMultiple: number | null;

  // Metadata
  slippageBps: number;
  regime: RegimeLabel;
}

// ─── Day State ───

export interface DayState {
  date: string;
  entriesCount: number;
  dailyPnl: number;
  consecutiveLosses: number;
  cooldownMap: Map<string, number>;  // symbol -> cooldown-until timestamp in ms
  peakEquity: number;
}

// ─── Metrics ───

export interface BacktestMetrics {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  profitFactor: number;
  expectancy: number;
  expectancyR: number;

  totalReturn: number;
  totalReturnDollars: number;
  annualizedReturn: number;

  maxDrawdownDollars: number;
  maxDrawdownPercent: number;
  maxDrawdownDurationDays: number;

  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;

  avgHoldingTimeMinutes: number;
  totalTradingDays: number;
  daysWithTrades: number;
  avgTradesPerDay: number;

  equityCurve: { date: string; equity: number; drawdownPct: number }[];
  dailyReturns: { date: string; returnPct: number; pnl: number }[];
}

export interface StrategyMetrics extends BacktestMetrics {
  strategyName: string;
}

export interface BacktestReport {
  config: BacktestConfig;
  runTimestamp: string;
  aggregate: BacktestMetrics;
  perStrategy: StrategyMetrics[];
  trades: BacktestTrade[];
}
