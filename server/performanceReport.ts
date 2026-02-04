import * as alpaca from "./alpaca";
import * as fs from "fs";
import * as path from "path";
import { getEasternTime, toEasternDateString, getPtDateString, getPtNowString } from "./timezone";
import { parseTradeId } from "./tradeId";
import * as skipCounters from "./skipCounters";
import * as executionQuality from "./executionQuality";
import * as activityLedger from "./activityLedger";
import * as tradeLifecycle from "./tradeLifecycle";

interface AlpacaOrder {
  id: string;
  client_order_id: string;
  symbol: string;
  qty: string;
  filled_qty: string;
  side: string;
  type: string;
  status: string;
  filled_avg_price: string | null;
  created_at: string;
  filled_at: string | null;
}

interface JoinedTrade {
  trade_id: string;
  symbol: string;
  strategy: string;
  tier: number;
  side: string;
  qty: number;
  entry_time: string;
  entry_price: number;
  exit_time: string | null;
  exit_price: number | null;
  pnl: number | null;
  exit_reason: string | null;
  order_id: string;
  status: string;
  match_confidence: "HIGH" | "MED" | "LOW";
  has_trade_id: boolean;
}

interface TierStats {
  trades: number;
  wins: number;
  losses: number;
  totalPnl: number;
  winRate: number;
  expectancy: number;
}

interface StrategyStats {
  trades: number;
  wins: number;
  losses: number;
  totalPnl: number;
  winRate: number;
  expectancy: number;
}

interface ConfidenceMetrics {
  trade_count: number;
  win_rate: number;
  expectancy: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number;
}

interface SummaryReport {
  reportDate: string;
  periodStart: string;
  periodEnd: string;
  tradingDays: number;
  reportIsRolling?: boolean; // True if this is a rolling multi-day report
  marketClosed?: boolean;  // True if market was closed on reportDate (holiday/weekend)
  sessionOpen?: string;    // HH:MM Eastern when market opened
  sessionClose?: string;   // HH:MM Eastern when market closed
  baseline_mode: boolean;
  all_trades: ConfidenceMetrics;
  high_confidence_only: ConfidenceMetrics;
  overall: {
    totalScans: number;
    totalSignals: number;
    totalTrades: number;
    wins: number;
    losses: number;
    winRate: number;
    avgWin: number;
    avgLoss: number;
    expectancy: number;
    totalPnl: number;
  };
  byTier: {
    tier1: TierStats;
    tier2: TierStats;
  };
  byStrategy: Record<string, StrategyStats>;
  skipReasons: Array<{ reason: string; count: number; percent: number }>;
  signalCounters?: {
    no_signal_count: number;
    blocked_count: number;
  };
  dataQuality: {
    trades_with_trade_id: number;
    trades_without_trade_id: number;
    high_confidence_matches: number;
    med_confidence_matches: number;
    low_confidence_matches: number;
    paired_trades: number;
    unpaired_entries: number;
    unpaired_exits: number;
    legacy_orders_without_trade_id: number;
  };
  // P6: Execution quality metrics
  executionQuality?: {
    fillRate: number;
    avgTimeToFillMs: number;
    cancelRate: number;
    timeoutRate: number;
    slippageByStrategy: Record<string, { avg: number; median: number; worst: number }>;
    slippageByTimeWindow: Record<string, { avg: number; median: number; worst: number }>;
  };
  // Activity section: Proves bot was running
  activity?: {
    botWasRunning: boolean;
    scanTicks: number;
    symbolsEvaluated: number;
    totalSkips: number;
    noSignalCount: number;
    signalsGenerated: number;
    tradesProposed: number;
    tradesSubmitted: number;
    tradesFilled: number;
    topSkipReasons: Array<{ reason: string; count: number; percent: number }>;
    firstTickET: string | null;
    lastTickET: string | null;
    note?: string;  // Warning if bot wasn't running
  };
  // EOD Flatten section (B3)
  eod?: {
    eodFlattenTriggered: boolean;
    positionsFlattenedCount: number;
    overnightPositionsDetectedCount: number;
    lastFlattenTs: string | null;
  };
}

function getTradingDaysBack(days: number): Date[] {
  const result: Date[] = [];
  const now = new Date();
  let d = new Date(now);
  
  while (result.length < days) {
    const dayOfWeek = d.getDay();
    if (dayOfWeek !== 0 && dayOfWeek !== 6) {
      result.push(new Date(d));
    }
    d.setDate(d.getDate() - 1);
  }
  
  return result.reverse();
}

async function fetchOrdersLast5Days(): Promise<AlpacaOrder[]> {
  const tradingDays = getTradingDaysBack(5);
  if (tradingDays.length === 0) return [];
  
  const startDate = tradingDays[0];
  startDate.setHours(0, 0, 0, 0);
  
  const endDate = new Date();
  
  try {
    const orders = await alpaca.getOrdersWithDateRange(startDate, endDate, "all", 500);
    console.log(`[Report] Fetched ${orders.length} orders from last 5 trading days`);
    return orders;
  } catch (error) {
    console.error("[Report] Error fetching orders:", error);
    return [];
  }
}

/**
 * Fetch orders for a single market day only
 * Uses Alpaca calendar to get exact session open/close times
 */
async function fetchOrdersForDay(dateStr: string): Promise<{
  orders: AlpacaOrder[];
  calendarDay: alpaca.AlpacaCalendarDay | null;
}> {
  // Check if this was a trading day
  const calendarDay = await alpaca.getCalendarDay(dateStr);
  
  if (!calendarDay) {
    // Market was closed (holiday or weekend)
    return { orders: [], calendarDay: null };
  }
  
  // Build session timestamps in Eastern time
  // calendarDay.open = "09:30", calendarDay.close = "16:00"
  const sessionOpenET = new Date(`${dateStr}T${calendarDay.open}:00-05:00`);
  const sessionCloseET = new Date(`${dateStr}T${calendarDay.close}:00-05:00`);
  
  // Add buffer: start 1 minute before open, end 30 minutes after close (for fills)
  const queryStart = new Date(sessionOpenET.getTime() - 60 * 1000);
  const queryEnd = new Date(sessionCloseET.getTime() + 30 * 60 * 1000);
  
  try {
    const orders = await alpaca.getOrdersWithDateRange(queryStart, queryEnd, "all", 500);
    console.log(`[Report] Fetched ${orders.length} orders for ${dateStr} (session ${calendarDay.open}-${calendarDay.close} ET)`);
    return { orders, calendarDay };
  } catch (error) {
    console.error(`[Report] Error fetching orders for ${dateStr}:`, error);
    return { orders: [], calendarDay };
  }
}

function matchEntryToExit(
  entries: AlpacaOrder[],
  exits: AlpacaOrder[]
): Map<string, { entry: AlpacaOrder; exit: AlpacaOrder | null }> {
  const matches = new Map<string, { entry: AlpacaOrder; exit: AlpacaOrder | null }>();
  
  for (const entry of entries) {
    const tradeId = entry.client_order_id;
    
    const exitOrder = exits.find(exit => {
      if (exit.symbol !== entry.symbol) return false;
      
      if (exit.client_order_id.startsWith(entry.symbol)) {
        const entryTime = new Date(entry.filled_at || entry.created_at).getTime();
        const exitTime = new Date(exit.filled_at || exit.created_at).getTime();
        return exitTime > entryTime && exitTime - entryTime < 24 * 60 * 60 * 1000;
      }
      
      return false;
    });
    
    matches.set(tradeId, { entry, exit: exitOrder || null });
  }
  
  return matches;
}

function buildJoinedTrades(orders: AlpacaOrder[]): JoinedTrade[] {
  const buyOrders = orders.filter(o => o.side === "buy" && o.status === "filled");
  const sellOrders = orders.filter(o => o.side === "sell" && o.status === "filled");
  
  const joined: JoinedTrade[] = [];
  const usedSellOrders = new Set<string>(); // Track which sell orders are already matched
  
  for (const buy of buyOrders) {
    const parsed = parseTradeId(buy.client_order_id);
    const hasTradeId = parsed !== null && parsed.strategy !== "unknown";
    
    let matchingSell: AlpacaOrder | null = null;
    let matchConfidence: "HIGH" | "MED" | "LOW" = "LOW";
    
    // PRIORITY 1: Strict trade_id match (HIGH confidence)
    // Match entry trade_id to exit client_order_id with _EXIT or _PARTIAL suffix
    if (hasTradeId) {
      const entryTradeId = buy.client_order_id;
      
      // Look for exit order where client_order_id starts with entry trade_id
      // e.g., entry="SPY_breakout_buy_20251222_0936_T2" matches exit="SPY_breakout_buy_20251222_0936_T2_EXIT"
      const prefixMatch = sellOrders.find(s => 
        s.symbol === buy.symbol && 
        !usedSellOrders.has(s.id) &&
        (s.client_order_id === entryTradeId || 
         s.client_order_id === `${entryTradeId}_EXIT` ||
         s.client_order_id === `${entryTradeId}_PARTIAL` ||
         s.client_order_id.startsWith(`${entryTradeId}_`))
      );
      if (prefixMatch) {
        matchingSell = prefixMatch;
        matchConfidence = "HIGH";
      }
    }
    
    // PRIORITY 2: Time-based fallback (60 min window for trades with trade_id = MED, 60 min for legacy = LOW)
    if (!matchingSell) {
      const buyTime = new Date(buy.filled_at || buy.created_at).getTime();
      const timeWindow = 60 * 60 * 1000; // 60 minutes (reduced from 8 hours)
      
      const timeMatch = sellOrders.find(s => {
        if (s.symbol !== buy.symbol) return false;
        if (usedSellOrders.has(s.id)) return false;
        const sellTime = new Date(s.filled_at || s.created_at).getTime();
        return sellTime > buyTime && sellTime - buyTime < timeWindow;
      });
      
      if (timeMatch) {
        matchingSell = timeMatch;
        matchConfidence = hasTradeId ? "MED" : "LOW";
      }
    }
    
    // Mark sell order as used
    if (matchingSell) {
      usedSellOrders.add(matchingSell.id);
    }
    
    const entryPrice = parseFloat(buy.filled_avg_price || "0");
    const exitPrice = matchingSell ? parseFloat(matchingSell.filled_avg_price || "0") : null;
    const qty = parseInt(buy.filled_qty, 10);
    
    let pnl: number | null = null;
    if (exitPrice !== null && entryPrice > 0) {
      pnl = (exitPrice - entryPrice) * qty;
    }
    
    joined.push({
      trade_id: buy.client_order_id,
      symbol: buy.symbol,
      strategy: parsed?.strategy || "unknown",
      tier: parsed?.tier || 2,
      side: "buy",
      qty,
      entry_time: buy.filled_at || buy.created_at,
      entry_price: entryPrice,
      exit_time: matchingSell?.filled_at || null,
      exit_price: exitPrice,
      pnl,
      exit_reason: matchingSell ? "sell_order" : null,
      order_id: buy.id,
      status: matchingSell ? "closed" : "open",
      match_confidence: matchConfidence,
      has_trade_id: hasTradeId,
    });
  }
  
  return joined;
}

function calculateStats(trades: JoinedTrade[]): {
  wins: number;
  losses: number;
  totalPnl: number;
  avgWin: number;
  avgLoss: number;
} {
  const closedTrades = trades.filter(t => t.pnl !== null);
  const wins = closedTrades.filter(t => (t.pnl || 0) > 0);
  const losses = closedTrades.filter(t => (t.pnl || 0) <= 0);
  
  const totalPnl = closedTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
  const avgWin = wins.length > 0 ? wins.reduce((s, t) => s + (t.pnl || 0), 0) / wins.length : 0;
  const avgLoss = losses.length > 0 ? Math.abs(losses.reduce((s, t) => s + (t.pnl || 0), 0) / losses.length) : 0;
  
  return {
    wins: wins.length,
    losses: losses.length,
    totalPnl,
    avgWin,
    avgLoss,
  };
}

function calculateConfidenceMetrics(trades: JoinedTrade[]): ConfidenceMetrics {
  const closedTrades = trades.filter(t => t.pnl !== null);
  const wins = closedTrades.filter(t => (t.pnl || 0) > 0);
  const losses = closedTrades.filter(t => (t.pnl || 0) <= 0);
  
  const totalWinPnl = wins.reduce((s, t) => s + (t.pnl || 0), 0);
  const totalLossPnl = Math.abs(losses.reduce((s, t) => s + (t.pnl || 0), 0));
  const totalPnl = closedTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
  
  const avgWin = wins.length > 0 ? totalWinPnl / wins.length : 0;
  const avgLoss = losses.length > 0 ? totalLossPnl / losses.length : 0;
  const winRate = closedTrades.length > 0 ? (wins.length / closedTrades.length) * 100 : 0;
  const expectancy = closedTrades.length > 0 ? totalPnl / closedTrades.length : 0;
  const profitFactor = totalLossPnl > 0 ? totalWinPnl / totalLossPnl : (totalWinPnl > 0 ? Infinity : 0);
  
  return {
    trade_count: trades.length,
    win_rate: winRate,
    expectancy,
    avg_win: avgWin,
    avg_loss: avgLoss,
    profit_factor: profitFactor === Infinity ? 999 : profitFactor,
  };
}

interface BuildSummaryOptions {
  reportDate: string;
  periodStart?: string;
  periodEnd?: string;
  tradingDays?: number;
  calendarDay?: alpaca.AlpacaCalendarDay | null;
  marketClosed?: boolean;
  reportIsRolling?: boolean;
}

function buildSummary(trades: JoinedTrade[], options?: BuildSummaryOptions): SummaryReport {
  const et = getEasternTime();
  const defaultTradingDays = getTradingDaysBack(5);
  
  // Use provided options or defaults
  const reportDate = options?.reportDate || et.dateString;
  const periodStart = options?.periodStart || (defaultTradingDays.length > 0 ? toEasternDateString(defaultTradingDays[0]) : et.dateString);
  const periodEnd = options?.periodEnd || et.dateString;
  const tradingDaysCount = options?.tradingDays ?? defaultTradingDays.length;
  const marketClosed = options?.marketClosed ?? false;
  const calendarDay = options?.calendarDay;
  
  const overall = calculateStats(trades);
  const closedCount = trades.filter(t => t.pnl !== null).length;
  const winRate = closedCount > 0 ? (overall.wins / closedCount) * 100 : 0;
  const expectancy = closedCount > 0 ? overall.totalPnl / closedCount : 0;
  
  const tier1Trades = trades.filter(t => t.tier === 1);
  const tier2Trades = trades.filter(t => t.tier === 2);
  
  const tier1Stats = calculateStats(tier1Trades);
  const tier1Closed = tier1Trades.filter(t => t.pnl !== null).length;
  
  const tier2Stats = calculateStats(tier2Trades);
  const tier2Closed = tier2Trades.filter(t => t.pnl !== null).length;
  
  const strategyGroups: Record<string, JoinedTrade[]> = {};
  for (const t of trades) {
    if (!strategyGroups[t.strategy]) strategyGroups[t.strategy] = [];
    strategyGroups[t.strategy].push(t);
  }
  
  const byStrategy: Record<string, StrategyStats> = {};
  for (const [strategy, stratTrades] of Object.entries(strategyGroups)) {
    const stats = calculateStats(stratTrades);
    const closed = stratTrades.filter(t => t.pnl !== null).length;
    byStrategy[strategy] = {
      trades: stratTrades.length,
      wins: stats.wins,
      losses: stats.losses,
      totalPnl: stats.totalPnl,
      winRate: closed > 0 ? (stats.wins / closed) * 100 : 0,
      expectancy: closed > 0 ? stats.totalPnl / closed : 0,
    };
  }
  
  const skipCounts = skipCounters.getSkipCounts();
  const totalSkips = skipCounters.getTotalSkips();
  const skipReasons: Array<{ reason: string; count: number; percent: number }> = [];
  
  skipCounts.forEach((count, reason) => {
    skipReasons.push({
      reason,
      count,
      percent: totalSkips > 0 ? (count / totalSkips) * 100 : 0,
    });
  });
  
  skipReasons.sort((a, b) => b.count - a.count);
  const top10Skips = skipReasons.slice(0, 10);
  
  // Data quality counters
  const tradesWithTradeId = trades.filter(t => t.has_trade_id).length;
  const tradesWithoutTradeId = trades.filter(t => !t.has_trade_id).length;
  const highConfidenceTrades = trades.filter(t => t.match_confidence === "HIGH");
  const medConfidence = trades.filter(t => t.match_confidence === "MED").length;
  const lowConfidence = trades.filter(t => t.match_confidence === "LOW").length;
  
  // A4 trade_id pairing counters - now sourced from tradeLifecycle for accuracy
  // Get authoritative pairing data from lifecycle (uses parent_order_id linking for brackets)
  const lifecyclePairing = tradeLifecycle.getPairingMetrics(reportDate);
  
  // Fallback to order-based matching if lifecycle has no data
  const pairedTrades = lifecyclePairing.pairedTrades > 0 
    ? lifecyclePairing.pairedTrades 
    : trades.filter(t => t.match_confidence === "HIGH" && t.pnl !== null).length;
  
  const unpairedEntries = lifecyclePairing.unpairedEntries > 0
    ? lifecyclePairing.unpairedEntries
    : trades.filter(t => t.has_trade_id && t.status === "open").length;
  
  // unpaired_exits: now properly tracked via lifecycle (0 if all exits linked)
  const unpairedExits = lifecyclePairing.unpairedExits;
  
  // legacy_orders: orders without ato_ format trade_id
  const legacyOrders = lifecyclePairing.legacyOrders > 0 
    ? lifecyclePairing.legacyOrders 
    : tradesWithoutTradeId;
  
  // Confidence-split metrics
  const allTradesMetrics = calculateConfidenceMetrics(trades);
  const highConfidenceMetrics = calculateConfidenceMetrics(highConfidenceTrades);
  
  // Get activity ledger data FIRST for totalScans and signal counters
  // For rolling reports, get activity for the entire period
  const activitySummary = options?.reportIsRolling && options?.periodStart && options?.periodEnd
    ? activityLedger.getActivitySummary({ start: options.periodStart, end: options.periodEnd })
    : activityLedger.getActivitySummary(reportDate);
  
  // Signal counters: use activity ledger as source of truth when bot was running
  // This fixes the bug where Signal Counters showed 0 even when Activity section had data
  const noSignalCount = activitySummary.botWasRunning 
    ? activitySummary.noSignalCount 
    : skipCounters.getNoSignalCount();
  const blockedCount = activitySummary.botWasRunning 
    ? activitySummary.totalSkips 
    : skipCounters.getBlockedCount();
  
  const result: SummaryReport = {
    reportDate,
    periodStart,
    periodEnd,
    tradingDays: tradingDaysCount,
    reportIsRolling: options?.reportIsRolling ?? false,
    baseline_mode: true,
    all_trades: allTradesMetrics,
    high_confidence_only: highConfidenceMetrics,
    overall: {
      totalScans: activitySummary.scanTicks,  // Wire from activity ledger
      totalSignals: activitySummary.signalsGenerated,  // Signals with valid indicators
      totalTrades: trades.length,
      wins: overall.wins,
      losses: overall.losses,
      winRate,
      avgWin: overall.avgWin,
      avgLoss: overall.avgLoss,
      expectancy,
      totalPnl: overall.totalPnl,
    },
    byTier: {
      tier1: {
        trades: tier1Trades.length,
        wins: tier1Stats.wins,
        losses: tier1Stats.losses,
        totalPnl: tier1Stats.totalPnl,
        winRate: tier1Closed > 0 ? (tier1Stats.wins / tier1Closed) * 100 : 0,
        expectancy: tier1Closed > 0 ? tier1Stats.totalPnl / tier1Closed : 0,
      },
      tier2: {
        trades: tier2Trades.length,
        wins: tier2Stats.wins,
        losses: tier2Stats.losses,
        totalPnl: tier2Stats.totalPnl,
        winRate: tier2Closed > 0 ? (tier2Stats.wins / tier2Closed) * 100 : 0,
        expectancy: tier2Closed > 0 ? tier2Stats.totalPnl / tier2Closed : 0,
      },
    },
    byStrategy,
    skipReasons: top10Skips,
    signalCounters: {
      no_signal_count: noSignalCount,
      blocked_count: blockedCount,
    },
    dataQuality: {
      trades_with_trade_id: tradesWithTradeId,
      trades_without_trade_id: tradesWithoutTradeId,
      high_confidence_matches: highConfidenceTrades.length,
      med_confidence_matches: medConfidence,
      low_confidence_matches: lowConfidence,
      paired_trades: pairedTrades,
      unpaired_entries: unpairedEntries,
      unpaired_exits: unpairedExits,
      legacy_orders_without_trade_id: legacyOrders,
    },
    // P6: Execution quality metrics
    executionQuality: executionQuality.getExecutionReport(),
  };
  
  // Add market closed flag and session times if applicable
  if (marketClosed) {
    result.marketClosed = true;
  }
  if (calendarDay) {
    result.sessionOpen = calendarDay.open;
    result.sessionClose = calendarDay.close;
  }
  
  // ACTIVITY SECTION: Proves bot was running (activitySummary already fetched above)
  result.activity = {
    botWasRunning: activitySummary.botWasRunning,
    scanTicks: activitySummary.scanTicks,
    symbolsEvaluated: activitySummary.symbolsEvaluated,
    totalSkips: activitySummary.totalSkips,
    noSignalCount: activitySummary.noSignalCount,
    signalsGenerated: activitySummary.signalsGenerated,
    tradesProposed: activitySummary.tradesProposed,
    tradesSubmitted: activitySummary.tradesSubmitted,
    tradesFilled: activitySummary.tradesFilled,
    topSkipReasons: activitySummary.topSkipReasons,
    firstTickET: activitySummary.firstTickET,
    lastTickET: activitySummary.lastTickET,
  };
  
  // If bot wasn't running, add warning note
  if (!activitySummary.botWasRunning) {
    result.activity.note = "No scan activity recorded (bot likely not running). Trade counts may reflect Alpaca history only.";
    // Mark signal counters as unknown when bot wasn't running
    result.signalCounters = undefined;
  }
  
  // A4: Consistency Guard - warn if bot was running but scans are 0
  if (activitySummary.botWasRunning && activitySummary.scanTicks === 0) {
    console.log("[Report] WARNING: botWasRunning=true but scanTicks=0 - possible data mapping issue");
    if (!result.activity.note) {
      result.activity.note = "Warning: Bot marked as running but no scan ticks recorded.";
    }
  }
  
  // A4: Warn if trades exist but no signals/scans
  if (trades.length > 0 && activitySummary.scanTicks === 0 && !activitySummary.botWasRunning) {
    result.activity.note = (result.activity.note || "") + " Trades exist but no scans recorded (Alpaca history only).";
  }
  
  // B3: EOD Flatten section - dynamically import to avoid circular deps
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const eodManager = require("./eodManager");
    const eodStatus = eodManager.getEODStatus();
    result.eod = {
      eodFlattenTriggered: eodStatus.flattenTriggered,
      positionsFlattenedCount: eodStatus.positionsFlattenedCount,
      overnightPositionsDetectedCount: eodStatus.overnightPositionsDetected,
      lastFlattenTs: eodStatus.lastFlattenTs,
    };
  } catch {
    // EOD manager not loaded yet
    result.eod = {
      eodFlattenTriggered: false,
      positionsFlattenedCount: 0,
      overnightPositionsDetectedCount: 0,
      lastFlattenTs: null,
    };
  }
  
  return result;
}

function tradesArrayToCsv(trades: JoinedTrade[]): string {
  const headers = [
    "trade_id",
    "symbol",
    "strategy",
    "tier",
    "side",
    "qty",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "pnl",
    "exit_reason",
    "match_confidence",
    "has_trade_id",
  ];
  
  const rows = trades.map(t => [
    t.trade_id,
    t.symbol,
    t.strategy,
    t.tier.toString(),
    t.side,
    t.qty.toString(),
    t.entry_time,
    t.entry_price.toFixed(2),
    t.exit_time || "",
    t.exit_price?.toFixed(2) || "",
    t.pnl?.toFixed(2) || "",
    t.exit_reason || "",
    t.match_confidence,
    t.has_trade_id ? "true" : "false",
  ]);
  
  return [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
}

function writeDailyReport(summary: SummaryReport, dateOverride?: string): void {
  const dailyDir = path.join(process.cwd(), "daily_reports");
  if (!fs.existsSync(dailyDir)) {
    fs.mkdirSync(dailyDir, { recursive: true });
  }
  
  const dateStr = dateOverride || summary.reportDate;
  console.log(`[REPORT] now_pt=${getPtNowString()} reportDate=${dateStr}`);
  const jsonContent = JSON.stringify(summary, null, 2);
  
  const datePath = path.join(dailyDir, `${dateStr}.json`);
  const latestPath = path.join(dailyDir, "latest.json");
  const summaryTxtPath = path.join(dailyDir, `${dateStr}_summary.txt`);
  
  fs.writeFileSync(datePath, jsonContent);
  fs.writeFileSync(latestPath, jsonContent);
  
  const dq = summary.dataQuality;
  const o = summary.overall;
  const sc = summary.signalCounters;
  const top5Skips = summary.skipReasons.slice(0, 5).map(s => `  ${s.reason}: ${s.count} (${s.percent.toFixed(1)}%)`).join("\n");
  
  // Market status line
  const marketStatusLine = summary.marketClosed 
    ? `Market Status: CLOSED (Holiday/Weekend)\n` 
    : (summary.sessionOpen && summary.sessionClose 
        ? `Session: ${summary.sessionOpen} - ${summary.sessionClose} ET\n` 
        : "");
  
  const isRolling = summary.reportIsRolling === true;
  const headerLine = isRolling 
    ? `AtoBot Rolling Report (Last 5 Trading Days) - Generated ${dateStr}`
    : `AtoBot Daily Report - ${dateStr}`;
  const periodLine = isRolling && summary.periodStart && summary.periodEnd
    ? `Period: ${summary.periodStart} → ${summary.periodEnd} | tradingDays=${summary.tradingDays ?? 0}\n`
    : "";
  
  // Activity section for TXT
  const act = summary.activity;
  const activityTopSkips = act?.topSkipReasons?.slice(0, 3).map(s => `    ${s.reason}: ${s.count}`).join("\n") || "    (none)";
  const activityNote = act?.note ? `  Note: ${act.note}\n` : "";
  const activitySection = act ? `
Activity (Bot Scans):
  Bot Was Running: ${act.botWasRunning ? "YES" : "NO"}
  Scan Ticks: ${act.scanTicks}
  Symbols Evaluated: ${act.symbolsEvaluated}
  First Tick: ${act.firstTickET || "N/A"}
  Last Tick: ${act.lastTickET || "N/A"}
  Top Skip Reasons:
${activityTopSkips}
${activityNote}` : "";

  const txtContent = `${headerLine}
================================
${periodLine}${marketStatusLine}Trades: ${o.totalTrades} | Wins: ${o.wins} | Losses: ${o.losses}
Win Rate: ${o.winRate.toFixed(1)}%
Total P&L: $${o.totalPnl.toFixed(2)}
Expectancy: $${o.expectancy.toFixed(2)}
Total Scans: ${o.totalScans} | Total Signals: ${o.totalSignals}

Signal Counters:
  No Signal: ${sc?.no_signal_count ?? 0}
  Blocked: ${sc?.blocked_count ?? 0}

Data Quality:
  Paired Trades: ${dq.paired_trades}
  Unpaired Entries: ${dq.unpaired_entries}
  Unpaired Exits: ${dq.unpaired_exits}
  Legacy (no trade_id): ${dq.legacy_orders_without_trade_id}
  HIGH confidence: ${dq.high_confidence_matches}
${activitySection}
Top Skip Reasons:
${top5Skips || "  (none)"}
`;
  
  fs.writeFileSync(summaryTxtPath, txtContent);
  
  console.log(`[Report] Daily reports written to daily_reports/:`);
  console.log(`  - ${dateStr}.json`);
  console.log(`  - latest.json`);
  console.log(`  - ${dateStr}_summary.txt`);
}

function skipReasonsToCsv(): string {
  const skipCounts = skipCounters.getSkipCounts();
  const totalSkips = skipCounters.getTotalSkips();
  
  const headers = ["reason", "count", "percent"];
  const entries: Array<[string, number]> = [];
  skipCounts.forEach((count, reason) => {
    entries.push([reason, count]);
  });
  
  entries.sort((a, b) => b[1] - a[1]);
  
  const rows = entries.map(([reason, count]) => [
    reason,
    count.toString(),
    totalSkips > 0 ? ((count / totalSkips) * 100).toFixed(2) : "0",
  ]);
  
  return [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
}

/**
 * Generate a rolling 5-day report ending on a specific date
 * Uses Alpaca calendar to determine trading days and fetches last 5 trading days of orders
 */
export async function generateDailyReportForDate(dateStr: string): Promise<SummaryReport> {
  console.log(`[Report] Generating rolling 5-day report ending ${dateStr}...`);
  
  // Check if the specified date was a trading day
  const calendarDay = await alpaca.getCalendarDay(dateStr);
  
  if (!calendarDay) {
    // Market was closed on this day (holiday or weekend)
    console.log(`[Report] Market was CLOSED on ${dateStr} (holiday/weekend)`);
    const summary = buildSummary([], {
      reportDate: dateStr,
      periodStart: dateStr,
      periodEnd: dateStr,
      tradingDays: 0,
      marketClosed: true,
      calendarDay: null,
      reportIsRolling: true,
    });
    writeDailyReport(summary, dateStr);
    return summary;
  }
  
  // Fetch rolling 5-day orders ending on this date
  const orders = await fetchOrdersLast5Days();
  const joinedTrades = buildJoinedTrades(orders);
  const summary = buildSummary(joinedTrades, {
    reportDate: dateStr,
    reportIsRolling: true,
  });
  
  writeDailyReport(summary, dateStr);
  console.log(`[Report] ${dateStr}: ${summary.overall.totalTrades} trades, Period: ${summary.periodStart} - ${summary.periodEnd}`);
  
  return summary;
}

/**
 * Generate rolling 5-day report
 * This is the main entry point called by the report scheduler
 */
export async function generateReport(): Promise<{
  summary: SummaryReport;
  tradesPath: string;
  summaryPath: string;
  skipReasonsPath: string;
}> {
  const et = getEasternTime();
  const todayStr = et.dateString;
  
  console.log(`[Report] Generating rolling 5-day report for ${todayStr}...`);
  
  const reportsDir = path.join(process.cwd(), "reports");
  if (!fs.existsSync(reportsDir)) {
    fs.mkdirSync(reportsDir, { recursive: true });
  }
  
  const orders = await fetchOrdersLast5Days();
  const joinedTrades = buildJoinedTrades(orders);
  const summary = buildSummary(joinedTrades, {
    reportDate: todayStr,
    reportIsRolling: true,
  });
  
  const tradesCsv = tradesArrayToCsv(joinedTrades);
  const skipsCsv = skipReasonsToCsv();
  
  const tradesPath = path.join(reportsDir, "trades_joined_last_5d.csv");
  const summaryPath = path.join(reportsDir, "summary_last_5d.json");
  const skipReasonsPath = path.join(reportsDir, "skip_reasons_last_5d.csv");
  
  fs.writeFileSync(tradesPath, tradesCsv);
  fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
  fs.writeFileSync(skipReasonsPath, skipsCsv);
  
  // Write rolling report to daily_reports/ as well
  writeDailyReport(summary, todayStr);
  
  console.log(`[Report] Generated reports:`);
  console.log(`  - ${tradesPath}`);
  console.log(`  - ${summaryPath}`);
  console.log(`  - ${skipReasonsPath}`);
  console.log(`[Report] Rolling 5d: ${summary.overall.totalTrades} trades, Period: ${summary.periodStart} - ${summary.periodEnd}`);
  
  return {
    summary,
    tradesPath,
    summaryPath,
    skipReasonsPath,
  };
}

/**
 * Generate reports for multiple dates (useful for backfilling)
 */
export async function generateReportsForDates(dates: string[]): Promise<void> {
  console.log(`[Report] Generating reports for ${dates.length} dates...`);
  for (const dateStr of dates) {
    await generateDailyReportForDate(dateStr);
  }
  console.log(`[Report] Completed generating ${dates.length} reports`);
}

