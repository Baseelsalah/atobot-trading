/**
 * Export Last 4 Trading Days Performance Packet
 * Reads stored artifacts + Alpaca history and packages them
 */

import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";
import * as alpaca from "../server/alpaca.js";

const REPORTS_DIR = path.join(process.cwd(), "reports");
const DIAGNOSTICS_DIR = path.join(REPORTS_DIR, "diagnostics");
const OUTPUT_DIR = path.join(DIAGNOSTICS_DIR, "last4tradingdays");

interface TradingDay {
  date: string;
  open: string;
  close: string;
}

interface ActivityTick {
  tickId: string;
  tsET: string;
  ptDate: string;
  etDate: string;
  symbolsEvaluated: number;
  validQuotes: number;
  validBars: number;
  noSignalCount: number;
  skipCount: number;
  skipReasonCounts: Record<string, number>;
  tradesAttempted: number;
  tradesFilled: number;
}

interface DailyReport {
  reportDate: string;
  activity?: {
    botWasRunning: boolean;
    scanTicks: number;
    symbolsEvaluated: number;
    totalSkips: number;
    noSignalCount: number;
    tradesAttempted: number;
    tradesFilled: number;
    topSkipReasons: Array<{ reason: string; count: number }>;
    firstTickET: string | null;
    lastTickET: string | null;
  };
  eod?: {
    eodFlattenTriggered: boolean;
    positionsFlattenedCount: number;
    overnightPositionsDetectedCount: number;
    lastFlattenTs: string | null;
  };
  executionQuality?: {
    fillRate: number;
    avgTimeToFillMs: number;
    cancelRate: number;
    timeoutRate: number;
  };
}

async function getLast4TradingDays(): Promise<TradingDay[]> {
  const now = new Date();
  const startDate = new Date(now);
  startDate.setDate(startDate.getDate() - 14); // Look back 14 days to find 4 trading days
  
  const calendar = await alpaca.getCalendar(
    startDate.toISOString().split("T")[0],
    now.toISOString().split("T")[0]
  );
  
  // Get last 4 trading days (excluding today if market still open)
  const tradingDays = calendar
    .filter((day: any) => new Date(day.date) < now)
    .slice(-4)
    .map((day: any) => ({
      date: day.date,
      open: day.open,
      close: day.close,
    }));
  
  return tradingDays;
}

function readActivityLedger(date: string): ActivityTick[] {
  const filePath = path.join(REPORTS_DIR, "activity", `activity_${date}.json`);
  if (fs.existsSync(filePath)) {
    try {
      return JSON.parse(fs.readFileSync(filePath, "utf-8"));
    } catch {
      return [];
    }
  }
  return [];
}

function readDailyReport(date: string): DailyReport | null {
  const jsonPath = path.join(REPORTS_DIR, "daily_reports", `${date}.json`);
  if (fs.existsSync(jsonPath)) {
    try {
      return JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
    } catch {
      return null;
    }
  }
  return null;
}

function generateActivityAggregate(tradingDays: TradingDay[]): any {
  const aggregate: any = {
    tradingDays: tradingDays.map(d => d.date),
    perDay: {} as Record<string, any>,
    topSkipReasons: {} as Record<string, number>,
    totals: {
      scanTicks: 0,
      symbolsEvaluated: 0,
      noSignalCount: 0,
      skipCount: 0,
      tradesAttempted: 0,
      tradesFilled: 0,
    },
  };

  for (const day of tradingDays) {
    const ticks = readActivityLedger(day.date);
    const dayData = {
      date: day.date,
      scanTicks: ticks.length,
      symbolsEvaluated: ticks.reduce((sum, t) => sum + t.symbolsEvaluated, 0),
      noSignalCount: ticks.reduce((sum, t) => sum + t.noSignalCount, 0),
      skipCount: ticks.reduce((sum, t) => sum + t.skipCount, 0),
      tradesAttempted: ticks.reduce((sum, t) => sum + t.tradesAttempted, 0),
      tradesFilled: ticks.reduce((sum, t) => sum + t.tradesFilled, 0),
      firstTickET: ticks.length > 0 ? ticks[0].tsET : null,
      lastTickET: ticks.length > 0 ? ticks[ticks.length - 1].tsET : null,
      skipReasons: {} as Record<string, number>,
    };

    // Aggregate skip reasons
    for (const tick of ticks) {
      for (const [reason, count] of Object.entries(tick.skipReasonCounts)) {
        dayData.skipReasons[reason] = (dayData.skipReasons[reason] || 0) + count;
        aggregate.topSkipReasons[reason] = (aggregate.topSkipReasons[reason] || 0) + count;
      }
    }

    aggregate.perDay[day.date] = dayData;
    aggregate.totals.scanTicks += dayData.scanTicks;
    aggregate.totals.symbolsEvaluated += dayData.symbolsEvaluated;
    aggregate.totals.noSignalCount += dayData.noSignalCount;
    aggregate.totals.skipCount += dayData.skipCount;
    aggregate.totals.tradesAttempted += dayData.tradesAttempted;
    aggregate.totals.tradesFilled += dayData.tradesFilled;
  }

  // Sort skip reasons by count
  aggregate.topSkipReasons = Object.entries(aggregate.topSkipReasons)
    .sort((a, b) => (b[1] as number) - (a[1] as number))
    .slice(0, 15)
    .reduce((obj, [k, v]) => ({ ...obj, [k]: v }), {});

  return aggregate;
}

async function getAlpacaOrders(startDate: string, endDate: string): Promise<any[]> {
  const start = new Date(startDate);
  start.setHours(0, 0, 0, 0);
  const end = new Date(endDate);
  end.setDate(end.getDate() + 1);
  
  try {
    const orders = await alpaca.getOrdersWithDateRange(start, end, "all", 500);
    return orders;
  } catch (error) {
    console.log(`Error fetching orders: ${error}`);
    return [];
  }
}

function generateTradesJoined(orders: any[]): any[] {
  const trades: any[] = [];
  const ordersByClientId = new Map<string, any>();
  const ordersByParentId = new Map<string, any[]>();

  // Index orders
  for (const order of orders) {
    if (order.client_order_id) {
      ordersByClientId.set(order.client_order_id, order);
    }
    if (order.parent_order_id) {
      const siblings = ordersByParentId.get(order.parent_order_id) || [];
      siblings.push(order);
      ordersByParentId.set(order.parent_order_id, siblings);
    }
  }

  // Find entry orders (bracket parents)
  const entryOrders = orders.filter(o => 
    o.side === "buy" && 
    o.order_class === "bracket" &&
    o.client_order_id?.startsWith("ato_")
  );

  for (const entry of entryOrders) {
    const trade: any = {
      trade_id: entry.client_order_id?.replace("ato_", "") || "",
      symbol: entry.symbol,
      strategyName: "",
      regimeLabel: "",
      timeWindow: "",
      tier: "",
      entry_time: entry.filled_at || entry.created_at,
      entry_signal_price: "",
      entry_fill_price: entry.filled_avg_price || "",
      entry_slippage_bps: "",
      qty: entry.filled_qty || entry.qty,
      exit_time: "",
      exit_fill_price: "",
      exit_slippage_bps: "",
      exitReason: "",
      status: entry.status === "filled" ? "OPEN" : entry.status,
      realized_pnl: "",
      usedAtrFallback: "",
      atr: "",
      atrPct: "",
      stopDistance: "",
      rr: "",
      parent_order_id: entry.id,
      stop_order_id: "",
      tp_order_id: "",
    };

    // Find child orders (stop loss, take profit)
    const legs = entry.legs || [];
    for (const leg of legs) {
      if (leg.type === "stop" || leg.stop_price) {
        trade.stop_order_id = leg.id;
        if (leg.status === "filled") {
          trade.exit_time = leg.filled_at;
          trade.exit_fill_price = leg.filled_avg_price;
          trade.exitReason = "SL";
          trade.status = "CLOSED";
        }
      } else if (leg.type === "limit" || leg.limit_price) {
        trade.tp_order_id = leg.id;
        if (leg.status === "filled") {
          trade.exit_time = leg.filled_at;
          trade.exit_fill_price = leg.filled_avg_price;
          trade.exitReason = "TP";
          trade.status = "CLOSED";
        }
      }
    }

    // Calculate P&L if closed
    if (trade.status === "CLOSED" && trade.entry_fill_price && trade.exit_fill_price) {
      const entryPrice = parseFloat(trade.entry_fill_price);
      const exitPrice = parseFloat(trade.exit_fill_price);
      const qty = parseInt(trade.qty);
      trade.realized_pnl = ((exitPrice - entryPrice) * qty).toFixed(2);
    }

    trades.push(trade);
  }

  return trades;
}

function generateExecutionSummary(orders: any[], tradingDays: TradingDay[]): any {
  const summary: any = {
    tradingDays: tradingDays.map(d => d.date),
    fillRate: 0,
    cancelRate: 0,
    timeoutRate: 0,
    avgTimeToFillMs: 0,
    slippage: {
      avg: 0,
      median: 0,
      worst: 0,
    },
    slippageByStrategy: {},
    slippageByTimeWindow: {},
    skipReasonCounts: {
      QUOTE_STALE_OR_MISSING: 0,
      UNFILLED_TIMEOUT: 0,
      SPREAD_NEAR_MAX: 0,
      ORDER_REJECTED: 0,
    },
  };

  const entryOrders = orders.filter(o => o.side === "buy");
  if (entryOrders.length === 0) {
    return summary;
  }

  const filled = entryOrders.filter(o => o.status === "filled");
  const cancelled = entryOrders.filter(o => o.status === "canceled" || o.status === "cancelled");
  const expired = entryOrders.filter(o => o.status === "expired");

  summary.fillRate = filled.length / entryOrders.length;
  summary.cancelRate = cancelled.length / entryOrders.length;
  summary.timeoutRate = expired.length / entryOrders.length;

  // Calculate time to fill
  const fillTimes: number[] = [];
  for (const order of filled) {
    if (order.created_at && order.filled_at) {
      const created = new Date(order.created_at).getTime();
      const filled = new Date(order.filled_at).getTime();
      fillTimes.push(filled - created);
    }
  }
  if (fillTimes.length > 0) {
    summary.avgTimeToFillMs = fillTimes.reduce((a, b) => a + b, 0) / fillTimes.length;
  }

  return summary;
}

function generateEodOvernightSummary(tradingDays: TradingDay[]): any {
  const summary: any = {
    tradingDays: tradingDays.map(d => d.date),
    perDay: {} as Record<string, any>,
    totals: {
      eodFlattenTriggered: 0,
      positionsFlattenedCount: 0,
      overnightPositionsDetectedCount: 0,
      entriesBlockedOccurrences: 0,
    },
    alertFiles: [] as string[],
  };

  for (const day of tradingDays) {
    const report = readDailyReport(day.date);
    const dayData = {
      date: day.date,
      eodFlattenTriggered: report?.eod?.eodFlattenTriggered || false,
      positionsFlattenedCount: report?.eod?.positionsFlattenedCount || 0,
      overnightPositionsDetectedCount: report?.eod?.overnightPositionsDetectedCount || 0,
      lastFlattenTs: report?.eod?.lastFlattenTs || null,
    };

    summary.perDay[day.date] = dayData;
    if (dayData.eodFlattenTriggered) summary.totals.eodFlattenTriggered++;
    summary.totals.positionsFlattenedCount += dayData.positionsFlattenedCount;
    summary.totals.overnightPositionsDetectedCount += dayData.overnightPositionsDetectedCount;
  }

  // Check for alert files
  const alertsDir = path.join(REPORTS_DIR, "alerts");
  if (fs.existsSync(alertsDir)) {
    const alertFiles = fs.readdirSync(alertsDir).filter(f => f.startsWith("overnight_risk_"));
    for (const file of alertFiles) {
      const dateMatch = file.match(/overnight_risk_(\d{4}-\d{2}-\d{2})/);
      if (dateMatch && tradingDays.some(d => d.date === dateMatch[1])) {
        summary.alertFiles.push(file);
      }
    }
  }

  return summary;
}

async function main() {
  console.log("========================================");
  console.log("EXPORT: Last 4 Trading Days Performance");
  console.log("========================================\n");

  // Create output directory
  if (fs.existsSync(OUTPUT_DIR)) {
    fs.rmSync(OUTPUT_DIR, { recursive: true });
  }
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });

  // Get last 4 trading days
  console.log("Fetching trading calendar...");
  const tradingDays = await getLast4TradingDays();
  console.log(`Trading days: ${tradingDays.map(d => d.date).join(", ")}\n`);

  if (tradingDays.length === 0) {
    console.log("ERROR: No trading days found in range");
    return;
  }

  const startDate = tradingDays[0].date;
  const endDate = tradingDays[tradingDays.length - 1].date;

  // 1. Copy daily reports
  console.log("Collecting daily reports...");
  const dailyReportsDir = path.join(OUTPUT_DIR, "daily_reports");
  fs.mkdirSync(dailyReportsDir, { recursive: true });
  
  for (const day of tradingDays) {
    const jsonSrc = path.join(REPORTS_DIR, "daily_reports", `${day.date}.json`);
    const txtSrc = path.join(REPORTS_DIR, "daily_reports", `${day.date}_summary.txt`);
    if (fs.existsSync(jsonSrc)) {
      fs.copyFileSync(jsonSrc, path.join(dailyReportsDir, `${day.date}.json`));
    }
    if (fs.existsSync(txtSrc)) {
      fs.copyFileSync(txtSrc, path.join(dailyReportsDir, `${day.date}_summary.txt`));
    }
  }

  // Copy rolling report
  const rollingJson = path.join(REPORTS_DIR, "summary_last_5d.json");
  if (fs.existsSync(rollingJson)) {
    fs.copyFileSync(rollingJson, path.join(dailyReportsDir, "rolling_summary.json"));
  }

  // 2. Copy weekly scorecard if exists
  console.log("Collecting weekly scorecard...");
  const weeklyDir = path.join(OUTPUT_DIR, "weekly");
  fs.mkdirSync(weeklyDir, { recursive: true });
  
  const weeklyFiles = fs.readdirSync(REPORTS_DIR).filter(f => f.startsWith("weekly_"));
  for (const file of weeklyFiles) {
    fs.copyFileSync(path.join(REPORTS_DIR, file), path.join(weeklyDir, file));
  }

  // 3. Copy activity ledger files and generate aggregate
  console.log("Collecting activity ledger...");
  const activityDir = path.join(OUTPUT_DIR, "activity");
  fs.mkdirSync(activityDir, { recursive: true });
  
  for (const day of tradingDays) {
    const src = path.join(REPORTS_DIR, "activity", `activity_${day.date}.json`);
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, path.join(activityDir, `activity_${day.date}.json`));
    }
  }

  const activityAggregate = generateActivityAggregate(tradingDays);
  fs.writeFileSync(
    path.join(OUTPUT_DIR, "activity_4d_aggregate.json"),
    JSON.stringify(activityAggregate, null, 2)
  );

  // 4. Get Alpaca orders and generate trades joined
  console.log("Fetching Alpaca orders...");
  const orders = await getAlpacaOrders(startDate, endDate);
  console.log(`Found ${orders.length} orders`);

  // Save orders
  fs.writeFileSync(
    path.join(OUTPUT_DIR, "orders_4d.json"),
    JSON.stringify(orders, null, 2)
  );

  // Generate CSV
  const ordersCsv = [
    "id,client_order_id,symbol,side,type,order_class,qty,filled_qty,limit_price,stop_price,filled_avg_price,status,created_at,filled_at",
    ...orders.map((o: any) => [
      o.id,
      o.client_order_id || "",
      o.symbol,
      o.side,
      o.type,
      o.order_class || "",
      o.qty,
      o.filled_qty || "",
      o.limit_price || "",
      o.stop_price || "",
      o.filled_avg_price || "",
      o.status,
      o.created_at,
      o.filled_at || "",
    ].join(","))
  ].join("\n");
  fs.writeFileSync(path.join(OUTPUT_DIR, "orders_4d.csv"), ordersCsv);

  // Generate trades joined
  const trades = generateTradesJoined(orders);
  const tradesCsv = [
    "trade_id,symbol,strategyName,regimeLabel,timeWindow,tier,entry_time,entry_signal_price,entry_fill_price,entry_slippage_bps,qty,exit_time,exit_fill_price,exit_slippage_bps,exitReason,status,realized_pnl,usedAtrFallback,atr,atrPct,stopDistance,rr,parent_order_id,stop_order_id,tp_order_id",
    ...trades.map(t => Object.values(t).join(","))
  ].join("\n");
  fs.writeFileSync(path.join(OUTPUT_DIR, "trades_4d_joined.csv"), tradesCsv);

  // 5. Generate execution summary
  console.log("Generating execution summary...");
  const executionSummary = generateExecutionSummary(orders, tradingDays);
  fs.writeFileSync(
    path.join(OUTPUT_DIR, "execution_4d_summary.json"),
    JSON.stringify(executionSummary, null, 2)
  );

  // 6. Generate EOD/overnight summary
  console.log("Generating EOD/overnight summary...");
  const eodSummary = generateEodOvernightSummary(tradingDays);
  fs.writeFileSync(
    path.join(OUTPUT_DIR, "eod_overnight_4d.json"),
    JSON.stringify(eodSummary, null, 2)
  );

  // Copy alert files
  const alertsDir = path.join(REPORTS_DIR, "alerts");
  if (fs.existsSync(alertsDir)) {
    const alertsOutDir = path.join(OUTPUT_DIR, "alerts");
    fs.mkdirSync(alertsOutDir, { recursive: true });
    for (const file of eodSummary.alertFiles) {
      fs.copyFileSync(path.join(alertsDir, file), path.join(alertsOutDir, file));
    }
  }

  // 7. Get positions and account
  console.log("Fetching account snapshot...");
  try {
    const positions = await alpaca.getPositions();
    fs.writeFileSync(
      path.join(OUTPUT_DIR, "positions_snapshot.json"),
      JSON.stringify(positions, null, 2)
    );

    const account = await alpaca.getAccount();
    fs.writeFileSync(
      path.join(OUTPUT_DIR, "account_snapshot.json"),
      JSON.stringify({
        equity: account.equity,
        cash: account.cash,
        buying_power: account.buying_power,
        portfolio_value: account.portfolio_value,
        last_equity: account.last_equity,
        daytrade_count: account.daytrade_count,
        pattern_day_trader: account.pattern_day_trader,
      }, null, 2)
    );
  } catch (error) {
    console.log(`Warning: Could not fetch account/positions: ${error}`);
  }

  // 8. Create zip
  console.log("\nCreating zip archive...");
  const today = new Date().toISOString().split("T")[0];
  const zipPath = path.join(DIAGNOSTICS_DIR, `ato_last4tradingdays_${today}.zip`);
  
  try {
    execSync(`cd "${OUTPUT_DIR}" && zip -r "${zipPath}" .`, { stdio: "pipe" });
  } catch {
    console.log("Warning: zip command failed, files are still in folder");
  }

  // Print summary
  console.log("\n========================================");
  console.log("EXPORT SUMMARY");
  console.log("========================================");
  console.log(`Trading Days Covered: ${tradingDays.length}`);
  console.log(`Date Range: ${startDate} to ${endDate}`);
  console.log(`\nActivity:`);
  console.log(`  Scan Ticks Total: ${activityAggregate.totals.scanTicks}`);
  console.log(`  Symbols Evaluated: ${activityAggregate.totals.symbolsEvaluated}`);
  console.log(`  No Signal Count: ${activityAggregate.totals.noSignalCount}`);
  console.log(`  Skip Count: ${activityAggregate.totals.skipCount}`);
  console.log(`  Trades Attempted: ${activityAggregate.totals.tradesAttempted}`);
  console.log(`  Trades Filled: ${activityAggregate.totals.tradesFilled}`);
  
  console.log(`\nTrades:`);
  const openTrades = trades.filter(t => t.status === "OPEN").length;
  const closedTrades = trades.filter(t => t.status === "CLOSED").length;
  console.log(`  Total: ${trades.length}`);
  console.log(`  Open: ${openTrades}`);
  console.log(`  Closed: ${closedTrades}`);
  
  console.log(`\nExecution Quality:`);
  console.log(`  Fill Rate: ${(executionSummary.fillRate * 100).toFixed(1)}%`);
  console.log(`  Cancel Rate: ${(executionSummary.cancelRate * 100).toFixed(1)}%`);
  console.log(`  Timeout Rate: ${(executionSummary.timeoutRate * 100).toFixed(1)}%`);
  console.log(`  Avg Time to Fill: ${executionSummary.avgTimeToFillMs.toFixed(0)}ms`);
  
  console.log(`\nTop 10 Skip Reasons:`);
  const skipEntries = Object.entries(activityAggregate.topSkipReasons).slice(0, 10);
  if (skipEntries.length === 0) {
    console.log(`  (none recorded)`);
  } else {
    for (const [reason, count] of skipEntries) {
      console.log(`  ${reason}: ${count}`);
    }
  }
  
  console.log(`\nEOD/Overnight Events:`);
  console.log(`  EOD Flatten Triggered: ${eodSummary.totals.eodFlattenTriggered} days`);
  console.log(`  Positions Flattened: ${eodSummary.totals.positionsFlattenedCount}`);
  console.log(`  Overnight Positions Detected: ${eodSummary.totals.overnightPositionsDetectedCount}`);
  console.log(`  Alert Files: ${eodSummary.alertFiles.length}`);
  
  console.log(`\nOutput:`);
  console.log(`  Folder: ${OUTPUT_DIR}`);
  console.log(`  Zip: ${zipPath}`);
  console.log("\n========================================");
}

main().catch(console.error);
