/**
 * Diagnostic Bundle Exporter
 * 
 * Exports the last 2 weeks (14 calendar days / ~10 trading days) of:
 * - Reports (daily, rolling, weekly scorecards)
 * - Activity ledger data with aggregated summary
 * - Trade lifecycle records (CSV/JSON)
 * - Alpaca orders/positions snapshots
 * - Relevant log sections
 * - Validation reports
 * - Manifest with metadata
 * 
 * Usage: npx tsx scripts/export_diagnostics.ts
 */

import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
import * as alpaca from '../server/alpaca';
import * as tradeLifecycle from '../server/tradeLifecycle';
import { DAY_TRADER_CONFIG } from '../server/dayTraderConfig';

const DAYS_TO_EXPORT = 14;
const REPORTS_DIR = path.join(process.cwd(), 'reports');
const DIAGNOSTICS_DIR = path.join(REPORTS_DIR, 'diagnostics');
const DAILY_REPORTS_DIR = path.join(REPORTS_DIR, 'daily_reports');
const ACTIVITY_DIR = path.join(REPORTS_DIR, 'activity');
const VALIDATION_DIR = path.join(REPORTS_DIR, 'validation');

interface ManifestData {
  exportDate: string;
  dateRangeStart: string;
  dateRangeEnd: string;
  calendarDays: number;
  gitCommit: string | null;
  runtimeSettings: {
    baselineMode: boolean;
    riskPerTrade: string;
    maxTradesPerDay: number;
    maxOpenPositions: number;
    allowedSymbols: string[];
    entryWindow: string;
    forceCloseTime: string;
  };
  fileCounts: {
    dailyReports: number;
    rollingReports: number;
    weeklyScorecards: number;
    activityLedger: number;
    validationReports: number;
    tradeRecords: number;
    alpacaOrders: number;
    logExtracts: number;
  };
  summary: {
    totalScanTicks: number;
    totalSymbolsEvaluated: number;
    totalTrades: number;
    topSkipReasons: Array<{ reason: string; count: number }>;
  };
}

interface ActivityTick {
  tickId: string;
  timestamp: string;
  symbolsEvaluated: number;
  validQuotes: number;
  validBars: number;
  noSignalCount: number;
  skipCount: number;
  skipReasonCounts: Record<string, number>;
  tradesAttempted: number;
  tradesFilled: number;
}

interface ActivityDaySummary {
  date: string;
  totalTicks: number;
  symbolsEvaluated: number;
  validQuotes: number;
  validBars: number;
  noSignalCount: number;
  skipCount: number;
  tradesAttempted: number;
  tradesFilled: number;
  firstTickTs: string | null;
  lastTickTs: string | null;
  skipReasonCounts: Record<string, number>;
}

function getDateRange(): { startDate: Date; endDate: Date; dateStrings: string[] } {
  const endDate = new Date();
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - DAYS_TO_EXPORT);
  
  const dateStrings: string[] = [];
  const current = new Date(startDate);
  while (current <= endDate) {
    dateStrings.push(current.toISOString().split('T')[0]);
    current.setDate(current.getDate() + 1);
  }
  
  return { startDate, endDate, dateStrings };
}

function ensureDir(dir: string): void {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function getGitCommit(): string | null {
  try {
    return execSync('git rev-parse --short HEAD', { encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
}

function copyMatchingFiles(srcDir: string, destDir: string, dateStrings: string[], pattern?: RegExp): number {
  ensureDir(destDir);
  let count = 0;
  
  if (!fs.existsSync(srcDir)) {
    return 0;
  }
  
  const files = fs.readdirSync(srcDir);
  for (const file of files) {
    const matchesDate = dateStrings.some(d => file.includes(d));
    const matchesPattern = !pattern || pattern.test(file);
    const isLatest = file === 'latest.json' || file === 'latest.md';
    
    if ((matchesDate && matchesPattern) || isLatest) {
      const srcPath = path.join(srcDir, file);
      const destPath = path.join(destDir, file);
      if (fs.statSync(srcPath).isFile()) {
        fs.copyFileSync(srcPath, destPath);
        count++;
      }
    }
  }
  
  return count;
}

function aggregateActivityLedger(dateStrings: string[]): { 
  daySummaries: ActivityDaySummary[]; 
  totalTicks: number;
  totalSymbols: number;
  aggregatedSkipReasons: Record<string, number>;
} {
  const daySummaries: ActivityDaySummary[] = [];
  let totalTicks = 0;
  let totalSymbols = 0;
  const aggregatedSkipReasons: Record<string, number> = {};
  
  for (const dateStr of dateStrings) {
    const filePath = path.join(ACTIVITY_DIR, `activity_${dateStr}.json`);
    if (!fs.existsSync(filePath)) continue;
    
    try {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      const ticks: ActivityTick[] = data.ticks || [];
      
      if (ticks.length === 0) continue;
      
      const summary: ActivityDaySummary = {
        date: dateStr,
        totalTicks: ticks.length,
        symbolsEvaluated: 0,
        validQuotes: 0,
        validBars: 0,
        noSignalCount: 0,
        skipCount: 0,
        tradesAttempted: 0,
        tradesFilled: 0,
        firstTickTs: ticks[0]?.timestamp || null,
        lastTickTs: ticks[ticks.length - 1]?.timestamp || null,
        skipReasonCounts: {},
      };
      
      for (const tick of ticks) {
        summary.symbolsEvaluated += tick.symbolsEvaluated || 0;
        summary.validQuotes += tick.validQuotes || 0;
        summary.validBars += tick.validBars || 0;
        summary.noSignalCount += tick.noSignalCount || 0;
        summary.skipCount += tick.skipCount || 0;
        summary.tradesAttempted += tick.tradesAttempted || 0;
        summary.tradesFilled += tick.tradesFilled || 0;
        
        for (const [reason, count] of Object.entries(tick.skipReasonCounts || {})) {
          summary.skipReasonCounts[reason] = (summary.skipReasonCounts[reason] || 0) + (count as number);
          aggregatedSkipReasons[reason] = (aggregatedSkipReasons[reason] || 0) + (count as number);
        }
      }
      
      totalTicks += summary.totalTicks;
      totalSymbols += summary.symbolsEvaluated;
      daySummaries.push(summary);
    } catch (e) {
      console.log(`[Diagnostics] Failed to parse ${filePath}: ${e}`);
    }
  }
  
  return { daySummaries, totalTicks, totalSymbols, aggregatedSkipReasons };
}

async function exportAlpacaOrders(startDate: Date, endDate: Date, destDir: string): Promise<number> {
  ensureDir(destDir);
  
  try {
    const orders = await alpaca.getOrders({
      status: 'all',
      after: startDate.toISOString(),
      until: endDate.toISOString(),
      limit: 500,
    });
    
    const simplified = orders.map((o: any) => ({
      id: o.id,
      client_order_id: o.client_order_id,
      parent_order_id: o.parent_order_id || null,
      symbol: o.symbol,
      side: o.side,
      type: o.type,
      order_class: o.order_class,
      status: o.status,
      qty: o.qty,
      filled_qty: o.filled_qty,
      filled_avg_price: o.filled_avg_price,
      created_at: o.created_at,
      filled_at: o.filled_at,
      submitted_at: o.submitted_at,
      expired_at: o.expired_at,
      canceled_at: o.canceled_at,
      limit_price: o.limit_price,
      stop_price: o.stop_price,
    }));
    
    fs.writeFileSync(
      path.join(destDir, 'alpaca_orders.json'),
      JSON.stringify(simplified, null, 2)
    );
    
    const csvHeader = 'id,client_order_id,parent_order_id,symbol,side,type,order_class,status,qty,filled_qty,filled_avg_price,created_at,filled_at\n';
    const csvRows = simplified.map((o: any) => 
      `${o.id},${o.client_order_id},${o.parent_order_id || ''},${o.symbol},${o.side},${o.type},${o.order_class || ''},${o.status},${o.qty},${o.filled_qty || ''},${o.filled_avg_price || ''},${o.created_at},${o.filled_at || ''}`
    ).join('\n');
    fs.writeFileSync(path.join(destDir, 'alpaca_orders.csv'), csvHeader + csvRows);
    
    return simplified.length;
  } catch (e) {
    console.log(`[Diagnostics] Failed to fetch Alpaca orders: ${e}`);
    return 0;
  }
}

async function exportAlpacaPositions(destDir: string): Promise<void> {
  ensureDir(destDir);
  
  try {
    const positions = await alpaca.getPositions();
    fs.writeFileSync(
      path.join(destDir, 'alpaca_positions_current.json'),
      JSON.stringify(positions, null, 2)
    );
  } catch (e) {
    console.log(`[Diagnostics] Failed to fetch Alpaca positions: ${e}`);
  }
}

async function exportAlpacaAccount(destDir: string): Promise<void> {
  ensureDir(destDir);
  
  try {
    const account = await alpaca.getAccount();
    fs.writeFileSync(
      path.join(destDir, 'alpaca_account.json'),
      JSON.stringify({
        equity: account.equity,
        buying_power: account.buying_power,
        cash: account.cash,
        portfolio_value: account.portfolio_value,
        pattern_day_trader: account.pattern_day_trader,
        daytrade_count: account.daytrade_count,
        daytrading_buying_power: account.daytrading_buying_power,
      }, null, 2)
    );
  } catch (e) {
    console.log(`[Diagnostics] Failed to fetch Alpaca account: ${e}`);
  }
}

function exportTradeLifecycleRecords(destDir: string): number {
  ensureDir(destDir);
  
  try {
    const activeTrades = tradeLifecycle.getActiveTrades();
    const allTrades = (tradeLifecycle as any).getAllTradesForScorecard 
      ? (tradeLifecycle as any).getAllTradesForScorecard() 
      : activeTrades;
    
    const records = allTrades.map((t: any) => ({
      trade_id: t.tradeId,
      symbol: t.symbol,
      side: t.side,
      status: t.status,
      strategyName: t.strategyName || 'UNKNOWN',
      regimeLabel: t.regimeLabel || 'unknown',
      timeWindow: t.timeWindow || 'unknown',
      entrySignalPrice: t.entrySignalPrice,
      entryFillPrice: t.entryFillPrice,
      entrySlippageBps: t.entrySlippageBps,
      exitFillPrice: t.exitFillPrice,
      exitSlippageBps: t.exitSlippageBps,
      exitReason: t.exitReason,
      openedAt: t.createdAt,
      closedAt: t.closedAt,
      realizedPnl: t.realizedPnl,
      parent_order_id: t.entryOrderId,
      stop_order_id: t.stopOrderId,
      tp_order_id: t.tpOrderId,
      usedAtrFallback: t.usedAtrFallback,
      atr: t.atr,
      atrPct: t.atrPct,
      stopDistance: t.stopDistance,
      rr: t.rr,
    }));
    
    fs.writeFileSync(
      path.join(destDir, 'trade_lifecycle_records.json'),
      JSON.stringify(records, null, 2)
    );
    
    if (records.length > 0) {
      const headers = Object.keys(records[0]).join(',');
      const rows = records.map((r: any) => Object.values(r).map(v => v ?? '').join(','));
      fs.writeFileSync(
        path.join(destDir, 'trade_lifecycle_records.csv'),
        headers + '\n' + rows.join('\n')
      );
    }
    
    return records.length;
  } catch (e) {
    console.log(`[Diagnostics] Failed to export trade lifecycle: ${e}`);
    return 0;
  }
}

function extractLogSections(destDir: string, dateStrings: string[]): number {
  ensureDir(destDir);
  
  const patterns = [
    { name: 'eod_flatten', pattern: 'EOD_FLATTEN|EOD_SUMMARY|FORCE_CLOSE|positionsFlattened' },
    { name: 'overnight_positions', pattern: 'overnight_position_detected|OVERNIGHT_FLATTEN|OVERNIGHT_WATCHDOG' },
    { name: 'order_issues', pattern: 'ORDER_REJECTED|UNFILLED_TIMEOUT|QUOTE_STALE_OR_MISSING|SPREAD_NEAR_MAX' },
    { name: 'killswitch', pattern: 'KILL_SWITCH|killswitch|strategy.*disabled' },
    { name: 'trade_actions', pattern: 'ACTION=ENTRY|ACTION=EXIT|ACTION=SKIP|TRADE_OPENED|TRADE_CLOSED' },
    { name: 'errors', pattern: 'ERROR|FAILED|error:|Error:' },
  ];
  
  let totalLines = 0;
  
  for (const { name, pattern } of patterns) {
    try {
      const logFiles = [
        '/tmp/logs/Start_application_*.log',
      ].join(' ');
      
      const result = execSync(
        `grep -iE "${pattern}" ${logFiles} 2>/dev/null | tail -1000 || true`,
        { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 }
      );
      
      if (result.trim()) {
        fs.writeFileSync(path.join(destDir, `log_${name}.txt`), result);
        totalLines += result.split('\n').length;
      }
    } catch {
    }
  }
  
  return totalLines > 0 ? patterns.length : 0;
}

function getRuntimeSettings(): ManifestData['runtimeSettings'] {
  try {
    const config = DAY_TRADER_CONFIG;
    
    return {
      baselineMode: config.BASELINE_MODE ?? true,
      riskPerTrade: `${(config.RISK_PER_TRADE * 100).toFixed(2)}%`,
      maxTradesPerDay: config.MAX_NEW_ENTRIES_PER_DAY ?? 10,
      maxOpenPositions: config.MAX_OPEN_POSITIONS ?? 5,
      allowedSymbols: config.BASELINE_MODE ? config.BASELINE_UNIVERSE : config.ALLOWED_SYMBOLS,
      entryWindow: `${config.WAIT_AFTER_OPEN_MINUTES ?? 5} min after open`,
      forceCloseTime: `${config.MARKET_CLOSE_BUFFER_MINUTES ?? 15} min before close`,
    };
  } catch {
    return {
      baselineMode: true,
      riskPerTrade: 'unknown',
      maxTradesPerDay: 0,
      maxOpenPositions: 0,
      allowedSymbols: [],
      entryWindow: 'unknown',
      forceCloseTime: 'unknown',
    };
  }
}

function createZipBundle(bundleDir: string, outputPath: string): void {
  try {
    execSync(`cd "${bundleDir}" && zip -r "${outputPath}" .`, { encoding: 'utf8' });
  } catch (e) {
    console.log(`[Diagnostics] Failed to create zip: ${e}`);
  }
}

async function main() {
  console.log('========================================');
  console.log('ATOBOT DIAGNOSTIC BUNDLE EXPORTER');
  console.log('========================================');
  
  const { startDate, endDate, dateStrings } = getDateRange();
  const exportDate = new Date().toISOString().split('T')[0];
  const bundleName = `ato_diagnostics_last2weeks_${exportDate}`;
  const bundleDir = path.join(DIAGNOSTICS_DIR, bundleName);
  
  ensureDir(DIAGNOSTICS_DIR);
  ensureDir(bundleDir);
  
  console.log(`\nDate range: ${dateStrings[0]} to ${dateStrings[dateStrings.length - 1]}`);
  console.log(`Bundle dir: ${bundleDir}`);
  
  const manifest: ManifestData = {
    exportDate,
    dateRangeStart: dateStrings[0],
    dateRangeEnd: dateStrings[dateStrings.length - 1],
    calendarDays: DAYS_TO_EXPORT,
    gitCommit: getGitCommit(),
    runtimeSettings: getRuntimeSettings(),
    fileCounts: {
      dailyReports: 0,
      rollingReports: 0,
      weeklyScorecards: 0,
      activityLedger: 0,
      validationReports: 0,
      tradeRecords: 0,
      alpacaOrders: 0,
      logExtracts: 0,
    },
    summary: {
      totalScanTicks: 0,
      totalSymbolsEvaluated: 0,
      totalTrades: 0,
      topSkipReasons: [],
    },
  };
  
  console.log('\n[1/7] Collecting daily reports...');
  manifest.fileCounts.dailyReports = copyMatchingFiles(
    DAILY_REPORTS_DIR, 
    path.join(bundleDir, 'daily_reports'),
    dateStrings
  );
  console.log(`  -> ${manifest.fileCounts.dailyReports} files`);
  
  console.log('\n[2/7] Collecting rolling reports & weekly scorecards...');
  manifest.fileCounts.rollingReports = copyMatchingFiles(
    REPORTS_DIR,
    path.join(bundleDir, 'rolling_reports'),
    dateStrings,
    /summary|trades_joined|skip_reasons|latest/
  );
  manifest.fileCounts.weeklyScorecards = copyMatchingFiles(
    REPORTS_DIR,
    path.join(bundleDir, 'weekly_scorecards'),
    dateStrings,
    /weekly_/
  );
  console.log(`  -> ${manifest.fileCounts.rollingReports} rolling, ${manifest.fileCounts.weeklyScorecards} weekly`);
  
  console.log('\n[3/7] Aggregating activity ledger...');
  manifest.fileCounts.activityLedger = copyMatchingFiles(
    ACTIVITY_DIR,
    path.join(bundleDir, 'activity_ledger'),
    dateStrings
  );
  const { daySummaries, totalTicks, totalSymbols, aggregatedSkipReasons } = aggregateActivityLedger(dateStrings);
  manifest.summary.totalScanTicks = totalTicks;
  manifest.summary.totalSymbolsEvaluated = totalSymbols;
  
  const sortedSkipReasons = Object.entries(aggregatedSkipReasons)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15)
    .map(([reason, count]) => ({ reason, count }));
  manifest.summary.topSkipReasons = sortedSkipReasons;
  
  fs.writeFileSync(
    path.join(bundleDir, 'activity_ledger', 'aggregated_summary.json'),
    JSON.stringify({
      dateRange: { start: dateStrings[0], end: dateStrings[dateStrings.length - 1] },
      totalTicks,
      totalSymbolsEvaluated: totalSymbols,
      topSkipReasons: sortedSkipReasons,
      dailySummaries: daySummaries,
    }, null, 2)
  );
  console.log(`  -> ${manifest.fileCounts.activityLedger} files, ${totalTicks} total ticks`);
  
  console.log('\n[4/7] Exporting trade lifecycle records...');
  manifest.fileCounts.tradeRecords = exportTradeLifecycleRecords(
    path.join(bundleDir, 'trade_lifecycle')
  );
  manifest.summary.totalTrades = manifest.fileCounts.tradeRecords;
  console.log(`  -> ${manifest.fileCounts.tradeRecords} trade records`);
  
  console.log('\n[5/7] Fetching Alpaca snapshots...');
  manifest.fileCounts.alpacaOrders = await exportAlpacaOrders(
    startDate, 
    endDate,
    path.join(bundleDir, 'alpaca_snapshots')
  );
  await exportAlpacaPositions(path.join(bundleDir, 'alpaca_snapshots'));
  await exportAlpacaAccount(path.join(bundleDir, 'alpaca_snapshots'));
  console.log(`  -> ${manifest.fileCounts.alpacaOrders} orders`);
  
  console.log('\n[6/7] Extracting log sections...');
  manifest.fileCounts.logExtracts = extractLogSections(
    path.join(bundleDir, 'logs'),
    dateStrings
  );
  console.log(`  -> ${manifest.fileCounts.logExtracts} log extracts`);
  
  console.log('\n[7/7] Collecting validation reports...');
  manifest.fileCounts.validationReports = copyMatchingFiles(
    VALIDATION_DIR,
    path.join(bundleDir, 'validation_reports'),
    dateStrings
  );
  console.log(`  -> ${manifest.fileCounts.validationReports} validation reports`);
  
  fs.writeFileSync(
    path.join(bundleDir, 'manifest.json'),
    JSON.stringify(manifest, null, 2)
  );
  
  console.log('\nCreating zip bundle...');
  const zipPath = path.join(DIAGNOSTICS_DIR, `${bundleName}.zip`);
  createZipBundle(bundleDir, zipPath);
  
  console.log('\n========================================');
  console.log('DIAGNOSTIC BUNDLE COMPLETE');
  console.log('========================================');
  console.log(`\nZip file: ${zipPath}`);
  console.log(`\nMANIFEST SUMMARY:`);
  console.log(`  Date range: ${manifest.dateRangeStart} to ${manifest.dateRangeEnd}`);
  console.log(`  Git commit: ${manifest.gitCommit || 'N/A'}`);
  console.log(`  Files:`);
  console.log(`    - Daily reports: ${manifest.fileCounts.dailyReports}`);
  console.log(`    - Rolling reports: ${manifest.fileCounts.rollingReports}`);
  console.log(`    - Weekly scorecards: ${manifest.fileCounts.weeklyScorecards}`);
  console.log(`    - Activity ledger: ${manifest.fileCounts.activityLedger}`);
  console.log(`    - Trade records: ${manifest.fileCounts.tradeRecords}`);
  console.log(`    - Alpaca orders: ${manifest.fileCounts.alpacaOrders}`);
  console.log(`    - Validation reports: ${manifest.fileCounts.validationReports}`);
  console.log(`    - Log extracts: ${manifest.fileCounts.logExtracts}`);
  console.log(`  Summary:`);
  console.log(`    - Total scan ticks: ${manifest.summary.totalScanTicks}`);
  console.log(`    - Symbols evaluated: ${manifest.summary.totalSymbolsEvaluated}`);
  console.log(`    - Total trades: ${manifest.summary.totalTrades}`);
  console.log(`    - Top skip reasons: ${manifest.summary.topSkipReasons.slice(0, 5).map(r => `${r.reason}(${r.count})`).join(', ') || 'none'}`);
  console.log(`  Runtime settings:`);
  console.log(`    - Baseline mode: ${manifest.runtimeSettings.baselineMode}`);
  console.log(`    - Risk per trade: ${manifest.runtimeSettings.riskPerTrade}`);
  console.log(`    - Max trades/day: ${manifest.runtimeSettings.maxTradesPerDay}`);
  console.log(`    - Allowed symbols: ${manifest.runtimeSettings.allowedSymbols.join(', ')}`);
  console.log('========================================');
}

main().catch(console.error);
