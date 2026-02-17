/**
 * AtoBot Backtesting CLI
 *
 * Usage:
 *   npx tsx server/backtest/index.ts [options]
 *
 * Options:
 *   --start YYYY-MM-DD        Start date (default: 3 months ago)
 *   --end YYYY-MM-DD          End date (default: yesterday)
 *   --symbols SYM1,SYM2       Symbols to test (default: all from config)
 *   --strategies S1,S2        Strategies to enable (default: all 3)
 *   --equity NUMBER           Starting equity (default: 100000)
 *   --slippage fixed|none     Slippage model (default: fixed)
 *   --slippage-bps NUMBER     Slippage basis points (default: 5)
 *   --confidence NUMBER       Min confidence threshold (default: 55)
 *   --output PATH             Output directory (default: reports/backtest)
 *   --format json,csv         Output formats (default: json,csv)
 *   --no-cache                Force re-fetch data
 *   --verbose                 Detailed console output
 */

import { BacktestEngine } from "./backtestEngine";
import { DataFetcher } from "./dataFetcher";
import { MetricsCalculator } from "./metricsCalculator";
import { ReportGenerator } from "./reportGenerator";
import { DEFAULT_CONFIG, type BacktestConfig } from "./types";
import type { StrategyName } from "../strategyEngine";

function parseArgs(argv: string[]): BacktestConfig {
  const config = { ...DEFAULT_CONFIG };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = argv[i + 1];

    switch (arg) {
      case "--start":
        config.startDate = next;
        i++;
        break;
      case "--end":
        config.endDate = next;
        i++;
        break;
      case "--symbols":
        config.symbols = next.split(",").map(s => s.trim().toUpperCase());
        i++;
        break;
      case "--strategies":
        config.strategies = next.split(",").map(s => s.trim()) as StrategyName[];
        i++;
        break;
      case "--equity":
        config.initialEquity = parseFloat(next);
        i++;
        break;
      case "--slippage":
        config.slippageMode = next as "none" | "fixed" | "pctOfSpread";
        i++;
        break;
      case "--slippage-bps":
        config.slippageFixedBps = parseFloat(next);
        i++;
        break;
      case "--confidence":
        config.minConfidence = parseFloat(next);
        i++;
        break;
      case "--output":
        config.outputPath = next;
        i++;
        break;
      case "--format":
        config.outputFormat = next.split(",").map(s => s.trim()) as ("json" | "csv")[];
        i++;
        break;
      case "--no-cache":
        config.noCache = true;
        break;
      case "--verbose":
        config.verbose = true;
        break;
      default:
        if (arg.startsWith("--")) {
          console.warn(`Unknown option: ${arg}`);
        }
    }
  }

  // Default dates: 3 months ago to yesterday
  if (!config.startDate) {
    const d = new Date();
    d.setMonth(d.getMonth() - 3);
    config.startDate = d.toISOString().substring(0, 10);
  }
  if (!config.endDate) {
    const d = new Date();
    d.setDate(d.getDate() - 1);
    config.endDate = d.toISOString().substring(0, 10);
  }

  return config;
}

function validateConfig(config: BacktestConfig): void {
  if (!config.startDate || !config.endDate) {
    throw new Error("Start and end dates are required");
  }
  if (config.startDate >= config.endDate) {
    throw new Error("Start date must be before end date");
  }
  if (config.initialEquity <= 0) {
    throw new Error("Initial equity must be positive");
  }
  if (config.symbols.length === 0) {
    throw new Error("At least one symbol is required");
  }
  if (!process.env.ALPACA_API_KEY && !process.env.APCA_API_KEY_ID) {
    throw new Error("ALPACA_API_KEY or APCA_API_KEY_ID environment variable is required");
  }
  if (!process.env.ALPACA_SECRET_KEY && !process.env.ALPACA_API_SECRET && !process.env.APCA_API_SECRET_KEY) {
    throw new Error("ALPACA_SECRET_KEY or ALPACA_API_SECRET environment variable is required");
  }
}

async function main(): Promise<void> {
  const config = parseArgs(process.argv.slice(2));

  try {
    validateConfig(config);
  } catch (error: any) {
    console.error(`Error: ${error.message}`);
    console.log("\nUsage: npx tsx server/backtest/index.ts [options]");
    console.log("  --start YYYY-MM-DD   Start date (default: 3 months ago)");
    console.log("  --end YYYY-MM-DD     End date (default: yesterday)");
    console.log("  --symbols SYM1,SYM2  Symbols (default: full config universe)");
    console.log("  --strategies S1,S2   VWAP_REVERSION,ORB,EMA_CROSSOVER");
    console.log("  --equity NUMBER      Starting equity (default: 100000)");
    console.log("  --slippage MODE      none|fixed|pctOfSpread (default: fixed)");
    console.log("  --slippage-bps N     Slippage basis points (default: 5)");
    console.log("  --confidence N       Min confidence threshold (default: 55)");
    console.log("  --output PATH        Output directory (default: reports/backtest)");
    console.log("  --format FMT         json,csv (default: json,csv)");
    console.log("  --no-cache           Force re-fetch data");
    console.log("  --verbose            Detailed output");
    process.exit(1);
  }

  console.log("=".repeat(60));
  console.log("  AtoBot Backtesting Framework");
  console.log("=".repeat(60));
  console.log(`  Period:     ${config.startDate} to ${config.endDate}`);
  console.log(`  Symbols:    ${config.symbols.join(", ")} (${config.symbols.length})`);
  console.log(`  Strategies: ${config.strategies.join(", ")}`);
  console.log(`  Equity:     $${config.initialEquity.toLocaleString()}`);
  console.log(`  Slippage:   ${config.slippageMode} (${config.slippageFixedBps}bps)`);
  console.log(`  Confidence: >= ${config.minConfidence}%`);
  console.log("");

  // Get API keys
  const apiKey = process.env.ALPACA_API_KEY || process.env.APCA_API_KEY_ID || "";
  const secretKey = process.env.ALPACA_SECRET_KEY || process.env.ALPACA_API_SECRET || process.env.APCA_API_SECRET_KEY || "";

  // 1. Fetch historical data
  console.log("Fetching historical data...");
  const fetcher = new DataFetcher(apiKey, secretKey);

  const tradingDays = await fetcher.fetchTradingCalendar(config.startDate, config.endDate);
  console.log(`  Trading days: ${tradingDays.length}`);

  const data = await fetcher.fetchAllData(
    config.symbols,
    config.startDate,
    config.endDate,
    config.cachePath,
    config.noCache
  );

  // Log data summary
  let totalBars = 0;
  for (const [symbol, symbolData] of Array.from(data.entries())) {
    totalBars += symbolData.bars5min.length + symbolData.bars1min.length;
    if (config.verbose) {
      console.log(`  ${symbol}: ${symbolData.bars5min.length} 5-min bars, ${symbolData.bars1min.length} 1-min bars`);
    }
  }
  console.log(`  Total bars loaded: ${totalBars.toLocaleString()}`);

  // 2. Run backtest
  console.log("\nRunning backtest...");
  const engine = new BacktestEngine(config);
  const partialReport = await engine.run(data, tradingDays);

  // 3. Compute metrics
  console.log("Computing metrics...");
  const equityCurve = engine.getEquityCurve();
  const aggregate = MetricsCalculator.compute(
    partialReport.trades,
    equityCurve,
    config.initialEquity,
    tradingDays.length
  );
  const perStrategy = MetricsCalculator.computePerStrategy(
    partialReport.trades,
    equityCurve,
    config.initialEquity,
    tradingDays.length
  );

  const report = {
    ...partialReport,
    aggregate,
    perStrategy,
  };

  // 4. Output results
  ReportGenerator.printSummary(report);

  if (config.outputFormat.includes("json")) {
    ReportGenerator.writeJSON(report, config.outputPath);
  }
  if (config.outputFormat.includes("csv")) {
    ReportGenerator.writeCSV(report.trades, config.outputPath);
  }
}

main().catch(err => {
  console.error("Backtest failed:", err);
  process.exit(1);
});
