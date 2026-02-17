/**
 * Report Generator - Output JSON reports, CSV trade logs, and console summaries
 */

import * as fs from "fs";
import * as path from "path";
import type { BacktestReport, BacktestTrade, BacktestMetrics, StrategyMetrics } from "./types";

export class ReportGenerator {
  /**
   * Write the full JSON report to disk.
   */
  static writeJSON(report: BacktestReport, outputPath: string): string {
    if (!fs.existsSync(outputPath)) {
      fs.mkdirSync(outputPath, { recursive: true });
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").substring(0, 19);
    const filename = path.join(outputPath, `backtest_${timestamp}.json`);

    // Strip large equity curve data from JSON for manageability
    const reportForFile = {
      ...report,
      aggregate: {
        ...report.aggregate,
        equityCurve: `[${report.aggregate.equityCurve.length} points - see separate file]`,
        dailyReturns: `[${report.aggregate.dailyReturns.length} points - see separate file]`,
      },
    };

    fs.writeFileSync(filename, JSON.stringify(reportForFile, null, 2), "utf-8");

    // Write equity curve as separate file
    const eqFilename = path.join(outputPath, `equity_curve_${timestamp}.json`);
    fs.writeFileSync(eqFilename, JSON.stringify(report.aggregate.equityCurve, null, 2), "utf-8");

    console.log(`\nJSON report saved to: ${filename}`);
    console.log(`Equity curve saved to: ${eqFilename}`);
    return filename;
  }

  /**
   * Write trade log as CSV.
   */
  static writeCSV(trades: BacktestTrade[], outputPath: string): string {
    if (!fs.existsSync(outputPath)) {
      fs.mkdirSync(outputPath, { recursive: true });
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").substring(0, 19);
    const filename = path.join(outputPath, `trades_${timestamp}.csv`);

    const headers = [
      "id", "symbol", "strategy", "side", "confidence",
      "signal_price", "entry_price", "slippage_bps",
      "stop_price", "target_price",
      "exit_price", "exit_time", "exit_reason",
      "quantity", "pnl", "pnl_pct",
      "holding_minutes", "r_multiple", "regime",
      "entry_time",
    ].join(",");

    const rows = trades.map(t => [
      t.id,
      t.symbol,
      t.strategyName,
      t.side,
      t.confidence,
      t.signalPrice?.toFixed(4),
      t.entryPrice?.toFixed(4),
      t.slippageBps?.toFixed(2),
      t.initialStopPrice?.toFixed(4),
      t.targetPrice?.toFixed(4),
      t.exitPrice?.toFixed(4) ?? "",
      t.exitTime ?? "",
      t.exitReason ?? "",
      t.quantity,
      t.realizedPnl?.toFixed(2) ?? "",
      t.realizedPnlPct?.toFixed(4) ?? "",
      t.holdingTimeMinutes?.toFixed(1) ?? "",
      t.rMultiple?.toFixed(3) ?? "",
      t.regime,
      t.entryTime,
    ].join(","));

    fs.writeFileSync(filename, [headers, ...rows].join("\n"), "utf-8");
    console.log(`CSV trade log saved to: ${filename}`);
    return filename;
  }

  /**
   * Print summary to console.
   */
  static printSummary(report: BacktestReport): void {
    const m = report.aggregate;
    const cfg = report.config;

    const finalEquity = cfg.initialEquity + m.totalReturnDollars;

    console.log("\n" + "=".repeat(60));
    console.log("  AtoBot Backtest Results");
    console.log("=".repeat(60));
    console.log(`  Period: ${cfg.startDate} to ${cfg.endDate} (${m.totalTradingDays} trading days)`);
    console.log(`  Symbols: ${cfg.symbols.length} | Strategies: ${cfg.strategies.join(", ")}`);
    console.log(`  Slippage: ${cfg.slippageMode} (${cfg.slippageFixedBps}bps)`);
    console.log(`  Initial Equity: $${cfg.initialEquity.toLocaleString()} -> Final: $${finalEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);

    console.log("\n  AGGREGATE METRICS:");
    console.log(`    Total Trades:     ${m.totalTrades}`);
    console.log(`    Win Rate:         ${m.winRate.toFixed(1)}% (${m.winningTrades}W / ${m.losingTrades}L)`);
    console.log(`    Profit Factor:    ${m.profitFactor === Infinity ? "Inf" : m.profitFactor.toFixed(2)}`);
    console.log(`    Expectancy:       $${m.expectancy.toFixed(2)}/trade`);
    console.log(`    Expectancy R:     ${m.expectancyR.toFixed(3)}R`);
    console.log(`    Total Return:     ${m.totalReturn.toFixed(2)}% ($${m.totalReturnDollars.toFixed(2)})`);
    console.log(`    Annualized:       ${m.annualizedReturn.toFixed(2)}%`);
    console.log(`    Sharpe Ratio:     ${m.sharpeRatio.toFixed(2)}`);
    console.log(`    Sortino Ratio:    ${m.sortinoRatio === Infinity ? "Inf" : m.sortinoRatio.toFixed(2)}`);
    console.log(`    Calmar Ratio:     ${m.calmarRatio.toFixed(2)}`);
    console.log(`    Max Drawdown:     -$${m.maxDrawdownDollars.toFixed(2)} (-${m.maxDrawdownPercent.toFixed(2)}%)`);
    console.log(`    Max DD Duration:  ${m.maxDrawdownDurationDays} days`);
    console.log(`    Avg Win:          $${m.avgWin.toFixed(2)}`);
    console.log(`    Avg Loss:         -$${m.avgLoss.toFixed(2)}`);
    console.log(`    Largest Win:      $${m.largestWin.toFixed(2)}`);
    console.log(`    Largest Loss:     $${m.largestLoss.toFixed(2)}`);
    console.log(`    Avg Hold Time:    ${m.avgHoldingTimeMinutes.toFixed(1)} min`);
    console.log(`    Trading Days:     ${m.daysWithTrades}/${m.totalTradingDays} (${m.avgTradesPerDay.toFixed(1)} trades/day)`);

    if (report.perStrategy.length > 0) {
      console.log("\n  PER STRATEGY:");
      for (const s of report.perStrategy) {
        const stratPnl = s.totalReturnDollars;
        const pnlSign = stratPnl >= 0 ? "+" : "";
        console.log(
          `    ${s.strategyName.padEnd(18)} ${String(s.totalTrades).padStart(3)} trades | ` +
          `${s.winRate.toFixed(1)}% WR | ` +
          `PF ${s.profitFactor === Infinity ? "Inf" : s.profitFactor.toFixed(2)} | ` +
          `${pnlSign}$${stratPnl.toFixed(2)}`
        );
      }
    }

    // Exit reason breakdown
    const exitReasons = new Map<string, number>();
    for (const trade of report.trades) {
      if (trade.exitReason) {
        exitReasons.set(trade.exitReason, (exitReasons.get(trade.exitReason) || 0) + 1);
      }
    }
    if (exitReasons.size > 0) {
      console.log("\n  EXIT REASONS:");
      for (const [reason, count] of Array.from(exitReasons.entries()).sort((a, b) => b[1] - a[1])) {
        console.log(`    ${reason.padEnd(20)} ${count}`);
      }
    }

    console.log("\n" + "=".repeat(60));
  }
}
