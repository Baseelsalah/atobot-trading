#!/usr/bin/env tsx
/**
 * Monthly Strategy Analysis Agent
 *
 * Analyzes strategy performance over the past month and makes auto-tuning decisions.
 * Can automatically disable underperforming strategies.
 *
 * Usage: npm run monthly:analyze
 */

import fs from "fs";
import path from "path";

interface StrategyDecision {
  strategy: string;
  action: "KEEP_ACTIVE" | "PROBATION" | "DISABLE" | "ENABLE";
  reason: string;
  stats: {
    trades: number;
    winRate: number;
    profitFactor: number;
    totalPnL: number;
    expectancy: number;
  };
}

const DAILY_REPORTS_DIR = path.join(process.cwd(), "daily_reports");
const MONTHLY_REPORTS_DIR = path.join(process.cwd(), "monthly_reports");
const ALERT_DIR = path.join(process.cwd(), "reports/alerts");
const CONFIG_FILE = path.join(process.cwd(), "server/strategyConfig.json");

function ensureDirectories(): void {
  if (!fs.existsSync(MONTHLY_REPORTS_DIR)) {
    fs.mkdirSync(MONTHLY_REPORTS_DIR, { recursive: true });
  }
}

function getLastMonthDates(): string[] {
  const now = new Date();
  const lastMonth = new Date(now.getFullYear(), now.getMonth() - 1, 1);
  const lastDay = new Date(now.getFullYear(), now.getMonth(), 0);

  const dates: string[] = [];
  for (let d = new Date(lastMonth); d <= lastDay; d.setDate(d.getDate() + 1)) {
    const dayOfWeek = d.getDay();
    if (dayOfWeek !== 0 && dayOfWeek !== 6) {
      // Skip weekends
      dates.push(d.toISOString().split("T")[0]);
    }
  }

  return dates;
}

function loadDailyReports(dates: string[]): any[] {
  const reports: any[] = [];

  for (const date of dates) {
    const filepath = path.join(DAILY_REPORTS_DIR, `${date}.json`);
    if (fs.existsSync(filepath)) {
      try {
        const content = fs.readFileSync(filepath, "utf-8");
        reports.push(JSON.parse(content));
      } catch (error) {
        console.warn(`Failed to load report for ${date}`);
      }
    }
  }

  return reports;
}

function analyzeStrategies(reports: any[]): Map<string, any> {
  const strategyData = new Map<string, any>();

  for (const report of reports) {
    if (!report.strategies) continue;

    for (const [strategy, stats] of Object.entries(report.strategies as any)) {
      if (!strategyData.has(strategy)) {
        strategyData.set(strategy, {
          trades: 0,
          wins: 0,
          losses: 0,
          totalPnL: 0,
          grossProfit: 0,
          grossLoss: 0,
          trades_list: [],
        });
      }

      const data = strategyData.get(strategy);
      data.trades += stats.trades || 0;
      data.wins += stats.wins || 0;
      data.losses += (stats.trades || 0) - (stats.wins || 0);
      data.totalPnL += stats.totalPnL || 0;
      data.grossProfit += stats.grossProfit || 0;
      data.grossLoss += Math.abs(stats.grossLoss || 0);
    }
  }

  return strategyData;
}

function makeStrategyDecisions(strategyData: Map<string, any>): StrategyDecision[] {
  const decisions: StrategyDecision[] = [];

  for (const [strategy, data] of strategyData.entries()) {
    const winRate = data.trades > 0 ? data.wins / data.trades : 0;
    const profitFactor = data.grossLoss > 0 ? data.grossProfit / data.grossLoss : 0;
    const avgPnL = data.trades > 0 ? data.totalPnL / data.trades : 0;
    const expectancy = winRate * avgPnL - (1 - winRate) * Math.abs(avgPnL);

    const stats = {
      trades: data.trades,
      winRate,
      profitFactor,
      totalPnL: data.totalPnL,
      expectancy,
    };

    let action: StrategyDecision["action"] = "KEEP_ACTIVE";
    let reason = "Strategy performing within acceptable parameters";

    // Decision criteria (conservative - require significant data before disabling)
    if (data.trades >= 30) {
      if (profitFactor < 0.9 || winRate < 0.30) {
        action = "DISABLE";
        reason = `Poor performance: ${(winRate * 100).toFixed(1)}% win rate, ${profitFactor.toFixed(2)} profit factor`;
      } else if (profitFactor < 1.1 || winRate < 0.38) {
        action = "PROBATION";
        reason = `Underperforming: ${(winRate * 100).toFixed(1)}% win rate, ${profitFactor.toFixed(2)} profit factor`;
      } else if (profitFactor > 2.0 && winRate > 0.55) {
        action = "KEEP_ACTIVE";
        reason = `Excellent performance: ${(winRate * 100).toFixed(1)}% win rate, ${profitFactor.toFixed(2)} profit factor`;
      }
    } else if (data.trades >= 10) {
      if (profitFactor < 0.8) {
        action = "PROBATION";
        reason = `Early warning: Low profit factor ${profitFactor.toFixed(2)} (limited data)`;
      }
    } else {
      reason = "Insufficient data for decision (need 10+ trades)";
    }

    decisions.push({ strategy, action, reason, stats });
  }

  return decisions;
}

function applyStrategyDecisions(decisions: StrategyDecision[]): void {
  console.log("\n📋 Strategy Decisions:");
  console.log("=====================\n");

  const configData: any = {
    lastUpdate: new Date().toISOString(),
    strategies: {},
  };

  for (const decision of decisions) {
    const icon =
      decision.action === "DISABLE"
        ? "🔴"
        : decision.action === "PROBATION"
        ? "⚠️"
        : "✅";

    console.log(`${icon} ${decision.strategy}: ${decision.action}`);
    console.log(`   Reason: ${decision.reason}`);
    console.log(`   Stats: ${decision.stats.trades} trades, ${(decision.stats.winRate * 100).toFixed(1)}% win rate`);
    console.log(`   P/L: $${decision.stats.totalPnL.toFixed(2)}, PF: ${decision.stats.profitFactor.toFixed(2)}`);
    console.log("");

    configData.strategies[decision.strategy] = {
      enabled: decision.action !== "DISABLE",
      status: decision.action,
      lastAnalysis: new Date().toISOString(),
      stats: decision.stats,
    };

    // Create alert for disabled strategies
    if (decision.action === "DISABLE") {
      const alertFile = path.join(
        ALERT_DIR,
        `WARNING_strategy_disabled_${decision.strategy}_${Date.now()}.txt`
      );
      const alertContent = `
========================================
STRATEGY DISABLED: ${decision.strategy}
========================================

Reason: ${decision.reason}

Performance Stats:
- Trades: ${decision.stats.trades}
- Win Rate: ${(decision.stats.winRate * 100).toFixed(1)}%
- Profit Factor: ${decision.stats.profitFactor.toFixed(2)}
- Total P/L: $${decision.stats.totalPnL.toFixed(2)}
- Expectancy: $${decision.stats.expectancy.toFixed(2)}/trade

Action: Strategy has been automatically disabled.
To re-enable, edit server/strategyConfig.json

Generated: ${new Date().toISOString()}
========================================
      `;

      if (!fs.existsSync(ALERT_DIR)) {
        fs.mkdirSync(ALERT_DIR, { recursive: true });
      }
      fs.writeFileSync(alertFile, alertContent, "utf-8");
    }
  }

  // Save config
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(configData, null, 2), "utf-8");
  console.log(`✅ Strategy configuration saved: ${CONFIG_FILE}\n`);
}

function generateMonthlyReport(decisions: StrategyDecision[], reports: any[]): void {
  const monthYear = new Date().toISOString().slice(0, 7); // YYYY-MM
  const reportFile = path.join(MONTHLY_REPORTS_DIR, `monthly_${monthYear}.txt`);

  const lines: string[] = [];
  lines.push("========================================");
  lines.push(`MONTHLY STRATEGY ANALYSIS - ${monthYear}`);
  lines.push("========================================");
  lines.push("");
  lines.push(`Trading Days Analyzed: ${reports.length}`);
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push("");

  lines.push("STRATEGY DECISIONS");
  lines.push("------------------");
  for (const decision of decisions) {
    const icon =
      decision.action === "DISABLE"
        ? "🔴 DISABLED"
        : decision.action === "PROBATION"
        ? "⚠️  PROBATION"
        : "✅ ACTIVE";
    lines.push(`${icon} - ${decision.strategy}`);
    lines.push(`   ${decision.reason}`);
    lines.push(
      `   Trades: ${decision.stats.trades}, Win Rate: ${(decision.stats.winRate * 100).toFixed(1)}%, PF: ${decision.stats.profitFactor.toFixed(2)}`
    );
    lines.push(`   P/L: $${decision.stats.totalPnL.toFixed(2)}`);
    lines.push("");
  }

  lines.push("========================================");

  fs.writeFileSync(reportFile, lines.join("\n"), "utf-8");
  console.log(`✅ Monthly report saved: ${reportFile}\n`);
}

async function main(): Promise<void> {
  console.log("\n📊 Monthly Strategy Analysis");
  console.log("========================================\n");

  ensureDirectories();

  const dates = getLastMonthDates();
  console.log(`Analyzing: ${dates[0]} to ${dates[dates.length - 1]}`);
  console.log(`Trading days: ${dates.length}\n`);

  const reports = loadDailyReports(dates);
  console.log(`Loaded ${reports.length} daily reports\n`);

  if (reports.length < 5) {
    console.log("⚠️ Insufficient data for monthly analysis (need 5+ trading days)");
    console.log("   This is normal if you just started trading.\n");
    return;
  }

  const strategyData = analyzeStrategies(reports);
  console.log(`Analyzed ${strategyData.size} strategies\n`);

  const decisions = makeStrategyDecisions(strategyData);
  applyStrategyDecisions(decisions);
  generateMonthlyReport(decisions, reports);

  console.log("✅ Monthly analysis complete\n");
}

main().catch((error) => {
  console.error("❌ Monthly analysis failed:", error);
  process.exit(1);
});
