#!/usr/bin/env tsx
/**
 * Weekly Performance Report Agent
 *
 * Analyzes the week's trading performance and sends a comprehensive report.
 * Runs automatically every Sunday evening.
 *
 * Usage: npm run weekly:report
 */

import fs from "fs";
import path from "path";

interface DailyReport {
  date: string;
  totalTrades: number;
  wins: number;
  losses: number;
  winRate: number;
  totalPnL: number;
  largestWin: number;
  largestLoss: number;
  avgWin: number;
  avgLoss: number;
  strategies: Record<string, any>;
}

interface WeeklyReport {
  weekStart: string;
  weekEnd: string;
  totalTrades: number;
  totalWins: number;
  totalLosses: number;
  totalBreakeven: number;
  overallWinRate: number;
  totalPnL: number;
  bestDay: { date: string; pnl: number };
  worstDay: { date: string; pnl: number };
  avgDailyPnL: number;
  profitFactor: number;
  largestWin: number;
  largestLoss: number;
  avgWin: number;
  avgLoss: number;
  expectancy: number;
  strategyPerformance: Record<string, StrategyStats>;
  symbolPerformance: Record<string, SymbolStats>;
  recommendations: string[];
}

interface StrategyStats {
  trades: number;
  wins: number;
  winRate: number;
  totalPnL: number;
  avgPnL: number;
  profitFactor: number;
  status: "ACTIVE" | "PROBATION" | "DISABLED";
}

interface SymbolStats {
  trades: number;
  wins: number;
  winRate: number;
  totalPnL: number;
  avgPnL: number;
}

const REPORTS_DIR = path.join(process.cwd(), "daily_reports");
const WEEKLY_DIR = path.join(process.cwd(), "weekly_reports");
const ALERT_DIR = path.join(process.cwd(), "reports/alerts");

function ensureDirectories(): void {
  if (!fs.existsSync(WEEKLY_DIR)) {
    fs.mkdirSync(WEEKLY_DIR, { recursive: true });
  }
}

function getLastWeekDates(): { start: Date; end: Date; dates: string[] } {
  const now = new Date();
  const dayOfWeek = now.getDay(); // 0 = Sunday

  // Last Sunday
  const lastSunday = new Date(now);
  lastSunday.setDate(now.getDate() - dayOfWeek - 7);
  lastSunday.setHours(0, 0, 0, 0);

  // Last Saturday
  const lastSaturday = new Date(lastSunday);
  lastSaturday.setDate(lastSunday.getDate() + 6);
  lastSaturday.setHours(23, 59, 59, 999);

  // Trading days (Mon-Fri only)
  const dates: string[] = [];
  for (let i = 1; i <= 5; i++) {
    const date = new Date(lastSunday);
    date.setDate(lastSunday.getDate() + i);
    dates.push(date.toISOString().split("T")[0]);
  }

  return { start: lastSunday, end: lastSaturday, dates };
}

function loadDailyReports(dates: string[]): DailyReport[] {
  const reports: DailyReport[] = [];

  for (const date of dates) {
    const filepath = path.join(REPORTS_DIR, `${date}.json`);
    if (fs.existsSync(filepath)) {
      try {
        const content = fs.readFileSync(filepath, "utf-8");
        const report = JSON.parse(content) as DailyReport;
        reports.push(report);
      } catch (error) {
        console.warn(`Failed to load report for ${date}:`, error);
      }
    }
  }

  return reports;
}

function analyzeWeek(reports: DailyReport[]): WeeklyReport {
  if (reports.length === 0) {
    throw new Error("No trading data found for last week");
  }

  let totalTrades = 0;
  let totalWins = 0;
  let totalLosses = 0;
  let totalBreakeven = 0;
  let totalPnL = 0;
  let totalGrossProfit = 0;
  let totalGrossLoss = 0;
  let largestWin = 0;
  let largestLoss = 0;
  let bestDay = { date: "", pnl: -Infinity };
  let worstDay = { date: "", pnl: Infinity };

  const strategyStats: Record<string, any> = {};
  const symbolStats: Record<string, any> = {};

  for (const report of reports) {
    totalTrades += report.totalTrades || 0;
    totalWins += report.wins || 0;
    totalLosses += report.losses || 0;

    const dayPnL = report.totalPnL || 0;
    totalPnL += dayPnL;

    if (dayPnL > bestDay.pnl) {
      bestDay = { date: report.date, pnl: dayPnL };
    }
    if (dayPnL < worstDay.pnl) {
      worstDay = { date: report.date, pnl: dayPnL };
    }

    if (report.largestWin > largestWin) largestWin = report.largestWin;
    if (report.largestLoss < largestLoss) largestLoss = report.largestLoss;

    // Aggregate strategy stats
    if (report.strategies) {
      for (const [strategy, stats] of Object.entries(report.strategies)) {
        if (!strategyStats[strategy]) {
          strategyStats[strategy] = {
            trades: 0,
            wins: 0,
            totalPnL: 0,
            grossProfit: 0,
            grossLoss: 0,
          };
        }
        strategyStats[strategy].trades += stats.trades || 0;
        strategyStats[strategy].wins += stats.wins || 0;
        strategyStats[strategy].totalPnL += stats.totalPnL || 0;
        strategyStats[strategy].grossProfit += stats.grossProfit || 0;
        strategyStats[strategy].grossLoss += Math.abs(stats.grossLoss || 0);
      }
    }
  }

  const overallWinRate = totalTrades > 0 ? totalWins / totalTrades : 0;
  const avgDailyPnL = reports.length > 0 ? totalPnL / reports.length : 0;

  // Calculate strategy performance
  const strategyPerformance: Record<string, StrategyStats> = {};
  for (const [strategy, stats] of Object.entries(strategyStats)) {
    const winRate = stats.trades > 0 ? stats.wins / stats.trades : 0;
    const avgPnL = stats.trades > 0 ? stats.totalPnL / stats.trades : 0;
    const profitFactor = stats.grossLoss > 0 ? stats.grossProfit / stats.grossLoss : 0;

    let status: "ACTIVE" | "PROBATION" | "DISABLED" = "ACTIVE";
    if (stats.trades >= 10) {
      if (winRate < 0.35 || profitFactor < 1.0) {
        status = "DISABLED";
      } else if (winRate < 0.40 || profitFactor < 1.2) {
        status = "PROBATION";
      }
    }

    strategyPerformance[strategy] = {
      trades: stats.trades,
      wins: stats.wins,
      winRate,
      totalPnL: stats.totalPnL,
      avgPnL,
      profitFactor,
      status,
    };
  }

  // Calculate profit factor
  if (totalGrossProfit === 0) totalGrossProfit = totalWins * 10; // Estimate
  if (totalGrossLoss === 0) totalGrossLoss = Math.abs(totalLosses * 10); // Estimate
  const profitFactor = totalGrossLoss > 0 ? totalGrossProfit / totalGrossLoss : 0;

  // Calculate expectancy
  const avgWin = totalWins > 0 ? totalGrossProfit / totalWins : 0;
  const avgLoss = totalLosses > 0 ? totalGrossLoss / totalLosses : 0;
  const expectancy = overallWinRate * avgWin - (1 - overallWinRate) * avgLoss;

  // Generate recommendations
  const recommendations = generateRecommendations({
    totalTrades,
    overallWinRate,
    totalPnL,
    profitFactor,
    expectancy,
    strategyPerformance,
  });

  return {
    weekStart: reports[0]?.date || "",
    weekEnd: reports[reports.length - 1]?.date || "",
    totalTrades,
    totalWins,
    totalLosses,
    totalBreakeven,
    overallWinRate,
    totalPnL,
    bestDay,
    worstDay,
    avgDailyPnL,
    profitFactor,
    largestWin,
    largestLoss,
    avgWin,
    avgLoss,
    expectancy,
    strategyPerformance,
    symbolPerformance: {},
    recommendations,
  };
}

function generateRecommendations(data: any): string[] {
  const recs: string[] = [];

  // Win rate analysis
  if (data.overallWinRate < 0.40) {
    recs.push("⚠️ Win rate below 40% - Review entry signals and consider tighter filters");
  } else if (data.overallWinRate > 0.60) {
    recs.push("✅ Excellent win rate - Consider increasing position sizes");
  }

  // Profit factor analysis
  if (data.profitFactor < 1.0) {
    recs.push("🔴 CRITICAL: Profit factor below 1.0 - Bot is losing money overall");
  } else if (data.profitFactor < 1.2) {
    recs.push("⚠️ Profit factor below 1.2 - Reduce risk or disable losing strategies");
  } else if (data.profitFactor > 2.0) {
    recs.push("✅ Strong profit factor - Current strategies are profitable");
  }

  // Expectancy analysis
  if (data.expectancy < 0) {
    recs.push("🔴 Negative expectancy - Expected to lose money per trade");
  } else if (data.expectancy > 5) {
    recs.push("✅ Positive expectancy - Expected to profit $" + data.expectancy.toFixed(2) + " per trade");
  }

  // Strategy-specific
  for (const [strategy, stats] of Object.entries(data.strategyPerformance)) {
    if (stats.status === "DISABLED") {
      recs.push(`🔴 DISABLE ${strategy} - Underperforming (${(stats.winRate * 100).toFixed(1)}% win rate)`);
    } else if (stats.status === "PROBATION") {
      recs.push(`⚠️ WATCH ${strategy} - On probation (${(stats.winRate * 100).toFixed(1)}% win rate)`);
    }
  }

  // Volume analysis
  if (data.totalTrades < 10) {
    recs.push("⚠️ Low trade volume - Not enough data for reliable analysis");
  }

  return recs;
}

function formatReport(report: WeeklyReport): string {
  const lines: string[] = [];

  lines.push("========================================");
  lines.push("📊 WEEKLY PERFORMANCE REPORT");
  lines.push("========================================");
  lines.push("");
  lines.push(`Week: ${report.weekStart} to ${report.weekEnd}`);
  lines.push("");

  lines.push("SUMMARY");
  lines.push("--------");
  lines.push(`Total Trades: ${report.totalTrades}`);
  lines.push(`Wins: ${report.totalWins} (${(report.overallWinRate * 100).toFixed(1)}%)`);
  lines.push(`Losses: ${report.totalLosses}`);
  lines.push(`Total P/L: $${report.totalPnL.toFixed(2)}`);
  lines.push(`Avg Daily P/L: $${report.avgDailyPnL.toFixed(2)}`);
  lines.push(`Profit Factor: ${report.profitFactor.toFixed(2)}`);
  lines.push(`Expectancy: $${report.expectancy.toFixed(2)}/trade`);
  lines.push("");

  lines.push("BEST/WORST");
  lines.push("-----------");
  lines.push(`Best Day: ${report.bestDay.date} ($${report.bestDay.pnl.toFixed(2)})`);
  lines.push(`Worst Day: ${report.worstDay.date} ($${report.worstDay.pnl.toFixed(2)})`);
  lines.push(`Largest Win: $${report.largestWin.toFixed(2)}`);
  lines.push(`Largest Loss: $${report.largestLoss.toFixed(2)}`);
  lines.push("");

  lines.push("STRATEGY PERFORMANCE");
  lines.push("--------------------");
  for (const [strategy, stats] of Object.entries(report.strategyPerformance)) {
    const statusIcon = stats.status === "ACTIVE" ? "✅" : stats.status === "PROBATION" ? "⚠️" : "🔴";
    lines.push(`${statusIcon} ${strategy}:`);
    lines.push(`   Trades: ${stats.trades}, Win Rate: ${(stats.winRate * 100).toFixed(1)}%`);
    lines.push(`   P/L: $${stats.totalPnL.toFixed(2)}, Avg: $${stats.avgPnL.toFixed(2)}`);
    lines.push(`   Profit Factor: ${stats.profitFactor.toFixed(2)}`);
    lines.push(`   Status: ${stats.status}`);
    lines.push("");
  }

  lines.push("RECOMMENDATIONS");
  lines.push("----------------");
  if (report.recommendations.length === 0) {
    lines.push("✅ No issues detected - Continue current approach");
  } else {
    for (const rec of report.recommendations) {
      lines.push(rec);
    }
  }
  lines.push("");

  lines.push("========================================");
  lines.push("Generated: " + new Date().toISOString());
  lines.push("========================================");

  return lines.join("\n");
}

async function sendReport(report: WeeklyReport, formatted: string): Promise<void> {
  // Save to file
  const filename = `weekly_${report.weekEnd}.json`;
  fs.writeFileSync(
    path.join(WEEKLY_DIR, filename),
    JSON.stringify(report, null, 2),
    "utf-8"
  );

  const textFilename = `weekly_${report.weekEnd}.txt`;
  fs.writeFileSync(
    path.join(WEEKLY_DIR, textFilename),
    formatted,
    "utf-8"
  );

  console.log(`✅ Weekly report saved: ${filename}`);
  console.log(`✅ Text report saved: ${textFilename}`);

  // Create alert
  if (!fs.existsSync(ALERT_DIR)) {
    fs.mkdirSync(ALERT_DIR, { recursive: true });
  }

  const alertFile = path.join(ALERT_DIR, `INFO_weekly_report_${Date.now()}.txt`);
  fs.writeFileSync(alertFile, formatted, "utf-8");

  // Future: Send email/SMS
  console.log("\n📧 Email delivery not configured yet - check weekly_reports/ directory");
}

async function main(): Promise<void> {
  console.log("\n📊 Generating Weekly Performance Report");
  console.log("========================================\n");

  ensureDirectories();

  const { start, end, dates } = getLastWeekDates();
  console.log(`Week: ${start.toISOString().split("T")[0]} to ${end.toISOString().split("T")[0]}`);
  console.log(`Trading Days: ${dates.join(", ")}\n`);

  const dailyReports = loadDailyReports(dates);
  console.log(`Loaded ${dailyReports.length} daily reports\n`);

  if (dailyReports.length === 0) {
    console.log("⚠️ No trading data found for last week");
    console.log("   This is normal if you haven't traded yet.\n");
    return;
  }

  const weeklyReport = analyzeWeek(dailyReports);
  const formatted = formatReport(weeklyReport);

  console.log(formatted);
  console.log("");

  await sendReport(weeklyReport, formatted);

  console.log("\n✅ Weekly report generation complete\n");
}

main().catch((error) => {
  console.error("❌ Weekly report generation failed:", error);
  process.exit(1);
});
