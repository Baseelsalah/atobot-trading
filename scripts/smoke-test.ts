#!/usr/bin/env tsx
/**
 * Pre-Market Smoke Test & Validation
 *
 * Run this script before market open (e.g., 8:30-9:20 AM ET) to verify:
 * - Bot connectivity to Alpaca
 * - Clock synchronization
 * - All critical systems operational
 * - No overnight positions (day trading rule)
 * - Configuration is correct
 *
 * Usage: npm run smoke-test
 */

import "dotenv/config";
import * as alpaca from "../server/alpaca.js";
import { getEasternTime } from "../server/timezone.js";
import * as dayTraderConfig from "../server/dayTraderConfig.js";
import { loadDailyState } from "../server/persistentState.js";

interface TestResult {
  name: string;
  status: "PASS" | "FAIL" | "WARN";
  message: string;
  details?: any;
}

const results: TestResult[] = [];

function logTest(name: string, status: "PASS" | "FAIL" | "WARN", message: string, details?: any) {
  results.push({ name, status, message, details });
  const icon = status === "PASS" ? "✅" : status === "FAIL" ? "❌" : "⚠️";
  console.log(`${icon} ${name}: ${message}`);
  if (details) {
    console.log(`   Details: ${JSON.stringify(details, null, 2)}`);
  }
}

async function testAlpacaConnection() {
  console.log("\n========================================");
  console.log("1. ALPACA CONNECTION TEST");
  console.log("========================================");

  try {
    const account = await alpaca.getAccount();
    logTest(
      "Alpaca API Connection",
      "PASS",
      `Connected to account: ${account.account_number}`,
      {
        buyingPower: account.buying_power,
        cash: account.cash,
        portfolioValue: account.portfolio_value,
      }
    );

    // Check if paper trading
    const isPaper = account.account_number.startsWith("PA");
    if (isPaper) {
      logTest("Account Type", "WARN", "Using PAPER trading account (safe for testing)");
    } else {
      logTest("Account Type", "PASS", "Using LIVE trading account");
    }

    // Check buying power
    const buyingPower = parseFloat(account.buying_power);
    if (buyingPower < 1000) {
      logTest("Buying Power", "WARN", `Low buying power: $${buyingPower.toFixed(2)}`);
    } else {
      logTest("Buying Power", "PASS", `Sufficient buying power: $${buyingPower.toFixed(2)}`);
    }
  } catch (error) {
    logTest(
      "Alpaca API Connection",
      "FAIL",
      `Failed to connect: ${error instanceof Error ? error.message : error}`
    );
    return false;
  }

  return true;
}

async function testClockSync() {
  console.log("\n========================================");
  console.log("2. CLOCK SYNCHRONIZATION TEST");
  console.log("========================================");

  try {
    const clock = await alpaca.getClock();
    const localET = getEasternTime();

    logTest(
      "Alpaca Clock Fetch",
      "PASS",
      `Market is ${clock.is_open ? "OPEN" : "CLOSED"}`,
      {
        alpacaTime: clock.timestamp,
        nextOpen: clock.next_open,
        nextClose: clock.next_close,
      }
    );

    // Check if it's a trading day
    const now = new Date();
    const dayOfWeek = now.getDay(); // 0 = Sunday, 6 = Saturday
    if (dayOfWeek === 0 || dayOfWeek === 6) {
      logTest("Trading Day Check", "WARN", "Today is a weekend - no trading");
    } else {
      logTest("Trading Day Check", "PASS", "Today is a weekday");
    }

    // Check time until market open/close
    const nextOpen = new Date(clock.next_open);
    const nextClose = new Date(clock.next_close);
    const minutesUntilOpen = Math.floor((nextOpen.getTime() - now.getTime()) / 60000);
    const minutesUntilClose = Math.floor((nextClose.getTime() - now.getTime()) / 60000);

    if (clock.is_open) {
      logTest(
        "Market Status",
        "PASS",
        `Market is OPEN - closes in ${minutesUntilClose} minutes`,
        { nextClose: clock.next_close }
      );
    } else {
      logTest(
        "Market Status",
        "PASS",
        `Market is CLOSED - opens in ${minutesUntilOpen} minutes`,
        { nextOpen: clock.next_open }
      );
    }

    // Check for early close
    const closeHour = nextClose.getHours();
    if (closeHour < 16) {
      logTest(
        "Early Close Detection",
        "WARN",
        `Early close detected: ${clock.next_close} (normal close is 4:00 PM ET)`
      );
    }

    // Local time check
    logTest(
      "Local Time (ET)",
      "PASS",
      `Current ET: ${localET.displayTime}`,
      { dateET: localET.dateString }
    );
  } catch (error) {
    logTest(
      "Clock Synchronization",
      "FAIL",
      `Failed to fetch clock: ${error instanceof Error ? error.message : error}`
    );
    return false;
  }

  return true;
}

async function testPositions() {
  console.log("\n========================================");
  console.log("3. POSITION CHECK (Day Trading Rule)");
  console.log("========================================");

  try {
    const positions = await alpaca.getPositions();

    if (positions.length === 0) {
      logTest("Overnight Positions", "PASS", "No positions held overnight (day trading compliant)");
    } else {
      logTest(
        "Overnight Positions",
        "FAIL",
        `Found ${positions.length} position(s) held overnight - MUST BE FLAT!`,
        positions.map((p) => ({
          symbol: p.symbol,
          qty: p.qty,
          unrealizedPL: p.unrealized_pl,
        }))
      );
    }
  } catch (error) {
    logTest(
      "Position Check",
      "FAIL",
      `Failed to fetch positions: ${error instanceof Error ? error.message : error}`
    );
    return false;
  }

  return true;
}

async function testDailyState() {
  console.log("\n========================================");
  console.log("4. DAILY STATE CHECK");
  console.log("========================================");

  const state = loadDailyState();
  const currentDate = getEasternTime().dateString;

  if (!state) {
    logTest("Daily State", "PASS", "No state from previous day - clean start");
  } else if (state.date !== currentDate) {
    logTest(
      "Daily State",
      "PASS",
      `State from previous day (${state.date}) - will reset for ${currentDate}`
    );
  } else {
    logTest(
      "Daily State",
      "WARN",
      `State already exists for today (${currentDate})`,
      {
        entries: state.newEntriesToday,
        pnl: state.dailyPnL.toFixed(2),
        resetCompleted: state.resetCompleted,
      }
    );
  }

  // Check daily limits
  const maxEntries = dayTraderConfig.DAY_TRADER_CONFIG.MAX_NEW_ENTRIES_PER_DAY;
  const maxLoss = dayTraderConfig.DAY_TRADER_CONFIG.DAILY_MAX_LOSS;
  const maxProfit = dayTraderConfig.DAY_TRADER_CONFIG.DAILY_MAX_PROFIT;

  logTest(
    "Daily Limits Config",
    "PASS",
    `Max ${maxEntries} entries, Loss limit: $${maxLoss}, Profit limit: $${maxProfit}`
  );
}

async function testConfiguration() {
  console.log("\n========================================");
  console.log("5. CONFIGURATION VALIDATION");
  console.log("========================================");

  // Check environment variables
  const requiredEnvVars = [
    "ALPACA_API_KEY",
    "ALPACA_API_SECRET",
  ];

  const optionalEnvVars = [
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "DRY_RUN",
    "TIME_GUARD_OVERRIDE",
  ];

  let allRequired = true;
  for (const envVar of requiredEnvVars) {
    if (process.env[envVar]) {
      logTest(`Env: ${envVar}`, "PASS", "Configured");
    } else {
      logTest(`Env: ${envVar}`, "FAIL", "Missing required environment variable");
      allRequired = false;
    }
  }

  for (const envVar of optionalEnvVars) {
    if (process.env[envVar]) {
      logTest(
        `Env: ${envVar}`,
        "PASS",
        `Set to: ${envVar.includes("KEY") ? "[REDACTED]" : process.env[envVar]}`
      );
    }
  }

  // Check DRY_RUN mode
  const isDryRun = process.env.DRY_RUN === "1";
  if (isDryRun) {
    logTest("Trading Mode", "WARN", "DRY_RUN enabled - no real orders will be placed");
  } else {
    logTest("Trading Mode", "PASS", "LIVE trading mode - real orders will be placed");
  }

  // Check TIME_GUARD_OVERRIDE
  const timeGuardOverride = process.env.TIME_GUARD_OVERRIDE === "1";
  if (timeGuardOverride) {
    logTest("Time Guard", "WARN", "TIME_GUARD_OVERRIDE enabled - FORT KNOX disabled (testing only)");
  } else {
    logTest("Time Guard", "PASS", "FORT KNOX time guard enabled (safe)");
  }

  return allRequired;
}

async function testSystemResources() {
  console.log("\n========================================");
  console.log("6. SYSTEM RESOURCES");
  console.log("========================================");

  const memoryUsage = process.memoryUsage();
  const memoryMB = Math.round(memoryUsage.heapUsed / 1024 / 1024);

  logTest("Memory Usage", "PASS", `${memoryMB} MB heap used`);

  // Check disk space for reports directory
  try {
    const fs = await import("fs");
    const path = await import("path");

    const reportsDir = path.join(process.cwd(), "reports");
    if (fs.existsSync(reportsDir)) {
      logTest("Reports Directory", "PASS", "Reports directory exists");
    } else {
      logTest("Reports Directory", "WARN", "Reports directory will be created on first run");
    }

    const stateDir = path.join(process.cwd(), "reports", "state");
    if (fs.existsSync(stateDir)) {
      logTest("State Directory", "PASS", "State directory exists");
    } else {
      logTest("State Directory", "WARN", "State directory will be created on first use");
    }
  } catch (error) {
    logTest("Filesystem Check", "WARN", "Could not check filesystem");
  }
}

async function runSmokeTest() {
  console.log("\n");
  console.log("╔════════════════════════════════════════╗");
  console.log("║   ATOBOT PRE-MARKET SMOKE TEST        ║");
  console.log("╔════════════════════════════════════════╗");
  console.log("");

  const startTime = Date.now();

  // Run all tests
  await testAlpacaConnection();
  await testClockSync();
  await testPositions();
  await testDailyState();
  await testConfiguration();
  await testSystemResources();

  const duration = ((Date.now() - startTime) / 1000).toFixed(2);

  // Summary
  console.log("\n========================================");
  console.log("SUMMARY");
  console.log("========================================");

  const passed = results.filter((r) => r.status === "PASS").length;
  const failed = results.filter((r) => r.status === "FAIL").length;
  const warned = results.filter((r) => r.status === "WARN").length;
  const total = results.length;

  console.log(`✅ PASSED: ${passed}/${total}`);
  console.log(`❌ FAILED: ${failed}/${total}`);
  console.log(`⚠️  WARNINGS: ${warned}/${total}`);
  console.log(`⏱️  Duration: ${duration}s`);
  console.log("");

  if (failed > 0) {
    console.log("🔴 SMOKE TEST FAILED - DO NOT START TRADING");
    console.log("   Fix the issues above before starting the bot.");
    process.exit(1);
  } else if (warned > 0) {
    console.log("🟡 SMOKE TEST PASSED WITH WARNINGS");
    console.log("   Review warnings above before starting the bot.");
    process.exit(0);
  } else {
    console.log("🟢 SMOKE TEST PASSED - READY FOR TRADING");
    console.log("   You can safely start the bot with: npm run pm2:start");
    process.exit(0);
  }
}

// Run the smoke test
runSmokeTest().catch((error) => {
  console.error("\n❌ SMOKE TEST CRASHED:");
  console.error(error);
  process.exit(1);
});
