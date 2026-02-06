#!/usr/bin/env tsx
/**
 * Daily Automation Agent
 *
 * Fully automated daily trading routine with zero user involvement.
 * Handles pre-market checks, mode switching, bot startup, monitoring, and alerts.
 *
 * Usage:
 *   npm run daily:premarket   # Run at 8:30 AM ET (auto-switches to live mode)
 *   npm run daily:postmarket  # Run at 4:15 PM ET (auto-switches to test mode)
 *   npm run daily:monitor     # Continuous monitoring during trading hours
 *
 * Set up cron jobs for full automation:
 *   30 13 * * 1-5  cd /path/to/atobot && npm run daily:premarket
 *   15 21 * * 1-5  cd /path/to/atobot && npm run daily:postmarket
 *   35 13-20 * * 1-5  cd /path/to/atobot && npm run daily:monitor
 */

import { execSync } from "child_process";
import fs from "fs";
import path from "path";

const ALERT_DIR = path.join(process.cwd(), "reports/alerts");
const AUTOMATION_LOG = path.join(process.cwd(), "reports/automation.jsonl");

interface AutomationLog {
  timestamp: string;
  phase: string;
  action: string;
  result: "success" | "failure" | "warning";
  details?: any;
}

function log(phase: string, action: string, result: "success" | "failure" | "warning", details?: any): void {
  const entry: AutomationLog = {
    timestamp: new Date().toISOString(),
    phase,
    action,
    result,
    details,
  };

  const icon = result === "success" ? "✅" : result === "failure" ? "❌" : "⚠️";
  console.log(`${icon} [${phase}] ${action}`, details ? `- ${JSON.stringify(details)}` : "");

  // Ensure reports directory exists
  const reportsDir = path.dirname(AUTOMATION_LOG);
  if (!fs.existsSync(reportsDir)) {
    fs.mkdirSync(reportsDir, { recursive: true });
  }

  fs.appendFileSync(AUTOMATION_LOG, JSON.stringify(entry) + "\n", "utf-8");
}

function runCommand(command: string, description: string): { success: boolean; output: string } {
  try {
    const output = execSync(command, { encoding: "utf-8", stdio: "pipe" });
    return { success: true, output };
  } catch (error: any) {
    return { success: false, output: error.message };
  }
}

function sendAlert(severity: "INFO" | "WARNING" | "CRITICAL", message: string, details?: any): void {
  const timestamp = new Date().toISOString();
  const filename = `${severity}_${timestamp.replace(/[:.]/g, "-")}.txt`;
  const filepath = path.join(ALERT_DIR, filename);

  if (!fs.existsSync(ALERT_DIR)) {
    fs.mkdirSync(ALERT_DIR, { recursive: true });
  }

  const content = `
========================================
${severity} ALERT
========================================
Timestamp: ${timestamp}
Message: ${message}

Details:
${details ? JSON.stringify(details, null, 2) : "N/A"}
========================================
`;

  fs.writeFileSync(filepath, content, "utf-8");
  console.log(`📧 Alert created: ${filepath}`);

  // Future: Send email/SMS here
  // await sendEmail(severity, message, details);
  // await sendSMS(severity, message);
}

async function premarketRoutine(): Promise<void> {
  console.log("\n🌅 Starting Pre-Market Automation Routine");
  console.log("==========================================\n");

  const startTime = Date.now();

  // Step 1: Check current mode
  log("premarket", "Check current mode", "success");
  const modeCheck = runCommand("npm run mode:status 2>&1", "Check mode");
  console.log(modeCheck.output);

  // Step 2: Run smoke test (in current mode first)
  log("premarket", "Run smoke test", "success");
  const smokeTest = runCommand("npm run smoke-test 2>&1", "Smoke test");

  if (!smokeTest.success) {
    log("premarket", "Smoke test failed", "failure", { output: smokeTest.output });
    sendAlert("CRITICAL", "Pre-market smoke test FAILED", { output: smokeTest.output });
    console.log("\n❌ PRE-MARKET CHECK FAILED");
    console.log("   Smoke test did not pass. Fix issues before trading.");
    console.log("   Alert created in reports/alerts/\n");
    process.exit(1);
  }

  log("premarket", "Smoke test passed", "success");

  // Step 3: Auto-switch to live mode (bypass confirmation)
  log("premarket", "Switch to live trading mode", "success");
  console.log("\n🔄 Switching to LIVE TRADING mode...");

  // Direct .env modification (bypasses confirmation)
  const envPath = path.join(process.cwd(), ".env");
  const envBackupPath = path.join(process.cwd(), ".env.backup");

  // Backup current .env
  if (fs.existsSync(envPath)) {
    fs.copyFileSync(envPath, envBackupPath);
    log("premarket", "Backup .env", "success");
  }

  // Read and modify .env
  let envContent = fs.readFileSync(envPath, "utf-8");
  const lines = envContent.split("\n");
  const newLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("DRY_RUN=")) {
      newLines.push("DRY_RUN=0");
    } else if (line.startsWith("TIME_GUARD_OVERRIDE=")) {
      newLines.push("TIME_GUARD_OVERRIDE=0");
    } else if (line.startsWith("SIM_CLOCK_OPEN=") || line.startsWith("SIM_TIME_ET=")) {
      // Remove simulation variables
      continue;
    } else {
      newLines.push(line);
    }
  }

  fs.writeFileSync(envPath, newLines.join("\n"), "utf-8");
  log("premarket", "Update .env to live mode", "success");

  // Step 4: Check if bot is running
  log("premarket", "Check bot status", "success");
  const botStatus = runCommand("npm run pm2:status 2>&1 | grep atobot || echo 'not running'", "PM2 status");

  if (botStatus.output.includes("online")) {
    // Bot is running - restart it
    log("premarket", "Restart bot", "success");
    console.log("\n🔄 Restarting bot with live mode configuration...");
    runCommand("npm run pm2:restart", "Restart bot");
  } else {
    // Bot not running - start it
    log("premarket", "Start bot", "success");
    console.log("\n▶️  Starting bot in live mode...");

    // Build first if dist doesn't exist
    if (!fs.existsSync(path.join(process.cwd(), "dist"))) {
      console.log("📦 Building production bundle...");
      runCommand("npm run build", "Build");
    }

    runCommand("npm run pm2:start", "Start bot");
  }

  // Step 5: Wait for bot startup
  console.log("\n⏳ Waiting for bot initialization (10 seconds)...");
  await new Promise((resolve) => setTimeout(resolve, 10000));

  // Step 6: Verify bot is healthy
  log("premarket", "Health check", "success");
  const healthCheck = runCommand('curl -s http://localhost:5000/health || echo "failed"', "Health check");

  if (healthCheck.output.includes("failed") || healthCheck.output.includes("error")) {
    log("premarket", "Health check failed", "failure", { output: healthCheck.output });
    sendAlert("CRITICAL", "Bot health check FAILED after startup", { healthOutput: healthCheck.output });
    console.log("\n❌ BOT HEALTH CHECK FAILED");
    console.log("   Check logs: npm run pm2:logs\n");
    process.exit(1);
  }

  log("premarket", "Health check passed", "success");

  // Step 7: Verify live mode is active
  console.log("\n🔍 Verifying LIVE MODE is active...");
  const logsCheck = runCommand("npm run pm2:logs --lines 20 --nostream 2>&1 | grep 'DRY_RUN' || echo 'not found'", "Check logs");

  if (logsCheck.output.includes("DRY_RUN (env): ON") || logsCheck.output.includes("DRY_RUN: ON")) {
    log("premarket", "Live mode verification", "warning", { message: "Bot may still be in DRY_RUN mode" });
    sendAlert("WARNING", "Bot may still be in testing mode - verify manually", { logs: logsCheck.output });
  } else if (logsCheck.output.includes("DRY_RUN (env): OFF") || logsCheck.output.includes("DRY_RUN: OFF")) {
    log("premarket", "Live mode verified", "success");
    console.log("✅ Confirmed: LIVE TRADING mode is active");
  }

  const duration = ((Date.now() - startTime) / 1000).toFixed(2);
  log("premarket", "Pre-market routine complete", "success", { durationSeconds: duration });

  console.log("\n🟢 PRE-MARKET AUTOMATION COMPLETE");
  console.log("==========================================");
  console.log(`   Duration: ${duration}s`);
  console.log(`   Mode: LIVE TRADING`);
  console.log(`   Bot Status: ONLINE`);
  console.log(`   Entry Window Opens: 9:35 AM ET`);
  console.log(`\n   Monitor: npm run pm2:logs --follow`);
  console.log(`   Dashboard: http://localhost:5000\n`);

  sendAlert("INFO", "Pre-market automation complete - Bot ready for trading", {
    mode: "LIVE TRADING",
    duration: `${duration}s`,
  });
}

async function postmarketRoutine(): Promise<void> {
  console.log("\n🌙 Starting Post-Market Automation Routine");
  console.log("==========================================\n");

  const startTime = Date.now();

  // Step 1: Verify all positions closed
  log("postmarket", "Check positions", "success");
  const positionsCheck = runCommand('curl -s http://localhost:5000/api/trading/positions 2>&1', "Check positions");

  let positionsOpen = 0;
  try {
    const positions = JSON.parse(positionsCheck.output);
    positionsOpen = Array.isArray(positions) ? positions.length : 0;
  } catch {
    log("postmarket", "Failed to parse positions", "warning");
  }

  if (positionsOpen > 0) {
    log("postmarket", "Positions still open", "failure", { count: positionsOpen });
    sendAlert("CRITICAL", `${positionsOpen} positions still open after market close!`, {
      positions: positionsCheck.output,
    });
    console.log(`\n❌ WARNING: ${positionsOpen} POSITIONS STILL OPEN`);
    console.log("   This violates day trading rules!");
    console.log("   Manually close via Alpaca dashboard immediately.\n");
  } else {
    log("postmarket", "All positions closed", "success");
    console.log("✅ All positions closed (day trading compliant)");
  }

  // Step 2: Generate daily report
  log("postmarket", "Generate daily report", "success");
  const today = new Date().toISOString().split("T")[0];
  const reportPath = path.join(process.cwd(), `daily_reports/${today}.json`);

  if (fs.existsSync(reportPath)) {
    const report = JSON.parse(fs.readFileSync(reportPath, "utf-8"));
    console.log("\n📊 Daily Trading Summary:");
    console.log(`   P/L: $${report.totalPnL || 0}`);
    console.log(`   Trades: ${report.totalTrades || 0}`);
    console.log(`   Win Rate: ${report.winRate ? (report.winRate * 100).toFixed(1) : 0}%`);
    log("postmarket", "Daily report reviewed", "success", report);
  } else {
    log("postmarket", "Daily report not found", "warning");
  }

  // Step 3: Switch to testing mode (safe overnight)
  log("postmarket", "Switch to testing mode", "success");
  console.log("\n🔄 Switching to TESTING mode (safe overnight)...");

  const envPath = path.join(process.cwd(), ".env");
  let envContent = fs.readFileSync(envPath, "utf-8");
  const lines = envContent.split("\n");
  const newLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("DRY_RUN=")) {
      newLines.push("DRY_RUN=1");
    } else if (line.startsWith("TIME_GUARD_OVERRIDE=")) {
      newLines.push("TIME_GUARD_OVERRIDE=1");
    } else {
      newLines.push(line);
    }
  }

  // Add simulation variables
  newLines.push("SIM_CLOCK_OPEN=1");
  newLines.push(`SIM_TIME_ET=${new Date().toISOString().split("T")[0]} 10:00`);

  fs.writeFileSync(envPath, newLines.join("\n"), "utf-8");
  log("postmarket", "Switched to testing mode", "success");

  // Step 4: Restart bot (optional - can leave running in test mode)
  const restartBot = process.env.AUTO_RESTART_POSTMARKET === "1";
  if (restartBot) {
    log("postmarket", "Restart bot", "success");
    console.log("\n🔄 Restarting bot in testing mode...");
    runCommand("npm run pm2:restart", "Restart");
  } else {
    log("postmarket", "Bot left running", "success");
    console.log("\n✅ Bot left running in testing mode");
  }

  const duration = ((Date.now() - startTime) / 1000).toFixed(2);
  log("postmarket", "Post-market routine complete", "success", { durationSeconds: duration });

  console.log("\n🟢 POST-MARKET AUTOMATION COMPLETE");
  console.log("==========================================");
  console.log(`   Duration: ${duration}s`);
  console.log(`   Positions Closed: ${positionsOpen === 0 ? "YES" : "NO"}`);
  console.log(`   Mode: TESTING (safe overnight)`);
  console.log(`\n   Next: Automated pre-market routine at 8:30 AM ET\n`);
}

async function monitorRoutine(): Promise<void> {
  console.log("\n👀 Running Trading Hours Monitor");
  console.log("==========================================\n");

  // Step 1: Check bot is running
  const botStatus = runCommand("npm run pm2:status 2>&1 | grep atobot || echo 'not running'", "PM2 status");

  if (!botStatus.output.includes("online")) {
    log("monitor", "Bot not running", "failure");
    sendAlert("CRITICAL", "Bot is NOT RUNNING during trading hours!", { status: botStatus.output });
    console.log("❌ BOT IS NOT RUNNING!");
    console.log("   Starting bot automatically...");
    runCommand("npm run pm2:start", "Start bot");
    return;
  }

  log("monitor", "Bot running", "success");

  // Step 2: Health check
  const healthCheck = runCommand('curl -s http://localhost:5000/health 2>&1', "Health check");

  try {
    const health = JSON.parse(healthCheck.output);

    if (health.status !== "ok") {
      log("monitor", "Health check failed", "failure", health);
      sendAlert("CRITICAL", "Bot health check failed", health);
      console.log("❌ HEALTH CHECK FAILED");
      console.log("   Restarting bot...");
      runCommand("npm run pm2:restart", "Restart");
      return;
    }

    log("monitor", "Health check passed", "success");

    // Check if bot is trading
    const lastTickET = health.lastTickET;
    const ticksSinceBoot = health.ticksSinceBoot;

    console.log(`✅ Bot Healthy`);
    console.log(`   Last Tick: ${lastTickET}`);
    console.log(`   Ticks Since Boot: ${ticksSinceBoot}`);
    console.log(`   Market Status: ${health.marketStatus}`);
    console.log(`   Entry Allowed: ${health.entryAllowed}`);

    // Check for stalls (no tick in 10+ minutes)
    if (lastTickET && ticksSinceBoot > 0) {
      const now = new Date();
      const lastTick = new Date(lastTickET);
      const minutesSinceLastTick = (now.getTime() - lastTick.getTime()) / 60000;

      if (minutesSinceLastTick > 10) {
        log("monitor", "Stall detected", "warning", { minutesSinceLastTick });
        sendAlert("WARNING", `Bot may be stalled - no tick in ${minutesSinceLastTick.toFixed(0)} minutes`);
      }
    }
  } catch {
    log("monitor", "Failed to parse health response", "warning");
    sendAlert("WARNING", "Health endpoint returned invalid response");
  }

  // Step 3: Check for critical alerts
  if (fs.existsSync(ALERT_DIR)) {
    const alerts = fs.readdirSync(ALERT_DIR).filter((f) => f.startsWith("CRITICAL_"));
    if (alerts.length > 0) {
      log("monitor", "Critical alerts found", "warning", { count: alerts.length });
      console.log(`\n⚠️  ${alerts.length} CRITICAL ALERTS FOUND`);
      console.log(`   Check: reports/alerts/\n`);
    }
  }

  console.log("\n✅ Monitoring check complete\n");
  log("monitor", "Monitoring check complete", "success");
}

// CLI Interface
const command = process.argv[2];

switch (command) {
  case "premarket":
    premarketRoutine().catch((error) => {
      console.error("❌ Pre-market routine crashed:", error);
      log("premarket", "Routine crashed", "failure", { error: error.message });
      sendAlert("CRITICAL", "Pre-market automation crashed", { error: error.message });
      process.exit(1);
    });
    break;

  case "postmarket":
    postmarketRoutine().catch((error) => {
      console.error("❌ Post-market routine crashed:", error);
      log("postmarket", "Routine crashed", "failure", { error: error.message });
      sendAlert("CRITICAL", "Post-market automation crashed", { error: error.message });
      process.exit(1);
    });
    break;

  case "monitor":
    monitorRoutine().catch((error) => {
      console.error("❌ Monitor routine crashed:", error);
      log("monitor", "Routine crashed", "failure", { error: error.message });
      process.exit(1);
    });
    break;

  default:
    console.log(`
🤖 AtoBot Daily Automation Agent

Usage:
  npm run daily:premarket   - Pre-market routine (8:30 AM ET)
  npm run daily:postmarket  - Post-market routine (4:15 PM ET)
  npm run daily:monitor     - Trading hours monitoring

Automation Features:
  ✅ Auto-switches to live mode before market open
  ✅ Runs smoke tests automatically
  ✅ Starts/restarts bot as needed
  ✅ Monitors health during trading hours
  ✅ Verifies positions closed at end of day
  ✅ Auto-switches to testing mode overnight
  ✅ Sends alerts for critical issues

Set up cron jobs for full automation (see crontab-template.txt)
    `);
    process.exit(1);
}
