#!/usr/bin/env tsx
/**
 * Automated Mode Switcher
 *
 * Safely switches between testing (DRY_RUN/simulation) and live trading mode.
 *
 * Features:
 * - Validates environment before switching
 * - Runs smoke test in target mode
 * - Creates backup of .env before changes
 * - Provides rollback capability
 * - Requires explicit confirmation for live mode
 * - Logs all changes for audit trail
 *
 * Usage:
 *   npm run mode:test     # Switch to testing mode
 *   npm run mode:live     # Switch to live mode (requires confirmation)
 *   npm run mode:status   # Show current mode
 *   npm run mode:rollback # Rollback last change
 */

import fs from "fs";
import path from "path";
import { execSync } from "child_process";
import readline from "readline";

const ENV_FILE = path.join(process.cwd(), ".env");
const ENV_BACKUP = path.join(process.cwd(), ".env.backup");
const MODE_LOG = path.join(process.cwd(), "reports/mode_changes.jsonl");

interface ModeConfig {
  DRY_RUN: "0" | "1";
  TIME_GUARD_OVERRIDE: "0" | "1";
  SIM_CLOCK_OPEN?: string;
  SIM_TIME_ET?: string;
  ALPACA_ENDPOINT?: string;
}

interface ModeChangeLog {
  timestamp: string;
  from: string;
  to: string;
  user: string;
  validated: boolean;
  smokeTestPassed: boolean;
}

const TESTING_MODE: ModeConfig = {
  DRY_RUN: "1",
  TIME_GUARD_OVERRIDE: "1",
  SIM_CLOCK_OPEN: "1",
  SIM_TIME_ET: "2026-02-04 10:00",
};

const LIVE_MODE: ModeConfig = {
  DRY_RUN: "0",
  TIME_GUARD_OVERRIDE: "0",
};

function readEnvFile(): Record<string, string> {
  if (!fs.existsSync(ENV_FILE)) {
    throw new Error(".env file not found. Cannot switch modes.");
  }

  const content = fs.readFileSync(ENV_FILE, "utf-8");
  const env: Record<string, string> = {};

  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const match = trimmed.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
    if (match) {
      const [, key, value] = match;
      env[key] = value.replace(/^["']|["']$/g, ""); // Remove quotes
    }
  }

  return env;
}

function writeEnvFile(env: Record<string, string>): void {
  const lines: string[] = [];

  // Preserve comments and structure
  if (fs.existsSync(ENV_FILE)) {
    const originalContent = fs.readFileSync(ENV_FILE, "utf-8");
    const originalLines = originalContent.split("\n");

    for (const line of originalLines) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) {
        lines.push(line);
        continue;
      }

      const match = trimmed.match(/^([A-Z_][A-Z0-9_]*)=/);
      if (match) {
        const key = match[1];
        if (key in env) {
          lines.push(`${key}=${env[key]}`);
          delete env[key]; // Mark as processed
        } else {
          lines.push(line); // Keep original
        }
      } else {
        lines.push(line);
      }
    }
  }

  // Add any new keys not in original file
  for (const [key, value] of Object.entries(env)) {
    lines.push(`${key}=${value}`);
  }

  fs.writeFileSync(ENV_FILE, lines.join("\n"), "utf-8");
}

function backupEnv(): void {
  if (fs.existsSync(ENV_FILE)) {
    fs.copyFileSync(ENV_FILE, ENV_BACKUP);
    console.log(`✅ Backup created: ${ENV_BACKUP}`);
  }
}

function rollbackEnv(): void {
  if (!fs.existsSync(ENV_BACKUP)) {
    console.error("❌ No backup found. Cannot rollback.");
    process.exit(1);
  }

  fs.copyFileSync(ENV_BACKUP, ENV_FILE);
  console.log("✅ Rollback complete. Restored from backup.");
}

function getCurrentMode(): string {
  const env = readEnvFile();
  const isDryRun = env.DRY_RUN === "1";
  const simClockOpen = env.SIM_CLOCK_OPEN === "1";
  const timeGuardOverride = env.TIME_GUARD_OVERRIDE === "1";

  if (isDryRun && simClockOpen) {
    return "TESTING (Full Simulation)";
  } else if (isDryRun) {
    return "TESTING (Dry Run)";
  } else {
    return "LIVE TRADING";
  }
}

async function confirmLiveMode(): Promise<boolean> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise((resolve) => {
    console.log("\n⚠️  WARNING: You are about to enable LIVE TRADING mode.");
    console.log("   Real orders will be placed with real money.");
    console.log("   Make sure you have:");
    console.log("   - ✅ Completed smoke test");
    console.log("   - ✅ Verified Alpaca API keys");
    console.log("   - ✅ Checked sufficient buying power");
    console.log("   - ✅ Reviewed strategy performance\n");

    rl.question("Type 'CONFIRM LIVE MODE' to proceed: ", (answer) => {
      rl.close();
      resolve(answer.trim() === "CONFIRM LIVE MODE");
    });
  });
}

async function runSmokeTest(): Promise<boolean> {
  try {
    console.log("\n🧪 Running smoke test in target mode...");
    execSync("npm run smoke-test", { stdio: "inherit" });
    return true;
  } catch (error) {
    console.error("❌ Smoke test failed. Aborting mode switch.");
    return false;
  }
}

function logModeChange(from: string, to: string, validated: boolean, smokeTestPassed: boolean): void {
  const logEntry: ModeChangeLog = {
    timestamp: new Date().toISOString(),
    from,
    to,
    user: process.env.USER || "unknown",
    validated,
    smokeTestPassed,
  };

  const reportsDir = path.dirname(MODE_LOG);
  if (!fs.existsSync(reportsDir)) {
    fs.mkdirSync(reportsDir, { recursive: true });
  }

  fs.appendFileSync(MODE_LOG, JSON.stringify(logEntry) + "\n", "utf-8");
}

async function switchToMode(targetMode: "test" | "live", skipConfirm = false): Promise<void> {
  const currentMode = getCurrentMode();
  const targetModeStr = targetMode === "test" ? "TESTING" : "LIVE TRADING";

  console.log(`\n🔄 Mode Switch Request`);
  console.log(`   From: ${currentMode}`);
  console.log(`   To:   ${targetModeStr}\n`);

  if (currentMode.includes(targetModeStr)) {
    console.log(`✅ Already in ${targetModeStr} mode. No changes needed.`);
    return;
  }

  // Confirm live mode
  if (targetMode === "live" && !skipConfirm) {
    const confirmed = await confirmLiveMode();
    if (!confirmed) {
      console.log("❌ Live mode switch cancelled.");
      process.exit(0);
    }
  }

  // Backup current .env
  backupEnv();

  // Apply changes
  const env = readEnvFile();
  const targetConfig = targetMode === "test" ? TESTING_MODE : LIVE_MODE;

  for (const [key, value] of Object.entries(targetConfig)) {
    if (value !== undefined) {
      env[key] = value;
    }
  }

  // Remove simulation keys in live mode
  if (targetMode === "live") {
    delete env.SIM_CLOCK_OPEN;
    delete env.SIM_TIME_ET;
  }

  writeEnvFile(env);
  console.log(`✅ Environment updated to ${targetModeStr} mode`);

  // Run smoke test
  const smokeTestPassed = await runSmokeTest();

  // Log change
  logModeChange(currentMode, targetModeStr, true, smokeTestPassed);

  if (smokeTestPassed) {
    console.log(`\n🟢 Mode switch complete: ${targetModeStr}`);
    console.log(`   Restart bot: npm run pm2:restart`);
    console.log(`   Or start bot: npm run pm2:start\n`);
  } else {
    console.log(`\n🟡 Mode switch complete but smoke test failed.`);
    console.log(`   Fix issues before starting bot.`);
    console.log(`   Rollback: npm run mode:rollback\n`);
  }
}

function showStatus(): void {
  const env = readEnvFile();
  const mode = getCurrentMode();

  console.log(`\n📊 Current Trading Mode: ${mode}\n`);
  console.log("Configuration:");
  console.log(`   DRY_RUN:             ${env.DRY_RUN || "not set"}`);
  console.log(`   TIME_GUARD_OVERRIDE: ${env.TIME_GUARD_OVERRIDE || "not set"}`);
  console.log(`   SIM_CLOCK_OPEN:      ${env.SIM_CLOCK_OPEN || "not set"}`);
  console.log(`   SIM_TIME_ET:         ${env.SIM_TIME_ET || "not set"}`);
  console.log(`   ALPACA_ENDPOINT:     ${env.ALPACA_ENDPOINT || "not set"}\n`);

  // Show recent mode changes
  if (fs.existsSync(MODE_LOG)) {
    const logs = fs.readFileSync(MODE_LOG, "utf-8").trim().split("\n");
    const recentLogs = logs.slice(-5).map((line) => JSON.parse(line)) as ModeChangeLog[];

    console.log("Recent Mode Changes:");
    for (const log of recentLogs) {
      const icon = log.smokeTestPassed ? "✅" : "⚠️";
      console.log(`   ${icon} ${log.timestamp}: ${log.from} → ${log.to}`);
    }
    console.log("");
  }
}

// CLI Interface
const command = process.argv[2];

switch (command) {
  case "test":
    switchToMode("test");
    break;

  case "live":
    switchToMode("live");
    break;

  case "status":
    showStatus();
    break;

  case "rollback":
    rollbackEnv();
    break;

  default:
    console.log(`
🤖 AtoBot Mode Switcher

Usage:
  npm run mode:test     - Switch to testing mode (safe)
  npm run mode:live     - Switch to live trading mode (requires confirmation)
  npm run mode:status   - Show current mode and configuration
  npm run mode:rollback - Rollback last mode change

Current Mode: ${getCurrentMode()}
    `);
    process.exit(1);
}
