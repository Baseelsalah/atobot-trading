/**
 * Verify Runtime Today - Check if bot was running during market hours
 * 
 * Usage: npx tsx scripts/verify_runtime_today.ts
 * 
 * Prints:
 * - Market status, next open, next close
 * - Ticks today, last tick time
 * - Heartbeat count, last heartbeat time
 * - Path existence of did_run_YYYY-MM-DD.json
 */

import * as alpaca from "../server/alpaca";
import * as activityLedger from "../server/activityLedger";
import * as runtimeMonitor from "../server/runtimeMonitor";
import { getEasternTime } from "../server/timezone";
import * as fs from "fs";

async function main(): Promise<void> {
  console.log("========================================");
  console.log("VERIFY RUNTIME TODAY");
  console.log("========================================\n");

  const et = getEasternTime();
  const ptNow = new Date().toLocaleString("en-US", { 
    timeZone: "America/Los_Angeles", 
    hour12: false,
    year: "numeric",
    month: "2-digit", 
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });

  console.log(`Current Time:`);
  console.log(`  PT: ${ptNow}`);
  console.log(`  ET: ${et.displayTime} (${et.dateString})`);
  console.log("");

  // Market status
  try {
    const clock = await alpaca.getClock();
    console.log(`Market Status:`);
    console.log(`  is_open: ${clock.is_open ? "OPEN" : "CLOSED"}`);
    console.log(`  next_open: ${clock.next_open}`);
    console.log(`  next_close: ${clock.next_close}`);
  } catch (err) {
    console.log(`Market Status: ERROR - Unable to fetch Alpaca clock`);
  }
  console.log("");

  // Activity summary for today
  const summary = activityLedger.getActivitySummary(et.dateString);
  console.log(`Activity Today (${et.dateString}):`);
  console.log(`  botWasRunning: ${summary.botWasRunning}`);
  console.log(`  scanTicks: ${summary.scanTicks}`);
  console.log(`  symbolsEvaluated: ${summary.symbolsEvaluated}`);
  console.log(`  tradesAttempted: ${summary.tradesAttempted}`);
  console.log(`  tradesFilled: ${summary.tradesFilled}`);
  console.log(`  firstTickET: ${summary.firstTickET || "none"}`);
  console.log(`  lastTickET: ${summary.lastTickET || "none"}`);
  console.log("");

  // Did run file check
  const didRunPath = `reports/runtime/did_run_${et.dateString}.json`;
  const didRunExists = fs.existsSync(didRunPath);
  console.log(`Did Run File:`);
  console.log(`  path: ${didRunPath}`);
  console.log(`  exists: ${didRunExists}`);
  
  if (didRunExists) {
    try {
      const data = fs.readFileSync(didRunPath, "utf-8");
      const report = JSON.parse(data);
      console.log(`  heartbeatCount: ${report.heartbeatCount}`);
      console.log(`  uptimeMinutes: ${report.uptimeMinutes}`);
      console.log(`  lastHeartbeatET: ${report.lastHeartbeatET || "none"}`);
      console.log(`  lastTickET: ${report.lastTickET || "none"}`);
    } catch (err) {
      console.log(`  (unable to parse file)`);
    }
  }
  console.log("");

  // Activity ledger file check
  const activityPath = `reports/activity/activity_${et.dateString}.json`;
  const activityExists = fs.existsSync(activityPath);
  console.log(`Activity Ledger File:`);
  console.log(`  path: ${activityPath}`);
  console.log(`  exists: ${activityExists}`);
  
  if (activityExists) {
    try {
      const data = fs.readFileSync(activityPath, "utf-8");
      const ticks = JSON.parse(data);
      console.log(`  tickCount: ${ticks.length}`);
      if (ticks.length > 0) {
        console.log(`  firstTick: ${ticks[0].tsET}`);
        console.log(`  lastTick: ${ticks[ticks.length - 1].tsET}`);
      }
    } catch (err) {
      console.log(`  (unable to parse file)`);
    }
  }
  console.log("");

  // Summary verdict
  console.log("========================================");
  if (summary.botWasRunning && summary.scanTicks > 0) {
    console.log("VERDICT: Bot was running today");
    console.log(`         ${summary.scanTicks} scan ticks recorded`);
  } else {
    console.log("VERDICT: No bot activity recorded today");
    console.log("         (may be outside market hours or bot not started)");
  }
  console.log("========================================");
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
