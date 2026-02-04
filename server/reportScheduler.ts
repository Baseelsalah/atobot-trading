import * as fs from "fs";
import * as path from "path";
import { getPtDateString, getPtNowString } from "./timezone";
import { generateReport } from "./performanceReport";

const REPORT_CHECK_INTERVAL_MS = 30 * 60 * 1000;

let intervalHandle: NodeJS.Timeout | null = null;

export function dailyReportExists(dateStr: string): boolean {
  const dailyDir = path.join(process.cwd(), "daily_reports");
  const filePath = path.join(dailyDir, `${dateStr}.json`);
  return fs.existsSync(filePath);
}

async function ensureDailyReport(): Promise<void> {
  const todayPt = getPtDateString();
  
  if (dailyReportExists(todayPt)) {
    console.log(`[REPORT] skipped reportDate=${todayPt} reason=already_exists`);
    return;
  }
  
  try {
    console.log(`[REPORT] Generating auto daily report for ${todayPt}...`);
    await generateReport();
    console.log(`[REPORT] generated reportDate=${todayPt} reason=auto_daily`);
  } catch (error) {
    console.error(`[REPORT] Failed to auto-generate report:`, error);
  }
}

export function startReportScheduler(): void {
  console.log(`[REPORT] Starting report scheduler (check interval: ${REPORT_CHECK_INTERVAL_MS / 60000} min)`);
  
  ensureDailyReport();
  
  intervalHandle = setInterval(() => {
    ensureDailyReport();
  }, REPORT_CHECK_INTERVAL_MS);
}

export function stopReportScheduler(): void {
  if (intervalHandle) {
    clearInterval(intervalHandle);
    intervalHandle = null;
    console.log("[REPORT] Report scheduler stopped");
  }
}
