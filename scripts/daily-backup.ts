#!/usr/bin/env tsx
/**
 * Daily Backup Agent
 *
 * Backs up all critical trading data daily.
 * Runs automatically at 11:55 PM every night.
 *
 * Usage: npm run daily:backup
 */

import { execSync } from "child_process";
import fs from "fs";
import path from "path";

const BACKUP_DIR = path.join(process.cwd(), "backups");
const REPORTS_DIR = path.join(process.cwd(), "reports");
const DAILY_REPORTS_DIR = path.join(process.cwd(), "daily_reports");
const WEEKLY_REPORTS_DIR = path.join(process.cwd(), "weekly_reports");

function ensureBackupDir(): void {
  if (!fs.existsSync(BACKUP_DIR)) {
    fs.mkdirSync(BACKUP_DIR, { recursive: true });
  }
}

function getTimestamp(): string {
  return new Date().toISOString().split("T")[0];
}

function backupReports(): string {
  const timestamp = getTimestamp();
  const filename = `reports_${timestamp}.tar.gz`;
  const filepath = path.join(BACKUP_DIR, filename);

  console.log(`📦 Backing up reports directory...`);

  if (fs.existsSync(REPORTS_DIR)) {
    execSync(`tar -czf "${filepath}" -C "${process.cwd()}" reports/`, {
      stdio: "pipe",
    });
    const size = fs.statSync(filepath).size;
    console.log(`✅ Reports backed up: ${filename} (${(size / 1024).toFixed(2)} KB)`);
    return filepath;
  } else {
    console.log(`⚠️ Reports directory not found - skipping`);
    return "";
  }
}

function backupDailyReports(): string {
  const timestamp = getTimestamp();
  const filename = `daily_reports_${timestamp}.tar.gz`;
  const filepath = path.join(BACKUP_DIR, filename);

  console.log(`📦 Backing up daily reports...`);

  if (fs.existsSync(DAILY_REPORTS_DIR)) {
    execSync(`tar -czf "${filepath}" -C "${process.cwd()}" daily_reports/`, {
      stdio: "pipe",
    });
    const size = fs.statSync(filepath).size;
    console.log(`✅ Daily reports backed up: ${filename} (${(size / 1024).toFixed(2)} KB)`);
    return filepath;
  } else {
    console.log(`⚠️ Daily reports directory not found - skipping`);
    return "";
  }
}

function backupWeeklyReports(): string {
  const timestamp = getTimestamp();
  const filename = `weekly_reports_${timestamp}.tar.gz`;
  const filepath = path.join(BACKUP_DIR, filename);

  console.log(`📦 Backing up weekly reports...`);

  if (fs.existsSync(WEEKLY_REPORTS_DIR)) {
    execSync(`tar -czf "${filepath}" -C "${process.cwd()}" weekly_reports/`, {
      stdio: "pipe",
    });
    const size = fs.statSync(filepath).size;
    console.log(`✅ Weekly reports backed up: ${filename} (${(size / 1024).toFixed(2)} KB)`);
    return filepath;
  } else {
    console.log(`⚠️ Weekly reports directory not found - skipping`);
    return "";
  }
}

function backupEnv(): string {
  const timestamp = getTimestamp();
  const filename = `.env.backup_${timestamp}`;
  const filepath = path.join(BACKUP_DIR, filename);

  console.log(`📦 Backing up .env configuration...`);

  const envPath = path.join(process.cwd(), ".env");
  if (fs.existsSync(envPath)) {
    fs.copyFileSync(envPath, filepath);
    console.log(`✅ .env backed up: ${filename}`);
    return filepath;
  } else {
    console.log(`⚠️ .env file not found - skipping`);
    return "";
  }
}

function cleanOldBackups(days: number = 30): void {
  console.log(`\n🧹 Cleaning backups older than ${days} days...`);

  if (!fs.existsSync(BACKUP_DIR)) {
    return;
  }

  const files = fs.readdirSync(BACKUP_DIR);
  const now = Date.now();
  const maxAge = days * 24 * 60 * 60 * 1000;

  let deletedCount = 0;
  let deletedSize = 0;

  for (const file of files) {
    const filepath = path.join(BACKUP_DIR, file);
    const stats = fs.statSync(filepath);
    const age = now - stats.mtimeMs;

    if (age > maxAge) {
      deletedSize += stats.size;
      fs.unlinkSync(filepath);
      deletedCount++;
      console.log(`   Deleted: ${file}`);
    }
  }

  if (deletedCount > 0) {
    console.log(`✅ Cleaned ${deletedCount} old backups (${(deletedSize / 1024 / 1024).toFixed(2)} MB freed)`);
  } else {
    console.log(`✅ No old backups to clean`);
  }
}

function createBackupSummary(backups: string[]): void {
  const timestamp = getTimestamp();
  const summaryPath = path.join(BACKUP_DIR, `backup_summary_${timestamp}.txt`);

  const lines: string[] = [];
  lines.push("========================================");
  lines.push("BACKUP SUMMARY");
  lines.push("========================================");
  lines.push("");
  lines.push(`Date: ${new Date().toISOString()}`);
  lines.push(`Backup Directory: ${BACKUP_DIR}`);
  lines.push("");
  lines.push("Files Backed Up:");
  lines.push("----------------");

  for (const backup of backups) {
    if (backup) {
      const stats = fs.statSync(backup);
      const size = (stats.size / 1024).toFixed(2);
      lines.push(`✅ ${path.basename(backup)} (${size} KB)`);
    }
  }

  lines.push("");
  lines.push("========================================");

  fs.writeFileSync(summaryPath, lines.join("\n"), "utf-8");
  console.log(`\n📄 Backup summary: ${summaryPath}`);
}

async function main(): Promise<void> {
  console.log("\n💾 Starting Daily Backup");
  console.log("========================================\n");

  ensureBackupDir();

  const backups: string[] = [];

  // Perform backups
  backups.push(backupReports());
  backups.push(backupDailyReports());
  backups.push(backupWeeklyReports());
  backups.push(backupEnv());

  // Create summary
  createBackupSummary(backups.filter((b) => b !== ""));

  // Clean old backups
  cleanOldBackups(30);

  console.log("\n✅ Daily backup complete");
  console.log(`📁 Backups saved to: ${BACKUP_DIR}\n`);
}

main().catch((error) => {
  console.error("❌ Backup failed:", error);
  process.exit(1);
});
