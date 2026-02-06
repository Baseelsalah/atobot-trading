/**
 * Persistent Daily State Management
 *
 * Ensures that daily trading state (counters, P/L) persists across bot restarts.
 * Prevents mid-day restarts from resetting trade counts and P/L limits.
 */

import fs from "fs";
import path from "path";
import { getEasternTime } from "./timezone.js";

const STATE_DIR = path.join(process.cwd(), "reports", "state");
const STATE_FILE = path.join(STATE_DIR, "daily_state.json");

export interface DailyState {
  date: string; // YYYY-MM-DD in ET
  newEntriesToday: number;
  dailyPnL: number;
  dailyRealizedPL: number; // From profitManager
  resetCompleted: boolean;
  lastUpdated: string; // ISO timestamp
}

/**
 * Ensure state directory exists
 */
function ensureStateDir(): void {
  if (!fs.existsSync(STATE_DIR)) {
    fs.mkdirSync(STATE_DIR, { recursive: true });
  }
}

/**
 * Get current trading date in Eastern Time (YYYY-MM-DD)
 */
function getTradingDateET(): string {
  return getEasternTime().dateString;
}

/**
 * Load daily state from disk
 * Returns null if file doesn't exist or is from a previous trading day
 */
export function loadDailyState(): DailyState | null {
  ensureStateDir();

  if (!fs.existsSync(STATE_FILE)) {
    return null;
  }

  try {
    const data = fs.readFileSync(STATE_FILE, "utf-8");
    const state: DailyState = JSON.parse(data);

    const currentDate = getTradingDateET();

    // If state is from previous day, return null (new day)
    if (state.date !== currentDate) {
      console.log(
        `[PersistentState] State file is from previous day (${state.date}), starting fresh for ${currentDate}`
      );
      return null;
    }

    console.log(
      `[PersistentState] Loaded state for ${state.date}: ` +
      `${state.newEntriesToday} entries, P/L: $${state.dailyPnL.toFixed(2)}`
    );
    return state;
  } catch (error) {
    console.error("[PersistentState] Error loading state file:", error);
    return null;
  }
}

/**
 * Save daily state to disk
 */
export function saveDailyState(state: Partial<DailyState>): void {
  ensureStateDir();

  const currentDate = getTradingDateET();
  const existingState = loadDailyState() || createEmptyState(currentDate);

  const updatedState: DailyState = {
    ...existingState,
    ...state,
    date: currentDate,
    lastUpdated: new Date().toISOString(),
  };

  try {
    fs.writeFileSync(STATE_FILE, JSON.stringify(updatedState, null, 2), "utf-8");
  } catch (error) {
    console.error("[PersistentState] Error saving state file:", error);
  }
}

/**
 * Create empty state for a new trading day
 */
function createEmptyState(date: string): DailyState {
  return {
    date,
    newEntriesToday: 0,
    dailyPnL: 0,
    dailyRealizedPL: 0,
    resetCompleted: true,
    lastUpdated: new Date().toISOString(),
  };
}

/**
 * Update trade count in persistent state
 */
export function incrementPersistentEntries(): void {
  const state = loadDailyState() || createEmptyState(getTradingDateET());
  saveDailyState({
    newEntriesToday: state.newEntriesToday + 1,
  });
}

/**
 * Update P/L in persistent state
 */
export function updatePersistentPnL(dailyPnL: number, dailyRealizedPL: number): void {
  saveDailyState({
    dailyPnL,
    dailyRealizedPL,
  });
}

/**
 * Mark daily reset as completed
 */
export function markResetCompleted(): void {
  saveDailyState({
    resetCompleted: true,
  });
}

/**
 * Check if reset has been completed today
 * Used to prevent duplicate resets on multiple bot starts
 */
export function isResetCompletedToday(): boolean {
  const state = loadDailyState();
  return state?.resetCompleted ?? false;
}

/**
 * Get persistent state summary for logging
 */
export function getStateSummary(): string {
  const state = loadDailyState();
  if (!state) {
    return `No state loaded (new day or first boot)`;
  }
  return `Date: ${state.date}, Entries: ${state.newEntriesToday}, P/L: $${state.dailyPnL.toFixed(2)}, Realized: $${state.dailyRealizedPL.toFixed(2)}`;
}
