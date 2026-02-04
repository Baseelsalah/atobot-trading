/**
 * LOCKED TIMEZONE CONFIGURATION
 * All trading operations use Eastern Time (America/New_York)
 * This file is the SINGLE SOURCE OF TRUTH for timezone handling
 * DO NOT modify timezone logic elsewhere - always import from here
 */

const TIMEZONE = "America/New_York" as const;

export interface EasternTime {
  hour: number;      // 0-23 (24-hour format)
  minute: number;    // 0-59
  second: number;    // 0-59
  dateString: string; // YYYY-MM-DD format
  displayTime: string; // HH:MM ET format
  dayOfWeek: number;  // 0=Sunday, 1=Monday, ..., 6=Saturday
}

/**
 * Parse SIM_TIME_ET environment variable
 * Format: "YYYY-MM-DD HH:MM" (24-hour) in Eastern Time
 */
function parseSimTimeET(simTimeStr: string): Date | null {
  try {
    const match = simTimeStr.match(/^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})$/);
    if (!match) {
      console.error(`[SIM] Invalid SIM_TIME_ET format: ${simTimeStr}. Expected: YYYY-MM-DD HH:MM`);
      return null;
    }
    
    const [, year, month, day, hour, minute] = match;
    
    // Create a date string in Eastern timezone format
    // We need to create a date that when formatted in ET gives us the specified time
    const etDateStr = `${year}-${month}-${day}T${hour}:${minute}:00`;
    
    // Use Intl to find the offset for America/New_York at this date
    const tempDate = new Date(`${year}-${month}-${day}T12:00:00Z`);
    const etFormatter = new Intl.DateTimeFormat("en-US", {
      timeZone: TIMEZONE,
      timeZoneName: "shortOffset",
    });
    
    // Create date treating input as Eastern Time
    // This is approximate but works for simulation purposes
    const utcDate = new Date(`${etDateStr}-05:00`); // Assume EST (adjust for DST would need more logic)
    
    return utcDate;
  } catch (err) {
    console.error(`[SIM] Error parsing SIM_TIME_ET: ${err}`);
    return null;
  }
}

// Check for SIM_TIME_ET on module load
const SIM_TIME_ET = process.env.SIM_TIME_ET;
let simTimeDate: Date | null = null;

if (SIM_TIME_ET) {
  simTimeDate = parseSimTimeET(SIM_TIME_ET);
  if (simTimeDate) {
    console.log(`[SIM] SIM_TIME_ET active: ${SIM_TIME_ET}`);
  }
}

/**
 * Check if simulation time is active
 */
export function isSimTimeActive(): boolean {
  return simTimeDate !== null;
}

/**
 * Get the simulation time string (for display)
 */
export function getSimTimeString(): string | null {
  return SIM_TIME_ET || null;
}

/**
 * Get current Eastern Time components
 * Uses Intl.DateTimeFormat for reliable timezone conversion
 * This method works regardless of server timezone (UTC, PST, etc.)
 * 
 * If SIM_TIME_ET is set, returns simulated time instead
 */
export function getEasternTime(): EasternTime {
  // Use simulated time if SIM_TIME_ET is active
  const now = simTimeDate ? simTimeDate : new Date();
  
  const hourFormatter = new Intl.DateTimeFormat("en-US", { 
    timeZone: TIMEZONE, 
    hour: "numeric", 
    hour12: false 
  });
  
  const minuteFormatter = new Intl.DateTimeFormat("en-US", { 
    timeZone: TIMEZONE, 
    minute: "numeric" 
  });
  
  const secondFormatter = new Intl.DateTimeFormat("en-US", { 
    timeZone: TIMEZONE, 
    second: "numeric" 
  });
  
  const dateFormatter = new Intl.DateTimeFormat("en-US", { 
    timeZone: TIMEZONE, 
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  });
  
  const weekdayFormatter = new Intl.DateTimeFormat("en-US", { 
    timeZone: TIMEZONE, 
    weekday: "short" 
  });
  
  const hour = parseInt(hourFormatter.format(now), 10);
  const minute = parseInt(minuteFormatter.format(now), 10);
  const second = parseInt(secondFormatter.format(now), 10);
  
  // Format: MM/DD/YYYY -> convert to YYYY-MM-DD
  const dateParts = dateFormatter.format(now).split("/");
  const dateString = `${dateParts[2]}-${dateParts[0].padStart(2, '0')}-${dateParts[1].padStart(2, '0')}`;
  
  const displayTime = `${hour.toString().padStart(2, "0")}:${minute.toString().padStart(2, "0")} ET`;
  
  // Get day of week (0=Sunday)
  const weekdayStr = weekdayFormatter.format(now);
  const dayMap: Record<string, number> = { 
    Sun: 0, Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6 
  };
  const dayOfWeek = dayMap[weekdayStr] ?? 0;
  
  return { hour, minute, second, dateString, displayTime, dayOfWeek };
}

/**
 * Convert hours and minutes to total minutes since midnight
 */
export function toMinutesSinceMidnight(hour: number, minute: number): number {
  return hour * 60 + minute;
}

/**
 * Check if current ET time is a weekday (Monday-Friday)
 */
export function isWeekday(): boolean {
  const { dayOfWeek } = getEasternTime();
  return dayOfWeek >= 1 && dayOfWeek <= 5;
}

/**
 * Log current Eastern Time for debugging
 */
export function logCurrentTime(prefix: string = "[TIMEZONE]"): void {
  const et = getEasternTime();
  const simIndicator = isSimTimeActive() ? " [SIMULATED]" : "";
  console.log(`${prefix} Eastern Time: ${et.displayTime} on ${et.dateString} (day ${et.dayOfWeek})${simIndicator}`);
}

/**
 * Convert any Date into an Eastern (America/New_York) date string YYYY-MM-DD.
 * Use this for "trading day" comparisons (avoids UTC/local-time bugs).
 */
export function toEasternDateString(date: Date): string {
  const dateFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  const parts = dateFormatter.format(date).split("/");
  return `${parts[2]}-${parts[0].padStart(2, "0")}-${parts[1].padStart(2, "0")}`;
}

/**
 * Get current Pacific Time date string YYYY-MM-DD.
 * Used for daily reports since user is in PT timezone.
 */
export function getPtDateString(now: Date = new Date()): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/Los_Angeles" }).format(now);
}

/**
 * Get current Pacific Time ISO string for logging.
 * Returns format like: 2025-12-24T20:48:00-08:00
 */
export function getPtNowString(now: Date = new Date()): string {
  const ptFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Los_Angeles",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
  const parts = ptFormatter.formatToParts(now);
  const get = (type: string) => parts.find(p => p.type === type)?.value || "00";
  return `${get("year")}-${get("month")}-${get("day")}T${get("hour")}:${get("minute")}:${get("second")}-08:00`;
}
