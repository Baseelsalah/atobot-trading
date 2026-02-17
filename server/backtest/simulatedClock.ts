/**
 * Simulated Clock for Backtesting
 *
 * Manages the simulated time for the backtest engine.
 * Sets the timezone module's simTime so that strategies calling
 * getEasternTime() get the correct bar timestamp.
 */

import { setSimTime } from "../timezone";

const TIMEZONE = "America/New_York";

// Reusable formatters (created once for performance)
const hourFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: TIMEZONE,
  hour: "numeric",
  hour12: false,
});

const minuteFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: TIMEZONE,
  minute: "numeric",
});

const dateFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: TIMEZONE,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
});

export class SimulatedClock {
  private currentTime: Date;

  constructor(initialTime?: Date) {
    this.currentTime = initialTime || new Date();
  }

  /** Advance the simulated clock and update the timezone module */
  setTime(timestamp: Date): void {
    this.currentTime = timestamp;
    setSimTime(timestamp);
  }

  /** Set time from a bar timestamp string (ISO format) */
  setTimeFromBar(barTimestamp: string): void {
    this.setTime(new Date(barTimestamp));
  }

  /** Get the current simulated time */
  getTime(): Date {
    return this.currentTime;
  }

  /** Get ET components from the current simulated time */
  getEasternTime(): { hour: number; minute: number; dateString: string } {
    return SimulatedClock.toET(this.currentTime);
  }

  /** Get minutes since midnight in ET */
  getMinutesSinceMidnightET(): number {
    const et = this.getEasternTime();
    return et.hour * 60 + et.minute;
  }

  /** Check if current time is within entry window */
  isWithinEntryWindow(startMinutes: number, endMinutes: number): boolean {
    const minutes = this.getMinutesSinceMidnightET();
    return minutes >= startMinutes && minutes < endMinutes;
  }

  /** Check if current time is past force close */
  isPastForceClose(forceCloseMinutes: number): boolean {
    const minutes = this.getMinutesSinceMidnightET();
    return minutes >= forceCloseMinutes;
  }

  /** Clear simulation time (restore to real clock) */
  reset(): void {
    setSimTime(null);
  }

  // ─── Static utilities ───

  /** Convert a Date to ET components (pure, no global state) */
  static toET(date: Date): { hour: number; minute: number; dateString: string } {
    const hour = parseInt(hourFormatter.format(date), 10);
    const minute = parseInt(minuteFormatter.format(date), 10);
    const dateParts = dateFormatter.format(date).split("/");
    const dateString = `${dateParts[2]}-${dateParts[0].padStart(2, "0")}-${dateParts[1].padStart(2, "0")}`;
    return { hour, minute, dateString };
  }

  /** Get minutes elapsed between two timestamps */
  static minutesBetween(start: Date, end: Date): number {
    return (end.getTime() - start.getTime()) / 60000;
  }

  /** Parse bar timestamp string to Date */
  static parseBarTimestamp(barTimestamp: string): Date {
    return new Date(barTimestamp);
  }

  /** Get ET date string from a bar timestamp */
  static barDateET(barTimestamp: string): string {
    return SimulatedClock.toET(new Date(barTimestamp)).dateString;
  }
}
