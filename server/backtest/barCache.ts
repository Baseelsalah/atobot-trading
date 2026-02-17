/**
 * Bar Cache - Local JSON file caching for historical bar data
 *
 * Caches fetched Alpaca bar data to disk to avoid re-fetching on repeated runs.
 * Cache files stored in data/backtest_cache/ as JSON.
 */

import * as fs from "fs";
import * as path from "path";
import type { AlpacaBar } from "./types";

export class BarCache {
  private cachePath: string;

  constructor(cachePath: string) {
    this.cachePath = cachePath;
    // Ensure cache directory exists
    if (!fs.existsSync(this.cachePath)) {
      fs.mkdirSync(this.cachePath, { recursive: true });
    }
  }

  /** Generate cache filename */
  private getFilename(symbol: string, timeframe: string, startDate: string, endDate: string): string {
    return path.join(this.cachePath, `${symbol}_${timeframe}_${startDate}_${endDate}.json`);
  }

  /** Check if cached data exists */
  has(symbol: string, timeframe: string, startDate: string, endDate: string): boolean {
    return fs.existsSync(this.getFilename(symbol, timeframe, startDate, endDate));
  }

  /** Load cached bars */
  load(symbol: string, timeframe: string, startDate: string, endDate: string): AlpacaBar[] | null {
    const filename = this.getFilename(symbol, timeframe, startDate, endDate);
    if (!fs.existsSync(filename)) return null;

    try {
      const data = fs.readFileSync(filename, "utf-8");
      const bars = JSON.parse(data) as AlpacaBar[];
      return bars;
    } catch {
      return null;
    }
  }

  /** Save bars to cache */
  save(symbol: string, timeframe: string, startDate: string, endDate: string, bars: AlpacaBar[]): void {
    const filename = this.getFilename(symbol, timeframe, startDate, endDate);
    fs.writeFileSync(filename, JSON.stringify(bars), "utf-8");
  }

  /** Clear cache for a symbol or all symbols */
  clear(symbol?: string): void {
    if (symbol) {
      const files = fs.readdirSync(this.cachePath).filter(f => f.startsWith(symbol + "_"));
      for (const file of files) {
        fs.unlinkSync(path.join(this.cachePath, file));
      }
    } else {
      const files = fs.readdirSync(this.cachePath).filter(f => f.endsWith(".json"));
      for (const file of files) {
        fs.unlinkSync(path.join(this.cachePath, file));
      }
    }
  }
}
