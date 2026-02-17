/**
 * Data Fetcher - Fetch historical bar data from Alpaca REST API
 *
 * Handles pagination, rate limiting, and caching.
 * Uses IEX feed (Alpaca free tier).
 */

import { BarCache } from "./barCache";
import type { AlpacaBar, SymbolBarData } from "./types";

const ALPACA_DATA_URL = "https://data.alpaca.markets";
const MAX_BARS_PER_REQUEST = 10000;
const RATE_LIMIT_PER_MINUTE = 150; // Leave headroom below Alpaca's 200/min

class RateLimiter {
  private requests: number[] = [];
  private maxRequests: number;
  private windowMs: number;

  constructor(maxRequests: number, windowMs: number) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
  }

  async waitForSlot(): Promise<void> {
    const now = Date.now();
    // Remove expired entries
    this.requests = this.requests.filter(t => now - t < this.windowMs);

    if (this.requests.length >= this.maxRequests) {
      const oldest = this.requests[0];
      const waitMs = this.windowMs - (now - oldest) + 50; // 50ms buffer
      await new Promise(resolve => setTimeout(resolve, waitMs));
    }

    this.requests.push(Date.now());
  }
}

export class DataFetcher {
  private apiKey: string;
  private secretKey: string;
  private rateLimiter: RateLimiter;

  constructor(apiKey: string, secretKey: string) {
    this.apiKey = apiKey;
    this.secretKey = secretKey;
    this.rateLimiter = new RateLimiter(RATE_LIMIT_PER_MINUTE, 60000);
  }

  /**
   * Fetch bars for a symbol across a date range with pagination.
   */
  async fetchBars(
    symbol: string,
    timeframe: "1Min" | "5Min",
    startDate: string,
    endDate: string
  ): Promise<AlpacaBar[]> {
    const allBars: AlpacaBar[] = [];
    let pageToken: string | undefined;

    // Convert dates to RFC3339 timestamps
    const startRfc = `${startDate}T00:00:00Z`;
    const endRfc = `${endDate}T23:59:59Z`;

    do {
      await this.rateLimiter.waitForSlot();

      const params = new URLSearchParams({
        timeframe,
        start: startRfc,
        end: endRfc,
        limit: String(MAX_BARS_PER_REQUEST),
        feed: "iex",
        sort: "asc",
      });

      if (pageToken) {
        params.set("page_token", pageToken);
      }

      const url = `${ALPACA_DATA_URL}/v2/stocks/${symbol}/bars?${params.toString()}`;

      const response = await this.fetchWithRetry(url, 3);
      if (!response) {
        console.error(`[DataFetcher] Failed to fetch bars for ${symbol} (${timeframe})`);
        break;
      }

      const data = await response.json() as {
        bars?: AlpacaBar[];
        next_page_token?: string;
      };

      if (data.bars && data.bars.length > 0) {
        allBars.push(...data.bars);
      }

      pageToken = data.next_page_token;
    } while (pageToken);

    return allBars;
  }

  /**
   * Fetch trading calendar from Alpaca.
   */
  async fetchTradingCalendar(
    startDate: string,
    endDate: string
  ): Promise<{ date: string; open: string; close: string }[]> {
    await this.rateLimiter.waitForSlot();

    const params = new URLSearchParams({ start: startDate, end: endDate });
    const url = `https://api.alpaca.markets/v2/calendar?${params.toString()}`;

    const response = await this.fetchWithRetry(url, 3);
    if (!response) {
      throw new Error("Failed to fetch trading calendar");
    }

    return response.json() as Promise<{ date: string; open: string; close: string }[]>;
  }

  /**
   * Fetch all data needed for a backtest.
   * Checks cache first, fetches missing data.
   */
  async fetchAllData(
    symbols: string[],
    startDate: string,
    endDate: string,
    cachePath: string,
    noCache: boolean = false
  ): Promise<Map<string, SymbolBarData>> {
    const cache = new BarCache(cachePath);
    const result = new Map<string, SymbolBarData>();

    // SPY must always be included for regime filter
    const allSymbols = symbols.includes("SPY") ? symbols : ["SPY", ...symbols];

    const total = allSymbols.length * 2; // 2 timeframes per symbol
    let completed = 0;

    for (const symbol of allSymbols) {
      let bars5min: AlpacaBar[];
      let bars1min: AlpacaBar[];

      // Fetch 5-min bars
      if (!noCache && cache.has(symbol, "5Min", startDate, endDate)) {
        bars5min = cache.load(symbol, "5Min", startDate, endDate) || [];
        completed++;
        this.logProgress(completed, total, `${symbol} 5Min (cached)`);
      } else {
        bars5min = await this.fetchBars(symbol, "5Min", startDate, endDate);
        cache.save(symbol, "5Min", startDate, endDate, bars5min);
        completed++;
        this.logProgress(completed, total, `${symbol} 5Min (${bars5min.length} bars)`);
      }

      // Fetch 1-min bars
      if (!noCache && cache.has(symbol, "1Min", startDate, endDate)) {
        bars1min = cache.load(symbol, "1Min", startDate, endDate) || [];
        completed++;
        this.logProgress(completed, total, `${symbol} 1Min (cached)`);
      } else {
        bars1min = await this.fetchBars(symbol, "1Min", startDate, endDate);
        cache.save(symbol, "1Min", startDate, endDate, bars1min);
        completed++;
        this.logProgress(completed, total, `${symbol} 1Min (${bars1min.length} bars)`);
      }

      // Filter invalid bars
      bars5min = bars5min.filter(b => Number.isFinite(b.c) && b.c > 0);
      bars1min = bars1min.filter(b => Number.isFinite(b.c) && b.c > 0);

      result.set(symbol, {
        symbol,
        bars5min,
        bars1min,
        fetchedAt: new Date().toISOString(),
      });
    }

    return result;
  }

  private logProgress(completed: number, total: number, detail: string): void {
    const pct = Math.round((completed / total) * 100);
    console.log(`  [${pct}%] ${detail}`);
  }

  private async fetchWithRetry(url: string, retries: number): Promise<Response | null> {
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const response = await fetch(url, {
          headers: {
            "APCA-API-KEY-ID": this.apiKey,
            "APCA-API-SECRET-KEY": this.secretKey,
          },
        });

        if (response.status === 429) {
          // Rate limited - wait and retry
          const waitMs = Math.pow(2, attempt) * 2000;
          console.warn(`[DataFetcher] Rate limited, waiting ${waitMs}ms...`);
          await new Promise(resolve => setTimeout(resolve, waitMs));
          continue;
        }

        if (!response.ok) {
          if (attempt < retries) {
            const waitMs = Math.pow(2, attempt) * 1000;
            await new Promise(resolve => setTimeout(resolve, waitMs));
            continue;
          }
          console.error(`[DataFetcher] HTTP ${response.status}: ${await response.text()}`);
          return null;
        }

        return response;
      } catch (error) {
        if (attempt < retries) {
          const waitMs = Math.pow(2, attempt) * 1000;
          await new Promise(resolve => setTimeout(resolve, waitMs));
          continue;
        }
        console.error(`[DataFetcher] Fetch error: ${error}`);
        return null;
      }
    }
    return null;
  }
}
