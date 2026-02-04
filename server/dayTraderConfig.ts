export const DAY_TRADER_CONFIG = {
  ACCOUNT_SIZE: 100_000,
  RISK_PER_TRADE: 0.005,        // 0.5% risk per trade
  DAILY_MAX_LOSS: -500,          // Stop new entries at -$500
  DAILY_MAX_PROFIT: 500,         // Stop new entries at +$500 (paper safety band)
  MAX_OPEN_POSITIONS: 5,
  MAX_NEW_ENTRIES_PER_DAY: 10,
  COOLDOWN_MINUTES: 30,
  PARTIAL_PROFIT_PCT: 0.33,
  TRADE_TIMEOUT_MINUTES: 15,
  SCAN_INTERVAL_MINUTES: 5,
  WAIT_AFTER_OPEN_MINUTES: 5,
  MIN_DOLLAR_VOLUME: 50_000_000,
  MAX_SPREAD_PCT: 0.001,         // Skip if spread > 0.10%
  MARKET_CLOSE_BUFFER_MINUTES: 15,
  
  // FULL UNIVERSE: Ultra-liquid ETFs and mega-cap stocks for expanded opportunity
  ALLOWED_SYMBOLS: [
    // Major Index ETFs
    "SPY", "QQQ", "IWM", "DIA",
    // Bond & Commodity ETFs
    "TLT", "GLD", "SLV",
    // Sector ETFs
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY",
    // Mega-cap stocks
    "AAPL", "MSFT", "NVDA", "AMZN", "TSLA",
    // Bearish hedge
    "SH"
  ],
  
  // BASELINE MODE: Curated ultra-liquid universe for expanded sample collection
  // Set to true to restrict entries to baseline universe only (exits still work for any position)
  BASELINE_MODE: true,
  BASELINE_UNIVERSE: [
    // Major Index ETFs
    "SPY", "QQQ", "IWM", "DIA",
    // Bond & Commodity ETFs
    "TLT", "GLD", "SLV",
    // Sector ETFs
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY",
    // Mega-cap stocks
    "AAPL", "MSFT", "NVDA", "AMZN", "TSLA"
  ],
  
  // BASELINE MAX HOLD: Close positions after this many minutes (only in BASELINE_MODE)
  // Reduces unmatched/open trades for faster learning
  BASELINE_MAX_HOLD_MINUTES: 45,
  
  // BREAKEVEN RULE: When unrealized gain reaches this % of entry, move stop to breakeven
  // Prevents winners from turning into losers (fixes avg loss > avg win leak)
  BREAKEVEN_TRIGGER_PCT: 0.5,    // 0.5% gain triggers breakeven stop
  BREAKEVEN_OFFSET_PCT: 0.05,   // 0.05% buffer above entry to avoid spread/micro-noise
};

let newEntriesToday = 0;
let dailyPnL = 0;
const cooldownMap: Map<string, number> = new Map();

export function getNewEntriesToday(): number {
  return newEntriesToday;
}

export function incrementNewEntries(): void {
  newEntriesToday++;
}

export function canOpenNewEntry(): boolean {
  return newEntriesToday < DAY_TRADER_CONFIG.MAX_NEW_ENTRIES_PER_DAY;
}

export function getDailyPnL(): number {
  return dailyPnL;
}

export function updateDailyPnL(amount: number): void {
  dailyPnL += amount;
}

export function setDailyPnL(amount: number): void {
  dailyPnL = amount;
}

export function isDailyLossLimitHit(): boolean {
  return dailyPnL <= DAY_TRADER_CONFIG.DAILY_MAX_LOSS;
}

export function isDailyProfitLimitHit(): boolean {
  return dailyPnL >= DAY_TRADER_CONFIG.DAILY_MAX_PROFIT;
}

export function isDailyKillThresholdHit(): boolean {
  return isDailyLossLimitHit() || isDailyProfitLimitHit();
}

export function isSymbolAllowed(symbol: string): boolean {
  return DAY_TRADER_CONFIG.ALLOWED_SYMBOLS.includes(symbol.toUpperCase());
}

/**
 * Check if symbol is allowed for ENTRY in current mode
 * In baseline mode, only BASELINE_UNIVERSE symbols can enter
 * Exits are always allowed for any open position
 */
export function isSymbolAllowedForEntry(symbol: string): { allowed: boolean; reason: string } {
  const upperSymbol = symbol.toUpperCase();
  
  // First check full universe
  if (!DAY_TRADER_CONFIG.ALLOWED_SYMBOLS.includes(upperSymbol)) {
    return { allowed: false, reason: "SYMBOL_NOT_IN_UNIVERSE" };
  }
  
  // In baseline mode, further restrict to baseline universe
  if (DAY_TRADER_CONFIG.BASELINE_MODE) {
    if (!DAY_TRADER_CONFIG.BASELINE_UNIVERSE.includes(upperSymbol)) {
      return { allowed: false, reason: "BASELINE_UNIVERSE_RESTRICTED" };
    }
  }
  
  return { allowed: true, reason: "SYMBOL_ALLOWED" };
}

export function getAllowedSymbols(): string[] {
  return [...DAY_TRADER_CONFIG.ALLOWED_SYMBOLS];
}

export function getEntryUniverse(): string[] {
  if (DAY_TRADER_CONFIG.BASELINE_MODE) {
    return [...DAY_TRADER_CONFIG.BASELINE_UNIVERSE];
  }
  return [...DAY_TRADER_CONFIG.ALLOWED_SYMBOLS];
}

export function isBaselineMode(): boolean {
  return DAY_TRADER_CONFIG.BASELINE_MODE;
}

export function setCooldown(symbol: string): void {
  cooldownMap.set(symbol, Date.now() + DAY_TRADER_CONFIG.COOLDOWN_MINUTES * 60 * 1000);
}

export function isOnCooldown(symbol: string): boolean {
  const cooldownUntil = cooldownMap.get(symbol);
  if (!cooldownUntil) return false;
  return Date.now() < cooldownUntil;
}

export function getCooldownRemaining(symbol: string): number {
  const cooldownUntil = cooldownMap.get(symbol);
  if (!cooldownUntil) return 0;
  const remaining = cooldownUntil - Date.now();
  return remaining > 0 ? Math.ceil(remaining / 60000) : 0;
}

export function resetDaily(): void {
  newEntriesToday = 0;
  dailyPnL = 0;
  cooldownMap.clear();
  console.log("[DayTraderConfig] Daily reset complete");
}

export function setNewEntriesToday(count: number): void {
  newEntriesToday = count;
}

export async function rehydrateFromTrades(trades: { side: string; realizedPL: number }[]): Promise<void> {
  // Count today's buy orders as entries
  const todayEntries = trades.filter(t => t.side === "buy").length;
  newEntriesToday = todayEntries;
  
  // Sum up realized P/L from today's trades
  const totalPnL = trades.reduce((sum, t) => sum + (t.realizedPL || 0), 0);
  dailyPnL = totalPnL;
  
  console.log(`[DayTraderConfig] Rehydrated from trades: ${newEntriesToday} entries, P/L: $${dailyPnL.toFixed(2)}`);
}

export function getDayTraderStatus() {
  return {
    newEntriesToday,
    maxEntriesAllowed: DAY_TRADER_CONFIG.MAX_NEW_ENTRIES_PER_DAY,
    entriesRemaining: DAY_TRADER_CONFIG.MAX_NEW_ENTRIES_PER_DAY - newEntriesToday,
    dailyPnL,
    dailyMaxLoss: DAY_TRADER_CONFIG.DAILY_MAX_LOSS,
    dailyMaxProfit: DAY_TRADER_CONFIG.DAILY_MAX_PROFIT,
    lossLimitHit: isDailyLossLimitHit(),
    profitLimitHit: isDailyProfitLimitHit(),
    killThresholdHit: isDailyKillThresholdHit(),
    allowedSymbols: DAY_TRADER_CONFIG.ALLOWED_SYMBOLS,
    cooldownSymbols: Array.from(cooldownMap.entries()).filter(([_, time]) => time > Date.now()).map(([sym, time]) => ({
      symbol: sym,
      minutesRemaining: Math.ceil((time - Date.now()) / 60000)
    })),
  };
}
