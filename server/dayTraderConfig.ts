import {
  loadDailyState,
  saveDailyState,
  incrementPersistentEntries,
  updatePersistentPnL,
  markResetCompleted,
} from "./persistentState.js";

export const DAY_TRADER_CONFIG = {
  ACCOUNT_SIZE: 100_000,
  RISK_PER_TRADE: 1.00,           // 100% — cap is always the binding constraint at aggressive paper trading leverage
  DAILY_MAX_LOSS: -100000,         // Effectively uncapped for aggressive paper trading
  DAILY_MAX_PROFIT: 200000,        // Effectively uncapped for aggressive paper trading
  MAX_OPEN_POSITIONS: 5,         // 5 concurrent for frequent trading (backtested: 34 trades/6mo)
  MAX_NEW_ENTRIES_PER_DAY: 12,   // More entry opportunities per day
  COOLDOWN_MINUTES: 5,           // Fast re-entry on winning symbols
  PARTIAL_PROFIT_PCT: 0.33,      // Proven balanced profit capture
  TRADE_TIMEOUT_MINUTES: 30,     // Proven value — 60 min timeouts were losers
  SCAN_INTERVAL_MINUTES: 3,     // Aligned with storage.ts analysisInterval (180s = 3min)
  WAIT_AFTER_OPEN_MINUTES: 5,
  MIN_DOLLAR_VOLUME: 50_000_000,
  MAX_SPREAD_PCT: 0.001,         // Skip if spread > 0.10%
  MARKET_CLOSE_BUFFER_MINUTES: 15,
  
  // FINAL UNIVERSE: 18 symbols proven profitable with EMA_CROSSOVER strategy
  // 6-month backtest: PF 3.10, 82.4% WR, +$51K on $100K equity at 700% cap
  // Excluded COIN and AMD (net -$118K destroyers)
  ALLOWED_SYMBOLS: [
    "SLV", "NVDA", "QQQ", "TSLA", "AAPL", "MSFT", "SPY", "META",
    "AMZN", "GOOG", "NFLX", "BA", "JPM", "GS", "UBER", "PLTR", "XOM", "DIS"
  ],

  // BASELINE MODE: Disabled - use full TRADING_UNIVERSE from .env for maximum opportunity
  BASELINE_MODE: false,
  BASELINE_UNIVERSE: [
    "SLV", "NVDA", "QQQ", "TSLA", "AAPL", "MSFT", "SPY", "META",
    "AMZN", "GOOG", "NFLX", "BA", "JPM", "GS", "UBER", "PLTR", "XOM", "DIS"
  ],
  
  // BASELINE MAX HOLD: Close positions after this many minutes (only in BASELINE_MODE)
  // Reduces unmatched/open trades for faster learning
  BASELINE_MAX_HOLD_MINUTES: 45,
  
  // BREAKEVEN RULE: When unrealized gain reaches this % of entry, move stop to breakeven
  // Prevents winners from turning into losers (fixes avg loss > avg win leak)
  BREAKEVEN_TRIGGER_PCT: 0.3,    // 0.3% gain triggers breakeven stop — protects remaining position after partial
  BREAKEVEN_OFFSET_PCT: 0.05,   // 0.05% buffer above entry to avoid spread/micro-noise
};

const UNIVERSE_OVERRIDE = process.env.TRADING_UNIVERSE
  ? process.env.TRADING_UNIVERSE.split(",").map(s => s.trim().toUpperCase()).filter(Boolean)
  : [];

if (UNIVERSE_OVERRIDE.length > 0) {
  console.log(`[DayTraderConfig] Universe override active: ${UNIVERSE_OVERRIDE.join(",")}`);
}

function getAllowedUniverse(): string[] {
  if (UNIVERSE_OVERRIDE.length > 0) return UNIVERSE_OVERRIDE;
  return DAY_TRADER_CONFIG.ALLOWED_SYMBOLS;
}

function getBaselineUniverse(): string[] {
  if (UNIVERSE_OVERRIDE.length > 0) return UNIVERSE_OVERRIDE;
  return DAY_TRADER_CONFIG.BASELINE_UNIVERSE;
}

let newEntriesToday = 0;
let dailyPnL = 0;
const cooldownMap: Map<string, number> = new Map();

export function getNewEntriesToday(): number {
  return newEntriesToday;
}

export function incrementNewEntries(): void {
  newEntriesToday++;
  incrementPersistentEntries();
}

export function canOpenNewEntry(): boolean {
  return newEntriesToday < DAY_TRADER_CONFIG.MAX_NEW_ENTRIES_PER_DAY;
}

export function getDailyPnL(): number {
  return dailyPnL;
}

export function updateDailyPnL(amount: number): void {
  dailyPnL += amount;
  updatePersistentPnL(dailyPnL, dailyPnL);
}

export function setDailyPnL(amount: number): void {
  dailyPnL = amount;
  updatePersistentPnL(dailyPnL, dailyPnL);
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
  return getAllowedUniverse().includes(symbol.toUpperCase());
}

/**
 * Check if symbol is allowed for ENTRY in current mode
 * In baseline mode, only BASELINE_UNIVERSE symbols can enter
 * Exits are always allowed for any open position
 */
export function isSymbolAllowedForEntry(symbol: string): { allowed: boolean; reason: string } {
  const upperSymbol = symbol.toUpperCase();
  
  // First check full universe
  if (!getAllowedUniverse().includes(upperSymbol)) {
    return { allowed: false, reason: "SYMBOL_NOT_IN_UNIVERSE" };
  }
  
  // In baseline mode, further restrict to baseline universe
  if (DAY_TRADER_CONFIG.BASELINE_MODE) {
    if (!getBaselineUniverse().includes(upperSymbol)) {
      return { allowed: false, reason: "BASELINE_UNIVERSE_RESTRICTED" };
    }
  }
  
  return { allowed: true, reason: "SYMBOL_ALLOWED" };
}

export function getAllowedSymbols(): string[] {
  return [...getAllowedUniverse()];
}

export function getEntryUniverse(): string[] {
  if (DAY_TRADER_CONFIG.BASELINE_MODE) {
    return [...getBaselineUniverse()];
  }
  return [...getAllowedUniverse()];
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
  markResetCompleted();
  console.log("[DayTraderConfig] Daily reset complete");
}

export function setNewEntriesToday(count: number): void {
  newEntriesToday = count;
}

export async function rehydrateFromTrades(trades: { side: string; realizedPL: number }[]): Promise<void> {
  // First, try to load from persistent state
  const persistedState = loadDailyState();

  if (persistedState) {
    // Use persisted state if available (more reliable than recalculating)
    newEntriesToday = persistedState.newEntriesToday;
    dailyPnL = persistedState.dailyPnL;
    console.log(`[DayTraderConfig] Rehydrated from persistent state: ${newEntriesToday} entries, P/L: $${dailyPnL.toFixed(2)}`);
  } else {
    // Fallback: Count today's buy orders as entries
    const todayEntries = trades.filter(t => t.side === "buy").length;
    newEntriesToday = todayEntries;

    // Sum up realized P/L from today's trades
    const totalPnL = trades.reduce((sum, t) => sum + (t.realizedPL || 0), 0);
    dailyPnL = totalPnL;

    // Save to persistent state
    updatePersistentPnL(dailyPnL, dailyPnL);
    saveDailyState({ newEntriesToday });

    console.log(`[DayTraderConfig] Rehydrated from trades: ${newEntriesToday} entries, P/L: $${dailyPnL.toFixed(2)}`);
  }
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
