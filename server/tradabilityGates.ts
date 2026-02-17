/**
 * Tradability Gates - Pre-strategy hard gates per symbol
 * Block trades if symbol fails any tradability check BEFORE strategy evaluation
 * 
 * FAIL-CLOSED: If data needed for a gate is missing, SKIP with reason.
 * Gates run BEFORE any strategy/validator can approve a trade.
 * 
 * Gates:
 * A) Spread Gate - Block wide bid-ask spreads
 * B) Liquidity Gate - Block low volume symbols
 * C) Volatility Gate (ATR%) - Block too-low or too-high volatility
 * D) Extreme Move Gate - Block symbols with >12% intraday moves
 * E) Time-of-day Gate - Block first 2 min after open, last 10 min before close
 */

import * as alpaca from "./alpaca";
import { getEasternTime } from "./timezone";

export interface TradabilityCheck {
  passed: boolean;
  reasons: string[];
}

export interface TradabilityConfig {
  spreadMaxDollars: number;
  spreadMaxPercent: number;
  minDailyVolume: number;
  atrMinPercent: number;
  atrMaxPercent: number;
  extremeMovePercent: number;
}

const DEFAULT_CONFIG: TradabilityConfig = {
  spreadMaxDollars: 0.05,
  spreadMaxPercent: 0.12,
  minDailyVolume: 500000,
  atrMinPercent: 0.25,
  atrMaxPercent: 6.0,
  extremeMovePercent: 12.0,
};

const TIME_GATE_CONFIG = {
  OPEN_BUFFER_MINUTES: 2,   // Block first 2 minutes after open
  CLOSE_BUFFER_MINUTES: 10, // Block last 10 minutes before close
};

let config = { ...DEFAULT_CONFIG };

export function updateConfig(newConfig: Partial<TradabilityConfig>): void {
  config = { ...config, ...newConfig };
}

export function getConfig(): TradabilityConfig {
  return { ...config };
}

// P6: Spread near-max threshold - block if spread > 90% of max
const SPREAD_NEAR_MAX_PCT = 0.95;

export async function checkSpreadGate(
  symbol: string,
  bid: number,
  ask: number,
  price: number
): Promise<{ passed: boolean; reason?: string; spreadInfo?: { spread: number; maxAllowed: number } }> {
  if (!bid || !ask || !price || bid <= 0 || ask <= 0 || price <= 0) {
    return { passed: false, reason: "spread:missing" };
  }

  const spread = ask - bid;
  const spreadPercent = (spread / price) * 100;
  const maxAllowed = Math.max(config.spreadMaxDollars, (config.spreadMaxPercent / 100) * price);

  if (spread > maxAllowed) {
    return {
      passed: false,
      reason: `spread:${spread.toFixed(3)}>${maxAllowed.toFixed(3)}`,
    };
  }
  
  // P6: Block if spread is > 90% of max (spread:nearMax gate)
  const nearMaxThreshold = maxAllowed * SPREAD_NEAR_MAX_PCT;
  if (spread > nearMaxThreshold) {
    const pctOfMax = (spread / maxAllowed) * 100;
    return {
      passed: false,
      reason: `spread:nearMax_${pctOfMax.toFixed(0)}%`,
    };
  }
  
  return { passed: true, spreadInfo: { spread, maxAllowed } };
}

/**
 * P6: Check if quote is fresh (within 5 seconds)
 */
export async function checkQuoteFreshnessGate(
  quoteTimestamp: string | null | undefined
): Promise<{ passed: boolean; reason?: string; ageMs?: number }> {
  const MAX_QUOTE_AGE_MS = 5000; // 5 seconds
  
  if (!quoteTimestamp) {
    return { passed: false, reason: "quote:missing" };
  }
  
  try {
    const quoteTime = new Date(quoteTimestamp).getTime();
    const now = Date.now();
    const ageMs = now - quoteTime;
    
    if (ageMs > MAX_QUOTE_AGE_MS) {
      return {
        passed: false,
        reason: `quote:stale_age=${Math.round(ageMs / 1000)}s`,
        ageMs,
      };
    }
    
    return { passed: true, ageMs };
  } catch {
    return { passed: false, reason: "quote:invalid_timestamp" };
  }
}

export interface LiquidityPaceData {
  barVolume: number;
  minutesSinceOpen: number;
  projectedDailyVolume: number;
  isMarketOpen: boolean;
  gateMode: 'projected' | 'raw' | 'warmup';
  usedValueForGate: number;
}

export async function checkLiquidityGate(
  symbol: string,
  volume: number | null | undefined,
  paceData?: LiquidityPaceData
): Promise<{ passed: boolean; reason?: string }> {
  if (volume === null || volume === undefined) {
    return { passed: false, reason: "liquidity:missing" };
  }
  
  // LIQUIDITY-PACE-1: Use projected volume during market hours
  let usedValue = volume;
  let gateMode: 'projected' | 'raw' | 'warmup' = 'raw';
  
  if (paceData) {
    gateMode = paceData.gateMode;
    usedValue = paceData.usedValueForGate;
    
    // Warmup mode: first 2 minutes after open, skip check (always pass)
    if (gateMode === 'warmup') {
      console.log(`ACTION=LIQUIDITY_DEBUG symbol=${symbol} gateMode=warmup minutesSinceOpen=${paceData.minutesSinceOpen} barVol=${paceData.barVolume} projectedVol=${Math.floor(paceData.projectedDailyVolume)} PASS_WARMUP`);
      return { passed: true };
    }
  }
  
  if (usedValue < config.minDailyVolume) {
    // Structured log line for liquidity gate failure
    console.log(`ACTION=LIQUIDITY_DEBUG symbol=${symbol} gateMode=${gateMode} minutesSinceOpen=${paceData?.minutesSinceOpen || 0} barVol=${paceData?.barVolume || volume} projectedVol=${paceData?.projectedDailyVolume ? Math.floor(paceData.projectedDailyVolume) : 0} usedVal=${Math.floor(usedValue)} minVol=${config.minDailyVolume} FAIL`);
    return {
      passed: false,
      reason: `liquidity:${gateMode === 'projected' ? 'proj' : 'vol'}=${Math.floor(usedValue / 1000)}k<${config.minDailyVolume / 1000}k`,
    };
  }
  
  // LIQUIDITY-PACE-MINUTES-1: Log PASS case for debugging
  console.log(`ACTION=LIQUIDITY_DEBUG symbol=${symbol} gateMode=${gateMode} minutesSinceOpen=${paceData?.minutesSinceOpen || 0} barVol=${paceData?.barVolume || volume} projectedVol=${paceData?.projectedDailyVolume ? Math.floor(paceData.projectedDailyVolume) : 0} usedVal=${Math.floor(usedValue)} minVol=${config.minDailyVolume} PASS`);
  return { passed: true };
}

export async function checkVolatilityGate(
  symbol: string,
  atrPercent: number | null | undefined
): Promise<{ passed: boolean; reason?: string }> {
  if (atrPercent === null || atrPercent === undefined || !Number.isFinite(atrPercent)) {
    return { passed: false, reason: "atrPct:missing" };
  }
  
  if (atrPercent < config.atrMinPercent) {
    const atrDecimal = atrPercent / 100;
    const minDecimal = config.atrMinPercent / 100;
    return {
      passed: false,
      reason: `atrPct:${parseFloat(atrDecimal.toFixed(4))}<${parseFloat(minDecimal.toFixed(4))}`,
    };
  }
  
  if (atrPercent > config.atrMaxPercent) {
    const atrDecimal = atrPercent / 100;
    const maxDecimal = config.atrMaxPercent / 100;
    return {
      passed: false,
      reason: `atrPct:${parseFloat(atrDecimal.toFixed(4))}>${parseFloat(maxDecimal.toFixed(2))}`,
    };
  }
  
  return { passed: true };
}

export async function checkExtremeMoveGate(
  symbol: string,
  changePercent: number | null | undefined
): Promise<{ passed: boolean; reason?: string }> {
  if (changePercent === null || changePercent === undefined || !Number.isFinite(changePercent)) {
    return { passed: false, reason: "extremeMove:missing" };
  }
  
  if (Math.abs(changePercent) > config.extremeMovePercent) {
    const changeDecimal = Math.abs(changePercent) / 100;
    const maxDecimal = config.extremeMovePercent / 100;
    return {
      passed: false,
      reason: `extremeMove:${parseFloat(changeDecimal.toFixed(4))}>${parseFloat(maxDecimal.toFixed(2))}`,
    };
  }
  
  return { passed: true };
}

export function checkTimeGate(): { passed: boolean; reason?: string } {
  const et = getEasternTime();
  
  const marketOpenHour = 9;
  const marketOpenMinute = 30;
  const marketCloseHour = 16;
  const marketCloseMinute = 0;
  
  const minutesSinceOpen = (et.hour - marketOpenHour) * 60 + (et.minute - marketOpenMinute);
  const minutesUntilClose = (marketCloseHour - et.hour) * 60 + (marketCloseMinute - et.minute);
  
  if (minutesSinceOpen >= 0 && minutesSinceOpen < TIME_GATE_CONFIG.OPEN_BUFFER_MINUTES) {
    return { passed: false, reason: "timeGate:openBuffer" };
  }
  
  if (minutesUntilClose >= 0 && minutesUntilClose < TIME_GATE_CONFIG.CLOSE_BUFFER_MINUTES) {
    return { passed: false, reason: "timeGate:closeBuffer" };
  }
  
  return { passed: true };
}

export interface SymbolMarketData {
  bid: number;
  ask: number;
  price: number;
  volume: number;
  atrPercent: number;
  changePercent: number;
  barCount?: number;
  barReason?: string;
  // Liquidity truth fields (LIQUIDITY-PACE-1)
  volSource?: string;
  timeframe?: string;
  timestampET?: string;
  barVolume?: number;
  sessionVolume?: number;
  isMarketOpen?: boolean;
  minutesSinceOpen?: number;
  projectedDailyVolume?: number;
  usedValueForGate?: number;
  gateMode?: 'projected' | 'raw' | 'warmup';
}

export async function runAllTradabilityGates(
  symbol: string,
  data: SymbolMarketData,
  includeTimeGate: boolean = true
): Promise<TradabilityCheck> {
  const reasons: string[] = [];
  
  const spreadCheck = await checkSpreadGate(symbol, data.bid, data.ask, data.price);
  if (!spreadCheck.passed && spreadCheck.reason) {
    reasons.push(spreadCheck.reason);
  }
  
  // LIQUIDITY-PACE-1: Pass pace data to liquidity gate
  const paceData: LiquidityPaceData | undefined = data.gateMode ? {
    barVolume: data.barVolume || 0,
    minutesSinceOpen: data.minutesSinceOpen || 0,
    projectedDailyVolume: data.projectedDailyVolume || 0,
    isMarketOpen: data.isMarketOpen || false,
    gateMode: data.gateMode,
    usedValueForGate: data.usedValueForGate || data.volume,
  } : undefined;
  
  const liquidityCheck = await checkLiquidityGate(symbol, data.volume, paceData);
  if (!liquidityCheck.passed && liquidityCheck.reason) {
    reasons.push(liquidityCheck.reason);
  }
  
  const volatilityCheck = await checkVolatilityGate(symbol, data.atrPercent);
  if (!volatilityCheck.passed && volatilityCheck.reason) {
    reasons.push(volatilityCheck.reason);
  }
  
  const extremeCheck = await checkExtremeMoveGate(symbol, data.changePercent);
  if (!extremeCheck.passed && extremeCheck.reason) {
    reasons.push(extremeCheck.reason);
  }
  
  if (includeTimeGate) {
    const timeCheck = checkTimeGate();
    if (!timeCheck.passed && timeCheck.reason) {
      reasons.push(timeCheck.reason);
    }
  }
  
  return {
    passed: reasons.length === 0,
    reasons,
  };
}

export interface TradabilityResult {
  ok: boolean;
  reasons: string[];
}

export async function evaluateTradability(
  symbol: string,
  includeTimeGate: boolean = true
): Promise<TradabilityResult> {
  const data = await fetchSymbolMarketData(symbol);
  
  // BUGFIX-P1-DATA-1: Fail-closed with NO_QUOTE when quote data unavailable
  if (!data) {
    return {
      ok: false,
      reasons: ["NO_QUOTE"],
    };
  }
  
  const check = await runAllTradabilityGates(symbol, data, includeTimeGate);
  
  return {
    ok: check.passed,
    reasons: check.reasons,
  };
}

export function logTradabilitySkip(symbol: string, result: TradabilityResult | TradabilityCheck, data?: SymbolMarketData | null): void {
  const reasons = 'ok' in result ? result.reasons : result.reasons;
  const passed = 'ok' in result ? result.ok : result.passed;
  
  if (!passed && reasons.length > 0) {
    // BUGFIX-P1-DATA-1: Include numeric evidence in logs
    if (data) {
      const spreadCents = ((data.ask - data.bid) * 100).toFixed(1);
      const spreadPct = (((data.ask - data.bid) / data.price) * 100).toFixed(3);
      const maxSpreadDollars = Math.max(DEFAULT_CONFIG.spreadMaxDollars, (DEFAULT_CONFIG.spreadMaxPercent / 100) * data.price);
      const maxSpreadCents = (maxSpreadDollars * 100).toFixed(1);
      console.log(`ACTION=SKIP symbol=${symbol} SKIP_REASONS=[${reasons.join(",")}] ` +
        `spreadCents=${spreadCents} spreadPct=${spreadPct}% maxAllowedCents=${maxSpreadCents} ` +
        `atrPct=${data.atrPercent.toFixed(3)}% atrMin=${DEFAULT_CONFIG.atrMinPercent}% atrMax=${DEFAULT_CONFIG.atrMaxPercent}% ` +
        `vol=${Math.floor(data.volume / 1000)}k minVol=${DEFAULT_CONFIG.minDailyVolume / 1000}k`);
    } else {
      console.log(`ACTION=SKIP symbol=${symbol} SKIP_REASONS=[${reasons.join(",")}]`);
    }
  }
}

// BUGFIX-P1-DATA-1: Enhanced evaluation with logging
export async function evaluateTradabilityWithLogging(
  symbol: string,
  includeTimeGate: boolean = true
): Promise<TradabilityResult> {
  const data = await fetchSymbolMarketData(symbol);
  
  if (!data) {
    console.log(`ACTION=SKIP symbol=${symbol} SKIP_REASONS=[NO_QUOTE] (no valid bid/ask data)`);
    return {
      ok: false,
      reasons: ["NO_QUOTE"],
    };
  }
  
  const check = await runAllTradabilityGates(symbol, data, includeTimeGate);
  
  if (!check.passed) {
    logTradabilitySkip(symbol, check, data);
  } else {
    // Log PASS with numeric evidence
    const spreadCents = ((data.ask - data.bid) * 100).toFixed(1);
    const spreadPct = (((data.ask - data.bid) / data.price) * 100).toFixed(3);
    console.log(`ACTION=PASS symbol=${symbol} spreadCents=${spreadCents} spreadPct=${spreadPct}% ` +
      `atrPct=${data.atrPercent.toFixed(3)}% vol=${Math.floor(data.volume / 1000)}k`);
  }
  
  return {
    ok: check.passed,
    reasons: check.reasons,
  };
}

export async function fetchSymbolMarketData(symbol: string): Promise<SymbolMarketData | null> {
  try {
    // BUGFIX-P1-DATA-1: Use getExtendedQuote for REAL bid/ask (not mock spread)
    const quote = await alpaca.getExtendedQuote(symbol);
    
    // Fail-closed: NO_QUOTE if missing price or bid/ask
    if (!quote || quote.price <= 0) {
      console.log(`[TRADABILITY] ${symbol}: NO_QUOTE - price=${quote?.price || 0}`);
      return null;
    }
    
    const price = quote.price;
    const bid = quote.bid;  // REAL bid from Alpaca
    const ask = quote.ask;  // REAL ask from Alpaca
    
    // Fail-closed: NO_QUOTE if bid/ask missing or invalid
    if (bid <= 0 || ask <= 0) {
      console.log(`[TRADABILITY] ${symbol}: NO_QUOTE - bid=${bid} ask=${ask} (invalid bid/ask)`);
      return null;
    }
    
    // Use getBarsSafe with warm-start for reliable daily bar data (includes IEX feed)
    const barResult = await alpaca.getBarsSafe(symbol, "1Day", 10);
    const bars = barResult.bars;
    let volume = 0;
    let atrPercent = 0;
    let changePercent = 0;
    let barCount = bars.length;
    
    // Debug: log bar fetch result for ATR calculation
    if (barCount < 10) {
      console.log(`[TRADABILITY] ${symbol}: ATR_INSUFFICIENT_BARS barCount=${barCount} need=10 reason=${barResult.reason || 'unknown'}`);
    }
    
    // Liquidity truth tracking
    let volSource = "none";
    let timeframe = "1Day";
    let barTimestampRaw = "";
    let barVolume = 0;
    let sessionVolume = 0;
    
    // LIQUIDITY-PACE-MINUTES-1: Get market clock for isMarketOpen and minutesSinceOpen
    // Use getEasternTime() for reliable ET time calculation (same as checkTimeGate)
    let isMarketOpen = false;
    let minutesSinceOpen = 0;
    
    try {
      const clock = await alpaca.getClock();
      isMarketOpen = clock.is_open;
      
      // Calculate minutes since 9:30 AM ET if market is open
      if (isMarketOpen) {
        // Use getEasternTime() for reliable ET time (matches checkTimeGate)
        const et = getEasternTime();
        const marketOpenHour = 9;
        const marketOpenMinute = 30;
        
        // Calculate minutes since 9:30 ET
        minutesSinceOpen = (et.hour - marketOpenHour) * 60 + (et.minute - marketOpenMinute);
        
        // LIQUIDITY-PACE-MINUTES-1: Clamp to >= 1 during market hours to avoid projectedVol=0
        // This prevents division by zero and ensures volume projection always works
        if (minutesSinceOpen < 1) {
          minutesSinceOpen = 1;
        }
      }
    } catch (e) {
      // Default to false if clock fetch fails - will use raw volume mode
      console.log(`[TRADABILITY] Clock fetch failed: ${e}`);
    }
    
    if (bars && bars.length > 0) {
      const latestBar = bars[bars.length - 1];
      barVolume = latestBar.v || 0;
      barTimestampRaw = latestBar.t || "";
      volume = barVolume;
      volSource = "bar";
      
      if (bars.length >= 2) {
        const prevClose = bars[bars.length - 2].c;
        changePercent = prevClose > 0 ? ((price - prevClose) / prevClose) * 100 : 0;
      }
      
      if (bars.length >= 10) {
        let atrSum = 0;
        for (let i = 1; i < bars.length; i++) {
          const high = bars[i].h;
          const low = bars[i].l;
          const prevClose = bars[i - 1].c;
          const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
          atrSum += tr;
        }
        const atr = atrSum / (bars.length - 1);
        atrPercent = price > 0 ? (atr / price) * 100 : 0;
      }
    }
    
    // Try to get session cumulative volume from snapshot (if available)
    try {
      const snapshot = await alpaca.getSnapshot(symbol);
      if (snapshot && snapshot.volume) {
        sessionVolume = snapshot.volume || 0;
      }
    } catch (e) {
      // Snapshot not available, sessionVolume stays 0
    }
    
    // Convert bar timestamp to ET for display
    let timestampET = "";
    if (barTimestampRaw) {
      try {
        const d = new Date(barTimestampRaw);
        timestampET = d.toLocaleString("en-US", { timeZone: "America/New_York", hour: "2-digit", minute: "2-digit", hour12: false }) + " ET " + d.toISOString().split("T")[0];
      } catch (e) {
        timestampET = barTimestampRaw;
      }
    }
    
    // LIQUIDITY-PACE-1: Compute projected daily volume
    // projectedDailyVolume = (barVolume / max(minutesSinceOpen, 1)) * 390
    // 390 = total trading minutes in a day (9:30 AM - 4:00 PM ET)
    const TRADING_MINUTES_PER_DAY = 390;
    const WARMUP_MINUTES = 2; // First 2 minutes use warmup mode
    
    let projectedDailyVolume = 0;
    let usedValueForGate = barVolume;
    let gateMode: 'projected' | 'raw' | 'warmup' = 'raw';
    
    if (isMarketOpen && minutesSinceOpen > 0) {
      if (minutesSinceOpen <= WARMUP_MINUTES) {
        // Warmup mode: first 2 minutes, skip liquidity check
        gateMode = 'warmup';
        usedValueForGate = config.minDailyVolume; // Will always pass
      } else {
        // Projected mode: use volume pace projection
        gateMode = 'projected';
        projectedDailyVolume = (barVolume / Math.max(minutesSinceOpen, 1)) * TRADING_MINUTES_PER_DAY;
        usedValueForGate = projectedDailyVolume;
      }
    }
    // else: raw mode (market closed or minutesSinceOpen=0), use actual barVolume
    
    return {
      bid,
      ask,
      price,
      volume,
      atrPercent,
      changePercent,
      barCount,
      barReason: barResult.reason,
      volSource,
      timeframe,
      timestampET,
      barVolume,
      sessionVolume,
      isMarketOpen,
      minutesSinceOpen,
      projectedDailyVolume,
      usedValueForGate,
      gateMode,
    };
  } catch (error) {
    console.log(`[TRADABILITY] Failed to fetch market data for ${symbol}: ${error}`);
    return null;
  }
}
