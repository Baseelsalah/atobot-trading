/**
 * Replay Smoke Test - Market-Hours Replay (NO ORDERS)
 * 
 * Verifies the bot can run smoothly and generate signals during market hours
 * using historical bars, without placing orders.
 * 
 * Usage: npx tsx scripts/replay_smoke.ts [--date=YYYY-MM-DD]
 */

import * as fs from "fs";
import * as path from "path";

// Minimal Alpaca API wrapper for historical data
const ALPACA_API_KEY = process.env.ALPACA_API_KEY || "";
const ALPACA_API_SECRET = process.env.ALPACA_SECRET_KEY || process.env.ALPACA_API_SECRET || "";
const ALPACA_BASE_URL = "https://data.alpaca.markets";

interface AlpacaBar {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

interface ReplayTick {
  timeET: string;
  symbolsEvaluated: string[];
  regimeOk: boolean;
  regimeLabel: string;
  signals: ReplaySignal[];
  skipReasons: Record<string, number>;
  wouldTradeCount: number;
}

interface ReplaySignal {
  symbol: string;
  strategy: string;
  side: string;
  confidence: number;
  reason: string;
  wouldTrade: boolean;
}

interface ReplayReport {
  replayDate: string;
  windowStart: string;
  windowEnd: string;
  totalTicks: number;
  symbolsEvaluated: number;
  uniqueSymbols: string[];
  signalsByStrategy: Record<string, number>;
  signalsBySymbol: Record<string, number>;
  skipReasons: Record<string, number>;
  regimeBreakdown: Record<string, number>;
  wouldTradeCount: number;
  avgConfidence: number;
  errors: string[];
  ticks: ReplayTick[];
}

// Universe from dayTraderConfig (BASELINE_UNIVERSE)
const BASELINE_UNIVERSE = [
  "SPY", "QQQ", "IWM", "DIA",
  "TLT", "GLD", "SLV",
  "XLF", "XLK", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY",
  "AAPL", "MSFT", "NVDA", "AMZN", "TSLA"
];

// Gate thresholds (from tradabilityGates.ts)
const GATE_CONFIG = {
  spreadMaxDollars: 0.03,
  spreadMaxPercent: 0.08,
  minDailyVolume: 500000,
  atrMinPercent: 0.25,
  atrMaxPercent: 6.0,
  extremeMovePercent: 12.0,
};

// Strategy configs (from strategyEngine.ts)
const VWAP_CONFIG = {
  MIN_DEVIATION_PCT: 0.3,
  MAX_DEVIATION_PCT: 2.0,
  RSI_OVERSOLD: 35,
  RSI_OVERBOUGHT: 65,
  BASE_CONFIDENCE: 60,
  MAX_CONFIDENCE: 85,
};

const ORB_CONFIG = {
  RANGE_MINUTES: 15,
  BREAKOUT_BUFFER_PCT: 0.05,
  MIN_RANGE_PCT: 0.15,
  MAX_RANGE_PCT: 1.5,
  BASE_CONFIDENCE: 65,
  MAX_CONFIDENCE: 85,
};

// Simple indicator calculations
function ema(prices: number[], period: number): number[] {
  const result: number[] = [];
  const k = 2 / (period + 1);
  let ema = prices[0];
  result.push(ema);
  for (let i = 1; i < prices.length; i++) {
    ema = prices[i] * k + ema * (1 - k);
    result.push(ema);
  }
  return result;
}

function rsi(prices: number[], period: number = 14): number[] {
  const result: number[] = [];
  if (prices.length < period + 1) return result;
  
  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const change = prices[i] - prices[i - 1];
    if (change >= 0) gains += change;
    else losses += Math.abs(change);
  }
  
  let avgGain = gains / period;
  let avgLoss = losses / period;
  
  for (let i = 0; i <= period; i++) {
    result.push(50); // Fill early values with neutral
  }
  
  for (let i = period + 1; i < prices.length; i++) {
    const change = prices[i] - prices[i - 1];
    avgGain = (avgGain * (period - 1) + (change > 0 ? change : 0)) / period;
    avgLoss = (avgLoss * (period - 1) + (change < 0 ? Math.abs(change) : 0)) / period;
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    result.push(100 - (100 / (1 + rs)));
  }
  
  return result;
}

function atr(highs: number[], lows: number[], closes: number[], period: number = 14): number[] {
  const result: number[] = [];
  if (highs.length < 2) return result;
  
  const trueRanges: number[] = [highs[0] - lows[0]];
  for (let i = 1; i < highs.length; i++) {
    const tr = Math.max(
      highs[i] - lows[i],
      Math.abs(highs[i] - closes[i - 1]),
      Math.abs(lows[i] - closes[i - 1])
    );
    trueRanges.push(tr);
  }
  
  return ema(trueRanges, period);
}

function vwap(highs: number[], lows: number[], closes: number[], volumes: number[]): number[] {
  const result: number[] = [];
  let cumTypicalPriceVol = 0;
  let cumVolume = 0;
  
  for (let i = 0; i < closes.length; i++) {
    const typicalPrice = (highs[i] + lows[i] + closes[i]) / 3;
    cumTypicalPriceVol += typicalPrice * volumes[i];
    cumVolume += volumes[i];
    result.push(cumVolume > 0 ? cumTypicalPriceVol / cumVolume : closes[i]);
  }
  
  return result;
}

// Alpaca API request
async function alpacaDataRequest<T>(endpoint: string): Promise<T> {
  const url = `${ALPACA_BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      "APCA-API-KEY-ID": ALPACA_API_KEY,
      "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    },
  });
  
  if (!response.ok) {
    throw new Error(`Alpaca API error: ${response.status} ${response.statusText}`);
  }
  
  return response.json() as Promise<T>;
}

// Fetch historical bars for a symbol on a specific date
async function fetchBarsForDate(
  symbol: string,
  date: string,
  startHour: number,
  startMin: number,
  endHour: number,
  endMin: number
): Promise<AlpacaBar[]> {
  // Build date range in ET, then convert to UTC for API
  const startET = new Date(`${date}T${String(startHour).padStart(2, '0')}:${String(startMin).padStart(2, '0')}:00-05:00`);
  const endET = new Date(`${date}T${String(endHour).padStart(2, '0')}:${String(endMin).padStart(2, '0')}:00-05:00`);
  
  const startStr = startET.toISOString();
  const endStr = endET.toISOString();
  
  try {
    const response = await alpacaDataRequest<{
      bars?: AlpacaBar[] | Record<string, AlpacaBar[]>;
      [key: string]: AlpacaBar[] | Record<string, AlpacaBar[]> | undefined;
    }>(`/v2/stocks/${symbol}/bars?timeframe=1Min&start=${startStr}&end=${endStr}&limit=500&feed=iex`);
    
    if (Array.isArray(response.bars)) {
      return response.bars;
    } else if (response.bars && typeof response.bars === 'object') {
      return (response.bars as Record<string, AlpacaBar[]>)[symbol] || [];
    } else if (response[symbol] && Array.isArray(response[symbol])) {
      return response[symbol] as AlpacaBar[];
    }
    return [];
  } catch (error) {
    console.error(`[Replay] Failed to fetch bars for ${symbol}:`, error);
    return [];
  }
}

// Evaluate market regime from SPY bars
function evaluateRegime(spyBars: AlpacaBar[]): { ok: boolean; label: string } {
  if (spyBars.length < 21) {
    return { ok: false, label: "missing" };
  }
  
  const closes = spyBars.map(b => b.c);
  const ema9 = ema(closes, 9);
  const ema21 = ema(closes, 21);
  
  const latestEma9 = ema9[ema9.length - 1];
  const latestEma21 = ema21[ema21.length - 1];
  const spread = Math.abs(latestEma9 - latestEma21) / latestEma21 * 100;
  
  if (latestEma9 >= latestEma21) {
    return { ok: true, label: spread < 0.15 ? "chop" : "bull" };
  } else {
    return { ok: false, label: spread < 0.15 ? "chop" : "bear" };
  }
}

// Check tradability gates using bar data (approximations)
function checkGates(bars: AlpacaBar[]): { passed: boolean; reasons: string[] } {
  const reasons: string[] = [];
  
  if (!bars || bars.length < 20) {
    reasons.push("INSUFFICIENT_BARS");
    return { passed: false, reasons };
  }
  
  const closes = bars.map(b => b.c);
  const highs = bars.map(b => b.h);
  const lows = bars.map(b => b.l);
  const volumes = bars.map(b => b.v);
  
  const latestClose = closes[closes.length - 1];
  const latestHigh = highs[highs.length - 1];
  const latestLow = lows[lows.length - 1];
  
  // Approximate spread from high-low (not exact but useful for replay)
  const approxSpread = latestHigh - latestLow;
  const maxAllowed = Math.max(GATE_CONFIG.spreadMaxDollars, (GATE_CONFIG.spreadMaxPercent / 100) * latestClose);
  if (approxSpread > maxAllowed * 3) { // Allow 3x since we're using H-L not bid-ask
    reasons.push("SPREAD_TOO_WIDE");
  }
  
  // Check liquidity (sum recent volume)
  const recentVolume = volumes.slice(-20).reduce((a, b) => a + b, 0);
  if (recentVolume < GATE_CONFIG.minDailyVolume / 10) {
    reasons.push("LIQUIDITY_TOO_LOW");
  }
  
  // Check ATR%
  const atrValues = atr(highs, lows, closes, 14);
  const latestAtr = atrValues[atrValues.length - 1] || 0;
  const atrPct = (latestAtr / latestClose) * 100;
  
  if (atrPct < GATE_CONFIG.atrMinPercent) {
    reasons.push("ATR_TOO_LOW");
  }
  if (atrPct > GATE_CONFIG.atrMaxPercent) {
    reasons.push("ATR_TOO_HIGH");
  }
  
  // Check extreme move (use first bar of day as reference)
  const openPrice = bars[0].o;
  const moveFromOpen = Math.abs(latestClose - openPrice) / openPrice * 100;
  if (moveFromOpen > GATE_CONFIG.extremeMovePercent) {
    reasons.push("EXTREME_MOVE");
  }
  
  return { passed: reasons.length === 0, reasons };
}

// Evaluate VWAP Reversion strategy
function evaluateVwapReversion(bars: AlpacaBar[]): ReplaySignal | null {
  if (bars.length < 30) return null;
  
  const closes = bars.map(b => b.c);
  const highs = bars.map(b => b.h);
  const lows = bars.map(b => b.l);
  const volumes = bars.map(b => b.v);
  
  const vwapValues = vwap(highs, lows, closes, volumes);
  const rsiValues = rsi(closes, 14);
  
  const latestClose = closes[closes.length - 1];
  const latestVwap = vwapValues[vwapValues.length - 1];
  const latestRsi = rsiValues[rsiValues.length - 1];
  
  if (!latestVwap || latestVwap === 0) return null;
  
  const deviation = ((latestClose - latestVwap) / latestVwap) * 100;
  const absDeviation = Math.abs(deviation);
  
  if (absDeviation < VWAP_CONFIG.MIN_DEVIATION_PCT || absDeviation > VWAP_CONFIG.MAX_DEVIATION_PCT) {
    return null;
  }
  
  // Check RSI for confirmation
  let side: "buy" | "sell" | null = null;
  if (deviation < 0 && latestRsi < VWAP_CONFIG.RSI_OVERSOLD) {
    side = "buy";
  } else if (deviation > 0 && latestRsi > VWAP_CONFIG.RSI_OVERBOUGHT) {
    side = "sell";
  }
  
  if (!side) return null;
  
  const confidence = Math.min(
    VWAP_CONFIG.MAX_CONFIDENCE,
    VWAP_CONFIG.BASE_CONFIDENCE + (absDeviation - VWAP_CONFIG.MIN_DEVIATION_PCT) * 10
  );
  
  return {
    symbol: "",
    strategy: "VWAP_REVERSION",
    side,
    confidence,
    reason: `VWAP dev=${deviation.toFixed(2)}%, RSI=${latestRsi.toFixed(1)}`,
    wouldTrade: confidence >= 60,
  };
}

// Evaluate ORB strategy
function evaluateOrb(bars: AlpacaBar[], minutesSinceOpen: number): ReplaySignal | null {
  if (minutesSinceOpen < ORB_CONFIG.RANGE_MINUTES) return null;
  if (bars.length < ORB_CONFIG.RANGE_MINUTES + 5) return null;
  
  // Opening range is first N bars
  const rangeBars = bars.slice(0, ORB_CONFIG.RANGE_MINUTES);
  const orbHigh = Math.max(...rangeBars.map(b => b.h));
  const orbLow = Math.min(...rangeBars.map(b => b.l));
  const orbRange = orbHigh - orbLow;
  const midPrice = (orbHigh + orbLow) / 2;
  const rangePct = (orbRange / midPrice) * 100;
  
  if (rangePct < ORB_CONFIG.MIN_RANGE_PCT || rangePct > ORB_CONFIG.MAX_RANGE_PCT) {
    return null;
  }
  
  const closes = bars.map(b => b.c);
  const latestClose = closes[closes.length - 1];
  const ema20Values = ema(closes, 20);
  const latestEma20 = ema20Values[ema20Values.length - 1];
  
  // Check for breakout
  const breakoutBuffer = orbRange * ORB_CONFIG.BREAKOUT_BUFFER_PCT;
  let side: "buy" | "sell" | null = null;
  
  if (latestClose > orbHigh + breakoutBuffer && latestClose > latestEma20) {
    side = "buy";
  } else if (latestClose < orbLow - breakoutBuffer && latestClose < latestEma20) {
    side = "sell";
  }
  
  if (!side) return null;
  
  const confidence = Math.min(
    ORB_CONFIG.MAX_CONFIDENCE,
    ORB_CONFIG.BASE_CONFIDENCE + (rangePct - ORB_CONFIG.MIN_RANGE_PCT) * 5
  );
  
  return {
    symbol: "",
    strategy: "ORB",
    side,
    confidence,
    reason: `ORB ${side === "buy" ? "breakout UP" : "breakdown DOWN"}, range=${rangePct.toFixed(2)}%`,
    wouldTrade: confidence >= 60,
  };
}

// Get last trading day (simple logic - go back until we hit a weekday)
function getLastTradingDay(): string {
  const now = new Date();
  // Go back one day
  now.setDate(now.getDate() - 1);
  
  // Skip weekends
  while (now.getDay() === 0 || now.getDay() === 6) {
    now.setDate(now.getDate() - 1);
  }
  
  return now.toISOString().split("T")[0];
}

// Main replay function
async function runReplaySmoke(targetDate: string): Promise<ReplayReport> {
  console.log(`\n[Replay] ============================================`);
  console.log(`[Replay] PREMARKET SMOKE BACKTEST`);
  console.log(`[Replay] Date: ${targetDate}`);
  console.log(`[Replay] Window: 09:35 - 11:35 ET`);
  console.log(`[Replay] Universe: ${BASELINE_UNIVERSE.length} symbols`);
  console.log(`[Replay] ============================================\n`);
  
  const report: ReplayReport = {
    replayDate: targetDate,
    windowStart: "09:35",
    windowEnd: "11:35",
    totalTicks: 0,
    symbolsEvaluated: 0,
    uniqueSymbols: [],
    signalsByStrategy: {},
    signalsBySymbol: {},
    skipReasons: {},
    regimeBreakdown: { bull: 0, bear: 0, chop: 0, missing: 0 },
    wouldTradeCount: 0,
    avgConfidence: 0,
    errors: [],
    ticks: [],
  };
  
  // Step 1: Fetch SPY bars for regime evaluation (need warm-start data)
  console.log("[Replay] Fetching SPY bars for regime filter...");
  const spyBars = await fetchBarsForDate("SPY", targetDate, 9, 30, 11, 35);
  
  if (spyBars.length === 0) {
    report.errors.push("SPY data missing - cannot evaluate regime (FAIL-CLOSED)");
    console.error("[Replay] FATAL: SPY data missing - skipping all evaluation");
    return report;
  }
  
  console.log(`[Replay] SPY bars fetched: ${spyBars.length}`);
  
  // Step 2: Fetch bars for all symbols
  console.log("[Replay] Fetching bars for all symbols...");
  const symbolBars: Record<string, AlpacaBar[]> = { SPY: spyBars };
  
  for (const symbol of BASELINE_UNIVERSE) {
    if (symbol === "SPY") continue;
    
    const bars = await fetchBarsForDate(symbol, targetDate, 9, 30, 11, 35);
    if (bars.length > 0) {
      symbolBars[symbol] = bars;
      console.log(`[Replay] ${symbol}: ${bars.length} bars`);
    } else {
      console.log(`[Replay] ${symbol}: NO DATA (skip)`);
      report.skipReasons[`${symbol}:NO_DATA`] = (report.skipReasons[`${symbol}:NO_DATA`] || 0) + 1;
    }
    
    // Small delay to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  
  const availableSymbols = Object.keys(symbolBars);
  report.uniqueSymbols = availableSymbols;
  console.log(`\n[Replay] Data available for ${availableSymbols.length} symbols`);
  
  // Step 3: Simulate each minute tick (09:35 to 11:35 = 120 minutes)
  const tickMinutes = 120;
  let totalConfidence = 0;
  let confidenceCount = 0;
  
  for (let minuteOffset = 0; minuteOffset < tickMinutes; minuteOffset++) {
    const tickHour = Math.floor((9 * 60 + 35 + minuteOffset) / 60);
    const tickMinute = (9 * 60 + 35 + minuteOffset) % 60;
    const timeET = `${String(tickHour).padStart(2, '0')}:${String(tickMinute).padStart(2, '0')} ET`;
    
    const tick: ReplayTick = {
      timeET,
      symbolsEvaluated: [],
      regimeOk: false,
      regimeLabel: "unknown",
      signals: [],
      skipReasons: {},
      wouldTradeCount: 0,
    };
    
    // Find SPY bars up to this tick
    const spyBarsForTick = spyBars.filter(bar => {
      const barTime = new Date(bar.t);
      const barMinutesET = barTime.getUTCHours() * 60 + barTime.getUTCMinutes() - 5 * 60; // Adjust UTC to ET
      const tickMinutesET = tickHour * 60 + tickMinute;
      return barMinutesET <= tickMinutesET;
    });
    
    // Evaluate regime
    const regime = evaluateRegime(spyBarsForTick);
    tick.regimeOk = regime.ok;
    tick.regimeLabel = regime.label;
    report.regimeBreakdown[regime.label] = (report.regimeBreakdown[regime.label] || 0) + 1;
    
    if (!regime.ok) {
      tick.skipReasons["REGIME_BLOCKED"] = 1;
      report.skipReasons["REGIME_BLOCKED"] = (report.skipReasons["REGIME_BLOCKED"] || 0) + 1;
      report.ticks.push(tick);
      report.totalTicks++;
      continue;
    }
    
    // Evaluate each symbol
    for (const symbol of availableSymbols) {
      const bars = symbolBars[symbol];
      
      // Filter bars up to this tick
      const barsForTick = bars.filter(bar => {
        const barTime = new Date(bar.t);
        const barMinutesET = barTime.getUTCHours() * 60 + barTime.getUTCMinutes() - 5 * 60;
        const tickMinutesET = tickHour * 60 + tickMinute;
        return barMinutesET <= tickMinutesET;
      });
      
      if (barsForTick.length < 5) {
        tick.skipReasons["INSUFFICIENT_BARS"] = (tick.skipReasons["INSUFFICIENT_BARS"] || 0) + 1;
        continue;
      }
      
      tick.symbolsEvaluated.push(symbol);
      report.symbolsEvaluated++;
      
      // Check tradability gates
      const gates = checkGates(barsForTick);
      if (!gates.passed) {
        for (const reason of gates.reasons) {
          tick.skipReasons[reason] = (tick.skipReasons[reason] || 0) + 1;
          report.skipReasons[reason] = (report.skipReasons[reason] || 0) + 1;
        }
        continue;
      }
      
      // Evaluate strategies
      const minutesSinceOpen = minuteOffset + 5; // We start at 09:35, which is 5 min after open
      
      // VWAP Reversion
      const vwapSignal = evaluateVwapReversion(barsForTick);
      if (vwapSignal) {
        vwapSignal.symbol = symbol;
        tick.signals.push(vwapSignal);
        report.signalsByStrategy["VWAP_REVERSION"] = (report.signalsByStrategy["VWAP_REVERSION"] || 0) + 1;
        report.signalsBySymbol[symbol] = (report.signalsBySymbol[symbol] || 0) + 1;
        
        if (vwapSignal.wouldTrade) {
          tick.wouldTradeCount++;
          report.wouldTradeCount++;
          totalConfidence += vwapSignal.confidence;
          confidenceCount++;
        }
      }
      
      // ORB
      const orbSignal = evaluateOrb(barsForTick, minutesSinceOpen);
      if (orbSignal) {
        orbSignal.symbol = symbol;
        tick.signals.push(orbSignal);
        report.signalsByStrategy["ORB"] = (report.signalsByStrategy["ORB"] || 0) + 1;
        report.signalsBySymbol[symbol] = (report.signalsBySymbol[symbol] || 0) + 1;
        
        if (orbSignal.wouldTrade) {
          tick.wouldTradeCount++;
          report.wouldTradeCount++;
          totalConfidence += orbSignal.confidence;
          confidenceCount++;
        }
      }
    }
    
    // Aggregate skip reasons to report
    for (const [reason, count] of Object.entries(tick.skipReasons)) {
      report.skipReasons[reason] = (report.skipReasons[reason] || 0) + count;
    }
    
    report.ticks.push(tick);
    report.totalTicks++;
    
    // Progress update every 30 ticks
    if ((minuteOffset + 1) % 30 === 0) {
      console.log(`[Replay] Tick ${minuteOffset + 1}/${tickMinutes}: signals=${tick.signals.length}, wouldTrade=${tick.wouldTradeCount}`);
    }
  }
  
  report.avgConfidence = confidenceCount > 0 ? totalConfidence / confidenceCount : 0;
  
  return report;
}

// Generate summary text
function generateSummary(report: ReplayReport): string {
  const lines: string[] = [];
  lines.push("=".repeat(60));
  lines.push("REPLAY SMOKE TEST SUMMARY");
  lines.push("=".repeat(60));
  lines.push(`Date: ${report.replayDate}`);
  lines.push(`Window: ${report.windowStart} - ${report.windowEnd} ET`);
  lines.push(`Total Ticks: ${report.totalTicks}`);
  lines.push(`Errors: ${report.errors.length}`);
  lines.push("");
  lines.push("-".repeat(40));
  lines.push("SYMBOLS");
  lines.push("-".repeat(40));
  lines.push(`Unique Symbols Evaluated: ${report.uniqueSymbols.length}`);
  lines.push(`Total Symbol Evaluations: ${report.symbolsEvaluated}`);
  lines.push("");
  lines.push("-".repeat(40));
  lines.push("REGIME BREAKDOWN");
  lines.push("-".repeat(40));
  for (const [label, count] of Object.entries(report.regimeBreakdown)) {
    const pct = report.totalTicks > 0 ? (count / report.totalTicks * 100).toFixed(1) : "0";
    lines.push(`  ${label}: ${count} (${pct}%)`);
  }
  lines.push("");
  lines.push("-".repeat(40));
  lines.push("SIGNALS BY STRATEGY");
  lines.push("-".repeat(40));
  for (const [strategy, count] of Object.entries(report.signalsByStrategy)) {
    lines.push(`  ${strategy}: ${count}`);
  }
  const totalSignals = Object.values(report.signalsByStrategy).reduce((a, b) => a + b, 0);
  lines.push(`  TOTAL: ${totalSignals}`);
  lines.push("");
  lines.push("-".repeat(40));
  lines.push("SIGNALS BY SYMBOL (top 10)");
  lines.push("-".repeat(40));
  const sortedBySymbol = Object.entries(report.signalsBySymbol)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);
  for (const [symbol, count] of sortedBySymbol) {
    lines.push(`  ${symbol}: ${count}`);
  }
  lines.push("");
  lines.push("-".repeat(40));
  lines.push("SKIP REASONS (top 15)");
  lines.push("-".repeat(40));
  const sortedSkips = Object.entries(report.skipReasons)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15);
  for (const [reason, count] of sortedSkips) {
    lines.push(`  ${reason}: ${count}`);
  }
  lines.push("");
  lines.push("-".repeat(40));
  lines.push("WOULD-TRADE DECISIONS");
  lines.push("-".repeat(40));
  lines.push(`Count: ${report.wouldTradeCount}`);
  lines.push(`Average Confidence: ${report.avgConfidence.toFixed(1)}`);
  lines.push("");
  lines.push("-".repeat(40));
  lines.push("ERRORS");
  lines.push("-".repeat(40));
  if (report.errors.length === 0) {
    lines.push("  (none)");
  } else {
    for (const error of report.errors) {
      lines.push(`  - ${error}`);
    }
  }
  lines.push("");
  lines.push("=".repeat(60));
  lines.push(`RESULT: errors=${report.errors.length} symbolsEvaluated=${report.uniqueSymbols.length} signals=${totalSignals} wouldTrade=${report.wouldTradeCount}`);
  lines.push("=".repeat(60));
  
  return lines.join("\n");
}

// Main entry point
async function main() {
  // Parse command line args
  const args = process.argv.slice(2);
  let targetDate = getLastTradingDay();
  
  for (const arg of args) {
    if (arg.startsWith("--date=")) {
      targetDate = arg.split("=")[1];
    }
  }
  
  // Validate API keys
  if (!ALPACA_API_KEY || !ALPACA_API_SECRET) {
    console.error("[Replay] ERROR: ALPACA_API_KEY and ALPACA_API_SECRET required");
    process.exit(1);
  }
  
  try {
    const report = await runReplaySmoke(targetDate);
    
    // Create output directory
    const replayDir = path.join(process.cwd(), "reports", "replay");
    if (!fs.existsSync(replayDir)) {
      fs.mkdirSync(replayDir, { recursive: true });
    }
    
    // Write JSON report
    const jsonPath = path.join(replayDir, `replay_smoke_${targetDate}.json`);
    fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
    console.log(`\n[Replay] JSON report: ${jsonPath}`);
    
    // Write summary
    const summary = generateSummary(report);
    const summaryPath = path.join(replayDir, `replay_smoke_${targetDate}_summary.txt`);
    fs.writeFileSync(summaryPath, summary);
    console.log(`[Replay] Summary: ${summaryPath}`);
    
    // Print summary to console
    console.log("\n" + summary);
    
    // Exit with error if errors found
    if (report.errors.length > 0) {
      console.error("\n[Replay] FAILED: Errors encountered");
      process.exit(1);
    }
    
    console.log("\n[Replay] PASSED: No errors");
    process.exit(0);
    
  } catch (error) {
    console.error("[Replay] FATAL ERROR:", error);
    process.exit(1);
  }
}

main();
