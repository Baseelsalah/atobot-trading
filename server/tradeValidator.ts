// Trade Validator Module
// Blocks trades that don't meet quantitative requirements
// This is the key to making profitable trades instead of losing money

import * as alpaca from "./alpaca";

interface ValidationResult {
  approved: boolean;
  reason: string;
  score: number; // 0-100 confidence score
  technicals: {
    trendAlignment: boolean;
    volumeConfirmation: boolean;
    priceVsVwap: "above" | "below" | "near";
    momentumStrength: number; // -100 to 100
  };
}

interface BarData {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: string;
}

// Calculate VWAP from intraday bars
function calculateVWAP(bars: BarData[]): number {
  if (bars.length === 0) return 0;
  
  let cumulativeTPV = 0; // Typical Price * Volume
  let cumulativeVolume = 0;
  
  for (const bar of bars) {
    const typicalPrice = (bar.high + bar.low + bar.close) / 3;
    cumulativeTPV += typicalPrice * bar.volume;
    cumulativeVolume += bar.volume;
  }
  
  return cumulativeVolume > 0 ? cumulativeTPV / cumulativeVolume : 0;
}

// Calculate Average True Range for volatility
function calculateATR(bars: BarData[]): number {
  if (bars.length < 2) return 0;
  
  let atrSum = 0;
  for (let i = 1; i < bars.length; i++) {
    const high = bars[i].high;
    const low = bars[i].low;
    const prevClose = bars[i - 1].close;
    
    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    atrSum += tr;
  }
  
  return atrSum / (bars.length - 1);
}

// Calculate momentum using rate of change
function calculateMomentum(bars: BarData[], periods = 5): number {
  if (bars.length < periods + 1) return 0;
  
  const currentPrice = bars[bars.length - 1].close;
  const pastPrice = bars[bars.length - periods - 1].close;
  
  // Return percentage change
  return ((currentPrice - pastPrice) / pastPrice) * 100;
}

// Check if volume is above average
// PRO: Increased threshold from 1.2 to 1.5 (150% of average) for stronger confirmation
function hasVolumeConfirmation(bars: BarData[], threshold = 1.5): boolean {
  if (bars.length < 10) return false;
  
  const avgVolume = bars.slice(0, -1).reduce((sum, b) => sum + b.volume, 0) / (bars.length - 1);
  const currentVolume = bars[bars.length - 1].volume;
  
  return currentVolume > avgVolume * threshold;
}

// Detect trend direction from bars
function detectTrend(bars: BarData[]): "bullish" | "bearish" | "sideways" {
  if (bars.length < 5) return "sideways";
  
  const recentBars = bars.slice(-5);
  const firstClose = recentBars[0].close;
  const lastClose = recentBars[recentBars.length - 1].close;
  const changePercent = ((lastClose - firstClose) / firstClose) * 100;
  
  // Count higher highs and higher lows for bullish
  let higherHighs = 0;
  let lowerLows = 0;
  
  for (let i = 1; i < recentBars.length; i++) {
    if (recentBars[i].high > recentBars[i - 1].high) higherHighs++;
    if (recentBars[i].low < recentBars[i - 1].low) lowerLows++;
  }
  
  if (changePercent > 0.3 && higherHighs >= 2) return "bullish";
  if (changePercent < -0.3 && lowerLows >= 2) return "bearish";
  return "sideways";
}

// Stock preferences from brain (populated at runtime)
let stockPreferences: Record<string, number> = {};

export function setStockPreferences(prefs: Record<string, number>): void {
  stockPreferences = prefs;
}

// Main validation function
export async function validateTrade(
  symbol: string,
  side: "buy" | "sell",
  strategyType: "scalp" | "dip" | "breakout"
): Promise<ValidationResult> {
  console.log(`[TradeValidator] Validating ${side} ${symbol} (strategy: ${strategyType})...`);
  
  // BLACKLIST CHECK: Block trades on blacklisted symbols
  const stockPref = stockPreferences[symbol] || 50;
  if (stockPref < 20 && side === "buy") {
    console.log(`[TradeValidator] BLOCKED: ${symbol} is blacklisted (preference: ${stockPref})`);
    return {
      approved: false,
      reason: `Symbol ${symbol} is blacklisted due to poor performance (preference: ${stockPref}/100)`,
      score: 0,
      technicals: {
        trendAlignment: false,
        volumeConfirmation: false,
        priceVsVwap: "near",
        momentumStrength: 0,
      },
    };
  }
  
  try {
    // Get 5-minute bars for intraday analysis (with warm-start for reliability)
    const bars5minResult = await alpaca.getBarsSafe(symbol, "5Min", 10);
    const bars5min = bars5minResult.bars;
    // Get 1-minute bars for short-term momentum (with warm-start for reliability)
    const bars1minResult = await alpaca.getBarsSafe(symbol, "1Min", 5);
    const bars1min = bars1minResult.bars;
    // Get current price
    const quote = await alpaca.getLatestQuote(symbol);
    
    // FAIL-CLOSED: Block trades if we can't get current price
    if (!quote.price || quote.price <= 0) {
      console.log(`[TradeValidator] BLOCKED: Cannot get current price for ${symbol}`);
      return {
        approved: false,
        reason: "FAIL_CLOSED: Cannot get current price - blocking trade",
        score: 0,
        technicals: {
          trendAlignment: false,
          volumeConfirmation: false,
          priceVsVwap: "near",
          momentumStrength: 0,
        },
      };
    }
    
    // FAIL-CLOSED: Require minimum bar history (at least 10 bars for proper analysis)
    const MIN_BARS_REQUIRED = 10;
    if (bars5min.length < MIN_BARS_REQUIRED) {
      console.log(`[TradeValidator] BLOCKED: Insufficient bar data for ${symbol} (${bars5min.length}/${MIN_BARS_REQUIRED} bars)`);
      return {
        approved: false,
        reason: `FAIL_CLOSED: Insufficient bar data (${bars5min.length}/${MIN_BARS_REQUIRED} required)`,
        score: 0,
        technicals: {
          trendAlignment: false,
          volumeConfirmation: false,
          priceVsVwap: "near",
          momentumStrength: 0,
        },
      };
    }
    
    // Convert Alpaca bars to our format
    const formattedBars: BarData[] = bars5min.map(b => ({
      open: b.o,
      high: b.h,
      low: b.l,
      close: b.c,
      volume: b.v,
      timestamp: b.t,
    }));
    
    // Calculate technicals
    const vwap = calculateVWAP(formattedBars);
    const atr = calculateATR(formattedBars);
    const momentum = calculateMomentum(formattedBars);
    const trend = detectTrend(formattedBars);
    const volumeOk = hasVolumeConfirmation(formattedBars);
    
    const currentPrice = quote.price;
    const priceVsVwap = currentPrice > vwap * 1.002 ? "above" :
                        currentPrice < vwap * 0.998 ? "below" : "near";
    
    // Determine if trend aligns with trade direction
    const trendAlignment = 
      (side === "buy" && (trend === "bullish" || strategyType === "dip")) ||
      (side === "sell" && (trend === "bearish" || trend === "sideways"));
    
    // Score the trade (0-100)
    // FAIL-CLOSED: Start conservative, require strong signals
    let score = 40; // Start conservative - only trade with good signals
    
    // Add stock preference bonus (stocks Autopilot likes get a boost)
    if (stockPref >= 70) score += 10;
    else if (stockPref >= 50) score += 5;
    
    // Trend alignment is helpful but not required (+/- 10 points, reduced from 20)
    if (trendAlignment) score += 10;
    else score -= 5; // Reduced penalty - sideways can still work
    
    // Volume confirmation (+/- 15 points) - PRO: Increased importance
    if (volumeOk) score += 15;
    else score -= 10; // PRO: Increased penalty for weak volume
    
    // Price vs VWAP alignment (+/- 10 points)
    if (side === "buy" && priceVsVwap === "below") score += 10;
    else if (side === "buy" && priceVsVwap === "above") score -= 5; // Reduced penalty
    else if (side === "sell" && priceVsVwap === "above") score += 10;
    else if (side === "sell" && priceVsVwap === "below") score -= 5;
    
    // Momentum alignment (+/- 5 points - less important for day trading)
    if (side === "buy" && momentum > 0.2) score += 5;
    else if (side === "buy" && momentum < -1.0) score -= 10; // Only penalize severe downtrends
    else if (side === "sell" && momentum < -0.2) score += 5;
    
    // Strategy-specific adjustments (reduced penalties)
    if (strategyType === "scalp") {
      if (atr > currentPrice * 0.03) score -= 5; // Only penalize extreme volatility
    } else if (strategyType === "dip") {
      if (priceVsVwap !== "below") score -= 5; // Reduced from 20
      if (momentum < -2) score -= 5; // Reduced penalty
    } else if (strategyType === "breakout") {
      // PRO: MANDATORY volume for breakouts (blocks fake breakouts)
      if (!volumeOk) score -= 25; // Increased from 5 - breakouts NEED volume
      if (Math.abs(momentum) < 0.3) score -= 5; // Reduced from 15
    }
    
    // Clamp score
    score = Math.max(0, Math.min(100, score));
    
    // PRO: Raised threshold from 35 to 45 - filter weak setups
    // Volume confirmation is now more heavily weighted
    const APPROVAL_THRESHOLD = 45;
    const approved = score >= APPROVAL_THRESHOLD;
    
    // Build flags array (always computed, even for approved trades)
    const flags: string[] = [];
    if (!trendAlignment) flags.push("trend misaligned");
    if (!volumeOk) flags.push("weak volume");
    if (side === "buy" && priceVsVwap === "above") flags.push("price above VWAP");
    if (side === "buy" && momentum < -0.5) flags.push("falling momentum");
    
    let reason: string;
    if (approved) {
      reason = `Trade approved (score: ${score}): ${trend} trend, ${priceVsVwap} VWAP, ${volumeOk ? "volume confirmed" : "low volume"}`;
      console.log(`[VALIDATION_PASS] ${symbol} | score=${score} threshold=${APPROVAL_THRESHOLD} | flags=${flags.length ? flags.join(", ") : "none"}`);
    } else {
      reason = `Trade blocked (score: ${score}): ${flags.join(", ")}`;
      console.log(`[VALIDATION_BLOCK] ${symbol} | score=${score} threshold=${APPROVAL_THRESHOLD} | flags=${flags.join(", ")}`);
    }
    
    // Keep existing line for backward compatibility (remove after 2026-02-07)
    console.log(`[TradeValidator] ${symbol}: ${reason}`);
    
    return {
      approved,
      reason,
      score,
      technicals: {
        trendAlignment,
        volumeConfirmation: volumeOk,
        priceVsVwap,
        momentumStrength: momentum,
      },
    };
  } catch (error) {
    console.error(`[TradeValidator] Error validating ${symbol}:`, error);
    return {
      approved: false,
      reason: `Validation error: ${error}`,
      score: 0,
      technicals: {
        trendAlignment: false,
        volumeConfirmation: false,
        priceVsVwap: "near",
        momentumStrength: 0,
      },
    };
  }
}

// Check overall market conditions before trading
export async function checkMarketConditions(): Promise<{
  favorable: boolean;
  reason: string;
  vixLevel: "low" | "normal" | "high";
  spyTrend: "bullish" | "bearish" | "sideways";
}> {
  console.log("[TradeValidator] Checking market conditions...");
  
  try {
    // Check SPY for overall market trend (with warm-start for reliability)
    const spyBarResult = await alpaca.getBarsSafe("SPY", "5Min", 10);
    const spyBars = spyBarResult.bars;
    
    // If we can't get SPY bars, still allow trading - don't block
    // The market is open, we should trade
    if (spyBars.length < 3) {
      console.log(`[TradeValidator] Limited SPY data (${spyBars.length} bars), allowing trades anyway`);
      return {
        favorable: true, // Allow trading even with limited data
        reason: "Limited market data - proceeding with trades",
        vixLevel: "normal",
        spyTrend: "sideways",
      };
    }
    
    const formattedBars: BarData[] = spyBars.map(b => ({
      open: b.o,
      high: b.h,
      low: b.l,
      close: b.c,
      volume: b.v,
      timestamp: b.t,
    }));
    
    const spyTrend = detectTrend(formattedBars);
    const spyMomentum = calculateMomentum(formattedBars);
    const spyATR = calculateATR(formattedBars);
    const spyPrice = formattedBars[formattedBars.length - 1].close;
    
    // High ATR relative to price = high volatility (VIX proxy)
    const volatilityPercent = (spyATR / spyPrice) * 100;
    const vixLevel = volatilityPercent > 0.5 ? "high" :
                     volatilityPercent < 0.2 ? "low" : "normal";
    
    // Trading conditions - BE PERMISSIVE, we need to trade!
    // Only block trading in extreme conditions
    let favorable = true;
    const notes: string[] = [];
    
    // Only block in EXTREME volatility (>0.8% ATR) - otherwise trade through it
    if (volatilityPercent > 0.8) {
      favorable = false;
      notes.push("extreme volatility - waiting for calmer conditions");
    }
    
    // Note sideways markets but DON'T block - we can still scalp
    if (spyTrend === "sideways" && Math.abs(spyMomentum) < 0.15) {
      notes.push("sideways market (scalp mode)");
      // Don't set favorable=false - we can still trade range-bound stocks
    }
    
    // Add context about market state
    if (spyTrend === "bullish") {
      notes.push("bullish trend - favor longs");
    } else if (spyTrend === "bearish") {
      notes.push("bearish trend - be cautious with longs");
    }
    
    const reason = favorable
      ? `Market tradeable: SPY ${spyTrend}, volatility ${vixLevel}. ${notes.join(". ")}`
      : `Market conditions risky: ${notes.join(", ")}`;
    
    console.log(`[TradeValidator] ${reason}`);
    
    return {
      favorable,
      reason,
      vixLevel,
      spyTrend,
    };
  } catch (error) {
    console.error("[TradeValidator] Market conditions check error:", error);
    return {
      favorable: true, // Default to trading if we can't check
      reason: "Unable to check market conditions",
      vixLevel: "normal",
      spyTrend: "sideways",
    };
  }
}

// Calculate optimal position size based on ATR and risk
export async function calculatePositionSize(
  symbol: string,
  accountEquity: number,
  riskPerTrade: number = 0.01 // 1% risk per trade
): Promise<{
  shares: number;
  stopDistance: number;
  positionValue: number;
}> {
  try {
    const barResult = await alpaca.getBarsSafe(symbol, "5Min", 5);
    const bars = barResult.bars;
    const quote = await alpaca.getLatestQuote(symbol);
    
    if (bars.length < 5 || !quote.price) {
      // Default sizing
      const defaultPositionSize = accountEquity * 0.02; // 2% of account
      const shares = Math.floor(defaultPositionSize / quote.price);
      return {
        shares: Math.max(1, shares),
        stopDistance: quote.price * 0.015, // 1.5% default
        positionValue: shares * quote.price,
      };
    }
    
    const formattedBars: BarData[] = bars.map(b => ({
      open: b.o,
      high: b.h,
      low: b.l,
      close: b.c,
      volume: b.v,
      timestamp: b.t,
    }));
    
    const atr = calculateATR(formattedBars);
    const currentPrice = quote.price;
    
    // Stop loss at 1.5x ATR
    const stopDistance = atr * 1.5;
    
    // Risk amount in dollars
    const riskAmount = accountEquity * riskPerTrade;
    
    // Shares = risk amount / stop distance
    const shares = Math.floor(riskAmount / stopDistance);
    
    // Cap at 5% of account value
    const maxPositionValue = accountEquity * 0.05;
    const maxShares = Math.floor(maxPositionValue / currentPrice);
    
    const finalShares = Math.max(1, Math.min(shares, maxShares));
    
    console.log(`[TradeValidator] Position size for ${symbol}: ${finalShares} shares (ATR: $${atr.toFixed(2)}, stop: $${stopDistance.toFixed(2)})`);
    
    return {
      shares: finalShares,
      stopDistance,
      positionValue: finalShares * currentPrice,
    };
  } catch (error) {
    console.error(`[TradeValidator] Position sizing error for ${symbol}:`, error);
    // Return safe default
    return {
      shares: 1,
      stopDistance: 0,
      positionValue: 0,
    };
  }
}
