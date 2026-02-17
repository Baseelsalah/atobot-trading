import * as alpaca from "./alpaca";
import { storage } from "./storage";
import * as dayTraderConfig from "./dayTraderConfig";
import * as timeGuard from "./tradingTimeGuard";
import * as leaderLock from "./leaderLock";

interface VolatilityData {
  symbol: string;
  atr: number;
  atrPercent: number;
  dailyRange: number;
  volatilityLevel: "low" | "medium" | "high" | "extreme";
  lastUpdated: Date;
}

interface PositionSizeResult {
  symbol: string;
  recommendedShares: number;
  recommendedValue: number;
  maxShares: number;
  riskPerShare: number;
  volatilityAdjustment: number;
  stopLossPrice: number;
  takeProfitPrice: number;
  riskRewardRatio: number;
  reasoning: string;
}

interface RiskMetrics {
  portfolioVolatility: number;
  portfolioHeatLevel: number;
  maxDrawdown: number;
  currentDrawdown: number;
  riskCapacity: number;
  correlationRisk: number;
}

const volatilityCache: Map<string, VolatilityData> = new Map();
const CACHE_DURATION_MS = 5 * 60 * 1000;

export async function calculateVolatility(symbol: string): Promise<VolatilityData> {
  const cached = volatilityCache.get(symbol);
  if (cached && Date.now() - cached.lastUpdated.getTime() < CACHE_DURATION_MS) {
    return cached;
  }

  try {
    const barResult = await alpaca.getBarsSafe(symbol, "1Day", 14);
    const bars = barResult.bars;
    
    if (!bars || bars.length < 14) {
      return {
        symbol,
        atr: 0,
        atrPercent: 5,
        dailyRange: 0,
        volatilityLevel: "medium",
        lastUpdated: new Date(),
      };
    }

    let trSum = 0;
    for (let i = 1; i < bars.length; i++) {
      const high = bars[i].h;
      const low = bars[i].l;
      const prevClose = bars[i - 1].c;
      
      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      trSum += tr;
    }
    
    const atr = trSum / (bars.length - 1);
    const currentPrice = bars[bars.length - 1].c;
    const atrPercent = (atr / currentPrice) * 100;

    const dailyRanges = bars.map((b: { h: number; l: number }) => ((b.h - b.l) / b.l) * 100);
    const avgDailyRange = dailyRanges.reduce((a: number, b: number) => a + b, 0) / dailyRanges.length;

    let volatilityLevel: "low" | "medium" | "high" | "extreme";
    if (atrPercent < 1.5) {
      volatilityLevel = "low";
    } else if (atrPercent < 3) {
      volatilityLevel = "medium";
    } else if (atrPercent < 5) {
      volatilityLevel = "high";
    } else {
      volatilityLevel = "extreme";
    }

    const volatilityData: VolatilityData = {
      symbol,
      atr,
      atrPercent,
      dailyRange: avgDailyRange,
      volatilityLevel,
      lastUpdated: new Date(),
    };

    volatilityCache.set(symbol, volatilityData);
    return volatilityData;
  } catch (error) {
    console.error(`[RiskManager] Failed to calculate volatility for ${symbol}:`, error);
    return {
      symbol,
      atr: 0,
      atrPercent: 3,
      dailyRange: 2,
      volatilityLevel: "medium",
      lastUpdated: new Date(),
    };
  }
}

export async function calculateDynamicPositionSize(
  symbol: string,
  currentPrice: number,
  side: "buy" | "sell",
  portfolioValue: number,
  existingPositionsCount: number
): Promise<PositionSizeResult> {
  const settings = await storage.getSettings();
  
  if (!alpaca.isConfigured()) {
    const positionSize = settings.maxPositionSize || 1000;
    const quantity = Math.max(1, Math.floor(positionSize / currentPrice));
    const stopLossPrice = side === "buy" ? currentPrice * 0.98 : currentPrice * 1.02;
    const takeProfitPrice = side === "buy" ? currentPrice * 1.05 : currentPrice * 0.95;
    
    return {
      symbol,
      recommendedShares: quantity,
      recommendedValue: quantity * currentPrice,
      maxShares: quantity,
      riskPerShare: Math.abs(currentPrice - stopLossPrice),
      volatilityAdjustment: 1.0,
      stopLossPrice,
      takeProfitPrice,
      riskRewardRatio: 2.5,
      reasoning: "Default position sizing (Alpaca not configured for volatility data).",
    };
  }
  
  const volatility = await calculateVolatility(symbol);
  
  const baseRiskPercent = 1;
  const maxRiskPercent = 2;
  const maxPositionPercent = 700; // 700% of portfolio per position — balanced leverage (backtested: PF 3.10, $8.5K/month, -7.3% max DD)
  
  let volatilityMultiplier: number;
  switch (volatility.volatilityLevel) {
    case "low":
      volatilityMultiplier = 1.5;
      break;
    case "medium":
      volatilityMultiplier = 1.0;
      break;
    case "high":
      volatilityMultiplier = 0.6;
      break;
    case "extreme":
      volatilityMultiplier = 0.3;
      break;
  }

  const concentrationFactor = Math.max(0.5, 1 - (existingPositionsCount * 0.1));

  const atrMultiplier = 2;
  const stopLossDistance = volatility.atr > 0 ? volatility.atr * atrMultiplier : currentPrice * 0.02;
  const stopLossPrice = side === "buy" 
    ? currentPrice - stopLossDistance 
    : currentPrice + stopLossDistance;

  const takeProfitMultiplier = 3;
  const takeProfitDistance = stopLossDistance * takeProfitMultiplier;
  const takeProfitPrice = side === "buy"
    ? currentPrice + takeProfitDistance
    : currentPrice - takeProfitDistance;

  const riskRewardRatio = takeProfitDistance / stopLossDistance;

  const dollarRiskPerTrade = portfolioValue * (baseRiskPercent / 100);
  const adjustedRisk = dollarRiskPerTrade * volatilityMultiplier * concentrationFactor;

  const riskPerShare = Math.abs(currentPrice - stopLossPrice);
  const recommendedShares = riskPerShare > 0 
    ? Math.floor(adjustedRisk / riskPerShare)
    : Math.floor((settings.maxPositionSize || 1000) / currentPrice);

  const maxPositionValue = portfolioValue * (maxPositionPercent / 100);
  const settingsMaxValue = settings.maxPositionSize || 1000;
  const effectiveMaxValue = Math.min(maxPositionValue, settingsMaxValue);
  const maxShares = Math.floor(effectiveMaxValue / currentPrice);

  const finalShares = Math.max(1, Math.min(recommendedShares, maxShares));
  const recommendedValue = finalShares * currentPrice;

  const reasoning = `Volatility: ${volatility.volatilityLevel} (ATR: ${volatility.atrPercent.toFixed(2)}%). ` +
    `Position sized at ${((recommendedValue / portfolioValue) * 100).toFixed(1)}% of portfolio. ` +
    `Risk/Reward: 1:${riskRewardRatio.toFixed(1)}. ` +
    `Stop: $${stopLossPrice.toFixed(2)}, Target: $${takeProfitPrice.toFixed(2)}.`;

  return {
    symbol,
    recommendedShares: finalShares,
    recommendedValue,
    maxShares,
    riskPerShare,
    volatilityAdjustment: volatilityMultiplier,
    stopLossPrice,
    takeProfitPrice,
    riskRewardRatio,
    reasoning,
  };
}

export async function calculatePortfolioRisk(): Promise<RiskMetrics> {
  try {
    const positions = await storage.getPositions();
    const trades = await storage.getTrades();
    
    let portfolioValue = 100000;
    let startingValue = 100000;
    
    if (alpaca.isConfigured()) {
      try {
        const account = await alpaca.getAccount();
        portfolioValue = parseFloat(account.equity);
        startingValue = parseFloat(account.last_equity);
      } catch (e) {
        const positionValue = positions.reduce((sum, p) => sum + p.marketValue, 0);
        portfolioValue = 100000 + positionValue;
        startingValue = 100000;
      }
    } else {
      const positionValue = positions.reduce((sum, p) => sum + p.marketValue, 0);
      portfolioValue = 100000 + positionValue;
      startingValue = 100000;
    }

    let portfolioVolatility = 0;
    for (const position of positions) {
      const volatility = await calculateVolatility(position.symbol);
      const positionWeight = position.marketValue / portfolioValue;
      portfolioVolatility += volatility.atrPercent * positionWeight;
    }

    const totalExposure = positions.reduce((sum, p) => sum + p.marketValue, 0);
    const portfolioHeatLevel = (totalExposure / portfolioValue) * 100;

    const recentTrades = trades.filter(t => {
      if (!t.timestamp) return false;
      const tradeDate = new Date(t.timestamp);
      const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
      return tradeDate >= weekAgo;
    });

    let runningPL = 0;
    let maxDrawdown = 0;
    let peakValue = startingValue;

    for (const trade of recentTrades) {
      if (trade.side === "sell") {
        runningPL += trade.totalValue;
      } else {
        runningPL -= trade.totalValue;
      }
      
      const currentValue = startingValue + runningPL;
      if (currentValue > peakValue) {
        peakValue = currentValue;
      }
      const drawdown = ((peakValue - currentValue) / peakValue) * 100;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }

    const currentDrawdown = ((peakValue - portfolioValue) / peakValue) * 100;

    let riskCapacity = 100;
    riskCapacity -= portfolioHeatLevel * 0.5;
    riskCapacity -= currentDrawdown * 2;
    riskCapacity -= portfolioVolatility * 5;
    riskCapacity = Math.max(0, Math.min(100, riskCapacity));

    const uniqueSymbols = Array.from(new Set(positions.map(p => p.symbol)));
    const correlationRisk = uniqueSymbols.length < 3 ? 80 : 
                            uniqueSymbols.length < 5 ? 50 : 
                            uniqueSymbols.length < 8 ? 30 : 20;

    return {
      portfolioVolatility,
      portfolioHeatLevel,
      maxDrawdown,
      currentDrawdown: Math.max(0, currentDrawdown),
      riskCapacity,
      correlationRisk,
    };
  } catch (error) {
    console.error("[RiskManager] Failed to calculate portfolio risk:", error);
    return {
      portfolioVolatility: 3,
      portfolioHeatLevel: 50,
      maxDrawdown: 0,
      currentDrawdown: 0,
      riskCapacity: 50,
      correlationRisk: 50,
    };
  }
}

export async function shouldAllowTrade(
  symbol: string,
  side: "buy" | "sell",
  proposedValue: number
): Promise<{ allowed: boolean; reason: string }> {
  try {
    const settings = await storage.getSettings();
    
    // OPS-PROD-LOCK-1: Hard enforcement - only leader instance can enter trades
    if (side === "buy" && leaderLock.shouldBlockEntry()) {
      const status = leaderLock.getLeaderStatus();
      console.log(`[RiskManager] CRITICAL: Trade blocked - not leader instance (bootId: ${status.bootId}, error: ${status.lastError})`);
      return {
        allowed: false,
        reason: `BLOCKED: Not leader instance - cannot enter trades`,
      };
    }
    
    // FORT KNOX: Check symbol is in allowed universe (baseline mode restricts entry further)
    if (side === "buy") {
      const universeCheck = dayTraderConfig.isSymbolAllowedForEntry(symbol);
      if (!universeCheck.allowed) {
        return {
          allowed: false,
          reason: `SKIP: ${symbol} - ${universeCheck.reason}`,
        };
      }
    } else if (!dayTraderConfig.isSymbolAllowed(symbol)) {
      return {
        allowed: false,
        reason: `SKIP: ${symbol} not in allowed universe`,
      };
    }
    
    // FORT KNOX: For BUY orders, check if we're within entry window (9:35-11:35 AM ET)
    if (side === "buy") {
      const tradingStatus = timeGuard.getTradingStatus();
      if (!tradingStatus.canEnterNewPositions) {
        return {
          allowed: false,
          reason: `SKIP: Outside entry window (11:35 AM ET cutoff) - ${tradingStatus.reason}`,
        };
      }
      
      // FORT KNOX: Check P&L kill threshold (-$500 or +$500)
      if (dayTraderConfig.isDailyKillThresholdHit()) {
        const status = dayTraderConfig.getDayTraderStatus();
        const threshold = status.lossLimitHit ? "loss" : "profit";
        return {
          allowed: false,
          reason: `SKIP: P&L kill threshold hit ($${status.dailyPnL.toFixed(0)} ${threshold}) - no new entries`,
        };
      }
    }
    
    if (!alpaca.isConfigured()) {
      return {
        allowed: true,
        reason: "Risk checks bypassed - Alpaca not configured.",
      };
    }
    
    const riskMetrics = await calculatePortfolioRisk();

    if (riskMetrics.riskCapacity < 20) {
      return {
        allowed: false,
        reason: `Risk capacity too low (${riskMetrics.riskCapacity.toFixed(0)}%). Portfolio needs to stabilize before new trades.`,
      };
    }

    if (riskMetrics.currentDrawdown > 5) {
      return {
        allowed: false,
        reason: `Current drawdown (${riskMetrics.currentDrawdown.toFixed(1)}%) exceeds safety threshold. Wait for recovery.`,
      };
    }

    if (riskMetrics.portfolioHeatLevel > 1500) {
      return {
        allowed: false,
        reason: `Portfolio heat level too high (${riskMetrics.portfolioHeatLevel.toFixed(0)}%). Close some positions first.`,
      };
    }

    const volatility = await calculateVolatility(symbol);
    if (volatility.volatilityLevel === "extreme") {
      return {
        allowed: false,
        reason: `${symbol} volatility is extreme (ATR: ${volatility.atrPercent.toFixed(1)}%). Too risky for new positions.`,
      };
    }

    return {
      allowed: true,
      reason: "Trade passes all risk checks.",
    };
  } catch (error) {
    console.error("[RiskManager] Error checking trade permission:", error);
    return {
      allowed: true,
      reason: "Risk check failed, allowing trade with caution.",
    };
  }
}

export async function getRiskDashboardData(): Promise<{
  metrics: RiskMetrics;
  volatilityBySymbol: Record<string, VolatilityData>;
  recommendations: string[];
}> {
  const positions = await storage.getPositions();
  const metrics = await calculatePortfolioRisk();
  
  const volatilityBySymbol: Record<string, VolatilityData> = {};
  for (const position of positions) {
    volatilityBySymbol[position.symbol] = await calculateVolatility(position.symbol);
  }

  const recommendations: string[] = [];

  if (metrics.portfolioHeatLevel > 70) {
    recommendations.push("Consider reducing position sizes - portfolio exposure is high.");
  }
  
  if (metrics.correlationRisk > 60) {
    recommendations.push("Diversify holdings - too few unique positions increases risk.");
  }
  
  if (metrics.portfolioVolatility > 4) {
    recommendations.push("Portfolio volatility is elevated - consider adding stable assets.");
  }
  
  if (metrics.currentDrawdown > 3) {
    recommendations.push("Currently in drawdown - focus on risk management over new entries.");
  }
  
  if (metrics.riskCapacity > 80) {
    recommendations.push("Risk capacity is healthy - can consider new opportunities.");
  }

  const extremeVolatility = Object.values(volatilityBySymbol).filter(v => v.volatilityLevel === "extreme");
  if (extremeVolatility.length > 0) {
    recommendations.push(`High volatility alert: ${extremeVolatility.map(v => v.symbol).join(", ")}`);
  }

  return {
    metrics,
    volatilityBySymbol,
    recommendations,
  };
}
