import * as alpaca from "./alpaca";
import * as indicators from "./indicators";

export interface MarketRegime {
  isUptrend: boolean;
  isBullish: boolean;
  isBearish: boolean;
  isNeutral: boolean;
  qqq: {
    price: number;
    ema9: number;
    ema20: number;
    trend: "bullish" | "bearish" | "neutral";
  };
  spy: {
    price: number;
    ema9: number;
    ema20: number;
    trend: "bullish" | "bearish" | "neutral";
  };
  recommendation: "aggressive" | "normal" | "cautious" | "avoid";
}

let cachedRegime: MarketRegime | null = null;
let lastRegimeCheck = 0;
const REGIME_CACHE_MS = 60000;

export async function checkMarketRegime(): Promise<MarketRegime> {
  const now = Date.now();
  if (cachedRegime && (now - lastRegimeCheck) < REGIME_CACHE_MS) {
    return cachedRegime;
  }
  
  try {
    const [qqqBars, spyBars] = await Promise.all([
      alpaca.getBars("QQQ", "5Min", 30),
      alpaca.getBars("SPY", "5Min", 30),
    ]);
    
    const qqqData = analyzeTrend(qqqBars);
    const spyData = analyzeTrend(spyBars);
    
    const bothBullish = qqqData.trend === "bullish" && spyData.trend === "bullish";
    const bothBearish = qqqData.trend === "bearish" && spyData.trend === "bearish";
    
    let recommendation: "aggressive" | "normal" | "cautious" | "avoid";
    if (bothBullish) {
      recommendation = "aggressive";
    } else if (bothBearish) {
      recommendation = "avoid";
    } else if (qqqData.trend === "bullish" || spyData.trend === "bullish") {
      recommendation = "normal";
    } else {
      recommendation = "cautious";
    }
    
    cachedRegime = {
      isUptrend: bothBullish,
      isBullish: qqqData.trend === "bullish" || spyData.trend === "bullish",
      isBearish: bothBearish,
      isNeutral: !bothBullish && !bothBearish,
      qqq: qqqData,
      spy: spyData,
      recommendation,
    };
    
    lastRegimeCheck = now;
    console.log(`[MarketRegime] QQQ: ${qqqData.trend}, SPY: ${spyData.trend}, Recommendation: ${recommendation}`);
    
    return cachedRegime;
  } catch (error) {
    console.error("[MarketRegime] Error checking market regime:", error);
    return {
      isUptrend: false,
      isBullish: false,
      isBearish: false,
      isNeutral: true,
      qqq: { price: 0, ema9: 0, ema20: 0, trend: "neutral" },
      spy: { price: 0, ema9: 0, ema20: 0, trend: "neutral" },
      recommendation: "cautious",
    };
  }
}

function analyzeTrend(bars: Array<{ c: number }>): {
  price: number;
  ema9: number;
  ema20: number;
  trend: "bullish" | "bearish" | "neutral";
} {
  if (bars.length < 20) {
    return { price: 0, ema9: 0, ema20: 0, trend: "neutral" };
  }
  
  const closes = bars.map(b => b.c);
  const ema9 = indicators.ema(closes, 9);
  const ema20 = indicators.ema(closes, 20);
  
  const lastClose = closes[closes.length - 1];
  const lastEma9 = ema9[ema9.length - 1];
  const lastEma20 = ema20[ema20.length - 1];
  
  let trend: "bullish" | "bearish" | "neutral";
  if (lastClose > lastEma9 && lastEma9 > lastEma20) {
    trend = "bullish";
  } else if (lastClose < lastEma9 && lastEma9 < lastEma20) {
    trend = "bearish";
  } else {
    trend = "neutral";
  }
  
  return {
    price: lastClose,
    ema9: lastEma9,
    ema20: lastEma20,
    trend,
  };
}

export async function isMarketUptrend(): Promise<boolean> {
  const regime = await checkMarketRegime();
  return regime.isUptrend;
}

export function getRecommendation(): "aggressive" | "normal" | "cautious" | "avoid" {
  return cachedRegime?.recommendation || "cautious";
}

export function clearCache(): void {
  cachedRegime = null;
  lastRegimeCheck = 0;
}
