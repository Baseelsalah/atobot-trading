import OpenAI from "openai";

const DISABLE_OPENAI = process.env.DISABLE_OPENAI === "1";
if (DISABLE_OPENAI) {
  console.log("[OpenAI] Disabled via DISABLE_OPENAI=1 (skipping AI calls)");
}

const OPENAI_BASE_URL = process.env.AI_INTEGRATIONS_OPENAI_BASE_URL;
const OPENAI_API_KEY = process.env.AI_INTEGRATIONS_OPENAI_API_KEY;
let openai: OpenAI | null = null;

function getOpenAI(): OpenAI {
  if (DISABLE_OPENAI) {
    throw new Error("OpenAI disabled");
  }
  if (!OPENAI_API_KEY) {
    throw new Error("Missing AI_INTEGRATIONS_OPENAI_API_KEY");
  }
  if (!openai) {
    openai = new OpenAI({
      baseURL: OPENAI_BASE_URL,
      apiKey: OPENAI_API_KEY,
    });
  }
  return openai;
}

// the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
const MODEL = "gpt-5";

interface MarketAnalysis {
  summary: string;
  sentiment: "bullish" | "bearish" | "neutral";
  confidence: number;
  symbols: string[];
  recommendations: TradeRecommendation[];
  technicalIndicators: string[];
  newsFactors: string[];
}

interface TradeRecommendation {
  symbol: string;
  action: "buy" | "sell" | "hold";
  reason: string;
  confidence: number;
  riskLevel: "low" | "medium" | "high";
  targetPrice?: number;
  stopLoss?: number;
}

interface StockAnalysis {
  symbol: string;
  currentPrice: number;
  priceHistory: number[];
  analysis: string;
  recommendation: TradeRecommendation;
}

export async function analyzeMarket(
  watchlist: string[],
  portfolioValue: number,
  currentPositions: { symbol: string; qty: number; unrealizedPL: number }[],
  riskSettings: { maxPositionSize: number; stopLossPercent: number; takeProfitPercent: number }
): Promise<MarketAnalysis> {
  if (DISABLE_OPENAI) {
    return {
      summary: "OpenAI disabled",
      sentiment: "neutral",
      confidence: 0,
      symbols: watchlist,
      recommendations: [],
      technicalIndicators: [],
      newsFactors: ["OpenAI disabled"],
    };
  }
  const positionsSummary = currentPositions.length > 0
    ? currentPositions.map((p) => `${p.symbol}: ${p.qty} shares, P/L: $${p.unrealizedPL.toFixed(2)}`).join(", ")
    : "No current positions";

  const prompt = `You are an expert day trading analyst. Analyze the current market conditions and provide trading recommendations.

Current Portfolio:
- Total Value: $${portfolioValue.toFixed(2)}
- Current Positions: ${positionsSummary}

Risk Settings:
- Max Position Size: $${riskSettings.maxPositionSize}
- Stop Loss: ${riskSettings.stopLossPercent}%
- Take Profit: ${riskSettings.takeProfitPercent}%

Watchlist: ${watchlist.join(", ")}

Analyze the market and provide:
1. Overall market sentiment analysis
2. Specific recommendations for stocks in the watchlist
3. Risk assessment for each recommendation
4. Technical indicators to watch

Respond in JSON format:
{
  "summary": "Brief market overview",
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": 0-100,
  "symbols": ["symbols analyzed"],
  "recommendations": [
    {
      "symbol": "TICKER",
      "action": "buy" | "sell" | "hold",
      "reason": "Why this recommendation",
      "confidence": 0-100,
      "riskLevel": "low" | "medium" | "high",
      "targetPrice": optional number,
      "stopLoss": optional number
    }
  ],
  "technicalIndicators": ["indicators to watch"],
  "newsFactors": ["relevant news/events"]
}`;

  try {
    const response = await getOpenAI().chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      max_completion_tokens: 2048,
    });

    const content = response.choices[0]?.message?.content || "{}";
    console.log("[OpenAI] Raw response:", content.substring(0, 500));
    const rawAnalysis = JSON.parse(content);
    
    // Validate and provide defaults for all required fields
    const analysis: MarketAnalysis = {
      summary: rawAnalysis.summary || "Market analysis completed",
      sentiment: rawAnalysis.sentiment || "neutral",
      confidence: typeof rawAnalysis.confidence === "number" ? rawAnalysis.confidence : 50,
      symbols: Array.isArray(rawAnalysis.symbols) ? rawAnalysis.symbols : watchlist,
      recommendations: Array.isArray(rawAnalysis.recommendations) ? rawAnalysis.recommendations.map((rec: any) => ({
        symbol: rec.symbol || "UNKNOWN",
        action: rec.action || "hold",
        reason: rec.reason || "AI recommendation",
        confidence: typeof rec.confidence === "number" ? rec.confidence : 50,
        riskLevel: rec.riskLevel || "medium",
        targetPrice: rec.targetPrice,
        stopLoss: rec.stopLoss,
      })) : [],
      technicalIndicators: Array.isArray(rawAnalysis.technicalIndicators) ? rawAnalysis.technicalIndicators : [],
      newsFactors: Array.isArray(rawAnalysis.newsFactors) ? rawAnalysis.newsFactors : [],
    };
    
    console.log("[OpenAI] Parsed analysis - sentiment:", analysis.sentiment, "confidence:", analysis.confidence, "recommendations:", analysis.recommendations.length);
    
    // FAIL-CLOSED: If no recommendations, that's a valid outcome - do NOT generate fallback trades
    if (analysis.recommendations.length === 0) {
      console.log("[OpenAI] No actionable recommendations from AI - this is normal, not generating forced trades");
    }
    
    return analysis;
  } catch (error) {
    console.error("[OpenAI] Market analysis failed:", error);
    // FAIL-CLOSED: On API error, return empty recommendations - do NOT force trades
    console.log("[OpenAI] API error - returning empty recommendations (fail-closed behavior)");
    return {
      summary: "Market analysis unavailable due to API error",
      sentiment: "neutral",
      confidence: 0,
      symbols: watchlist,
      recommendations: [], // EMPTY - no forced trades on error
      technicalIndicators: [],
      newsFactors: ["API error - analysis unavailable"],
    };
  }
}

export async function analyzeStock(
  symbol: string,
  priceHistory: number[],
  currentPrice: number,
  portfolioContext: string
): Promise<StockAnalysis> {
  if (DISABLE_OPENAI) {
    return {
      symbol,
      currentPrice,
      priceHistory,
      analysis: "OpenAI disabled",
      recommendation: {
        symbol,
        action: "hold",
        reason: "OpenAI disabled",
        confidence: 0,
        riskLevel: "low",
      },
    };
  }
  const priceChange = priceHistory.length > 1
    ? ((currentPrice - priceHistory[0]) / priceHistory[0] * 100).toFixed(2)
    : "0";

  const prompt = `Analyze ${symbol} stock for day trading opportunity.

Current Price: $${currentPrice.toFixed(2)}
Recent Price History: ${priceHistory.slice(-10).map((p) => `$${p.toFixed(2)}`).join(", ")}
Price Change: ${priceChange}%

Portfolio Context: ${portfolioContext}

Provide a detailed analysis and trading recommendation in JSON format:
{
  "symbol": "${symbol}",
  "currentPrice": ${currentPrice},
  "priceHistory": [numbers],
  "analysis": "Detailed technical and fundamental analysis",
  "recommendation": {
    "symbol": "${symbol}",
    "action": "buy" | "sell" | "hold",
    "reason": "Specific reason for recommendation",
    "confidence": 0-100,
    "riskLevel": "low" | "medium" | "high",
    "targetPrice": optional target price,
    "stopLoss": optional stop loss price
  }
}`;

  try {
    const response = await getOpenAI().chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      max_completion_tokens: 1024,
    });

    const content = response.choices[0]?.message?.content || "{}";
    return JSON.parse(content) as StockAnalysis;
  } catch (error) {
    console.error(`Stock analysis for ${symbol} failed:`, error);
    return {
      symbol,
      currentPrice,
      priceHistory,
      analysis: "Unable to analyze stock at this time",
      recommendation: {
        symbol,
        action: "hold",
        reason: "Analysis unavailable",
        confidence: 0,
        riskLevel: "high",
      },
    };
  }
}

export async function generateTradingStrategy(
  marketConditions: string,
  riskTolerance: "conservative" | "moderate" | "aggressive",
  availableCapital: number
): Promise<string> {
  const prompt = `As an expert day trader, develop a trading strategy based on:

Market Conditions: ${marketConditions}
Risk Tolerance: ${riskTolerance}
Available Capital: $${availableCapital.toFixed(2)}

Provide a concise trading strategy including:
1. Entry and exit criteria
2. Position sizing recommendations
3. Risk management rules
4. Time-based considerations for day trading`;

  try {
    const response = await getOpenAI().chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      max_completion_tokens: 1024,
    });

    return response.choices[0]?.message?.content || "Unable to generate strategy";
  } catch (error) {
    console.error("Strategy generation failed:", error);
    return "Unable to generate trading strategy at this time";
  }
}
