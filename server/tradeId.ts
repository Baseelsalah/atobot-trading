import { getEasternTime } from "./timezone";

export interface TradeIdInfo {
  tradeId: string;
  symbol: string;
  strategy: string;
  side: string;
  timestamp: string;
  tier: number;
}

export function generateTradeId(
  symbol: string,
  strategy: string,
  side: "buy" | "sell",
  tier: number = 2
): string {
  const et = getEasternTime();
  const dateStr = et.dateString.replace(/-/g, "");
  const hourStr = String(et.hour).padStart(2, "0");
  const minStr = String(et.minute).padStart(2, "0");
  const secStr = String(Math.floor(Date.now() / 1000) % 60).padStart(2, "0");
  const randSuffix = Math.random().toString(36).substring(2, 6);
  
  const cleanStrategy = strategy
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "_")
    .replace(/_+/g, "_")
    .substring(0, 20);
  
  return `${symbol}_${cleanStrategy}_${side}_${dateStr}_${hourStr}${minStr}${secStr}_${randSuffix}_T${tier}`;
}

export function parseTradeId(tradeId: string): TradeIdInfo | null {
  try {
    const parts = tradeId.split("_");
    if (parts.length < 5) return null;
    
    const symbol = parts[0];
    const tierPart = parts[parts.length - 1];
    const tier = parseInt(tierPart.replace("T", ""), 10) || 2;
    
    const randPart = parts[parts.length - 2];
    const timePart = parts[parts.length - 3];
    const datePart = parts[parts.length - 4];
    const sidePart = parts[parts.length - 5];
    
    const strategyParts = parts.slice(1, -5);
    const strategy = strategyParts.join("_") || "unknown";
    
    const timestamp = `${datePart.substring(0,4)}-${datePart.substring(4,6)}-${datePart.substring(6,8)} ${timePart.substring(0,2)}:${timePart.substring(2,4)}:${timePart.substring(4,6)} ET`;
    
    return {
      tradeId,
      symbol,
      strategy,
      side: sidePart,
      timestamp,
      tier,
    };
  } catch {
    return null;
  }
}

export function formatLogWithTradeId(message: string, tradeId: string): string {
  return `${message} trade_id=${tradeId}`;
}
