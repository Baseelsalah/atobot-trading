import * as alpaca from "./alpaca";
import * as indicators from "./indicators";
import { storage } from "./storage";
import { DAY_TRADER_CONFIG } from "./dayTraderConfig";

const PARTIAL_PROFIT_PCT = 0.33;  // Proven balanced profit capture
const TRADE_TIMEOUT_MINUTES = 30;  // Proven value — 60 min timeouts were losers
const PARTIAL_PROFIT_R_MULTIPLE = 0.5;  // Take partial at +0.5R to activate trailing stop

interface ManagedPosition {
  symbol: string;
  entryPrice: number;
  stopPrice: number;
  targetPrice: number;
  quantity: number;
  startTime: number;
  partialTaken: boolean;
  rValue: number;
  trailingActive: boolean;
  tradeId: string | null;
  breakevenArmed: boolean;
}

const managedPositions: Map<string, ManagedPosition> = new Map();
let positionMonitorInterval: NodeJS.Timeout | null = null;

export function trackNewPosition(
  symbol: string,
  entryPrice: number,
  stopPrice: number,
  targetPrice: number,
  quantity: number,
  tradeId?: string
): void {
  const rValue = entryPrice - stopPrice;
  
  managedPositions.set(symbol, {
    symbol,
    entryPrice,
    stopPrice,
    targetPrice,
    quantity,
    startTime: Date.now(),
    partialTaken: false,
    rValue,
    trailingActive: false,
    tradeId: tradeId || null,
    breakevenArmed: false,
  });
  
  const tradeIdLog = tradeId ? ` trade_id=${tradeId}` : "";
  console.log(`[PositionManager] Tracking ${symbol}: Entry $${entryPrice.toFixed(2)}, Stop $${stopPrice.toFixed(2)}, R-Value $${rValue.toFixed(2)}${tradeIdLog}`);
}

export function getManagedPositions(): ManagedPosition[] {
  return Array.from(managedPositions.values());
}

export function getManagedPositionsSafe(): { symbol: string; partialTaken: boolean; trailingActive: boolean; elapsedMinutes: number }[] {
  return Array.from(managedPositions.values()).map(p => ({
    symbol: p.symbol,
    partialTaken: p.partialTaken,
    trailingActive: p.trailingActive,
    elapsedMinutes: Math.round((Date.now() - p.startTime) / 60000),
  }));
}

export async function rehydrateFromAlpaca(): Promise<void> {
  try {
    const positions = await alpaca.getPositions();
    
    if (positions.length === 0) {
      console.log("[PositionManager] No open positions to rehydrate");
      return;
    }
    
    for (const pos of positions) {
      if (managedPositions.has(pos.symbol)) continue;
      
      const entryPrice = parseFloat(pos.avg_entry_price);
      const qty = parseInt(pos.qty);
      
      // Use 1.5% of entry as default R-value when rehydrating (conservative estimate)
      const estimatedRValue = entryPrice * 0.015;
      const stopPrice = entryPrice - estimatedRValue;
      const targetPrice = entryPrice + (estimatedRValue * 2);
      
      // IMPORTANT: Always set partialTaken=false on rehydration
      // The position monitor will check if +1R is reached and take partials
      // This ensures we don't skip partial profit taking on restart
      managedPositions.set(pos.symbol, {
        symbol: pos.symbol,
        entryPrice,
        stopPrice,
        targetPrice,
        quantity: qty,
        startTime: Date.now(),
        partialTaken: false,  // Never assume partial was taken
        rValue: estimatedRValue,
        trailingActive: false, // Never assume trailing is active
        tradeId: null, // Unknown on rehydration
        breakevenArmed: false,
      });
      
      console.log(`[PositionManager] Rehydrated ${pos.symbol}: Entry $${entryPrice.toFixed(2)}, Stop $${stopPrice.toFixed(2)}, R=$${estimatedRValue.toFixed(2)}`);
    }
    
    console.log(`[PositionManager] Rehydrated ${positions.length} positions from Alpaca`);
  } catch (error) {
    console.error("[PositionManager] Error rehydrating positions:", error);
  }
}

export function isPositionManaged(symbol: string): boolean {
  return managedPositions.has(symbol);
}

export function removePosition(symbol: string): void {
  managedPositions.delete(symbol);
  console.log(`[PositionManager] Removed ${symbol} from tracking`);
}

export async function checkAndManagePositions(): Promise<void> {
  if (managedPositions.size === 0) return;
  
  try {
    const positions = await alpaca.getPositions();
    const positionMap = new Map(positions.map(p => [p.symbol, p]));
    
    for (const [symbol, managed] of Array.from(managedPositions.entries())) {
      const position = positionMap.get(symbol);
      
      if (!position) {
        console.log(`[PositionManager] ${symbol} no longer in portfolio, removing from tracking`);
        managedPositions.delete(symbol);
        continue;
      }
      
      const currentPrice = parseFloat(position.current_price);
      const currentQty = parseInt(position.qty);
      const elapsedMinutes = (Date.now() - managed.startTime) / (60 * 1000);
      
      // Check breakeven rule first (before stop loss, fixes avg loss > avg win leak)
      checkBreakevenRule(symbol, managed, currentPrice);
      
      const stopHit = await checkStopLoss(symbol, managed, currentPrice);
      if (stopHit) continue;
      
      const partialTaken = await checkPartialProfit(symbol, managed, currentPrice, currentQty);
      if (partialTaken) continue;
      
      if (managed.partialTaken) {
        await checkTrailingStop(symbol, managed, currentPrice);
      }
      
      // BASELINE_MAX_HOLD_MINUTES: Close positions held too long (only in BASELINE_MODE)
      if (DAY_TRADER_CONFIG.BASELINE_MODE && elapsedMinutes > DAY_TRADER_CONFIG.BASELINE_MAX_HOLD_MINUTES) {
        console.log(`[PositionManager] ${symbol} BASELINE TIME EXIT after ${elapsedMinutes.toFixed(1)} minutes (max: ${DAY_TRADER_CONFIG.BASELINE_MAX_HOLD_MINUTES})`);
        await closePositionWithLog(symbol, currentPrice, managed, "TIME_EXIT_BASELINE");
        continue;
      }
      
      // Legacy timeout for non-moving positions (shorter than BASELINE_MAX_HOLD)
      if (elapsedMinutes > TRADE_TIMEOUT_MINUTES && !managed.partialTaken) {
        console.log(`[PositionManager] ${symbol} timeout after ${elapsedMinutes.toFixed(1)} minutes`);
        await closePositionWithLog(symbol, currentPrice, managed, "timeout");
      }
    }
  } catch (error) {
    console.error("[PositionManager] Error checking positions:", error);
  }
}

async function checkStopLoss(symbol: string, managed: ManagedPosition, currentPrice: number): Promise<boolean> {
  if (currentPrice <= managed.stopPrice) {
    console.log(`[PositionManager] ${symbol} STOP HIT at $${currentPrice.toFixed(2)} (stop: $${managed.stopPrice.toFixed(2)})`);
    await closePositionWithLog(symbol, currentPrice, managed, "stop_loss");
    return true;
  }
  return false;
}

/**
 * Breakeven Rule: When unrealized gain reaches BREAKEVEN_TRIGGER_PCT, move stop to entry + offset.
 * This prevents winners from turning into losers (fixes avg loss > avg win leak).
 * The offset (BREAKEVEN_OFFSET_PCT) adds a small buffer above entry to avoid spread/micro-noise.
 * Returns true if breakeven was just armed this check, false otherwise.
 */
function checkBreakevenRule(symbol: string, managed: ManagedPosition, currentPrice: number): boolean {
  // Skip if breakeven already armed or partial was already taken (stop already moved)
  if (managed.breakevenArmed || managed.partialTaken) return false;
  
  const triggerPct = DAY_TRADER_CONFIG.BREAKEVEN_TRIGGER_PCT;
  const offsetPct = DAY_TRADER_CONFIG.BREAKEVEN_OFFSET_PCT;
  const gainPct = ((currentPrice - managed.entryPrice) / managed.entryPrice) * 100;
  
  if (gainPct >= triggerPct) {
    // Move stop to entry + offset (small buffer to avoid spread/micro-noise)
    const oldStop = managed.stopPrice;
    managed.stopPrice = managed.entryPrice * (1 + offsetPct / 100);
    managed.breakevenArmed = true;
    
    const tradeIdLog = managed.tradeId ? ` trade_id=${managed.tradeId}` : "";
    console.log(`[PositionManager] ${symbol} BREAKEVEN_ARMED trigger=${triggerPct}% stop=entry+${offsetPct}% ($${managed.stopPrice.toFixed(2)})${tradeIdLog}`);
    
    return true;
  }
  
  return false;
}

async function checkPartialProfit(symbol: string, managed: ManagedPosition, currentPrice: number, currentQty: number): Promise<boolean> {
  if (managed.partialTaken) return false;
  
  const profitTarget = managed.entryPrice + managed.rValue * PARTIAL_PROFIT_R_MULTIPLE;
  const tradeIdLog = managed.tradeId ? ` trade_id=${managed.tradeId}` : "";

  if (currentPrice >= profitTarget) {
    const qtyToSell = Math.max(1, Math.floor(currentQty * PARTIAL_PROFIT_PCT));

    console.log(`[PositionManager] ${symbol} +${PARTIAL_PROFIT_R_MULTIPLE}R reached! Taking ${PARTIAL_PROFIT_PCT * 100}% profit (${qtyToSell} shares)${tradeIdLog}`);
    
    try {
      // Use trade_id with _PARTIAL suffix for partial exits
      const partialClientOrderId = managed.tradeId ? `${managed.tradeId}_PARTIAL` : undefined;
      await alpaca.submitOrder(symbol, qtyToSell, "sell", "market", undefined, "partial_profit", partialClientOrderId);
      
      managed.partialTaken = true;
      managed.trailingActive = true;
      managed.stopPrice = managed.entryPrice + (0.1 * managed.rValue);
      managed.quantity = currentQty - qtyToSell;
      
      console.log(`[PositionManager] ${symbol} stop moved to breakeven+0.1R: $${managed.stopPrice.toFixed(2)}${tradeIdLog}`);
      
      await storage.createActivityLog({
        type: "trade",
        action: "Partial Profit Taken",
        description: `${symbol}: Sold ${qtyToSell} shares at +1R ($${currentPrice.toFixed(2)}). Stop moved to $${managed.stopPrice.toFixed(2)}${tradeIdLog}`,
      });
      
      return true;
    } catch (error) {
      console.error(`[PositionManager] Error taking partial profit on ${symbol}:`, error);
    }
  }
  
  return false;
}

async function checkTrailingStop(symbol: string, managed: ManagedPosition, currentPrice: number): Promise<void> {
  if (!managed.trailingActive) return;
  
  try {
    const barResult = await alpaca.getBarsSafe(symbol, "1Min", 10);
    const bars = barResult.bars;
    
    if (bars.length < 10) return;
    
    const closes = bars.map((b: { c: number }) => b.c);
    const ema9 = indicators.ema(closes, 9);
    const currentEma9 = ema9[ema9.length - 1];
    
    if (currentPrice < currentEma9) {
      console.log(`[PositionManager] ${symbol} trailing stop hit - price $${currentPrice.toFixed(2)} below EMA-9 $${currentEma9.toFixed(2)}`);
      await closePositionWithLog(symbol, currentPrice, managed, "trailing_stop");
    } else {
      const newStop = Math.max(managed.stopPrice, currentEma9 * 0.998);
      if (newStop > managed.stopPrice) {
        managed.stopPrice = newStop;
        console.log(`[PositionManager] ${symbol} trailing stop raised to $${newStop.toFixed(2)}`);
      }
    }
  } catch (error) {
    console.error(`[PositionManager] Error checking trailing stop for ${symbol}:`, error);
  }
}

async function closePositionWithLog(symbol: string, exitPrice: number, managed: ManagedPosition, reason: string): Promise<void> {
  try {
    const tradeIdLog = managed.tradeId ? ` trade_id=${managed.tradeId}` : "";
    await alpaca.closePosition(symbol, reason, managed.tradeId || undefined);
    
    const pnl = (exitPrice - managed.entryPrice) * managed.quantity;
    const pnlPercent = ((exitPrice - managed.entryPrice) / managed.entryPrice) * 100;
    const holdMinutes = Math.round((Date.now() - managed.startTime) / 60000);
    
    console.log(`[PositionManager] CLOSED ${symbol}: exit_reason=${reason} pnl=$${pnl.toFixed(2)} hold_minutes=${holdMinutes}${tradeIdLog}`);
    
    await storage.createActivityLog({
      type: "trade",
      action: `Position Closed - ${reason}`,
      description: `${symbol}: Exit at $${exitPrice.toFixed(2)}, P/L: $${pnl.toFixed(2)} (${pnlPercent.toFixed(2)}%), Hold: ${holdMinutes}m${tradeIdLog}`,
    });
    
    managedPositions.delete(symbol);
  } catch (error) {
    console.error(`[PositionManager] Error closing ${symbol}:`, error);
  }
}

export function startPositionMonitor(): void {
  if (positionMonitorInterval) return;
  
  positionMonitorInterval = setInterval(checkAndManagePositions, 5000);
  console.log("[PositionManager] Position monitor started (5-second intervals)");
}

export function stopPositionMonitor(): void {
  if (positionMonitorInterval) {
    clearInterval(positionMonitorInterval);
    positionMonitorInterval = null;
    console.log("[PositionManager] Position monitor stopped");
  }
}

export function resetDaily(): void {
  managedPositions.clear();
  console.log("[PositionManager] Daily reset - cleared all managed positions");
}
