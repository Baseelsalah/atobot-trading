/**
 * Position Simulator for Backtesting
 *
 * Reimplements positionManager.ts exit logic as pure simulation
 * without any Alpaca API calls. Processes bars to detect exits.
 *
 * Exit checks (in order, matching live system):
 * 1. Breakeven rule - move stop to entry + offset when gain >= trigger %
 * 2. Stop loss - price hits stop price
 * 3. Partial profit - sell 33% at +1R, move stop to breakeven
 * 4. Trailing stop - after partial, exit if price < EMA9 on 1-min bars
 * 5. Timeout - 15 min with no partial taken
 */

import { ema } from "../indicators";
import type { AlpacaBar, TradeExit, TradeExitReason } from "./types";

export interface SimulatedPosition {
  symbol: string;
  entryPrice: number;
  stopPrice: number;
  targetPrice: number;
  quantity: number;
  originalQuantity: number;
  startTimeMs: number;
  partialTaken: boolean;
  rValue: number;           // entryPrice - initial stopPrice (risk per share)
  trailingActive: boolean;
  breakevenArmed: boolean;
  strategyName: string;
  confidence: number;
  tradeIndex: number;       // Reference to BacktestTrade.id
  side: "buy" | "sell";
}

interface PositionExitConfig {
  partialProfitPct: number;          // 0.33
  partialProfitRMultiple: number;    // 0.5 (R-multiple at which to take partial)
  tradeTimeoutMinutes: number;       // 30
  breakevenTriggerPct: number;       // 0.3
  breakevenOffsetPct: number;        // 0.05
}

export class PositionSimulator {
  private positions: Map<string, SimulatedPosition> = new Map();

  /** Open a new simulated position */
  openPosition(pos: SimulatedPosition): void {
    this.positions.set(pos.symbol, pos);
  }

  /** Get all open positions */
  getOpenPositions(): SimulatedPosition[] {
    return Array.from(this.positions.values());
  }

  /** Check if symbol has open position */
  hasPosition(symbol: string): boolean {
    return this.positions.has(symbol);
  }

  /** Count of open positions */
  positionCount(): number {
    return this.positions.size;
  }

  /** Remove a position after exit */
  removePosition(symbol: string): void {
    this.positions.delete(symbol);
  }

  /** Clear all positions */
  clear(): void {
    this.positions.clear();
  }

  /**
   * Process a 1-minute bar for a specific symbol's position only.
   * Returns exit event if one occurred, null otherwise.
   */
  processBarForSymbol(
    symbol: string,
    bar: AlpacaBar,
    recentCloses1min: number[],
    currentTimeMs: number,
    config: PositionExitConfig
  ): TradeExit | null {
    const pos = this.positions.get(symbol);
    if (!pos) return null;

    const exit = this.checkPositionExit(symbol, pos, bar, recentCloses1min, currentTimeMs, config);
    if (exit) {
      if (exit.reason !== "take_profit" || pos.quantity <= 0) {
        this.positions.delete(symbol);
      }
    }
    return exit;
  }

  /**
   * Force close a single symbol's position.
   */
  forceClosePosition(
    symbol: string,
    exitPrice: number,
    exitTime: string,
    reason: TradeExitReason
  ): TradeExit | null {
    const pos = this.positions.get(symbol);
    if (!pos) return null;

    const pnl = this.calculatePnl(pos, exitPrice, pos.quantity);
    this.positions.delete(symbol);

    return {
      symbol,
      price: exitPrice,
      quantity: pos.quantity,
      time: exitTime,
      reason,
      pnl,
    };
  }

  /**
   * Process a 1-minute bar against all open positions.
   * Returns array of exit events that occurred.
   * @deprecated Use processBarForSymbol() instead to avoid cross-symbol contamination.
   */
  processBar(
    bar: AlpacaBar,
    recentCloses1min: number[],  // last 10+ 1-min closes for EMA-9 trailing
    currentTimeMs: number,
    config: PositionExitConfig
  ): TradeExit[] {
    const exits: TradeExit[] = [];

    for (const [symbol, pos] of Array.from(this.positions.entries())) {
      const exit = this.checkPositionExit(symbol, pos, bar, recentCloses1min, currentTimeMs, config);
      if (exit) {
        exits.push(exit);
        // If full exit, remove position
        if (exit.reason !== "take_profit" || pos.quantity <= 0) {
          this.positions.delete(symbol);
        }
      }
    }

    return exits;
  }

  /**
   * Force close all positions (end of day or daily limit).
   */
  forceCloseAll(
    bar: AlpacaBar,
    reason: TradeExitReason
  ): TradeExit[] {
    const exits: TradeExit[] = [];

    for (const [symbol, pos] of Array.from(this.positions.entries())) {
      const exitPrice = bar.c;
      const pnl = this.calculatePnl(pos, exitPrice, pos.quantity);

      exits.push({
        symbol,
        price: exitPrice,
        quantity: pos.quantity,
        time: bar.t,
        reason,
        pnl,
      });

      this.positions.delete(symbol);
    }

    return exits;
  }

  /**
   * Check a single position against a 1-min bar for exit conditions.
   */
  private checkPositionExit(
    symbol: string,
    pos: SimulatedPosition,
    bar: AlpacaBar,
    recentCloses1min: number[],
    currentTimeMs: number,
    config: PositionExitConfig
  ): TradeExit | null {
    const currentPrice = bar.c;
    const elapsedMinutes = (currentTimeMs - pos.startTimeMs) / 60000;

    // For buy positions: check against bar low for stop, bar high for target
    // For sell positions: reversed (but we primarily handle buy-side logic)
    const isBuy = pos.side === "buy";

    // 1. Check breakeven rule (before stop loss)
    this.checkBreakevenRule(pos, isBuy ? bar.h : bar.l, config);

    // 2. Check stop loss
    const stopTriggered = isBuy
      ? bar.l <= pos.stopPrice
      : bar.h >= pos.stopPrice;

    if (stopTriggered) {
      // Use stop price as exit (or bar open if it gapped past)
      const exitPrice = isBuy
        ? Math.min(pos.stopPrice, bar.o)
        : Math.max(pos.stopPrice, bar.o);
      const pnl = this.calculatePnl(pos, exitPrice, pos.quantity);

      return {
        symbol,
        price: exitPrice,
        quantity: pos.quantity,
        time: bar.t,
        reason: pos.breakevenArmed && !pos.partialTaken ? "breakeven_stop" : "stop_loss",
        pnl,
      };
    }

    // 3. Check partial profit at configurable R-multiple (default +0.5R)
    if (!pos.partialTaken) {
      const rMultiple = config.partialProfitRMultiple ?? 1.0;
      const profitTarget = isBuy
        ? pos.entryPrice + pos.rValue * rMultiple
        : pos.entryPrice - pos.rValue * rMultiple;
      const targetHit = isBuy ? bar.h >= profitTarget : bar.l <= profitTarget;

      if (targetHit) {
        const partialQty = Math.max(1, Math.floor(pos.originalQuantity * config.partialProfitPct));
        const exitPrice = profitTarget;
        const pnl = this.calculatePnl(pos, exitPrice, partialQty);

        pos.partialTaken = true;
        pos.trailingActive = true;
        pos.quantity -= partialQty;
        // Move stop to breakeven + 0.1R
        pos.stopPrice = isBuy
          ? pos.entryPrice + (0.1 * pos.rValue)
          : pos.entryPrice - (0.1 * pos.rValue);

        // If no shares remain, full exit
        if (pos.quantity <= 0) {
          this.positions.delete(pos.symbol);
        }

        return {
          symbol,
          price: exitPrice,
          quantity: partialQty,
          time: bar.t,
          reason: "take_profit",
          pnl,
        };
      }
    }

    // 4. Check trailing stop (after partial taken)
    if (pos.partialTaken && pos.trailingActive && recentCloses1min.length >= 10) {
      const ema9Values = ema(recentCloses1min, 9);
      const currentEma9 = ema9Values[ema9Values.length - 1];

      if (Number.isFinite(currentEma9)) {
        const trailingHit = isBuy
          ? currentPrice < currentEma9
          : currentPrice > currentEma9;

        if (trailingHit) {
          const pnl = this.calculatePnl(pos, currentPrice, pos.quantity);
          return {
            symbol,
            price: currentPrice,
            quantity: pos.quantity,
            time: bar.t,
            reason: "trailing_stop",
            pnl,
          };
        }

        // Raise trailing stop
        if (isBuy) {
          const newStop = Math.max(pos.stopPrice, currentEma9 * 0.998);
          if (newStop > pos.stopPrice) pos.stopPrice = newStop;
        } else {
          const newStop = Math.min(pos.stopPrice, currentEma9 * 1.002);
          if (newStop < pos.stopPrice) pos.stopPrice = newStop;
        }
      }
    }

    // 5. Check timeout (only if no partial taken)
    if (elapsedMinutes > config.tradeTimeoutMinutes && !pos.partialTaken) {
      const pnl = this.calculatePnl(pos, currentPrice, pos.quantity);
      return {
        symbol,
        price: currentPrice,
        quantity: pos.quantity,
        time: bar.t,
        reason: "timeout",
        pnl,
      };
    }

    return null;
  }

  /**
   * Check and apply breakeven rule.
   * When unrealized gain reaches trigger%, move stop to entry + offset%.
   */
  private checkBreakevenRule(
    pos: SimulatedPosition,
    priceForCheck: number,
    config: PositionExitConfig
  ): void {
    if (pos.breakevenArmed || pos.partialTaken) return;

    const isBuy = pos.side === "buy";
    const gainPct = isBuy
      ? ((priceForCheck - pos.entryPrice) / pos.entryPrice) * 100
      : ((pos.entryPrice - priceForCheck) / pos.entryPrice) * 100;

    if (gainPct >= config.breakevenTriggerPct) {
      pos.stopPrice = isBuy
        ? pos.entryPrice * (1 + config.breakevenOffsetPct / 100)
        : pos.entryPrice * (1 - config.breakevenOffsetPct / 100);
      pos.breakevenArmed = true;
    }
  }

  /**
   * Calculate P&L for a position exit.
   */
  private calculatePnl(pos: SimulatedPosition, exitPrice: number, quantity: number): number {
    if (pos.side === "buy") {
      return (exitPrice - pos.entryPrice) * quantity;
    } else {
      return (pos.entryPrice - exitPrice) * quantity;
    }
  }
}
