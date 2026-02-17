/**
 * Slippage Model for Backtesting
 *
 * Applies configurable slippage to simulated trade fills.
 * Supports: no slippage, fixed basis points, or percentage of estimated spread.
 */

import type { AlpacaBar } from "./types";

export interface SlippageResult {
  adjustedPrice: number;
  slippageBps: number;
  slippageDollars: number;
}

/**
 * Estimate spread from bar data.
 * Uses a conservative proxy since historical bars lack bid/ask data.
 * Returns estimated spread in dollars.
 */
export function estimateSpread(bar: AlpacaBar): number {
  const barRange = bar.h - bar.l;
  // Use 10% of bar range or 1 basis point of close, whichever is larger
  return Math.max(bar.c * 0.0001, barRange * 0.1);
}

/**
 * Apply slippage to an execution price.
 * For BUY: price goes UP (worse fill)
 * For SELL: price goes DOWN (worse fill)
 */
export function applySlippage(
  price: number,
  side: "buy" | "sell",
  mode: "none" | "fixed" | "pctOfSpread",
  fixedBps: number,
  pctOfSpread: number,
  bar: AlpacaBar
): SlippageResult {
  if (mode === "none") {
    return { adjustedPrice: price, slippageBps: 0, slippageDollars: 0 };
  }

  let slippageDollars: number;

  if (mode === "fixed") {
    slippageDollars = price * (fixedBps / 10000);
  } else {
    // pctOfSpread mode
    const spread = estimateSpread(bar);
    slippageDollars = spread * pctOfSpread;
  }

  const slippageBps = (slippageDollars / price) * 10000;

  // Apply in adverse direction
  const adjustedPrice = side === "buy"
    ? price + slippageDollars
    : price - slippageDollars;

  return {
    adjustedPrice,
    slippageBps,
    slippageDollars,
  };
}
