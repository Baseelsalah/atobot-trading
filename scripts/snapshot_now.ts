#!/usr/bin/env npx tsx
/**
 * Snapshot Now - One-shot runtime state diagnostic
 * 
 * Prints a comprehensive RUNSTATE block showing:
 * - Current time (ET + PT)
 * - Market status and schedule
 * - Entry window status
 * - Scan loop activity
 * - Skip/signal counts
 * - Trade attempts and fills
 * - Open positions and orders from Alpaca
 * 
 * Usage: npx tsx scripts/snapshot_now.ts
 */

import * as alpaca from "../server/alpaca";
import * as activityLedger from "../server/activityLedger";
import * as timeGuard from "../server/tradingTimeGuard";
import { getEasternTime } from "../server/timezone";

interface SnapshotData {
  timestamp: {
    nowET: string;
    nowPT: string;
    dateET: string;
  };
  market: {
    status: string;
    nextOpen: string;
    nextClose: string;
    isEarlyClose: boolean;
  };
  entryWindow: {
    entryAllowed: boolean;
    canManagePositions: boolean;
    shouldForceClose: boolean;
    reason: string;
    entryCutoffET: string;
    forceCloseET: string;
  };
  scanLoop: {
    ticksToday: number;
    lastTickET: string | null;
    symbolsEvaluatedToday: number;
    botWasRunning: boolean;
  };
  signals: {
    noSignalCount: number;
    totalSkips: number;
    topSkipReasons: Array<{ reason: string; count: number; percent: number }>;
  };
  trades: {
    tradesAttemptedToday: number;
    tradesFilledToday: number;
  };
  tradeAccounting: {
    proposed: number;
    submitted: number;
    rejected: number;
    suppressed: number;
    canceled: number;
    lastRejectionReason: string | null;
    topSuppressReasons: Array<{ reason: string; count: number }>;
  };
  alpaca: {
    openPositions: Array<{ symbol: string; qty: string; side: string; unrealizedPl: string }>;
    openOrdersCount: number;
    recentOrders: Array<{ id: string; symbol: string; side: string; status: string; createdAt: string }>;
  };
}

async function getSnapshot(): Promise<SnapshotData> {
  const et = getEasternTime();
  const now = new Date();
  
  const nowPT = now.toLocaleString("en-US", {
    timeZone: "America/Los_Angeles",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
  
  let marketStatus = "UNKNOWN";
  let nextOpen = "unknown";
  let nextClose = "unknown";
  let isEarlyClose = false;
  
  // Use tradingTimeGuard for authoritative entry window status
  const tgStatus = timeGuard.getTimeGuardStatus();
  const entryAllowed = tgStatus.canOpenNewTrades;
  const canManagePositions = tgStatus.canManagePositions;
  const shouldForceClose = tgStatus.shouldForceClose;
  const timeGuardReason = tgStatus.reason;
  
  // Entry window times from time guard (always accurate)
  const entryCutoffET = "11:35 AM (or next_close - 5 min)";
  const forceCloseET = "3:45 PM (or next_close - 2 min)";
  
  try {
    const clock = await alpaca.getClock();
    marketStatus = clock.is_open ? "OPEN" : "CLOSED";
    nextOpen = clock.next_open;
    nextClose = clock.next_close;
    
    const nextCloseDate = new Date(clock.next_close);
    const closeHour = parseInt(nextCloseDate.toLocaleString("en-US", {
      timeZone: "America/New_York",
      hour: "2-digit",
      hour12: false,
    }));
    
    isEarlyClose = closeHour < 16;
  } catch (err) {
    console.error("[Snapshot] Error fetching Alpaca clock:", err);
  }
  
  const summary = activityLedger.getTodaysSummary();
  
  let openPositions: SnapshotData["alpaca"]["openPositions"] = [];
  let openOrdersCount = 0;
  let recentOrders: SnapshotData["alpaca"]["recentOrders"] = [];
  
  try {
    const positions = await alpaca.getPositions();
    openPositions = positions.map(p => ({
      symbol: p.symbol,
      qty: p.qty,
      side: p.side,
      unrealizedPl: p.unrealized_pl,
    }));
  } catch (err) {
    console.error("[Snapshot] Error fetching positions:", err);
  }
  
  try {
    const orders = await alpaca.getOrders("all", 10);
    openOrdersCount = orders.filter(o => ["new", "accepted", "pending_new", "partially_filled"].includes(o.status)).length;
    recentOrders = orders.slice(0, 5).map(o => ({
      id: o.id.slice(0, 8),
      symbol: o.symbol,
      side: o.side,
      status: o.status,
      createdAt: new Date(o.created_at).toLocaleTimeString("en-US", { timeZone: "America/New_York" }),
    }));
  } catch (err) {
    console.error("[Snapshot] Error fetching orders:", err);
  }
  
  return {
    timestamp: {
      nowET: et.displayTime,
      nowPT,
      dateET: et.dateString,
    },
    market: {
      status: marketStatus,
      nextOpen,
      nextClose,
      isEarlyClose,
    },
    entryWindow: {
      entryAllowed,
      canManagePositions,
      shouldForceClose,
      reason: timeGuardReason,
      entryCutoffET,
      forceCloseET,
    },
    scanLoop: {
      ticksToday: summary.scanTicks,
      lastTickET: summary.lastTickET,
      symbolsEvaluatedToday: summary.symbolsEvaluated,
      botWasRunning: summary.botWasRunning,
    },
    signals: {
      noSignalCount: summary.noSignalCount,
      totalSkips: summary.totalSkips,
      topSkipReasons: summary.topSkipReasons.slice(0, 10),
    },
    trades: {
      tradesAttemptedToday: summary.tradesAttempted,
      tradesFilledToday: summary.tradesFilled,
    },
    tradeAccounting: summary.tradeAccounting || {
      proposed: 0,
      submitted: 0,
      rejected: 0,
      suppressed: 0,
      canceled: 0,
      lastRejectionReason: null,
      topSuppressReasons: [],
    },
    alpaca: {
      openPositions,
      openOrdersCount,
      recentOrders,
    },
  };
}

function formatSnapshot(data: SnapshotData): string {
  const lines: string[] = [];
  
  lines.push("============================================================");
  lines.push("                    ATOBOT RUNSTATE SNAPSHOT                 ");
  lines.push("============================================================");
  lines.push("");
  
  lines.push(`[TIME]`);
  lines.push(`  nowET:    ${data.timestamp.nowET}`);
  lines.push(`  nowPT:    ${data.timestamp.nowPT}`);
  lines.push(`  dateET:   ${data.timestamp.dateET}`);
  lines.push("");
  
  lines.push(`[MARKET]`);
  lines.push(`  status:       ${data.market.status}`);
  lines.push(`  nextOpen:     ${data.market.nextOpen}`);
  lines.push(`  nextClose:    ${data.market.nextClose}`);
  lines.push(`  isEarlyClose: ${data.market.isEarlyClose}`);
  lines.push("");
  
  lines.push(`[ENTRY WINDOW] (from tradingTimeGuard)`);
  lines.push(`  entryAllowed:       ${data.entryWindow.entryAllowed ? "YES" : "NO"}`);
  lines.push(`  canManagePositions: ${data.entryWindow.canManagePositions ? "YES" : "NO"}`);
  lines.push(`  shouldForceClose:   ${data.entryWindow.shouldForceClose ? "YES" : "NO"}`);
  lines.push(`  reason:             ${data.entryWindow.reason}`);
  lines.push(`  entryCutoffET:      ${data.entryWindow.entryCutoffET}`);
  lines.push(`  forceCloseET:       ${data.entryWindow.forceCloseET}`);
  lines.push("");
  
  lines.push(`[SCAN LOOP]`);
  lines.push(`  botWasRunning:         ${data.scanLoop.botWasRunning ? "YES" : "NO"}`);
  lines.push(`  ticksToday:            ${data.scanLoop.ticksToday}`);
  lines.push(`  lastTickET:            ${data.scanLoop.lastTickET || "none"}`);
  lines.push(`  symbolsEvaluatedToday: ${data.scanLoop.symbolsEvaluatedToday}`);
  lines.push("");
  
  lines.push(`[SIGNALS]`);
  lines.push(`  noSignalCount: ${data.signals.noSignalCount}`);
  lines.push(`  totalSkips:    ${data.signals.totalSkips}`);
  if (data.signals.topSkipReasons.length > 0) {
    lines.push(`  topSkipReasons:`);
    for (const r of data.signals.topSkipReasons) {
      lines.push(`    - ${r.reason}: ${r.count} (${r.percent.toFixed(1)}%)`);
    }
  }
  lines.push("");
  
  lines.push(`[TRADES]`);
  lines.push(`  tradesAttemptedToday: ${data.trades.tradesAttemptedToday}`);
  lines.push(`  tradesFilledToday:    ${data.trades.tradesFilledToday}`);
  lines.push("");
  
  lines.push(`[TRADE ACCOUNTING] (Alpaca visibility)`);
  lines.push(`  proposed:   ${data.tradeAccounting.proposed}`);
  lines.push(`  submitted:  ${data.tradeAccounting.submitted} (accepted by Alpaca)`);
  lines.push(`  rejected:   ${data.tradeAccounting.rejected} (Alpaca rejected)`);
  lines.push(`  suppressed: ${data.tradeAccounting.suppressed} (blocked before submit)`);
  lines.push(`  canceled:   ${data.tradeAccounting.canceled} (timeout/canceled)`);
  if (data.tradeAccounting.lastRejectionReason) {
    lines.push(`  lastRejectionReason: ${data.tradeAccounting.lastRejectionReason}`);
  }
  if (data.tradeAccounting.topSuppressReasons.length > 0) {
    lines.push(`  topSuppressReasons:`);
    for (const r of data.tradeAccounting.topSuppressReasons) {
      lines.push(`    - ${r.reason}: ${r.count}`);
    }
  }
  // Invariant check: proposed = submitted + rejected + suppressed
  const expectedSubmitted = data.tradeAccounting.proposed - data.tradeAccounting.rejected - data.tradeAccounting.suppressed;
  if (expectedSubmitted !== data.tradeAccounting.submitted && data.tradeAccounting.proposed > 0) {
    lines.push(`  INVARIANT VIOLATION: proposed (${data.tradeAccounting.proposed}) != submitted (${data.tradeAccounting.submitted}) + rejected (${data.tradeAccounting.rejected}) + suppressed (${data.tradeAccounting.suppressed})`);
  }
  lines.push("");
  
  lines.push(`[ALPACA POSITIONS] (${data.alpaca.openPositions.length})`);
  if (data.alpaca.openPositions.length === 0) {
    lines.push(`  (no open positions)`);
  } else {
    for (const p of data.alpaca.openPositions) {
      lines.push(`  - ${p.symbol}: ${p.qty} shares (${p.side}) P&L: $${p.unrealizedPl}`);
    }
  }
  lines.push("");
  
  lines.push(`[ALPACA ORDERS]`);
  lines.push(`  openOrdersCount: ${data.alpaca.openOrdersCount}`);
  if (data.alpaca.recentOrders.length > 0) {
    lines.push(`  recent orders (last 5):`);
    for (const o of data.alpaca.recentOrders) {
      lines.push(`    - ${o.id} ${o.symbol} ${o.side} ${o.status} @ ${o.createdAt}`);
    }
  }
  lines.push("");
  
  lines.push("============================================================");
  lines.push("                         DIAGNOSIS                          ");
  lines.push("============================================================");
  
  const diagnosis: string[] = [];
  
  if (data.market.status === "CLOSED") {
    diagnosis.push("Market is CLOSED - no trading activity expected");
  } else if (!data.entryWindow.entryAllowed) {
    diagnosis.push("Market OPEN but outside entry window (9:35-11:35 ET) - managing existing positions only");
  } else {
    diagnosis.push("Market OPEN and within entry window - new trades allowed");
  }
  
  if (!data.scanLoop.botWasRunning) {
    diagnosis.push("WARNING: No scan ticks recorded today - bot may not be running");
  } else if (data.scanLoop.ticksToday === 0) {
    diagnosis.push("WARNING: Bot started but no ticks yet");
  } else {
    diagnosis.push(`Bot is active: ${data.scanLoop.ticksToday} ticks today, last at ${data.scanLoop.lastTickET}`);
  }
  
  if (data.signals.totalSkips > 0 && data.trades.tradesAttemptedToday === 0) {
    diagnosis.push(`All ${data.signals.totalSkips} evaluations were SKIPPED - check skip reasons above`);
  } else if (data.signals.noSignalCount > 0 && data.trades.tradesAttemptedToday === 0) {
    diagnosis.push(`${data.signals.noSignalCount} symbols had NO_SIGNAL - no trade opportunities found`);
  }
  
  if (data.trades.tradesAttemptedToday > 0) {
    diagnosis.push(`Trades attempted: ${data.trades.tradesAttemptedToday}, filled: ${data.trades.tradesFilledToday}`);
  }
  
  if (data.alpaca.openPositions.length > 0) {
    diagnosis.push(`Currently holding ${data.alpaca.openPositions.length} position(s)`);
  }
  
  if (data.alpaca.openOrdersCount > 0) {
    diagnosis.push(`${data.alpaca.openOrdersCount} order(s) pending in Alpaca`);
  }
  
  for (const d of diagnosis) {
    lines.push(`  ${d}`);
  }
  
  lines.push("");
  lines.push("============================================================");
  
  return lines.join("\n");
}

async function main(): Promise<void> {
  console.log("\nFetching runtime snapshot...\n");
  
  try {
    const snapshot = await getSnapshot();
    console.log(formatSnapshot(snapshot));
    
    console.log("\n[JSON DATA]");
    console.log(JSON.stringify(snapshot, null, 2));
    
  } catch (err) {
    console.error("Error generating snapshot:", err);
    process.exit(1);
  }
  
  process.exit(0);
}

main();

export { getSnapshot, formatSnapshot, SnapshotData };
