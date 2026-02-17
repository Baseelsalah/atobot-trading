/**
 * Backtest Engine - Main bar-by-bar replay engine
 *
 * Orchestrates the backtest:
 * 1. Iterates trading days chronologically
 * 2. At each 5-min bar: evaluates strategies (reuses strategyEngine.ts)
 * 3. Between 5-min bars: processes 1-min bars for position management
 * 4. Tracks equity, trades, and daily state
 */

import { ema, rsi, atr, macd } from "../indicators";
import { evaluateStrategies, selectBestSignal } from "../strategyEngine";
import { determineRegimeLabel, REGIME_MIN_EMA_SPREAD_PCT } from "../regimeFilter";
import type { Tier1Indicators, Tier2Indicators } from "../indicatorPipeline";
import type { StrategySignal } from "../strategyEngine";
import type { RegimeLabel } from "../regimeFilter";

import { SimulatedClock } from "./simulatedClock";
import { PositionSimulator, type SimulatedPosition } from "./positionSimulator";
import { applySlippage } from "./slippageModel";
import type {
  BacktestConfig,
  BacktestTrade,
  BacktestReport,
  DayState,
  TradeExit,
  AlpacaBar,
  SymbolBarData,
} from "./types";

const TIER_1_MIN_BARS = 130;
const TIER_2_MIN_BARS = 200;

export class BacktestEngine {
  private config: BacktestConfig;
  private clock: SimulatedClock;
  private positionSim: PositionSimulator;
  private trades: BacktestTrade[] = [];
  private tradeIdCounter = 0;
  private equity: number;
  private equityCurve: { date: string; equity: number }[] = [];
  private spyDailyCloses: number[] = []; // Track SPY daily closes for daily trend filter

  constructor(config: BacktestConfig) {
    this.config = config;
    this.clock = new SimulatedClock();
    this.positionSim = new PositionSimulator();
    this.equity = config.initialEquity;
  }

  /**
   * Run the full backtest.
   */
  async run(
    data: Map<string, SymbolBarData>,
    tradingDays: { date: string; open: string; close: string }[]
  ): Promise<BacktestReport> {
    const startTime = Date.now();

    // Index bars by date for efficient lookup
    const barsByDate5min = this.indexBarsByDate(data, "5min");
    const barsByDate1min = this.indexBarsByDate(data, "1min");

    // Track all 5-min bars per symbol for trailing indicator window
    const allBars5min = new Map<string, AlpacaBar[]>();
    for (const [symbol, symbolData] of Array.from(data.entries())) {
      allBars5min.set(symbol, [...symbolData.bars5min]);
    }

    let totalBarsProcessed = 0;

    for (const day of tradingDays) {
      const dayState: DayState = {
        date: day.date,
        entriesCount: 0,
        dailyPnl: 0,
        consecutiveLosses: 0,
        cooldownMap: new Map(),
        peakEquity: this.equity,
      };

      const barsProcessed = this.processDay(
        day,
        barsByDate5min,
        barsByDate1min,
        allBars5min,
        dayState
      );

      totalBarsProcessed += barsProcessed;

      // Record SPY daily close for daily trend filter
      const spyDayBars = barsByDate5min.get(day.date)?.get("SPY");
      if (spyDayBars && spyDayBars.length > 0) {
        this.spyDailyCloses.push(spyDayBars[spyDayBars.length - 1].c);
      }

      // Record end-of-day equity
      this.equityCurve.push({ date: day.date, equity: this.equity });

      if (this.config.verbose) {
        const dayTrades = this.trades.filter(t => t.entryTime.startsWith(day.date));
        const dayPnl = dayTrades.reduce((sum, t) => sum + (t.realizedPnl || 0), 0);
        console.log(
          `  ${day.date}: ${dayTrades.length} trades, P&L: $${dayPnl.toFixed(2)}, ` +
          `Equity: $${this.equity.toFixed(2)}, Bars: ${barsProcessed}`
        );
      }
    }

    // Clean up simulated clock
    this.clock.reset();

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`\nBacktest complete in ${elapsed}s. ${this.trades.length} trades, ${totalBarsProcessed} bars processed.`);

    // Metrics are computed externally
    return {
      config: this.config,
      runTimestamp: new Date().toISOString(),
      aggregate: null as any, // Filled by metricsCalculator
      perStrategy: [],
      trades: this.trades,
    };
  }

  getEquityCurve(): { date: string; equity: number }[] {
    return this.equityCurve;
  }

  getFinalEquity(): number {
    return this.equity;
  }

  /**
   * Process a single trading day.
   */
  private processDay(
    day: { date: string; open: string; close: string },
    barsByDate5min: Map<string, Map<string, AlpacaBar[]>>,
    barsByDate1min: Map<string, Map<string, AlpacaBar[]>>,
    allBars5min: Map<string, AlpacaBar[]>,
    dayState: DayState
  ): number {
    let barsProcessed = 0;

    // Get 5-min bars for this day across all symbols
    const dayBars5min = barsByDate5min.get(day.date) || new Map();
    const dayBars1min = barsByDate1min.get(day.date) || new Map();

    // Collect all unique 5-min timestamps for this day, sorted
    const timestamps5min = new Set<string>();
    for (const [, bars] of Array.from(dayBars5min.entries())) {
      for (const bar of bars) timestamps5min.add(bar.t);
    }
    const sortedTimestamps = Array.from(timestamps5min).sort();

    if (sortedTimestamps.length === 0) return 0;

    // Build 1-min bar timeline per symbol for position management
    const symbol1minBars = new Map<string, AlpacaBar[]>();
    for (const [symbol, bars] of Array.from(dayBars1min.entries())) {
      symbol1minBars.set(symbol, bars);
    }

    // Track which 1-min bars we've processed
    const symbol1minIndex = new Map<string, number>();
    for (const symbol of this.config.symbols) {
      symbol1minIndex.set(symbol, 0);
    }

    // Evaluate regime filter for the day using SPY bars
    const spyAllBars = allBars5min.get("SPY") || [];

    let prevTimestamp: string | null = null;

    for (const timestamp of sortedTimestamps) {
      this.clock.setTimeFromBar(timestamp);
      const minutesET = this.clock.getMinutesSinceMidnightET();
      barsProcessed++;

      // ─── Position Management on 1-min bars ───
      // Process 1-min bars between previous and current 5-min bar
      if (this.positionSim.positionCount() > 0) {
        const posExits = this.processPositionManagement(
          prevTimestamp,
          timestamp,
          symbol1minBars,
          symbol1minIndex,
          dayState
        );

        // Apply exits to equity
        for (const exit of posExits) {
          this.equity += exit.pnl;
          dayState.dailyPnl += exit.pnl;
        }
      }

      // Check force close
      if (minutesET >= this.config.forceCloseMinutes) {
        if (this.positionSim.positionCount() > 0) {
          // Force close each position using its own symbol's bar price
          const positions = this.positionSim.getOpenPositions();
          for (const pos of positions) {
            // Find this symbol's bar at the current timestamp
            const symBars = dayBars5min.get(pos.symbol);
            const bar = symBars?.find((b: AlpacaBar) => b.t === timestamp);
            const lastBar = bar || symBars?.[symBars.length - 1];
            if (lastBar) {
              const exit = this.positionSim.forceClosePosition(
                pos.symbol, lastBar.c, lastBar.t, "force_close_eod"
              );
              if (exit) {
                this.finalizeTradeExit(exit);
                this.equity += exit.pnl;
                dayState.dailyPnl += exit.pnl;
              }
            }
          }
          // Clear any remaining that couldn't be matched
          this.positionSim.clear();
        }
        prevTimestamp = timestamp;
        continue;
      }

      // ─── Strategy Evaluation on 5-min bars ───
      const inEntryWindow = minutesET >= this.config.entryWindowStartMinutes &&
        minutesET < this.config.entryWindowEndMinutes;

      if (!inEntryWindow) {
        prevTimestamp = timestamp;
        continue;
      }

      // Check daily limits
      if (dayState.entriesCount >= this.config.maxEntriesPerDay) {
        prevTimestamp = timestamp;
        continue;
      }
      if (dayState.dailyPnl <= this.config.dailyMaxLoss) {
        prevTimestamp = timestamp;
        continue;
      }
      if (dayState.dailyPnl >= this.config.dailyMaxProfit) {
        prevTimestamp = timestamp;
        continue;
      }
      if (dayState.consecutiveLosses >= this.config.maxConsecutiveLosses) {
        prevTimestamp = timestamp;
        continue;
      }

      // Evaluate regime filter
      const regime = this.evaluateRegime(spyAllBars, timestamp);
      if (!regime.allowed) {
        prevTimestamp = timestamp;
        continue;
      }

      // Evaluate strategies for each symbol
      for (const symbol of this.config.symbols) {
        if (this.positionSim.hasPosition(symbol)) continue;
        if (this.positionSim.positionCount() >= this.config.maxOpenPositions) break;

        // Check cooldown
        const cooldownUntil = dayState.cooldownMap.get(symbol);
        if (cooldownUntil && new Date(timestamp).getTime() < cooldownUntil) continue;

        // Get all 5-min bars up to this timestamp for this symbol
        const symbolAllBars = allBars5min.get(symbol);
        if (!symbolAllBars) continue;

        const barsUpToNow = symbolAllBars.filter(b => b.t <= timestamp);
        if (barsUpToNow.length < this.config.warmupBars) continue;

        const currentBar = barsUpToNow[barsUpToNow.length - 1];
        if (currentBar.t !== timestamp) {
          // No bar for this symbol at this timestamp
          continue;
        }

        // Compute indicators
        const indicators = this.computeIndicators(barsUpToNow);
        if (!indicators) continue;

        // ATR filter: skip symbols with insufficient intraday range
        // If ATR/price < 0.15%, the symbol doesn't move enough to hit targets
        const atrPct = (indicators.atr14 / currentBar.c) * 100;
        if (atrPct < 0.15) continue;

        // Get today's bars for this symbol (for VWAP/ORB which need intraday bars)
        const todayBars = barsUpToNow.filter(b => {
          const barDate = SimulatedClock.barDateET(b.t);
          return barDate === day.date;
        });

        // Calculate cumulative volume for the day
        const dailyVolume = todayBars.reduce((sum, b) => sum + b.v, 0);

        // Build SymbolData for strategy evaluation
        const symbolData = {
          symbol,
          currentPrice: currentBar.c,
          bid: currentBar.c - (currentBar.c * 0.0001), // Estimate bid/ask
          ask: currentBar.c + (currentBar.c * 0.0001),
          volume: dailyVolume,
          indicators,
          bars: todayBars,
        };

        // Evaluate strategies — use regime-adjusted confidence threshold
        const signals = evaluateStrategies(symbolData);
        const bestSignal = selectBestSignal(signals, regime.minConfidence);

        if (!bestSignal || bestSignal.side === "none") continue;

        // Per-strategy daily trend gate: block ORB when daily SPY trend is bearish
        // ORB breakouts fail in choppy/bearish markets; EMA_CROSSOVER is robust across regimes
        if (bestSignal.strategyName === "ORB" && !regime.dailyTrendBullish) continue;

        // Only allow strategies that are enabled in config
        if (!this.config.strategies.includes(bestSignal.strategyName as any)) continue;

        // Calculate position size
        const sizing = this.calculatePositionSize(
          symbol,
          bestSignal.entrySignalPrice,
          indicators.atr14,
          this.equity,
          bestSignal.side as "buy" | "sell"
        );

        if (sizing.qty <= 0) continue;

        // Apply slippage
        const fill = applySlippage(
          bestSignal.entrySignalPrice,
          bestSignal.side as "buy" | "sell",
          this.config.slippageMode,
          this.config.slippageFixedBps,
          this.config.slippagePctOfSpread,
          currentBar
        );

        // Create trade record
        const trade: BacktestTrade = {
          id: this.tradeIdCounter++,
          symbol,
          strategyName: bestSignal.strategyName,
          side: bestSignal.side as "buy" | "sell",
          signalPrice: bestSignal.entrySignalPrice,
          entryPrice: fill.adjustedPrice,
          entryTime: timestamp,
          quantity: sizing.qty,
          confidence: bestSignal.confidence,
          initialStopPrice: sizing.stopPrice,
          targetPrice: sizing.targetPrice,
          exitPrice: null,
          exitTime: null,
          exitReason: null,
          exits: [],
          realizedPnl: null,
          realizedPnlPct: null,
          holdingTimeMinutes: null,
          rMultiple: null,
          slippageBps: fill.slippageBps,
          regime: regime.label,
        };

        this.trades.push(trade);

        // Open simulated position
        const simPos: SimulatedPosition = {
          symbol,
          entryPrice: fill.adjustedPrice,
          stopPrice: sizing.stopPrice,
          targetPrice: sizing.targetPrice,
          quantity: sizing.qty,
          originalQuantity: sizing.qty,
          startTimeMs: new Date(timestamp).getTime(),
          partialTaken: false,
          rValue: Math.abs(fill.adjustedPrice - sizing.stopPrice),
          trailingActive: false,
          breakevenArmed: false,
          strategyName: bestSignal.strategyName,
          confidence: bestSignal.confidence,
          tradeIndex: trade.id,
          side: bestSignal.side as "buy" | "sell",
        };

        this.positionSim.openPosition(simPos);

        // Set cooldown
        const cooldownMs = this.config.cooldownMinutes * 60 * 1000;
        dayState.cooldownMap.set(symbol, new Date(timestamp).getTime() + cooldownMs);
        dayState.entriesCount++;
      }

      prevTimestamp = timestamp;
    }

    // Force close any positions still open at end of day data
    if (this.positionSim.positionCount() > 0) {
      const positions = this.positionSim.getOpenPositions();
      for (const pos of positions) {
        const symBars = dayBars5min.get(pos.symbol);
        const lastBar = symBars?.[symBars.length - 1];
        if (lastBar) {
          const exit = this.positionSim.forceClosePosition(
            pos.symbol, lastBar.c, lastBar.t, "force_close_eod"
          );
          if (exit) {
            this.finalizeTradeExit(exit);
            this.equity += exit.pnl;
            dayState.dailyPnl += exit.pnl;
          }
        }
      }
      this.positionSim.clear();
    }

    return barsProcessed;
  }

  /**
   * Process 1-min bars for position management between two 5-min bar timestamps.
   * Processes each symbol's bars independently to avoid cross-symbol contamination.
   */
  private processPositionManagement(
    prevTimestamp: string | null,
    currentTimestamp: string,
    symbol1minBars: Map<string, AlpacaBar[]>,
    symbol1minIndex: Map<string, number>,
    dayState: DayState
  ): TradeExit[] {
    const allExits: TradeExit[] = [];

    // Process each symbol with open positions independently
    const openSymbols = this.positionSim.getOpenPositions().map(p => p.symbol);

    for (const symbol of openSymbols) {
      if (!this.positionSim.hasPosition(symbol)) continue; // May have been closed by earlier iteration

      const bars1min = symbol1minBars.get(symbol) || [];
      let idx = symbol1minIndex.get(symbol) || 0;

      // Find 1-min bars between prev and current 5-min timestamps for this symbol
      while (idx < bars1min.length && bars1min[idx].t < currentTimestamp) {
        const bar1m = bars1min[idx];

        // Skip bars before prev timestamp
        if (prevTimestamp && bar1m.t <= prevTimestamp) {
          idx++;
          continue;
        }

        const bar1mTime = new Date(bar1m.t).getTime();

        // Build recent closes for trailing stop EMA-9 (from this symbol's bars only)
        const recentCloses: number[] = [];
        const lookback = Math.min(idx + 1, 15);
        for (let i = idx - lookback + 1; i <= idx; i++) {
          if (i >= 0) recentCloses.push(bars1min[i].c);
        }

        // Process this bar for this symbol's position only
        const exit = this.positionSim.processBarForSymbol(
          symbol,
          bar1m,
          recentCloses,
          bar1mTime,
          {
            partialProfitPct: this.config.partialProfitPct,
            partialProfitRMultiple: this.config.partialProfitRMultiple,
            tradeTimeoutMinutes: this.config.tradeTimeoutMinutes,
            breakevenTriggerPct: this.config.breakevenTriggerPct,
            breakevenOffsetPct: this.config.breakevenOffsetPct,
          }
        );

        if (exit) {
          this.finalizeTradeExit(exit);
          allExits.push(exit);

          // Track consecutive losses for daily safety
          if (exit.pnl < 0) {
            dayState.consecutiveLosses++;
          } else {
            dayState.consecutiveLosses = 0;
          }

          // If position was fully closed, stop processing this symbol's bars
          if (!this.positionSim.hasPosition(symbol)) {
            idx++;
            break;
          }
        }

        idx++;
      }

      symbol1minIndex.set(symbol, idx);
    }

    return allExits;
  }

  /**
   * Finalize a trade exit - update the BacktestTrade record.
   * Matches exit to trade by symbol.
   */
  private finalizeTradeExit(exit: TradeExit): void {
    // Find the matching open trade by symbol (most recent unclosed trade for this symbol)
    for (let i = this.trades.length - 1; i >= 0; i--) {
      const trade = this.trades[i];

      // Must match symbol and be an unclosed trade
      if (trade.symbol !== exit.symbol) continue;
      if (trade.exitReason !== null && exit.reason !== "take_profit") continue;

      // For partial exits (take_profit), allow adding to a trade that already has partial exits
      // For full exits, only match trades without a final exitReason
      if (trade.exitReason !== null) continue;

      trade.exits.push(exit);

      // Calculate aggregate P&L from all exits
      const totalPnl = trade.exits.reduce((sum, e) => sum + e.pnl, 0);
      const totalQty = trade.exits.reduce((sum, e) => sum + e.quantity, 0);
      const avgExitPrice = trade.exits.reduce((sum, e) => sum + e.price * e.quantity, 0) / totalQty;

      // If this is the final exit (not a partial take_profit, or full position closed)
      if (exit.reason !== "take_profit" || totalQty >= trade.quantity) {
        trade.exitPrice = avgExitPrice;
        trade.exitTime = exit.time;
        trade.exitReason = exit.reason as any;
        trade.realizedPnl = totalPnl;
        trade.realizedPnlPct = (totalPnl / (trade.entryPrice * trade.quantity)) * 100;
        trade.holdingTimeMinutes = SimulatedClock.minutesBetween(
          new Date(trade.entryTime),
          new Date(exit.time)
        );
        const riskPerShare = Math.abs(trade.entryPrice - trade.initialStopPrice);
        trade.rMultiple = riskPerShare > 0 ? totalPnl / (riskPerShare * trade.quantity) : 0;
      }

      break;
    }
  }

  /**
   * Evaluate regime filter using SPY EMA9/EMA21 (intraday) + optional daily trend (close > 20-day SMA).
   * Returns both overall regime and whether daily trend supports ORB breakout trading.
   */
  private evaluateRegime(
    spyAllBars: AlpacaBar[],
    upToTimestamp: string
  ): { allowed: boolean; label: RegimeLabel; minConfidence: number; dailyTrendBullish: boolean } {
    // Layer 1: Daily trend filter — SPY close vs 20-day SMA
    const DAILY_SMA_PERIOD = 20;
    let dailyTrendBullish = true;
    if (this.spyDailyCloses.length >= DAILY_SMA_PERIOD) {
      const recentCloses = this.spyDailyCloses.slice(-DAILY_SMA_PERIOD);
      const sma20 = recentCloses.reduce((a, b) => a + b, 0) / DAILY_SMA_PERIOD;
      const latestClose = this.spyDailyCloses[this.spyDailyCloses.length - 1];
      dailyTrendBullish = latestClose >= sma20;
    }

    // Layer 2: Intraday EMA9/EMA21 filter (existing logic)
    const barsUpTo = spyAllBars.filter(b => b.t <= upToTimestamp);
    if (barsUpTo.length < 21) return { allowed: true, label: "chop", minConfidence: 75, dailyTrendBullish };

    const closes = barsUpTo.map(b => b.c);
    const ema9Array = ema(closes, 9);
    const ema21Array = ema(closes, 21);
    const latestEma9 = ema9Array[ema9Array.length - 1];
    const latestEma21 = ema21Array[ema21Array.length - 1];

    if (!Number.isFinite(latestEma9) || !Number.isFinite(latestEma21)) {
      return { allowed: false, label: "chop", minConfidence: 100, dailyTrendBullish };
    }

    const label = determineRegimeLabel(latestEma9, latestEma21);
    const isBullish = latestEma9 >= latestEma21;

    // Strict bull-only on intraday: chop and bear both blocked
    if (!isBullish || label === "chop") {
      return { allowed: false, label, minConfidence: 100, dailyTrendBullish };
    }

    // Intraday bullish — trade normally (daily filter applied per-strategy elsewhere)
    return { allowed: true, label, minConfidence: this.config.minConfidence, dailyTrendBullish };
  }

  /**
   * Compute indicators from a window of 5-min bars.
   */
  private computeIndicators(bars: AlpacaBar[]): (Tier1Indicators & Partial<Tier2Indicators>) | null {
    if (bars.length < TIER_1_MIN_BARS) return null;

    const closes = bars.map(b => b.c);
    const highs = bars.map(b => b.h);
    const lows = bars.map(b => b.l);

    const ema9Array = ema(closes, 9);
    const ema20Array = ema(closes, 20);
    const rsi14Array = rsi(closes, 14);
    const atr14Array = atr(highs, lows, closes, 14);

    const latest = bars.length - 1;

    const indicators: Tier1Indicators & Partial<Tier2Indicators> = {
      ema9: ema9Array[latest],
      ema20: ema20Array[latest],
      rsi14: rsi14Array[latest],
      atr14: atr14Array[latest],
      latestClose: closes[latest],
    };

    // Validate
    if (!Number.isFinite(indicators.ema9) || !Number.isFinite(indicators.ema20) ||
      !Number.isFinite(indicators.rsi14) || !Number.isFinite(indicators.atr14)) {
      return null;
    }

    // Add MACD if we have enough bars (Tier 2)
    if (bars.length >= TIER_2_MIN_BARS) {
      const macdResult = macd(closes, 12, 26, 9);
      const macdLine = macdResult.macd[latest];
      const macdSignal = macdResult.signal[latest];

      if (Number.isFinite(macdLine) && Number.isFinite(macdSignal)) {
        indicators.macdLine = macdLine;
        indicators.macdSignal = macdSignal;
      }
    }

    return indicators;
  }

  /**
   * Calculate position size using ATR-based sizing.
   * Pure reimplementation of riskEngine.calculateATRPositionSize.
   */
  private calculatePositionSize(
    symbol: string,
    entryPrice: number,
    atrValue: number,
    equity: number,
    side: "buy" | "sell"
  ): { qty: number; stopPrice: number; targetPrice: number } {
    const config = this.config;

    // Calculate stop distance from ATR
    let stopMultiplier = config.atrStopMultiplier;

    // Time-of-day adjustment: widen stops in first 30 minutes
    const minutesET = this.clock.getMinutesSinceMidnightET();
    const minutesSinceOpen = minutesET - (9 * 60 + 30);
    if (minutesSinceOpen >= 0 && minutesSinceOpen < 30) {
      stopMultiplier *= 1.25; // EARLY_STOP_MULTIPLIER
    }

    let stopDistance = atrValue * stopMultiplier;

    // Enforce hard caps
    const stopPct = stopDistance / entryPrice;
    if (stopPct < config.minStopPct) {
      stopDistance = entryPrice * config.minStopPct;
    } else if (stopPct > config.maxStopPct) {
      stopDistance = entryPrice * config.maxStopPct;
    }

    // Calculate risk amount and position size
    const riskAmount = equity * config.riskPerTradePct;
    let qty = Math.floor(riskAmount / stopDistance);
    qty = Math.max(1, qty);

    // Enforce max position cap
    let notional = qty * entryPrice;
    const maxNotional = equity * config.maxPositionPct;
    if (notional > maxNotional) {
      qty = Math.floor(maxNotional / entryPrice);
      qty = Math.max(1, qty);
    }

    // Calculate stop and target
    let stopPrice: number;
    let targetPrice: number;

    if (side === "buy") {
      stopPrice = entryPrice - stopDistance;
      targetPrice = entryPrice + (stopDistance * config.rewardRiskRatio);
    } else {
      stopPrice = entryPrice + stopDistance;
      targetPrice = entryPrice - (stopDistance * config.rewardRiskRatio);
    }

    return { qty, stopPrice, targetPrice };
  }

  /**
   * Index all bars by ET date for efficient day-by-day processing.
   * Returns Map<date, Map<symbol, AlpacaBar[]>>
   */
  private indexBarsByDate(
    data: Map<string, SymbolBarData>,
    timeframe: "5min" | "1min"
  ): Map<string, Map<string, AlpacaBar[]>> {
    const result = new Map<string, Map<string, AlpacaBar[]>>();

    for (const [symbol, symbolData] of Array.from(data.entries())) {
      const bars = timeframe === "5min" ? symbolData.bars5min : symbolData.bars1min;

      for (const bar of bars) {
        const dateStr = SimulatedClock.barDateET(bar.t);
        if (!result.has(dateStr)) {
          result.set(dateStr, new Map());
        }
        const dayMap = result.get(dateStr)!;
        if (!dayMap.has(symbol)) {
          dayMap.set(symbol, []);
        }
        dayMap.get(symbol)!.push(bar);
      }
    }

    return result;
  }
}
