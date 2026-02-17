/**
 * Metrics Calculator - Compute aggregate and per-strategy performance metrics
 *
 * Computes: win rate, profit factor, Sharpe, Sortino, max drawdown,
 * expectancy, R-multiples, and more.
 */

import type { BacktestTrade, BacktestMetrics, StrategyMetrics } from "./types";

export class MetricsCalculator {
  /**
   * Compute all aggregate metrics from trades and equity curve.
   */
  static compute(
    trades: BacktestTrade[],
    equityCurve: { date: string; equity: number }[],
    initialEquity: number,
    totalTradingDays: number
  ): BacktestMetrics {
    const closedTrades = trades.filter(t => t.realizedPnl !== null);
    return MetricsCalculator.computeFromTrades(closedTrades, equityCurve, initialEquity, totalTradingDays);
  }

  /**
   * Compute per-strategy breakdown.
   */
  static computePerStrategy(
    trades: BacktestTrade[],
    equityCurve: { date: string; equity: number }[],
    initialEquity: number,
    totalTradingDays: number
  ): StrategyMetrics[] {
    const closedTrades = trades.filter(t => t.realizedPnl !== null);
    const strategyNames = Array.from(new Set(closedTrades.map(t => t.strategyName)));

    return strategyNames.map(name => {
      const stratTrades = closedTrades.filter(t => t.strategyName === name);
      const metrics = MetricsCalculator.computeFromTrades(
        stratTrades,
        equityCurve,
        initialEquity,
        totalTradingDays
      );
      return { ...metrics, strategyName: name };
    });
  }

  private static computeFromTrades(
    trades: BacktestTrade[],
    equityCurve: { date: string; equity: number }[],
    initialEquity: number,
    totalTradingDays: number
  ): BacktestMetrics {
    if (trades.length === 0) {
      return MetricsCalculator.emptyMetrics(equityCurve, initialEquity, totalTradingDays);
    }

    const winners = trades.filter(t => (t.realizedPnl || 0) > 0);
    const losers = trades.filter(t => (t.realizedPnl || 0) < 0);
    const breakeven = trades.filter(t => (t.realizedPnl || 0) === 0);

    const winPnls = winners.map(t => t.realizedPnl!);
    const lossPnls = losers.map(t => t.realizedPnl!);

    const grossProfit = winPnls.reduce((s, v) => s + v, 0);
    const grossLoss = Math.abs(lossPnls.reduce((s, v) => s + v, 0));
    const totalPnl = grossProfit - grossLoss;

    const avgWin = winPnls.length > 0 ? grossProfit / winPnls.length : 0;
    const avgLoss = lossPnls.length > 0 ? grossLoss / lossPnls.length : 0;

    const finalEquity = equityCurve.length > 0
      ? equityCurve[equityCurve.length - 1].equity
      : initialEquity + totalPnl;

    const totalReturn = ((finalEquity - initialEquity) / initialEquity) * 100;
    const annualizedReturn = totalTradingDays > 0
      ? (Math.pow(finalEquity / initialEquity, 252 / totalTradingDays) - 1) * 100
      : 0;

    // Daily returns for Sharpe/Sortino
    const dailyReturns = MetricsCalculator.computeDailyReturns(equityCurve);
    const dailyReturnValues = dailyReturns.map(d => d.returnPct);

    // Drawdown
    const dd = MetricsCalculator.computeMaxDrawdown(equityCurve);

    // Equity curve with drawdown
    const equityCurveWithDD = MetricsCalculator.addDrawdownToEquityCurve(equityCurve);

    // R-multiples
    const rMultiples = trades
      .map(t => t.rMultiple)
      .filter((r): r is number => r !== null && Number.isFinite(r));

    // Days with trades
    const tradeDates = new Set(trades.map(t => t.entryTime.substring(0, 10)));

    // Holding time
    const holdingTimes = trades
      .map(t => t.holdingTimeMinutes)
      .filter((h): h is number => h !== null);

    return {
      totalTrades: trades.length,
      winningTrades: winners.length,
      losingTrades: losers.length,
      winRate: (winners.length / trades.length) * 100,
      avgWin,
      avgLoss,
      largestWin: winPnls.length > 0 ? Math.max(...winPnls) : 0,
      largestLoss: lossPnls.length > 0 ? Math.min(...lossPnls) : 0,
      profitFactor: grossLoss > 0 ? grossProfit / grossLoss : grossProfit > 0 ? Infinity : 0,
      expectancy: totalPnl / trades.length,
      expectancyR: rMultiples.length > 0
        ? rMultiples.reduce((s, v) => s + v, 0) / rMultiples.length
        : 0,

      totalReturn,
      totalReturnDollars: totalPnl,
      annualizedReturn,

      maxDrawdownDollars: dd.dollars,
      maxDrawdownPercent: dd.percent,
      maxDrawdownDurationDays: dd.durationDays,

      sharpeRatio: MetricsCalculator.sharpeRatio(dailyReturnValues),
      sortinoRatio: MetricsCalculator.sortinoRatio(dailyReturnValues),
      calmarRatio: dd.percent > 0 ? annualizedReturn / dd.percent : 0,

      avgHoldingTimeMinutes: holdingTimes.length > 0
        ? holdingTimes.reduce((s, v) => s + v, 0) / holdingTimes.length
        : 0,
      totalTradingDays,
      daysWithTrades: tradeDates.size,
      avgTradesPerDay: tradeDates.size > 0 ? trades.length / tradeDates.size : 0,

      equityCurve: equityCurveWithDD,
      dailyReturns,
    };
  }

  /**
   * Sharpe ratio (annualized from daily returns)
   * = (mean / std) * sqrt(252)
   */
  static sharpeRatio(dailyReturns: number[]): number {
    if (dailyReturns.length < 2) return 0;

    const mean = dailyReturns.reduce((s, v) => s + v, 0) / dailyReturns.length;
    const variance = dailyReturns.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / (dailyReturns.length - 1);
    const std = Math.sqrt(variance);

    if (std === 0) return 0;
    return (mean / std) * Math.sqrt(252);
  }

  /**
   * Sortino ratio (uses only downside deviation)
   * = (mean / downside_std) * sqrt(252)
   */
  static sortinoRatio(dailyReturns: number[]): number {
    if (dailyReturns.length < 2) return 0;

    const mean = dailyReturns.reduce((s, v) => s + v, 0) / dailyReturns.length;
    const downsideReturns = dailyReturns.filter(r => r < 0);

    if (downsideReturns.length === 0) return mean > 0 ? Infinity : 0;

    const downsideVariance = downsideReturns.reduce((s, v) => s + Math.pow(v, 2), 0) / downsideReturns.length;
    const downsideStd = Math.sqrt(downsideVariance);

    if (downsideStd === 0) return 0;
    return (mean / downsideStd) * Math.sqrt(252);
  }

  /**
   * Maximum drawdown from equity curve.
   */
  static computeMaxDrawdown(
    equityCurve: { date: string; equity: number }[]
  ): { dollars: number; percent: number; durationDays: number } {
    if (equityCurve.length === 0) {
      return { dollars: 0, percent: 0, durationDays: 0 };
    }

    let peak = equityCurve[0].equity;
    let maxDdDollars = 0;
    let maxDdPercent = 0;
    let maxDdDuration = 0;

    let currentDdStart = 0;

    for (let i = 0; i < equityCurve.length; i++) {
      const equity = equityCurve[i].equity;

      if (equity > peak) {
        peak = equity;
        currentDdStart = i;
      }

      const ddDollars = peak - equity;
      const ddPercent = (ddDollars / peak) * 100;

      if (ddDollars > maxDdDollars) {
        maxDdDollars = ddDollars;
        maxDdPercent = ddPercent;
        maxDdDuration = i - currentDdStart;
      }
    }

    return {
      dollars: maxDdDollars,
      percent: maxDdPercent,
      durationDays: maxDdDuration,
    };
  }

  /**
   * Compute daily returns from equity curve.
   */
  static computeDailyReturns(
    equityCurve: { date: string; equity: number }[]
  ): { date: string; returnPct: number; pnl: number }[] {
    if (equityCurve.length < 2) return [];

    const returns: { date: string; returnPct: number; pnl: number }[] = [];
    for (let i = 1; i < equityCurve.length; i++) {
      const prevEquity = equityCurve[i - 1].equity;
      const curEquity = equityCurve[i].equity;
      const pnl = curEquity - prevEquity;
      const returnPct = (pnl / prevEquity) * 100;
      returns.push({ date: equityCurve[i].date, returnPct, pnl });
    }
    return returns;
  }

  /**
   * Add drawdown percentage to each equity curve point.
   */
  static addDrawdownToEquityCurve(
    equityCurve: { date: string; equity: number }[]
  ): { date: string; equity: number; drawdownPct: number }[] {
    let peak = equityCurve[0]?.equity || 0;
    return equityCurve.map(point => {
      if (point.equity > peak) peak = point.equity;
      const drawdownPct = peak > 0 ? ((peak - point.equity) / peak) * 100 : 0;
      return { ...point, drawdownPct };
    });
  }

  private static emptyMetrics(
    equityCurve: { date: string; equity: number }[],
    initialEquity: number,
    totalTradingDays: number
  ): BacktestMetrics {
    return {
      totalTrades: 0, winningTrades: 0, losingTrades: 0,
      winRate: 0, avgWin: 0, avgLoss: 0, largestWin: 0, largestLoss: 0,
      profitFactor: 0, expectancy: 0, expectancyR: 0,
      totalReturn: 0, totalReturnDollars: 0, annualizedReturn: 0,
      maxDrawdownDollars: 0, maxDrawdownPercent: 0, maxDrawdownDurationDays: 0,
      sharpeRatio: 0, sortinoRatio: 0, calmarRatio: 0,
      avgHoldingTimeMinutes: 0, totalTradingDays, daysWithTrades: 0, avgTradesPerDay: 0,
      equityCurve: equityCurve.map(e => ({ ...e, drawdownPct: 0 })),
      dailyReturns: [],
    };
  }
}
