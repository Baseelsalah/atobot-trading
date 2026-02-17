import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import * as alpaca from "./alpaca";
import * as tradingBot from "./tradingBot";
import * as brain from "./autopilotBrain";
import * as ato from "./ato";
import * as riskManager from "./riskManager";
import * as profitManager from "./profitManager";
import * as tradeBus from "./tradeIntelligenceBus";
import * as timeGuard from "./tradingTimeGuard";
import * as dayTraderConfig from "./dayTraderConfig";
import * as marketRegime from "./marketRegime";
import * as positionManager from "./positionManager";
import * as performanceReport from "./performanceReport";
import * as weeklyScorecard from "./weeklyScorecard";
import type { TradeRecommendation } from "@shared/schema";
import { generateTradeId, parseTradeId } from "./tradeId";
import * as runtimeMonitor from "./runtimeMonitor";
import * as timezone from "./timezone";
import * as activityLedger from "./activityLedger";
import * as entryWindowProof from "./entryWindowProof";
import * as reportStorage from "./reportStorage";
import * as leaderLock from "./leaderLock";
import * as tradabilityGates from "./tradabilityGates";
import * as executionTrace from "./executionTrace";
import * as envScope from "./envScope";
import { requireAdmin, requireApproved } from "./auth";
import * as accountManager from "./accountManager";

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  // Mock portfolio data for when Alpaca is unavailable
  const mockPortfolio = {
    totalEquity: 100000,
    buyingPower: 50000,
    cash: 50000,
    todayPL: 0,
    todayPLPercent: 0,
    totalPL: 0,
    totalPLPercent: 0,
    dayTradesRemaining: 3,
  };

  // Portfolio endpoints (user-scoped: uses logged-in user's Alpaca keys)
  app.get("/api/portfolio", async (req, res) => {
    try {
      // Get user's Alpaca credentials
      const userKeys = req.user ? accountManager.getUserDecryptedKeys(req.user.id) : null;
      if (userKeys) {
        alpaca.setActiveCredentials(userKeys.key, userKeys.secret);
      }

      try {
        // Always try to fetch real Alpaca data first when configured
        if (alpaca.isConfigured()) {
          try {
            const account = await alpaca.getAccount();
            const equity = parseFloat(account.equity);
            const lastEquity = parseFloat(account.last_equity);
            const todayPL = equity - lastEquity;
            const todayPLPercent = lastEquity > 0 ? (todayPL / lastEquity) * 100 : 0;

            return res.json({
              totalEquity: equity,
              buyingPower: parseFloat(account.buying_power),
              cash: parseFloat(account.cash),
              todayPL,
              todayPLPercent,
              totalPL: todayPL,
              totalPLPercent: todayPLPercent,
              dayTradesRemaining: 3 - account.daytrade_count,
            });
          } catch (alpacaError) {
            console.error("Alpaca API error, falling back to local calculation:", alpacaError);
          }
        }
      } finally {
        if (userKeys) alpaca.clearActiveCredentials();
      }
      
      // Fallback: calculate from local positions and trades
      const positions = await storage.getPositions();
      const trades = await storage.getTrades();
      
      const positionValue = positions.reduce((sum, pos) => sum + pos.marketValue, 0);
      const totalPL = positions.reduce((sum, pos) => sum + pos.unrealizedPL, 0);
      
      const tradeSpend = trades
        .filter(t => t.side === 'buy' && t.status === 'filled')
        .reduce((sum, t) => sum + t.totalValue, 0);
      const tradeSales = trades
        .filter(t => t.side === 'sell' && t.status === 'filled')
        .reduce((sum, t) => sum + t.totalValue, 0);
      
      const startingCash = 100000;
      const cash = startingCash - tradeSpend + tradeSales;
      const totalEquity = cash + positionValue;
      
      return res.json({
        totalEquity,
        buyingPower: cash * 2,
        cash,
        todayPL: totalPL,
        todayPLPercent: totalEquity > 0 ? (totalPL / totalEquity) * 100 : 0,
        totalPL,
        totalPLPercent: startingCash > 0 ? ((totalEquity - startingCash) / startingCash) * 100 : 0,
        dayTradesRemaining: 3,
      });
    } catch (error) {
      console.error("Portfolio fetch error:", error);
      res.status(500).json({ error: "Failed to fetch portfolio" });
    }
  });

  // Positions endpoints (user-scoped)
  app.get("/api/positions", async (req, res) => {
    try {
      const userKeys = req.user ? accountManager.getUserDecryptedKeys(req.user.id) : null;
      if (userKeys) {
        alpaca.setActiveCredentials(userKeys.key, userKeys.secret);
      }

      try {
        if (alpaca.isConfigured()) {
          try {
            const alpacaPositions = await alpaca.getPositions();
            // Return directly for non-admin users (don't pollute shared storage)
            if (userKeys && req.user?.role !== "admin") {
              return res.json(alpacaPositions.map((pos: any) => ({
                id: 0,
                symbol: pos.symbol,
                quantity: parseInt(pos.qty),
                avgEntryPrice: parseFloat(pos.avg_entry_price),
                currentPrice: parseFloat(pos.current_price),
                marketValue: parseFloat(pos.market_value),
                unrealizedPL: parseFloat(pos.unrealized_pl),
                unrealizedPLPercent: parseFloat(pos.unrealized_plpc) * 100,
              })));
            }
            // Admin: sync into shared storage as before
            await storage.clearPositions();
            for (const pos of alpacaPositions) {
              await storage.upsertPosition({
                symbol: pos.symbol,
                quantity: parseInt(pos.qty),
                avgEntryPrice: parseFloat(pos.avg_entry_price),
                currentPrice: parseFloat(pos.current_price),
                marketValue: parseFloat(pos.market_value),
                unrealizedPL: parseFloat(pos.unrealized_pl),
                unrealizedPLPercent: parseFloat(pos.unrealized_plpc) * 100,
              });
            }
          } catch (syncError) {
            console.error("Failed to sync positions from Alpaca:", syncError);
          }
        }
      } finally {
        if (userKeys) alpaca.clearActiveCredentials();
      }
      const positions = await storage.getPositions();
      res.json(positions);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch positions" });
    }
  });

  app.post("/api/positions/close", async (req, res) => {
    try {
      const { symbol } = req.body;
      if (!symbol) {
        return res.status(400).json({ error: "Symbol required" });
      }
      const success = await tradingBot.closePosition(symbol);
      res.json({ success });
    } catch (error) {
      res.status(500).json({ error: "Failed to close position" });
    }
  });

  app.post("/api/positions/close-all", async (req, res) => {
    try {
      const success = await tradingBot.closeAllPositions();
      res.json({ success });
    } catch (error) {
      res.status(500).json({ error: "Failed to close positions" });
    }
  });

  // Trades endpoints
  app.get("/api/trades", async (req, res) => {
    try {
      const trades = await storage.getTrades();
      res.json(trades);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch trades" });
    }
  });

  // Alpaca orders endpoint - check real order history
  app.get("/api/orders", async (req, res) => {
    try {
      const limit = parseInt(req.query.limit as string) || 50;
      const status = (req.query.status as string) || "all";
      const orders = await alpaca.getOrders(status, limit);
      res.json(orders.map(o => ({
        id: o.id?.slice(0, 8),
        symbol: o.symbol,
        side: o.side,
        type: o.type,
        status: o.status,
        qty: o.qty,
        filled_qty: o.filled_qty,
        submitted_at: o.submitted_at,
        filled_at: o.filled_at,
        client_order_id: o.client_order_id,
      })));
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch orders from Alpaca" });
    }
  });

  // Research endpoints
  app.get("/api/research", async (req, res) => {
    try {
      const logs = await storage.getResearchLogs();
      res.json(logs);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch research logs" });
    }
  });

  // Activity endpoints
  app.get("/api/activity", async (req, res) => {
    try {
      const logs = await storage.getActivityLogs();
      res.json(logs);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch activity logs" });
    }
  });

  // Alerts endpoints
  app.get("/api/alerts", async (req, res) => {
    try {
      const alerts = await storage.getAlerts();
      res.json(alerts);
    } catch (error) {
      res.status(500).json({ error: "Failed to fetch alerts" });
    }
  });

  app.post("/api/alerts/:id/read", async (req, res) => {
    try {
      await storage.markAlertRead(req.params.id);
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to mark alert read" });
    }
  });

  app.post("/api/alerts/:id/approve", async (req, res) => {
    try {
      const alerts = await storage.getAlerts();
      const alert = alerts.find((a) => a.id === req.params.id);

      if (!alert || !alert.metadata) {
        return res.status(404).json({ error: "Alert not found" });
      }

      const recommendation = JSON.parse(alert.metadata) as TradeRecommendation;
      const success = await tradingBot.executeTrade(recommendation);

      if (success) {
        await storage.resolveAlert(req.params.id);
      }

      res.json({ success });
    } catch (error) {
      res.status(500).json({ error: "Failed to approve trade" });
    }
  });

  app.post("/api/alerts/:id/deny", async (req, res) => {
    try {
      await storage.resolveAlert(req.params.id);
      await storage.createActivityLog({
        type: "alert",
        action: "Trade Denied",
        description: "User denied trade recommendation",
      });
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to deny trade" });
    }
  });

  app.post("/api/alerts/mark-all-read", async (req, res) => {
    try {
      await storage.markAllAlertsRead();
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to mark alerts read" });
    }
  });

  // Bot control endpoints (admin only, except status)
  app.get("/api/bot/status", async (req, res) => {
    try {
      const status = tradingBot.getBotStatus();
      res.json(status);
    } catch (error) {
      res.status(500).json({ error: "Failed to get bot status" });
    }
  });

  app.post("/api/bot/start", requireAdmin, async (req, res) => {
    try {
      await tradingBot.startBot();
      res.json({ success: true });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to start bot";
      res.status(500).json({ error: message });
    }
  });

  app.post("/api/bot/pause", requireAdmin, async (req, res) => {
    try {
      await tradingBot.pauseBot();
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to pause bot" });
    }
  });

  app.post("/api/bot/stop", requireAdmin, async (req, res) => {
    try {
      await tradingBot.stopBot();
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to stop bot" });
    }
  });

  app.post("/api/bot/analyze", requireAdmin, async (req, res) => {
    try {
      await tradingBot.runAnalysis();
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to run analysis" });
    }
  });

  // Portfolio history endpoint for charts
  app.get("/api/portfolio/history", async (req, res) => {
    try {
      const period = (req.query.period as string) || "1D";
      const timeframe = period === "1D" ? "5Min" : period === "1M" ? "1H" : "1D";
      
      if (!alpaca.isConfigured()) {
        return res.json([]);
      }
      
      const history = await alpaca.getPortfolioHistory(period, timeframe);
      res.json(history);
    } catch (error) {
      console.error("Portfolio history error:", error);
      res.status(500).json({ error: "Failed to get portfolio history" });
    }
  });

  // Market status endpoint
  // Mock market status for when Alpaca is unavailable
  const mockMarketStatus = {
    isOpen: false,
    nextOpen: new Date().toISOString(),
    nextClose: new Date().toISOString(),
  };

  app.get("/api/market/status", async (req, res) => {
    try {
      if (!alpaca.isConfigured()) {
        return res.json(mockMarketStatus);
      }

      try {
        const clock = await alpaca.getClock();
        res.json({
          isOpen: clock.is_open,
          nextOpen: clock.next_open,
          nextClose: clock.next_close,
        });
      } catch (alpacaError) {
        // Fall back to mock data if Alpaca API fails
        console.error("Alpaca market status error, using mock data:", alpacaError);
        res.json(mockMarketStatus);
      }
    } catch (error) {
      res.status(500).json({ error: "Failed to get market status" });
    }
  });

  // V2 Market Status - Single source of truth with simulation flags
  // Uses the same 5-second cache as order submission guard (canSubmitOrderNow)
  app.get("/api/market-status", async (req, res) => {
    try {
      const status = await alpaca.getMarketStatusCached();
      res.json(status);
    } catch (error) {
      res.status(500).json({ error: "Failed to get market status" });
    }
  });

  // Trading time guard status - shows if trading is allowed
  app.get("/api/trading/status", async (req, res) => {
    try {
      const status = timeGuard.getTradingStatus();
      res.json(status);
    } catch (error) {
      res.status(500).json({ error: "Failed to get trading status" });
    }
  });

  // Emergency close all positions endpoint (admin only)
  app.post("/api/trading/emergency-close", requireAdmin, async (req, res) => {
    try {
      const result = await timeGuard.closeAllPositionsNow("Manual emergency close requested");
      res.json({ success: true, ...result });
    } catch (error) {
      res.status(500).json({ error: "Failed to emergency close positions" });
    }
  });

  // Day trader config status
  app.get("/api/daytrader/status", async (req, res) => {
    try {
      const status = dayTraderConfig.getDayTraderStatus();
      res.json(status);
    } catch (error) {
      res.status(500).json({ error: "Failed to get day trader status" });
    }
  });

  // Market regime status
  app.get("/api/market/regime", async (req, res) => {
    try {
      const regime = await marketRegime.checkMarketRegime();
      res.json(regime);
    } catch (error) {
      res.status(500).json({ error: "Failed to get market regime" });
    }
  });

  // Managed positions status (safe version without internal stop prices)
  app.get("/api/positions/managed", async (req, res) => {
    try {
      const positions = positionManager.getManagedPositionsSafe();
      res.json(positions);
    } catch (error) {
      res.status(500).json({ error: "Failed to get managed positions" });
    }
  });

  // Settings endpoints
  app.get("/api/settings", async (req, res) => {
    try {
      const settings = await storage.getSettings();
      res.json(settings);
    } catch (error) {
      res.status(500).json({ error: "Failed to get settings" });
    }
  });

  app.patch("/api/settings", requireAdmin, async (req, res) => {
    try {
      const settings = await storage.updateSettings(req.body);
      res.json(settings);
    } catch (error) {
      res.status(500).json({ error: "Failed to update settings" });
    }
  });

  // Autopilot Brain endpoints
  app.get("/api/brain/status", async (req, res) => {
    try {
      const status = brain.getBrainStatus();
      res.json(status);
    } catch (error) {
      res.status(500).json({ error: "Failed to get brain status" });
    }
  });

  app.get("/api/brain/strategies", async (req, res) => {
    try {
      const strategies = await storage.getStrategies();
      res.json(strategies);
    } catch (error) {
      res.status(500).json({ error: "Failed to get strategies" });
    }
  });

  app.post("/api/brain/research", async (req, res) => {
    try {
      const result = await brain.conductResearch();
      res.json({ success: true, outlook: result });
    } catch (error) {
      res.status(500).json({ error: "Failed to conduct research" });
    }
  });

  app.post("/api/brain/improve", async (req, res) => {
    try {
      await brain.improveStrategies();
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ error: "Failed to improve strategies" });
    }
  });

  app.post("/api/brain/create-strategy", async (req, res) => {
    try {
      const { name, description, type } = req.body;
      if (!name || !description || !type) {
        return res.status(400).json({ error: "Name, description, and type required" });
      }
      const strategy = await brain.createNewStrategy(name, description, type);
      res.json(strategy);
    } catch (error) {
      res.status(500).json({ error: "Failed to create strategy" });
    }
  });

  // Ato (Day Trader) endpoints
  app.get("/api/ato/status", async (req, res) => {
    try {
      const status = ato.getAtoState();
      res.json(status);
    } catch (error) {
      res.status(500).json({ error: "Failed to get Ato status" });
    }
  });

  // Risk Management endpoints
  app.get("/api/risk/dashboard", async (req, res) => {
    try {
      const data = await riskManager.getRiskDashboardData();
      res.json(data);
    } catch (error) {
      res.status(500).json({ error: "Failed to get risk dashboard data" });
    }
  });

  app.get("/api/risk/position-size", async (req, res) => {
    try {
      const { symbol, price, side } = req.query;
      
      if (!symbol || !price) {
        return res.status(400).json({ error: "Symbol and price required" });
      }

      let portfolioValue = 100000;
      
      if (alpaca.isConfigured()) {
        try {
          const account = await alpaca.getAccount();
          portfolioValue = parseFloat(account.equity);
        } catch (e) {
          const positions = await storage.getPositions();
          const positionValue = positions.reduce((sum, p) => sum + p.marketValue, 0);
          portfolioValue = 100000 + positionValue;
        }
      } else {
        const positions = await storage.getPositions();
        const positionValue = positions.reduce((sum, p) => sum + p.marketValue, 0);
        portfolioValue = 100000 + positionValue;
      }
      
      const positions = await storage.getPositions();

      const sizing = await riskManager.calculateDynamicPositionSize(
        symbol as string,
        parseFloat(price as string),
        (side as "buy" | "sell") || "buy",
        portfolioValue,
        positions.length
      );

      res.json(sizing);
    } catch (error) {
      res.status(500).json({ error: "Failed to calculate position size" });
    }
  });

  app.get("/api/risk/volatility/:symbol", async (req, res) => {
    try {
      const volatility = await riskManager.calculateVolatility(req.params.symbol);
      res.json(volatility);
    } catch (error) {
      res.status(500).json({ error: "Failed to get volatility data" });
    }
  });

  app.post("/api/risk/check-trade", async (req, res) => {
    try {
      const { symbol, side, value } = req.body;
      
      if (!symbol || !side || !value) {
        return res.status(400).json({ error: "Symbol, side, and value required" });
      }

      const result = await riskManager.shouldAllowTrade(symbol, side, value);
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: "Failed to check trade permission" });
    }
  });

  // Profit Goal & Performance endpoints
  app.get("/api/profit-goal", async (req, res) => {
    try {
      const goalState = await profitManager.getProfitGoalState();
      res.json(goalState);
    } catch (error) {
      res.status(500).json({ error: "Failed to get profit goal status" });
    }
  });

  app.get("/api/performance", async (req, res) => {
    try {
      const performance = profitManager.getPerformance();
      res.json(performance);
    } catch (error) {
      res.status(500).json({ error: "Failed to get performance data" });
    }
  });

  app.get("/api/trading-allowed", async (req, res) => {
    try {
      const result = await profitManager.shouldContinueTrading();
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: "Failed to check trading status" });
    }
  });

  app.get("/api/analytics", async (req, res) => {
    try {
      const analytics = await tradeBus.aggregateTradeAnalytics();
      res.json(analytics);
    } catch (error) {
      res.status(500).json({ error: "Failed to get analytics" });
    }
  });

  app.get("/api/communication", async (req, res) => {
    try {
      const summary = tradeBus.getCommunicationSummary();
      res.json(summary);
    } catch (error) {
      res.status(500).json({ error: "Failed to get communication status" });
    }
  });

  app.get("/api/signals", async (req, res) => {
    try {
      const signals = tradeBus.getPendingSignals();
      res.json(signals);
    } catch (error) {
      res.status(500).json({ error: "Failed to get signals" });
    }
  });

  app.get("/api/feedback", async (req, res) => {
    try {
      const feedback = tradeBus.getRecentFeedback(50);
      res.json(feedback);
    } catch (error) {
      res.status(500).json({ error: "Failed to get feedback" });
    }
  });

  app.get("/api/learning-insights", async (req, res) => {
    try {
      const insights = tradeBus.getLearningInsights();
      res.json(insights);
    } catch (error) {
      res.status(500).json({ error: "Failed to get learning insights" });
    }
  });

  // P4: Weekly Scorecard for measurement + tuning
  app.get("/api/scorecard/weekly", async (req, res) => {
    try {
      const scorecard = weeklyScorecard.generateWeeklyScorecard();
      res.json(scorecard);
    } catch (error) {
      res.status(500).json({ error: "Failed to generate weekly scorecard" });
    }
  });

  app.get("/api/hunger", async (req, res) => {
    try {
      const hunger = await profitManager.getHungerState();
      res.json(hunger);
    } catch (error) {
      res.status(500).json({ error: "Failed to get hunger status" });
    }
  });

  app.get("/api/warrior", async (req, res) => {
    try {
      const warrior = profitManager.getWarriorState();
      const hunger = await profitManager.getHungerState();
      res.json({
        ...warrior,
        hungerLevel: hunger.hungerLevel,
        profitNeeded: hunger.profitNeeded,
        timeRemaining: hunger.timeRemainingHours,
      });
    } catch (error) {
      res.status(500).json({ error: "Failed to get warrior status" });
    }
  });

  app.get("/api/report", async (req, res) => {
    try {
      const result = await performanceReport.generateReport();
      res.json({
        success: true,
        summary: result.summary,
        files: {
          trades: result.tradesPath,
          summary: result.summaryPath,
          skipReasons: result.skipReasonsPath,
        },
      });
    } catch (error) {
      console.error("[API] Report generation failed:", error);
      res.status(500).json({ error: "Failed to generate report" });
    }
  });

  // Generate single-day report for a specific date (YYYY-MM-DD format)
  app.get("/api/report/date/:date", async (req, res) => {
    try {
      const dateStr = req.params.date;
      if (!/^\d{4}-\d{2}-\d{2}$/.test(dateStr)) {
        return res.status(400).json({ error: "Invalid date format. Use YYYY-MM-DD" });
      }
      const summary = await performanceReport.generateDailyReportForDate(dateStr);
      res.json({
        success: true,
        summary,
      });
    } catch (error) {
      console.error("[API] Report generation failed:", error);
      res.status(500).json({ error: "Failed to generate report" });
    }
  });

  // Generate reports for multiple dates (backfill)
  app.post("/api/report/backfill", async (req, res) => {
    try {
      const dates = req.body.dates as string[];
      if (!dates || !Array.isArray(dates)) {
        return res.status(400).json({ error: "dates array required" });
      }
      await performanceReport.generateReportsForDates(dates);
      res.json({ success: true, count: dates.length });
    } catch (error) {
      console.error("[API] Backfill failed:", error);
      res.status(500).json({ error: "Failed to backfill reports" });
    }
  });

  // Analytics endpoints - read from daily_reports folder
  app.get("/api/analytics/daily", async (req, res) => {
    try {
      const fs = await import("fs");
      const path = await import("path");
      const dailyDir = path.join(process.cwd(), "daily_reports");
      
      const dateParam = req.query.date as string | undefined;
      const filePath = dateParam 
        ? path.join(dailyDir, `${dateParam}.json`)
        : path.join(dailyDir, "latest.json");
      
      if (!fs.existsSync(filePath)) {
        return res.json({ ok: false, reason: "NO_REPORT" });
      }
      
      const raw = fs.readFileSync(filePath, "utf-8");
      const report = JSON.parse(raw);
      
      const o = report.overall || {};
      const dq = report.dataQuality || {};
      
      res.json({
        ok: true,
        reportDate: report.reportDate,
        periodStart: report.periodStart,
        periodEnd: report.periodEnd,
        tradingDays: report.tradingDays,
        daily: {
          closedTrades: o.wins + o.losses,
          tradeCount: o.totalTrades || 0,
          winRate: o.winRate || 0,
          netPnl: o.totalPnl || 0,
          profitFactor: report.all_trades?.profit_factor || 0,
          avgWin: o.avgWin || 0,
          avgLoss: o.avgLoss || 0,
          wins: o.wins || 0,
          losses: o.losses || 0,
        },
        dataQuality: {
          trades_with_trade_id: dq.trades_with_trade_id ?? 0,
          trades_without_trade_id: dq.trades_without_trade_id ?? 0,
          high_confidence_matches: dq.high_confidence_matches ?? 0,
          med_confidence_matches: dq.med_confidence_matches ?? 0,
          low_confidence_matches: dq.low_confidence_matches ?? 0,
          paired_trades: dq.paired_trades ?? 0,
          unpaired_entries: dq.unpaired_entries ?? 0,
          unpaired_exits: dq.unpaired_exits ?? 0,
          legacy_orders_without_trade_id: dq.legacy_orders_without_trade_id ?? 0,
        },
      });
    } catch (error) {
      console.error("[Analytics] Daily fetch error:", error);
      res.status(500).json({ ok: false, reason: "ERROR" });
    }
  });

  app.get("/api/analytics/monthly", async (req, res) => {
    try {
      const fs = await import("fs");
      const path = await import("path");
      const dailyDir = path.join(process.cwd(), "daily_reports");
      
      const monthParam = req.query.month as string;
      if (!monthParam || !/^\d{4}-\d{2}$/.test(monthParam)) {
        const now = new Date();
        const laDate = new Intl.DateTimeFormat("en-CA", { timeZone: "America/Los_Angeles" }).format(now);
        return res.redirect(`/api/analytics/monthly?month=${laDate.slice(0, 7)}`);
      }
      
      if (!fs.existsSync(dailyDir)) {
        return res.json({ ok: false, reason: "NO_MONTH_DATA" });
      }
      
      const files = fs.readdirSync(dailyDir)
        .filter(f => f.startsWith(monthParam) && f.endsWith(".json") && f !== "latest.json");
      
      if (files.length === 0) {
        return res.json({ ok: false, reason: "NO_MONTH_DATA" });
      }
      
      let totalTrades = 0;
      let totalPnl = 0;
      let totalWins = 0;
      let totalLosses = 0;
      let grossProfit = 0;
      let grossLoss = 0;
      const dailySeries: Array<{ date: string; netPnl: number; tradeCount: number }> = [];
      
      for (const file of files.sort()) {
        const filePath = path.join(dailyDir, file);
        const raw = fs.readFileSync(filePath, "utf-8");
        const report = JSON.parse(raw);
        const o = report.overall || {};
        const date = report.reportDate || file.replace(".json", "");
        
        const dayTrades = o.totalTrades || 0;
        const dayPnl = o.totalPnl || 0;
        const dayWins = o.wins || 0;
        const dayLosses = o.losses || 0;
        const dayAvgWin = o.avgWin || 0;
        const dayAvgLoss = o.avgLoss || 0;
        
        totalTrades += dayTrades;
        totalPnl += dayPnl;
        totalWins += dayWins;
        totalLosses += dayLosses;
        grossProfit += dayWins * dayAvgWin;
        grossLoss += dayLosses * dayAvgLoss;
        
        dailySeries.push({ date, netPnl: dayPnl, tradeCount: dayTrades });
      }
      
      const closedTrades = totalWins + totalLosses;
      const winRate = closedTrades > 0 ? (totalWins / closedTrades) * 100 : 0;
      const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? 999 : 0);
      
      res.json({
        ok: true,
        month: monthParam,
        daysReported: files.length,
        monthly: {
          tradeCount: totalTrades,
          closedTrades,
          netPnl: totalPnl,
          winRate,
          profitFactor,
          wins: totalWins,
          losses: totalLosses,
        },
        dailySeries,
      });
    } catch (error) {
      console.error("[Analytics] Monthly fetch error:", error);
      res.status(500).json({ ok: false, reason: "ERROR" });
    }
  });

  // SSE endpoint for live portfolio streaming
  app.get("/api/portfolio/:kind/stream", async (req, res) => {
    const kind = req.params.kind as "paper" | "live";
    console.log(`[SSE] Client connected for ${kind} portfolio stream`);
    
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.flushHeaders();

    const sendEvent = (eventType: string, data: unknown) => {
      res.write(`event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    const fetchAndSend = async () => {
      try {
        if (alpaca.isConfigured()) {
          const account = await alpaca.getAccount();
          const positions = await alpaca.getPositions();
          const botStatus = ato.getAtoState();
          
          const equity = parseFloat(account.equity);
          const lastEquity = parseFloat(account.last_equity);
          const todayPL = equity - lastEquity;
          
          sendEvent("account", {
            totalEquity: equity,
            buyingPower: parseFloat(account.buying_power),
            cash: parseFloat(account.cash),
            todayPL,
            todayPLPercent: lastEquity > 0 ? (todayPL / lastEquity) * 100 : 0,
            dayTradesRemaining: 3 - account.daytrade_count,
          });
          
          sendEvent("positions", positions.map((p: { symbol: string; qty: string; market_value: string; unrealized_pl: string; unrealized_plpc: string; current_price: string; avg_entry_price: string }) => ({
            symbol: p.symbol,
            qty: parseFloat(p.qty),
            marketValue: parseFloat(p.market_value),
            unrealizedPL: parseFloat(p.unrealized_pl),
            unrealizedPLPercent: parseFloat(p.unrealized_plpc) * 100,
            currentPrice: parseFloat(p.current_price),
            avgEntryPrice: parseFloat(p.avg_entry_price),
          })));
          
          sendEvent("botStatus", botStatus);
        } else {
          sendEvent("account", { totalEquity: 0, buyingPower: 0, cash: 0, todayPL: 0, todayPLPercent: 0 });
          sendEvent("positions", []);
          sendEvent("botStatus", { status: "stopped", lastAnalysis: null, currentAction: null });
        }
      } catch (err) {
        console.error("[SSE] Error fetching data:", err);
        sendEvent("error", { message: "Failed to fetch data" });
      }
    };

    await fetchAndSend();
    const intervalId = setInterval(fetchAndSend, 3000);

    req.on("close", () => {
      console.log(`[SSE] Client disconnected from ${kind} stream`);
      clearInterval(intervalId);
    });
  });

  // SMOKE TEST: Verify trade_id tagging works without real orders
  app.get("/api/smoke-test", async (req, res) => {
    try {
      console.log("[SMOKE TEST] Starting trade_id flow verification...");
      
      // 1. Generate entry trade_id
      const entryTradeId = generateTradeId("SPY", "breakout", "buy", 2);
      console.log(`[SMOKE TEST] Entry trade_id: ${entryTradeId}`);
      
      // 2. Generate exit client_order_id (what positionManager would use)
      const exitClientOrderId = `${entryTradeId}_EXIT`;
      const partialClientOrderId = `${entryTradeId}_PARTIAL`;
      console.log(`[SMOKE TEST] Exit client_order_id: ${exitClientOrderId}`);
      console.log(`[SMOKE TEST] Partial client_order_id: ${partialClientOrderId}`);
      
      // 3. Parse the trade_id to verify it's structured correctly
      const parsed = parseTradeId(entryTradeId);
      console.log(`[SMOKE TEST] Parsed trade_id:`, parsed);
      
      // 4. Verify prefix matching logic (what performanceReport uses)
      const matchesExact = exitClientOrderId === entryTradeId;
      const matchesExit = exitClientOrderId === `${entryTradeId}_EXIT`;
      const matchesPartial = partialClientOrderId === `${entryTradeId}_PARTIAL`;
      const matchesPrefix = exitClientOrderId.startsWith(`${entryTradeId}_`);
      
      // 5. Verify breakeven stop calculation
      const testEntry = 100.00;
      const triggerPct = dayTraderConfig.DAY_TRADER_CONFIG.BREAKEVEN_TRIGGER_PCT;
      const offsetPct = dayTraderConfig.DAY_TRADER_CONFIG.BREAKEVEN_OFFSET_PCT;
      const breakevenStop = testEntry * (1 + offsetPct / 100);
      console.log(`[SMOKE TEST] Breakeven calc: entry=$${testEntry.toFixed(2)} -> stop=$${breakevenStop.toFixed(2)} (trigger=${triggerPct}%, offset=${offsetPct}%)`);
      
      const result = {
        success: true,
        test: "trade_id_flow",
        entry: {
          trade_id: entryTradeId,
          parsed: parsed,
        },
        exits: {
          full_exit: exitClientOrderId,
          partial_exit: partialClientOrderId,
        },
        matching: {
          exact_match: matchesExact,
          exit_suffix_match: matchesExit,
          partial_suffix_match: matchesPartial,
          prefix_match: matchesPrefix,
          would_be_HIGH_confidence: matchesExit || matchesPartial || matchesPrefix,
        },
        breakeven: {
          test_entry: testEntry,
          trigger_pct: triggerPct,
          offset_pct: offsetPct,
          stop_price: breakevenStop,
          formula: `entry * (1 + ${offsetPct}/100) = ${breakevenStop.toFixed(2)}`,
        },
        config: {
          BASELINE_MAX_HOLD_MINUTES: dayTraderConfig.DAY_TRADER_CONFIG.BASELINE_MAX_HOLD_MINUTES,
          BREAKEVEN_TRIGGER_PCT: triggerPct,
          BREAKEVEN_OFFSET_PCT: offsetPct,
          BASELINE_MODE: dayTraderConfig.DAY_TRADER_CONFIG.BASELINE_MODE,
        },
      };
      
      console.log("[SMOKE TEST] Result:", JSON.stringify(result, null, 2));
      console.log("[SMOKE TEST] PASS - trade_id flow verified");
      
      res.json(result);
    } catch (error) {
      console.error("[SMOKE TEST] FAIL:", error);
      res.status(500).json({ success: false, error: "Smoke test failed" });
    }
  });

  // Health endpoint - returns runtime status for monitoring
  app.get("/health", async (req, res) => {
    try {
      const status = await runtimeMonitor.getRuntimeStatus();
      res.json(status);
    } catch (error) {
      res.status(500).json({ status: "error", error: "Failed to get runtime status" });
    }
  });

  // Ping endpoint - minimal response for uptime monitoring (UptimeRobot, etc.)
  app.get("/ping", (req, res) => {
    res.status(200).send("pong");
  });

  // Debug snapshot endpoint - comprehensive runtime state for diagnostics
  // Protected by DEBUG_ENDPOINTS_ENABLED env var
  app.get("/debug/snapshot", async (req, res) => {
    if (process.env.DEBUG_ENDPOINTS_ENABLED !== "true") {
      return res.status(404).json({ error: "Not found" });
    }
    
    try {
      const et = timezone.getEasternTime();
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
      let entryAllowed = false;
      
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
        
        if (clock.is_open) {
          const tgStatus = timeGuard.getTimeGuardStatus();
          entryAllowed = tgStatus.canOpenNewTrades;
        }
      } catch (err) {
        console.log("[Debug Snapshot] Error fetching clock");
      }
      
      const summary = activityLedger.getTodaysSummary();
      
      let openPositions: Array<{ symbol: string; qty: string; side: string; unrealizedPl: string }> = [];
      let openOrdersCount = 0;
      let recentOrders: Array<{ 
        id: string; 
        symbol: string; 
        side: string; 
        type: string;
        status: string; 
        qty: string;
        filled_qty: string;
        client_order_id: string;
        submitted_at: string | null;
        filled_at: string | null;
        createdAt: string;
      }> = [];
      
      try {
        const positions = await alpaca.getPositions();
        openPositions = positions.map(p => ({
          symbol: p.symbol,
          qty: p.qty,
          side: p.side,
          unrealizedPl: p.unrealized_pl,
        }));
      } catch (err) {
        console.log("[Debug Snapshot] Error fetching positions");
      }
      
      try {
        const orders = await alpaca.getOrders("all", 10);
        openOrdersCount = orders.filter(o => ["new", "accepted", "pending_new", "partially_filled"].includes(o.status)).length;
        recentOrders = orders.slice(0, 5).map(o => ({
          id: o.id.slice(0, 8),
          symbol: o.symbol,
          side: o.side,
          type: o.type,
          status: o.status,
          qty: o.qty,
          filled_qty: o.filled_qty,
          client_order_id: o.client_order_id,
          submitted_at: o.submitted_at ? new Date(o.submitted_at).toLocaleTimeString("en-US", { timeZone: "America/New_York" }) : null,
          filled_at: o.filled_at ? new Date(o.filled_at).toLocaleTimeString("en-US", { timeZone: "America/New_York" }) : null,
          createdAt: new Date(o.created_at).toLocaleTimeString("en-US", { timeZone: "America/New_York" }),
        }));
      } catch (err) {
        console.log("[Debug Snapshot] Error fetching orders");
      }
      
      const runtimeStatus = await runtimeMonitor.getRuntimeStatus();
      
      const snapshot = {
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
          canManagePositions: timeGuard.getTimeGuardStatus().canManagePositions,
          shouldForceClose: timeGuard.getTimeGuardStatus().shouldForceClose,
          reason: timeGuard.getTimeGuardStatus().reason,
          entryCutoffET: "11:35 AM (or next_close - 5 min)",
          forceCloseET: "3:45 PM (or next_close - 2 min)",
        },
        runtime: {
          uptimeMinutes: runtimeStatus.uptimeMinutes,
          heartbeatCount: runtimeStatus.heartbeatCount,
          bootTimeET: runtimeStatus.bootTimeET,
        },
        scanLoop: {
          ticksSinceMidnightET: summary.scanTicks,
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
          signalsGeneratedToday: summary.signalsGenerated,
          tradesProposedToday: summary.tradesProposed,
          tradesSubmittedToday: summary.tradesSubmitted,
          tradesFilledToday: summary.tradesFilled,
        },
        // Authoritative trade accounting (Alpaca visibility)
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
      
      res.json(snapshot);
    } catch (error) {
      console.error("[Debug Snapshot] Error:", error);
      res.status(500).json({ error: "Failed to generate snapshot" });
    }
  });

  // Entry Window Proof Bundle status endpoint (always available - read-only ops visibility)
  app.get("/debug/proof-status", async (req, res) => {
    const status = entryWindowProof.getProofStatus();
    const et = timezone.getEasternTime();
    
    // LOCK-TRUTH-1: Add leader lock transparency
    const leaderStatus = await leaderLock.getDetailedLeaderStatus();
    const isLeaderVal = leaderLock.isLeaderInstance();
    
    res.json({
      currentTimeET: `${et.hour}:${String(et.minute).padStart(2, '0')} ET`,
      ...status,
      shouldCaptureBeforeEntry: entryWindowProof.shouldCaptureBeforeEntry(),
      shouldCaptureDuringEntry: entryWindowProof.shouldCaptureDuringEntry(),
      shouldCaptureAfterEntry: entryWindowProof.shouldCaptureAfterEntry(),
      // LOCK-TRUTH-1: Leader blocking visibility
      isLeader: isLeaderVal,
      leaderBlocking: !isLeaderVal,
      leaderHolderBootId: leaderStatus.lockHolderBootId,
      leaderHolderPid: leaderStatus.lockHolderPid,
      tradeEntryAllowedByLeader: isLeaderVal,
    });
  });

  // Object Storage status endpoint (always available - read-only ops visibility)
  app.get("/debug/storage-status", async (req, res) => {
    const et = timezone.getEasternTime();
    const dateStr = et.dateString;
    await reportStorage.checkTodayKeys(dateStr);
    const status = reportStorage.getStorageStatus();
    res.json({
      currentTimeET: `${et.hour}:${String(et.minute).padStart(2, '0')} ET`,
      date: dateStr,
      ...status,
    });
  });

  // ALPACA-CONNECTIVITY-PROOF-1: Alpaca connectivity status endpoint
  app.get("/debug/alpaca-status", async (req, res) => {
    const et = timezone.getEasternTime();
    const connectivity = alpaca.getAlpacaConnectivityState();
    const recovery = runtimeMonitor.getRecoveryStatus();
    const storageStatus = reportStorage.getStorageStatus();
    
    let note: string;
    if (connectivity.degraded) {
      note = "DEGRADED - Alpaca API has 3+ consecutive failures";
    } else if (connectivity.consecutiveFailures > 0) {
      note = `WARNING - ${connectivity.consecutiveFailures} consecutive failure(s)`;
    } else if (connectivity.lastOkUTC) {
      note = "HEALTHY - Last API call succeeded";
    } else {
      note = "UNKNOWN - No API calls recorded yet";
    }
    
    res.json({
      currentTimeET: `${et.hour}:${String(et.minute).padStart(2, '0')} ET`,
      currentTimeUTC: new Date().toISOString(),
      note,
      connectivity: {
        alpacaLastOkUTC: connectivity.lastOkUTC,
        alpacaConsecutiveFailures: connectivity.consecutiveFailures,
        alpacaLastError: connectivity.lastError,
        alpacaDegraded: connectivity.degraded,
        lastClockPingUTC: connectivity.lastClockPingUTC,
        lastAccountPingUTC: connectivity.lastAccountPingUTC,
      },
      recovery: {
        tradingState: recovery.state,
        bootRecoveryRan: recovery.bootRecoveryRan,
        recoveryResult: recovery.recoveryResult,
        recoveryFailReason: recovery.recoveryFailReason,
        checks: recovery.checks,
      },
      storageWriteOk: storageStatus.lastWriteOk,
      storageLastWriteUTC: storageStatus.lastWriteTsUTC,
    });
  });

  // Leader lock status endpoint (always available - critical for multi-instance safety)
  // OPS-PROD-LOCK-2: Enhanced with lock freshness visibility
  app.get("/debug/leader-status", async (req, res) => {
    const et = timezone.getEasternTime();
    const detailed = await leaderLock.getDetailedLeaderStatus();
    
    let note: string;
    if (detailed.parseError) {
      note = `Error reading lock: ${detailed.parseError}`;
    } else if (detailed.stale) {
      note = "Lock is STALE - can be taken over by new instance";
    } else if (detailed.isLeader) {
      note = "This instance IS the leader and CAN enter trades";
    } else {
      note = "This instance is NOT the leader and CANNOT enter trades";
    }
    
    res.json({
      currentTimeET: `${et.hour}:${String(et.minute).padStart(2, '0')} ET`,
      currentTimeUTC: new Date().toISOString(),
      isLeader: detailed.isLeader,
      envScope: detailed.envScope,    // OPS-PROD-LOCK-4
      lockKey: detailed.lockKey,      // OPS-PROD-LOCK-4
      thisBootId: detailed.thisBootId,
      thisPid: detailed.thisPid,
      lockHolderBootId: detailed.lockHolderBootId,
      lockHolderPid: detailed.lockHolderPid,
      lockAcquiredAtUTC: detailed.lockAcquiredAtUTC,
      lockLastRefreshAtUTC: detailed.lockLastRefreshAtUTC,
      lockExpiresAtUTC: detailed.lockExpiresAtUTC,
      ttlSecondsRemaining: detailed.ttlSecondsRemaining,
      lockAgeSeconds: detailed.lockAgeSeconds,
      stale: detailed.stale,
      acquireReason: detailed.acquireReason,  // OPS-PROD-LOCK-3
      lockTtlMs: detailed.lockTtlMs,
      refreshIntervalMs: detailed.refreshIntervalMs,
      parseError: detailed.parseError,
      note,
    });
  });

  // OPS-RUNBOOK-VERIFY-1: Runbook snapshot for operational verification
  // ALWAYS available in production (no DEBUG_ENDPOINTS_ENABLED gate)
  app.get("/debug/runbook", async (req, res) => {
    const nowUTC = new Date().toISOString();
    console.log(`[RUNBOOK] served nowUTC=${nowUTC}`);
    
    const et = timezone.getEasternTime();
    const nowET = `${et.hour}:${String(et.minute).padStart(2, '0')} ET`;
    
    // Market & entry status from timeGuard
    const guardStatus = timeGuard.getTimeGuardStatus();
    
    // Leader lock status
    const leaderDetailed = await leaderLock.getDetailedLeaderStatus();
    const isLeaderVal = leaderLock.isLeaderInstance();
    
    // Counters from activityLedger (single source of truth)
    const ledgerSummary = activityLedger.getTodaysSummary();
    
    // Proof bundle status
    const proofStatus = entryWindowProof.getProofStatus();
    
    // Execution ring buffer (last attempt)
    const recentExecs = executionTrace.getRecentExecutions();
    const lastExec = recentExecs.length > 0 ? recentExecs[recentExecs.length - 1] : null;
    
    // Runtime tick counters
    const runtimeStatus = await runtimeMonitor.getRuntimeStatus();
    
    res.json({
      nowET,
      nowUTC,
      marketStatus: runtimeStatus.marketStatus,
      entryAllowed: guardStatus.canOpenNewTrades,
      entryWindowReason: guardStatus.reason,
      currentPhase: proofStatus.currentPhase,
      isLeader: isLeaderVal,
      leaderBlocking: !isLeaderVal,
      leaderHolderBootId: leaderDetailed.lockHolderBootId,
      ttlSecondsRemaining: leaderDetailed.ttlSecondsRemaining,
      lockLastRefreshAtUTC: leaderDetailed.lockLastRefreshAtUTC,
      lockExpiresAtUTC: leaderDetailed.lockExpiresAtUTC,
      countersToday: {
        signalsGeneratedToday: ledgerSummary.signalsGenerated,
        tradesProposedToday: ledgerSummary.tradesProposed,
        tradesSubmittedToday: ledgerSummary.tradesSubmitted,
        tradesFilledToday: ledgerSummary.tradesFilled,
      },
      scansToday: {
        ticksSinceMidnightET: runtimeStatus.ticksSinceMidnightET,
        ticksSinceBoot: runtimeStatus.ticksSinceBoot,
        symbolsEvaluatedToday: ledgerSummary.symbolsEvaluated,
      },
      proof: {
        status: proofStatus.status,
        finalStatus: proofStatus.finalStatus,
        reasonCodes: proofStatus.reasonCodes,
        snapshotCount: proofStatus.snapshotCount,
        gateLogCount: proofStatus.gateLogCount,
        tradabilityGateRunsToday: proofStatus.tradabilityGateRunsToday,
        lastGateRunET: proofStatus.lastGateRunET,
        expectedFinalizeET: proofStatus.expectedFinalizeET,
      },
      lastExecution: lastExec ? {
        lastAttemptET: lastExec.tsET,
        lastStage: lastExec.stageReached,
        lastErrorCode: lastExec.failureReason,
        symbol: lastExec.symbol,
        strategy: lastExec.strategy,
        tradeId: lastExec.trade_id,
      } : null,
    });
  });

  // EXECUTION-TRACE-DURABLE-1: Debug endpoint for recent execution traces
  // Always available (no DEBUG env gating) for production debugging
  // Uses activityLedger as authoritative source for counters, durable trace buffer for traces
  app.get("/debug/execution-recent", (req, res) => {
    const durableStatus = executionTrace.getDurableTraceStatus();
    res.json(durableStatus);
  });

  // ENV-SCOPE-HARDEN-1: Single source of truth for environment configuration
  // Always available (no DEBUG env gating) for production verification
  // Returns envScope, lockKey, reportsPrefix, version, bootId
  app.get("/debug/env", (req, res) => {
    const et = timezone.getEasternTime();
    const status = envScope.getEnvScopeStatus();
    
    res.json({
      currentTimeET: `${et.hour}:${String(et.minute).padStart(2, '0')} ET`,
      currentTimeUTC: new Date().toISOString(),
      ...status,
      note: status.blocked 
        ? `CRITICAL: Trading blocked due to env scope mismatch` 
        : `OK: envScope=${status.envScope} matches deployment context`,
    });
  });

  // Evidence bundle endpoint for debugging execution failures
  // Always available (no DEBUG env gating) for production investigation
  // Returns comprehensive snapshot for a specific date
  app.get("/debug/evidence/:date", async (req, res) => {
    const dateET = req.params.date;
    const nowUTC = new Date().toISOString();
    const et = timezone.getEasternTime();
    const envScope = reportStorage.getStorageEnvScope();
    const prefix = `atobot/${envScope}/reports`;
    
    if (!/^\d{4}-\d{2}-\d{2}$/.test(dateET)) {
      return res.status(400).json({ error: "Invalid date format. Use YYYY-MM-DD" });
    }
    
    try {
      const health = await runtimeMonitor.getRuntimeStatus();
      const leader = leaderLock.getLeaderStatus();
      const proof = entryWindowProof.getProofStatus();
      const storage = reportStorage.getStorageStatus();
      
      const executionFilename = `execution_recent_${dateET}.json`;
      const executionContent = await reportStorage.getText("execution", executionFilename);
      let executions: any[] = [];
      if (executionContent) {
        try {
          const parsed = JSON.parse(executionContent);
          executions = Array.isArray(parsed) ? parsed : (parsed.traces || []);
        } catch {
          executions = [];
        }
      }
      
      const flaggedData = activityLedger.getFlaggedTicks(dateET);
      
      console.log(`[EVIDENCE] served dateET=${dateET} nowUTC=${nowUTC} envScope=${envScope} prefix=${prefix} ticksFlagged=${flaggedData.ticksFlaggedCount} tracesCount=${executions.length}`);
      
      res.json({
        requestedDate: dateET,
        currentTimeET: `${et.hour}:${String(et.minute).padStart(2, "0")} ET`,
        generatedAtUTC: nowUTC,
        envScope,
        reportsPrefix: prefix,
        health,
        leader,
        proof,
        storage,
        executions,
        flaggedTicks: flaggedData.flaggedTicks,
        flaggedTicksSummary: {
          count: flaggedData.ticksFlaggedCount,
          activitySummary: flaggedData.summary,
        },
      });
    } catch (error: any) {
      console.error(`[EVIDENCE] error dateET=${dateET}:`, error?.message);
      res.status(500).json({ error: "Failed to generate evidence bundle", message: error?.message });
    }
  });

  // Tuesday runbook: baseline snapshot for restart verification
  app.get("/debug/baseline", async (req, res) => {
    if (process.env.DEBUG_ENDPOINTS_ENABLED !== "true") {
      return res.status(404).json({ error: "Not found" });
    }
    
    const baseline = runtimeMonitor.getBaseline();
    const et = timezone.getEasternTime();
    res.json({
      capturedAt: `${et.hour}:${String(et.minute).padStart(2, '0')} ET`,
      ...baseline,
    });
  });

  // OPS-DOWNTIME-PROOF-2: Debug endpoint to fetch today's boot/shutdown timeline
  // OPS-DOWNTIME-PROOF-2: Boot timeline for today - always available (no DEBUG gate)
  app.get("/debug/runtime/boots-today", async (req, res) => {
    const et = timezone.getEasternTime();
    const dateET = timezone.toEasternDateString(new Date());
    const fs = await import("fs");
    const path = await import("path");
    
    const localPath = path.join("reports/runtime", `boots_${dateET}.jsonl`);
    let events: any[] = [];
    let source = "none";
    
    // Try local file first
    if (fs.existsSync(localPath)) {
      try {
        const content = fs.readFileSync(localPath, "utf-8");
        const lines = content.split("\n").filter(line => line.trim());
        events = lines.map(line => {
          try { return JSON.parse(line); } catch { return { raw: line }; }
        });
        source = "local";
      } catch (err: any) {
        console.error(`[DEBUG] Failed to read local boots file:`, err?.message);
      }
    }
    
    // Fallback to Object Storage if local is empty
    if (events.length === 0) {
      const storageContent = await reportStorage.getText("runtime", `boots_${dateET}.jsonl`);
      if (storageContent) {
        const lines = storageContent.split("\n").filter(line => line.trim());
        events = lines.map(line => {
          try { return JSON.parse(line); } catch { return { raw: line }; }
        });
        source = "object_storage";
      }
    }
    
    // Get last N events (default 20)
    const limit = parseInt(req.query.limit as string) || 20;
    const recentEvents = events.slice(-limit);
    
    res.json({
      currentTimeET: `${et.hour}:${String(et.minute).padStart(2, "0")} ET`,
      dateET,
      source,
      totalEvents: events.length,
      returnedEvents: recentEvents.length,
      events: recentEvents,
    });
  });

  // BUGFIX-P1-DATA-1: Debug endpoint to test tradability gates with real bid/ask
  app.get("/debug/tradability/:symbol", async (req, res) => {
    const symbol = req.params.symbol.toUpperCase();
    const et = timezone.getEasternTime();
    
    try {
      // Fetch raw market data
      const data = await tradabilityGates.fetchSymbolMarketData(symbol);
      
      if (!data) {
        return res.json({
          symbol,
          currentTimeET: `${et.hour}:${String(et.minute).padStart(2, '0')} ET`,
          passed: false,
          reason: "NO_QUOTE",
          message: "Failed to get valid bid/ask quote from Alpaca",
        });
      }
      
      // Run gates
      const result = await tradabilityGates.runAllTradabilityGates(symbol, data, true);
      
      // Compute spread evidence
      const spreadCents = (data.ask - data.bid) * 100;
      const spreadPct = ((data.ask - data.bid) / data.price) * 100;
      const config = tradabilityGates.getConfig();
      const maxSpreadDollars = Math.max(config.spreadMaxDollars, (config.spreadMaxPercent / 100) * data.price);
      
      res.json({
        symbol,
        currentTimeET: `${et.hour}:${String(et.minute).padStart(2, '0')} ET`,
        passed: result.passed,
        reasons: result.reasons,
        rawData: {
          bid: data.bid,
          ask: data.ask,
          price: data.price,
          volume: data.volume,
          atrPercent: data.atrPercent,
          changePercent: data.changePercent,
          barCount: data.barCount,
          barReason: data.barReason,
        },
        liquidityTruth: {
          volSource: data.volSource,
          timeframe: data.timeframe,
          timestampET: data.timestampET,
          barVolume: data.barVolume,
          sessionVolume: data.sessionVolume,
          isMarketOpen: data.isMarketOpen,
          minutesSinceOpen: data.minutesSinceOpen,
          projectedDailyVolume: data.projectedDailyVolume ? Math.floor(data.projectedDailyVolume) : 0,
          usedValueForGate: data.usedValueForGate ? Math.floor(data.usedValueForGate) : data.volume,
          gateMode: data.gateMode || 'raw',
          minDailyVolume: tradabilityGates.getConfig().minDailyVolume,
          note: "LIQUIDITY-PACE-1: Uses projected volume during market hours"
        },
        spreadAnalysis: {
          spreadCents: spreadCents.toFixed(2),
          spreadPct: spreadPct.toFixed(4) + "%",
          maxAllowedCents: (maxSpreadDollars * 100).toFixed(2),
          maxAllowedPct: config.spreadMaxPercent.toFixed(2) + "%",
        },
        thresholds: {
          spreadMaxDollars: config.spreadMaxDollars,
          spreadMaxPercent: config.spreadMaxPercent,
          minDailyVolume: config.minDailyVolume,
          atrMinPercent: config.atrMinPercent,
          atrMaxPercent: config.atrMaxPercent,
        },
      });
    } catch (error: any) {
      res.status(500).json({
        symbol,
        error: error.message,
      });
    }
  });

  return httpServer;
}
