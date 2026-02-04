import { randomUUID } from "crypto";
import type {
  User,
  InsertUser,
  Trade,
  InsertTrade,
  Position,
  InsertPosition,
  ResearchLog,
  InsertResearchLog,
  ActivityLog,
  InsertActivityLog,
  Alert,
  InsertAlert,
  BotSettings,
  InsertBotSettings,
  TradingStrategy,
  InsertTradingStrategy,
  StrategyPerformance,
  InsertStrategyPerformance,
  AutopilotBrain,
  InsertAutopilotBrain,
} from "@shared/schema";

export interface IStorage {
  // Users
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;

  // Trades
  getTrades(limit?: number): Promise<Trade[]>;
  createTrade(trade: InsertTrade): Promise<Trade>;
  updateTradeStatus(id: string, status: string, orderId?: string): Promise<void>;

  // Positions
  getPositions(): Promise<Position[]>;
  getPosition(symbol: string): Promise<Position | undefined>;
  upsertPosition(position: InsertPosition): Promise<Position>;
  deletePosition(symbol: string): Promise<void>;
  clearPositions(): Promise<void>;

  // Research Logs
  getResearchLogs(limit?: number): Promise<ResearchLog[]>;
  createResearchLog(log: InsertResearchLog): Promise<ResearchLog>;

  // Activity Logs
  getActivityLogs(limit?: number): Promise<ActivityLog[]>;
  createActivityLog(log: InsertActivityLog): Promise<ActivityLog>;

  // Alerts
  getAlerts(): Promise<Alert[]>;
  getUnreadAlerts(): Promise<Alert[]>;
  createAlert(alert: InsertAlert): Promise<Alert>;
  markAlertRead(id: string): Promise<void>;
  resolveAlert(id: string): Promise<void>;
  markAllAlertsRead(): Promise<void>;

  // Bot Settings
  getSettings(): Promise<BotSettings>;
  updateSettings(settings: Partial<InsertBotSettings>): Promise<BotSettings>;

  // Trading Strategies
  getStrategies(): Promise<TradingStrategy[]>;
  getStrategy(id: string): Promise<TradingStrategy | undefined>;
  createStrategy(strategy: InsertTradingStrategy): Promise<TradingStrategy>;
  updateStrategy(id: string, updates: Partial<InsertTradingStrategy>): Promise<void>;

  // Strategy Performance
  recordStrategyPerformance(perf: InsertStrategyPerformance): Promise<StrategyPerformance>;
  getStrategyPerformance(strategyId: string): Promise<StrategyPerformance[]>;

  // Autopilot Brain
  getAutopilotBrain(): Promise<AutopilotBrain | undefined>;
  updateAutopilotBrain(updates: Partial<InsertAutopilotBrain>): Promise<AutopilotBrain>;
}

export class MemStorage implements IStorage {
  private users: Map<string, User>;
  private trades: Map<string, Trade>;
  private positions: Map<string, Position>;
  private researchLogs: Map<string, ResearchLog>;
  private activityLogs: Map<string, ActivityLog>;
  private alerts: Map<string, Alert>;
  private settings: BotSettings;
  private strategies: Map<string, TradingStrategy>;
  private strategyPerformance: Map<string, StrategyPerformance>;
  private autopilotBrain: AutopilotBrain | undefined;

  constructor() {
    this.users = new Map();
    this.trades = new Map();
    this.positions = new Map();
    this.researchLogs = new Map();
    this.activityLogs = new Map();
    this.alerts = new Map();
    this.strategies = new Map();
    this.strategyPerformance = new Map();
    this.settings = {
      id: randomUUID(),
      isActive: false,
      isPaperTrading: true,
      maxPositionSize: 1000,
      maxDailyLoss: 500,
      maxPositions: 5,
      stopLossPercent: 2,
      takeProfitPercent: 5,
      tradingHoursOnly: true,
      analysisInterval: 300,
      updatedAt: new Date(),
    };
  }

  // Users
  async getUser(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find((user) => user.username === username);
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  // Trades
  async getTrades(limit = 100): Promise<Trade[]> {
    const trades = Array.from(this.trades.values())
      .sort((a, b) => {
        const dateA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
        const dateB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
        return dateB - dateA;
      })
      .slice(0, limit);
    return trades;
  }

  async createTrade(insertTrade: InsertTrade): Promise<Trade> {
    const id = randomUUID();
    const trade: Trade = {
      ...insertTrade,
      id,
      status: insertTrade.status || "pending",
      orderId: insertTrade.orderId || null,
      reason: insertTrade.reason || null,
      timestamp: new Date(),
    };
    this.trades.set(id, trade);
    return trade;
  }

  async updateTradeStatus(id: string, status: string, orderId?: string): Promise<void> {
    const trade = this.trades.get(id);
    if (trade) {
      trade.status = status;
      if (orderId) trade.orderId = orderId;
    }
  }

  // Positions
  async getPositions(): Promise<Position[]> {
    return Array.from(this.positions.values());
  }

  async getPosition(symbol: string): Promise<Position | undefined> {
    return this.positions.get(symbol);
  }

  async upsertPosition(insertPosition: InsertPosition): Promise<Position> {
    const existing = this.positions.get(insertPosition.symbol);
    if (existing) {
      const updated: Position = {
        ...existing,
        ...insertPosition,
        updatedAt: new Date(),
      };
      this.positions.set(insertPosition.symbol, updated);
      return updated;
    }
    const id = randomUUID();
    const position: Position = {
      ...insertPosition,
      id,
      updatedAt: new Date(),
    };
    this.positions.set(insertPosition.symbol, position);
    return position;
  }

  async deletePosition(symbol: string): Promise<void> {
    this.positions.delete(symbol);
  }

  async clearPositions(): Promise<void> {
    this.positions.clear();
  }

  // Research Logs
  async getResearchLogs(limit = 50): Promise<ResearchLog[]> {
    return Array.from(this.researchLogs.values())
      .sort((a, b) => {
        const dateA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
        const dateB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
        return dateB - dateA;
      })
      .slice(0, limit);
  }

  async createResearchLog(insertLog: InsertResearchLog): Promise<ResearchLog> {
    const id = randomUUID();
    const log: ResearchLog = {
      ...insertLog,
      id,
      timestamp: new Date(),
      symbol: insertLog.symbol ?? null,
      details: insertLog.details ?? null,
      confidence: insertLog.confidence ?? null,
      sources: insertLog.sources ?? null,
    };
    this.researchLogs.set(id, log);
    return log;
  }

  // Activity Logs
  async getActivityLogs(limit = 100): Promise<ActivityLog[]> {
    return Array.from(this.activityLogs.values())
      .sort((a, b) => {
        const dateA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
        const dateB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
        return dateB - dateA;
      })
      .slice(0, limit);
  }

  async createActivityLog(insertLog: InsertActivityLog): Promise<ActivityLog> {
    const id = randomUUID();
    const log: ActivityLog = {
      ...insertLog,
      id,
      timestamp: new Date(),
      metadata: insertLog.metadata ?? null,
    };
    this.activityLogs.set(id, log);
    return log;
  }

  // Alerts
  async getAlerts(): Promise<Alert[]> {
    return Array.from(this.alerts.values())
      .sort((a, b) => {
        const dateA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
        const dateB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
        return dateB - dateA;
      });
  }

  async getUnreadAlerts(): Promise<Alert[]> {
    return (await this.getAlerts()).filter((a) => !a.isRead);
  }

  async createAlert(insertAlert: InsertAlert): Promise<Alert> {
    const id = randomUUID();
    const alert: Alert = {
      ...insertAlert,
      id,
      isRead: insertAlert.isRead ?? false,
      isResolved: insertAlert.isResolved ?? false,
      requiresApproval: insertAlert.requiresApproval ?? false,
      timestamp: new Date(),
      metadata: insertAlert.metadata ?? null,
    };
    this.alerts.set(id, alert);
    return alert;
  }

  async markAlertRead(id: string): Promise<void> {
    const alert = this.alerts.get(id);
    if (alert) {
      alert.isRead = true;
    }
  }

  async resolveAlert(id: string): Promise<void> {
    const alert = this.alerts.get(id);
    if (alert) {
      alert.isResolved = true;
      alert.isRead = true;
    }
  }

  async markAllAlertsRead(): Promise<void> {
    this.alerts.forEach((alert) => {
      alert.isRead = true;
    });
  }

  // Bot Settings
  async getSettings(): Promise<BotSettings> {
    return this.settings;
  }

  async updateSettings(updates: Partial<InsertBotSettings>): Promise<BotSettings> {
    this.settings = {
      ...this.settings,
      ...updates,
      updatedAt: new Date(),
    };
    return this.settings;
  }

  // Trading Strategies
  async getStrategies(): Promise<TradingStrategy[]> {
    return Array.from(this.strategies.values());
  }

  async getStrategy(id: string): Promise<TradingStrategy | undefined> {
    return this.strategies.get(id);
  }

  async createStrategy(insertStrategy: InsertTradingStrategy): Promise<TradingStrategy> {
    const id = randomUUID();
    const strategy: TradingStrategy = {
      ...insertStrategy,
      id,
      symbols: insertStrategy.symbols || null,
      confidence: insertStrategy.confidence ?? 50,
      winRate: insertStrategy.winRate ?? 0,
      totalTrades: insertStrategy.totalTrades ?? 0,
      totalProfit: insertStrategy.totalProfit ?? 0,
      isActive: insertStrategy.isActive ?? true,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    this.strategies.set(id, strategy);
    return strategy;
  }

  async updateStrategy(id: string, updates: Partial<InsertTradingStrategy>): Promise<void> {
    const strategy = this.strategies.get(id);
    if (strategy) {
      const updated: TradingStrategy = {
        ...strategy,
        ...updates,
        updatedAt: new Date(),
      };
      this.strategies.set(id, updated);
    }
  }

  // Strategy Performance
  async recordStrategyPerformance(insertPerf: InsertStrategyPerformance): Promise<StrategyPerformance> {
    const id = randomUUID();
    const perf: StrategyPerformance = {
      ...insertPerf,
      id,
      entryReason: insertPerf.entryReason || null,
      exitReason: insertPerf.exitReason || null,
      marketConditions: insertPerf.marketConditions || null,
      timestamp: new Date(),
    };
    this.strategyPerformance.set(id, perf);
    return perf;
  }

  async getStrategyPerformance(strategyId: string): Promise<StrategyPerformance[]> {
    return Array.from(this.strategyPerformance.values())
      .filter(p => p.strategyId === strategyId);
  }

  // Autopilot Brain
  async getAutopilotBrain(): Promise<AutopilotBrain | undefined> {
    return this.autopilotBrain;
  }

  async updateAutopilotBrain(updates: Partial<InsertAutopilotBrain>): Promise<AutopilotBrain> {
    if (!this.autopilotBrain) {
      this.autopilotBrain = {
        id: randomUUID(),
        marketInsights: updates.marketInsights || null,
        stockPreferences: updates.stockPreferences || null,
        riskProfile: updates.riskProfile || null,
        lastResearch: updates.lastResearch || null,
        totalLearningCycles: updates.totalLearningCycles ?? 0,
        overallWinRate: updates.overallWinRate ?? 0,
        bestPerformingStrategy: updates.bestPerformingStrategy || null,
        updatedAt: new Date(),
      };
    } else {
      this.autopilotBrain = {
        ...this.autopilotBrain,
        ...updates,
        updatedAt: new Date(),
      };
    }
    return this.autopilotBrain;
  }
}

export const storage = new MemStorage();
