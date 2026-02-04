import { pgTable, text, varchar, integer, real, boolean, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";
import { sql } from "drizzle-orm";

// Users table
export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

// Trades table - stores all executed trades
export const trades = pgTable("trades", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  symbol: text("symbol").notNull(),
  side: text("side").notNull(), // 'buy' or 'sell'
  quantity: integer("quantity").notNull(),
  price: real("price").notNull(),
  totalValue: real("total_value").notNull(),
  status: text("status").notNull().default("pending"), // pending, filled, cancelled, rejected
  orderId: text("order_id"),
  reason: text("reason"), // GPT reasoning for the trade
  timestamp: timestamp("timestamp").defaultNow(),
});

export const insertTradeSchema = createInsertSchema(trades).omit({ id: true, timestamp: true });
export type InsertTrade = z.infer<typeof insertTradeSchema>;
export type Trade = typeof trades.$inferSelect;

// Positions table - current holdings
export const positions = pgTable("positions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  symbol: text("symbol").notNull().unique(),
  quantity: integer("quantity").notNull(),
  avgEntryPrice: real("avg_entry_price").notNull(),
  currentPrice: real("current_price").notNull(),
  marketValue: real("market_value").notNull(),
  unrealizedPL: real("unrealized_pl").notNull(),
  unrealizedPLPercent: real("unrealized_pl_percent").notNull(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertPositionSchema = createInsertSchema(positions).omit({ id: true, updatedAt: true });
export type InsertPosition = z.infer<typeof insertPositionSchema>;
export type Position = typeof positions.$inferSelect;

// Research logs - GPT analysis entries
export const researchLogs = pgTable("research_logs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  type: text("type").notNull(), // 'analysis', 'news', 'technical', 'recommendation'
  symbol: text("symbol"),
  summary: text("summary").notNull(),
  details: text("details"),
  confidence: real("confidence"), // 0-100
  sources: text("sources"), // JSON array of source strings
  timestamp: timestamp("timestamp").defaultNow(),
});

export const insertResearchLogSchema = createInsertSchema(researchLogs).omit({ id: true, timestamp: true });
export type InsertResearchLog = z.infer<typeof insertResearchLogSchema>;
export type ResearchLog = typeof researchLogs.$inferSelect;

// Activity logs - all bot actions
export const activityLogs = pgTable("activity_logs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  type: text("type").notNull(), // 'trade', 'analysis', 'error', 'system', 'alert'
  action: text("action").notNull(),
  description: text("description").notNull(),
  metadata: text("metadata"), // JSON string for additional data
  timestamp: timestamp("timestamp").defaultNow(),
});

export const insertActivityLogSchema = createInsertSchema(activityLogs).omit({ id: true, timestamp: true });
export type InsertActivityLog = z.infer<typeof insertActivityLogSchema>;
export type ActivityLog = typeof activityLogs.$inferSelect;

// Alerts - notifications requiring user attention
export const alerts = pgTable("alerts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  type: text("type").notNull(), // 'critical', 'warning', 'info'
  title: text("title").notNull(),
  message: text("message").notNull(),
  requiresApproval: boolean("requires_approval").default(false),
  isRead: boolean("is_read").default(false),
  isResolved: boolean("is_resolved").default(false),
  metadata: text("metadata"), // JSON for trade approval data
  timestamp: timestamp("timestamp").defaultNow(),
});

export const insertAlertSchema = createInsertSchema(alerts).omit({ id: true, timestamp: true });
export type InsertAlert = z.infer<typeof insertAlertSchema>;
export type Alert = typeof alerts.$inferSelect;

// Bot settings
export const botSettings = pgTable("bot_settings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  isActive: boolean("is_active").default(false),
  isPaperTrading: boolean("is_paper_trading").default(true),
  maxPositionSize: real("max_position_size").default(1000), // Max $ per position
  maxDailyLoss: real("max_daily_loss").default(500), // Stop trading if daily loss exceeds this
  maxPositions: integer("max_positions").default(5), // Max concurrent positions
  stopLossPercent: real("stop_loss_percent").default(2), // Default stop loss %
  takeProfitPercent: real("take_profit_percent").default(5), // Default take profit %
  tradingHoursOnly: boolean("trading_hours_only").default(true),
  analysisInterval: integer("analysis_interval").default(300), // Seconds between analyses
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertBotSettingsSchema = createInsertSchema(botSettings).omit({ id: true, updatedAt: true });
export type InsertBotSettings = z.infer<typeof insertBotSettingsSchema>;
export type BotSettings = typeof botSettings.$inferSelect;

// Trading Strategies - learned and created by autopilot brain
export const tradingStrategies = pgTable("trading_strategies", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  description: text("description").notNull(),
  type: text("type").notNull(), // 'momentum', 'mean_reversion', 'breakout', 'custom'
  rules: text("rules").notNull(), // JSON rules for entry/exit
  symbols: text("symbols"), // JSON array of preferred symbols
  confidence: real("confidence").default(50), // How confident autopilot is in this strategy
  winRate: real("win_rate").default(0), // Historical win rate
  totalTrades: integer("total_trades").default(0),
  totalProfit: real("total_profit").default(0),
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertTradingStrategySchema = createInsertSchema(tradingStrategies).omit({ id: true, createdAt: true, updatedAt: true });
export type InsertTradingStrategy = z.infer<typeof insertTradingStrategySchema>;
export type TradingStrategy = typeof tradingStrategies.$inferSelect;

// Strategy Performance - tracks individual strategy performance over time
export const strategyPerformance = pgTable("strategy_performance", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  strategyId: varchar("strategy_id").notNull(),
  tradeId: varchar("trade_id").notNull(),
  symbol: text("symbol").notNull(),
  profitLoss: real("profit_loss").notNull(),
  wasSuccessful: boolean("was_successful").notNull(),
  entryReason: text("entry_reason"),
  exitReason: text("exit_reason"),
  marketConditions: text("market_conditions"), // JSON snapshot
  timestamp: timestamp("timestamp").defaultNow(),
});

export const insertStrategyPerformanceSchema = createInsertSchema(strategyPerformance).omit({ id: true, timestamp: true });
export type InsertStrategyPerformance = z.infer<typeof insertStrategyPerformanceSchema>;
export type StrategyPerformance = typeof strategyPerformance.$inferSelect;

// Autopilot Brain State - stores the AI's learned knowledge
export const autopilotBrain = pgTable("autopilot_brain", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  marketInsights: text("market_insights"), // JSON of learned market patterns
  stockPreferences: text("stock_preferences"), // JSON of stocks with their learned weights
  riskProfile: text("risk_profile"), // JSON of learned risk parameters
  lastResearch: timestamp("last_research"),
  totalLearningCycles: integer("total_learning_cycles").default(0),
  overallWinRate: real("overall_win_rate").default(0),
  bestPerformingStrategy: varchar("best_performing_strategy"),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertAutopilotBrainSchema = createInsertSchema(autopilotBrain).omit({ id: true, updatedAt: true });
export type InsertAutopilotBrain = z.infer<typeof insertAutopilotBrainSchema>;
export type AutopilotBrain = typeof autopilotBrain.$inferSelect;

// TypeScript interfaces for API responses
export interface PortfolioSummary {
  totalEquity: number;
  buyingPower: number;
  cash: number;
  todayPL: number;
  todayPLPercent: number;
  totalPL: number;
  totalPLPercent: number;
  dayTradesRemaining: number;
}

export interface MarketStatus {
  isOpen: boolean;
  nextOpen: string;
  nextClose: string;
}

// V2 Market Status with simulation flags (single source of truth)
export interface MarketStatusV2 {
  is_open: boolean;
  timestamp: string;
  next_open: string | null;
  next_close: string | null;
  source: "alpaca_clock";
  simulated: boolean;
}

export interface BotStatus {
  status: 'active' | 'paused' | 'analyzing' | 'error' | 'stopped';
  lastAnalysis: string | null;
  currentAction: string | null;
  errorMessage: string | null;
}

export interface TradeRecommendation {
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  reason: string;
  confidence: number;
  riskLevel: 'low' | 'medium' | 'high';
  // P3: ATR-based adaptive stops (optional - falls back to fixed % if not set)
  stopPrice?: number;
  takeProfitPrice?: number;
  atr?: number;
  // P4: Regime context for measurement + tuning
  regime?: 'bull' | 'bear' | 'chop';
  // P5: Strategy name for A/B evaluation in weekly scorecards
  strategyName?: string;
}
