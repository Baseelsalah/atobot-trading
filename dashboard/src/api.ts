const API_BASE = "";

// ── Auth Token Management ─────────────────────────────
let authToken: string | null = localStorage.getItem("atobot-token");

export function setAuthToken(token: string | null): void {
  authToken = token;
  if (token) {
    localStorage.setItem("atobot-token", token);
  } else {
    localStorage.removeItem("atobot-token");
  }
}

export function getAuthToken(): string | null {
  return authToken;
}

async function handleResponse<T>(res: Response, path: string): Promise<T> {
  if (res.status === 401 && !path.startsWith("/api/auth/")) {
    setAuthToken(null);
    window.location.reload();
    throw new Error("Session expired");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.error || `API ${path}: ${res.status}`);
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const headers: Record<string, string> = {};
  if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
  const res = await fetch(`${API_BASE}${path}`, { headers });
  return handleResponse(res, path);
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  return handleResponse(res, path);
}

async function del<T>(path: string): Promise<T> {
  const headers: Record<string, string> = {};
  if (authToken) headers["Authorization"] = `Bearer ${authToken}`;
  const res = await fetch(`${API_BASE}${path}`, { method: "DELETE", headers });
  return handleResponse(res, path);
}

// ── Auth Types ────────────────────────────────────────

export interface AuthUser {
  id: string;
  email: string;
  displayName: string;
  role: "admin" | "user";
  status: "pending" | "approved" | "rejected" | "suspended";
}

export interface LoginResponse {
  token: string;
  user: AuthUser;
}

export interface AdminUser {
  id: string;
  email: string;
  display_name: string;
  role: "admin" | "user";
  status: "pending" | "approved" | "rejected" | "suspended";
  is_paper: number;
  created_at: string;
  approved_at: string | null;
  approved_by: string | null;
}

// ── Auth API ──────────────────────────────────────────

export const login = (email: string, password: string) =>
  post<LoginResponse>("/api/auth/login", { email, password });

export const register = (email: string, password: string, displayName: string) =>
  post<LoginResponse>("/api/auth/register", { email, password, displayName });

export const fetchMe = () => get<AuthUser>("/api/auth/me");

export const saveApiKeys = (alpacaKey: string, alpacaSecret: string, isPaper: boolean) =>
  post<{ success: boolean }>("/api/auth/api-keys", { alpacaKey, alpacaSecret, isPaper });

export const fetchApiKeyStatus = () =>
  get<{ hasKeys: boolean; isPaper: boolean }>("/api/auth/api-keys/status");

export const deleteApiKeys = () =>
  del<{ success: boolean }>("/api/auth/api-keys");

export const testApiKeys = (alpacaKey: string, alpacaSecret: string, isPaper: boolean) =>
  post<{ success: boolean; equity?: string; status?: string; error?: string }>("/api/auth/test-keys", { alpacaKey, alpacaSecret, isPaper });

// ── Admin API ─────────────────────────────────────────

export const fetchAllUsers = () => get<AdminUser[]>("/api/admin/users");
export const approveUser = (id: string) => post<{ success: boolean }>(`/api/admin/users/${id}/approve`);
export const rejectUser = (id: string) => post<{ success: boolean }>(`/api/admin/users/${id}/reject`);
export const suspendUser = (id: string) => post<{ success: boolean }>(`/api/admin/users/${id}/suspend`);

// ── Types ──────────────────────────────────────────────

export interface Portfolio {
  totalEquity: number;
  buyingPower: number;
  cash: number;
  todayPL: number;
  todayPLPercent: number;
  totalPL: number;
  totalPLPercent: number;
  dayTradesRemaining: number;
}

export interface Position {
  symbol: string;
  quantity: number;
  avgEntryPrice: number;
  currentPrice: number;
  marketValue: number;
  unrealizedPL: number;
  unrealizedPLPercent: number;
}

export interface ManagedPosition {
  symbol: string;
  side: string;
  qty: number;
  entryPrice: number;
  currentPrice: number;
  stopLoss: number;
  takeProfit: number;
  unrealizedPL: number;
  unrealizedPLPercent: number;
  entryTime: string;
  strategy: string;
}

export interface Trade {
  id: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  totalValue: number;
  status: string;
  reason: string | null;
  timestamp: string | null;
}

export interface BotStatus {
  status: "active" | "paused" | "analyzing" | "error" | "stopped";
  lastAnalysis: string | null;
  currentAction: string | null;
  errorMessage: string | null;
}

export interface MarketStatus {
  isOpen: boolean;
  nextOpen: string;
  nextClose: string;
}

export interface Performance {
  totalTrades: number;
  wins: number;
  losses: number;
  totalProfit: number;
  totalLoss: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  expectancy: number;
  profitFactor: number;
  largestWin: number;
  largestLoss: number;
  consecutiveWins: number;
  consecutiveLosses: number;
}

export interface TradingStatus {
  canTrade: boolean;
  canEnterNewPositions: boolean;
  reason: string;
  currentTimeET: string;
  entryStartET: string;
  entryCutoffET: string;
  forceCloseET: string;
  forceClosePT: string;
}

export interface Settings {
  isActive: boolean;
  isPaperTrading: boolean;
  maxPositionSize: number;
  maxDailyLoss: number;
  maxPositions: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  tradingHoursOnly: boolean;
  analysisInterval: number;
}

export interface Alert {
  id: string;
  type: string;
  title: string;
  message: string;
  isRead: boolean;
  isResolved: boolean;
  timestamp: string | null;
}

export interface PortfolioHistoryPoint {
  timestamp: number;
  equity: number;
  profit_loss: number;
  profit_loss_pct: number;
}

export interface DailyAnalytics {
  ok: boolean;
  reason?: string;
  reportDate?: string;
  daily?: {
    closedTrades: number;
    tradeCount: number;
    winRate: number;
    netPnl: number;
    profitFactor: number;
    avgWin: number;
    avgLoss: number;
    wins: number;
    losses: number;
  };
}

export interface RiskDashboard {
  metrics: {
    portfolioVolatility: number;
    portfolioHeatLevel: number;
    maxDrawdown: number;
    currentDrawdown: number;
    riskCapacity: number;
    correlationRisk: number;
  };
  volatilityBySymbol: Record<string, number>;
  recommendations: string[];
}

export interface ProfitGoal {
  dailyGoal: number;
  currentProfit: number;
  realizedProfit: number;
  unrealizedProfit: number;
  progressPercent: number;
  goalMet: boolean;
  tradesNeeded: number;
  avgProfitPerTrade: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  expectancy: number;
  profitFactor: number;
}

export interface MarketRegime {
  isUptrend: boolean;
  isBullish: boolean;
  isBearish: boolean;
  isNeutral: boolean;
  qqq: { price: number; ema9: number; ema20: number; trend: string };
  spy: { price: number; ema9: number; ema20: number; trend: string };
  recommendation: string;
}

export interface Hunger {
  hungerLevel: string;
  urgency: number;
  aggressiveness: number;
  positionSizeMultiplier: number;
  thresholdReduction: number;
  message: string;
  profitNeeded: number;
  timeRemainingHours: number;
  profitPerHourNeeded: number;
}

export interface BrainStatus {
  strategies: number;
  activeStrategies: number;
  learningCycles: number;
  topStrategy: string;
  overallConfidence: number;
}

export interface Strategy {
  id: string;
  name: string;
  description: string;
  type: string;
  rules: string;
  symbols: string;
  confidence: number;
  isActive: boolean;
  winRate: number;
  totalTrades: number;
  totalProfit: number;
  createdAt: string;
  updatedAt: string;
}

export interface Signal {
  id: string;
  symbol: string;
  side: string;
  confidence: number;
  reason: string;
  timestamp: string;
}

export interface WeeklyScorecard {
  periodStart: string;
  periodEnd: string;
  generatedAt: string;
  summary: {
    totalTrades: number;
    wins: number;
    losses: number;
    winRate: number;
    totalPnl: number;
    avgPnl: number;
    expectancy: number;
    profitFactor: number;
    sharpeEstimate: number;
  };
  byRegime: Array<{
    regime: string;
    trades: number;
    wins: number;
    losses: number;
    winRate: number;
    totalPnl: number;
    avgPnl: number;
    expectancy: number;
    profitFactor: number;
  }>;
  byTimeWindow: Array<{
    timeWindow: string;
    trades: number;
    wins: number;
    losses: number;
    winRate: number;
    totalPnl: number;
    avgPnl: number;
  }>;
  byStrategy: Array<{
    strategy: string;
    trades: number;
    wins: number;
    losses: number;
    winRate: number;
    totalPnl: number;
  }>;
  skipReasons: Array<{
    reason: string;
    count: number;
    pctOfTotal: number;
  }>;
  recommendations: string[];
}

// ── GET endpoints ──────────────────────────────────────

export const fetchPortfolio = () => get<Portfolio>("/api/portfolio");
export const fetchPositions = () => get<Position[]>("/api/positions");
export const fetchManagedPositions = () => get<ManagedPosition[]>("/api/positions/managed");
export const fetchTrades = () => get<Trade[]>("/api/trades");
export const fetchBotStatus = () => get<BotStatus>("/api/bot/status");
export const fetchMarketStatus = () => get<MarketStatus>("/api/market/status");
export const fetchPerformance = () => get<Performance>("/api/performance");
export const fetchTradingStatus = () => get<TradingStatus>("/api/trading/status");
export const fetchSettings = () => get<Settings>("/api/settings");
export const fetchAlerts = () => get<Alert[]>("/api/alerts");
export const fetchDailyAnalytics = (date: string) =>
  get<DailyAnalytics>(`/api/analytics/daily?date=${date}`);
export const fetchPortfolioHistory = (period: string) =>
  get<PortfolioHistoryPoint[]>(`/api/portfolio/history?period=${period}`);
export const fetchRiskDashboard = () => get<RiskDashboard>("/api/risk/dashboard");
export const fetchProfitGoal = () => get<ProfitGoal>("/api/profit-goal");
export const fetchMarketRegime = () => get<MarketRegime>("/api/market/regime");
export const fetchHunger = () => get<Hunger>("/api/hunger");
export const fetchBrainStatus = () => get<BrainStatus>("/api/brain/status");
export const fetchStrategies = () => get<Strategy[]>("/api/brain/strategies");
export const fetchSignals = () => get<Signal[]>("/api/signals");
export const fetchWeeklyScorecard = () => get<WeeklyScorecard>("/api/scorecard/weekly");

// ── POST endpoints ─────────────────────────────────────

export const startBot = () => post<{ success: boolean }>("/api/bot/start");
export const stopBot = () => post<{ success: boolean }>("/api/bot/stop");
export const pauseBot = () => post<{ success: boolean }>("/api/bot/pause");
export const emergencyClose = () =>
  post<{ success: boolean; closed: number; totalPL: number }>("/api/trading/emergency-close");
export const closePosition = (symbol: string) =>
  post<{ success: boolean }>("/api/positions/close", { symbol });
export const closeAllPositions = () =>
  post<{ success: boolean }>("/api/positions/close-all");
export const approveAlert = (id: string) =>
  post<{ success: boolean }>(`/api/alerts/${id}/approve`);
export const denyAlert = (id: string) =>
  post<{ success: boolean }>(`/api/alerts/${id}/deny`);
export const markAlertRead = (id: string) =>
  post<{ success: boolean }>(`/api/alerts/${id}/read`);
export const markAllAlertsRead = () =>
  post<{ success: boolean }>("/api/alerts/mark-all-read");
export const triggerResearch = () =>
  post<{ success: boolean }>("/api/brain/research");
