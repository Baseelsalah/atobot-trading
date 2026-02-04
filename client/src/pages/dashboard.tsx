import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { useState, useEffect, useMemo } from "react";
import { useLocation, useSearch } from "wouter";
import { usePortfolioStream } from "@/hooks/use-portfolio-stream";
import { usePortfolio } from "@/lib/portfolio-context";
import { motion } from "framer-motion";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { AlertModal } from "@/components/alert-modal";
import { PageContainer, PageGrid, GridCell } from "@/components/layout";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";

import {
  TrendingUp,
  TrendingDown,
  Play,
  Pause,
  Square,
  AlertTriangle,
  Wifi,
  WifiOff,
  Loader2,
  Zap,
  BarChart3,
  Target,
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  CircleDot,
  FileText,
  DollarSign,
  Wallet,
  History,
  Download,
  Calendar,
  Clock,
  Radio,
  Layers,
} from "lucide-react";

import type {
  BotStatus,
  BotSettings,
  MarketStatusV2,
  Alert,
  Trade,
} from "@shared/schema";

function formatCurrency(val: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(val);
}

function formatPercent(val: number): string {
  const sign = val >= 0 ? "+" : "";
  return `${sign}${val.toFixed(2)}%`;
}

function StatusIndicator({ status }: { status: string }) {
  const colors: Record<string, string> = {
    active: "bg-emerald-500",
    paused: "bg-amber-500",
    analyzing: "bg-blue-500",
    stopped: "bg-slate-500",
    error: "bg-red-500",
  };
  
  return (
    <span className="relative flex h-2.5 w-2.5">
      <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${colors[status] || colors.stopped}`} />
      <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${colors[status] || colors.stopped}`} />
    </span>
  );
}

const WATCHLIST_SYMBOLS = ["SPY", "QQQ"];

function generateMockChartData(symbol: string, timeframe: string) {
  const points = timeframe === "1D" ? 78 : timeframe === "1W" ? 35 : 30;
  const basePrice = symbol === "SPY" ? 590 : symbol === "QQQ" ? 520 : 15;
  const volatility = symbol === "SH" ? 0.002 : 0.003;
  
  const data = [];
  let price = basePrice;
  const now = Date.now();
  const interval = timeframe === "1D" ? 5 * 60 * 1000 : timeframe === "1W" ? 4 * 60 * 60 * 1000 : 24 * 60 * 60 * 1000;
  
  for (let i = points; i >= 0; i--) {
    const change = (Math.random() - 0.48) * basePrice * volatility;
    price = Math.max(price + change, basePrice * 0.95);
    data.push({
      time: now - i * interval,
      price: parseFloat(price.toFixed(2)),
    });
  }
  return data;
}

const TERMINAL_TABS = [
  { id: "pnl", label: "P&L", icon: DollarSign },
  { id: "assets", label: "Assets", icon: Wallet },
  { id: "orders", label: "Orders", icon: FileText },
  { id: "history", label: "History", icon: History },
];

const TRADING_UNIVERSE_SIZE = 20;

interface StatusStripProps {
  marketStatus: MarketStatusV2 | undefined;
  botStatus: BotStatus;
  isConnected: boolean;
  settings: BotSettings | undefined;
}

function StatusStrip({ marketStatus, botStatus, isConnected, settings }: StatusStripProps) {
  const isMarketOpen = marketStatus?.is_open ?? false;
  const entryWindow = isMarketOpen && marketStatus?.next_close;
  const isPaper = settings?.isPaperTrading !== false;
  
  const getMarketStatusColor = () => {
    if (!isConnected) return "text-red-400";
    if (isMarketOpen) return "text-emerald-400";
    return "text-muted-foreground";
  };

  const getEntryStatusColor = () => {
    if (botStatus.status === "active" && isMarketOpen) return "text-emerald-400";
    if (botStatus.status === "paused") return "text-amber-400";
    return "text-muted-foreground";
  };

  return (
    <div className="status-strip flex-wrap gap-3 sm:gap-6">
      <div className="flex items-center gap-2">
        <Radio className={`w-3.5 h-3.5 ${getMarketStatusColor()}`} />
        <span className="text-xs">
          <span className="text-muted-foreground">Market:</span>{" "}
          <span className={getMarketStatusColor()}>{isMarketOpen ? "Open" : "Closed"}</span>
        </span>
      </div>
      
      <div className="flex items-center gap-2">
        <Clock className={`w-3.5 h-3.5 ${getEntryStatusColor()}`} />
        <span className="text-xs">
          <span className="text-muted-foreground">Entry:</span>{" "}
          <span className={getEntryStatusColor()}>
            {botStatus.status === "active" && isMarketOpen ? "Allowed" : "Blocked"}
          </span>
        </span>
      </div>

      <div className="flex items-center gap-2">
        <Badge 
          variant="outline" 
          className={`text-[10px] px-2 py-0.5 ${isPaper 
            ? "bg-amber-500/10 text-amber-400 border-amber-500/30" 
            : "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
          }`}
          data-testid="badge-trading-mode"
        >
          {isPaper ? "PAPER" : "LIVE"}
        </Badge>
      </div>

      <div className="flex items-center gap-2">
        <Layers className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="text-xs">
          <span className="text-muted-foreground">Universe:</span>{" "}
          <span className="font-mono num-webull">{TRADING_UNIVERSE_SIZE}</span>
        </span>
      </div>
    </div>
  );
}

interface DailyAnalytics {
  ok: boolean;
  reason?: string;
  reportDate?: string;
  periodStart?: string;
  periodEnd?: string;
  tradingDays?: number;
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

function StatCard({ title, value, subtitle, trend }: {
  title: string;
  value: string;
  subtitle?: string;
  trend?: "up" | "down" | "neutral";
}) {
  const trendColor = trend === "up" ? "ticker-positive" : trend === "down" ? "ticker-negative" : "text-foreground";
  const trendBg = trend === "up" ? "bg-emerald-500/5" : trend === "down" ? "bg-red-500/5" : "";
  
  return (
    <Card className={`border-border/40 ${trendBg} overflow-visible`}>
      <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
        <CardTitle className="label-webull text-[10px]">{title}</CardTitle>
        <div className={`p-1 rounded ${trend === "up" ? "bg-emerald-500/10" : trend === "down" ? "bg-red-500/10" : "bg-muted"}`}>
          {trend === "up" && <TrendingUp className="w-3 h-3 text-emerald-400" />}
          {trend === "down" && <TrendingDown className="w-3 h-3 text-red-400" />}
          {trend === "neutral" && <Activity className="w-3 h-3 text-muted-foreground" />}
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className={`text-xl sm:text-2xl font-bold font-mono tracking-tight ${trendColor}`}>
          {value}
        </div>
        {subtitle && (
          <p className="text-[10px] text-muted-foreground mt-1.5 num-webull truncate">
            {subtitle}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function PnLTab({ account }: { account: any }) {
  const dailyQuery = useQuery<DailyAnalytics>({
    queryKey: ["/api/analytics/daily"],
    refetchInterval: 30000,
  });

  const todayPL = account?.todayPL ?? 0;
  const todayPLPercent = account?.todayPLPercent ?? 0;
  const pnlTrend = todayPL > 0 ? "up" : todayPL < 0 ? "down" : "neutral";

  const d = dailyQuery.data?.daily;
  const periodStart = dailyQuery.data?.periodStart;
  const periodEnd = dailyQuery.data?.periodEnd;
  const tradingDays = dailyQuery.data?.tradingDays;
  const periodSubtitle = tradingDays && periodStart && periodEnd 
    ? `Last ${tradingDays} trading days (${periodStart} → ${periodEnd})`
    : formatPercent(todayPLPercent);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard 
          title="Period P&L" 
          value={formatCurrency(todayPL)}
          subtitle={periodSubtitle}
          trend={pnlTrend}
        />
        <StatCard 
          title="Win Rate" 
          value={d ? `${d.winRate.toFixed(1)}%` : "--"}
          subtitle={d ? `${d.wins}W / ${d.losses}L` : "No data"}
          trend={d && d.winRate >= 50 ? "up" : d && d.winRate < 50 ? "down" : "neutral"}
        />
        <StatCard 
          title="Total Trades" 
          value={d ? d.tradeCount.toString() : "--"}
          subtitle={d ? `${d.closedTrades} closed` : "No data"}
          trend="neutral"
        />
        <StatCard 
          title="Profit Factor" 
          value={d ? d.profitFactor.toFixed(2) : "--"}
          trend={d && d.profitFactor >= 1 ? "up" : d && d.profitFactor < 1 ? "down" : "neutral"}
        />
      </div>

      {d && (
        <div className="grid grid-cols-2 gap-4">
          <StatCard 
            title="Avg Win" 
            value={formatCurrency(d.avgWin)}
            trend="up"
          />
          <StatCard 
            title="Avg Loss" 
            value={formatCurrency(d.avgLoss)}
            trend="down"
          />
        </div>
      )}

      {!d && !dailyQuery.isLoading && (
        <Card className="border-border/50">
          <CardContent className="py-12 text-center">
            <DollarSign className="w-10 h-10 mx-auto mb-3 text-muted-foreground/30" />
            <p className="text-muted-foreground">No P&L report available yet</p>
            <p className="text-xs text-muted-foreground/70 mt-1">Generate a report to see period analytics</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function AssetsTab({ 
  account, 
  positions, 
  isConnected,
  activeSymbol,
  setActiveSymbol,
  chartTimeframe,
  setChartTimeframe,
  chartData,
  isPriceUp,
  priceChange,
  priceChangePercent,
  botStatus,
  startBotMutation,
  pauseBotMutation,
  setConfirmAction,
  recentOrders,
}: any) {
  const topPositions = positions.slice(0, 3);

  return (
    <PageGrid className="lg:items-start">
      <GridCell span={3} className="order-2 lg:order-1">
        <Card className="border-border/50">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-muted-foreground" />
              <CardTitle className="heading-webull text-sm">Watchlist</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-1">
            {WATCHLIST_SYMBOLS.map((symbol) => {
              const isActive = symbol === activeSymbol;
              const pos = positions.find((p: any) => p.symbol === symbol);
              return (
                <button
                  key={symbol}
                  onClick={() => setActiveSymbol(symbol)}
                  className={`w-full flex items-center justify-between p-3 rounded-md transition-colors ${
                    isActive 
                      ? 'bg-accent/50 border border-accent-border' 
                      : 'hover-elevate border border-transparent'
                  }`}
                  data-testid={`watchlist-${symbol}`}
                >
                  <div className="flex items-center gap-2">
                    <CircleDot className={`w-3 h-3 ${isActive ? 'text-accent-foreground' : 'text-muted-foreground'}`} />
                    <span className={`font-mono heading-webull ${isActive ? '' : 'text-foreground/80'}`}>
                      {symbol}
                    </span>
                    {pos && (
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                        {pos.qty}
                      </Badge>
                    )}
                  </div>
                  <div className="text-right">
                    {pos ? (
                      <span className={`text-xs font-mono num-webull ${pos.unrealizedPL >= 0 ? 'ticker-positive' : 'ticker-negative'}`}>
                        {formatCurrency(pos.unrealizedPL)}
                      </span>
                    ) : (
                      <span className="text-xs text-muted-foreground">--</span>
                    )}
                  </div>
                </button>
              );
            })}
          </CardContent>
        </Card>
      </GridCell>

      <GridCell span={6} className="order-1 lg:order-2">
        <Card className="border-border/50">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <span className="text-xl heading-webull font-mono" data-testid="text-chart-symbol">{activeSymbol}</span>
                <div className="flex items-center gap-1">
                  {isPriceUp ? (
                    <ArrowUpRight className="w-4 h-4 text-emerald-500" />
                  ) : (
                    <ArrowDownRight className="w-4 h-4 text-red-500" />
                  )}
                  <span className={`text-sm font-mono num-webull ${isPriceUp ? 'ticker-positive' : 'ticker-negative'}`}>
                    {isPriceUp ? '+' : ''}{priceChange.toFixed(2)} ({formatPercent(priceChangePercent)})
                  </span>
                </div>
              </div>
              <Tabs value={chartTimeframe} onValueChange={setChartTimeframe}>
                <TabsList className="h-8">
                  <TabsTrigger value="1D" className="text-xs px-2.5" data-testid="button-timeframe-1D">1D</TabsTrigger>
                  <TabsTrigger value="1W" className="text-xs px-2.5" data-testid="button-timeframe-1W">1W</TabsTrigger>
                  <TabsTrigger value="1M" className="text-xs px-2.5" data-testid="button-timeframe-1M">1M</TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-[240px] sm:h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                  <XAxis 
                    dataKey="time" 
                    tickFormatter={(t) => {
                      const d = new Date(t);
                      return chartTimeframe === "1D" 
                        ? d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
                        : d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                    }}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                    minTickGap={40}
                  />
                  <YAxis
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={(v) => `$${v.toFixed(0)}`}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                    width={50}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(var(--card))',
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '6px',
                      fontSize: '12px',
                    }}
                    labelFormatter={(t) => new Date(t).toLocaleString()}
                    formatter={(value: number) => [`$${value.toFixed(2)}`, activeSymbol]}
                  />
                  <Line
                    type="monotone"
                    dataKey="price"
                    stroke={isPriceUp ? '#22c55e' : '#ef4444'}
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </GridCell>

      <GridCell span={3} className="order-3 space-y-4">
        <Card className="border-border/50">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-muted-foreground" />
                <CardTitle className="heading-webull text-sm">Bot Controls</CardTitle>
              </div>
              <StatusIndicator status={botStatus.status} />
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="label-webull">Status</span>
              <Badge 
                variant={botStatus.status === "active" ? "default" : "secondary"}
                className={`text-[10px] ${botStatus.status === "active" ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" : ""}`}
                data-testid="badge-bot-status"
              >
                {botStatus.status.charAt(0).toUpperCase() + botStatus.status.slice(1)}
              </Badge>
            </div>
            
            <div className="grid grid-cols-2 gap-2">
              <Button
                size="sm"
                onClick={() => startBotMutation.mutate()}
                disabled={botStatus.status === "active" || startBotMutation.isPending}
                className="bg-emerald-600 hover:bg-emerald-700 text-white text-xs"
                data-testid="button-start-bot"
              >
                {startBotMutation.isPending ? (
                  <Loader2 className="w-3 h-3 animate-spin mr-1" />
                ) : (
                  <Play className="w-3 h-3 mr-1" />
                )}
                Start
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={() => pauseBotMutation.mutate()}
                disabled={botStatus.status !== "active" || pauseBotMutation.isPending}
                className="text-xs"
                data-testid="button-pause-bot"
              >
                <Pause className="w-3 h-3 mr-1" />
                Pause
              </Button>
            </div>
            
            <div className="grid grid-cols-2 gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => setConfirmAction("stop")}
                disabled={botStatus.status === "stopped"}
                className="text-xs"
                data-testid="button-stop-bot"
              >
                <Square className="w-3 h-3 mr-1" />
                Stop
              </Button>
              <Button
                size="sm"
                variant="destructive"
                onClick={() => setConfirmAction("halt")}
                className="text-xs"
                data-testid="button-halt-bot"
              >
                <AlertTriangle className="w-3 h-3 mr-1" />
                Halt
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/50">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Target className="w-4 h-4 text-muted-foreground" />
                <CardTitle className="heading-webull text-sm">Positions</CardTitle>
              </div>
              <Badge variant="outline" className="text-[10px] num-webull">
                {positions.length}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            {topPositions.length === 0 ? (
              <div className="py-6 text-center text-xs text-muted-foreground">
                No open positions
              </div>
            ) : (
              <div className="space-y-2">
                {topPositions.map((pos: any) => (
                  <div
                    key={pos.symbol}
                    className={`flex items-center justify-between p-2 rounded-md border ${
                      pos.symbol === activeSymbol 
                        ? 'bg-accent/30 border-accent-border' 
                        : 'border-border/50'
                    }`}
                    data-testid={`position-row-${pos.symbol}`}
                  >
                    <div>
                      <span className="font-mono heading-webull text-sm">{pos.symbol}</span>
                      <div className="text-[10px] text-muted-foreground num-webull">
                        {pos.qty} @ {formatCurrency(pos.avgEntryPrice)}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`font-mono text-sm num-webull ${pos.unrealizedPL >= 0 ? 'ticker-positive' : 'ticker-negative'}`}>
                        {formatCurrency(pos.unrealizedPL)}
                      </div>
                      <div className={`text-[10px] num-webull ${pos.unrealizedPL >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {formatPercent(pos.unrealizedPLPercent)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="border-border/50">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-muted-foreground" />
                <CardTitle className="heading-webull text-sm">Recent Orders</CardTitle>
              </div>
              <span className="label-webull font-mono">{activeSymbol}</span>
            </div>
          </CardHeader>
          <CardContent>
            {recentOrders.length === 0 ? (
              <div className="py-6 text-center text-xs text-muted-foreground">
                No orders for {activeSymbol}
              </div>
            ) : (
              <div className="space-y-2">
                {recentOrders.map((order: any, idx: number) => (
                  <div
                    key={order.id || idx}
                    className="flex items-center justify-between p-2 rounded-md border border-border/50 text-xs"
                    data-testid={`order-row-${idx}`}
                  >
                    <div className="flex items-center gap-2">
                      <Badge 
                        variant="outline" 
                        className={`text-[10px] px-1.5 ${order.side === 'buy' ? 'text-emerald-400 border-emerald-500/30' : 'text-red-400 border-red-500/30'}`}
                      >
                        {order.side?.toUpperCase()}
                      </Badge>
                      <span className="font-mono num-webull">{order.quantity}</span>
                    </div>
                    <div className="text-right">
                      <span className="font-mono num-webull">{formatCurrency(order.price || 0)}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </GridCell>
    </PageGrid>
  );
}

function OrdersTab({ trades }: { trades: Trade[] }) {
  const recentTrades = trades.slice(0, 20);

  return (
    <Card className="border-border/50">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-muted-foreground" />
            <CardTitle className="heading-webull text-sm">Recent Orders</CardTitle>
          </div>
          <Badge variant="outline" className="text-[10px] num-webull">{recentTrades.length}</Badge>
        </div>
      </CardHeader>
      <CardContent>
        {recentTrades.length === 0 ? (
          <div className="py-12 text-center text-muted-foreground">
            <FileText className="w-10 h-10 mx-auto mb-3 opacity-30" />
            <p className="text-sm">No orders yet</p>
          </div>
        ) : (
          <div className="space-y-2">
            {recentTrades.map((trade) => (
              <div
                key={trade.id}
                className="flex items-center justify-between p-3 rounded-md border border-border/50"
                data-testid={`trade-row-${trade.id}`}
              >
                <div className="flex items-center gap-3">
                  {trade.side === "buy" ? (
                    <ArrowUpRight className="w-4 h-4 text-emerald-500" />
                  ) : (
                    <ArrowDownRight className="w-4 h-4 text-red-500" />
                  )}
                  <div>
                    <span className="font-mono heading-webull">{trade.symbol}</span>
                    <span className="text-muted-foreground text-sm ml-2 num-webull">
                      {trade.quantity} @ {formatCurrency(Number(trade.price))}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={trade.status === "filled" ? "default" : "secondary"} className="text-[10px]">
                    {trade.status}
                  </Badge>
                  <span className="text-xs text-muted-foreground font-mono num-webull">
                    {trade.timestamp ? new Date(trade.timestamp).toLocaleTimeString() : "—"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface MonthlyAnalytics {
  ok: boolean;
  month: string;
  summary: {
    trades: number;
    netPnl: number;
    winRate: number;
    profitFactor: number;
    wins: number;
    losses: number;
  };
  dailySeries: Array<{ date: string; netPnl: number; tradeCount: number }>;
}

function HistoryTab({ trades }: { trades: Trade[] }) {
  const [dateFilter, setDateFilter] = useState<string>("all");
  
  const currentMonth = new Date().toISOString().slice(0, 7);
  const analyticsQuery = useQuery<MonthlyAnalytics>({
    queryKey: ["/api/analytics/monthly", currentMonth],
    refetchInterval: 30000,
  });

  const filteredTrades = useMemo(() => {
    if (dateFilter === "all") return trades;
    const now = new Date();
    const filterDate = new Date();
    if (dateFilter === "today") {
      filterDate.setHours(0, 0, 0, 0);
    } else if (dateFilter === "week") {
      filterDate.setDate(now.getDate() - 7);
    } else if (dateFilter === "month") {
      filterDate.setMonth(now.getMonth() - 1);
    }
    return trades.filter(t => t.timestamp && new Date(t.timestamp) >= filterDate);
  }, [trades, dateFilter]);

  const performanceData = useMemo(() => {
    const dailySeries = analyticsQuery.data?.dailySeries;
    if (!dailySeries || dailySeries.length === 0) return [];
    
    let cumPnl = 0;
    return dailySeries.map(d => {
      cumPnl += d.netPnl;
      return {
        date: new Date(d.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
        cumPnl: parseFloat(cumPnl.toFixed(2)),
        dailyPnl: parseFloat(d.netPnl.toFixed(2)),
      };
    });
  }, [analyticsQuery.data?.dailySeries]);

  const periodDates = useMemo(() => {
    const dailySeries = analyticsQuery.data?.dailySeries;
    if (!dailySeries || dailySeries.length === 0) return { start: "--", end: "--" };
    return {
      start: dailySeries[0].date,
      end: dailySeries[dailySeries.length - 1].date,
    };
  }, [analyticsQuery.data?.dailySeries]);

  const handleExportCSV = () => {
    const headers = ["ID", "Symbol", "Side", "Quantity", "Price", "Status", "Timestamp", "Order ID"];
    const rows = filteredTrades.map(t => [
      t.id,
      t.symbol,
      t.side,
      t.quantity,
      t.price,
      t.status,
      t.timestamp ? new Date(t.timestamp).toISOString() : "",
      t.orderId || "",
    ]);
    const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trades_${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      <Card className="border-border/50">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between gap-2">
            <div>
              <CardTitle className="heading-webull text-sm flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-muted-foreground" />
                Performance
              </CardTitle>
              <p className="text-xs text-muted-foreground mt-0.5 num-webull">
                Period: {periodDates.start} → {periodDates.end}
              </p>
            </div>
            <Tabs value={dateFilter} onValueChange={setDateFilter}>
              <TabsList className="h-8">
                <TabsTrigger value="all" className="text-xs px-2" data-testid="button-filter-all">All</TabsTrigger>
                <TabsTrigger value="today" className="text-xs px-2" data-testid="button-filter-today">Today</TabsTrigger>
                <TabsTrigger value="week" className="text-xs px-2" data-testid="button-filter-week">Week</TabsTrigger>
                <TabsTrigger value="month" className="text-xs px-2" data-testid="button-filter-month">Month</TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        </CardHeader>
        <CardContent>
          {performanceData.length === 0 ? (
            <div className="py-8 text-center text-muted-foreground">
              <Activity className="w-8 h-8 mx-auto mb-2 opacity-30" />
              <p className="text-sm">No performance data for this period yet.</p>
            </div>
          ) : (
            <div className="h-[160px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={performanceData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="cumPnlGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(142, 60%, 50%)" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(142, 60%, 50%)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis 
                    dataKey="date" 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                  />
                  <YAxis 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                    tickFormatter={(v) => `$${v}`}
                    width={50}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      background: 'hsl(var(--card))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '6px',
                      fontSize: '12px'
                    }}
                    formatter={(value: number) => [`$${value.toFixed(2)}`, 'Cumulative P&L']}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="cumPnl" 
                    stroke="hsl(142, 60%, 50%)" 
                    strokeWidth={2}
                    fill="url(#cumPnlGradient)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="border-border/50">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="flex items-center gap-2">
              <History className="w-4 h-4 text-muted-foreground" />
              <CardTitle className="heading-webull text-sm">Trade History</CardTitle>
              <Badge variant="outline" className="text-[10px] num-webull">{filteredTrades.length}</Badge>
            </div>
            <Button
              size="sm"
              variant="outline"
              onClick={handleExportCSV}
              className="text-xs"
              data-testid="button-export-csv"
            >
              <Download className="w-3 h-3 mr-1" />
              CSV
            </Button>
          </div>
        </CardHeader>
        <CardContent>
        {filteredTrades.length === 0 ? (
          <div className="py-12 text-center text-muted-foreground">
            <History className="w-10 h-10 mx-auto mb-3 opacity-30" />
            <p className="text-sm">No trades in this period</p>
          </div>
        ) : (
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            {filteredTrades.map((trade) => (
              <div
                key={trade.id}
                className="flex items-center justify-between p-3 rounded-md border border-border/50"
                data-testid={`history-row-${trade.id}`}
              >
                <div className="flex items-center gap-3">
                  {trade.side === "buy" ? (
                    <ArrowUpRight className="w-4 h-4 text-emerald-500" />
                  ) : (
                    <ArrowDownRight className="w-4 h-4 text-red-500" />
                  )}
                  <div>
                    <span className="font-mono heading-webull">{trade.symbol}</span>
                    <span className="text-muted-foreground text-sm ml-2 num-webull">
                      {trade.quantity} @ {formatCurrency(Number(trade.price))}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={trade.status === "filled" ? "default" : "secondary"} className="text-[10px]">
                    {trade.status}
                  </Badge>
                  <span className="text-xs text-muted-foreground font-mono num-webull">
                    {trade.timestamp ? new Date(trade.timestamp).toLocaleDateString() : "—"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
        </CardContent>
      </Card>
    </div>
  );
}

export default function Dashboard() {
  const { toast } = useToast();
  const { kind } = usePortfolio();
  const stream = usePortfolioStream();
  const [, setLocation] = useLocation();
  const searchString = useSearch();
  
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [confirmAction, setConfirmAction] = useState<"stop" | "halt" | null>(null);
  const [activeSymbol, setActiveSymbol] = useState<string>("SPY");
  const [chartTimeframe, setChartTimeframe] = useState<string>("1D");

  const params = new URLSearchParams(searchString);
  const tabFromUrl = params.get("tab");
  const validTabs = TERMINAL_TABS.map(t => t.id);
  const initialTab = tabFromUrl && validTabs.includes(tabFromUrl) ? tabFromUrl : "pnl";
  const [activeTab, setActiveTab] = useState(initialTab);

  useEffect(() => {
    const params = new URLSearchParams(searchString);
    const tab = params.get("tab");
    if (tab && validTabs.includes(tab)) {
      setActiveTab(tab);
    } else if (tab && !validTabs.includes(tab)) {
      setActiveTab("pnl");
    }
  }, [searchString]);

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    setLocation(`/dashboard?tab=${tab}`);
  };

  const botStatusQuery = useQuery<BotStatus>({
    queryKey: ["/api/bot/status"],
    refetchInterval: 2000,
  });

  const settingsQuery = useQuery<BotSettings>({
    queryKey: ["/api/settings"],
  });

  const marketStatusQuery = useQuery<MarketStatusV2>({
    queryKey: ["/api/market-status"],
    refetchInterval: 30000,
  });

  const alertsQuery = useQuery<Alert[]>({
    queryKey: ["/api/alerts"],
    refetchInterval: 5000,
  });

  const tradesQuery = useQuery<Trade[]>({
    queryKey: ["/api/trades"],
    refetchInterval: 10000,
  });

  const startBotMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/bot/start"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bot/status"] });
      toast({ title: "Bot started", description: "AtoBot is now actively trading" });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to start bot", variant: "destructive" });
    },
  });

  const pauseBotMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/bot/pause"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bot/status"] });
      toast({ title: "Bot paused" });
    },
  });

  const stopBotMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/bot/stop"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/bot/status"] });
      toast({ title: "Bot stopped" });
    },
  });

  const approveAlertMutation = useMutation({
    mutationFn: (alertId: string) => apiRequest("POST", `/api/alerts/${alertId}/approve`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/alerts"] });
      setSelectedAlert(null);
      toast({ title: "Trade approved" });
    },
  });

  const denyAlertMutation = useMutation({
    mutationFn: (alertId: string) => apiRequest("POST", `/api/alerts/${alertId}/deny`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/alerts"] });
      setSelectedAlert(null);
      toast({ title: "Trade denied" });
    },
  });

  const pendingAlerts = alertsQuery.data?.filter((a) => a.requiresApproval && !a.isResolved) || [];
  if (pendingAlerts.length > 0 && !selectedAlert) {
    setSelectedAlert(pendingAlerts[0]);
  }

  const botStatus = botStatusQuery.data ?? { status: "stopped", lastAnalysis: null, currentAction: null, errorMessage: null };
  const account = stream.account;
  const positions = stream.positions;
  const isConnected = stream.connected;
  const isMarketOpen = marketStatusQuery.data?.is_open ?? false;
  const trades = tradesQuery.data || [];

  const chartData = useMemo(() => {
    return generateMockChartData(activeSymbol, chartTimeframe);
  }, [activeSymbol, chartTimeframe]);

  const recentOrders = useMemo(() => {
    return trades
      .filter(t => t.symbol === activeSymbol)
      .slice(0, 5);
  }, [trades, activeSymbol]);

  const handleConfirmAction = () => {
    if (confirmAction === "stop") {
      stopBotMutation.mutate();
    } else if (confirmAction === "halt") {
      stopBotMutation.mutate();
    }
    setConfirmAction(null);
  };

  const pnlTrend = (account?.todayPL ?? 0) >= 0;

  const priceChange = chartData.length >= 2 
    ? chartData[chartData.length - 1].price - chartData[0].price 
    : 0;
  const priceChangePercent = chartData.length >= 2 && chartData[0].price > 0
    ? (priceChange / chartData[0].price) * 100
    : 0;
  const isPriceUp = priceChange >= 0;

  const settings = settingsQuery.data;

  return (
    <div className="min-h-screen bg-background">
      <PageContainer className="space-y-3 sm:space-y-4">
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="terminal-container p-4 sm:p-6"
        >
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className={`p-2 rounded-lg flex-shrink-0 ${isConnected ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
                {isConnected ? (
                  <Wifi className="w-5 h-5 text-emerald-400" />
                ) : (
                  <WifiOff className="w-5 h-5 text-red-400" />
                )}
              </div>
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <h1 className="text-lg sm:text-xl heading-webull">Terminal</h1>
                  <StatusIndicator status={botStatus.status} />
                </div>
                <p className="text-xs text-muted-foreground capitalize">{botStatus.status}</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3 flex-shrink-0">
              {account && (
                <div className="flex items-center gap-3">
                  <div className="text-right">
                    <p className="label-webull text-[10px]">Equity</p>
                    <p className="font-mono num-webull text-sm">{formatCurrency(account.totalEquity)}</p>
                  </div>
                  <div className="text-right">
                    <p className="label-webull text-[10px]">Today</p>
                    <p className={`font-mono num-webull text-sm ${pnlTrend ? 'ticker-positive' : 'ticker-negative'}`}>
                      {formatPercent(account.todayPLPercent)}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          <StatusStrip 
            marketStatus={marketStatusQuery.data} 
            botStatus={botStatus}
            isConnected={isConnected}
            settings={settings}
          />

          <div className="mt-4">
            <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
              <TabsList className="tabs-segmented h-10 w-full grid grid-cols-4 sm:w-auto sm:inline-flex">
                {TERMINAL_TABS.map((tab) => (
                  <TabsTrigger
                    key={tab.id}
                    value={tab.id}
                    className="flex items-center justify-center gap-1.5 text-xs sm:text-sm px-3 sm:px-5 rounded-md transition-all"
                    data-testid={`tab-${tab.id}`}
                  >
                    <tab.icon className="w-3.5 h-3.5" />
                    <span>{tab.label}</span>
                  </TabsTrigger>
                ))}
              </TabsList>
            </Tabs>
          </div>

          <div className="mt-4">
            {activeTab === "pnl" && <PnLTab account={account} />}
            {activeTab === "assets" && (
              <AssetsTab
                account={account}
                positions={positions}
                isConnected={isConnected}
                activeSymbol={activeSymbol}
                setActiveSymbol={setActiveSymbol}
                chartTimeframe={chartTimeframe}
                setChartTimeframe={setChartTimeframe}
                chartData={chartData}
                isPriceUp={isPriceUp}
                priceChange={priceChange}
                priceChangePercent={priceChangePercent}
                botStatus={botStatus}
                startBotMutation={startBotMutation}
                pauseBotMutation={pauseBotMutation}
                setConfirmAction={setConfirmAction}
                recentOrders={recentOrders}
              />
            )}
            {activeTab === "orders" && <OrdersTab trades={trades} />}
            {activeTab === "history" && <HistoryTab trades={trades} />}
          </div>
        </motion.div>
      </PageContainer>

      <AlertDialog open={confirmAction !== null} onOpenChange={() => setConfirmAction(null)}>
        <AlertDialogContent className="border-border/50">
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2">
              {confirmAction === "halt" && <AlertTriangle className="w-5 h-5 text-red-500" />}
              {confirmAction === "halt" ? "Emergency Halt" : "Stop Bot"}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {confirmAction === "halt"
                ? "This will immediately stop the bot and close all positions. This action cannot be undone."
                : "This will stop the bot. Open positions will remain. Are you sure?"}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={handleConfirmAction}
              className={confirmAction === "halt" ? "bg-red-600 hover:bg-red-700" : ""}
            >
              {confirmAction === "halt" ? "Halt Now" : "Stop Bot"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertModal
        alert={selectedAlert}
        onApprove={(id) => approveAlertMutation.mutate(id)}
        onDeny={(id) => denyAlertMutation.mutate(id)}
        onClose={() => setSelectedAlert(null)}
      />
    </div>
  );
}
