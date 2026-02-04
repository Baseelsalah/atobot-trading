import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, 
  ResponsiveContainer, Cell, PieChart, Pie, Legend 
} from "recharts";
import { TrendingUp, TrendingDown, Target, Activity, DollarSign, Percent } from "lucide-react";
import { format } from "date-fns";
import type { Trade } from "@shared/schema";

interface PerformanceStats {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalProfit: number;
  totalLoss: number;
  netProfitLoss: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  profitFactor: number;
}

interface BuyLot {
  price: number;
  quantity: number;
  consumed: boolean;
}

function calculateStats(trades: Trade[]): PerformanceStats {
  const filledTrades = trades.filter(t => t.status === "filled");
  
  const buyLotsBySymbol: Record<string, BuyLot[]> = {};
  const sellTrades: Trade[] = [];
  
  const sortedTrades = [...filledTrades].sort(
    (a, b) => new Date(a.timestamp || 0).getTime() - new Date(b.timestamp || 0).getTime()
  );
  
  for (const trade of sortedTrades) {
    if (trade.side === "buy") {
      if (!buyLotsBySymbol[trade.symbol]) {
        buyLotsBySymbol[trade.symbol] = [];
      }
      buyLotsBySymbol[trade.symbol].push({
        price: trade.price,
        quantity: trade.quantity,
        consumed: false,
      });
    } else {
      sellTrades.push(trade);
    }
  }
  
  let totalProfit = 0;
  let totalLoss = 0;
  let winningTrades = 0;
  let losingTrades = 0;
  let largestWin = 0;
  let largestLoss = 0;
  
  for (const sell of sellTrades) {
    const buyLots = buyLotsBySymbol[sell.symbol] || [];
    const unconsumedBuy = buyLots.find(b => !b.consumed);
    
    if (unconsumedBuy) {
      unconsumedBuy.consumed = true;
      const profit = (sell.price - unconsumedBuy.price) * sell.quantity;
      
      if (profit > 0) {
        totalProfit += profit;
        winningTrades++;
        largestWin = Math.max(largestWin, profit);
      } else {
        totalLoss += Math.abs(profit);
        losingTrades++;
        largestLoss = Math.max(largestLoss, Math.abs(profit));
      }
    }
  }
  
  const totalTrades = winningTrades + losingTrades;
  const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;
  const avgWin = winningTrades > 0 ? totalProfit / winningTrades : 0;
  const avgLoss = losingTrades > 0 ? totalLoss / losingTrades : 0;
  const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? Infinity : 0;
  
  return {
    totalTrades,
    winningTrades,
    losingTrades,
    winRate,
    totalProfit,
    totalLoss,
    netProfitLoss: totalProfit - totalLoss,
    avgWin,
    avgLoss,
    largestWin,
    largestLoss,
    profitFactor,
  };
}

function StatCard({ 
  title, 
  value, 
  icon: Icon, 
  trend, 
  subtitle 
}: { 
  title: string; 
  value: string; 
  icon: React.ElementType; 
  trend?: "up" | "down" | "neutral";
  subtitle?: string;
}) {
  const trendColor = trend === "up" ? "text-green-500" : trend === "down" ? "text-red-500" : "text-muted-foreground";
  
  return (
    <div className="flex items-center gap-3 p-3 rounded-md bg-muted/30">
      <div className={`p-2 rounded-md ${trend === "up" ? "bg-green-500/10" : trend === "down" ? "bg-red-500/10" : "bg-muted"}`}>
        <Icon className={`h-4 w-4 ${trendColor}`} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-muted-foreground">{title}</p>
        <p className={`text-sm font-semibold ${trendColor}`}>{value}</p>
        {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
      </div>
    </div>
  );
}

export function PerformanceAnalytics() {
  const tradesQuery = useQuery<Trade[]>({
    queryKey: ["/api/trades"],
    refetchInterval: 10000,
  });

  const trades = tradesQuery.data || [];
  const stats = calculateStats(trades);
  
  const tradesByDay = trades.reduce((acc, trade) => {
    const day = format(new Date(trade.timestamp || new Date()), "MMM d");
    if (!acc[day]) {
      acc[day] = { day, buys: 0, sells: 0, volume: 0 };
    }
    if (trade.side === "buy") {
      acc[day].buys++;
    } else {
      acc[day].sells++;
    }
    acc[day].volume += trade.totalValue;
    return acc;
  }, {} as Record<string, { day: string; buys: number; sells: number; volume: number }>);
  
  const dailyData = Object.values(tradesByDay).slice(-7);
  
  const winLossData = [
    { name: "Wins", value: stats.winningTrades, color: "#22c55e" },
    { name: "Losses", value: stats.losingTrades, color: "#ef4444" },
  ].filter(d => d.value > 0);

  const tradesBySymbol = trades.reduce((acc, trade) => {
    if (!acc[trade.symbol]) {
      acc[trade.symbol] = { symbol: trade.symbol, count: 0, volume: 0 };
    }
    acc[trade.symbol].count++;
    acc[trade.symbol].volume += trade.totalValue;
    return acc;
  }, {} as Record<string, { symbol: string; count: number; volume: number }>);
  
  const symbolData = Object.values(tradesBySymbol)
    .sort((a, b) => b.volume - a.volume)
    .slice(0, 5);

  if (tradesQuery.isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base font-medium">Performance Analytics</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-64 w-full" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
        <CardTitle className="text-base font-medium flex items-center gap-2">
          <Activity className="h-4 w-4" />
          Performance Analytics
        </CardTitle>
        <Badge variant="outline" className="text-xs">
          {stats.totalTrades} closed trades
        </Badge>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="overview" data-testid="tab-overview">Overview</TabsTrigger>
            <TabsTrigger value="charts" data-testid="tab-charts">Charts</TabsTrigger>
            <TabsTrigger value="breakdown" data-testid="tab-breakdown">Breakdown</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <StatCard
                title="Win Rate"
                value={`${stats.winRate.toFixed(1)}%`}
                icon={Target}
                trend={stats.winRate >= 50 ? "up" : stats.winRate > 0 ? "down" : "neutral"}
              />
              <StatCard
                title="Net P/L"
                value={`$${stats.netProfitLoss.toFixed(2)}`}
                icon={DollarSign}
                trend={stats.netProfitLoss >= 0 ? "up" : "down"}
              />
              <StatCard
                title="Profit Factor"
                value={stats.profitFactor === Infinity ? "N/A" : stats.profitFactor.toFixed(2)}
                icon={TrendingUp}
                trend={stats.profitFactor >= 1 ? "up" : "down"}
              />
              <StatCard
                title="Total Trades"
                value={stats.totalTrades.toString()}
                icon={Activity}
                trend="neutral"
                subtitle={`${stats.winningTrades}W / ${stats.losingTrades}L`}
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <StatCard
                title="Average Win"
                value={`$${stats.avgWin.toFixed(2)}`}
                icon={TrendingUp}
                trend="up"
              />
              <StatCard
                title="Average Loss"
                value={`$${stats.avgLoss.toFixed(2)}`}
                icon={TrendingDown}
                trend="down"
              />
              <StatCard
                title="Largest Win"
                value={`$${stats.largestWin.toFixed(2)}`}
                icon={TrendingUp}
                trend="up"
              />
              <StatCard
                title="Largest Loss"
                value={`$${stats.largestLoss.toFixed(2)}`}
                icon={TrendingDown}
                trend="down"
              />
            </div>
          </TabsContent>

          <TabsContent value="charts" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-medium mb-2">Win/Loss Distribution</p>
                {winLossData.length > 0 ? (
                  <div className="h-40">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={winLossData}
                          cx="50%"
                          cy="50%"
                          innerRadius={40}
                          outerRadius={60}
                          paddingAngle={5}
                          dataKey="value"
                        >
                          {winLossData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Legend />
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-40 flex items-center justify-center text-sm text-muted-foreground">
                    No closed trades yet
                  </div>
                )}
              </div>

              <div>
                <p className="text-sm font-medium mb-2">Daily Trading Activity</p>
                {dailyData.length > 0 ? (
                  <div className="h-40">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={dailyData}>
                        <XAxis dataKey="day" tick={{ fontSize: 10 }} />
                        <YAxis tick={{ fontSize: 10 }} />
                        <Tooltip />
                        <Bar dataKey="buys" fill="#22c55e" name="Buys" />
                        <Bar dataKey="sells" fill="#ef4444" name="Sells" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-40 flex items-center justify-center text-sm text-muted-foreground">
                    No trade data yet
                  </div>
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="breakdown" className="space-y-4">
            <div>
              <p className="text-sm font-medium mb-2">Top Traded Symbols</p>
              {symbolData.length > 0 ? (
                <div className="space-y-2">
                  {symbolData.map((item) => (
                    <div
                      key={item.symbol}
                      className="flex items-center justify-between p-2 rounded-md bg-muted/30"
                      data-testid={`symbol-breakdown-${item.symbol}`}
                    >
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{item.symbol}</Badge>
                        <span className="text-sm text-muted-foreground">{item.count} trades</span>
                      </div>
                      <span className="text-sm font-medium">
                        ${item.volume.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="h-32 flex items-center justify-center text-sm text-muted-foreground">
                  No trades yet
                </div>
              )}
            </div>

            <div>
              <p className="text-sm font-medium mb-2">P/L Summary</p>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-md bg-green-500/10">
                  <p className="text-xs text-muted-foreground">Total Profits</p>
                  <p className="text-lg font-semibold text-green-500">
                    ${stats.totalProfit.toFixed(2)}
                  </p>
                </div>
                <div className="p-3 rounded-md bg-red-500/10">
                  <p className="text-xs text-muted-foreground">Total Losses</p>
                  <p className="text-lg font-semibold text-red-500">
                    ${stats.totalLoss.toFixed(2)}
                  </p>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
