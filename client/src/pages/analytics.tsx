import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BarChart3, TrendingUp, TrendingDown, Activity, AlertCircle } from "lucide-react";
import { PerformanceAnalytics } from "@/components/performance-analytics";
import { RiskDashboard } from "@/components/risk-dashboard";
import { CommunicationStatus } from "@/components/communication-status";
import { PageContainer, PageGrid, GridCell } from "@/components/layout";

interface PeriodData {
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
  dataQuality?: {
    paired_trades: number;
    unpaired_entries: number;
    unpaired_exits: number;
    legacy_orders_without_trade_id: number;
  };
}

interface MonthlyData {
  ok: boolean;
  reason?: string;
  month?: string;
  daysReported?: number;
  monthly?: {
    tradeCount: number;
    closedTrades: number;
    netPnl: number;
    winRate: number;
    profitFactor: number;
    wins: number;
    losses: number;
  };
  dailySeries?: Array<{ date: string; netPnl: number; tradeCount: number }>;
}

function getCurrentMonth(): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/Los_Angeles" })
    .format(new Date())
    .slice(0, 7);
}

function StatCard({ title, value, subtitle, icon: Icon, trend }: {
  title: string;
  value: string;
  subtitle?: string;
  icon: typeof TrendingUp;
  trend?: "up" | "down" | "neutral";
}) {
  const trendColor = trend === "up" ? "ticker-positive" : trend === "down" ? "ticker-negative" : "text-muted-foreground";
  return (
    <Card className="metric-card border-border/50">
      <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
        <CardTitle className="text-xs font-medium text-muted-foreground">{title}</CardTitle>
        <Icon className={`w-3.5 h-3.5 ${trend === "up" ? "text-emerald-400" : trend === "down" ? "text-red-400" : "text-muted-foreground"}`} />
      </CardHeader>
      <CardContent>
        <div className={`text-xl sm:text-2xl font-bold font-mono ${trendColor}`}>{value}</div>
        {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
      </CardContent>
    </Card>
  );
}

function PeriodTab() {
  const { data, isLoading } = useQuery<PeriodData>({
    queryKey: ["/api/analytics/daily"],
    refetchInterval: 30000,
  });

  if (isLoading) {
    return <div className="text-muted-foreground">Loading period analytics...</div>;
  }

  if (!data?.ok || !data.daily) {
    return (
      <Card>
        <CardContent className="py-8 text-center">
          <AlertCircle className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
          <p className="text-muted-foreground">No report yet. Generate a report first.</p>
        </CardContent>
      </Card>
    );
  }

  const d = data.daily;
  const dq = data.dataQuality;
  const pnlTrend = d.netPnl > 0 ? "up" : d.netPnl < 0 ? "down" : "neutral";

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">Report Date: {data.reportDate}</p>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard 
          title="Net P&L" 
          value={`$${d.netPnl.toFixed(2)}`}
          icon={pnlTrend === "up" ? TrendingUp : TrendingDown}
          trend={pnlTrend}
        />
        <StatCard 
          title="Win Rate" 
          value={`${d.winRate.toFixed(1)}%`}
          subtitle={`${d.wins}W / ${d.losses}L`}
          icon={Activity}
          trend={d.winRate >= 50 ? "up" : "down"}
        />
        <StatCard 
          title="Total Trades" 
          value={d.tradeCount.toString()}
          subtitle={`${d.closedTrades} closed`}
          icon={BarChart3}
        />
        <StatCard 
          title="Profit Factor" 
          value={d.profitFactor.toFixed(2)}
          icon={TrendingUp}
          trend={d.profitFactor >= 1 ? "up" : "down"}
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <StatCard 
          title="Avg Win" 
          value={`$${d.avgWin.toFixed(2)}`}
          icon={TrendingUp}
          trend="up"
        />
        <StatCard 
          title="Avg Loss" 
          value={`$${d.avgLoss.toFixed(2)}`}
          icon={TrendingDown}
          trend="down"
        />
      </div>

      {dq && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Data Quality</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Paired Trades</p>
                <p className="font-medium">{dq.paired_trades}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Unpaired Entries</p>
                <p className="font-medium">{dq.unpaired_entries}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Unpaired Exits</p>
                <p className="font-medium">{dq.unpaired_exits}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Legacy Orders</p>
                <p className="font-medium">{dq.legacy_orders_without_trade_id}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function MonthlyTab() {
  const month = getCurrentMonth();
  const { data, isLoading } = useQuery<MonthlyData>({
    queryKey: ["/api/analytics/monthly", month],
    refetchInterval: 60000,
  });

  if (isLoading) {
    return <div className="text-muted-foreground">Loading monthly analytics...</div>;
  }

  if (!data?.ok || !data.monthly) {
    return (
      <Card>
        <CardContent className="py-8 text-center">
          <AlertCircle className="w-8 h-8 mx-auto mb-2 text-muted-foreground" />
          <p className="text-muted-foreground">No data for {month}. Generate daily reports first.</p>
        </CardContent>
      </Card>
    );
  }

  const m = data.monthly;
  const pnlTrend = m.netPnl > 0 ? "up" : m.netPnl < 0 ? "down" : "neutral";

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">Month: {data.month} ({data.daysReported} days reported)</p>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard 
          title="Net P&L" 
          value={`$${m.netPnl.toFixed(2)}`}
          icon={pnlTrend === "up" ? TrendingUp : TrendingDown}
          trend={pnlTrend}
        />
        <StatCard 
          title="Win Rate" 
          value={`${m.winRate.toFixed(1)}%`}
          subtitle={`${m.wins}W / ${m.losses}L`}
          icon={Activity}
          trend={m.winRate >= 50 ? "up" : "down"}
        />
        <StatCard 
          title="Total Trades" 
          value={m.tradeCount.toString()}
          subtitle={`${m.closedTrades} closed`}
          icon={BarChart3}
        />
        <StatCard 
          title="Profit Factor" 
          value={m.profitFactor.toFixed(2)}
          icon={TrendingUp}
          trend={m.profitFactor >= 1 ? "up" : "down"}
        />
      </div>

      {data.dailySeries && data.dailySeries.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Period Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {data.dailySeries.map((day) => (
                <div key={day.date} className="flex items-center justify-between text-sm border-b pb-1">
                  <span className="text-muted-foreground">{day.date}</span>
                  <span className="font-medium tabular-nums">{day.tradeCount} trades</span>
                  <span className={`tabular-nums text-right ${day.netPnl >= 0 ? "ticker-positive" : "ticker-negative"}`}>
                    ${day.netPnl.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

export default function AnalyticsPage() {
  return (
    <PageContainer className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-xl sm:text-2xl font-bold flex items-center gap-2 tracking-tight">
          <div className="p-2 rounded-lg bg-muted">
            <BarChart3 className="w-5 h-5 text-foreground" />
          </div>
          Analytics
        </h1>
        <p className="text-sm text-muted-foreground mt-1">Performance metrics and risk analysis</p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
      <Tabs defaultValue="period" className="w-full">
        <TabsList className="bg-muted/50">
          <TabsTrigger value="period" data-testid="tab-period">Period</TabsTrigger>
          <TabsTrigger value="monthly" data-testid="tab-monthly">Monthly</TabsTrigger>
          <TabsTrigger value="overview" data-testid="tab-overview">Overview</TabsTrigger>
        </TabsList>
        
        <TabsContent value="period" className="mt-4">
          <PeriodTab />
        </TabsContent>
        
        <TabsContent value="monthly" className="mt-4">
          <MonthlyTab />
        </TabsContent>
        
        <TabsContent value="overview" className="mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PerformanceAnalytics />
            <div className="space-y-6">
              <CommunicationStatus />
              <RiskDashboard />
            </div>
          </div>
        </TabsContent>
      </Tabs>
      </motion.div>
    </PageContainer>
  );
}
