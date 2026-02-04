import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { RefreshCw } from "lucide-react";
import { format } from "date-fns";

interface PortfolioHistoryPoint {
  timestamp: number;
  equity: number;
  profitLoss: number;
}

interface PortfolioChartProps {
  currentValue: number;
  todayPL: number;
  todayPLPercent: number;
}

const periods = [
  { label: "1D", value: "1D" },
  { label: "1M", value: "1M" },
  { label: "1Y", value: "1Y" },
  { label: "All", value: "all" },
];

export function PortfolioChart({ currentValue, todayPL, todayPLPercent }: PortfolioChartProps) {
  const [selectedPeriod, setSelectedPeriod] = useState("1D");

  const historyQuery = useQuery<PortfolioHistoryPoint[]>({
    queryKey: ["/api/portfolio/history", selectedPeriod],
    queryFn: async () => {
      const res = await fetch(`/api/portfolio/history?period=${selectedPeriod}`);
      if (!res.ok) throw new Error("Failed to fetch");
      return res.json();
    },
    refetchInterval: 60000,
  });

  const data = historyQuery.data || [];
  const isPositive = todayPL >= 0;
  const chartColor = isPositive ? "#22c55e" : "#ef4444";
  const fillColor = isPositive ? "rgba(34, 197, 94, 0.1)" : "rgba(239, 68, 68, 0.1)";

  const formatXAxis = (timestamp: number) => {
    if (selectedPeriod === "1D") {
      return format(new Date(timestamp), "h:mm a");
    } else if (selectedPeriod === "1M") {
      return format(new Date(timestamp), "MMM d");
    } else {
      return format(new Date(timestamp), "MMM yyyy");
    }
  };

  const formatTooltip = (value: number) => {
    return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const minEquity = data.length > 0 ? Math.min(...data.map(d => d.equity)) * 0.999 : 0;
  const maxEquity = data.length > 0 ? Math.max(...data.map(d => d.equity)) * 1.001 : 100000;

  return (
    <Card>
      <CardHeader className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2">
        <CardTitle className="text-sm sm:text-base font-medium">Your Portfolio</CardTitle>
        <div className="flex items-center gap-1 flex-wrap">
          {periods.map((period) => (
            <Button
              key={period.value}
              variant={selectedPeriod === period.value ? "default" : "ghost"}
              size="sm"
              className="h-8 px-2 sm:px-3 text-xs"
              onClick={() => setSelectedPeriod(period.value)}
              data-testid={`button-period-${period.value}`}
            >
              {period.label}
            </Button>
          ))}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => historyQuery.refetch()}
            data-testid="button-refresh-chart"
          >
            <RefreshCw className={`h-4 w-4 ${historyQuery.isFetching ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 sm:space-y-4">
        <div>
          <div className="flex items-baseline gap-2 flex-wrap">
            <span className="text-xl sm:text-2xl font-bold" data-testid="text-portfolio-value">
              ${currentValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
            <span className={`text-xs sm:text-sm font-medium ${isPositive ? 'text-green-500' : 'text-red-500'}`} data-testid="text-portfolio-change">
              {isPositive ? '+' : ''}{todayPLPercent.toFixed(2)}%
            </span>
          </div>
          <p className="text-[10px] sm:text-xs text-muted-foreground">
            {format(new Date(), "MMMM d, hh:mm a")} PST
          </p>
        </div>

        {historyQuery.isLoading ? (
          <Skeleton className="h-40 sm:h-48 w-full" />
        ) : data.length === 0 ? (
          <div className="h-40 sm:h-48 flex items-center justify-center text-xs sm:text-sm text-muted-foreground">
            No portfolio history available yet
          </div>
        ) : (
          <div className="h-40 sm:h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data} margin={{ top: 5, right: 5, left: 0, bottom: 5 }}>
                <defs>
                  <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={chartColor} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={chartColor} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={formatXAxis}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                  minTickGap={50}
                />
                <YAxis
                  domain={[minEquity, maxEquity]}
                  tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }}
                  width={45}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'hsl(var(--card))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '6px',
                    fontSize: '12px',
                  }}
                  labelFormatter={(label) => format(new Date(label), "MMM d, h:mm a")}
                  formatter={(value: number) => [formatTooltip(value), "Portfolio"]}
                />
                <Area
                  type="monotone"
                  dataKey="equity"
                  stroke={chartColor}
                  strokeWidth={2}
                  fill="url(#colorEquity)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
