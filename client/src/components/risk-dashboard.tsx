import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { Shield, AlertTriangle, Activity, TrendingUp, Percent, Target } from "lucide-react";

interface VolatilityData {
  symbol: string;
  atr: number;
  atrPercent: number;
  dailyRange: number;
  volatilityLevel: "low" | "medium" | "high" | "extreme";
}

interface RiskMetrics {
  portfolioVolatility: number;
  portfolioHeatLevel: number;
  maxDrawdown: number;
  currentDrawdown: number;
  riskCapacity: number;
  correlationRisk: number;
}

interface RiskDashboardData {
  metrics: RiskMetrics;
  volatilityBySymbol: Record<string, VolatilityData>;
  recommendations: string[];
}

function MetricCard({
  label,
  value,
  max,
  icon: Icon,
  status,
}: {
  label: string;
  value: number;
  max: number;
  icon: React.ElementType;
  status: "good" | "warning" | "danger";
}) {
  const statusColors = {
    good: "text-green-500 bg-green-500/10",
    warning: "text-yellow-500 bg-yellow-500/10",
    danger: "text-red-500 bg-red-500/10",
  };

  const progressColors = {
    good: "bg-green-500",
    warning: "bg-yellow-500",
    danger: "bg-red-500",
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`p-1.5 rounded-md ${statusColors[status]}`}>
            <Icon className="h-3.5 w-3.5" />
          </div>
          <span className="text-sm text-muted-foreground">{label}</span>
        </div>
        <span className={`text-sm font-semibold ${status === "danger" ? "text-red-500" : status === "warning" ? "text-yellow-500" : ""}`}>
          {value.toFixed(1)}%
        </span>
      </div>
      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full ${progressColors[status]} transition-all duration-300`}
          style={{ width: `${Math.min((value / max) * 100, 100)}%` }}
        />
      </div>
    </div>
  );
}

function VolatilityBadge({ level }: { level: "low" | "medium" | "high" | "extreme" }) {
  const config = {
    low: { label: "Low", className: "bg-green-500/20 text-green-700 dark:text-green-400" },
    medium: { label: "Med", className: "bg-yellow-500/20 text-yellow-700 dark:text-yellow-400" },
    high: { label: "High", className: "bg-orange-500/20 text-orange-700 dark:text-orange-400" },
    extreme: { label: "Extreme", className: "bg-red-500/20 text-red-700 dark:text-red-400" },
  };

  return (
    <Badge variant="secondary" className={`text-xs ${config[level].className}`}>
      {config[level].label}
    </Badge>
  );
}

export function RiskDashboard() {
  const riskQuery = useQuery<RiskDashboardData>({
    queryKey: ["/api/risk/dashboard"],
    refetchInterval: 30000,
  });

  if (riskQuery.isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base font-medium">Risk Management</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-48 w-full" />
        </CardContent>
      </Card>
    );
  }

  const data = riskQuery.data;
  const metrics = data?.metrics;
  const volatilityBySymbol = data?.volatilityBySymbol || {};
  const recommendations = data?.recommendations || [];

  const getRiskStatus = (value: number, thresholds: [number, number]): "good" | "warning" | "danger" => {
    if (value < thresholds[0]) return "good";
    if (value < thresholds[1]) return "warning";
    return "danger";
  };

  const getCapacityStatus = (value: number): "good" | "warning" | "danger" => {
    if (value > 60) return "good";
    if (value > 30) return "warning";
    return "danger";
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
        <CardTitle className="text-base font-medium flex items-center gap-2">
          <Shield className="h-4 w-4" />
          Risk Management
        </CardTitle>
        {metrics && (
          <Badge
            variant={metrics.riskCapacity > 60 ? "default" : metrics.riskCapacity > 30 ? "secondary" : "destructive"}
            className="text-xs"
          >
            {metrics.riskCapacity > 60 ? "Healthy" : metrics.riskCapacity > 30 ? "Caution" : "High Risk"}
          </Badge>
        )}
      </CardHeader>
      <CardContent className="space-y-4">
        {metrics && (
          <div className="space-y-3">
            <MetricCard
              label="Risk Capacity"
              value={metrics.riskCapacity}
              max={100}
              icon={Target}
              status={getCapacityStatus(metrics.riskCapacity)}
            />
            <MetricCard
              label="Portfolio Heat"
              value={metrics.portfolioHeatLevel}
              max={100}
              icon={Activity}
              status={getRiskStatus(metrics.portfolioHeatLevel, [50, 75])}
            />
            <MetricCard
              label="Volatility"
              value={metrics.portfolioVolatility}
              max={10}
              icon={TrendingUp}
              status={getRiskStatus(metrics.portfolioVolatility, [3, 5])}
            />
            <MetricCard
              label="Current Drawdown"
              value={metrics.currentDrawdown}
              max={10}
              icon={Percent}
              status={getRiskStatus(metrics.currentDrawdown, [2, 5])}
            />
          </div>
        )}

        {Object.keys(volatilityBySymbol).length > 0 && (
          <div className="pt-2 border-t">
            <p className="text-sm font-medium mb-2">Position Volatility</p>
            <div className="space-y-1.5">
              {Object.entries(volatilityBySymbol).map(([symbol, vol]) => (
                <div
                  key={symbol}
                  className="flex items-center justify-between p-2 rounded-md bg-muted/30"
                  data-testid={`volatility-${symbol}`}
                >
                  <span className="text-sm font-medium">{symbol}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">
                      ATR: {vol.atrPercent.toFixed(1)}%
                    </span>
                    <VolatilityBadge level={vol.volatilityLevel} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {recommendations.length > 0 && (
          <div className="pt-2 border-t">
            <p className="text-sm font-medium mb-2 flex items-center gap-1">
              <AlertTriangle className="h-3.5 w-3.5 text-yellow-500" />
              Recommendations
            </p>
            <div className="space-y-1.5">
              {recommendations.map((rec, i) => (
                <div
                  key={i}
                  className="text-xs text-muted-foreground p-2 rounded-md bg-muted/30"
                >
                  {rec}
                </div>
              ))}
            </div>
          </div>
        )}

        {!metrics && (
          <div className="text-center text-sm text-muted-foreground py-4">
            Risk data unavailable
          </div>
        )}
      </CardContent>
    </Card>
  );
}
