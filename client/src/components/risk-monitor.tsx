import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Shield, AlertTriangle, TrendingDown, Layers } from "lucide-react";
import type { BotSettings, PortfolioSummary } from "@shared/schema";

interface RiskMonitorProps {
  settings: BotSettings | null;
  portfolio: PortfolioSummary | null;
  positionsCount: number;
  isLoading: boolean;
}

export function RiskMonitor({ settings, portfolio, positionsCount, isLoading }: RiskMonitorProps) {
  if (isLoading || !settings) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg font-semibold flex items-center gap-2">
            <Shield className="w-5 h-5" />
            Risk Monitor
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="space-y-2">
                <div className="h-4 bg-muted rounded w-1/3" />
                <div className="h-2 bg-muted rounded" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  const dailyLossUsed = portfolio?.todayPL && portfolio.todayPL < 0 
    ? Math.abs(portfolio.todayPL) 
    : 0;
  const dailyLossPercent = Math.min((dailyLossUsed / settings.maxDailyLoss) * 100, 100);
  const positionPercent = (positionsCount / settings.maxPositions) * 100;

  const getDailyLossVariant = () => {
    if (dailyLossPercent >= 90) return "destructive";
    if (dailyLossPercent >= 70) return "secondary";
    return "default";
  };

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-semibold flex items-center gap-2">
          <Shield className="w-5 h-5" />
          Risk Monitor
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-sm">
              <TrendingDown className="w-4 h-4 text-muted-foreground" />
              <span>Daily Loss Limit</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-mono">
                ${dailyLossUsed.toFixed(2)} / ${settings.maxDailyLoss.toFixed(2)}
              </span>
              {dailyLossPercent >= 70 && (
                <AlertTriangle className="w-4 h-4 text-destructive" />
              )}
            </div>
          </div>
          <Progress value={dailyLossPercent} className="h-2" />
          {dailyLossPercent >= 90 && (
            <Badge variant="destructive" className="text-xs">
              Near daily loss limit - trading may be paused
            </Badge>
          )}
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-sm">
              <Layers className="w-4 h-4 text-muted-foreground" />
              <span>Open Positions</span>
            </div>
            <span className="text-sm font-mono">
              {positionsCount} / {settings.maxPositions}
            </span>
          </div>
          <Progress value={positionPercent} className="h-2" />
        </div>

        <div className="grid grid-cols-2 gap-4 pt-2">
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground uppercase tracking-wide">Stop Loss</span>
            <p className="text-lg font-mono font-semibold">{settings.stopLossPercent}%</p>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground uppercase tracking-wide">Take Profit</span>
            <p className="text-lg font-mono font-semibold">{settings.takeProfitPercent}%</p>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground uppercase tracking-wide">Max Position</span>
            <p className="text-lg font-mono font-semibold">${settings.maxPositionSize}</p>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground uppercase tracking-wide">Mode</span>
            <Badge variant={settings.isPaperTrading ? "secondary" : "default"} className="text-xs">
              {settings.isPaperTrading ? "Paper" : "Live"}
            </Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
