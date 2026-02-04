import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Target, TrendingUp, TrendingDown, Trophy, AlertTriangle } from "lucide-react";

interface ProfitGoalState {
  dailyGoal: number;
  currentProfit: number;
  realizedProfit: number;
  unrealizedProfit: number;
  progressPercent: number;
  goalMet: boolean;
  remaining: number;
}

interface PerformanceData {
  totalTrades: number;
  winCount: number;
  lossCount: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  expectancy: number;
  profitFactor: number;
  consecutiveLosses: number;
}

function formatCurrency(value: number): string {
  const sign = value >= 0 ? "" : "-";
  return `${sign}$${Math.abs(value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function ProfitGoalCard() {
  const profitGoalQuery = useQuery<ProfitGoalState>({
    queryKey: ["/api/profit-goal"],
    refetchInterval: 3000,
  });

  const performanceQuery = useQuery<PerformanceData>({
    queryKey: ["/api/performance"],
    refetchInterval: 5000,
  });

  const goal = profitGoalQuery.data;
  const perf = performanceQuery.data;

  if (profitGoalQuery.isLoading) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Target className="w-4 h-4" />
            Daily Profit Goal
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-3">
            <div className="h-4 bg-muted rounded w-3/4" />
            <div className="h-2 bg-muted rounded w-full" />
            <div className="h-4 bg-muted rounded w-1/2" />
          </div>
        </CardContent>
      </Card>
    );
  }

  const isPositive = (goal?.currentProfit ?? 0) >= 0;
  const progressValue = Math.max(0, Math.min(100, goal?.progressPercent ?? 0));

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm sm:text-base font-medium flex items-center justify-between gap-2">
          <span className="flex items-center gap-1.5 sm:gap-2">
            <Target className="w-4 h-4 flex-shrink-0" />
            <span>Daily Profit Goal</span>
          </span>
          {goal?.goalMet && (
            <span className="flex items-center gap-1 text-chart-2 text-xs sm:text-sm">
              <Trophy className="w-3 h-3 sm:w-4 sm:h-4" />
              <span className="hidden sm:inline">Goal Met</span>
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 sm:space-y-4">
        <div className="flex items-baseline justify-between gap-2 flex-wrap">
          <span
            className={`text-xl sm:text-2xl font-bold font-mono ${isPositive ? "text-chart-2" : "text-destructive"}`}
            data-testid="text-current-profit"
          >
            {formatCurrency(goal?.currentProfit ?? 0)}
          </span>
          <span className="text-muted-foreground text-xs sm:text-sm">
            / {formatCurrency(goal?.dailyGoal ?? 3000)}
          </span>
        </div>

        <div className="space-y-2">
          <Progress 
            value={progressValue} 
            className="h-2"
            data-testid="progress-profit-goal"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{progressValue.toFixed(1)}% complete</span>
            <span>{formatCurrency(goal?.remaining ?? 3000)} to go</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 pt-2 border-t">
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground">Realized</span>
            <div className={`text-sm font-mono ${(goal?.realizedProfit ?? 0) >= 0 ? "text-chart-2" : "text-destructive"}`}>
              {formatCurrency(goal?.realizedProfit ?? 0)}
            </div>
          </div>
          <div className="space-y-1">
            <span className="text-xs text-muted-foreground">Unrealized</span>
            <div className={`text-sm font-mono ${(goal?.unrealizedProfit ?? 0) >= 0 ? "text-chart-2" : "text-destructive"}`}>
              {formatCurrency(goal?.unrealizedProfit ?? 0)}
            </div>
          </div>
        </div>

        {perf && (
          <div className="space-y-3 pt-2 border-t">
            <div className="text-xs text-muted-foreground font-medium">Performance</div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <span className="text-xs text-muted-foreground">Win Rate</span>
                <div className="flex items-center gap-1 text-sm font-mono">
                  {perf.winRate >= 50 ? (
                    <TrendingUp className="w-3 h-3 text-chart-2" />
                  ) : (
                    <TrendingDown className="w-3 h-3 text-destructive" />
                  )}
                  {perf.winRate.toFixed(1)}%
                </div>
              </div>
              <div className="space-y-1">
                <span className="text-xs text-muted-foreground">Trades</span>
                <div className="text-sm font-mono">
                  {perf.winCount}W / {perf.lossCount}L
                </div>
              </div>
              <div className="space-y-1">
                <span className="text-xs text-muted-foreground">Expectancy</span>
                <div className={`text-sm font-mono ${perf.expectancy >= 0 ? "text-chart-2" : "text-destructive"}`}>
                  {formatCurrency(perf.expectancy)}
                </div>
              </div>
              <div className="space-y-1">
                <span className="text-xs text-muted-foreground">Profit Factor</span>
                <div className="text-sm font-mono">
                  {perf.profitFactor === Infinity ? "N/A" : perf.profitFactor.toFixed(2)}
                </div>
              </div>
            </div>
            {perf.consecutiveLosses >= 2 && (
              <div className="flex items-center gap-2 text-xs text-amber-500 dark:text-amber-400">
                <AlertTriangle className="w-3 h-3" />
                {perf.consecutiveLosses} consecutive losses
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
