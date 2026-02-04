import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Brain, Zap, TrendingUp, Target, Activity, User, BarChart2 } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";

interface BrainStatusData {
  strategies: number;
  activeStrategies: number;
  learningCycles: number;
  topStrategy: string | null;
  overallConfidence: number;
}

interface AtoState {
  mood: "aggressive" | "conservative" | "neutral";
  currentFocus: string[];
  dailyStats: {
    tradesExecuted: number;
    profitLoss: number;
    winCount: number;
    lossCount: number;
    bestTrade: string | null;
    worstTrade: string | null;
  };
  tradingStyle: {
    preferredSetups: string[];
    avoidPatterns: string[];
  };
}

interface TradingStrategy {
  id: string;
  name: string;
  description: string;
  type: string;
  confidence: number;
  winRate: number;
  totalTrades: number;
  totalProfit: number;
  isActive: boolean;
}

export function BrainStatus() {
  const brainQuery = useQuery<BrainStatusData>({
    queryKey: ["/api/brain/status"],
    refetchInterval: 10000,
  });

  const atoQuery = useQuery<AtoState>({
    queryKey: ["/api/ato/status"],
    refetchInterval: 5000,
  });

  const strategiesQuery = useQuery<TradingStrategy[]>({
    queryKey: ["/api/brain/strategies"],
    refetchInterval: 10000,
  });

  const brain = brainQuery.data;
  const atoState = atoQuery.data;
  const strategies = strategiesQuery.data ?? [];

  if (brainQuery.isLoading) {
    return (
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Brain className="h-4 w-4" />
            System Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    );
  }

  const moodColors = {
    aggressive: "text-red-500",
    conservative: "text-blue-500",
    neutral: "text-gray-500",
  };

  const moodBadgeVariant = {
    aggressive: "destructive" as const,
    conservative: "secondary" as const,
    neutral: "outline" as const,
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
        <CardTitle className="text-base font-medium flex items-center gap-2">
          <Brain className="h-4 w-4 text-purple-500" />
          Autopilot + Ato
        </CardTitle>
        <Badge variant="outline" className="text-xs">
          <Activity className="h-3 w-3 mr-1" />
          Cycle {brain?.learningCycles ?? 0}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div className="p-2 rounded-md bg-purple-500/10">
            <div className="flex items-center gap-1 mb-1">
              <Brain className="h-3 w-3 text-purple-500" />
              <span className="text-xs font-medium text-purple-500">AUTOPILOT</span>
            </div>
            <div className="text-sm text-muted-foreground">The Brain</div>
            <div className="text-lg font-semibold" data-testid="text-active-strategies">
              {brain?.activeStrategies ?? 0} strategies
            </div>
            <div className="text-xs text-muted-foreground">
              {brain?.overallConfidence ?? 0}% confidence
            </div>
          </div>
          <div className="p-2 rounded-md bg-blue-500/10">
            <div className="flex items-center gap-1 mb-1">
              <User className="h-3 w-3 text-blue-500" />
              <span className="text-xs font-medium text-blue-500">ATO</span>
            </div>
            <div className="text-sm text-muted-foreground">The Trader</div>
            <div className="flex items-center gap-2">
              <Badge 
                variant={moodBadgeVariant[atoState?.mood ?? "neutral"]}
                className="text-xs"
              >
                {atoState?.mood ?? "neutral"}
              </Badge>
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {atoState?.dailyStats?.tradesExecuted ?? 0} trades today
            </div>
          </div>
        </div>

        {atoState?.dailyStats && (
          <div className="flex items-center gap-2 p-2 rounded-md bg-muted/50">
            <BarChart2 className="h-4 w-4 text-chart-1" />
            <div className="flex-1">
              <span className="text-xs text-muted-foreground">Today's P/L</span>
              <p className={`text-sm font-mono font-medium ${atoState.dailyStats.profitLoss >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {atoState.dailyStats.profitLoss >= 0 ? '+' : ''}${atoState.dailyStats.profitLoss.toFixed(2)}
              </p>
            </div>
            <div className="text-right text-xs">
              <span className="text-green-500">{atoState.dailyStats.winCount}W</span>
              <span className="text-muted-foreground"> / </span>
              <span className="text-red-500">{atoState.dailyStats.lossCount}L</span>
            </div>
          </div>
        )}

        {brain?.topStrategy && (
          <div className="flex items-center gap-2 p-2 rounded-md bg-muted/50">
            <Target className="h-4 w-4 text-green-500" />
            <div className="flex-1">
              <span className="text-xs text-muted-foreground">Top Strategy</span>
              <p className="text-sm font-medium" data-testid="text-top-strategy">{brain.topStrategy}</p>
            </div>
          </div>
        )}

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-yellow-500" />
            <span className="text-sm font-medium">Active Strategies</span>
          </div>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {strategies.filter(s => s.isActive).map((strategy) => (
              <div
                key={strategy.id}
                className="flex items-center justify-between p-2 rounded-md bg-muted/30"
                data-testid={`strategy-card-${strategy.id}`}
              >
                <div className="flex-1 min-w-0">
                  <span className="text-sm font-medium truncate">{strategy.name}</span>
                  <div className="text-xs text-muted-foreground">{strategy.type}</div>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <div className="text-right">
                    <div className="font-medium">{(strategy.winRate ?? 0).toFixed(0)}%</div>
                    <div className="text-muted-foreground">win</div>
                  </div>
                </div>
              </div>
            ))}
            {strategies.filter(s => s.isActive).length === 0 && (
              <p className="text-sm text-muted-foreground text-center py-2">
                No active strategies
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <TrendingUp className="h-3 w-3" />
          <span>Autopilot guides Ato based on research and performance</span>
        </div>
      </CardContent>
    </Card>
  );
}
