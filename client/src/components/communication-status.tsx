import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Brain, Bot, ArrowLeftRight, Lightbulb, Signal, MessageSquare } from "lucide-react";

interface CommunicationSummary {
  autopilotActive: boolean;
  atoActive: boolean;
  pendingSignals: number;
  recentFeedbackCount: number;
  learningInsightsCount: number;
  strategiesTracked: number;
  lastAutopilotUpdate: string | null;
  lastAtoUpdate: string | null;
  topStrategies: { name: string; successRate: number; trades: number }[];
}

interface LearningInsight {
  id: string;
  timestamp: string;
  strategy: string;
  symbol: string;
  wasSuccessful: boolean;
  profitLoss: number;
  lesson: string;
}

function formatTimeAgo(dateString: string | null): string {
  if (!dateString) return "Never";
  const date = new Date(dateString);
  const now = new Date();
  const seconds = Math.floor((now.getTime() - date.getTime()) / 1000);
  
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

export function CommunicationStatus() {
  const commQuery = useQuery<CommunicationSummary>({
    queryKey: ["/api/communication"],
    refetchInterval: 5000,
  });

  const insightsQuery = useQuery<LearningInsight[]>({
    queryKey: ["/api/learning-insights"],
    refetchInterval: 10000,
  });

  const summary = commQuery.data;
  const insights = insightsQuery.data || [];
  const recentInsights = insights.slice(-5).reverse();

  if (commQuery.isLoading) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <ArrowLeftRight className="w-4 h-4" />
            Autopilot - Ato Communication
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-3">
            <div className="h-4 bg-muted rounded w-3/4" />
            <div className="h-4 bg-muted rounded w-1/2" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base font-medium flex items-center justify-between gap-2">
          <span className="flex items-center gap-2">
            <ArrowLeftRight className="w-4 h-4" />
            Autopilot - Ato Communication
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div className="flex items-center gap-2 p-2 rounded-md bg-muted/30">
            <Brain className="w-4 h-4 text-purple-500" />
            <div className="flex-1 min-w-0">
              <div className="text-xs text-muted-foreground">Autopilot</div>
              <div className="flex items-center gap-2">
                <Badge 
                  variant={summary?.autopilotActive ? "default" : "secondary"}
                  className="text-xs"
                >
                  {summary?.autopilotActive ? "Active" : "Idle"}
                </Badge>
                <span className="text-xs text-muted-foreground truncate">
                  {formatTimeAgo(summary?.lastAutopilotUpdate || null)}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 p-2 rounded-md bg-muted/30">
            <Bot className="w-4 h-4 text-blue-500" />
            <div className="flex-1 min-w-0">
              <div className="text-xs text-muted-foreground">Ato</div>
              <div className="flex items-center gap-2">
                <Badge 
                  variant={summary?.atoActive ? "default" : "secondary"}
                  className="text-xs"
                >
                  {summary?.atoActive ? "Active" : "Idle"}
                </Badge>
                <span className="text-xs text-muted-foreground truncate">
                  {formatTimeAgo(summary?.lastAtoUpdate || null)}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="p-2 rounded-md bg-muted/30">
            <Signal className="w-4 h-4 mx-auto text-muted-foreground" />
            <div className="text-lg font-semibold">{summary?.pendingSignals || 0}</div>
            <div className="text-xs text-muted-foreground">Signals</div>
          </div>
          <div className="p-2 rounded-md bg-muted/30">
            <MessageSquare className="w-4 h-4 mx-auto text-muted-foreground" />
            <div className="text-lg font-semibold">{summary?.recentFeedbackCount || 0}</div>
            <div className="text-xs text-muted-foreground">Feedback</div>
          </div>
          <div className="p-2 rounded-md bg-muted/30">
            <Lightbulb className="w-4 h-4 mx-auto text-muted-foreground" />
            <div className="text-lg font-semibold">{summary?.learningInsightsCount || 0}</div>
            <div className="text-xs text-muted-foreground">Insights</div>
          </div>
        </div>

        {summary?.topStrategies && summary.topStrategies.length > 0 && (
          <div className="space-y-2 pt-2 border-t">
            <div className="text-xs text-muted-foreground font-medium">Top Strategies</div>
            {summary.topStrategies.slice(0, 3).map((strat, idx) => (
              <div key={idx} className="flex items-center justify-between text-sm">
                <span className="truncate">{strat.name}</span>
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">
                    {strat.successRate.toFixed(0)}%
                  </Badge>
                  <span className="text-xs text-muted-foreground">{strat.trades} trades</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {recentInsights.length > 0 && (
          <div className="space-y-2 pt-2 border-t">
            <div className="text-xs text-muted-foreground font-medium">Recent Learning</div>
            <div className="space-y-1 max-h-32 overflow-y-auto">
              {recentInsights.map((insight) => (
                <div 
                  key={insight.id} 
                  className={`text-xs p-2 rounded-md ${insight.wasSuccessful ? "bg-green-500/10" : "bg-red-500/10"}`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium">{insight.symbol}</span>
                    <Badge 
                      variant={insight.wasSuccessful ? "default" : "destructive"}
                      className="text-xs"
                    >
                      {insight.profitLoss >= 0 ? "+" : ""}${insight.profitLoss.toFixed(2)}
                    </Badge>
                  </div>
                  <p className="text-muted-foreground truncate">{insight.lesson}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
