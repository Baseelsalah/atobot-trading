import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Brain, TrendingUp, Newspaper, BarChart3 } from "lucide-react";
import { BrainStatus } from "@/components/brain-status";
import type { ResearchLog } from "@shared/schema";

const typeConfig: Record<string, { icon: typeof Brain; label: string; color: string }> = {
  analysis: { icon: Brain, label: "Analysis", color: "text-chart-1" },
  news: { icon: Newspaper, label: "News", color: "text-chart-4" },
  technical: { icon: BarChart3, label: "Technical", color: "text-chart-3" },
  recommendation: { icon: TrendingUp, label: "Recommendation", color: "text-chart-2" },
};

function formatDateTime(timestamp: Date | string | null): string {
  if (!timestamp) return "";
  const date = typeof timestamp === "string" ? new Date(timestamp) : timestamp;
  return date.toLocaleString();
}

function getConfidenceVariant(confidence: number | null): "default" | "secondary" | "destructive" {
  if (!confidence) return "secondary";
  if (confidence >= 75) return "default";
  if (confidence >= 50) return "secondary";
  return "destructive";
}

export default function ResearchPage() {
  const researchQuery = useQuery<ResearchLog[]>({
    queryKey: ["/api/research"],
    refetchInterval: 10000,
  });

  const logs = researchQuery.data ?? [];

  const analysisCount = logs.filter((l) => l.type === "analysis").length;
  const newsCount = logs.filter((l) => l.type === "news").length;
  const technicalCount = logs.filter((l) => l.type === "technical").length;
  const recommendationCount = logs.filter((l) => l.type === "recommendation").length;

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-semibold flex items-center gap-2">
          <Brain className="w-6 h-6" />
          GPT Research
        </h1>
        <p className="text-muted-foreground">AI-powered market analysis and insights</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground mb-1">
                  <Brain className="w-4 h-4" />
                  Analysis
                </div>
                <div className="text-2xl font-bold font-mono">{analysisCount}</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground mb-1">
                  <Newspaper className="w-4 h-4" />
                  News
                </div>
                <div className="text-2xl font-bold font-mono">{newsCount}</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground mb-1">
                  <BarChart3 className="w-4 h-4" />
                  Technical
                </div>
                <div className="text-2xl font-bold font-mono">{technicalCount}</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground mb-1">
                  <TrendingUp className="w-4 h-4" />
                  Recommendations
                </div>
                <div className="text-2xl font-bold font-mono">{recommendationCount}</div>
              </CardContent>
            </Card>
          </div>
        </div>
        <div>
          <BrainStatus />
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">Research Log</CardTitle>
        </CardHeader>
        <CardContent>
          {researchQuery.isLoading ? (
            <div className="animate-pulse space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-32 bg-muted rounded-md" />
              ))}
            </div>
          ) : logs.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Brain className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg">No research entries yet</p>
              <p className="text-sm">GPT will analyze markets when the bot is active</p>
            </div>
          ) : (
            <ScrollArea className="h-[600px]">
              <div className="space-y-4 pr-4">
                {logs.map((log) => {
                  const config = typeConfig[log.type] || typeConfig.analysis;
                  const Icon = config.icon;
                  let sources: string[] = [];
                  try {
                    sources = log.sources ? JSON.parse(log.sources) : [];
                  } catch {
                    sources = [];
                  }

                  return (
                    <Card key={log.id} data-testid={`card-research-${log.id}`}>
                      <CardContent className="pt-6 space-y-3">
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex items-center gap-2 flex-wrap">
                            <Icon className={`w-5 h-5 ${config.color}`} />
                            <Badge variant="outline">{config.label}</Badge>
                            {log.symbol && (
                              <Badge variant="secondary" className="font-mono">
                                {log.symbol}
                              </Badge>
                            )}
                            {log.confidence && (
                              <Badge variant={getConfidenceVariant(log.confidence)}>
                                {log.confidence}% confidence
                              </Badge>
                            )}
                          </div>
                          <span className="text-xs text-muted-foreground font-mono whitespace-nowrap">
                            {formatDateTime(log.timestamp)}
                          </span>
                        </div>

                        <p className="text-sm font-medium">{log.summary}</p>

                        {log.details && (
                          <p className="text-sm text-muted-foreground">{log.details}</p>
                        )}

                        {sources.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {sources.map((source, i) => (
                              <Badge key={i} variant="outline" className="text-xs">
                                {source}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
