import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Brain, TrendingUp, Newspaper, BarChart3 } from "lucide-react";
import type { ResearchLog } from "@shared/schema";

interface ResearchPanelProps {
  logs: ResearchLog[];
  isLoading: boolean;
}

const typeConfig: Record<string, { icon: typeof Brain; label: string }> = {
  analysis: { icon: Brain, label: "Analysis" },
  news: { icon: Newspaper, label: "News" },
  technical: { icon: BarChart3, label: "Technical" },
  recommendation: { icon: TrendingUp, label: "Recommendation" },
};

function getConfidenceVariant(confidence: number | null): "default" | "secondary" | "destructive" {
  if (!confidence) return "secondary";
  if (confidence >= 75) return "default";
  if (confidence >= 50) return "secondary";
  return "destructive";
}

function formatTime(timestamp: Date | string | null): string {
  if (!timestamp) return "";
  const date = typeof timestamp === "string" ? new Date(timestamp) : timestamp;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function ResearchPanel({ logs, isLoading }: ResearchPanelProps) {
  if (isLoading) {
    return (
      <Card className="h-full">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg font-semibold flex items-center gap-2">
            <Brain className="w-5 h-5" />
            GPT Research
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-24 bg-muted rounded-md" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-2 flex-shrink-0">
        <CardTitle className="text-lg font-semibold flex items-center gap-2">
          <Brain className="w-5 h-5" />
          GPT Research
          <Badge variant="outline" className="ml-auto text-xs">
            {logs.length} entries
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 min-h-0">
        <ScrollArea className="h-72 sm:h-96">
          {logs.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Brain className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No research logs yet</p>
              <p className="text-sm">GPT will analyze markets when the bot is active</p>
            </div>
          ) : (
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
                  <div
                    key={log.id}
                    className="p-3 sm:p-4 bg-muted/50 rounded-md space-y-2"
                    data-testid={`card-research-${log.id}`}
                  >
                    <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-1 sm:gap-2">
                      <div className="flex items-center gap-2">
                        <Icon className="w-4 h-4 text-muted-foreground" />
                        <Badge variant="outline" className="text-xs">
                          {config.label}
                        </Badge>
                        {log.symbol && (
                          <Badge variant="secondary" className="text-xs font-mono">
                            {log.symbol}
                          </Badge>
                        )}
                      </div>
                      <span className="text-xs text-muted-foreground font-mono">
                        {formatTime(log.timestamp)}
                      </span>
                    </div>
                    <p className="text-sm">{log.summary}</p>
                    {log.confidence && (
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">Confidence:</span>
                        <Badge variant={getConfidenceVariant(log.confidence)} className="text-xs">
                          {log.confidence}%
                        </Badge>
                      </div>
                    )}
                    {sources.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {sources.map((source, i) => (
                          <Badge key={i} variant="outline" className="text-xs">
                            {source}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
