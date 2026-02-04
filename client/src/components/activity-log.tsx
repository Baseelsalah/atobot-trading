import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Activity, TrendingUp, Brain, AlertTriangle, Settings, Zap } from "lucide-react";
import type { ActivityLog as ActivityLogType } from "@shared/schema";

interface ActivityLogProps {
  logs: ActivityLogType[];
  isLoading: boolean;
}

const typeConfig: Record<string, { icon: typeof Activity; variant: "default" | "secondary" | "destructive" | "outline"; className: string }> = {
  trade: { icon: TrendingUp, variant: "outline", className: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30" },
  analysis: { icon: Brain, variant: "outline", className: "bg-blue-500/10 text-blue-400 border-blue-500/30" },
  error: { icon: AlertTriangle, variant: "outline", className: "bg-red-500/10 text-red-400 border-red-500/30" },
  system: { icon: Settings, variant: "outline", className: "bg-muted text-muted-foreground border-border" },
  alert: { icon: Zap, variant: "outline", className: "bg-amber-500/10 text-amber-400 border-amber-500/30" },
  exit: { icon: TrendingUp, variant: "outline", className: "bg-purple-500/10 text-purple-400 border-purple-500/30" },
  skip: { icon: Activity, variant: "outline", className: "bg-slate-500/10 text-slate-400 border-slate-500/30" },
  no_signal: { icon: Activity, variant: "outline", className: "bg-slate-500/10 text-slate-400/70 border-slate-500/20" },
};

function formatTime(timestamp: Date | string | null): string {
  if (!timestamp) return "";
  const date = typeof timestamp === "string" ? new Date(timestamp) : timestamp;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function ActivityLogComponent({ logs, isLoading }: ActivityLogProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg font-semibold flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Activity Log
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-2">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-8 bg-muted rounded-md" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base sm:text-lg font-semibold flex items-center gap-2">
          <Activity className="w-4 h-4 sm:w-5 sm:h-5 flex-shrink-0" />
          <span>Activity Log</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-56 sm:h-72 md:h-96">
          {logs.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Activity className="w-12 h-12 mx-auto mb-2 opacity-50" />
              <p>No activity yet</p>
              <p className="text-sm">Bot activity will appear here</p>
            </div>
          ) : (
            <div className="space-y-1 pr-4">
              {logs.map((log) => {
                const config = typeConfig[log.type] || typeConfig.system;
                const Icon = config.icon;

                return (
                  <div
                    key={log.id}
                    className="flex flex-col sm:flex-row sm:items-center gap-1.5 sm:gap-3 py-2.5 border-b border-border/30 last:border-0"
                    data-testid={`log-entry-${log.id}`}
                  >
                    <div className="flex items-center gap-2 sm:gap-3">
                      <span className="text-[10px] font-mono text-muted-foreground/70 whitespace-nowrap w-[70px]">
                        {formatTime(log.timestamp)}
                      </span>
                      <Badge 
                        variant={config.variant} 
                        className={`text-[10px] px-2 py-0.5 font-medium uppercase tracking-wider ${config.className}`}
                      >
                        <Icon className="w-2.5 h-2.5 mr-1" />
                        {log.action}
                      </Badge>
                    </div>
                    <span className="text-xs sm:text-sm text-foreground/90 flex-1 ml-[78px] sm:ml-0">{log.description}</span>
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
