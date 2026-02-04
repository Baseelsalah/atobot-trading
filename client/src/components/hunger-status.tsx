import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Flame, Target, Clock, TrendingUp } from "lucide-react";

interface HungerState {
  hungerLevel: "starving" | "hungry" | "fed" | "satisfied" | "full";
  urgency: number;
  aggressiveness: number;
  positionSizeMultiplier: number;
  thresholdReduction: number;
  message: string;
  profitNeeded: number;
  timeRemainingHours: number;
  profitPerHourNeeded: number;
}

function getHungerColor(level: string): string {
  switch (level) {
    case "starving": return "text-red-500";
    case "hungry": return "text-orange-500";
    case "fed": return "text-yellow-500";
    case "satisfied": return "text-green-500";
    case "full": return "text-emerald-500";
    default: return "text-muted-foreground";
  }
}

function getHungerBadgeVariant(level: string): "destructive" | "default" | "secondary" {
  switch (level) {
    case "starving": return "destructive";
    case "hungry": return "destructive";
    case "fed": return "secondary";
    case "satisfied": return "default";
    case "full": return "default";
    default: return "secondary";
  }
}

export function HungerStatus() {
  const hungerQuery = useQuery<HungerState>({
    queryKey: ["/api/hunger"],
    refetchInterval: 5000,
  });

  const hunger = hungerQuery.data;

  if (hungerQuery.isLoading || !hunger) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Flame className="w-4 h-4" />
            Bot Hunger Status
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

  const flameCount = hunger.hungerLevel === "starving" ? 5 : 
                     hunger.hungerLevel === "hungry" ? 4 :
                     hunger.hungerLevel === "fed" ? 2 : 
                     hunger.hungerLevel === "satisfied" ? 1 : 0;

  return (
    <Card data-testid="card-hunger-status" className={hunger.hungerLevel === "starving" ? "border-red-500/50" : hunger.hungerLevel === "hungry" ? "border-orange-500/50" : ""}>
      <CardHeader className="pb-2">
        <CardTitle className="text-base font-medium flex items-center justify-between gap-2">
          <span className="flex items-center gap-2">
            <Flame className={`w-4 h-4 ${getHungerColor(hunger.hungerLevel)}`} />
            Bot Hunger Status
          </span>
          <Badge data-testid="badge-hunger-level" variant={getHungerBadgeVariant(hunger.hungerLevel)}>
            {hunger.hungerLevel.toUpperCase()}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-1">
          {Array.from({ length: 5 }).map((_, i) => (
            <Flame 
              key={i} 
              className={`w-5 h-5 ${i < flameCount ? getHungerColor(hunger.hungerLevel) : "text-muted"}`}
              fill={i < flameCount ? "currentColor" : "none"}
            />
          ))}
        </div>

        <p data-testid="text-hunger-message" className="text-sm font-medium">{hunger.message}</p>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Urgency</span>
            <span className="font-medium">{hunger.urgency}%</span>
          </div>
          <Progress value={hunger.urgency} className="h-2" />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="flex items-center gap-2 p-2 rounded-md bg-muted/30">
            <Target className="w-4 h-4 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">Profit Needed</div>
              <div className="font-semibold text-sm">${hunger.profitNeeded.toFixed(0)}</div>
            </div>
          </div>

          <div className="flex items-center gap-2 p-2 rounded-md bg-muted/30">
            <Clock className="w-4 h-4 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">Time Left</div>
              <div className="font-semibold text-sm">{hunger.timeRemainingHours.toFixed(1)}h</div>
            </div>
          </div>

          <div className="flex items-center gap-2 p-2 rounded-md bg-muted/30">
            <TrendingUp className="w-4 h-4 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">$/Hour Needed</div>
              <div className="font-semibold text-sm">${hunger.profitPerHourNeeded.toFixed(0)}</div>
            </div>
          </div>

          <div className="flex items-center gap-2 p-2 rounded-md bg-muted/30">
            <Flame className="w-4 h-4 text-muted-foreground" />
            <div>
              <div className="text-xs text-muted-foreground">Aggression</div>
              <div className="font-semibold text-sm">{(hunger.aggressiveness * 100).toFixed(0)}%</div>
            </div>
          </div>
        </div>

        {hunger.positionSizeMultiplier > 1 && (
          <div className="text-xs text-center p-2 rounded-md bg-orange-500/10 text-orange-500">
            Position sizes boosted by {((hunger.positionSizeMultiplier - 1) * 100).toFixed(0)}%
          </div>
        )}
      </CardContent>
    </Card>
  );
}
