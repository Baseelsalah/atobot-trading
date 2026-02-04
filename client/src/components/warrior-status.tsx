import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Sword, Target, Shield, Trophy, Flame, Zap } from "lucide-react";

interface WarriorState {
  mode: "hunt" | "attack" | "defend" | "celebrate" | "regroup";
  killCount: number;
  conquestStreak: number;
  biggestKill: number;
  missionActive: boolean;
  currentMission: string | null;
  warCry: string;
  momentumScore: number;
  battleReadiness: number;
  coolingOff: boolean;
  hungerLevel: string;
  profitNeeded: number;
  timeRemaining: number;
}

export function WarriorStatus() {
  const { data: warrior, isLoading } = useQuery<WarriorState>({
    queryKey: ["/api/warrior"],
    refetchInterval: 3000,
  });

  if (isLoading || !warrior) {
    return (
      <Card data-testid="card-warrior-status">
        <CardHeader className="pb-2">
          <CardTitle className="text-base font-medium flex items-center gap-2">
            <Sword className="w-4 h-4" />
            Warrior Command
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-2">
            <div className="h-4 bg-muted rounded w-3/4"></div>
            <div className="h-4 bg-muted rounded w-1/2"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getModeIcon = (mode: string) => {
    switch (mode) {
      case "hunt": return <Target className="w-4 h-4" />;
      case "attack": return <Zap className="w-4 h-4" />;
      case "defend": return <Shield className="w-4 h-4" />;
      case "celebrate": return <Trophy className="w-4 h-4" />;
      case "regroup": return <Shield className="w-4 h-4" />;
      default: return <Sword className="w-4 h-4" />;
    }
  };

  const getModeColor = (mode: string) => {
    switch (mode) {
      case "hunt": return "text-orange-500";
      case "attack": return "text-red-500";
      case "defend": return "text-blue-500";
      case "celebrate": return "text-green-500";
      case "regroup": return "text-yellow-500";
      default: return "text-muted-foreground";
    }
  };

  const getModeBadge = (mode: string) => {
    switch (mode) {
      case "attack": return "destructive";
      case "hunt": return "default";
      case "celebrate": return "default";
      default: return "secondary";
    }
  };

  return (
    <Card 
      data-testid="card-warrior-status"
      className={warrior.mode === "attack" ? "border-red-500/50" : 
                 warrior.conquestStreak >= 3 ? "border-orange-500/50" : ""}
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-base font-medium flex items-center justify-between gap-2">
          <span className="flex items-center gap-2">
            <Sword className={`w-4 h-4 ${getModeColor(warrior.mode)}`} />
            Warrior Command
          </span>
          <Badge data-testid="badge-warrior-mode" variant={getModeBadge(warrior.mode) as any}>
            {getModeIcon(warrior.mode)}
            <span className="ml-1">{warrior.mode.toUpperCase()}</span>
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div data-testid="text-war-cry" className="text-sm font-bold text-center p-2 bg-muted/50 rounded-md">
          {warrior.warCry}
        </div>

        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="p-2 bg-muted/30 rounded-md">
            <div data-testid="text-kill-count" className="text-lg font-bold text-green-500">
              {warrior.killCount}
            </div>
            <div className="text-xs text-muted-foreground">Kills</div>
          </div>
          <div className="p-2 bg-muted/30 rounded-md">
            <div data-testid="text-conquest-streak" className="text-lg font-bold text-orange-500">
              {warrior.conquestStreak}
            </div>
            <div className="text-xs text-muted-foreground">Streak</div>
          </div>
          <div className="p-2 bg-muted/30 rounded-md">
            <div data-testid="text-biggest-kill" className="text-lg font-bold">
              ${warrior.biggestKill.toFixed(0)}
            </div>
            <div className="text-xs text-muted-foreground">Biggest</div>
          </div>
        </div>

        <div className="space-y-3">
          <div className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground flex items-center gap-1">
                <Flame className="w-3 h-3" /> Momentum
              </span>
              <span data-testid="text-momentum" className="font-medium">{warrior.momentumScore}%</span>
            </div>
            <Progress value={warrior.momentumScore} className="h-2" />
          </div>
          <div className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground flex items-center gap-1">
                <Shield className="w-3 h-3" /> Battle Ready
              </span>
              <span data-testid="text-battle-readiness" className="font-medium">{warrior.battleReadiness}%</span>
            </div>
            <Progress value={warrior.battleReadiness} className="h-2" />
          </div>
        </div>

        {warrior.missionActive && warrior.currentMission && (
          <div className="p-2 bg-orange-500/10 border border-orange-500/30 rounded-md">
            <div className="text-xs text-orange-500 font-medium">ACTIVE MISSION</div>
            <div data-testid="text-current-mission" className="text-sm font-medium">
              {warrior.currentMission}
            </div>
          </div>
        )}

        <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
          <span>Target: $3,000</span>
          <span>Need: ${warrior.profitNeeded.toFixed(0)}</span>
          <span>{warrior.timeRemaining.toFixed(1)}h left</span>
        </div>
      </CardContent>
    </Card>
  );
}
