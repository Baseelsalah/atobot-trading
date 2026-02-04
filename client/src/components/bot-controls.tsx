import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Play, Pause, StopCircle, RefreshCw } from "lucide-react";
import type { BotStatus, BotSettings, MarketStatusV2 } from "@shared/schema";

interface BotControlsProps {
  botStatus: BotStatus;
  settings: BotSettings | null;
  marketStatus: MarketStatusV2 | null;
  onStart: () => void;
  onPause: () => void;
  onStop: () => void;
  onAnalyze: () => void;
  onTogglePaperTrading: (isPaper: boolean) => void;
  isLoading: boolean;
}

function getStatusColor(status: BotStatus["status"]): string {
  switch (status) {
    case "active":
      return "bg-chart-2";
    case "analyzing":
      return "bg-chart-4";
    case "paused":
      return "bg-chart-1";
    case "error":
      return "bg-destructive";
    default:
      return "bg-muted";
  }
}

export function BotControls({
  botStatus,
  settings,
  marketStatus,
  onStart,
  onPause,
  onStop,
  onAnalyze,
  onTogglePaperTrading,
  isLoading,
}: BotControlsProps) {
  const isActive = botStatus.status === "active" || botStatus.status === "analyzing";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base sm:text-lg font-semibold flex items-center justify-between gap-2">
          <span>Bot Controls</span>
          <div className="flex items-center gap-1.5 sm:gap-2">
            <div className={`w-2 h-2 rounded-full ${getStatusColor(botStatus.status)} animate-pulse`} />
            <span className="text-xs sm:text-sm font-normal capitalize">{botStatus.status}</span>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 sm:space-y-4">
        {marketStatus && (
          <div className="flex items-center justify-between gap-2 p-2.5 sm:p-3 bg-muted/50 rounded-md">
            <span className="text-xs sm:text-sm">Market Status</span>
            <div className="flex items-center gap-1.5 sm:gap-2">
              {marketStatus.simulated && (
                <Badge variant="outline" className="text-[10px] sm:text-xs">SIM</Badge>
              )}
              <Badge 
                variant={marketStatus.is_open ? "default" : "secondary"}
                className="text-[10px] sm:text-xs"
                data-testid="badge-market-status"
              >
                {marketStatus.is_open ? "Open" : "Closed"}
              </Badge>
            </div>
          </div>
        )}

        <div className="flex items-center justify-between gap-2">
          <Label htmlFor="paper-trading" className="text-xs sm:text-sm">Paper Trading Mode</Label>
          <Switch
            id="paper-trading"
            checked={settings?.isPaperTrading ?? true}
            onCheckedChange={onTogglePaperTrading}
            disabled={isLoading || isActive}
            data-testid="switch-paper-trading"
          />
        </div>

        {settings?.isPaperTrading && (
          <Badge variant="secondary" className="w-full justify-center text-[10px] sm:text-xs py-1.5">
            Using paper trading - no real money at risk
          </Badge>
        )}

        <div className="grid grid-cols-2 gap-2">
          {!isActive ? (
            <Button
              onClick={onStart}
              disabled={isLoading}
              className="col-span-2"
              data-testid="button-start-bot"
            >
              <Play className="w-4 h-4 mr-1.5 sm:mr-2" />
              <span className="text-sm">Start Bot</span>
            </Button>
          ) : (
            <>
              <Button
                variant="secondary"
                onClick={onPause}
                disabled={isLoading || botStatus.status === "paused"}
                data-testid="button-pause-bot"
              >
                <Pause className="w-4 h-4 mr-1.5 sm:mr-2" />
                <span className="text-sm">Pause</span>
              </Button>
              <Button
                variant="destructive"
                onClick={onStop}
                disabled={isLoading}
                data-testid="button-stop-bot"
              >
                <StopCircle className="w-4 h-4 mr-1.5 sm:mr-2" />
                <span className="text-sm">Stop</span>
              </Button>
            </>
          )}
        </div>

        <Button
          variant="outline"
          onClick={onAnalyze}
          disabled={isLoading || botStatus.status === "analyzing"}
          className="w-full"
          data-testid="button-analyze"
        >
          <RefreshCw className={`w-4 h-4 mr-1.5 sm:mr-2 ${botStatus.status === "analyzing" ? "animate-spin" : ""}`} />
          <span className="text-sm">Run Analysis Now</span>
        </Button>

        {botStatus.errorMessage && (
          <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-md">
            <p className="text-sm text-destructive">{botStatus.errorMessage}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
