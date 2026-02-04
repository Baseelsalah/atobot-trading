import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import type { Alert, TradeRecommendation } from "@shared/schema";

interface AlertModalProps {
  alert: Alert | null;
  onApprove: (alertId: string) => void;
  onDeny: (alertId: string) => void;
  onClose: () => void;
}

function getRiskBadgeVariant(risk: string): "default" | "secondary" | "destructive" {
  switch (risk) {
    case "low":
      return "default";
    case "medium":
      return "secondary";
    case "high":
      return "destructive";
    default:
      return "secondary";
  }
}

export function AlertModal({ alert, onApprove, onDeny, onClose }: AlertModalProps) {
  if (!alert) return null;

  let recommendation: TradeRecommendation | null = null;
  if (alert.metadata) {
    try {
      recommendation = JSON.parse(alert.metadata) as TradeRecommendation;
    } catch {
      recommendation = null;
    }
  }

  return (
    <Dialog open={!!alert} onOpenChange={() => onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-chart-4" />
            {alert.title}
          </DialogTitle>
          <DialogDescription>{alert.message}</DialogDescription>
        </DialogHeader>

        {recommendation && (
          <div className="space-y-4 py-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <span className="text-xs text-muted-foreground uppercase">Symbol</span>
                <p className="text-lg font-semibold font-mono">{recommendation.symbol}</p>
              </div>
              <div>
                <span className="text-xs text-muted-foreground uppercase">Side</span>
                <p className={`text-lg font-semibold ${recommendation.side === "buy" ? "text-chart-2" : "text-destructive"}`}>
                  {recommendation.side.toUpperCase()}
                </p>
              </div>
              <div>
                <span className="text-xs text-muted-foreground uppercase">Quantity</span>
                <p className="text-lg font-mono">{recommendation.quantity}</p>
              </div>
              <div>
                <span className="text-xs text-muted-foreground uppercase">Price</span>
                <p className="text-lg font-mono">${recommendation.price.toFixed(2)}</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground uppercase">Risk Level:</span>
              <Badge variant={getRiskBadgeVariant(recommendation.riskLevel)}>
                {recommendation.riskLevel}
              </Badge>
              <span className="text-xs text-muted-foreground uppercase ml-4">Confidence:</span>
              <Badge variant="outline">{recommendation.confidence}%</Badge>
            </div>

            <div className="p-3 bg-muted/50 rounded-md">
              <span className="text-xs text-muted-foreground uppercase">GPT Reasoning</span>
              <p className="text-sm mt-1">{recommendation.reason}</p>
            </div>
          </div>
        )}

        {alert.requiresApproval && (
          <DialogFooter className="gap-2">
            <Button
              variant="outline"
              onClick={() => onDeny(alert.id)}
              className="flex-1"
              data-testid="button-deny-trade"
            >
              <XCircle className="w-4 h-4 mr-2" />
              Deny
            </Button>
            <Button
              onClick={() => onApprove(alert.id)}
              className="flex-1"
              data-testid="button-approve-trade"
            >
              <CheckCircle className="w-4 h-4 mr-2" />
              Approve
            </Button>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  );
}
