import { useState } from "react";
import { useMutation, useQueryClient, useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Power, Loader2, X } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

interface Position {
  symbol: string;
  quantity: number;
  marketValue: number;
  unrealizedPL: number;
}

export function KillSwitch() {
  const queryClient = useQueryClient();
  const [dialogOpen, setDialogOpen] = useState(false);

  const { data: positions = [] } = useQuery<Position[]>({
    queryKey: ["/api/positions"],
    refetchInterval: 5000,
  });

  const closeAllMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("POST", "/api/positions/close-all");
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/positions"] });
      queryClient.invalidateQueries({ queryKey: ["/api/portfolio"] });
      queryClient.invalidateQueries({ queryKey: ["/api/trades"] });
      setDialogOpen(false);
    },
  });

  const openPositionCount = positions.length;
  const totalValue = positions.reduce((sum, p) => sum + p.marketValue, 0);
  const totalPL = positions.reduce((sum, p) => sum + p.unrealizedPL, 0);

  if (openPositionCount === 0) {
    return (
      <Card className="border-green-500/30 bg-green-500/5">
        <CardContent className="p-4">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Power className="w-5 h-5 text-green-500" />
              <span className="font-medium text-green-600 dark:text-green-400">All Clear</span>
            </div>
            <Badge variant="outline" className="text-green-600 border-green-500/50">
              No Open Positions
            </Badge>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-red-500/50 bg-red-500/5">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between gap-2 text-base">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <span className="text-red-600 dark:text-red-400">Kill Switch</span>
          </div>
          <Badge variant="destructive" data-testid="badge-open-positions">
            {openPositionCount} Open
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <div className="text-muted-foreground">Total Value</div>
            <div className="font-medium" data-testid="text-total-value">
              ${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </div>
          <div>
            <div className="text-muted-foreground">Unrealized P/L</div>
            <div 
              className={`font-medium ${totalPL >= 0 ? 'text-green-600' : 'text-red-600'}`}
              data-testid="text-total-pl"
            >
              {totalPL >= 0 ? '+' : ''}{totalPL.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </div>
        </div>

        <div className="text-xs text-muted-foreground space-y-1">
          {positions.slice(0, 5).map((pos) => (
            <div key={pos.symbol} className="flex justify-between">
              <span>{pos.symbol}</span>
              <span className={pos.unrealizedPL >= 0 ? 'text-green-600' : 'text-red-600'}>
                {pos.unrealizedPL >= 0 ? '+' : ''}${pos.unrealizedPL.toFixed(2)}
              </span>
            </div>
          ))}
          {positions.length > 5 && (
            <div className="text-center">... and {positions.length - 5} more</div>
          )}
        </div>

        <AlertDialog open={dialogOpen} onOpenChange={setDialogOpen}>
          <AlertDialogTrigger asChild>
            <Button 
              variant="destructive" 
              className="w-full gap-2"
              data-testid="button-kill-switch"
            >
              <X className="w-4 h-4" />
              CLOSE ALL POSITIONS
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-red-500" />
                Confirm Kill Switch
              </AlertDialogTitle>
              <AlertDialogDescription className="space-y-2">
                <p>This will immediately close ALL {openPositionCount} open positions:</p>
                <ul className="text-sm space-y-1 mt-2">
                  {positions.map((pos) => (
                    <li key={pos.symbol} className="flex justify-between">
                      <span className="font-medium">{pos.symbol}</span>
                      <span>{pos.quantity} shares @ ${(pos.marketValue / pos.quantity).toFixed(2)}</span>
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-red-600 dark:text-red-400 font-medium">
                  Total P/L to be realized: {totalPL >= 0 ? '+' : ''}${totalPL.toFixed(2)}
                </p>
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel data-testid="button-cancel-kill">Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={() => closeAllMutation.mutate()}
                className="bg-red-600 hover:bg-red-700"
                disabled={closeAllMutation.isPending}
                data-testid="button-confirm-kill"
              >
                {closeAllMutation.isPending ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Closing...
                  </>
                ) : (
                  <>
                    <X className="w-4 h-4 mr-2" />
                    CONFIRM CLOSE ALL
                  </>
                )}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </CardContent>
    </Card>
  );
}
