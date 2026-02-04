import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, ArrowUpRight, ArrowDownRight } from "lucide-react";
import type { Trade } from "@shared/schema";

export default function TradesPage() {
  const { data: trades, isLoading } = useQuery<Trade[]>({
    queryKey: ["/api/trades"],
    refetchInterval: 10000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Trade History</h1>
        <p className="text-muted-foreground">All executed trades</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Recent Trades</CardTitle>
        </CardHeader>
        <CardContent>
          {!trades || trades.length === 0 ? (
            <p className="text-muted-foreground text-center py-8">No trades yet</p>
          ) : (
            <div className="space-y-2">
              {trades.slice(0, 50).map((trade) => (
                <div
                  key={trade.id}
                  className="flex items-center justify-between p-3 border rounded-md"
                  data-testid={`trade-row-${trade.id}`}
                >
                  <div className="flex items-center gap-3">
                    {trade.side === "buy" ? (
                      <ArrowUpRight className="w-4 h-4 text-green-500" />
                    ) : (
                      <ArrowDownRight className="w-4 h-4 text-red-500" />
                    )}
                    <div>
                      <span className="font-medium font-mono">{trade.symbol}</span>
                      <span className="text-muted-foreground text-sm ml-2">
                        {trade.quantity} @ ${Number(trade.price).toFixed(2)}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={trade.status === "filled" ? "default" : "secondary"}>
                      {trade.status}
                    </Badge>
                    <span className="text-xs text-muted-foreground font-mono">
                      {trade.timestamp ? new Date(trade.timestamp).toLocaleTimeString() : "—"}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
