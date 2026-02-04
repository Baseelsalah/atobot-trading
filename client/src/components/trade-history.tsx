import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { History, ArrowUpRight, ArrowDownRight, Clock, DollarSign, Filter } from "lucide-react";
import { format } from "date-fns";
import { useState } from "react";
import type { Trade } from "@shared/schema";

type FilterType = "all" | "buy" | "sell";

export function TradeHistory() {
  const [filter, setFilter] = useState<FilterType>("all");

  const tradesQuery = useQuery<Trade[]>({
    queryKey: ["/api/trades"],
    refetchInterval: 10000,
  });

  const trades = tradesQuery.data || [];
  const filteredTrades = filter === "all" 
    ? trades 
    : trades.filter(t => t.side === filter);

  const sortedTrades = [...filteredTrades].sort(
    (a, b) => new Date(b.timestamp || new Date()).getTime() - new Date(a.timestamp || new Date()).getTime()
  );

  if (tradesQuery.isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base font-medium">Trade History</CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-64 w-full" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between gap-2 pb-2">
        <CardTitle className="text-base font-medium flex items-center gap-2">
          <History className="h-4 w-4" />
          Trade History
        </CardTitle>
        <div className="flex items-center gap-1">
          <Button
            variant={filter === "all" ? "default" : "ghost"}
            size="sm"
            className="h-7 px-2 text-xs"
            onClick={() => setFilter("all")}
            data-testid="filter-all"
          >
            All
          </Button>
          <Button
            variant={filter === "buy" ? "default" : "ghost"}
            size="sm"
            className="h-7 px-2 text-xs"
            onClick={() => setFilter("buy")}
            data-testid="filter-buy"
          >
            Buys
          </Button>
          <Button
            variant={filter === "sell" ? "default" : "ghost"}
            size="sm"
            className="h-7 px-2 text-xs"
            onClick={() => setFilter("sell")}
            data-testid="filter-sell"
          >
            Sells
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {sortedTrades.length === 0 ? (
          <div className="h-64 flex flex-col items-center justify-center text-sm text-muted-foreground gap-2">
            <History className="h-8 w-8 opacity-50" />
            <p>No trades yet</p>
            <p className="text-xs">Trades will appear here once executed</p>
          </div>
        ) : (
          <ScrollArea className="h-80">
            <div className="space-y-2 pr-4">
              {sortedTrades.map((trade) => (
                <div
                  key={trade.id}
                  className="flex items-center gap-3 p-3 rounded-md bg-muted/30 hover-elevate"
                  data-testid={`trade-row-${trade.id}`}
                >
                  <div className={`p-2 rounded-md ${trade.side === "buy" ? "bg-green-500/10" : "bg-red-500/10"}`}>
                    {trade.side === "buy" ? (
                      <ArrowUpRight className="h-4 w-4 text-green-500" />
                    ) : (
                      <ArrowDownRight className="h-4 w-4 text-red-500" />
                    )}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{trade.symbol}</span>
                      <Badge 
                        variant={trade.side === "buy" ? "default" : "secondary"}
                        className={`text-xs ${trade.side === "buy" ? "bg-green-500/20 text-green-700 dark:text-green-400" : "bg-red-500/20 text-red-700 dark:text-red-400"}`}
                      >
                        {trade.side.toUpperCase()}
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {trade.status}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground mt-1">
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {format(new Date(trade.timestamp || new Date()), "MMM d, h:mm a")}
                      </span>
                      {trade.reason && (
                        <span className="truncate max-w-[200px]" title={trade.reason}>
                          {trade.reason}
                        </span>
                      )}
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="font-medium">
                      {trade.quantity} @ ${trade.price.toFixed(2)}
                    </div>
                    <div className="text-xs text-muted-foreground flex items-center justify-end gap-1">
                      <DollarSign className="h-3 w-3" />
                      {trade.totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        )}
        
        {sortedTrades.length > 0 && (
          <div className="mt-3 pt-3 border-t flex items-center justify-between text-xs text-muted-foreground">
            <span>Showing {sortedTrades.length} trades</span>
            <span>
              Total Volume: ${sortedTrades.reduce((sum, t) => sum + t.totalValue, 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
