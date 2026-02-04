import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";
import { History, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { TradeHistory } from "@/components/trade-history";
import { KillSwitch } from "@/components/kill-switch";
import type { Trade } from "@shared/schema";

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(value);
}

function formatDateTime(timestamp: Date | string | null): string {
  if (!timestamp) return "";
  const date = typeof timestamp === "string" ? new Date(timestamp) : timestamp;
  return date.toLocaleString();
}

function getStatusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "filled":
      return "default";
    case "pending":
      return "outline";
    case "cancelled":
      return "secondary";
    case "rejected":
      return "destructive";
    default:
      return "secondary";
  }
}

export default function HistoryPage() {
  const tradesQuery = useQuery<Trade[]>({
    queryKey: ["/api/trades"],
    refetchInterval: 10000,
  });

  const trades = tradesQuery.data ?? [];
  const totalTrades = trades.length;
  const buyTrades = trades.filter((t) => t.side === "buy").length;
  const sellTrades = trades.filter((t) => t.side === "sell").length;
  const filledTrades = trades.filter((t) => t.status === "filled").length;

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-6">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold flex items-center gap-2">
            <History className="w-6 h-6" />
            Trade History
          </h1>
          <p className="text-muted-foreground">View all executed trades</p>
        </div>
        <div className="w-full sm:w-72">
          <KillSwitch />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
              Total Trades
            </div>
            <div className="text-2xl font-bold font-mono">{totalTrades}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
              Buy Orders
            </div>
            <div className="text-2xl font-bold font-mono text-chart-2">{buyTrades}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
              Sell Orders
            </div>
            <div className="text-2xl font-bold font-mono text-destructive">{sellTrades}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
              Fill Rate
            </div>
            <div className="text-2xl font-bold font-mono">
              {totalTrades > 0 ? ((filledTrades / totalTrades) * 100).toFixed(0) : 0}%
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-lg font-semibold">All Trades</CardTitle>
          </CardHeader>
          <CardContent>
            {tradesQuery.isLoading ? (
              <div className="animate-pulse space-y-2">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="h-12 bg-muted rounded-md" />
                ))}
              </div>
            ) : trades.length === 0 ? (
              <div className="text-center py-12 text-muted-foreground">
                <History className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg">No trades yet</p>
                <p className="text-sm">Executed trades will appear here</p>
              </div>
            ) : (
              <ScrollArea className="h-[500px]">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="text-xs uppercase">Date/Time</TableHead>
                      <TableHead className="text-xs uppercase">Symbol</TableHead>
                      <TableHead className="text-xs uppercase">Side</TableHead>
                      <TableHead className="text-xs uppercase text-right">Quantity</TableHead>
                      <TableHead className="text-xs uppercase text-right">Price</TableHead>
                      <TableHead className="text-xs uppercase text-right">Total Value</TableHead>
                      <TableHead className="text-xs uppercase text-right">Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {trades.map((trade) => (
                      <TableRow key={trade.id} data-testid={`row-trade-${trade.id}`}>
                        <TableCell className="font-mono text-xs text-muted-foreground">
                          {formatDateTime(trade.timestamp)}
                        </TableCell>
                        <TableCell className="font-semibold">{trade.symbol}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-1">
                            {trade.side === "buy" ? (
                              <ArrowUpRight className="w-4 h-4 text-chart-2" />
                            ) : (
                              <ArrowDownRight className="w-4 h-4 text-destructive" />
                            )}
                            <span className={trade.side === "buy" ? "text-chart-2" : "text-destructive"}>
                              {trade.side.toUpperCase()}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell className="text-right font-mono">{trade.quantity}</TableCell>
                        <TableCell className="text-right font-mono">
                          {formatCurrency(trade.price)}
                        </TableCell>
                        <TableCell className="text-right font-mono">
                          {formatCurrency(trade.totalValue)}
                        </TableCell>
                        <TableCell className="text-right">
                          <Badge variant={getStatusVariant(trade.status)}>
                            {trade.status}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </ScrollArea>
            )}
          </CardContent>
        </Card>
        <TradeHistory />
      </div>
    </div>
  );
}
