import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";
import type { Trade } from "@shared/schema";

interface RecentTradesProps {
  trades: Trade[];
  isLoading: boolean;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(value);
}

function formatTime(timestamp: Date | string | null): string {
  if (!timestamp) return "";
  const date = typeof timestamp === "string" ? new Date(timestamp) : timestamp;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
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

export function RecentTrades({ trades, isLoading }: RecentTradesProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg font-semibold">Recent Trades</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-2">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-10 bg-muted rounded-md" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base sm:text-lg font-semibold flex items-center justify-between gap-2">
          <span>Recent Trades</span>
          <Badge variant="outline" className="text-[10px] sm:text-xs">
            {trades.length} trades
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {trades.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <p>No trades executed yet</p>
            <p className="text-sm">AtoBot will execute trades based on GPT analysis</p>
          </div>
        ) : (
          <ScrollArea className="h-64">
            <div className="overflow-x-auto">
            <Table className="min-w-[500px]">
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs uppercase">Time</TableHead>
                  <TableHead className="text-xs uppercase">Symbol</TableHead>
                  <TableHead className="text-xs uppercase">Side</TableHead>
                  <TableHead className="text-xs uppercase text-right">Qty</TableHead>
                  <TableHead className="text-xs uppercase text-right">Price</TableHead>
                  <TableHead className="text-xs uppercase text-right">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {trades.slice(0, 10).map((trade) => (
                  <TableRow key={trade.id} data-testid={`row-trade-${trade.id}`}>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      {formatTime(trade.timestamp)}
                    </TableCell>
                    <TableCell className="font-semibold">{trade.symbol}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1">
                        {trade.side === "buy" ? (
                          <ArrowUpRight className="w-3 h-3 text-chart-2" />
                        ) : (
                          <ArrowDownRight className="w-3 h-3 text-destructive" />
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
                    <TableCell className="text-right">
                      <Badge variant={getStatusVariant(trade.status)} className="text-xs">
                        {trade.status}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            </div>
          </ScrollArea>
        )}
      </CardContent>
    </Card>
  );
}
