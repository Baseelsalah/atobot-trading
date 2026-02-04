import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { TrendingUp, TrendingDown, X } from "lucide-react";
import type { Position } from "@shared/schema";

interface PositionsTableProps {
  positions: Position[];
  isLoading: boolean;
  onClosePosition?: (symbol: string) => void;
}

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(value);
}

function formatPercent(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

export function PositionsTable({ positions, isLoading, onClosePosition }: PositionsTableProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-lg font-semibold">Active Positions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-2">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-12 bg-muted rounded-md" />
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
          <span>Active Positions</span>
          <span className="text-xs sm:text-sm font-normal text-muted-foreground">
            {positions.length} position{positions.length !== 1 ? "s" : ""}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {positions.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <p>No active positions</p>
            <p className="text-sm">AtoBot will open positions based on market analysis</p>
          </div>
        ) : (
          <div className="overflow-x-auto -mx-4 sm:mx-0">
          <Table className="min-w-[600px]">
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs uppercase">Symbol</TableHead>
                <TableHead className="text-xs uppercase text-right">Qty</TableHead>
                <TableHead className="text-xs uppercase text-right">Entry</TableHead>
                <TableHead className="text-xs uppercase text-right">Current</TableHead>
                <TableHead className="text-xs uppercase text-right">P/L</TableHead>
                <TableHead className="text-xs uppercase text-right">P/L %</TableHead>
                <TableHead className="text-xs uppercase text-right w-12"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {positions.map((position) => {
                const isPositive = position.unrealizedPL >= 0;
                return (
                  <TableRow key={position.id} data-testid={`row-position-${position.symbol}`}>
                    <TableCell className="font-semibold">{position.symbol}</TableCell>
                    <TableCell className="text-right font-mono">{position.quantity}</TableCell>
                    <TableCell className="text-right font-mono">
                      {formatCurrency(position.avgEntryPrice)}
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {formatCurrency(position.currentPrice)}
                    </TableCell>
                    <TableCell className={`text-right font-mono ${isPositive ? "text-chart-2" : "text-destructive"}`}>
                      <div className="flex items-center justify-end gap-1">
                        {isPositive ? (
                          <TrendingUp className="w-3 h-3" />
                        ) : (
                          <TrendingDown className="w-3 h-3" />
                        )}
                        {formatCurrency(position.unrealizedPL)}
                      </div>
                    </TableCell>
                    <TableCell className={`text-right font-mono ${isPositive ? "text-chart-2" : "text-destructive"}`}>
                      {formatPercent(position.unrealizedPLPercent)}
                    </TableCell>
                    <TableCell className="text-right">
                      {onClosePosition && (
                        <Button
                          size="icon"
                          variant="ghost"
                          onClick={() => onClosePosition(position.symbol)}
                          data-testid={`button-close-${position.symbol}`}
                        >
                          <X className="w-4 h-4" />
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
