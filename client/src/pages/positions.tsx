import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { motion } from "framer-motion";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { TrendingUp, TrendingDown, X, Layers, DollarSign, Target, Activity } from "lucide-react";
import { RiskMonitor } from "@/components/risk-monitor";
import type { Position, PortfolioSummary, BotSettings } from "@shared/schema";

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

export default function PositionsPage() {
  const { toast } = useToast();

  const positionsQuery = useQuery<Position[]>({
    queryKey: ["/api/positions"],
    refetchInterval: 5000,
  });

  const portfolioQuery = useQuery<PortfolioSummary>({
    queryKey: ["/api/portfolio"],
    refetchInterval: 5000,
  });

  const settingsQuery = useQuery<BotSettings>({
    queryKey: ["/api/settings"],
  });

  const closePositionMutation = useMutation({
    mutationFn: (symbol: string) => apiRequest("POST", "/api/positions/close", { symbol }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/positions"] });
      queryClient.invalidateQueries({ queryKey: ["/api/trades"] });
      queryClient.invalidateQueries({ queryKey: ["/api/portfolio"] });
      toast({ title: "Position closed" });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to close position", variant: "destructive" });
    },
  });

  const closeAllMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/positions/close-all"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/positions"] });
      queryClient.invalidateQueries({ queryKey: ["/api/trades"] });
      queryClient.invalidateQueries({ queryKey: ["/api/portfolio"] });
      toast({ title: "All positions closed" });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to close positions", variant: "destructive" });
    },
  });

  const positions = positionsQuery.data ?? [];
  const totalUnrealizedPL = positions.reduce((sum, p) => sum + p.unrealizedPL, 0);
  const totalMarketValue = positions.reduce((sum, p) => sum + p.marketValue, 0);

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-screen-xl mx-auto space-y-6">
      <motion.div 
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between gap-4 flex-wrap"
      >
        <div>
          <h1 className="text-xl sm:text-2xl font-bold flex items-center gap-2 tracking-tight">
            <div className="p-2 rounded-lg bg-muted">
              <Layers className="w-5 h-5 text-foreground" />
            </div>
            Positions
          </h1>
          <p className="text-sm text-muted-foreground mt-1">Manage your current holdings</p>
        </div>
        {positions.length > 0 && (
          <Button
            variant="destructive"
            size="sm"
            onClick={() => closeAllMutation.mutate()}
            disabled={closeAllMutation.isPending}
            data-testid="button-close-all"
          >
            <X className="w-4 h-4 mr-1.5" />
            Close All
          </Button>
        )}
      </motion.div>

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid grid-cols-2 lg:grid-cols-4 gap-4"
      >
        <Card className="metric-card border-border/50">
          <CardContent className="pt-5 pb-4">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-1">
              <DollarSign className="w-3.5 h-3.5" />
              Market Value
            </div>
            <div className="text-xl sm:text-2xl font-bold font-mono">
              {formatCurrency(totalMarketValue)}
            </div>
          </CardContent>
        </Card>
        <Card className="metric-card border-border/50">
          <CardContent className="pt-5 pb-4">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-1">
              {totalUnrealizedPL >= 0 ? <TrendingUp className="w-3.5 h-3.5 text-emerald-400" /> : <TrendingDown className="w-3.5 h-3.5 text-red-400" />}
              Unrealized P/L
            </div>
            <div className={`text-xl sm:text-2xl font-bold font-mono ${totalUnrealizedPL >= 0 ? "ticker-positive" : "ticker-negative"}`}>
              {formatCurrency(totalUnrealizedPL)}
            </div>
          </CardContent>
        </Card>
        <Card className="metric-card border-border/50">
          <CardContent className="pt-5 pb-4">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-1">
              <Target className="w-3.5 h-3.5" />
              Open Positions
            </div>
            <div className="text-xl sm:text-2xl font-bold font-mono">
              {positions.length}
            </div>
          </CardContent>
        </Card>
        <RiskMonitor
          settings={settingsQuery.data ?? null}
          portfolio={portfolioQuery.data ?? null}
          positionsCount={positions.length}
          isLoading={settingsQuery.isLoading}
        />
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
      <Card className="border-border/50">
        <CardHeader className="flex flex-row items-center justify-between gap-4 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-muted">
              <Activity className="w-4 h-4 text-foreground" />
            </div>
            <CardTitle className="text-base font-semibold">All Positions</CardTitle>
          </div>
          <Badge variant="outline" className="text-xs">{positions.length} Active</Badge>
        </CardHeader>
        <CardContent>
          {positions.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <div className="p-4 rounded-full bg-muted/50 mx-auto mb-4 w-fit">
                <Layers className="w-8 h-8 opacity-50" />
              </div>
              <p className="text-base font-medium">No open positions</p>
              <p className="text-sm text-muted-foreground/70">AtoBot will open positions based on market analysis</p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs uppercase">Symbol</TableHead>
                  <TableHead className="text-xs uppercase text-right">Quantity</TableHead>
                  <TableHead className="text-xs uppercase text-right">Avg Entry</TableHead>
                  <TableHead className="text-xs uppercase text-right">Current Price</TableHead>
                  <TableHead className="text-xs uppercase text-right">Market Value</TableHead>
                  <TableHead className="text-xs uppercase text-right">Unrealized P/L</TableHead>
                  <TableHead className="text-xs uppercase text-right">P/L %</TableHead>
                  <TableHead className="text-xs uppercase text-right w-20">Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {positions.map((position) => {
                  const isPositive = position.unrealizedPL >= 0;
                  return (
                    <TableRow key={position.id} data-testid={`row-position-${position.symbol}`}>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-lg">{position.symbol}</span>
                        </div>
                      </TableCell>
                      <TableCell className="text-right font-mono">{position.quantity}</TableCell>
                      <TableCell className="text-right font-mono">
                        {formatCurrency(position.avgEntryPrice)}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {formatCurrency(position.currentPrice)}
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {formatCurrency(position.marketValue)}
                      </TableCell>
                      <TableCell className={`text-right font-mono ${isPositive ? "text-chart-2" : "text-destructive"}`}>
                        <div className="flex items-center justify-end gap-1">
                          {isPositive ? (
                            <TrendingUp className="w-4 h-4" />
                          ) : (
                            <TrendingDown className="w-4 h-4" />
                          )}
                          {formatCurrency(position.unrealizedPL)}
                        </div>
                      </TableCell>
                      <TableCell className={`text-right font-mono ${isPositive ? "text-chart-2" : "text-destructive"}`}>
                        {formatPercent(position.unrealizedPLPercent)}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          size="sm"
                          variant="destructive"
                          onClick={() => closePositionMutation.mutate(position.symbol)}
                          disabled={closePositionMutation.isPending}
                          data-testid={`button-close-${position.symbol}`}
                        >
                          Close
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
      </motion.div>
    </div>
  );
}
