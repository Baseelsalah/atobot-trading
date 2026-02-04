import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp, TrendingDown, DollarSign, Wallet, Calendar } from "lucide-react";
import type { PortfolioSummary as PortfolioSummaryType } from "@shared/schema";

interface PortfolioSummaryProps {
  data: PortfolioSummaryType | null;
  isLoading: boolean;
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

export function PortfolioSummary({ data, isLoading }: PortfolioSummaryProps) {
  if (isLoading || !data) {
    return (
      <Card className="col-span-full">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg font-semibold">Portfolio Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            <div className="h-12 bg-muted rounded-md w-1/3" />
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-16 bg-muted rounded-md" />
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const isPositiveToday = data.todayPL >= 0;

  return (
    <Card className="col-span-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-semibold">Portfolio Summary</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-baseline gap-2 sm:gap-4 flex-wrap">
          <span
            className="text-2xl sm:text-4xl font-bold font-mono"
            data-testid="text-total-equity"
          >
            {formatCurrency(data.totalEquity)}
          </span>
          <div className={`flex items-center gap-1 ${isPositiveToday ? "text-chart-2" : "text-destructive"}`}>
            {isPositiveToday ? (
              <TrendingUp className="w-4 h-4" />
            ) : (
              <TrendingDown className="w-4 h-4" />
            )}
            <span className="font-mono text-xs sm:text-sm font-medium" data-testid="text-today-pl">
              {formatCurrency(data.todayPL)} ({formatPercent(data.todayPLPercent)})
            </span>
            <span className="text-muted-foreground text-xs sm:text-sm">today</span>
          </div>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 sm:gap-4">
          <div className="p-3 sm:p-4 bg-muted/50 rounded-md">
            <div className="flex items-center gap-1 sm:gap-2 text-muted-foreground text-[10px] sm:text-xs uppercase tracking-wide font-medium mb-1">
              <Wallet className="w-3 h-3 sm:w-4 sm:h-4 flex-shrink-0" />
              <span className="truncate">Buying Power</span>
            </div>
            <span className="text-lg sm:text-2xl font-bold font-mono" data-testid="text-buying-power">
              {formatCurrency(data.buyingPower)}
            </span>
          </div>

          <div className="p-3 sm:p-4 bg-muted/50 rounded-md">
            <div className="flex items-center gap-1 sm:gap-2 text-muted-foreground text-[10px] sm:text-xs uppercase tracking-wide font-medium mb-1">
              <DollarSign className="w-3 h-3 sm:w-4 sm:h-4 flex-shrink-0" />
              <span className="truncate">Cash</span>
            </div>
            <span className="text-lg sm:text-2xl font-bold font-mono" data-testid="text-cash">
              {formatCurrency(data.cash)}
            </span>
          </div>

          <div className="p-3 sm:p-4 bg-muted/50 rounded-md">
            <div className="flex items-center gap-1 sm:gap-2 text-muted-foreground text-[10px] sm:text-xs uppercase tracking-wide font-medium mb-1">
              <TrendingUp className="w-3 h-3 sm:w-4 sm:h-4 flex-shrink-0" />
              <span className="truncate">Total P/L</span>
            </div>
            <span
              className={`text-lg sm:text-2xl font-bold font-mono ${data.totalPL >= 0 ? "text-chart-2" : "text-destructive"}`}
              data-testid="text-total-pl"
            >
              {formatCurrency(data.totalPL)}
            </span>
          </div>

          <div className="p-3 sm:p-4 bg-muted/50 rounded-md">
            <div className="flex items-center gap-1 sm:gap-2 text-muted-foreground text-[10px] sm:text-xs uppercase tracking-wide font-medium mb-1">
              <Calendar className="w-3 h-3 sm:w-4 sm:h-4 flex-shrink-0" />
              <span className="truncate">Day Trades</span>
            </div>
            <span className="text-lg sm:text-2xl font-bold font-mono" data-testid="text-day-trades">
              {data.dayTradesRemaining}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
