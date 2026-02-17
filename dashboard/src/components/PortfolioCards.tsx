import { formatCurrency, formatPercent, plColor } from "../hooks";
import type { Portfolio } from "../api";

interface PortfolioCardsProps {
  portfolio: Portfolio | null;
  loading: boolean;
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div
      className="rounded-xl p-4 border"
      style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}
    >
      <div className="text-xs font-medium mb-1" style={{ color: "var(--text-muted)" }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function Skeleton() {
  return <div className="h-7 w-28 rounded bg-gray-700/30 animate-pulse mt-1" />;
}

export default function PortfolioCards({ portfolio, loading }: PortfolioCardsProps) {
  if (loading || !portfolio) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i} title="Loading..."><Skeleton /></Card>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <Card title="Total Equity">
        <div className="text-2xl font-bold font-mono">
          {formatCurrency(portfolio.totalEquity)}
        </div>
        <div className={`text-sm font-mono ${plColor(portfolio.totalPL)}`}>
          {formatCurrency(portfolio.totalPL)} ({formatPercent(portfolio.totalPLPercent)})
        </div>
      </Card>

      <Card title="Today's P&L">
        <div className={`text-2xl font-bold font-mono ${plColor(portfolio.todayPL)}`}>
          {portfolio.todayPL >= 0 ? "+" : ""}{formatCurrency(portfolio.todayPL)}
        </div>
        <div className={`text-sm font-mono ${plColor(portfolio.todayPLPercent)}`}>
          {formatPercent(portfolio.todayPLPercent)}
        </div>
      </Card>

      <Card title="Buying Power">
        <div className="text-2xl font-bold font-mono">
          {formatCurrency(portfolio.buyingPower)}
        </div>
        <div className="text-sm font-mono" style={{ color: "var(--text-secondary)" }}>
          Cash: {formatCurrency(portfolio.cash)}
        </div>
      </Card>

      <Card title="Day Trades Left">
        <div className={`text-2xl font-bold font-mono ${portfolio.dayTradesRemaining <= 1 ? "text-loss" : ""}`}>
          {portfolio.dayTradesRemaining}
        </div>
        <div className="text-sm" style={{ color: "var(--text-secondary)" }}>
          of 3 allowed
        </div>
      </Card>
    </div>
  );
}
