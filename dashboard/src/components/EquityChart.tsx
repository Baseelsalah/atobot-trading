import { useState, useCallback } from "react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { usePolling, formatCurrency } from "../hooks";
import { fetchPortfolioHistory, type PortfolioHistoryPoint } from "../api";

const periods = ["1D", "1W", "1M", "3M"] as const;

export default function EquityChart() {
  const [period, setPeriod] = useState<string>("1D");

  const fetcher = useCallback(() => fetchPortfolioHistory(period), [period]);
  const { data, loading } = usePolling(fetcher, 30000);

  const chartData = (data || []).map((p: PortfolioHistoryPoint) => ({
    time: new Date(p.timestamp).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" }),
    equity: p.equity,
    pl: p.profit_loss,
  }));

  const isPositive = chartData.length > 0 && chartData[chartData.length - 1].equity >= chartData[0].equity;

  return (
    <div className="rounded-xl p-4 border" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
          Equity Curve
        </h3>
        <div className="flex gap-1">
          {periods.map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`px-3 py-1 text-xs rounded-md font-medium transition-colors ${
                period === p
                  ? "bg-blue-600 text-white"
                  : "hover:opacity-70"
              }`}
              style={period !== p ? { color: "var(--text-muted)", background: "var(--bg-secondary)" } : undefined}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {loading && !data ? (
        <div className="h-64 flex items-center justify-center" style={{ color: "var(--text-muted)" }}>
          Loading chart...
        </div>
      ) : chartData.length === 0 ? (
        <div className="h-64 flex items-center justify-center" style={{ color: "var(--text-muted)" }}>
          No data available for this period
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0.3} />
                <stop offset="100%" stopColor={isPositive ? "#22c55e" : "#ef4444"} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="time"
              tick={{ fontSize: 11, fill: "var(--text-muted)" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 11, fill: "var(--text-muted)" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
              domain={["dataMin - 100", "dataMax + 100"]}
            />
            <Tooltip
              contentStyle={{
                background: "var(--bg-card)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              formatter={(value: number) => [formatCurrency(value), "Equity"]}
            />
            <Area
              type="monotone"
              dataKey="equity"
              stroke={isPositive ? "#22c55e" : "#ef4444"}
              strokeWidth={2}
              fill="url(#equityGrad)"
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
