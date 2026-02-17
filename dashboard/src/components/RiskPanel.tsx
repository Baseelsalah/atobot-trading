import { useCallback } from "react";
import { usePolling } from "../hooks";
import { fetchRiskDashboard } from "../api";
import { ShieldAlert } from "lucide-react";

export default function RiskPanel() {
  const fetcher = useCallback(() => fetchRiskDashboard(), []);
  const { data } = usePolling(fetcher, 15000);

  const m = data?.metrics;
  if (!m) return null;

  const items = [
    { label: "Portfolio Heat", value: m.portfolioHeatLevel, max: 100 },
    { label: "Drawdown", value: m.currentDrawdown, max: 20 },
    { label: "Volatility", value: m.portfolioVolatility, max: 50 },
    { label: "Risk Capacity", value: m.riskCapacity, max: 100, invert: true },
  ];

  function barColor(value: number, max: number, invert?: boolean) {
    const pct = value / max;
    if (invert) return pct > 0.5 ? "#22c55e" : pct > 0.2 ? "#eab308" : "#ef4444";
    return pct < 0.3 ? "#22c55e" : pct < 0.6 ? "#eab308" : "#ef4444";
  }

  return (
    <div className="rounded-xl p-4 border" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
      <h3 className="text-sm font-medium mb-3 flex items-center gap-2" style={{ color: "var(--text-secondary)" }}>
        <ShieldAlert className="h-4 w-4" />
        Risk Monitor
      </h3>

      <div className="grid grid-cols-4 gap-4">
        {items.map((item) => {
          const pct = Math.min((item.value / item.max) * 100, 100);
          const color = barColor(item.value, item.max, item.invert);
          return (
            <div key={item.label}>
              <div className="flex justify-between text-xs mb-1">
                <span style={{ color: "var(--text-muted)" }}>{item.label}</span>
                <span className="font-mono font-medium" style={{ color }}>{item.value.toFixed(1)}%</span>
              </div>
              <div className="h-2 rounded-full overflow-hidden" style={{ background: "var(--bg-secondary)" }}>
                <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, background: color }} />
              </div>
            </div>
          );
        })}
      </div>

      {data?.recommendations && data.recommendations.length > 0 && (
        <div className="mt-3 text-xs" style={{ color: "var(--text-muted)" }}>
          {data.recommendations[0]}
        </div>
      )}
    </div>
  );
}
