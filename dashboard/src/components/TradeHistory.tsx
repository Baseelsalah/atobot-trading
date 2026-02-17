import { useCallback } from "react";
import { usePolling, formatCurrency, formatTime, plColor } from "../hooks";
import { fetchTrades, fetchPerformance, type Trade } from "../api";

export default function TradeHistory() {
  const fetchTradesStable = useCallback(() => fetchTrades(), []);
  const fetchPerfStable = useCallback(() => fetchPerformance(), []);
  const { data: trades, loading: tradesLoading } = usePolling(fetchTradesStable, 5000);
  const { data: perf } = usePolling(fetchPerfStable, 10000);

  const sortedTrades = [...(trades || [])].sort((a, b) => {
    const ta = a.timestamp ? new Date(a.timestamp).getTime() : 0;
    const tb = b.timestamp ? new Date(b.timestamp).getTime() : 0;
    return tb - ta;
  });

  return (
    <div className="space-y-4">
      {/* 5 key stats */}
      {perf && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          {[
            { label: "Win Rate", value: `${perf.winRate.toFixed(1)}%`, color: perf.winRate >= 50 ? "text-profit" : perf.totalTrades > 0 ? "text-loss" : "" },
            { label: "Profit Factor", value: perf.profitFactor === Infinity ? "---" : perf.profitFactor.toFixed(2), color: perf.profitFactor >= 1 ? "text-profit" : perf.totalTrades > 0 ? "text-loss" : "" },
            { label: "Expectancy", value: formatCurrency(perf.expectancy), color: plColor(perf.expectancy) },
            { label: "Trades", value: `${perf.wins}W / ${perf.losses}L` },
            { label: "Streaks", value: `${perf.consecutiveWins}W / ${perf.consecutiveLosses}L` },
          ].map((s) => (
            <div key={s.label} className="rounded-xl p-3 border" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
              <div className="text-xs" style={{ color: "var(--text-muted)" }}>{s.label}</div>
              <div className={`text-lg font-bold font-mono ${s.color || ""}`}>{s.value}</div>
            </div>
          ))}
        </div>
      )}

      {/* Trade log */}
      {tradesLoading && !trades ? (
        <div className="text-center py-12" style={{ color: "var(--text-muted)" }}>Loading trades...</div>
      ) : sortedTrades.length === 0 ? (
        <div className="rounded-xl border p-12 text-center" style={{ background: "var(--bg-card)", borderColor: "var(--border)", color: "var(--text-muted)" }}>
          No trades yet -- trades will appear here when the market opens.
        </div>
      ) : (
        <div className="rounded-xl border overflow-hidden" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
          <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0" style={{ background: "var(--bg-card)" }}>
                <tr className="border-b" style={{ borderColor: "var(--border)" }}>
                  {["Time", "Symbol", "Side", "Qty", "Price", "Value", "Status"].map((h) => (
                    <th key={h} className="px-4 py-3 text-left text-xs font-medium" style={{ color: "var(--text-muted)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedTrades.map((t: Trade) => (
                  <tr key={t.id} className="border-b last:border-0" style={{ borderColor: "var(--border)" }}>
                    <td className="px-4 py-2.5 text-xs font-mono" style={{ color: "var(--text-secondary)" }}>{formatTime(t.timestamp)}</td>
                    <td className="px-4 py-2.5 font-bold font-mono">{t.symbol}</td>
                    <td className="px-4 py-2.5">
                      <span className={`text-xs font-bold px-2 py-0.5 rounded ${t.side === "buy" ? "bg-green-500/20 text-profit" : "bg-red-500/20 text-loss"}`}>
                        {t.side.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 font-mono">{t.quantity}</td>
                    <td className="px-4 py-2.5 font-mono">{formatCurrency(t.price)}</td>
                    <td className="px-4 py-2.5 font-mono">{formatCurrency(t.totalValue)}</td>
                    <td className="px-4 py-2.5 text-xs" style={{ color: "var(--text-muted)" }}>{t.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
