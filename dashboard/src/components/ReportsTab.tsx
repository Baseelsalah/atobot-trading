import { useState, useCallback } from "react";
import { usePolling, formatCurrency, plColor } from "../hooks";
import { fetchWeeklyScorecard, fetchDailyAnalytics } from "../api";
import { Calendar, BarChart3, TrendingUp } from "lucide-react";

export default function ReportsTab() {
  const today = new Date().toISOString().slice(0, 10);
  const [selectedDate, setSelectedDate] = useState(today);

  const fetchScorecard = useCallback(() => fetchWeeklyScorecard(), []);
  const fetchDaily = useCallback(() => fetchDailyAnalytics(selectedDate), [selectedDate]);
  const { data: scorecard, loading: scLoading } = usePolling(fetchScorecard, 60000);
  const { data: daily, loading: dailyLoading } = usePolling(fetchDaily, 30000);

  return (
    <div className="space-y-4">
      {/* Daily Report */}
      <div className="rounded-xl p-5 border" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold flex items-center gap-2">
            <Calendar className="h-4 w-4 text-blue-500" />
            Daily Report
          </h3>
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="px-3 py-1.5 rounded-lg text-sm font-mono border cursor-pointer"
            style={{ background: "var(--bg-secondary)", borderColor: "var(--border)", color: "var(--text-primary)" }}
          />
        </div>

        {dailyLoading && !daily ? (
          <div className="text-center py-6" style={{ color: "var(--text-muted)" }}>Loading...</div>
        ) : !daily?.ok || !daily.daily ? (
          <div className="text-center py-6" style={{ color: "var(--text-muted)" }}>
            No report for {selectedDate}. Pick a trading day above.
          </div>
        ) : (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[
              { label: "Net P&L", value: formatCurrency(daily.daily.netPnl), color: plColor(daily.daily.netPnl) },
              { label: "Win Rate", value: `${daily.daily.winRate.toFixed(1)}%`, color: daily.daily.winRate >= 50 ? "text-profit" : "text-loss" },
              { label: "Trades", value: `${daily.daily.wins}W / ${daily.daily.losses}L` },
              { label: "Profit Factor", value: daily.daily.profitFactor === Infinity ? "---" : daily.daily.profitFactor.toFixed(2), color: daily.daily.profitFactor >= 1 ? "text-profit" : "text-loss" },
            ].map((s) => (
              <div key={s.label} className="rounded-lg p-3" style={{ background: "var(--bg-secondary)" }}>
                <div className="text-xs" style={{ color: "var(--text-muted)" }}>{s.label}</div>
                <div className={`text-xl font-bold font-mono ${s.color || ""}`}>{s.value}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Weekly Scorecard */}
      <div className="rounded-xl p-5 border" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-blue-500" />
          Weekly Scorecard
        </h3>

        {scLoading && !scorecard ? (
          <div className="text-center py-6" style={{ color: "var(--text-muted)" }}>Loading...</div>
        ) : !scorecard ? (
          <div className="text-center py-6" style={{ color: "var(--text-muted)" }}>No scorecard data yet.</div>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              {[
                { label: "Total P&L", value: formatCurrency(scorecard.summary.totalPnl), color: plColor(scorecard.summary.totalPnl) },
                { label: "Win Rate", value: `${scorecard.summary.winRate.toFixed(1)}%`, color: scorecard.summary.winRate >= 50 ? "text-profit" : scorecard.summary.totalTrades > 0 ? "text-loss" : "" },
                { label: "Trades", value: `${scorecard.summary.wins}W / ${scorecard.summary.losses}L` },
                { label: "Expectancy", value: formatCurrency(scorecard.summary.expectancy), color: plColor(scorecard.summary.expectancy) },
                { label: "Sharpe", value: scorecard.summary.sharpeEstimate.toFixed(2), color: scorecard.summary.sharpeEstimate >= 1 ? "text-profit" : "" },
              ].map((s) => (
                <div key={s.label} className="rounded-lg p-3" style={{ background: "var(--bg-secondary)" }}>
                  <div className="text-xs" style={{ color: "var(--text-muted)" }}>{s.label}</div>
                  <div className={`text-lg font-bold font-mono ${s.color || ""}`}>{s.value}</div>
                </div>
              ))}
            </div>

            {scorecard.recommendations.length > 0 && (
              <div>
                <h4 className="text-sm font-medium mb-2 flex items-center gap-2" style={{ color: "var(--text-secondary)" }}>
                  <TrendingUp className="h-3.5 w-3.5" />
                  Recommendations
                </h4>
                <div className="space-y-1.5">
                  {scorecard.recommendations.map((r, i) => (
                    <div key={i} className="px-3 py-2 rounded text-sm" style={{ background: "var(--bg-secondary)", color: "var(--text-secondary)" }}>
                      {r}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
