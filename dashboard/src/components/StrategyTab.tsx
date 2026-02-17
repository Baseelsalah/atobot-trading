import { useCallback, useState } from "react";
import { usePolling, formatCurrency, addToast } from "../hooks";
import { fetchBrainStatus, fetchStrategies, triggerResearch, type Strategy } from "../api";
import { Brain, RefreshCw } from "lucide-react";

export default function StrategyTab() {
  const [researchLoading, setResearchLoading] = useState(false);
  const fetchBrain = useCallback(() => fetchBrainStatus(), []);
  const fetchStrats = useCallback(() => fetchStrategies(), []);
  const { data: brain } = usePolling(fetchBrain, 15000);
  const { data: strategies, loading: stratsLoading, refetch } = usePolling(fetchStrats, 30000);

  const handleResearch = async () => {
    setResearchLoading(true);
    try {
      await triggerResearch();
      addToast("Research triggered", "success");
      refetch();
    } catch (e) {
      addToast(`Research failed: ${e}`, "error");
    } finally {
      setResearchLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Brain summary + research button */}
      <div className="rounded-xl p-5 border" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain className="h-5 w-5 text-purple-500" />
            <div>
              <h3 className="font-semibold">AI Brain</h3>
              {brain && (
                <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
                  {brain.activeStrategies}/{brain.strategies} strategies active
                  &middot; {brain.overallConfidence}% confidence
                  &middot; Top: {brain.topStrategy}
                </span>
              )}
            </div>
          </div>
          <button
            onClick={handleResearch}
            disabled={researchLoading}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-40 transition-colors"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${researchLoading ? "animate-spin" : ""}`} />
            {researchLoading ? "Running..." : "Run Research"}
          </button>
        </div>
      </div>

      {/* Strategy Cards */}
      {stratsLoading && !strategies ? (
        <div className="text-center py-8" style={{ color: "var(--text-muted)" }}>Loading strategies...</div>
      ) : !strategies || strategies.length === 0 ? (
        <div className="rounded-xl border p-8 text-center" style={{ background: "var(--bg-card)", borderColor: "var(--border)", color: "var(--text-muted)" }}>
          No strategies configured yet.
        </div>
      ) : (
        <div className="grid gap-3">
          {strategies.map((s: Strategy) => {
            let symbols: string[] = [];
            try { symbols = JSON.parse(s.symbols); } catch { /* skip */ }

            return (
              <div key={s.id} className="rounded-xl p-4 border" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${s.isActive ? "bg-green-500" : "bg-gray-500"}`} />
                    <span className="font-bold">{s.name}</span>
                    <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(139,92,246,0.2)", color: "#a78bfa" }}>
                      {s.type.replace("_", " ")}
                    </span>
                  </div>
                  <span className={`font-mono font-bold ${s.confidence >= 60 ? "text-profit" : ""}`}>
                    {s.confidence}%
                  </span>
                </div>

                <p className="text-sm mb-2" style={{ color: "var(--text-secondary)" }}>{s.description}</p>

                {/* Symbols + stats on one line */}
                <div className="flex items-center justify-between">
                  <div className="flex flex-wrap gap-1">
                    {symbols.map((sym) => (
                      <span key={sym} className="text-xs px-1.5 py-0.5 rounded font-mono"
                        style={{ background: "var(--bg-secondary)", color: "var(--text-secondary)" }}>
                        {sym}
                      </span>
                    ))}
                  </div>
                  <div className="flex gap-4 text-xs font-mono" style={{ color: "var(--text-muted)" }}>
                    <span>{s.totalTrades} trades</span>
                    <span className={s.winRate >= 50 ? "text-profit" : s.totalTrades > 0 ? "text-loss" : ""}>{s.winRate.toFixed(0)}% win</span>
                    <span className={s.totalProfit >= 0 ? "text-profit" : "text-loss"}>{formatCurrency(s.totalProfit)}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
