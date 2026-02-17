import { useCallback } from "react";
import { usePolling, formatCurrency } from "../hooks";
import { fetchProfitGoal, fetchHunger, type ProfitGoal, type Hunger } from "../api";
import { Target, Flame } from "lucide-react";

export default function ProfitGoalCard() {
  const fetchGoalStable = useCallback(() => fetchProfitGoal(), []);
  const fetchHungerStable = useCallback(() => fetchHunger(), []);
  const { data: goal } = usePolling(fetchGoalStable, 10000);
  const { data: hunger } = usePolling(fetchHungerStable, 10000);

  if (!goal) return null;

  const pct = Math.min(Math.max(goal.progressPercent, 0), 100);
  const barColor = goal.goalMet ? "#22c55e" : pct > 50 ? "#3b82f6" : "#eab308";

  const hungerColors: Record<string, string> = {
    fed: "#22c55e",
    hungry: "#eab308",
    starving: "#f97316",
    desperate: "#ef4444",
  };
  const hungerColor = hunger ? hungerColors[hunger.hungerLevel] || "#6b7280" : "#6b7280";

  return (
    <div className="rounded-xl p-4 border" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium flex items-center gap-2" style={{ color: "var(--text-secondary)" }}>
          <Target className="h-4 w-4" />
          Daily Profit Goal
        </h3>
        {hunger && (
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium"
            style={{ background: `${hungerColor}20`, color: hungerColor, border: `1px solid ${hungerColor}40` }}>
            <Flame className="h-3 w-3" />
            {hunger.hungerLevel.toUpperCase()}
          </div>
        )}
      </div>

      {/* Progress bar */}
      <div className="relative mb-2">
        <div className="h-4 rounded-full overflow-hidden" style={{ background: "var(--bg-secondary)" }}>
          <div
            className="h-full rounded-full transition-all duration-700 flex items-center justify-end pr-2"
            style={{ width: `${Math.max(pct, 2)}%`, background: barColor }}
          >
            {pct > 15 && <span className="text-[10px] font-bold text-white">{pct.toFixed(0)}%</span>}
          </div>
        </div>
      </div>

      {/* Numbers */}
      <div className="flex justify-between text-xs mb-3" style={{ color: "var(--text-muted)" }}>
        <span>
          <span className="font-mono font-medium" style={{ color: "var(--text-primary)" }}>
            {formatCurrency(goal.currentProfit)}
          </span>
          {" "}/ {formatCurrency(goal.dailyGoal)}
        </span>
        {goal.goalMet ? (
          <span className="text-profit font-medium">GOAL MET</span>
        ) : (
          <span>{goal.tradesNeeded} trades needed</span>
        )}
      </div>

      {/* Breakdown */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div className="rounded p-2" style={{ background: "var(--bg-secondary)" }}>
          <div style={{ color: "var(--text-muted)" }}>Realized</div>
          <div className="font-mono font-medium text-profit">{formatCurrency(goal.realizedProfit)}</div>
        </div>
        <div className="rounded p-2" style={{ background: "var(--bg-secondary)" }}>
          <div style={{ color: "var(--text-muted)" }}>Unrealized</div>
          <div className="font-mono font-medium">{formatCurrency(goal.unrealizedProfit)}</div>
        </div>
        <div className="rounded p-2" style={{ background: "var(--bg-secondary)" }}>
          <div style={{ color: "var(--text-muted)" }}>Avg/Trade</div>
          <div className="font-mono font-medium">{formatCurrency(goal.avgProfitPerTrade)}</div>
        </div>
      </div>

      {/* Hunger details */}
      {hunger && hunger.profitNeeded > 0 && (
        <div className="mt-2 text-xs px-3 py-1.5 rounded" style={{ background: "var(--bg-secondary)", color: "var(--text-secondary)" }}>
          {hunger.message}
        </div>
      )}
    </div>
  );
}
