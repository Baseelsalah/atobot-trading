import { Sun, Moon, Zap, LogOut } from "lucide-react";
import type { BotStatus, MarketStatus, MarketRegime, AuthUser } from "../api";

interface HeaderProps {
  dark: boolean;
  onToggleTheme: () => void;
  botStatus: BotStatus | null;
  marketStatus: MarketStatus | null;
  regime: MarketRegime | null;
  alertCount: number;
  user?: AuthUser;
  onLogout?: () => void;
}

const statusConfig: Record<string, { color: string; label: string }> = {
  active: { color: "#22c55e", label: "Active" },
  paused: { color: "#eab308", label: "Paused" },
  analyzing: { color: "#3b82f6", label: "Analyzing" },
  error: { color: "#ef4444", label: "Error" },
  stopped: { color: "#6b7280", label: "Stopped" },
};

const regimeConfig: Record<string, { color: string; bg: string; label: string }> = {
  bullish: { color: "#22c55e", bg: "rgba(34,197,94,0.15)", label: "BULL" },
  bearish: { color: "#ef4444", bg: "rgba(239,68,68,0.15)", label: "BEAR" },
  cautious: { color: "#eab308", bg: "rgba(234,179,8,0.15)", label: "CAUTIOUS" },
  neutral: { color: "#6b7280", bg: "rgba(107,114,128,0.15)", label: "NEUTRAL" },
  aggressive: { color: "#22c55e", bg: "rgba(34,197,94,0.15)", label: "AGGRESSIVE" },
};

export default function Header({ dark, onToggleTheme, botStatus, marketStatus, regime, alertCount, user, onLogout }: HeaderProps) {
  const status = botStatus ? statusConfig[botStatus.status] || statusConfig.stopped : null;
  const regimeKey = regime?.recommendation || "neutral";
  const regimeStyle = regimeConfig[regimeKey] || regimeConfig.neutral;

  return (
    <header
      className="border-b px-4 py-3 flex items-center justify-between"
      style={{ borderColor: "var(--border)", background: "var(--bg-secondary)" }}
    >
      <div className="flex items-center gap-3">
        <div className="h-8 w-8 rounded-lg bg-blue-600 flex items-center justify-center">
          <Zap className="h-4 w-4 text-white" />
        </div>
        <div>
          <span className="font-bold text-lg tracking-wide">ATOBOT</span>
          <span className="text-xs ml-2" style={{ color: "var(--text-muted)" }}>Trading Dashboard</span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Bot Status */}
        {status && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium"
            style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}>
            <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: status.color }} />
            {status.label}
          </div>
        )}

        {/* Market Status */}
        {marketStatus && (
          <div className="flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium"
            style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}>
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: marketStatus.isOpen ? "#22c55e" : "#6b7280" }}
            />
            {marketStatus.isOpen ? "Market Open" : "Market Closed"}
          </div>
        )}

        {/* Market Regime */}
        {regime && (
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold"
            style={{ background: regimeStyle.bg, color: regimeStyle.color, border: `1px solid ${regimeStyle.color}30` }}>
            {regimeStyle.label}
          </div>
        )}

        {/* Paper Trading Badge */}
        <div className="px-3 py-1 rounded-full text-xs font-medium bg-amber-500/20 text-amber-500 border border-amber-500/30">
          PAPER
        </div>

        {/* Alert Count Badge */}
        {alertCount > 0 && (
          <div className="px-2 py-0.5 rounded-full text-xs font-bold bg-red-500 text-white min-w-[20px] text-center">
            {alertCount}
          </div>
        )}

        {/* Theme Toggle */}
        <button
          onClick={onToggleTheme}
          className="p-2 rounded-lg hover:opacity-70 transition-opacity"
          style={{ color: "var(--text-secondary)" }}
        >
          {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>

        {/* User + Logout */}
        {user && (
          <div className="flex items-center gap-2 ml-2 pl-2 border-l" style={{ borderColor: "var(--border)" }}>
            <span className="text-xs" style={{ color: "var(--text-secondary)" }}>{user.displayName}</span>
            {onLogout && (
              <button onClick={onLogout} className="p-1.5 rounded hover:bg-gray-700/50 transition-colors"
                style={{ color: "var(--text-secondary)" }} title="Logout">
                <LogOut className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
