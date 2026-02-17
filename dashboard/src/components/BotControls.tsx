import { useState, useCallback } from "react";
import { usePolling, formatCurrency, formatTime, addToast } from "../hooks";
import { fetchTradingStatus, fetchSettings, startBot, stopBot, pauseBot, emergencyClose, type BotStatus, type MarketStatus } from "../api";
import { Play, Pause, Square, AlertTriangle, Clock } from "lucide-react";

interface BotControlsProps {
  botStatus: BotStatus | null;
  marketStatus: MarketStatus | null;
  onRefresh: () => void;
}

export default function BotControls({ botStatus, onRefresh }: BotControlsProps) {
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const fetchTradingStatusStable = useCallback(() => fetchTradingStatus(), []);
  const fetchSettingsStable = useCallback(() => fetchSettings(), []);
  const { data: tradingStatus } = usePolling(fetchTradingStatusStable, 5000);
  const { data: settings } = usePolling(fetchSettingsStable, 30000);

  const doAction = async (name: string, fn: () => Promise<unknown>) => {
    setActionLoading(name);
    try {
      await fn();
      addToast(`Bot ${name}ed`, "success");
      onRefresh();
    } catch (e) {
      addToast(`${name} failed: ${e}`, "error");
    } finally {
      setActionLoading(null);
    }
  };

  const handleEmergency = async () => {
    if (!window.confirm("EMERGENCY CLOSE ALL POSITIONS?\n\nThis cannot be undone.")) return;
    setActionLoading("emergency");
    try {
      const result = await emergencyClose();
      addToast(`Emergency close: ${result.closed} positions, P&L: ${formatCurrency(result.totalPL)}`, result.totalPL >= 0 ? "success" : "warning");
      onRefresh();
    } catch (e) {
      addToast(`Emergency close failed: ${e}`, "error");
    } finally {
      setActionLoading(null);
    }
  };

  const status = botStatus?.status || "stopped";

  return (
    <div className="space-y-4">
      {/* Bot control + Emergency in one row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Status + Buttons */}
        <div className="lg:col-span-2 rounded-xl p-5 border" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-xl font-bold capitalize">{status}</div>
              {botStatus?.currentAction && (
                <div className="text-sm" style={{ color: "var(--text-secondary)" }}>{botStatus.currentAction}</div>
              )}
              {botStatus?.lastAnalysis && (
                <div className="text-xs" style={{ color: "var(--text-muted)" }}>Last scan: {formatTime(botStatus.lastAnalysis)}</div>
              )}
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => doAction("start", startBot)}
                disabled={status === "active" || status === "analyzing" || actionLoading !== null}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-green-600 text-white hover:bg-green-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <Play className="h-4 w-4" />
                {actionLoading === "start" ? "..." : "Start"}
              </button>
              <button
                onClick={() => doAction("pause", pauseBot)}
                disabled={status !== "active" || actionLoading !== null}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-amber-600 text-white hover:bg-amber-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <Pause className="h-4 w-4" />
                {actionLoading === "pause" ? "..." : "Pause"}
              </button>
              <button
                onClick={() => doAction("stop", stopBot)}
                disabled={status === "stopped" || actionLoading !== null}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium border disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
              >
                <Square className="h-4 w-4" />
                {actionLoading === "stop" ? "..." : "Stop"}
              </button>
            </div>
          </div>
          {botStatus?.errorMessage && (
            <div className="p-2 rounded bg-red-500/10 text-red-500 text-sm">{botStatus.errorMessage}</div>
          )}
        </div>

        {/* Emergency */}
        <div className="rounded-xl p-5 border border-red-500/30 flex flex-col justify-between" style={{ background: "var(--bg-card)" }}>
          <div className="flex items-center gap-2 text-red-500 font-semibold mb-3">
            <AlertTriangle className="h-4 w-4" />
            Kill Switch
          </div>
          <button
            onClick={handleEmergency}
            disabled={actionLoading !== null}
            className="w-full py-2.5 rounded-lg text-sm font-bold bg-red-600 text-white hover:bg-red-700 disabled:opacity-40 transition-colors"
          >
            {actionLoading === "emergency" ? "CLOSING..." : "CLOSE ALL POSITIONS"}
          </button>
        </div>
      </div>

      {/* Time Guard + Settings in one row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-xl p-5 border" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Clock className="h-4 w-4 text-blue-500" />
            Time Guard
          </h3>
          {tradingStatus ? (
            <div className="space-y-2 text-sm">
              {[
                ["Time (ET)", tradingStatus.currentTimeET],
                ["Can Trade", tradingStatus.canEnterNewPositions ? "YES" : "NO"],
                ["Entry Window", `${tradingStatus.entryStartET} - ${tradingStatus.entryCutoffET}`],
                ["Force Close", tradingStatus.forceCloseET],
              ].map(([label, value]) => (
                <div key={label} className="flex justify-between">
                  <span style={{ color: "var(--text-muted)" }}>{label}</span>
                  <span className={`font-mono ${value === "YES" ? "text-profit" : value === "NO" ? "text-loss" : ""}`}>{value}</span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: "var(--text-muted)" }}>Loading...</div>
          )}
        </div>

        <div className="rounded-xl p-5 border" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
          <h3 className="font-semibold mb-3">Configuration</h3>
          {settings ? (
            <div className="space-y-2 text-sm">
              {[
                ["Mode", settings.isPaperTrading ? "Paper" : "LIVE"],
                ["Max Position", formatCurrency(settings.maxPositionSize)],
                ["Max Daily Loss", formatCurrency(settings.maxDailyLoss)],
                ["Positions", String(settings.maxPositions)],
                ["SL / TP", `${settings.stopLossPercent}% / ${settings.takeProfitPercent}%`],
              ].map(([label, value]) => (
                <div key={label} className="flex justify-between">
                  <span style={{ color: "var(--text-muted)" }}>{label}</span>
                  <span className="font-mono">{value}</span>
                </div>
              ))}
            </div>
          ) : (
            <div style={{ color: "var(--text-muted)" }}>Loading...</div>
          )}
        </div>
      </div>
    </div>
  );
}
