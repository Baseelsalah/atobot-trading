import { useCallback } from "react";
import { usePolling, formatTime, addToast } from "../hooks";
import { fetchAlerts, fetchSignals, approveAlert, denyAlert, markAllAlertsRead, type Alert, type Signal } from "../api";
import { Bell, CheckCircle, XCircle, Eye, Activity } from "lucide-react";

export default function AlertsPanel() {
  const fetchAlertsStable = useCallback(() => fetchAlerts(), []);
  const fetchSignalsStable = useCallback(() => fetchSignals(), []);
  const { data: alerts, loading: alertsLoading, refetch: refetchAlerts } = usePolling(fetchAlertsStable, 5000);
  const { data: signals, loading: signalsLoading } = usePolling(fetchSignalsStable, 5000);

  const handleApprove = async (id: string) => {
    try {
      await approveAlert(id);
      addToast("Alert approved, trade executing", "success");
      refetchAlerts();
    } catch (e) {
      addToast(`Failed to approve: ${e}`, "error");
    }
  };

  const handleDeny = async (id: string) => {
    try {
      await denyAlert(id);
      addToast("Alert denied", "info");
      refetchAlerts();
    } catch (e) {
      addToast(`Failed to deny: ${e}`, "error");
    }
  };

  const handleMarkAllRead = async () => {
    try {
      await markAllAlertsRead();
      refetchAlerts();
    } catch (e) {
      addToast(`Failed: ${e}`, "error");
    }
  };

  const unreadCount = (alerts || []).filter((a) => !a.isRead).length;

  return (
    <div className="space-y-6">
      {/* Signals Section */}
      <div>
        <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
          <Activity className="h-5 w-5 text-blue-500" />
          Pending Signals
        </h3>
        {signalsLoading && !signals ? (
          <div className="text-center py-8" style={{ color: "var(--text-muted)" }}>Loading signals...</div>
        ) : !signals || signals.length === 0 ? (
          <div className="rounded-xl border p-8 text-center" style={{ background: "var(--bg-card)", borderColor: "var(--border)", color: "var(--text-muted)" }}>
            No pending signals
          </div>
        ) : (
          <div className="grid gap-3">
            {signals.map((s: Signal) => (
              <div key={s.id} className="rounded-xl p-4 border" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="font-bold font-mono text-lg">{s.symbol}</span>
                    <span className={`text-xs font-bold px-2 py-0.5 rounded ${s.side === "buy" ? "bg-green-500/20 text-profit" : "bg-red-500/20 text-loss"}`}>
                      {s.side.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-mono">{s.confidence}% confidence</div>
                    <div className="text-xs" style={{ color: "var(--text-muted)" }}>{formatTime(s.timestamp)}</div>
                  </div>
                </div>
                <div className="text-sm mt-2" style={{ color: "var(--text-secondary)" }}>{s.reason}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Alerts Section */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Bell className="h-5 w-5 text-amber-500" />
            Alerts
            {unreadCount > 0 && (
              <span className="px-2 py-0.5 rounded-full text-xs font-bold bg-amber-500 text-white">{unreadCount}</span>
            )}
          </h3>
          {unreadCount > 0 && (
            <button
              onClick={handleMarkAllRead}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors hover:opacity-80"
              style={{ background: "var(--bg-secondary)", color: "var(--text-secondary)" }}
            >
              <Eye className="h-3 w-3" />
              Mark All Read
            </button>
          )}
        </div>

        {alertsLoading && !alerts ? (
          <div className="text-center py-8" style={{ color: "var(--text-muted)" }}>Loading alerts...</div>
        ) : !alerts || alerts.length === 0 ? (
          <div className="rounded-xl border p-8 text-center" style={{ background: "var(--bg-card)", borderColor: "var(--border)", color: "var(--text-muted)" }}>
            No alerts
          </div>
        ) : (
          <div className="space-y-2">
            {[...alerts].reverse().map((a: Alert) => (
              <div
                key={a.id}
                className={`rounded-xl p-4 border ${!a.isRead ? "border-l-4 border-l-amber-500" : ""}`}
                style={{ background: "var(--bg-card)", borderColor: !a.isRead ? undefined : "var(--border)" }}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs px-2 py-0.5 rounded font-medium"
                        style={{ background: a.type === "trade" ? "rgba(59,130,246,0.2)" : "rgba(107,114,128,0.2)", color: a.type === "trade" ? "#3b82f6" : "var(--text-muted)" }}>
                        {a.type.toUpperCase()}
                      </span>
                      <span className="font-medium text-sm">{a.title}</span>
                    </div>
                    <p className="text-sm mt-1" style={{ color: "var(--text-secondary)" }}>{a.message}</p>
                    <div className="text-xs mt-1" style={{ color: "var(--text-muted)" }}>{formatTime(a.timestamp)}</div>
                  </div>
                  {a.type === "trade" && !a.isResolved && (
                    <div className="flex gap-2 ml-4">
                      <button
                        onClick={() => handleApprove(a.id)}
                        className="p-2 rounded-lg bg-green-500/20 text-green-500 hover:bg-green-500/30 transition-colors"
                        title="Approve trade"
                      >
                        <CheckCircle className="h-4 w-4" />
                      </button>
                      <button
                        onClick={() => handleDeny(a.id)}
                        className="p-2 rounded-lg bg-red-500/20 text-red-500 hover:bg-red-500/30 transition-colors"
                        title="Deny trade"
                      >
                        <XCircle className="h-4 w-4" />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
