import { useState, useCallback, useEffect, useRef } from "react";
import { useAuth, useTheme, usePolling, useToasts, timeAgo, type Toast, addToast } from "./hooks";
import * as api from "./api";
import Header from "./components/Header";
import PortfolioCards from "./components/PortfolioCards";
import EquityChart from "./components/EquityChart";
import RiskPanel from "./components/RiskPanel";
import ProfitGoalCard from "./components/ProfitGoalCard";
import PositionsTable from "./components/PositionsTable";
import TradeHistory from "./components/TradeHistory";
import BotControls from "./components/BotControls";
import AlertsPanel from "./components/AlertsPanel";
import ReportsTab from "./components/ReportsTab";
import StrategyTab from "./components/StrategyTab";
import LoginPage from "./pages/LoginPage";
import AdminPanel from "./components/AdminPanel";
import ApiKeySettings from "./components/ApiKeySettings";

type Tab = "overview" | "positions" | "trades" | "alerts" | "strategy" | "reports" | "controls" | "settings" | "admin";

const toastColors: Record<string, { bg: string; border: string }> = {
  success: { bg: "rgba(34,197,94,0.15)", border: "#22c55e" },
  error: { bg: "rgba(239,68,68,0.15)", border: "#ef4444" },
  warning: { bg: "rgba(234,179,8,0.15)", border: "#eab308" },
  info: { bg: "rgba(59,130,246,0.15)", border: "#3b82f6" },
};

function ToastContainer() {
  const toasts = useToasts();
  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 space-y-2" style={{ maxWidth: "360px" }}>
      {toasts.map((t: Toast) => {
        const c = toastColors[t.type] || toastColors.info;
        return (
          <div
            key={t.id}
            className="px-4 py-3 rounded-lg text-sm font-medium shadow-lg animate-slide-in"
            style={{ background: c.bg, borderLeft: `3px solid ${c.border}`, color: "var(--text-primary)" }}
          >
            {t.message}
          </div>
        );
      })}
    </div>
  );
}

function PendingApproval({ onLogout }: { onLogout: () => void }) {
  return (
    <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-[#16161e] rounded-xl border border-gray-800 p-8 text-center">
        <div className="text-4xl mb-4">&#9203;</div>
        <h2 className="text-xl font-bold text-white mb-2">Pending Approval</h2>
        <p className="text-gray-400 mb-6">
          Your account is waiting for admin approval. Please check back later.
        </p>
        <button
          onClick={onLogout}
          className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600"
        >
          Logout
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const { user, token, loading: authLoading, login, registerUser, logout } = useAuth();
  const { dark, toggle } = useTheme();
  const [tab, setTab] = useState<Tab>("overview");

  // Auth gate
  if (authLoading) {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center">
        <p className="text-gray-400">Loading...</p>
      </div>
    );
  }

  if (!user || !token) {
    return <LoginPage onLogin={login} onRegister={registerUser} />;
  }

  if (user.status === "pending") {
    return <PendingApproval onLogout={logout} />;
  }

  if (user.status === "rejected" || user.status === "suspended") {
    return (
      <div className="min-h-screen bg-[#0a0a0f] flex items-center justify-center p-4">
        <div className="w-full max-w-md bg-[#16161e] rounded-xl border border-gray-800 p-8 text-center">
          <h2 className="text-xl font-bold text-white mb-2">Account {user.status}</h2>
          <p className="text-gray-400 mb-6">Contact the admin for more information.</p>
          <button onClick={logout} className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600">Logout</button>
        </div>
      </div>
    );
  }

  return <Dashboard user={user} dark={dark} toggle={toggle} tab={tab} setTab={setTab} onLogout={logout} />;
}

// The main dashboard (only rendered when authenticated and approved)
function Dashboard({
  user, dark, toggle, tab, setTab, onLogout,
}: {
  user: api.AuthUser;
  dark: boolean;
  toggle: () => void;
  tab: Tab;
  setTab: (t: Tab) => void;
  onLogout: () => void;
}) {
  const isAdmin = user.role === "admin";

  const tabs: { key: Tab; label: string }[] = [
    { key: "overview", label: "Overview" },
    { key: "positions", label: "Positions" },
    { key: "trades", label: "Trades" },
    { key: "alerts", label: "Alerts" },
    { key: "strategy", label: "Strategy" },
    { key: "reports", label: "Reports" },
    ...(isAdmin ? [{ key: "controls" as Tab, label: "Controls" }] : []),
    { key: "settings", label: "Settings" },
    ...(isAdmin ? [{ key: "admin" as Tab, label: "Admin" }] : []),
  ];

  const fetchPortfolio = useCallback(() => api.fetchPortfolio(), []);
  const fetchBotStatus = useCallback(() => api.fetchBotStatus(), []);
  const fetchMarketStatus = useCallback(() => api.fetchMarketStatus(), []);
  const fetchRegime = useCallback(() => api.fetchMarketRegime(), []);
  const fetchAlerts = useCallback(() => api.fetchAlerts(), []);

  const portfolio = usePolling(fetchPortfolio, 3000);
  const botStatus = usePolling(fetchBotStatus, 2000);
  const marketStatus = usePolling(fetchMarketStatus, 10000);
  const regime = usePolling(fetchRegime, 15000);
  const alerts = usePolling(fetchAlerts, 5000);

  // Trade fill notifications
  const prevTradeCount = useRef<number | null>(null);
  const fetchTrades = useCallback(() => api.fetchTrades(), []);
  const trades = usePolling(fetchTrades, 5000);

  useEffect(() => {
    if (!trades.data) return;
    const count = trades.data.length;
    if (prevTradeCount.current !== null && count > prevTradeCount.current) {
      const newest = trades.data[0];
      if (newest) {
        addToast(
          `${newest.side.toUpperCase()} ${newest.quantity} ${newest.symbol} @ $${newest.price.toFixed(2)}`,
          newest.side === "buy" ? "info" : "success"
        );
      }
    }
    prevTradeCount.current = count;
  }, [trades.data]);

  const unreadAlerts = (alerts.data || []).filter((a) => !a.isRead).length;

  return (
    <div className="min-h-screen" style={{ background: "var(--bg-primary)", color: "var(--text-primary)" }}>
      <Header
        dark={dark}
        onToggleTheme={toggle}
        botStatus={botStatus.data}
        marketStatus={marketStatus.data}
        regime={regime.data}
        alertCount={unreadAlerts}
        user={user}
        onLogout={onLogout}
      />

      {/* Tab navigation */}
      <nav className="border-b overflow-x-auto" style={{ borderColor: "var(--border)", background: "var(--bg-secondary)" }}>
        <div className="max-w-7xl mx-auto px-4 flex gap-1">
          {tabs.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`px-4 py-2.5 text-sm font-medium transition-colors whitespace-nowrap relative ${
                tab === t.key ? "border-b-2 border-blue-500 text-blue-500" : "hover:opacity-80"
              }`}
              style={tab !== t.key ? { color: "var(--text-secondary)" } : undefined}
            >
              {t.label}
              {t.key === "alerts" && unreadAlerts > 0 && (
                <span className="ml-1.5 px-1.5 py-0.5 rounded-full bg-red-500 text-white text-[10px] font-bold">
                  {unreadAlerts}
                </span>
              )}
            </button>
          ))}
        </div>
      </nav>

      {/* Last updated */}
      <div className="max-w-7xl mx-auto px-4 pt-2 flex justify-end">
        <span className="text-[10px]" style={{ color: "var(--text-muted)" }}>
          Updated {timeAgo(portfolio.lastUpdated)}
        </span>
      </div>

      <main className="max-w-7xl mx-auto px-4 py-4">
        {tab === "overview" && (
          <div className="space-y-4">
            <PortfolioCards portfolio={portfolio.data} loading={portfolio.loading} />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <EquityChart />
              <ProfitGoalCard />
            </div>
            <RiskPanel />
          </div>
        )}
        {tab === "positions" && <PositionsTable />}
        {tab === "trades" && <TradeHistory />}
        {tab === "alerts" && <AlertsPanel />}
        {tab === "strategy" && <StrategyTab />}
        {tab === "reports" && <ReportsTab />}
        {tab === "controls" && isAdmin && (
          <BotControls
            botStatus={botStatus.data}
            marketStatus={marketStatus.data}
            onRefresh={botStatus.refetch}
          />
        )}
        {tab === "settings" && <ApiKeySettings />}
        {tab === "admin" && isAdmin && <AdminPanel />}
      </main>

      <ToastContainer />
    </div>
  );
}
