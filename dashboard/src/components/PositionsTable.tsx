import { useCallback, useMemo } from "react";
import { usePolling, formatCurrency, formatPercent, plColor, addToast } from "../hooks";
import { fetchPositions, fetchManagedPositions, fetchPortfolio, closePosition, closeAllPositions, type Position, type ManagedPosition } from "../api";
import { X, AlertTriangle } from "lucide-react";

export default function PositionsTable() {
  const fetcher = useCallback(() => fetchPositions(), []);
  const managedFetcher = useCallback(() => fetchManagedPositions(), []);
  const portfolioFetcher = useCallback(() => fetchPortfolio(), []);
  const { data: positions, loading, refetch } = usePolling(fetcher, 3000);
  const { data: managed } = usePolling(managedFetcher, 3000);
  const { data: portfolio } = usePolling(portfolioFetcher, 5000);

  const managedMap = useMemo(() => {
    const map = new Map<string, ManagedPosition>();
    (managed || []).forEach((m) => map.set(m.symbol, m));
    return map;
  }, [managed]);

  const handleClose = async (symbol: string) => {
    if (!window.confirm(`Close ${symbol}?`)) return;
    try {
      await closePosition(symbol);
      addToast(`Closed ${symbol}`, "success");
      refetch();
    } catch (e) {
      addToast(`Failed to close ${symbol}: ${e}`, "error");
    }
  };

  const handleCloseAll = async () => {
    if (!window.confirm("Close ALL positions? This cannot be undone.")) return;
    try {
      await closeAllPositions();
      addToast("All positions closed", "success");
      refetch();
    } catch (e) {
      addToast(`Failed: ${e}`, "error");
    }
  };

  if (loading && !positions) {
    return <div className="text-center py-12" style={{ color: "var(--text-muted)" }}>Loading positions...</div>;
  }

  const list = positions || [];
  const totalUnrealized = list.reduce((sum, p) => sum + p.unrealizedPL, 0);
  const totalEquity = portfolio?.totalEquity || 100000;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-semibold">
            {list.length} Position{list.length !== 1 ? "s" : ""} Open
          </h3>
          <span className={`text-sm font-mono ${plColor(totalUnrealized)}`}>
            Unrealized: {totalUnrealized >= 0 ? "+" : ""}{formatCurrency(totalUnrealized)}
          </span>
        </div>
        {list.length > 0 && (
          <button
            onClick={handleCloseAll}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium bg-red-600 text-white hover:bg-red-700 transition-colors"
          >
            <AlertTriangle className="h-3.5 w-3.5" />
            Close All
          </button>
        )}
      </div>

      {list.length === 0 ? (
        <div className="rounded-xl border p-12 text-center" style={{ background: "var(--bg-card)", borderColor: "var(--border)", color: "var(--text-muted)" }}>
          No open positions. The bot will open positions when the market opens.
        </div>
      ) : (
        <div className="rounded-xl border overflow-hidden" style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}>
                {["Symbol", "Qty", "Entry", "Current", "Stop / TP", "P&L", "Alloc", ""].map((h) => (
                  <th key={h} className="px-4 py-3 text-left text-xs font-medium" style={{ color: "var(--text-muted)" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {list.map((p: Position) => {
                const m = managedMap.get(p.symbol);
                const alloc = (p.marketValue / totalEquity * 100);

                return (
                  <tr key={p.symbol} className="border-b last:border-0" style={{ borderColor: "var(--border)" }}>
                    <td className="px-4 py-3 font-bold font-mono">{p.symbol}</td>
                    <td className="px-4 py-3 font-mono">{p.quantity}</td>
                    <td className="px-4 py-3 font-mono">{formatCurrency(p.avgEntryPrice)}</td>
                    <td className="px-4 py-3 font-mono">{formatCurrency(p.currentPrice)}</td>
                    <td className="px-4 py-3 font-mono text-xs">
                      {m ? (
                        <span>
                          <span className="text-loss">{formatCurrency(m.stopLoss)}</span>
                          {" / "}
                          <span className="text-profit">{formatCurrency(m.takeProfit)}</span>
                        </span>
                      ) : (
                        <span style={{ color: "var(--text-muted)" }}>--</span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <div className={`font-mono font-medium ${plColor(p.unrealizedPL)}`}>
                        {p.unrealizedPL >= 0 ? "+" : ""}{formatCurrency(p.unrealizedPL)}
                      </div>
                      <div className={`text-xs font-mono ${plColor(p.unrealizedPLPercent)}`}>
                        {formatPercent(p.unrealizedPLPercent)}
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono text-xs">{alloc.toFixed(1)}%</td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => handleClose(p.symbol)}
                        className="p-1.5 rounded hover:bg-red-500/20 text-red-500 transition-colors"
                        title={`Close ${p.symbol}`}
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
