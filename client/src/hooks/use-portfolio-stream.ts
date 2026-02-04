import { useState, useEffect, useRef } from "react";
import { usePortfolio } from "@/lib/portfolio-context";

interface AccountData {
  totalEquity: number;
  buyingPower: number;
  cash: number;
  todayPL: number;
  todayPLPercent: number;
  dayTradesRemaining?: number;
}

interface PositionData {
  symbol: string;
  qty: number;
  marketValue: number;
  unrealizedPL: number;
  unrealizedPLPercent: number;
  currentPrice: number;
  avgEntryPrice: number;
}

interface BotStatusData {
  status: string;
  lastAnalysis: string | null;
  currentAction: string | null;
}

interface StreamState {
  account: AccountData | null;
  positions: PositionData[];
  botStatus: BotStatusData | null;
  connected: boolean;
  lastUpdate: Date | null;
}

export function usePortfolioStream() {
  const { kind } = usePortfolio();
  const [state, setState] = useState<StreamState>({
    account: null,
    positions: [],
    botStatus: null,
    connected: false,
    lastUpdate: null,
  });
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    const url = `/api/portfolio/${kind}/stream`;
    console.log(`[SSE] Connecting to ${url}`);
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onopen = () => {
      console.log("[SSE] Connected");
      setState((s) => ({ ...s, connected: true }));
    };

    es.addEventListener("account", (e) => {
      const data = JSON.parse(e.data) as AccountData;
      console.log("[SSE] account event:", data);
      setState((s) => ({ ...s, account: data, lastUpdate: new Date() }));
    });

    es.addEventListener("positions", (e) => {
      const data = JSON.parse(e.data) as PositionData[];
      console.log("[SSE] positions event:", data.length, "positions");
      setState((s) => ({ ...s, positions: data, lastUpdate: new Date() }));
    });

    es.addEventListener("botStatus", (e) => {
      const data = JSON.parse(e.data) as BotStatusData;
      console.log("[SSE] botStatus event:", data);
      setState((s) => ({ ...s, botStatus: data, lastUpdate: new Date() }));
    });

    es.onerror = () => {
      console.log("[SSE] Error or disconnected");
      setState((s) => ({ ...s, connected: false }));
    };

    return () => {
      console.log("[SSE] Closing connection");
      es.close();
    };
  }, [kind]);

  return state;
}
