import { useState, useEffect, useCallback, useRef } from "react";
import * as api from "./api";
import type { AuthUser } from "./api";

// ── Auth Hook ─────────────────────────────────────────

export function useAuth() {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [token, setToken] = useState<string | null>(api.getAuthToken());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const existing = api.getAuthToken();
    if (!existing) {
      setLoading(false);
      return;
    }
    api.fetchMe()
      .then((me) => {
        setUser(me);
        setToken(existing);
      })
      .catch(() => {
        api.setAuthToken(null);
        setToken(null);
      })
      .finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const result = await api.login(email, password);
    api.setAuthToken(result.token);
    setToken(result.token);
    setUser(result.user);
    return result;
  }, []);

  const registerUser = useCallback(async (email: string, password: string, displayName: string) => {
    const result = await api.register(email, password, displayName);
    api.setAuthToken(result.token);
    setToken(result.token);
    setUser(result.user);
    return result;
  }, []);

  const logout = useCallback(() => {
    api.setAuthToken(null);
    setToken(null);
    setUser(null);
  }, []);

  return { user, token, loading, login, registerUser, logout };
}

export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number
): { data: T | null; error: string | null; loading: boolean; refetch: () => void; lastUpdated: number | null } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const mountedRef = useRef(true);

  const doFetch = useCallback(async () => {
    try {
      const result = await fetcher();
      if (mountedRef.current) {
        setData(result);
        setError(null);
        setLastUpdated(Date.now());
      }
    } catch (e) {
      if (mountedRef.current) {
        setError(e instanceof Error ? e.message : "Unknown error");
      }
    } finally {
      if (mountedRef.current) setLoading(false);
    }
  }, [fetcher]);

  useEffect(() => {
    mountedRef.current = true;
    doFetch();
    const id = setInterval(doFetch, intervalMs);
    return () => {
      mountedRef.current = false;
      clearInterval(id);
    };
  }, [doFetch, intervalMs]);

  return { data, error, loading, refetch: doFetch, lastUpdated };
}

export function useTheme() {
  const [dark, setDark] = useState(() => {
    if (typeof window === "undefined") return true;
    const saved = localStorage.getItem("atobot-theme");
    return saved ? saved === "dark" : true;
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("atobot-theme", dark ? "dark" : "light");
  }, [dark]);

  return { dark, toggle: () => setDark((d) => !d) };
}

// ── Toast System ──────────────────────────────────────

export interface Toast {
  id: string;
  message: string;
  type: "success" | "error" | "info" | "warning";
}

let toastId = 0;
const listeners: Set<(toasts: Toast[]) => void> = new Set();
let toasts: Toast[] = [];

function notify() {
  listeners.forEach((fn) => fn([...toasts]));
}

export function addToast(message: string, type: Toast["type"] = "info") {
  const id = String(++toastId);
  toasts = [...toasts, { id, message, type }];
  notify();
  setTimeout(() => {
    toasts = toasts.filter((t) => t.id !== id);
    notify();
  }, 4000);
}

export function useToasts(): Toast[] {
  const [state, setState] = useState<Toast[]>([]);
  useEffect(() => {
    listeners.add(setState);
    return () => { listeners.delete(setState); };
  }, []);
  return state;
}

// ── Format Helpers ────────────────────────────────────

export function formatCurrency(n: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
  }).format(n);
}

export function formatPercent(n: number): string {
  return `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`;
}

export function formatTime(iso: string | null): string {
  if (!iso) return "--";
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function plColor(n: number): string {
  if (n > 0) return "text-profit";
  if (n < 0) return "text-loss";
  return "";
}

export function timeAgo(ts: number | null): string {
  if (!ts) return "";
  const seconds = Math.floor((Date.now() - ts) / 1000);
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  return `${Math.floor(seconds / 60)}m ago`;
}
