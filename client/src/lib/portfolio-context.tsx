import { createContext, useContext, useState, useEffect, type ReactNode } from "react";

export type PortfolioKind = "paper" | "live";

interface PortfolioContextType {
  kind: PortfolioKind;
  setKind: (kind: PortfolioKind) => void;
  apiPrefix: string;
}

const PortfolioContext = createContext<PortfolioContextType | null>(null);

export function PortfolioProvider({ children }: { children: ReactNode }) {
  const [kind, setKindState] = useState<PortfolioKind>(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem("portfolioKind");
      if (stored === "live" || stored === "paper") return stored;
    }
    return "paper";
  });

  const setKind = (newKind: PortfolioKind) => {
    setKindState(newKind);
    localStorage.setItem("portfolioKind", newKind);
  };

  useEffect(() => {
    localStorage.setItem("portfolioKind", kind);
  }, [kind]);

  const apiPrefix = `/api/portfolio/${kind}`;

  return (
    <PortfolioContext.Provider value={{ kind, setKind, apiPrefix }}>
      {children}
    </PortfolioContext.Provider>
  );
}

export function usePortfolio() {
  const ctx = useContext(PortfolioContext);
  if (!ctx) throw new Error("usePortfolio must be used within PortfolioProvider");
  return ctx;
}
