import { Switch, Route, Redirect, useLocation } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider, useQuery } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { PortfolioProvider } from "@/lib/portfolio-context";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Bell, AlertTriangle, CheckCircle2, Info } from "lucide-react";
import { PortfolioSwitcher } from "@/components/portfolio-switcher";
import { ThemeToggle } from "@/components/theme-toggle";
import { useState } from "react";

import Dashboard from "@/pages/dashboard";
import SettingsPage from "@/pages/settings";
import NotFound from "@/pages/not-found";

import type { BotStatus, Alert } from "@shared/schema";

function AlertsDrawer({ alerts, unreadCount }: { alerts: Alert[]; unreadCount: number }) {
  const [open, setOpen] = useState(false);

  const getAlertIcon = (type: string) => {
    switch (type) {
      case "critical":
        return <AlertTriangle className="w-4 h-4 text-red-500" />;
      case "warning":
        return <Info className="w-4 h-4 text-amber-500" />;
      default:
        return <CheckCircle2 className="w-4 h-4 text-emerald-500" />;
    }
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="relative"
          data-testid="button-alerts-bell"
        >
          <Bell className="w-4 h-4" />
          {unreadCount > 0 && (
            <Badge className="absolute -top-1 -right-1 h-4 min-w-4 px-1 text-[10px] bg-red-500 text-white border-0">
              {unreadCount > 9 ? "9+" : unreadCount}
            </Badge>
          )}
        </Button>
      </SheetTrigger>
      <SheetContent className="w-[380px] sm:w-[420px]">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <Bell className="w-4 h-4" />
            Alerts
            {unreadCount > 0 && (
              <Badge variant="secondary" className="text-xs">
                {unreadCount} new
              </Badge>
            )}
          </SheetTitle>
        </SheetHeader>
        <div className="mt-4 space-y-2 max-h-[calc(100vh-120px)] overflow-y-auto">
          {alerts.length === 0 ? (
            <div className="py-12 text-center text-muted-foreground">
              <Bell className="w-8 h-8 mx-auto mb-2 opacity-30" />
              <p className="text-sm">No alerts yet</p>
            </div>
          ) : (
            alerts.slice(0, 20).map((alert) => (
              <div
                key={alert.id}
                className={`p-3 rounded-md border ${
                  alert.isRead ? "border-border/50 bg-muted/30" : "border-accent-border bg-accent/20"
                }`}
              >
                <div className="flex items-start gap-2">
                  {getAlertIcon(alert.type)}
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm ${alert.isRead ? "text-muted-foreground" : "font-medium"}`}>
                      {alert.message}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {alert.timestamp ? new Date(alert.timestamp).toLocaleString() : "—"}
                    </p>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}

function AppContent() {
  const botStatusQuery = useQuery<BotStatus>({
    queryKey: ["/api/bot/status"],
    refetchInterval: 2000,
  });

  const alertsQuery = useQuery<Alert[]>({
    queryKey: ["/api/alerts"],
    refetchInterval: 5000,
  });

  const botStatus: BotStatus = botStatusQuery.data ?? {
    status: "stopped",
    lastAnalysis: null,
    currentAction: null,
    errorMessage: null,
  };

  const alerts = alertsQuery.data ?? [];
  const unreadAlerts = alerts.filter((a) => !a.isRead).length;

  const style = {
    "--sidebar-width": "16rem",
    "--sidebar-width-icon": "3.5rem",
  };

  return (
    <SidebarProvider style={style as React.CSSProperties} defaultOpen={false}>
      <div className="flex h-screen w-full">
        <AppSidebar botStatus={botStatus} />
        <div className="flex flex-col flex-1 min-w-0">
          <header className="flex items-center justify-between gap-2 px-3 sm:px-4 border-b border-border h-12 sm:h-14 flex-shrink-0 bg-background">
            <SidebarTrigger data-testid="button-sidebar-toggle" />
            <div className="flex items-center gap-1.5 sm:gap-2">
              <PortfolioSwitcher />
              <ThemeToggle />
              <AlertsDrawer alerts={alerts} unreadCount={unreadAlerts} />
            </div>
          </header>
          <main className="flex-1 overflow-auto bg-background scroll-container">
            <Switch>
              <Route path="/" component={Dashboard} />
              <Route path="/dashboard" component={Dashboard} />
              <Route path="/settings" component={SettingsPage} />
              
              <Route path="/scanner">{() => <Redirect to="/dashboard?tab=pnl" />}</Route>
              <Route path="/analytics">{() => <Redirect to="/dashboard?tab=pnl" />}</Route>
              <Route path="/trades">{() => <Redirect to="/dashboard?tab=orders" />}</Route>
              <Route path="/history">{() => <Redirect to="/dashboard?tab=history" />}</Route>
              <Route path="/alerts">{() => <Redirect to="/dashboard" />}</Route>
              <Route path="/positions">{() => <Redirect to="/dashboard?tab=assets" />}</Route>
              <Route path="/research">{() => <Redirect to="/dashboard" />}</Route>
              <Route path="/onboarding">{() => <Redirect to="/dashboard" />}</Route>
              
              <Route component={NotFound} />
            </Switch>
          </main>
        </div>
      </div>
    </SidebarProvider>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <PortfolioProvider>
          <Toaster />
          <AppContent />
        </PortfolioProvider>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
