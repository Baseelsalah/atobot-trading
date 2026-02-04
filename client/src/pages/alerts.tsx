import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { useState } from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AlertTriangle, Bell, CheckCircle, XCircle, Info, AlertOctagon } from "lucide-react";
import { AlertModal } from "@/components/alert-modal";
import type { Alert } from "@shared/schema";

function formatDateTime(timestamp: Date | string | null): string {
  if (!timestamp) return "";
  const date = typeof timestamp === "string" ? new Date(timestamp) : timestamp;
  return date.toLocaleString();
}

function getTypeIcon(type: string) {
  switch (type) {
    case "critical":
      return <AlertOctagon className="w-5 h-5 text-destructive" />;
    case "warning":
      return <AlertTriangle className="w-5 h-5 text-chart-4" />;
    case "info":
    default:
      return <Info className="w-5 h-5 text-chart-1" />;
  }
}

function getTypeVariant(type: string): "default" | "secondary" | "destructive" | "outline" {
  switch (type) {
    case "critical":
      return "destructive";
    case "warning":
      return "secondary";
    default:
      return "outline";
  }
}

export default function AlertsPage() {
  const { toast } = useToast();
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);

  const alertsQuery = useQuery<Alert[]>({
    queryKey: ["/api/alerts"],
    refetchInterval: 5000,
  });

  const markReadMutation = useMutation({
    mutationFn: (alertId: string) => apiRequest("POST", `/api/alerts/${alertId}/read`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/alerts"] });
    },
  });

  const approveAlertMutation = useMutation({
    mutationFn: (alertId: string) => apiRequest("POST", `/api/alerts/${alertId}/approve`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/alerts"] });
      queryClient.invalidateQueries({ queryKey: ["/api/trades"] });
      setSelectedAlert(null);
      toast({ title: "Trade approved" });
    },
  });

  const denyAlertMutation = useMutation({
    mutationFn: (alertId: string) => apiRequest("POST", `/api/alerts/${alertId}/deny`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/alerts"] });
      setSelectedAlert(null);
      toast({ title: "Trade denied" });
    },
  });

  const markAllReadMutation = useMutation({
    mutationFn: () => apiRequest("POST", "/api/alerts/mark-all-read"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/alerts"] });
      toast({ title: "All alerts marked as read" });
    },
  });

  const alerts = alertsQuery.data ?? [];
  const unreadCount = alerts.filter((a) => !a.isRead).length;
  const pendingApprovalCount = alerts.filter((a) => a.requiresApproval && !a.isResolved).length;
  const criticalCount = alerts.filter((a) => a.type === "critical" && !a.isResolved).length;

  return (
    <div className="p-6 max-w-screen-xl mx-auto space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold flex items-center gap-2">
            <Bell className="w-6 h-6" />
            Alerts
          </h1>
          <p className="text-muted-foreground">Notifications and trade approvals</p>
        </div>
        {unreadCount > 0 && (
          <Button
            variant="outline"
            onClick={() => markAllReadMutation.mutate()}
            disabled={markAllReadMutation.isPending}
            data-testid="button-mark-all-read"
          >
            Mark All as Read
          </Button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
              Unread Alerts
            </div>
            <div className="text-2xl font-bold font-mono">{unreadCount}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
              Pending Approval
            </div>
            <div className="text-2xl font-bold font-mono text-chart-4">{pendingApprovalCount}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">
              Critical
            </div>
            <div className="text-2xl font-bold font-mono text-destructive">{criticalCount}</div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">All Alerts</CardTitle>
        </CardHeader>
        <CardContent>
          {alertsQuery.isLoading ? (
            <div className="animate-pulse space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-20 bg-muted rounded-md" />
              ))}
            </div>
          ) : alerts.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Bell className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg">No alerts</p>
              <p className="text-sm">Critical events will appear here</p>
            </div>
          ) : (
            <ScrollArea className="h-[600px]">
              <div className="space-y-3 pr-4">
                {alerts.map((alert) => (
                  <Card
                    key={alert.id}
                    className={`${!alert.isRead ? "border-primary/50" : ""}`}
                    data-testid={`card-alert-${alert.id}`}
                  >
                    <CardContent className="pt-4 pb-4">
                      <div className="flex items-start gap-4">
                        <div className="flex-shrink-0 mt-1">
                          {getTypeIcon(alert.type)}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap mb-1">
                            <span className="font-semibold">{alert.title}</span>
                            <Badge variant={getTypeVariant(alert.type)} className="text-xs">
                              {alert.type}
                            </Badge>
                            {alert.requiresApproval && !alert.isResolved && (
                              <Badge variant="secondary" className="text-xs">
                                Needs Approval
                              </Badge>
                            )}
                            {alert.isResolved && (
                              <Badge variant="outline" className="text-xs">
                                Resolved
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground">{alert.message}</p>
                          <div className="flex items-center gap-4 mt-2 flex-wrap">
                            <span className="text-xs text-muted-foreground font-mono">
                              {formatDateTime(alert.timestamp)}
                            </span>
                            {alert.requiresApproval && !alert.isResolved && (
                              <div className="flex items-center gap-2">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => setSelectedAlert(alert)}
                                  data-testid={`button-view-alert-${alert.id}`}
                                >
                                  View Details
                                </Button>
                              </div>
                            )}
                            {!alert.isRead && (
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => markReadMutation.mutate(alert.id)}
                                data-testid={`button-mark-read-${alert.id}`}
                              >
                                Mark as Read
                              </Button>
                            )}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      <AlertModal
        alert={selectedAlert}
        onApprove={(id) => approveAlertMutation.mutate(id)}
        onDeny={(id) => denyAlertMutation.mutate(id)}
        onClose={() => setSelectedAlert(null)}
      />
    </div>
  );
}
