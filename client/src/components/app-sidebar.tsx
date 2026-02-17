import { Link, useLocation } from "wouter";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
  SidebarFooter,
} from "@/components/ui/sidebar";
import { Badge } from "@/components/ui/badge";
import {
  LayoutDashboard,
  Settings,
  Zap,
} from "lucide-react";
import type { BotStatus } from "@shared/schema";

interface AppSidebarProps {
  botStatus: BotStatus;
}

const menuItems = [
  { title: "Terminal", url: "/", icon: LayoutDashboard },
  { title: "Settings", url: "/settings", icon: Settings },
];

function getStatusBadge(status: BotStatus["status"]) {
  const config: Record<
    BotStatus["status"],
    { label: string; className: string }
  > = {
    active: { label: "Active", className: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/25" },
    paused: { label: "Paused", className: "bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/25" },
    analyzing: { label: "Analyzing", className: "bg-muted text-foreground border-border" },
    error: { label: "Error", className: "bg-destructive/15 text-destructive border-destructive/25" },
    stopped: { label: "Stopped", className: "bg-muted text-muted-foreground border-border" },
  };
  const { label, className } = config[status];
  return (
    <Badge variant="outline" className={`text-xs font-medium ${className}`}>
      {status === "active" && <Zap className="w-3 h-3 mr-1" />}
      {label}
    </Badge>
  );
}

export function AppSidebar({ botStatus }: AppSidebarProps) {
  const [location] = useLocation();

  const isActive = (url: string) => {
    if (url === "/") {
      return location === "/" || location === "/dashboard" || location.startsWith("/dashboard?");
    }
    return location === url || location.startsWith(url + "?");
  };

  return (
    <Sidebar className="border-r border-border/50">
      <SidebarHeader className="p-4 border-b border-border/50">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="h-9 w-9 rounded-lg bg-primary flex items-center justify-center">
              <Zap className="h-5 w-5 text-primary-foreground" />
            </div>
            {botStatus.status === "active" && (
              <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-emerald-500 rounded-full border-2 border-sidebar animate-pulse" />
            )}
          </div>
          <div className="flex flex-col">
            <span className="font-semibold text-base uppercase tracking-wider text-foreground" style={{ fontFamily: "'Oxanium', sans-serif" }}>ATOBOT</span>
            <span className="text-xs text-muted-foreground">Day Trading Bot</span>
          </div>
        </div>
        <div className="mt-3 flex items-center gap-2">
          {getStatusBadge(botStatus.status)}
        </div>
      </SidebarHeader>

      <SidebarContent className="px-2 py-4">
        <SidebarGroup>
          <SidebarGroupLabel className="text-[10px] uppercase tracking-widest font-medium text-muted-foreground/70 px-2 mb-2">
            Navigation
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu className="space-y-1">
              {menuItems.map((item) => {
                const active = isActive(item.url);
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      isActive={active}
                      className={`py-2.5 px-3 rounded-md transition-all duration-200 ${
                        active 
                          ? 'bg-accent border-l-2 border-l-foreground/40 text-foreground font-medium' 
                          : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                      }`}
                      data-testid={`link-nav-${item.title.toLowerCase().replace(' ', '-')}`}
                    >
                      <Link href={item.url}>
                        <item.icon className={`w-4 h-4 ${active ? 'text-foreground' : ''}`} />
                        <span className="text-sm font-medium">{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="p-4 border-t border-border/50">
        <div className="text-[11px] text-muted-foreground/70">
          {botStatus.lastAnalysis ? (
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-foreground/30" />
              <span>Last scan: {new Date(botStatus.lastAnalysis).toLocaleTimeString()}</span>
            </div>
          ) : (
            <span>No analysis yet</span>
          )}
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
