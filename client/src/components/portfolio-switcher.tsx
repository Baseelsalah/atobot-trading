import { usePortfolio, type PortfolioKind } from "@/lib/portfolio-context";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDown, FileText, Zap } from "lucide-react";

export function PortfolioSwitcher() {
  const { kind, setKind } = usePortfolio();

  const options: { value: PortfolioKind; label: string; icon: typeof FileText }[] = [
    { value: "paper", label: "Paper", icon: FileText },
    { value: "live", label: "Live", icon: Zap },
  ];

  const current = options.find((o) => o.value === kind) || options[0];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button 
          variant="ghost" 
          size="sm" 
          className="h-8 px-2 gap-1 text-xs font-medium" 
          data-testid="button-portfolio-switcher"
        >
          <Badge 
            variant={kind === "live" ? "destructive" : "secondary"} 
            className="text-[10px] px-1.5 py-0"
          >
            {kind === "live" ? "LIVE" : "PAPER"}
          </Badge>
          <ChevronDown className="w-3 h-3 opacity-50" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {options.map((opt) => (
          <DropdownMenuItem
            key={opt.value}
            onClick={() => setKind(opt.value)}
            data-testid={`menu-item-${opt.value}`}
          >
            <opt.icon className="w-4 h-4 mr-2" />
            {opt.label} Trading
            {kind === opt.value && <span className="ml-2 text-muted-foreground">(active)</span>}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
