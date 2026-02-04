import { cn } from "@/lib/utils";

interface PageContainerProps {
  children: React.ReactNode;
  className?: string;
}

export function PageContainer({ children, className }: PageContainerProps) {
  return (
    <div className={cn("max-w-[1200px] mx-auto px-3 sm:px-6 lg:px-8 py-3 sm:py-6", className)}>
      {children}
    </div>
  );
}

interface PageGridProps {
  children: React.ReactNode;
  className?: string;
}

export function PageGrid({ children, className }: PageGridProps) {
  return (
    <div className={cn("grid grid-cols-1 gap-4 lg:grid-cols-12 lg:gap-6", className)}>
      {children}
    </div>
  );
}

interface GridCellProps {
  children: React.ReactNode;
  span?: 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12;
  className?: string;
}

export function GridCell({ children, span = 12, className }: GridCellProps) {
  const spanClasses: Record<number, string> = {
    1: "lg:col-span-1",
    2: "lg:col-span-2",
    3: "lg:col-span-3",
    4: "lg:col-span-4",
    5: "lg:col-span-5",
    6: "lg:col-span-6",
    7: "lg:col-span-7",
    8: "lg:col-span-8",
    9: "lg:col-span-9",
    10: "lg:col-span-10",
    11: "lg:col-span-11",
    12: "lg:col-span-12",
  };
  
  return (
    <div className={cn(spanClasses[span], className)}>
      {children}
    </div>
  );
}

export function PageHeader({ 
  title, 
  subtitle, 
  icon: Icon, 
  children 
}: { 
  title: string; 
  subtitle?: string; 
  icon?: React.ComponentType<{ className?: string }>;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
      <div className="flex items-center gap-3">
        {Icon && (
          <div className="p-2 rounded-lg bg-muted">
            <Icon className="w-5 h-5 text-foreground" />
          </div>
        )}
        <div>
          <h1 className="text-xl sm:text-2xl font-bold tracking-tight">{title}</h1>
          {subtitle && <p className="text-sm text-muted-foreground">{subtitle}</p>}
        </div>
      </div>
      {children && <div className="flex items-center gap-2">{children}</div>}
    </div>
  );
}
