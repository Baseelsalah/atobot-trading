import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageContainer } from "@/components/layout";
import { Search, Construction } from "lucide-react";
import { motion } from "framer-motion";

export default function ScannerPage() {
  return (
    <PageContainer className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-xl sm:text-2xl font-bold flex items-center gap-2 tracking-tight">
          <div className="p-2 rounded-lg bg-muted">
            <Search className="w-5 h-5 text-foreground" />
          </div>
          Scanner
        </h1>
        <p className="text-sm text-muted-foreground mt-1">Market scanner and opportunity detection</p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="border-border/50">
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <Construction className="w-4 h-4" />
              Coming Soon
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="py-12 text-center text-muted-foreground">
              <Search className="w-12 h-12 mx-auto mb-4 opacity-30" />
              <p className="text-sm">Market scanner is under development</p>
              <p className="text-xs mt-2 opacity-70">
                This feature will allow scanning for trading opportunities based on custom filters.
              </p>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </PageContainer>
  );
}
