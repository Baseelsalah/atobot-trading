import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { motion } from "framer-motion";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Settings as SettingsIcon, Shield, Clock, Zap } from "lucide-react";
import type { BotSettings } from "@shared/schema";

const settingsFormSchema = z.object({
  isPaperTrading: z.boolean(),
  maxPositionSize: z.coerce.number().min(100).max(100000),
  maxDailyLoss: z.coerce.number().min(50).max(50000),
  maxPositions: z.coerce.number().min(1).max(20),
  stopLossPercent: z.coerce.number().min(0.5).max(20),
  takeProfitPercent: z.coerce.number().min(1).max(50),
  tradingHoursOnly: z.boolean(),
  analysisInterval: z.coerce.number().min(60).max(3600),
});

type SettingsFormValues = z.infer<typeof settingsFormSchema>;

export default function SettingsPage() {
  const { toast } = useToast();

  const settingsQuery = useQuery<BotSettings>({
    queryKey: ["/api/settings"],
  });

  const form = useForm<SettingsFormValues>({
    resolver: zodResolver(settingsFormSchema),
    defaultValues: {
      isPaperTrading: true,
      maxPositionSize: 1000,
      maxDailyLoss: 500,
      maxPositions: 5,
      stopLossPercent: 2,
      takeProfitPercent: 5,
      tradingHoursOnly: true,
      analysisInterval: 300,
    },
    values: settingsQuery.data ? {
      isPaperTrading: settingsQuery.data.isPaperTrading ?? true,
      maxPositionSize: settingsQuery.data.maxPositionSize ?? 1000,
      maxDailyLoss: settingsQuery.data.maxDailyLoss ?? 500,
      maxPositions: settingsQuery.data.maxPositions ?? 5,
      stopLossPercent: settingsQuery.data.stopLossPercent ?? 2,
      takeProfitPercent: settingsQuery.data.takeProfitPercent ?? 5,
      tradingHoursOnly: settingsQuery.data.tradingHoursOnly ?? true,
      analysisInterval: settingsQuery.data.analysisInterval ?? 300,
    } : undefined,
  });

  const updateSettingsMutation = useMutation({
    mutationFn: (values: SettingsFormValues) =>
      apiRequest("PATCH", "/api/settings", values),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/settings"] });
      toast({ title: "Settings saved", description: "Your trading settings have been updated" });
    },
    onError: () => {
      toast({ title: "Error", description: "Failed to save settings", variant: "destructive" });
    },
  });

  const onSubmit = (values: SettingsFormValues) => {
    updateSettingsMutation.mutate(values);
  };

  if (settingsQuery.isLoading) {
    return (
      <div className="p-6 max-w-screen-lg mx-auto">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-muted rounded w-1/4" />
          <div className="h-64 bg-muted rounded" />
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-screen-lg mx-auto space-y-6">
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-xl sm:text-2xl font-bold flex items-center gap-2 tracking-tight">
          <div className="p-2 rounded-lg bg-muted">
            <SettingsIcon className="w-5 h-5 text-foreground" />
          </div>
          Settings
        </h1>
        <p className="text-sm text-muted-foreground mt-1">Configure your AtoBot trading parameters</p>
      </motion.div>

      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <Card className="border-border/50">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <div className="p-1.5 rounded-md bg-muted">
                  <Zap className="w-4 h-4 text-foreground" />
                </div>
                Trading Mode
              </CardTitle>
              <CardDescription className="text-sm">
                Choose between paper trading (simulated) and live trading
              </CardDescription>
            </CardHeader>
            <CardContent>
              <FormField
                control={form.control}
                name="isPaperTrading"
                render={({ field }) => (
                  <FormItem className="flex items-center justify-between gap-4">
                    <div>
                      <FormLabel>Paper Trading Mode</FormLabel>
                      <FormDescription>
                        When enabled, trades are simulated and no real money is used
                      </FormDescription>
                    </div>
                    <FormControl>
                      <Switch
                        checked={field.value}
                        onCheckedChange={field.onChange}
                        data-testid="switch-paper-trading-settings"
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
            </CardContent>
          </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <Card className="border-border/50">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <div className="p-1.5 rounded-md bg-muted">
                  <Shield className="w-4 h-4 text-foreground" />
                </div>
                Risk Management
              </CardTitle>
              <CardDescription className="text-sm">
                Set limits to protect your capital
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <FormField
                  control={form.control}
                  name="maxPositionSize"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Max Position Size ($)</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          {...field}
                          data-testid="input-max-position"
                        />
                      </FormControl>
                      <FormDescription>
                        Maximum dollar amount per position
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="maxDailyLoss"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Max Daily Loss ($)</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          {...field}
                          data-testid="input-max-daily-loss"
                        />
                      </FormControl>
                      <FormDescription>
                        Stop trading if daily loss exceeds this
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="maxPositions"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Max Open Positions</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          {...field}
                          data-testid="input-max-positions"
                        />
                      </FormControl>
                      <FormDescription>
                        Maximum concurrent positions
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="stopLossPercent"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Stop Loss (%)</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          step="0.5"
                          {...field}
                          data-testid="input-stop-loss"
                        />
                      </FormControl>
                      <FormDescription>
                        Auto-sell if position drops by this %
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="takeProfitPercent"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Take Profit (%)</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          step="0.5"
                          {...field}
                          data-testid="input-take-profit"
                        />
                      </FormControl>
                      <FormDescription>
                        Auto-sell if position gains this %
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </CardContent>
          </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
          <Card className="border-border/50">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <div className="p-1.5 rounded-md bg-muted">
                  <Clock className="w-4 h-4 text-foreground" />
                </div>
                Trading Schedule
              </CardTitle>
              <CardDescription className="text-sm">
                Control when the bot trades
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <FormField
                control={form.control}
                name="tradingHoursOnly"
                render={({ field }) => (
                  <FormItem className="flex items-center justify-between gap-4">
                    <div>
                      <FormLabel>Trading Hours Only</FormLabel>
                      <FormDescription>
                        Only trade during market hours (9:30 AM - 4:00 PM ET)
                      </FormDescription>
                    </div>
                    <FormControl>
                      <Switch
                        checked={field.value}
                        onCheckedChange={field.onChange}
                        data-testid="switch-trading-hours"
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="analysisInterval"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Analysis Interval (seconds)</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        {...field}
                        data-testid="input-analysis-interval"
                      />
                    </FormControl>
                    <FormDescription>
                      How often GPT analyzes the market (60-3600 seconds)
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </CardContent>
          </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
          <Button
            type="submit"
            disabled={updateSettingsMutation.isPending}
            className="w-full"
            data-testid="button-save-settings"
          >
            {updateSettingsMutation.isPending ? "Saving..." : "Save Settings"}
          </Button>
          </motion.div>
        </form>
      </Form>
    </div>
  );
}
