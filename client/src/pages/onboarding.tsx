import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CheckCircle, Circle, ArrowRight } from "lucide-react";
import { Link } from "wouter";

const steps = [
  { id: 1, title: "Connect Alpaca Account", done: true },
  { id: 2, title: "Configure Risk Settings", done: true },
  { id: 3, title: "Set Trading Universe", done: true },
  { id: 4, title: "Enable Paper Trading", done: true },
  { id: 5, title: "Review Dashboard", done: false },
];

export default function OnboardingPage() {
  const completedSteps = steps.filter((s) => s.done).length;

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Welcome to AtoBot</h1>
        <p className="text-muted-foreground">Complete setup to start trading</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Setup Progress</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {steps.map((step) => (
            <div key={step.id} className="flex items-center gap-3">
              {step.done ? (
                <CheckCircle className="w-5 h-5 text-green-500" />
              ) : (
                <Circle className="w-5 h-5 text-muted-foreground" />
              )}
              <span className={step.done ? "text-muted-foreground line-through" : ""}>
                {step.title}
              </span>
            </div>
          ))}
        </CardContent>
      </Card>

      <div className="flex justify-between items-center">
        <span className="text-sm text-muted-foreground">
          {completedSteps} of {steps.length} complete
        </span>
        <Link href="/">
          <Button data-testid="button-go-dashboard">
            Go to Dashboard
            <ArrowRight className="w-4 h-4 ml-2" />
          </Button>
        </Link>
      </div>
    </div>
  );
}
