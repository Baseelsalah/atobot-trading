import express, { type Request, Response, NextFunction } from "express";
import { registerRoutes } from "./routes";
import { serveStatic } from "./static";
import { createServer } from "http";
import * as tradingBot from "./tradingBot";
import { startReportScheduler } from "./reportScheduler";
import * as runtimeMonitor from "./runtimeMonitor";
import * as activityLedger from "./activityLedger";
import * as tradeAccounting from "./tradeAccounting";
import * as reportStorage from "./reportStorage";
import * as envScopeModule from "./envScope";

// BOOT LOG: Durable startup log with pid and version
function logBoot(): void {
  const now = new Date();
  const utcTimestamp = now.toISOString();
  const ptFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Los_Angeles",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
  const etFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  const ptTimestamp = ptFormatter.format(now).replace(",", "");
  const etTime = etFormatter.format(now);
  const pid = process.pid;
  const version = envScopeModule.getVersion();
  
  // Durable boot line - single line for easy grep
  console.log(`[BOOT] bootTimeET=${etTime} pid=${pid} version=${version} timestamp_pt=${ptTimestamp} timestamp_utc=${utcTimestamp}`);
  
  // OPS-RUNBOOK-VERIFY-1: Log that runbook endpoint is available
  console.log(`[RUNBOOK] /debug/runbook enabled version=${version}`);
}

// Shutdown logging
function registerShutdownHandlers(): void {
  const logShutdown = (signal: string) => {
    const now = new Date();
    const etFormatter = new Intl.DateTimeFormat("en-US", {
      timeZone: "America/New_York",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
    const etTime = etFormatter.format(now);
    console.log(`[SHUTDOWN] signal=${signal} timeET=${etTime} pid=${process.pid}`);
  };
  
  process.on("SIGTERM", () => {
    tradeAccounting.persistOnShutdown();
    logShutdown("SIGTERM");
  });
  process.on("SIGINT", () => {
    tradeAccounting.persistOnShutdown();
    logShutdown("SIGINT");
  });
}

logBoot();

// ENV-SCOPE-HARDEN-1: Assert env scope consistency at startup
// If prod URL but envScope != prod, log CRITICAL and block trading
const envScopeCheck = envScopeModule.assertEnvScopeConsistency();
if (!envScopeCheck.ok) {
  console.error(`[STARTUP] ENV SCOPE MISMATCH DETECTED - TRADING WILL BE BLOCKED`);
}

tradeAccounting.initializeOnBoot();
registerShutdownHandlers();

const app = express();
const httpServer = createServer(app);

declare module "http" {
  interface IncomingMessage {
    rawBody: unknown;
  }
}

app.use(
  express.json({
    verify: (req, _res, buf) => {
      req.rawBody = buf;
    },
  }),
);

app.use(express.urlencoded({ extended: false }));

export function log(message: string, source = "express") {
  const formattedTime = new Date().toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });

  console.log(`${formattedTime} [${source}] ${message}`);
}

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      log(logLine);
    }
  });

  next();
});

(async () => {
  await registerRoutes(httpServer, app);

  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    res.status(status).json({ message });
    throw err;
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (process.env.NODE_ENV === "production") {
    serveStatic(app);
  } else {
    const { setupVite } = await import("./vite");
    await setupVite(httpServer, app);
  }

  // ALWAYS serve the app on the port specified in the environment variable PORT
  // Other ports are firewalled. Default to 5000 if not specified.
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = parseInt(process.env.PORT || "5000", 10);
  httpServer.listen(
    {
      port,
      host: "0.0.0.0",
      reusePort: true,
    },
    async () => {
      log(`serving on port ${port}`);
      
      // Initialize report storage FIRST (needed for leader lock during runtime init)
      reportStorage.initReportStorage();
      
      // Initialize runtime monitor (boot logging, heartbeat, stall detection, leader lock)
      await runtimeMonitor.initRuntimeMonitor();
      
      // Register tick callback so runtime monitor knows when scans happen
      // OPS-METRICS-1: Also increment ticksSinceBoot counter
      activityLedger.registerTickCallback((tickET) => {
        runtimeMonitor.notifyTick(tickET);
        runtimeMonitor.incrementTicksSinceBoot();
      });
      
      // Start the daily report scheduler (runs on boot + every 30 min)
      startReportScheduler();
      
      // Auto-start the trading bot (no manual intervention needed)
      try {
        await tradingBot.startBot();
        log("Trading bot auto-started successfully");
        
        // Register scan restart callback for stall recovery
        runtimeMonitor.registerScanRestartCallback(async () => {
          await tradingBot.restartScanLoop();
        });
      } catch (error) {
        log(`Trading bot auto-start failed: ${error}`);
      }
    },
  );
})();
