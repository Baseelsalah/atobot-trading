import { storage } from "./storage";
import * as alpaca from "./alpaca";
import * as openaiService from "./openai";
import * as brain from "./autopilotBrain";
import * as ato from "./ato";
import * as riskManager from "./riskManager";
import * as profitManager from "./profitManager";
import * as timeGuard from "./tradingTimeGuard";
import * as positionManager from "./positionManager";
import * as marketRegime from "./marketRegime";
import * as dayTraderConfig from "./dayTraderConfig";
import * as skipCounters from "./skipCounters";
import * as signalCounters from "./signalCounters";
import * as autoTestReporter from "./autoTestReporter";
import * as dataHealthCounters from "./dataHealthCounters";
export { dataHealthCounters };
import * as indicatorPipeline from "./indicatorPipeline";
import { getEasternTime, toEasternDateString, isSimTimeActive, getSimTimeString } from "./timezone";
import { generateTradeId } from "./tradeId";
import * as preflight from "./preflight";
import * as tradabilityGates from "./tradabilityGates";
import * as regimeFilter from "./regimeFilter";
import * as tradeLifecycle from "./tradeLifecycle";
import * as riskEngine from "./riskEngine";
import * as strategyEngine from "./strategyEngine";
import * as executionQuality from "./executionQuality";
import * as activityLedger from "./activityLedger";
import * as eodManager from "./eodManager";
import * as tradeAccounting from "./tradeAccounting";
import * as entryWindowProof from "./entryWindowProof";
import * as executionTrace from "./executionTrace";
import * as regimeState from "./regimeState";
import * as controlLoopTrace from "./controlLoopTrace";
import type { BotStatus, TradeRecommendation } from "@shared/schema";

// P5: Strategy-based signal generation (deterministic, no LLM dependency)
// When enabled, generates signals from VWAP_REVERSION and ORB strategies
const USE_STRATEGY_ENGINE = true;  // P5: Enable deterministic strategies

// Watchlist for day trading - RESTRICTED to allowed universe only
import { DAY_TRADER_CONFIG } from "./dayTraderConfig";
const ALLOWED_UNIVERSE = DAY_TRADER_CONFIG.ALLOWED_SYMBOLS; // SPY, QQQ, SH only

// TIER-BASED STRATEGY GATING: Explicit registry of strategies requiring MACD (Tier 2)
// This replaces fragile string-matching heuristics with explicit metadata
const MACD_REQUIRING_STRATEGIES = new Set([
  "scalp momentum",
  "momentum reversal",
  "macd cross",
  "macd divergence",
  "zerolag cross",
  "momentum breakout",
  "trend momentum",
]);

/**
 * Check if a strategy requires MACD indicators (Tier 2)
 * Uses explicit registry + fallback keyword matching for unknown strategies
 */
function strategyRequiresMacd(strategyName: string): boolean {
  const normalized = strategyName.toLowerCase().trim();
  
  // First check explicit registry
  if (MACD_REQUIRING_STRATEGIES.has(normalized)) {
    return true;
  }
  
  // Fallback: Check for MACD/momentum keywords (catches any new strategies)
  if (normalized.includes("macd") || normalized.includes("momentum")) {
    return true;
  }
  
  return false;
}

let botStatus: BotStatus = {
  status: "stopped",
  lastAnalysis: null,
  currentAction: null,
  errorMessage: null,
};

let analysisInterval: NodeJS.Timeout | null = null;
let brainCycleCounter = 0;

export function getBotStatus(): BotStatus {
  return { ...botStatus };
}

export async function startBot(): Promise<void> {
  if (botStatus.status === "active" || botStatus.status === "analyzing") {
    return;
  }

  if (!alpaca.isConfigured()) {
    botStatus = {
      status: "error",
      lastAnalysis: null,
      currentAction: null,
      errorMessage: "Alpaca API keys not configured",
    };
    throw new Error("Alpaca API keys not configured");
  }

  const settings = await storage.getSettings();

  // Run preflight check BEFORE starting trading
  const preflightResult = await preflight.runPreflightCheck();
  if (!preflight.preflightPassed(preflightResult)) {
    botStatus = {
      status: "error",
      lastAnalysis: null,
      currentAction: null,
      errorMessage: "Preflight check failed - trading blocked",
    };
    throw new Error("Preflight check failed - see logs for details");
  }

  botStatus = {
    status: "active",
    lastAnalysis: null,
    currentAction: "Initializing...",
    errorMessage: null,
  };

  await storage.createActivityLog({
    type: "system",
    action: "Bot Started",
    description: `AtoBot started in ${settings.isPaperTrading ? "paper" : "live"} trading mode`,
  });

  // Baseline readiness log - one-time startup confirmation
  const isDryRun = process.env.DRY_RUN === "1";
  const isSimulated = !!(process.env.SIM_CLOCK_OPEN || process.env.SIM_CLOCK_FAIL);
  const universe = dayTraderConfig.getEntryUniverse();
  console.log(`[BASELINE READY] paper=${settings.isPaperTrading} dry_run=${isDryRun} sim=${isSimulated} universe=[${universe.join(",")}]`);

  // Check for new trading day and reset daily tracking
  profitManager.checkAndResetDaily();
  dayTraderConfig.resetDaily();
  positionManager.resetDaily();
  marketRegime.clearCache();
  executionQuality.resetDailyMetrics();  // P6: Reset execution quality metrics
  
  // Initialize Entry Window Proof Bundle for today
  const etDate = toEasternDateString(new Date());
  entryWindowProof.initProofBundle(etDate);
  
  // P3: Initialize risk engine (registers callback, rehydrates from completed trades)
  riskEngine.initializeRiskEngine();
  
  // Rehydrate managed positions from existing Alpaca positions
  await positionManager.rehydrateFromAlpaca();
  
  // Rehydrate day trader config from today's trades only (filter by current Eastern date)
  const allTrades = await storage.getTrades();
  const today = getEasternTime().dateString; // YYYY-MM-DD in Eastern Time
  const todaysTrades = allTrades.filter(t => {
    if (!t.timestamp) return false;
    const tradeDate = toEasternDateString(new Date(t.timestamp));
    return tradeDate === today;
  });
  
  if (todaysTrades.length > 0) {
    await dayTraderConfig.rehydrateFromTrades(todaysTrades.map(t => ({
      side: t.side,
      realizedPL: 0 // P/L tracked separately by profitManager
    })));
  }
  
  // Initialize the autopilot brain (the intelligence)
  await brain.initializeBrain();
  
  await storage.createActivityLog({
    type: "system",
    action: "Autopilot Initialized",
    description: "Autopilot brain loaded strategies and market insights",
  });

  // Initialize Ato (the day trader executor)
  await ato.initializeAto();
  
  await storage.createActivityLog({
    type: "system",
    action: "Ato Initialized",
    description: "Ato day trader module ready to execute trades",
  });

  // Sync positions from Alpaca
  await syncPositions();

  // Start the time guard - FORT KNOX: entry 9:35-11:35 AM, close 3:45 PM ET
  await timeGuard.startTimeGuard();
  
  // Start position manager for trailing stops and partial profits
  positionManager.startPositionMonitor();
  
  // P2: Start trade lifecycle reconciliation loop (syncs with Alpaca every 30s)
  tradeLifecycle.startReconciliationLoop();
  
  // Start EOD Manager for calendar-aware flattening and overnight watchdog
  eodManager.startEODManager();
  
  // DAY TRADING: Use config-driven scan interval
  const DAY_TRADING_INTERVAL_MS = dayTraderConfig.DAY_TRADER_CONFIG.SCAN_INTERVAL_MINUTES * 60 * 1000;
  
  console.log(`[Bot] Starting day trading mode with ${dayTraderConfig.DAY_TRADER_CONFIG.SCAN_INTERVAL_MINUTES}-minute analysis cycles`);

  // Start the self-scheduling control loop (replaces fixed interval)
  startSelfSchedulingLoop();
  
  // Check for market open every 30 seconds to catch the opening bell
  startMarketOpenWatcher();
}

export async function pauseBot(): Promise<void> {
  if (analysisInterval) {
    clearInterval(analysisInterval);
    analysisInterval = null;
  }

  botStatus = {
    ...botStatus,
    status: "paused",
    currentAction: null,
  };

  await storage.createActivityLog({
    type: "system",
    action: "Bot Paused",
    description: "AtoBot has been paused",
  });
}

export async function stopBot(): Promise<void> {
  if (analysisInterval) {
    clearInterval(analysisInterval);
    analysisInterval = null;
  }

  // Stop the self-scheduling control loop
  stopSelfSchedulingLoop();
  
  // Stop the time guard
  timeGuard.stopTimeGuard();
  
  // Stop position manager
  positionManager.stopPositionMonitor();
  
  // Stop market open watcher
  stopMarketOpenWatcher();
  
  // Stop EOD Manager
  eodManager.stopEODManager();

  botStatus = {
    status: "stopped",
    lastAnalysis: null,
    currentAction: null,
    errorMessage: null,
  };

  await storage.createActivityLog({
    type: "system",
    action: "Bot Stopped",
    description: "AtoBot has been stopped",
  });
}

export async function runAnalysis(): Promise<void> {
  await runAnalysisCycle();
}

async function runAnalysisCycle(): Promise<void> {
  if (botStatus.status !== "active" && botStatus.status !== "paused") {
    return;
  }

  // Mark analysis start for coordination with report generation
  dataHealthCounters.markAnalysisStart();

  const previousStatus = botStatus.status;
  botStatus = {
    ...botStatus,
    status: "analyzing",
    currentAction: "Analyzing market conditions...",
  };

  // Get current state for funnel snapshot
  const et = getEasternTime();
  const clock = await alpaca.getClock().catch(() => ({ is_open: false }));
  const guardStatus = timeGuard.getTimeGuardStatus();
  const positions = await storage.getPositions();
  const dayStatus = dayTraderConfig.getDayTraderStatus();
  
  // FUNNEL SNAPSHOT: Log at start of each analysis tick
  console.log(`[ANALYSIS] Tick @ ${et.displayTime} | marketOpen=${clock.is_open} | canOpenNewTrades=${guardStatus.canOpenNewTrades} | positions=${positions.length} | entriesToday=${dayStatus.newEntriesToday}`);
  
  // CONTROL-LOOP-TRACE-1: Record analysis run for observability
  controlLoopTrace.recordAnalysisRun();
  
  // Log time status with dynamic cutoffs every scan cycle
  timeGuard.logTimeStatus();
  
  // MARKET_CLOCK and TRADING_ALLOWED logs for user visibility
  const nextClose = (clock as any).next_close ? new Date((clock as any).next_close).toISOString() : "unknown";
  console.log(`MARKET_CLOCK is_open=${clock.is_open} next_close=${nextClose}`);
  
  const tradingAllowed = clock.is_open && guardStatus.canManagePositions;
  const tradingReason = !clock.is_open ? "MARKET_CLOSED" : guardStatus.reason;
  console.log(`TRADING_ALLOWED=${tradingAllowed} reason=${tradingReason}`);
  
  // Track scan count for EOD summary
  signalCounters.recordScan();

  try {
    const settings = await storage.getSettings();

    // CRITICAL: Check time guard first - this is the absolute trading cutoff
    const tradingStatus = timeGuard.getTradingStatus();
    if (!tradingStatus.canTrade) {
      console.log(`[BLOCKED] guardrail=time_guard reason=${tradingStatus.reason}`);
      controlLoopTrace.recordAnalysisSkip(`time_guard:${tradingStatus.reason}`);
      controlLoopTrace.setTradingStateReason(tradingStatus.reason);
      botStatus = {
        ...botStatus,
        status: previousStatus,
        currentAction: tradingStatus.reason,
        lastAnalysis: new Date().toISOString(),
      };
      return;
    }
    
    // Check if we're past entry cutoff (11:35 AM ET) - can only manage existing positions
    const canEnterNewPositions = tradingStatus.canEnterNewPositions;
    if (!canEnterNewPositions) {
      console.log(`[BLOCKED] guardrail=entry_cutoff reason=${tradingStatus.reason}`);
    }
    
    // Check daily P&L kill threshold (-$500 or +$500)
    if (dayTraderConfig.isDailyKillThresholdHit()) {
      const status = dayTraderConfig.getDayTraderStatus();
      const killType = status.lossLimitHit ? "daily_max_loss" : "daily_profit_target";
      const reason = status.lossLimitHit 
        ? `Daily loss limit hit ($${Math.abs(status.dailyPnL).toFixed(0)} loss)`
        : `Daily profit target hit ($${status.dailyPnL.toFixed(0)} profit)`;
      console.log(`[BLOCKED] guardrail=${killType} current=$${status.dailyPnL.toFixed(0)} limit=$500`);
      botStatus = {
        ...botStatus,
        status: "active",
        currentAction: `${reason} - no new entries`,
        lastAnalysis: new Date().toISOString(),
      };
      // Continue to manage existing positions but don't enter new ones
    }

    // Check if within trading hours (secondary check using Alpaca clock)
    if (settings.tradingHoursOnly) {
      const clock = await alpaca.getClock();
      if (!clock.is_open) {
        botStatus = {
          ...botStatus,
          status: previousStatus,
          currentAction: "Market closed - waiting for open",
          lastAnalysis: new Date().toISOString(),
        };
        await storage.createResearchLog({
          type: "analysis",
          summary: "Market is currently closed. Waiting for market hours.",
          confidence: 100,
        });
        return;
      }
    }

    // Get account info
    const account = await alpaca.getAccount();
    const portfolioValue = parseFloat(account.equity);
    const buyingPower = parseFloat(account.buying_power);

    // Get current positions
    const positions = await storage.getPositions();
    const positionsForAnalysis = positions.map((p) => ({
      symbol: p.symbol,
      qty: p.quantity,
      unrealizedPL: p.unrealizedPL,
    }));

    // Check daily loss limit
    const todayPL = positions.reduce((sum, p) => sum + p.unrealizedPL, 0);
    if (todayPL < -(settings.maxDailyLoss || 500)) {
      await storage.createAlert({
        type: "critical",
        title: "Daily Loss Limit Reached",
        message: `Daily loss of $${Math.abs(todayPL).toFixed(2)} exceeds limit of $${settings.maxDailyLoss}. Trading paused.`,
        requiresApproval: false,
      });
      await pauseBot();
      return;
    }

    botStatus.currentAction = "Checking day trading limits...";

    // Check if Ato should stop trading today (daily limits) - BEFORE market regime
    if (await ato.shouldStopTradingToday()) {
      await storage.createAlert({
        type: "warning",
        title: "Day Trading Limits Reached",
        message: "Ato has reached daily trading limits. Monitoring only until tomorrow.",
        requiresApproval: false,
      });
      
      botStatus = {
        status: "paused",
        lastAnalysis: new Date().toISOString(),
        currentAction: "Daily limits reached - monitoring only",
        errorMessage: null,
      };
      return;
    }
    
    // Check profit goal status - stop if goal met or trading not allowed
    const tradingCheck = await profitManager.shouldContinueTrading();
    if (!tradingCheck.allowed) {
      console.log(`[Bot] Trading stopped: ${tradingCheck.reason}`);
      
      const goalState = await profitManager.getProfitGoalState();
      if (goalState.goalMet) {
        await storage.createAlert({
          type: "info",
          title: "DAILY PROFIT GOAL ACHIEVED!",
          message: `Target: $${goalState.dailyGoal} | Actual: $${goalState.currentProfit.toFixed(2)} | Great trading day!`,
          requiresApproval: false,
        });
      }
      
      botStatus = {
        status: "paused",
        lastAnalysis: new Date().toISOString(),
        currentAction: tradingCheck.reason,
        errorMessage: null,
      };
      return;
    }

    // REGIME FILTER v1: SPY EMA9 vs EMA21 trend filter (FAIL-CLOSED)
    // Must pass BEFORE any trading decisions are made
    const regimeResult = await regimeFilter.evaluateMarketRegime();
    if (!regimeResult.ok) {
      console.log(`ACTION=SKIP symbol=ALL SKIP_REASONS=[${regimeResult.reasons.join(",")}]`);
      controlLoopTrace.recordAnalysisSkip(`regime_filter:${regimeResult.reasons[0] || "unknown"}`);
      skipCounters.recordSkip("CHOP_REGIME");
      signalCounters.recordCandidate();
      botStatus = {
        ...botStatus,
        status: "active",
        currentAction: `Market regime filter blocked: ${regimeResult.reasons[0] || "unknown"}`,
        lastAnalysis: new Date().toISOString(),
      };
      return;
    }
    
    // Check market regime (QQQ/SPY trend filter) - AFTER daily limits
    // REGIME-BLOCK-ENTRIES-ONLY-1: Store regime state but continue evaluation pipeline
    // Block at precheck instead of early return to preserve receipts and learning data
    const regime = await marketRegime.checkMarketRegime();
    console.log(`[Bot] Market Regime: ${regime.recommendation} (QQQ: ${regime.qqq.trend}, SPY: ${regime.spy.trend})`);
    
    // Store regime state for precheck blocking
    regimeState.updateRegimeState(regime.recommendation, regime.qqq.trend, regime.spy.trend);
    
    if (regime.recommendation === "avoid") {
      console.log(`[REGIME-BLOCK-ENTRIES-ONLY-1] regime=avoid - continuing pipeline, will block at precheck`);
      skipCounters.recordSkip("CHOP_REGIME");
      signalCounters.recordCandidate();
      botStatus = {
        ...botStatus,
        status: "active",
        currentAction: "Market bearish - evaluating but blocking entries",
        lastAnalysis: new Date().toISOString(),
      };
      // REGIME-BLOCK-ENTRIES-ONLY-1: Do NOT return early - continue evaluation pipeline
    }

    botStatus.currentAction = "Autopilot analyzing market conditions...";

    // Run autopilot brain cycle every 5th analysis to learn and improve
    if (brainCycleCounter % 5 === 0) {
      console.log("[Autopilot] Running brain cycle - research, learn, improve...");
      await brain.runBrainCycle();
      
      // Autopilot guides Ato's trading style based on performance
      const styleGuidance = await brain.guideAtoStyle();
      ato.updateTradingStyle(styleGuidance);
      
      // Autopilot evaluates Ato for potential upgrades
      await brain.evaluateAndUpgrade();
    }
    brainCycleCounter++;

    // Ato reads the market like a day trader
    botStatus.currentAction = "Ato reading market conditions...";
    
    // Log current profit goal progress
    const goalProgress = await profitManager.getProfitGoalState();
    console.log(`[Bot] Profit Goal: $${goalProgress.currentProfit.toFixed(2)}/$${goalProgress.dailyGoal} (${goalProgress.progressPercent.toFixed(1)}%)`);
    console.log(`[Bot] Performance: Win Rate ${goalProgress.winRate.toFixed(1)}% | Expectancy $${goalProgress.expectancy.toFixed(2)}`);
    
    botStatus.currentAction = `Trading - Goal: ${goalProgress.progressPercent.toFixed(0)}% ($${goalProgress.currentProfit.toFixed(0)}/$${goalProgress.dailyGoal})`;

    // Run GPT analysis with DAY TRADING parameters
    // Key: Risk/Reward must favor profitability (1:2+ ratio)
    // 1% stop loss, 2.5% take profit = 2.5:1 risk/reward
    // This means we only need 30% win rate to break even
    const dayTradingStopLoss = 1.0; // Tight 1% stop loss - cut losses fast
    const dayTradingTakeProfit = 2.5; // 2.5% profit target - let winners run
    
    const analysis = await openaiService.analyzeMarket(
      ALLOWED_UNIVERSE, // SPY, QQQ, SH only
      portfolioValue,
      positionsForAnalysis,
      {
        maxPositionSize: settings.maxPositionSize || 1000,
        stopLossPercent: dayTradingStopLoss,
        takeProfitPercent: dayTradingTakeProfit,
      }
    );

    // Log the analysis
    const technicalIndicators = analysis.technicalIndicators || [];
    const newsFactors = analysis.newsFactors || [];
    await storage.createResearchLog({
      type: "analysis",
      summary: analysis.summary || "Market analysis completed",
      details: `Market sentiment: ${analysis.sentiment || "neutral"}. Technical indicators: ${technicalIndicators.length > 0 ? technicalIndicators.join(", ") : "N/A"}`,
      confidence: analysis.confidence || 50,
      sources: JSON.stringify(newsFactors),
    });

    // Process recommendations - AUTO-EXECUTE trades
    // Use brain to filter and adjust recommendation confidence
    const rawRecommendations = analysis.recommendations || [];
    const llmRecommendations = brain.filterRecommendations(rawRecommendations);
    const MIN_CONFIDENCE_THRESHOLD = 50; // Minimum confidence to execute a trade
    
    // FAIL-CLOSED: If no recommendations or only "hold" actions, log ACTION=NO_SIGNAL
    const actionableRecs = llmRecommendations.filter(r => r.action === "buy" || r.action === "sell");
    if (actionableRecs.length === 0) {
      console.log(`ACTION=NO_SIGNAL reason="no_actionable_recommendations" raw_count=${rawRecommendations.length} filtered_count=${llmRecommendations.length}`);
      skipCounters.recordSkip("NO_SIGNAL");
      // Continue to data health tracking but don't enter the trading loop
    }
    
    // P1: PRE-SCAN TRADABILITY EVALUATION
    // Run tradability gates on ALL symbols BEFORE strategy evaluation
    // This gives us actual symbolsEvaluated count and per-symbol skip reasons
    // Standard reason codes: NO_QUOTE, SPREAD_TOO_WIDE, VOLUME_TOO_LOW, ATR_TOO_LOW, ATR_TOO_HIGH, TIME_GUARD_BLOCKED, STRATEGY_NOT_ENABLED
    const entryUniverse = dayTraderConfig.getEntryUniverse();
    const tradabilityResults: Map<string, { passed: boolean; reasons: string[]; gates: { spreadPass: boolean; volumePass: boolean; atrPass: boolean } }> = new Map();
    const tradableSymbols: string[] = [];
    let symbolsEvaluated = 0;
    
    console.log(`[P1:TRADABILITY] Evaluating ${entryUniverse.length} symbols through tradability gates`);
    
    for (const symbol of entryUniverse) {
      symbolsEvaluated++;
      
      try {
        const marketData = await tradabilityGates.fetchSymbolMarketData(symbol);
        
        if (!marketData) {
          tradabilityResults.set(symbol, { 
            passed: false, 
            reasons: ["NO_QUOTE"], 
            gates: { spreadPass: false, volumePass: false, atrPass: false } 
          });
          console.log(`[P1:GATE] ${symbol} FAIL gates={spreadPass:false,volumePass:false,atrPass:false} skipReasons=[NO_QUOTE]`);
          skipCounters.recordSkip("NO_QUOTE");
          continue;
        }
        
        // Run individual gate checks to capture per-gate results
        const spreadCheck = await tradabilityGates.checkSpreadGate(symbol, marketData.bid, marketData.ask, marketData.price);
        const liquidityCheck = await tradabilityGates.checkLiquidityGate(symbol, marketData.volume);
        const volatilityCheck = await tradabilityGates.checkVolatilityGate(symbol, marketData.atrPercent);
        const extremeCheck = await tradabilityGates.checkExtremeMoveGate(symbol, marketData.changePercent);
        const timeCheck = tradabilityGates.checkTimeGate();
        
        const gates = {
          spreadPass: spreadCheck.passed,
          volumePass: liquidityCheck.passed,
          atrPass: volatilityCheck.passed,
        };
        
        // Collect skip reasons using ONLY standard codes:
        // NO_QUOTE, SPREAD_TOO_WIDE, VOLUME_TOO_LOW, ATR_TOO_LOW, ATR_TOO_HIGH, TIME_GUARD_BLOCKED, STRATEGY_NOT_ENABLED
        const skipReasons: string[] = [];
        if (!spreadCheck.passed) skipReasons.push("SPREAD_TOO_WIDE");
        if (!liquidityCheck.passed) skipReasons.push("VOLUME_TOO_LOW");
        if (!volatilityCheck.passed) {
          // Distinguish between too low and too high
          if (volatilityCheck.reason && volatilityCheck.reason.includes("<")) {
            skipReasons.push("ATR_TOO_LOW");
          } else {
            skipReasons.push("ATR_TOO_HIGH");
          }
        }
        // Map extreme move to ATR_TOO_HIGH (extreme volatility)
        if (!extremeCheck.passed) skipReasons.push("ATR_TOO_HIGH");
        if (!timeCheck.passed) skipReasons.push("TIME_GUARD_BLOCKED");
        
        const allPassed = skipReasons.length === 0;
        tradabilityResults.set(symbol, { passed: allPassed, reasons: skipReasons, gates });
        
        const logLine = `[P1:GATE] ${symbol} ${allPassed ? 'PASS' : 'FAIL'} gates={spreadPass:${gates.spreadPass},volumePass:${gates.volumePass},atrPass:${gates.atrPass}} skipReasons=[${skipReasons.join(",")}]`;
        console.log(logLine);
        entryWindowProof.recordP1GateLog(logLine, allPassed);
        
        if (allPassed) {
          tradableSymbols.push(symbol);
        } else {
          // Record each skip reason using standard codes
          for (const reason of skipReasons) {
            skipCounters.recordSkip(reason);
          }
        }
      } catch (err) {
        tradabilityResults.set(symbol, { 
          passed: false, 
          reasons: ["NO_QUOTE"], 
          gates: { spreadPass: false, volumePass: false, atrPass: false } 
        });
        const errorLogLine = `[P1:GATE] ${symbol} FAIL gates={spreadPass:false,volumePass:false,atrPass:false} skipReasons=[NO_QUOTE]`;
        console.log(errorLogLine);
        entryWindowProof.recordP1GateLog(errorLogLine, false);
        skipCounters.recordSkip("NO_QUOTE");
      }
    }
    
    // P1: Tick summary - top skip reasons + counts (peek only, don't reset - activity ledger does that)
    const p1TopReasons = skipCounters.getTopSkipReasons(5);
    const topReasonsStr = p1TopReasons.map(r => `${r.reason}:${r.count}`).join(", ");
    const tradabilitySummary = `[P1:TRADABILITY] ${tradableSymbols.length}/${symbolsEvaluated} passed | topSkipReasons=[${topReasonsStr || "none"}]`;
    console.log(tradabilitySummary);
    entryWindowProof.recordP1TradabilityLog(tradabilitySummary);
    
    // GATE-TRUTH-1: Record that tradability gates ran this tick
    const failedCount = symbolsEvaluated - tradableSymbols.length;
    entryWindowProof.recordTradabilityGateRun(symbolsEvaluated, tradableSymbols.length, failedCount);
    
    // HARDENED INDICATOR PIPELINE: Fetch bars and compute indicators safely (with warm-start & tiers)
    const pipelineResults = await indicatorPipeline.fetchIndicatorsForUniverse(ALLOWED_UNIVERSE);
    const validIndicatorCount = pipelineResults.validCount;
    const barsMin = pipelineResults.barsMin === Infinity ? 0 : pipelineResults.barsMin;
    const tierMin = pipelineResults.tierMin;
    const tier1Count = pipelineResults.tier1Count;
    const tier2Count = pipelineResults.tier2Count;
    const hasValidIndicators = validIndicatorCount > 0;
    
    // validPriceCount = symbols that have valid indicators (indicators require valid price data)
    let validPriceCount = validIndicatorCount;
    
    // P5: DETERMINISTIC STRATEGY ENGINE - generates signals without LLM dependency
    // These signals take priority over LLM recommendations when enabled
    // P1: Only evaluate symbols that passed tradability gates
    let strategySignals: strategyEngine.StrategySignal[] = [];
    if (!USE_STRATEGY_ENGINE) {
      console.log(`[StrategyEngine] DISABLED - recording STRATEGY_NOT_ENABLED skip`);
      skipCounters.recordSkip("STRATEGY_NOT_ENABLED");
    } else if (USE_STRATEGY_ENGINE && hasValidIndicators) {
      console.log(`[StrategyEngine] Evaluating ${tradableSymbols.length} tradable symbols with deterministic strategies`);
      strategyEngine.logStrategyEngineStatus();
      
      for (const symbol of tradableSymbols) {
        const symbolIndicators = pipelineResults.results.get(symbol);
        if (!symbolIndicators || !symbolIndicators.ok || !symbolIndicators.indicators) continue;
        
        try {
          // Fetch bars for VWAP/ORB calculation
          const bars = await strategyEngine.fetchBarsForStrategy(symbol);
          
          // Get current quote for price data (extended with bid/ask)
          const quote = await alpaca.getExtendedQuote(symbol);
          
          // Calculate today's volume from bars (sum of all bars' volume)
          // This is more reliable than quote.volume which is often 0
          let todayVolume = 0;
          if (bars && bars.length > 0) {
            // Sum volume from bars that are from today
            const today = new Date().toISOString().split('T')[0];
            for (const bar of bars) {
              if (bar.t.startsWith(today)) {
                todayVolume += bar.v;
              }
            }
            // If no bars from today yet (pre-market), use yesterday's total
            if (todayVolume === 0) {
              todayVolume = bars.slice(-20).reduce((sum, bar) => sum + bar.v, 0);
            }
          }
          
          // Build symbol data for strategy evaluation
          const symbolData: strategyEngine.SymbolData = {
            symbol,
            currentPrice: quote.price,
            bid: quote.bid,
            ask: quote.ask,
            volume: todayVolume,  // Use calculated volume from bars
            indicators: symbolIndicators.indicators,
            bars: bars || undefined,
          };
          
          // Generate signal from strategy engine
          const signal = await strategyEngine.generateSignal(symbolData);
          if (signal && signal.side !== "none") {
            strategySignals.push(signal);
            console.log(`[StrategyEngine] ${symbol}: ${signal.strategyName} -> ${signal.side.toUpperCase()} (confidence=${signal.confidence})`);
          }
        } catch (err) {
          console.log(`[StrategyEngine] ${symbol}: error evaluating strategies - ${err instanceof Error ? err.message : String(err)}`);
        }
      }
      
      console.log(`[StrategyEngine] Generated ${strategySignals.length} strategy signals`);
    }
    
    // P5: Convert strategy signals to recommendations format
    // Strategy signals override LLM recommendations for the same symbol
    const strategyRecommendations = strategySignals.map(signal => ({
      symbol: signal.symbol,
      action: signal.side as "buy" | "sell" | "hold",
      confidence: signal.confidence,
      reason: signal.reason,
      strategyName: signal.strategyName,  // P5: Tag with strategy name (maps to TradeRecommendation.strategyName)
      riskLevel: "medium" as const,
      targetPrice: signal.entrySignalPrice,
      stopLoss: signal.invalidation || undefined,
    }));
    
    // P5: Use strategy recommendations if available, otherwise fall back to LLM
    const recommendations = strategyRecommendations.length > 0 
      ? strategyRecommendations 
      : llmRecommendations;
    
    // Log source of recommendations
    if (strategyRecommendations.length > 0) {
      console.log(`[P5] Using ${strategyRecommendations.length} DETERMINISTIC strategy signals (LLM fallback disabled)`);
    } else if (rawRecommendations.length > 0) {
      console.log(`[P5] No strategy signals - using ${recommendations.length} LLM recommendations as fallback`);
    }
    
    // Legacy fallback for technical indicators from OpenAI analysis
    const technicalIndicatorsCount = (analysis.technicalIndicators || []).length;
    
    // DATA INTEGRITY WARNING: If no valid indicators, log warning but allow fallback path
    if (!hasValidIndicators && recommendations.length > 0) {
      console.log(`[Bot] WARNING: No valid indicators but ${recommendations.length} recommendations from OpenAI fallback`);
      console.log(`[Bot] Fallback mode - using OpenAI analysis instead of technical indicators`);
    }
    
    // Log tier distribution
    if (hasValidIndicators) {
      console.log(`[Bot] TIER STATUS: tier1=${tier1Count} tier2=${tier2Count} tierMin=${tierMin}`);
    }
    
    // SIGNAL TRACKING: Record raw signals BEFORE filters
    // RULE: Only count signals if indicators are valid (otherwise it's fallback/invalid data)
    let signalsThisCycle = 0;
    let executedThisCycle = 0;
    let symbolErrors = 0;
    for (const rec of recommendations) {
      // CRASH-PROOF: Wrap per-symbol processing in try/catch so one failure doesn't kill the loop
      try {
        if (rec.action === "hold") continue;
        
        // Record signal BEFORE any filters (this is a raw signal from the strategy)
        // Only count if we have valid indicator data (not just fallback recommendations)
        const signalType = `SIGNAL_${rec.action.toUpperCase()}_${rec.symbol.toUpperCase()}`;
        signalCounters.recordSignal(signalType);
        signalCounters.recordCandidate();
        // signalsThisCycle only increments if indicators are valid
        if (hasValidIndicators) {
          signalsThisCycle++;
        }

        // FORT KNOX: Check if symbol is in allowed universe (baseline mode restricts further)
        if (rec.action === "buy") {
          const universeCheck = dayTraderConfig.isSymbolAllowedForEntry(rec.symbol);
          if (!universeCheck.allowed) {
            console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[universe:${universeCheck.reason}]`);
            skipCounters.recordSkip(universeCheck.reason);
            continue;
          }
        } else if (!dayTraderConfig.isSymbolAllowed(rec.symbol)) {
          // For non-buy actions, still check full universe
          console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[universe:not_in_allowed_universe]`);
          skipCounters.recordSkip("SYMBOL_NOT_ALLOWED");
          continue;
        }
      
      // TRADABILITY GATES: Pre-strategy hard gates (spread, liquidity, volatility, extreme move, time)
      // Gates run BEFORE any strategy/validator can approve a trade
      try {
        const marketData = await tradabilityGates.fetchSymbolMarketData(rec.symbol);
        if (!marketData) {
          console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[liquidity:missing]`);
          skipCounters.recordSkip("LIQUIDITY_TOO_LOW");
          continue;
        }
        
        const tradabilityCheck = await tradabilityGates.runAllTradabilityGates(rec.symbol, marketData, true);
        if (!tradabilityCheck.passed) {
          console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[${tradabilityCheck.reasons.join(",")}]`);
          for (const reason of tradabilityCheck.reasons) {
            if (reason.startsWith("spread:")) {
              skipCounters.recordSkip("SPREAD_TOO_WIDE");
            } else if (reason.startsWith("liquidity:")) {
              skipCounters.recordSkip("LIQUIDITY_TOO_LOW");
            } else if (reason.startsWith("atrPct:") && reason.includes("<")) {
              skipCounters.recordSkip("ATR_TOO_LOW");
            } else if (reason.startsWith("atrPct:") && reason.includes(">")) {
              skipCounters.recordSkip("ATR_TOO_HIGH");
            } else if (reason.startsWith("extremeMove:")) {
              skipCounters.recordSkip("EXTREME_MOVE");
            } else if (reason.startsWith("timeGate:")) {
              skipCounters.recordSkip("TIME_GATE");
            } else {
              skipCounters.recordSkip("TRADABILITY_GATE_FAIL");
            }
          }
          continue;
        }
      } catch (tradabilityError) {
        console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[liquidity:missing]`);
        skipCounters.recordSkip("LIQUIDITY_TOO_LOW");
        continue;
      }

      // DATA INTEGRITY: Check if this symbol has valid indicators from the pipeline
      const symbolIndicators = pipelineResults.results.get(rec.symbol);
      if (!symbolIndicators || !symbolIndicators.ok) {
        const reason = symbolIndicators?.reason || "unknown";
        const bars = symbolIndicators?.barsLen || 0;
        console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[indicators:${reason}_tier=0_bars=${bars}]`);
        skipCounters.recordSkip("INDICATOR_PIPELINE_FAIL");
        continue;
      }
      
      // P5: Get strategy name from recommendation (deterministic strategies set strategyName)
      const recStrategyName = (rec as any).strategyName || "LLM_FALLBACK";
      
      // P3 PRE-TRADE CHECK: Kill-switch and daily safety controls
      if (rec.action === "buy") {
        const p3Check = riskEngine.preTradeCheck(recStrategyName, null);
        if (!p3Check.allowed) {
          console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[${p3Check.skipReason}]`);
          continue;
        }
      }
      
      // TIER-BASED STRATEGY GATING: Check if strategy requires Tier 2 (MACD)
      // Use explicit registry of MACD-requiring strategies (not fragile string matching)
      const symbolTier = symbolIndicators.tier;
      const requiresMacd = strategyRequiresMacd(recStrategyName);
      
      if (requiresMacd && symbolTier < 2) {
        console.log(`[Bot] SKIP: ${rec.symbol} - ${recStrategyName} blocked (MACD_REQUIRES_TIER_2) tier=${symbolTier} bars=${symbolIndicators.barsLen}`);
        skipCounters.recordSkip("MACD_REQUIRES_TIER_2");
        continue;
      }
      
      // FORT KNOX: For BUY orders, check if we're within entry window
      if (rec.action === "buy" && !canEnterNewPositions) {
        console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[entry_window:outside_entry_window]`);
        skipCounters.recordSkip("OUTSIDE_ENTRY_WINDOW");
        continue;
      }
      
      // FORT KNOX: Check P&L kill threshold for new entries
      if (rec.action === "buy" && dayTraderConfig.isDailyKillThresholdHit()) {
        const status = dayTraderConfig.getDayTraderStatus();
        const killType = status.lossLimitHit ? "daily_max_loss" : "daily_profit_target";
        console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[${killType}:pnl=$${status.dailyPnL.toFixed(0)}]`);
        skipCounters.recordSkip("KILL_THRESHOLD_HIT");
        continue;
      }

      // Check if we can open more positions
      if (rec.action === "buy" && positions.length >= (settings.maxPositions || 5)) {
        console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[max_positions:${positions.length}/${settings.maxPositions || 5}]`);
        skipCounters.recordSkip("MAX_POSITIONS");
        continue;
      }

      // Check buying power
      const estimatedCost = (settings.maxPositionSize || 1000);
      if (rec.action === "buy" && buyingPower < estimatedCost) {
        console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[buying_power:$${buyingPower.toFixed(0)}<$${estimatedCost}]`);
        skipCounters.recordSkip("INSUFFICIENT_BUYING_POWER");
        continue;
      }

      // Check confidence threshold
      if (rec.confidence < MIN_CONFIDENCE_THRESHOLD) {
        console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[low_confidence:${rec.confidence}%]`);
        skipCounters.recordSkip("LOW_CONFIDENCE");
        continue;
      }

      // Get current price for the symbol
      let currentPrice = (rec as any).targetPrice || 0;
      try {
        const quote = await alpaca.getLatestQuote(rec.symbol);
        if (quote.price > 0) {
          currentPrice = quote.price;
          // Note: validPriceCount is now based on validIndicatorCount (indicators require valid prices)
        }
        console.log(`[Bot] Current price for ${rec.symbol}: $${currentPrice}`);
      } catch (error) {
        console.log(`[Bot] Could not get price for ${rec.symbol}, using estimate`);
      }

      // Skip if we couldn't get a valid price
      if (currentPrice <= 0) {
        console.log(`[Bot] SKIP: ${rec.symbol} - no valid price available`);
        skipCounters.recordSkip("NO_VALID_PRICE");
        continue;
      }

      // Log cooldown status for observability (logging only - does not block)
      if (rec.action === "buy" && dayTraderConfig.isOnCooldown(rec.symbol)) {
        const remaining = dayTraderConfig.getCooldownRemaining(rec.symbol);
        const remainingMinutes = Math.ceil(remaining / 60000);
        console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[cooldown:${remainingMinutes}min]`);
        skipCounters.recordSkip("COOLDOWN_ACTIVE");
        // Note: Cooldown is informational only - actual enforcement happens in risk manager
      }

      // Advanced risk management: Check if trade is allowed
      const tradeCheck = await riskManager.shouldAllowTrade(
        rec.symbol,
        rec.action === "buy" ? "buy" : "sell",
        settings.maxPositionSize || 1000
      );
      
      if (!tradeCheck.allowed) {
        console.log(`ACTION=SKIP symbol=${rec.symbol} SKIP_REASONS=[risk_manager:${tradeCheck.reason}]`);
        skipCounters.recordSkip("RISK_CHECK_FAILED");
        await storage.createActivityLog({
          type: "system",
          action: "Trade Blocked by Risk Manager",
          description: `${rec.symbol}: ${tradeCheck.reason}`,
        });
        continue;
      }

      // P3: ATR-BASED POSITION SIZING (replaces fixed 5% sizing)
      // Use indicator pipeline ATR for risk-per-trade sizing
      const symbolATR = symbolIndicators.indicators?.atr14 || 0;
      const atrSizing = riskEngine.calculateATRPositionSize(
        rec.symbol,
        currentPrice,
        symbolATR,
        portfolioValue,
        rec.action === "buy" ? "buy" : "sell"
      );
      
      // Store P3 sizing results for bracket order calculation
      const p3StopDistance = atrSizing.stopDistance;
      const p3StopPrice = atrSizing.stopPrice;
      const p3TakeProfitPrice = atrSizing.takeProfitPrice;
      
      // Also get volatility-adjusted sizing as a safety cap
      const positionSizing = await riskManager.calculateDynamicPositionSize(
        rec.symbol,
        currentPrice,
        rec.action === "buy" ? "buy" : "sell",
        portfolioValue,
        positions.length
      );
      
      // Use the smaller of ATR-based or volatility-adjusted sizing for safety
      let quantity = Math.min(atrSizing.qty, positionSizing.recommendedShares);
      console.log(`[Bot] P3 ATR sizing: ${atrSizing.qty} shares ($${atrSizing.notional.toFixed(0)}) - ${atrSizing.reasoning}`);
      console.log(`[Bot] Volatility cap: ${positionSizing.recommendedShares} shares. ${positionSizing.reasoning}`);
      
      // TIER 1 RISK REDUCTION: 50% of normal position size if only Tier 1 indicators available
      const tier1SizingReduction = indicatorPipeline.TIER_1_SIZING_REDUCTION;
      if (symbolTier === 1) {
        const originalQty = quantity;
        quantity = Math.max(1, Math.floor(quantity * tier1SizingReduction));
        console.log(`[Bot] TIER=1 sizing_reduction=${tier1SizingReduction}: ${originalQty} -> ${quantity} shares`);
      }
      
      console.log(`[Bot] Using: ${quantity} shares for ${rec.symbol} (tier=${symbolTier})`);

      // P5: Get strategy name from recommendation (deterministic strategies set strategyName)
      const strategyName = (rec as any).strategyName || "LLM_FALLBACK";
      
      const recommendation: TradeRecommendation = {
        symbol: rec.symbol,
        side: rec.action === "buy" ? "buy" : "sell",
        quantity,
        price: currentPrice,
        reason: rec.reason,
        confidence: rec.confidence,
        riskLevel: (rec as any).riskLevel,
        // P3: Pass ATR-based stops through for bracket orders
        stopPrice: p3StopPrice,
        takeProfitPrice: p3TakeProfitPrice,
        atr: symbolATR,
        // P4: Pass regime for measurement + tuning
        regime: regimeResult.regimeLabel || "chop",
        // P5: Pass strategy name for A/B evaluation
        strategyName,
      };

      // Ato executes the trade (the body acts on the brain's decision)
      console.log(`[Ato] Executing: ${rec.action.toUpperCase()} ${quantity} shares of ${rec.symbol} at $${currentPrice}`);
      
      const tradeResult = await ato.executeTraderDecision(recommendation, settings);
      
      if (tradeResult.success) {
        // EXECUTION TRACKING: Record successful trade execution
        signalCounters.recordExecution(rec.symbol, rec.action);
        executedThisCycle++;
        
        await storage.createAlert({
          type: "info",
          title: `Ato Executed: ${rec.action.toUpperCase()} ${rec.symbol}`,
          message: `Bought ${quantity} shares at $${currentPrice.toFixed(2)}. Reason: ${rec.reason}`,
          requiresApproval: false,
        });
        
        // Record trade result for Ato's daily stats
        ato.recordTradeResult(rec.symbol, 0); // P/L will be calculated on close
      } else {
        // Trade passed all filters but execution failed
        skipCounters.recordSkip("EXECUTION_FAILED");
        await storage.createAlert({
          type: "critical",
          title: `Ato Trade Failed: ${rec.symbol}`,
          message: `Failed to execute ${rec.action}: ${tradeResult.message}`,
          requiresApproval: false,
        });
      }

      await storage.createResearchLog({
        type: "recommendation",
        symbol: rec.symbol,
        summary: `${rec.action.toUpperCase()} ${tradeResult.success ? "EXECUTED" : "FAILED"} with ${rec.confidence}% confidence`,
        details: rec.reason,
        confidence: rec.confidence,
      });
      } catch (symbolError) {
        // CRASH-PROOF: Log error and continue to next symbol
        symbolErrors++;
        console.error(`[ERROR] symbol=${rec.symbol} step=processing exception=${symbolError instanceof Error ? symbolError.message : String(symbolError)}`);
      }
    }
    
    // Log any symbol processing errors
    if (symbolErrors > 0) {
      console.log(`[ANALYSIS] Errors: ${symbolErrors} symbol(s) failed processing but scan continued`);
    }
    
    // FUNNEL OUTCOME: Log what happened this cycle
    if (executedThisCycle > 0) {
      console.log(`[ANALYSIS] Outcome: EXECUTED ${executedThisCycle} trade(s) from ${signalsThisCycle} signals`);
    } else if (signalsThisCycle > 0) {
      console.log(`[ANALYSIS] Outcome: SKIPPED (0/${signalsThisCycle} signals passed filters)`);
    } else {
      // ACTION=NO_SIGNAL: Strategy evaluated normally but produced no trade opportunity
      console.log(`ACTION=NO_SIGNAL symbols=[${ALLOWED_UNIVERSE.join(",")}] indicators_valid=${hasValidIndicators ? validIndicatorCount : 0}/${symbolsEvaluated}`);
      skipCounters.recordSkip("NO_SIGNAL");
    }
    
    // DATA HEALTH: Record tick metrics for pipeline diagnosis (with tier info)
    dataHealthCounters.recordTickMetrics({
      symbolsEvaluated,
      validPrices: validPriceCount,
      validIndicators: validIndicatorCount,
      rawSignals: signalsThisCycle,
      barsMin,
      tierMin,
      tier1Count,
      tier2Count,
    });
    
    // ACTIVITY LEDGER: Record per-tick scan summary for report truthfulness
    const tickSkips = skipCounters.getAndResetTickSkips();
    const execCounters = executionTrace.getExecutionCounters();
    activityLedger.recordTick({
      symbolsEvaluated,
      validQuotes: validPriceCount,
      validBars: barsMin,
      noSignalCount: signalsThisCycle === 0 ? 1 : 0,
      skipCount: tickSkips.totalSkips,
      skipReasonCounts: tickSkips.reasonCounts,
      signalsGenerated: signalsThisCycle,
      tradesProposed: execCounters.tradesProposed,
      tradesSubmitted: execCounters.tradesSubmitted,
      tradesFilled: executedThisCycle,
    });
    
    // Create day trading settings with tighter stops
    const dayTradingSettings = {
      ...settings,
      stopLossPercent: dayTradingStopLoss,
      takeProfitPercent: dayTradingTakeProfit,
    };
    
    // Use Alpaca clock for accurate market close timing
    try {
      const clock = await alpaca.getClock();
      const nextClose = new Date(clock.next_close);
      const now = new Date();
      const minutesUntilClose = (nextClose.getTime() - now.getTime()) / (1000 * 60);
      
      // If less than 15 minutes until close, flatten all positions (day trading rule)
      if (clock.is_open && minutesUntilClose <= 15 && minutesUntilClose > 0) {
        console.log(`[Ato] ${minutesUntilClose.toFixed(0)} min until close - closing all positions to stay flat overnight`);
        await ato.closeAllPositionsEndOfDay();
      } else {
        // Ato manages positions (stop-loss, take-profit) - the body maintains its health
        await ato.managePositions(dayTradingSettings);
      }
    } catch (clockError) {
      console.error("[Bot] Error getting market clock, managing positions normally:", clockError);
      await ato.managePositions(dayTradingSettings);
    }
    
    // Record outcomes for Autopilot brain learning
    for (const position of positions) {
      const plPercent = position.unrealizedPLPercent;
      
      if (plPercent <= -dayTradingStopLoss || plPercent >= dayTradingTakeProfit) {
        // Brain learns from closed positions
        await brain.recordTradeOutcome(
          position.symbol,
          "buy",
          position.avgEntryPrice,
          position.currentPrice,
          position.quantity
        );
      }
    }

    botStatus = {
      status: previousStatus,
      lastAnalysis: new Date().toISOString(),
      currentAction: null,
      errorMessage: null,
    };

    await storage.createActivityLog({
      type: "analysis",
      action: "Analysis Complete",
      description: `Market analysis complete. Sentiment: ${analysis.sentiment}, Confidence: ${analysis.confidence}%`,
    });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    botStatus = {
      ...botStatus,
      status: "error",
      currentAction: null,
      errorMessage,
    };

    await storage.createActivityLog({
      type: "error",
      action: "Analysis Failed",
      description: errorMessage,
    });
  }
}

async function syncPositions(): Promise<void> {
  try {
    const alpacaPositions = await alpaca.getPositions();

    // Clear existing positions and sync from Alpaca
    await storage.clearPositions();

    for (const pos of alpacaPositions) {
      await storage.upsertPosition({
        symbol: pos.symbol,
        quantity: parseInt(pos.qty),
        avgEntryPrice: parseFloat(pos.avg_entry_price),
        currentPrice: parseFloat(pos.current_price),
        marketValue: parseFloat(pos.market_value),
        unrealizedPL: parseFloat(pos.unrealized_pl),
        unrealizedPLPercent: parseFloat(pos.unrealized_plpc) * 100,
      });
    }
  } catch (error) {
    console.error("Failed to sync positions:", error);
  }
}

export async function executeTrade(recommendation: TradeRecommendation): Promise<boolean> {
  const settings = await storage.getSettings();

  // FORT KNOX: Check symbol is in allowed universe (baseline mode restricts entry further)
  if (recommendation.side === "buy") {
    const universeCheck = dayTraderConfig.isSymbolAllowedForEntry(recommendation.symbol);
    if (!universeCheck.allowed) {
      console.log(`[Bot] BLOCKED: ${recommendation.symbol} - ${universeCheck.reason}`);
      await storage.createActivityLog({
        type: "trade",
        action: "Trade Blocked - Universe",
        description: `BLOCKED: ${recommendation.symbol} - ${universeCheck.reason}`,
      });
      return false;
    }
  } else if (!dayTraderConfig.isSymbolAllowed(recommendation.symbol)) {
    // For exits, still check full universe
    console.log(`[Bot] BLOCKED: ${recommendation.symbol} not in allowed universe`);
    await storage.createActivityLog({
      type: "trade",
      action: "Trade Blocked - Symbol",
      description: `BLOCKED: ${recommendation.symbol} not in allowed universe`,
    });
    return false;
  }
  
  // FORT KNOX: Check trading time
  const tradingStatus = timeGuard.getTradingStatus();
  if (!tradingStatus.canTrade) {
    console.log(`[Bot] BLOCKED: Past force close (3:45 PM ET)`);
    return false;
  }
  
  // FORT KNOX: For BUY orders, check entry window and kill threshold
  if (recommendation.side === "buy") {
    if (!tradingStatus.canEnterNewPositions) {
      console.log(`[Bot] BLOCKED: Outside entry window (11:35 AM ET cutoff)`);
      await storage.createActivityLog({
        type: "trade",
        action: "Trade Blocked - Entry Cutoff",
        description: `BLOCKED: ${recommendation.symbol} - outside entry window`,
      });
      return false;
    }
    
    if (dayTraderConfig.isDailyKillThresholdHit()) {
      const status = dayTraderConfig.getDayTraderStatus();
      console.log(`[Bot] BLOCKED: P&L kill threshold hit ($${status.dailyPnL.toFixed(0)})`);
      await storage.createActivityLog({
        type: "trade",
        action: "Trade Blocked - P&L Kill",
        description: `BLOCKED: ${recommendation.symbol} - P&L kill threshold`,
      });
      return false;
    }
    
    // P2 IDEMPOTENCY GUARD: Prevent duplicate entries
    const idempotencyCheck = await tradeLifecycle.canEnterPosition(recommendation.symbol);
    if (!idempotencyCheck.allowed) {
      console.log(`ACTION=SKIP symbol=${recommendation.symbol} SKIP_REASONS=[${idempotencyCheck.reason}]`);
      skipCounters.recordSkip("IDEMPOTENCY");
      
      // TRADE ACCOUNTING: Record suppression for proposed trades blocked by idempotency
      const strategyType = recommendation.reason?.toLowerCase().includes("scalp") ? "scalp" :
                           recommendation.reason?.toLowerCase().includes("dip") ? "dip" :
                           recommendation.reason?.toLowerCase().includes("vwap") ? "vwap" : "breakout";
      const tempTradeId = generateTradeId(recommendation.symbol, strategyType, recommendation.side, 2);
      tradeAccounting.recordSuppress({
        tradeId: tempTradeId,
        symbol: recommendation.symbol,
        strategyName: strategyType,
        limitPrice: recommendation.price,
        qty: recommendation.quantity,
        reason: idempotencyCheck.reason,
      });
      
      await storage.createActivityLog({
        type: "trade",
        action: "Trade Blocked - Idempotency",
        description: `BLOCKED: ${recommendation.symbol} - ${idempotencyCheck.reason}`,
      });
      return false;
    }
  }

  // Generate trade_id for HIGH confidence pairing in reports (outside try for catch access)
  const strategyType = recommendation.reason?.toLowerCase().includes("scalp") ? "scalp" :
                       recommendation.reason?.toLowerCase().includes("dip") ? "dip" :
                       recommendation.reason?.toLowerCase().includes("vwap") ? "vwap" : "breakout";
  const tradeId = generateTradeId(recommendation.symbol, strategyType, recommendation.side, 2);
  let proposalRecorded = false;  // Track if PROPOSE was called for proper SUBMIT_FAIL handling
  
  try {
    
    // P3: Use ATR-based stops if provided, fallback to fixed percentages
    // ATR-based stops are computed upstream in the analysis loop
    let stopLossPrice: number;
    let takeProfitPrice: number;
    
    if (recommendation.stopPrice && recommendation.takeProfitPrice) {
      // P3: Use adaptive ATR-based stops
      stopLossPrice = recommendation.stopPrice;
      takeProfitPrice = recommendation.takeProfitPrice;
      const stopPct = ((recommendation.price - stopLossPrice) / recommendation.price * 100).toFixed(2);
      const tpPct = ((takeProfitPrice - recommendation.price) / recommendation.price * 100).toFixed(2);
      console.log(`[P3] Using ATR-based stops: SL=-${stopPct}%, TP=+${tpPct}%`);
    } else {
      // Fallback: Fixed percentage stops (legacy behavior)
      const stopLossPercent = 0.01;
      const takeProfitPercent = 0.025;
      stopLossPrice = recommendation.price * (1 - stopLossPercent);
      takeProfitPrice = recommendation.price * (1 + takeProfitPercent);
      console.log(`[P3] Fallback to fixed stops: SL=-1.0%, TP=+2.5%`);
    }
    
    // P2: Create trade lifecycle record for BUY orders (tracks slippage, links orders)
    // P4: Include regime metadata for measurement + tuning
    let lifecycleTrade: tradeLifecycle.TradeRecord | null = null;
    if (recommendation.side === "buy") {
      const stopDistance = recommendation.price - stopLossPrice;
      const rrRatio = (takeProfitPrice - recommendation.price) / stopDistance;
      const atrPct = recommendation.atr && recommendation.price > 0 
        ? (recommendation.atr / recommendation.price) * 100 
        : null;
      const usedFallback = !recommendation.stopPrice || !recommendation.takeProfitPrice;
      
      lifecycleTrade = tradeLifecycle.createTrade(
        recommendation.symbol,
        recommendation.side,
        strategyType,
        recommendation.price, // signal price for slippage tracking
        stopLossPrice,
        takeProfitPrice,
        undefined, // spreadAtSignal
        {
          regime: recommendation.regime || "chop",
          atr: recommendation.atr || null,
          atrPct,
          stopDistance,
          rr: rrRatio,
          usedAtrFallback: usedFallback,
          gatesPassed: true,
          gateFailReasons: [],
        }
      );
    }
    
    // Create trade record
    const trade = await storage.createTrade({
      symbol: recommendation.symbol,
      side: recommendation.side,
      quantity: recommendation.quantity,
      price: recommendation.price,
      totalValue: recommendation.quantity * recommendation.price,
      status: "pending",
      reason: recommendation.reason,
    });

    console.log(`ACTION=TRADE symbol=${recommendation.symbol} side=${recommendation.side} qty=${recommendation.quantity} price=${recommendation.price.toFixed(2)} strategy=${strategyType} trade_id=${tradeId}`);

    // Execute trade via Alpaca (works for both paper and live trading)
    // CRITICAL: Only BUY orders use bracket orders - SELL orders MUST use regular market orders
    // Bracket orders are for opening positions with attached stop loss / take profit
    // Sells/exits must go through regular market orders to close positions cleanly
    let order;
    if (recommendation.side === "buy") {
      // P6: Get fresh quote for limit order calculation and staleness check
      const freshQuote = await alpaca.getExtendedQuote(recommendation.symbol);
      
      // P6: Quote freshness gate - FAIL-CLOSED if quote is stale or missing
      const quoteCheck = await tradabilityGates.checkQuoteFreshnessGate(freshQuote.timestamp);
      if (!quoteCheck.passed) {
        console.log(`ACTION=SKIP symbol=${recommendation.symbol} SKIP_REASONS=[${quoteCheck.reason}]`);
        skipCounters.recordSkip("QUOTE_STALE_OR_MISSING");
        await storage.updateTradeStatus(trade.id, "rejected", undefined);
        return false;
      }
      
      // P6: LIMIT BRACKET ORDER - reduces slippage vs market orders
      // Calculate limit price: min(ask, lastPrice) + offset
      const { limitPrice, offset, reasoning } = executionQuality.calculateEntryLimit(
        freshQuote.bid,
        freshQuote.ask,
        recommendation.price
      );
      
      const slPct = ((limitPrice - stopLossPrice) / limitPrice * 100);
      const tpPct = ((takeProfitPrice - limitPrice) / limitPrice * 100);
      console.log(`[P6 LIMIT BRACKET] ${recommendation.symbol}: Limit=$${limitPrice.toFixed(2)} (${reasoning}), SL=$${stopLossPrice.toFixed(2)} (-${slPct.toFixed(2)}%), TP=$${takeProfitPrice.toFixed(2)} (+${tpPct.toFixed(2)}%)`);
      
      // Track order submission
      executionQuality.recordOrderSubmitted();
      const submitTime = Date.now();
      
      // TRADE ACCOUNTING: Record proposal before submitting to Alpaca
      tradeAccounting.recordProposal({
        tradeId,
        symbol: recommendation.symbol,
        strategyName: strategyType,
        limitPrice,
        qty: recommendation.quantity,
      });
      proposalRecorded = true;
      
      order = await alpaca.submitLimitBracketOrder(
        recommendation.symbol,
        recommendation.quantity,
        "buy",
        limitPrice,
        stopLossPrice,
        takeProfitPrice,
        tradeId
      );
      
      // TRADE ACCOUNTING: Record successful submission to Alpaca
      tradeAccounting.recordSubmitOk({
        tradeId,
        symbol: recommendation.symbol,
        strategyName: strategyType,
        limitPrice,
        qty: recommendation.quantity,
        alpacaOrderId: order.id,
      });
      
      // P2: Link parent order to trade lifecycle
      if (lifecycleTrade) {
        tradeLifecycle.linkParentOrder(lifecycleTrade.tradeId, order.id);
      }
      
      // P6: Wait for fill with timeout (45 seconds)
      // If not filled, cancel and skip - do NOT chase price upward
      if (order.status !== "filled") {
        console.log(`[P6] Waiting for limit order ${order.id} to fill (timeout: ${executionQuality.EXECUTION_CONFIG.FILL_TIMEOUT_MS}ms)...`);
        
        const fillResult = await executionQuality.waitForFill(order.id);
        
        if (fillResult.filled && fillResult.order) {
          // Order filled successfully
          order = { ...order, ...fillResult.order, status: "filled" };
          executionQuality.recordOrderFilled(fillResult.timeToFillMs);
          console.log(`[P6] Order FILLED in ${fillResult.timeToFillMs}ms at $${fillResult.order.filled_avg_price}`);
          
          // Record entry fill and slippage
          if (lifecycleTrade && fillResult.order.filled_avg_price) {
            const filledPrice = parseFloat(fillResult.order.filled_avg_price);
            tradeLifecycle.recordEntryFill(
              lifecycleTrade.tradeId,
              filledPrice,
              parseInt(fillResult.order.filled_qty || "0"),
              new Date().toISOString()
            );
            
            // P6: Track slippage (signal price vs fill price in basis points)
            // Use lifecycle trade's strategy and time window for consistent metadata
            const slippageBps = ((filledPrice - recommendation.price) / recommendation.price) * 10000;
            const strategyForSlippage = lifecycleTrade.strategy || recommendation.strategyName || strategyType;
            const timeWindowForSlippage = lifecycleTrade.metadata?.timeWindow || tradeLifecycle.determineTimeWindow();
            executionQuality.recordSlippage(strategyForSlippage, timeWindowForSlippage, slippageBps);
            console.log(`[P6] Slippage: ${slippageBps.toFixed(1)}bps strategy=${strategyForSlippage} window=${timeWindowForSlippage}`);
          }
        } else if (fillResult.timedOut) {
          // Timeout - cancel order and skip
          console.log(`ACTION=SKIP symbol=${recommendation.symbol} SKIP_REASONS=[entry:unfilledTimeout]`);
          skipCounters.recordSkip("UNFILLED_TIMEOUT");
          executionQuality.recordOrderTimedOut();
          
          // TRADE ACCOUNTING: Record cancellation for timeout
          tradeAccounting.recordCancel({
            tradeId,
            symbol: recommendation.symbol,
            strategyName: strategyType,
            alpacaOrderId: order.id,
            reason: "FILL_TIMEOUT",
          });
          
          await executionQuality.cancelUnfilledOrder(order.id);
          await storage.updateTradeStatus(trade.id, "cancelled", order.id);
          
          // DO NOT chase price upward - just return
          return false;
        } else {
          // Order cancelled or rejected by broker
          console.log(`ACTION=SKIP symbol=${recommendation.symbol} SKIP_REASONS=[entry:orderRejected]`);
          skipCounters.recordSkip("ORDER_REJECTED");
          await storage.updateTradeStatus(trade.id, order.status, order.id);
          return false;
        }
      } else {
        // Immediate fill (rare for limit orders, common in dry-run)
        const timeToFillMs = Date.now() - submitTime;
        executionQuality.recordOrderFilled(timeToFillMs);
        console.log(`[P6] Order IMMEDIATELY FILLED in ${timeToFillMs}ms`);
        
        if (lifecycleTrade && order.filled_avg_price) {
          const filledPrice = parseFloat(order.filled_avg_price);
          tradeLifecycle.recordEntryFill(
            lifecycleTrade.tradeId,
            filledPrice,
            parseInt(order.filled_qty),
            order.filled_at || new Date().toISOString()
          );
          
          // P6: Track slippage (signal price vs fill price in basis points)
          // Use lifecycle trade's strategy and time window for consistent metadata
          const slippageBps = ((filledPrice - recommendation.price) / recommendation.price) * 10000;
          const strategyForSlippage = lifecycleTrade.strategy || recommendation.strategyName || strategyType;
          const timeWindowForSlippage = lifecycleTrade.metadata?.timeWindow || tradeLifecycle.determineTimeWindow();
          executionQuality.recordSlippage(strategyForSlippage, timeWindowForSlippage, slippageBps);
          console.log(`[P6] Slippage: ${slippageBps.toFixed(1)}bps strategy=${strategyForSlippage} window=${timeWindowForSlippage}`);
        }
      }
    } else {
      // EXPLICIT GUARD: All non-buy orders (sell, exit) use regular market orders
      // This ensures exits work cleanly without bracket complications
      console.log(`[MARKET] ${recommendation.symbol}: Exit/sell order - using regular market order (not bracket)`);
      order = await alpaca.submitOrder(
        recommendation.symbol,
        recommendation.quantity,
        recommendation.side, // Should always be "sell" here
        "market",
        undefined,
        recommendation.reason,
        tradeId
      );
    }

    await storage.updateTradeStatus(trade.id, order.status, order.id);

    await storage.createActivityLog({
      type: "trade",
      action: settings.isPaperTrading ? `Paper ${recommendation.side.toUpperCase()}` : `${recommendation.side.toUpperCase()} Order`,
      description: `${recommendation.side.toUpperCase()} ${recommendation.quantity} shares of ${recommendation.symbol} at $${recommendation.price.toFixed(2)}`,
    });

    // Sync positions after trade
    setTimeout(syncPositions, 2000);

    return true;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    
    // TRADE ACCOUNTING: Record submission failure only if PROPOSE was recorded
    // This maintains invariant: proposed = submitted + rejected + suppressed
    if (proposalRecorded) {
      tradeAccounting.recordSubmitFail({
        tradeId,
        symbol: recommendation.symbol,
        strategyName: strategyType,
        limitPrice: recommendation.price,
        qty: recommendation.quantity,
        errorCode: "EXCEPTION",
        errorMessage,
      });
    }
    
    await storage.createActivityLog({
      type: "error",
      action: "Trade Failed",
      description: `Failed to execute ${recommendation.side} for ${recommendation.symbol}: ${errorMessage}`,
    });
    return false;
  }
}

export async function closePosition(symbol: string): Promise<boolean> {
  const settings = await storage.getSettings();

  try {
    const position = await storage.getPosition(symbol);
    if (!position) {
      return false;
    }

    // Look up tradeId from positionManager for HIGH confidence pairing
    const managedPos = positionManager.getManagedPositions().find(p => p.symbol === symbol);
    const tradeId = (managedPos as any)?.tradeId || undefined;
    
    console.log(`ACTION=EXIT symbol=${symbol} side=sell reason=manual_close trade_id=${tradeId || 'UNKNOWN'}`);

    // Close position via Alpaca (works for both paper and live)
    await alpaca.closePosition(symbol, "manual_close", tradeId);
    setTimeout(syncPositions, 2000);

    await storage.createActivityLog({
      type: "trade",
      action: "Position Closed",
      description: `Closed position in ${symbol}`,
    });

    return true;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    await storage.createActivityLog({
      type: "error",
      action: "Close Failed",
      description: `Failed to close ${symbol}: ${errorMessage}`,
    });
    return false;
  }
}

export async function closeAllPositions(): Promise<boolean> {
  const settings = await storage.getSettings();

  try {
    // Close all positions via Alpaca (works for both paper and live)
    await alpaca.closeAllPositions();
    setTimeout(syncPositions, 2000);

    await storage.createActivityLog({
      type: "trade",
      action: "All Positions Closed",
      description: "Closed all open positions",
    });

    return true;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error";
    await storage.createActivityLog({
      type: "error",
      action: "Close All Failed",
      description: errorMessage,
    });
    return false;
  }
}

// Market open watcher - catches the opening bell and starts trading immediately
let marketOpenWatcherInterval: NodeJS.Timeout | null = null;
let lastMarketOpenDate: string | null = null;

function startMarketOpenWatcher(): void {
  if (marketOpenWatcherInterval) {
    clearInterval(marketOpenWatcherInterval);
  }
  
  // FORT KNOX: Trading starts at 9:35 AM ET (5 min after market open)
  console.log("[Bot] Starting market open watcher - FORT KNOX: Trading starts 9:35 AM ET");
  
  // Check every 30 seconds for market open
  marketOpenWatcherInterval = setInterval(async () => {
    try {
      const clock = await alpaca.getClock();
      const todayStr = getEasternTime().dateString; // Use Eastern Time for day comparison
      
      // If market just opened today and we haven't processed today's open yet
      if (clock.is_open && lastMarketOpenDate !== todayStr) {
        // Check if we're past our trading start time (9:35 AM ET = 5 min after open)
        const tradingStatus = timeGuard.getTradingStatus();
        
        // Only start trading if we're within our FORT KNOX trading window
        if (tradingStatus.canEnterNewPositions) {
          console.log("[Bot] ENTRY WINDOW OPEN! FORT KNOX schedule active.");
          lastMarketOpenDate = todayStr;
          
          await storage.createActivityLog({
            type: "system",
            action: "Entry Window Open",
            description: `FORT KNOX: Entry window started at ${tradingStatus.currentTimeET}. Entry cutoff: 11:35 AM ET. Force close: 3:45 PM ET (12:45 PM PT).`,
          });
          
          await storage.createAlert({
            type: "info",
            title: "FORT KNOX Entry Window Active",
            message: "Entry window: 9:35 AM - 11:35 AM ET. Force close: 3:45 PM ET (12:45 PM Pacific).",
            requiresApproval: false,
          });
          
          // Reset daily stats for new day
          ato.resetDailyStats();
          
          // Run immediate analysis cycle
          console.log("[Bot] Running market open analysis...");
          await runAnalysisCycle();
        }
      }
    } catch (error) {
      console.error("[Bot] Market open watcher error:", error);
    }
  }, 30 * 1000); // Check every 30 seconds
}

function stopMarketOpenWatcher(): void {
  if (marketOpenWatcherInterval) {
    clearInterval(marketOpenWatcherInterval);
    marketOpenWatcherInterval = null;
  }
}

// Self-scheduling control loop - uses Alpaca clock + Time Guard to control when analysis runs
let selfSchedulingInterval: NodeJS.Timeout | null = null;
let lastAnalysisTime: number = 0;
let analysisLoopRunning = false;

/**
 * Compute combined status reason with proper precedence:
 * 1) MARKET_CLOSED (clock.is_open=false)
 * 2) BEFORE_ENTRY_WINDOW
 * 3) ENTRY_WINDOW_ACTIVE
 * 4) AFTER_ENTRY_CUTOFF_MANAGE_ONLY
 * 5) FORCE_CLOSE_REQUIRED
 */
function getCombinedReason(marketOpen: boolean, tgStatus: ReturnType<typeof timeGuard.getTimeGuardStatus>): string {
  // Priority 1: Market closed takes precedence
  if (!marketOpen) {
    return "MARKET_CLOSED";
  }
  // Priority 2-5: Use time guard reason directly (already in precedence order)
  return tgStatus.reason;
}

/**
 * Compute effective force close - only true when market is open AND time guard says force close
 * This prevents any close attempts during off-hours
 */
function getEffectiveForceClose(marketOpen: boolean, tgStatus: ReturnType<typeof timeGuard.getTimeGuardStatus>): boolean {
  return marketOpen && tgStatus.shouldForceClose;
}

function startSelfSchedulingLoop(): void {
  if (selfSchedulingInterval) {
    clearInterval(selfSchedulingInterval);
  }
  
  const scanIntervalMs = dayTraderConfig.DAY_TRADER_CONFIG.SCAN_INTERVAL_MINUTES * 60 * 1000;
  const envDryRun = alpaca.isDryRun();
  const autoTestEnabled = autoTestReporter.isAutoTestModeEnabled();
  
  console.log("[CONTROL] ============================================");
  console.log("[CONTROL] SELF-SCHEDULING CONTROL LOOP STARTED");
  console.log(`[CONTROL] DRY_RUN (env): ${envDryRun ? 'ON' : 'OFF'}`);
  if (autoTestEnabled) {
    console.log(`[CONTROL] AUTO_TEST_MODE: ENABLED (auto-DRY_RUN 9:30-9:40 ET, report at 10:30 ET)`);
  }
  if (isSimTimeActive()) {
    console.log(`[SIM] SIM_TIME_ET active: ${getSimTimeString()}`);
  }
  if (process.env.SIM_CLOCK_OPEN !== undefined) {
    console.log(`[SIM] SIM_CLOCK_OPEN active: ${process.env.SIM_CLOCK_OPEN}`);
  }
  if (envDryRun) {
    console.log(`[CONTROL] *** DRY RUN MODE *** No real orders will be placed`);
  }
  console.log("[CONTROL] ============================================");
  
  // Control loop runs every 30 seconds
  selfSchedulingInterval = setInterval(async () => {
    try {
      const et = getEasternTime();
      const tgStatus = timeGuard.getTimeGuardStatus();
      let clock: { is_open: boolean; next_open: string; next_close: string } | null = null;
      
      try {
        clock = await alpaca.getClock();
      } catch (err) {
        console.log("[CONTROL] Unable to fetch Alpaca clock - assuming market closed");
      }
      
      const marketOpen = clock?.is_open ?? false;
      alpaca.setMarketOpenState(marketOpen);  // Sync market state for order dry-run checks
      const combinedReason = getCombinedReason(marketOpen, tgStatus);
      const effectiveForceClose = getEffectiveForceClose(marketOpen, tgStatus);
      
      // Auto-test mode: compute effective DRY_RUN (env OR forced 9:30-9:40)
      const autoTestStatus = autoTestReporter.getAutoTestStatus(marketOpen);
      const dryRunEffective = autoTestReporter.isDryRunEffective(marketOpen);
      
      // Verification logging with standardized flags
      console.log(`[CONTROL] ----------------------------------------`);
      console.log(`[CONTROL] Eastern Time: ${et.displayTime} | Date: ${et.dateString}`);
      console.log(`[CONTROL] Alpaca Clock: is_open=${marketOpen}, next_open=${clock?.next_open ?? 'unknown'}, next_close=${clock?.next_close ?? 'unknown'}`);
      console.log(`[CONTROL] Time Guard: canOpenNewTrades=${tgStatus.canOpenNewTrades}, canManagePositions=${tgStatus.canManagePositions}, shouldForceClose=${tgStatus.shouldForceClose}`);
      console.log(`[CONTROL] Combined Reason: ${combinedReason} | effectiveForceClose: ${effectiveForceClose}`);
      console.log(`[CONTROL] Analysis Loop: ${analysisLoopRunning ? 'RUNNING' : 'STOPPED'} | DRY_RUN: ${dryRunEffective ? 'ON' : 'OFF'} | Scan Interval: ${dayTraderConfig.DAY_TRADER_CONFIG.SCAN_INTERVAL_MINUTES} min`);
      
      // Auto-test mode logging
      if (autoTestStatus.enabled) {
        if (autoTestStatus.forceDryRun) {
          console.log(`[AUTO_TEST] FORCE DRY_RUN ACTIVE (9:30-9:40 ET window)`);
        }
        if (autoTestStatus.reportPending) {
          console.log(`[AUTO_TEST] Report pending (will generate at 10:30 ET)`);
        }
      }
      
      // Handle auto-test tasks (report generation at 10:30 ET)
      if (clock) {
        autoTestReporter.handleAutoTestTasks(marketOpen, clock, tgStatus);
      }
      
      // Decision logic based on combined reason (proper precedence)
      if (combinedReason === "MARKET_CLOSED") {
        // Market closed - stop analysis if running
        if (analysisLoopRunning) {
          console.log("[CONTROL] Market CLOSED - stopping analysis loop");
          analysisLoopRunning = false;
        }
        // Check if time guard thinks force close but market is closed
        if (tgStatus.shouldForceClose) {
          console.log("[CONTROL] Force close suppressed (market closed)");
        }
        skipCounters.recordSkip("MARKET_CLOSED");
        // Record guard-skip tick so botWasRunning=true even when market closed
        activityLedger.recordTick({
          symbolsEvaluated: 0,
          validQuotes: 0,
          validBars: 0,
          noSignalCount: 0,
          skipCount: 1,
          skipReasonCounts: { "MARKET_CLOSED": 1 },
          signalsGenerated: 0,
          tradesProposed: 0,
          tradesSubmitted: 0,
          tradesFilled: 0,
        });
        return;
      }
      
      if (combinedReason === "FORCE_CLOSE_REQUIRED") {
        // Past 3:45 PM ET AND market is open - execute force close
        if (analysisLoopRunning) {
          console.log("[CONTROL] FORCE CLOSE REQUIRED - stopping analysis loop");
          analysisLoopRunning = false;
        }
        // Only actually close if effectiveForceClose is true (market open + shouldForceClose)
        if (effectiveForceClose) {
          console.log("[CONTROL] Executing force close (market open + shouldForceClose)");
          // Force close is handled by timeGuard.startTimeGuard() interval
        }
        skipCounters.recordSkip("FORCE_CLOSE_REQUIRED");
        // Record guard-skip tick so botWasRunning=true even during force close
        activityLedger.recordTick({
          symbolsEvaluated: 0,
          validQuotes: 0,
          validBars: 0,
          noSignalCount: 0,
          skipCount: 1,
          skipReasonCounts: { "FORCE_CLOSE_REQUIRED": 1 },
          signalsGenerated: 0,
          tradesProposed: 0,
          tradesSubmitted: 0,
          tradesFilled: 0,
        });
        return;
      }
      
      // Track whether analysis was executed this iteration (for activity ledger)
      let analysisExecuted = false;
      
      if (combinedReason === "ENTRY_WINDOW_ACTIVE") {
        // Within entry window (9:35-11:35 AM ET) - run analysis
        analysisLoopRunning = true;
        
        // Check if enough time has passed since last analysis
        const timeSinceLastAnalysis = Date.now() - lastAnalysisTime;
        if (timeSinceLastAnalysis >= scanIntervalMs) {
          console.log(`[CONTROL] ENTRY WINDOW ACTIVE - running analysis cycle`);
          lastAnalysisTime = Date.now();
          analysisExecuted = true;
          await runAnalysisCycle();
        } else {
          const waitSecs = Math.round((scanIntervalMs - timeSinceLastAnalysis) / 1000);
          console.log(`[CONTROL] ENTRY WINDOW ACTIVE - waiting ${waitSecs}s until next analysis`);
        }
      } else if (combinedReason === "AFTER_ENTRY_CUTOFF_MANAGE_ONLY") {
        // Within management window (11:35 AM - 3:45 PM ET) - no new entries, manage positions
        if (analysisLoopRunning) {
          console.log("[CONTROL] MANAGEMENT WINDOW - stopping new entry analysis, position management only");
          analysisLoopRunning = false;
        }
        skipCounters.recordSkip("AFTER_ENTRY_CUTOFF");
        // Position manager continues running independently
      } else if (combinedReason === "BEFORE_ENTRY_WINDOW") {
        // Before trading start (before 9:35 AM ET)
        if (analysisLoopRunning) {
          console.log("[CONTROL] PRE-MARKET - waiting for entry window");
          analysisLoopRunning = false;
        }
        skipCounters.recordSkip("BEFORE_ENTRY_WINDOW");
      }
      
      // ACTIVITY LEDGER: Record a tick every control loop iteration
      // This ensures botWasRunning=true in reports even when analysis is skipped
      if (!analysisExecuted) {
        // Record a guard-skip tick when analysis didn't run
        activityLedger.recordTick({
          symbolsEvaluated: 0,
          validQuotes: 0,
          validBars: 0,
          noSignalCount: 0,
          skipCount: 1,
          skipReasonCounts: { [combinedReason]: 1 },
          signalsGenerated: 0,
          tradesProposed: 0,
          tradesSubmitted: 0,
          tradesFilled: 0,
        });
      }
      // Note: When analysisExecuted=true, recordTick is called inside runAnalysisCycle()
      
      // PROOF BUNDLE: Auto-capture snapshots during key windows
      if (marketOpen && !entryWindowProof.isFinalized()) {
        const proofSnapshot = {
          timestamp: new Date().toISOString(),
          combinedReason,
          marketOpen,
          tgStatus: {
            canOpenNewTrades: tgStatus.canOpenNewTrades,
            canManagePositions: tgStatus.canManagePositions,
            shouldForceClose: tgStatus.shouldForceClose,
          },
          gateLogCount: entryWindowProof.getProofStatus().gateLogCount,
          analysisLoopRunning,
        };
        
        if (entryWindowProof.shouldCaptureBeforeEntry()) {
          entryWindowProof.captureSnapshot('before_entry', proofSnapshot);
        } else if (entryWindowProof.shouldCaptureDuringEntry()) {
          entryWindowProof.captureSnapshot('during_entry', proofSnapshot);
        } else if (entryWindowProof.shouldCaptureAfterEntry()) {
          entryWindowProof.captureSnapshot('after_entry', proofSnapshot);
          // Finalize proof bundle during after-entry window (OPS-ENTRY-PROOF-1)
          const topReasons = skipCounters.getTopSkipReasons(10);
          const summary = activityLedger.getTodaysSummary();
          entryWindowProof.finalizeProofBundle(topReasons, summary.symbolsEvaluated);
        }
      }
      
    } catch (error) {
      console.error("[CONTROL] Control loop error:", error);
    }
  }, 30 * 1000); // Check every 30 seconds
  
  // Run immediately on startup
  (async () => {
    const et = getEasternTime();
    const tgStatus = timeGuard.getTimeGuardStatus();
    let clock: { is_open: boolean; next_open: string; next_close: string } | null = null;
    
    try {
      clock = await alpaca.getClock();
    } catch (err) {
      console.log("[CONTROL] Unable to fetch Alpaca clock on startup");
    }
    
    const marketOpen = clock?.is_open ?? false;
    alpaca.setMarketOpenState(marketOpen);  // Sync market state for order dry-run checks
    const combinedReason = getCombinedReason(marketOpen, tgStatus);
    const effectiveForceClose = getEffectiveForceClose(marketOpen, tgStatus);
    const dryRunEffective = autoTestReporter.isDryRunEffective(marketOpen);
    
    console.log("[CONTROL] ============================================");
    console.log("[CONTROL] STARTUP VERIFICATION");
    console.log("[CONTROL] ============================================");
    console.log(`[CONTROL] Eastern Time: ${et.displayTime} | Date: ${et.dateString}`);
    console.log(`[CONTROL] Alpaca Clock: is_open=${marketOpen}, next_open=${clock?.next_open ?? 'unknown'}, next_close=${clock?.next_close ?? 'unknown'}`);
    console.log(`[CONTROL] Time Guard: canOpenNewTrades=${tgStatus.canOpenNewTrades}, canManagePositions=${tgStatus.canManagePositions}, shouldForceClose=${tgStatus.shouldForceClose}`);
    console.log(`[CONTROL] Combined Reason: ${combinedReason} | effectiveForceClose: ${effectiveForceClose}`);
    console.log(`[CONTROL] DRY_RUN: ${dryRunEffective ? 'ON' : 'OFF'} | Scan Interval: ${dayTraderConfig.DAY_TRADER_CONFIG.SCAN_INTERVAL_MINUTES} min`);
    console.log("[CONTROL] ============================================");
    
    // If within entry window on startup, run analysis immediately
    if (marketOpen && tgStatus.canOpenNewTrades) {
      console.log("[CONTROL] Within entry window on startup - running immediate analysis");
      analysisLoopRunning = true;
      lastAnalysisTime = Date.now();
      await runAnalysisCycle();
    }
  })();
}

function stopSelfSchedulingLoop(): void {
  if (selfSchedulingInterval) {
    clearInterval(selfSchedulingInterval);
    selfSchedulingInterval = null;
  }
  analysisLoopRunning = false;
}

/**
 * Restart scan loop - idempotent, used by runtime monitor for stall recovery
 */
export async function restartScanLoop(): Promise<void> {
  console.log("[CONTROL] Restarting scan loop (idempotent restart)...");
  
  // If already running, just log and return - idempotent
  if (selfSchedulingInterval) {
    console.log("[CONTROL] Scan loop already running");
    return;
  }
  
  // Only restart if bot is active
  if (botStatus.status !== "active" && botStatus.status !== "paused") {
    console.log(`[CONTROL] Cannot restart scan loop - bot status is ${botStatus.status}`);
    return;
  }
  
  startSelfSchedulingLoop();
  console.log("[CONTROL] Scan loop restarted successfully");
}
