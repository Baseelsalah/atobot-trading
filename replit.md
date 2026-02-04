# AtoBot Trading Dashboard

## Overview
AtoBot is an AI-powered stock trading dashboard integrated with the Alpaca trading API. It uses GPT-5 for market analysis and trade recommendations, enabling automated trading with configurable risk management. The dashboard offers a professional interface for monitoring portfolio information, positions, trade history, research, and alerts, designed with Carbon Design System principles. The project aims to provide a robust, AI-driven platform for automated stock trading with a focus on comprehensive risk management and user-friendly interaction.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend
- **Framework**: React 18 with TypeScript.
- **Routing**: Wouter.
- **State Management**: TanStack React Query for real-time data.
- **UI Components**: shadcn/ui built on Radix UI, styled with Tailwind CSS, following Carbon Design System principles.
- **Build Tool**: Vite.

### Backend
- **Runtime**: Node.js with Express.
- **Language**: TypeScript (ES modules).
- **API**: RESTful JSON endpoints (`/api/*`).
- **Development**: Vite middleware for HMR.
- **Production**: esbuild for server code bundling and static file serving.

### Data Storage
- **ORM**: Drizzle ORM (PostgreSQL dialect) with schema in `shared/schema.ts`.
- **Tables**: users, trades, positions, research_logs, activity_logs, alerts, bot_settings.
- **Migrations**: Drizzle Kit.
- **Current Implementation**: In-memory storage with a database interface for future PostgreSQL integration.

### Core System Features
- **Shared Schema Pattern**: Type safety across client/server via `drizzle-zod`.
- **Polling-Based Real-Time Updates**: React Query `refetchInterval` (2-5 seconds).
- **Paper Trading Default**: Safe defaults for risk management.
- **Trading Universe**: 20 ultra-liquid symbols (SPY, QQQ, IWM, DIA, TLT, GLD, SLV, XLF, XLK, XLE, XLV, XLI, XLP, XLU, XLY, AAPL, MSFT, NVDA, AMZN, TSLA).
- **Risk Management System**: Includes Position Manager (partial profit, trailing stops, trade timeout), Day Trader Config (risk % per trade, daily P&L limits, max open positions, daily entry limits), Market Regime Filter (QQQ/SPY trend analysis), Technical Indicators (EMA, ATR, RSI, MACD, Bollinger Bands, VWAP), and FORT KNOX Trading Time Guard (dynamic entry/management/force close windows).
- **Timezone Management**: Centralized module for America/New_York (Eastern Time).
- **Debugging & Analysis Tools**: Skip Reason Counters, Signal Counters & Execution Funnel, Data Health Counters, Indicator Pipeline, Preflight Check.
- **Tradability Gates**: 5 pre-strategy hard gates (spread, liquidity, ATR%, extreme move, time buffer). P1 enhancement: Pre-scan evaluation runs BEFORE strategy evaluation on all symbols in entry universe, with per-symbol PASS/FAIL logging and reason codes. LIQUIDITY-PACE-1: Uses volume pace projection during market hours (projectedDailyVolume = barVolume / minutesSinceOpen × 390) to prevent early-session false blocking. Includes 2-minute warmup mode after open. LIQUIDITY-PACE-MINUTES-1 (Feb 2 2026): Fixed minutesSinceOpen calculation to use getEasternTime() instead of Intl.DateTimeFormat for reliable ET time; clamps to ≥1 during market hours to prevent projectedVol=0.
- **Regime Filter**: SPY EMA9/EMA21 trend filter, blocks trades if bearish.
- **Automated Paper Test Mode**: For testing with automated safeguards and report generation.
- **FAIL-CLOSED Policy**: AI returning no recommendations is valid; trade validator requires minimum 10 bars and score ≥65.
- **Risk Engine**: ATR-based position sizing and adaptive bracket stops with kill-switch, daily safety controls, and callback integration.
- **Bracket Orders**: All BUY entries use Alpaca bracket orders with ATR-based stops.
- **Trade Lifecycle Manager**: Tracks complete trade lifecycle, including trade_id generation, order linking, Alpaca order reconciliation, idempotency guard, and slippage tracking.
- **Idempotency Guard**: Prevents duplicate entries by checking existing positions, open orders, and active trades.
- **Slippage Tracking**: Captures and reports slippage (signal price vs fill price).
- **Realized P&L Reporting**: Daily reports use filled trades for P&L calculations.
- **Measurement & Tuning Loop**: Rich metadata per trade and weekly scorecards for systematic tuning, including regime, time window, ATR analysis, and slippage analysis.
- **Deterministic Strategy Engine**: Generates trade signals without LLM dependency for A/B evaluation, including VWAP Reversion and ORB strategies, with a strategy registry and decision policy.
- **Execution Quality**: Limit order entries with slippage controls, fill timeout, quote freshness gate, and spread near-max gate. Tracks execution metrics like fill rate and cancel rate.
- **Activity Ledger**: Records per-tick scan summaries for truthful reporting, including symbols evaluated, valid quotes/bars, skip counts, and trade attempts.
- **Trade Accounting**: Authoritative trade attempt counters tracking full lifecycle - proposed (passed validation), submitted (Alpaca accepted), rejected (Alpaca rejected), suppressed (blocked by idempotency/preflight), and canceled (timeout). Maintains invariant: proposed = submitted + rejected + suppressed. Integrated into activity ledger summary and snapshot tools.
- **Trade Pairing**: Links bracket order child orders via Alpaca's `parent_order_id` to prevent double counting.
- **No Overnight Safety Guarantee**: Critical system ensuring positions are flat before market close with a multi-step flatten sequence and critical alerts for remaining positions.
- **Report Truth Alignment**: Signal counters in performance reports are sourced from the activity ledger.
- **Validation Test Suite**: Automated testing for system integrity covering activity ledger, trade pairing, restart safety, EOD flatten, rolling reports, and overnight guarantee.
- **Runtime Monitor**: Boot logging, heartbeat, stall detection, and "did run" proof files for autonomous operation. Includes auto-start and health/ping endpoints.
- **Single-Leader Lock** (OPS-PROD-LOCK-1): Uses Object Storage to ensure only one instance can enter trades even if multiple processes start. Lock key `atobot/locks/leader.json` with bootId, acquired on boot, refreshed every 60s. Non-leader instances have `entryAllowed=false` and log CRITICAL. Endpoints: `/debug/leader-status`, `/health` includes `isLeader`.
- **Durable Report Persistence**: Dual-write to filesystem and Object Storage for reports that survive redeploys. Storage categories: activity, runtime, proof, accounting, locks, alerts, execution.
- **Execution Trace System** (EXECUTION-TRACE-DURABLE-1): Durable 50-entry ring buffer persisting EXEC_START/EXEC_FAIL/EXEC_OK traces to Object Storage. Each trace captures: symbol, strategy, tier, tradeId, stage (precheck/risk/alpaca_submit), errorMessage, stack, isLeader, entryAllowed, marketStatus. All 9 failure paths in executeTraderDecision instrumented. Endpoint `/debug/execution-recent` returns ring buffer status and recent traces.
- **Storage Scope Isolation** (STORAGE-SCOPE-1): All Object Storage writes namespaced by envScope. Production writes to `atobot/prod/reports/*`, dev writes to `atobot/dev/reports/*`. Prevents dev/prod data contamination. Endpoint `/debug/storage-status` includes envScope and reportsPrefix fields.
- **Environment Scope Hardening** (ENV-SCOPE-HARDEN-1): Centralized `server/envScope.ts` module as single source of truth for environment scope configuration. Prevents P0 bug where prod URL could read from dev storage. Prod context detected via `REPLIT_DEPLOYMENT=1` OR `NODE_ENV=production`. Startup assertion logs CRITICAL and blocks trading if prod context but `envScope!=prod`. Endpoint `/debug/env` returns envScope, lockKey, reportsPrefix, version, bootId, and blocking status. `shouldBlockEntry()` integrates envScope blocking check.
- **Entry Window Proof Finalization** (OPS-ENTRY-PROOF-1): Proof bundle finalizes at 11:45 ET with explicit FINAL_OK or FINAL_FAIL status. FINAL_OK requires snapshotCount >= 2 AND (gateLogCount > 0 OR symbolsEvaluatedToday > 0). FINAL_FAIL includes reason codes (MISSED_ENTRY_WINDOW, NO_GATES_RAN, NO_SCANS, BOT_NOT_RUNNING, INSUFFICIENT_SNAPSHOTS) and writes CRITICAL alert file to `atobot/reports/alerts/`. Endpoint `/debug/proof-status` includes expectedFinalizeET, currentPhase, passCriteria fields.
- **Downtime Proof System** (OPS-DOWNTIME-PROOF-2): Durable receipts for downtime and boot events persisted to Object Storage. CRITICAL_downtime_detected_{date}.txt written when heartbeat gap detected during market hours. CRITICAL_boot_during_entry_{date}.txt written when process starts during entry window (9:35-11:35 ET). Storage status alerts check all CRITICAL_* keys. Endpoint `/debug/runtime/boots-today` returns boot/shutdown timeline for current day.
- **Alpaca Connectivity Proof** (ALPACA-CONNECTIVITY-PROOF-1): Request timeout (8s) + retry policy (2 retries for timeout/network/429/5xx). Alpaca heartbeat (clock: 45s, account: 5min). Connectivity tracking: alpacaLastOkUTC, alpacaConsecutiveFailures, alpacaLastError, alpacaDegraded (3+ failures). Endpoint `/debug/alpaca-status` with connectivity status and recovery state. `/health` includes tradingState and alpacaDegraded.
- **Recovery Mode** (RECOVERY-MODE-1): State machine for boot during market hours. States: INITIALIZING → RECOVERY_MODE → (ACTIVE_TRADING | MANAGE_ONLY | STOPPED). Recovery checks: leader lock, Alpaca clock/account reachable, positions/orders loaded, risk state. CRITICAL_recovery_failed_{date}.txt persisted on failure.
- **Regime Block Entries Only** (REGIME-BLOCK-ENTRIES-ONLY-1): Changed regime filter from early-return to precheck blocking. When regime=avoid, evaluation pipeline (gates + strategies + validator) still runs for receipts and learning data, but orders are blocked at precheck with REGIME_AVOID_BLOCK_ENTRY reason code. Regime state stored in `server/regimeState.ts` module.
- **Control Loop Trace** (CONTROL-LOOP-TRACE-1): Observability for "mystery zero" debugging. `/health` endpoint includes lastAnalysisRunET, lastSkipReason, tradingStateReason fields. Leader transition detection logs LEADER_CHANGE_RECOMPUTE when isLeader transitions false→true.
- **Export Script**: Generates multi-day performance packets (zip archives) for external analysis.
- **Replay Smoke Test**: Market-hours replay for signal validation without orders, evaluating regime filter, tradability gates, and strategies.

## External Dependencies

### Trading Services
- **Alpaca API**: For market data, account management, and trade execution.

### AI Services
- **Replit AI Integrations**: Utilizes GPT-5 for market analysis and trade recommendations.

### Database
- **PostgreSQL**: For persistent storage.

### Other Key Integrations
- **connect-pg-simple**: For PostgreSQL session storage.