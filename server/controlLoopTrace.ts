/**
 * CONTROL-LOOP-TRACE-1: Observability for the trading control loop
 * 
 * Tracks key metrics for debugging "mystery zero" sessions:
 * - lastAnalysisRun: When the analysis cycle last executed
 * - lastSkipReason: Why the last analysis was skipped (if applicable)
 * - tradingStateReason: Why the bot is in its current trading state
 */

import { getEasternTime } from "./timezone";

interface ControlLoopTraceState {
  lastAnalysisRunET: string | null;
  lastAnalysisRunUTC: string | null;
  lastSkipReason: string | null;
  tradingStateReason: string | null;
  analysisRunCount: number;
  analysisSkipCount: number;
  leaderTransitions: number;
  lastLeaderTransitionET: string | null;
  lastLeaderTransitionReason: string | null;
}

let traceState: ControlLoopTraceState = {
  lastAnalysisRunET: null,
  lastAnalysisRunUTC: null,
  lastSkipReason: null,
  tradingStateReason: null,
  analysisRunCount: 0,
  analysisSkipCount: 0,
  leaderTransitions: 0,
  lastLeaderTransitionET: null,
  lastLeaderTransitionReason: null,
};

export function recordAnalysisRun(): void {
  const now = new Date();
  const et = getEasternTime();
  traceState.lastAnalysisRunET = et.displayTime;
  traceState.lastAnalysisRunUTC = now.toISOString();
  traceState.analysisRunCount++;
  traceState.lastSkipReason = null;
}

export function recordAnalysisSkip(reason: string): void {
  traceState.lastSkipReason = reason;
  traceState.analysisSkipCount++;
}

export function setTradingStateReason(reason: string): void {
  traceState.tradingStateReason = reason;
}

export function recordLeaderTransition(fromLeader: boolean, toLeader: boolean, reason: string): void {
  const et = getEasternTime();
  traceState.leaderTransitions++;
  traceState.lastLeaderTransitionET = et.displayTime;
  traceState.lastLeaderTransitionReason = reason;
  console.log(`[LEADER_CHANGE_RECOMPUTE] transition=${fromLeader}->${toLeader} reason=${reason} et=${et.displayTime}`);
}

export function getTraceState(): ControlLoopTraceState {
  return { ...traceState };
}

export function resetDailyTrace(): void {
  traceState.analysisRunCount = 0;
  traceState.analysisSkipCount = 0;
  traceState.lastSkipReason = null;
}
