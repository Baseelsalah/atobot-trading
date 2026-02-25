"""Quick verification that all hardening v5 changes compile and work."""
import sys
try:
    from src.config.settings import Settings
    print("[OK] Settings imports")
    
    from src.risk.risk_manager import RiskManager
    print("[OK] RiskManager imports")
    
    from src.strategies.strategy_selector import AdaptiveStrategySelector
    print("[OK] AdaptiveStrategySelector imports")
    
    from src.scanner.regime_detector import MarketRegimeDetector
    print("[OK] MarketRegimeDetector imports")
    
    from src.core.engine import TradingEngine
    print("[OK] TradingEngine imports")
    
    # Verify new settings exist
    s = Settings(
        EXCHANGE="alpaca",
        ALPACA_API_KEY="test",
        ALPACA_API_SECRET="test",
        ALPACA_PAPER=True,
    )
    assert hasattr(s, "CIRCUIT_BREAKER_ENABLED"), "Missing CIRCUIT_BREAKER_ENABLED"
    assert hasattr(s, "GAP_FILTER_ENABLED"), "Missing GAP_FILTER_ENABLED"
    assert hasattr(s, "ATR_ADAPTIVE_STOPS"), "Missing ATR_ADAPTIVE_STOPS"
    assert hasattr(s, "CRISIS_SIZING_ENABLED"), "Missing CRISIS_SIZING_ENABLED"
    assert hasattr(s, "FORCE_EOD_FLATTEN"), "Missing FORCE_EOD_FLATTEN"
    assert hasattr(s, "EOD_BLOCK_ENTRIES_MINUTES"), "Missing EOD_BLOCK_ENTRIES_MINUTES"
    print("[OK] All new settings present")
    
    # Verify risk manager new methods
    rm = RiskManager(s)
    assert hasattr(rm, "update_symbol_atr"), "Missing update_symbol_atr"
    assert hasattr(rm, "update_spy_open"), "Missing update_spy_open"
    assert hasattr(rm, "check_circuit_breaker"), "Missing check_circuit_breaker"
    assert hasattr(rm, "is_circuit_breaker_active"), "Missing is_circuit_breaker_active"
    rm.update_symbol_atr("SPY", 0.5)
    rm.update_spy_open(450.0)
    assert not rm.check_circuit_breaker(445.0), "CB should not trigger at 1.1% drop"
    assert rm.check_circuit_breaker(430.0), "CB should trigger at 4.4% drop"
    assert rm.is_circuit_breaker_active, "CB should be active after trigger"
    print("[OK] Circuit breaker works")
    
    # Verify strategy selector extreme vol blocking
    sel = AdaptiveStrategySelector()
    sel.register_strategy("vwap_scalp")
    sel.register_strategy("orb")
    sel._volatility_regime = "extreme"
    ok, reason = sel.should_trade("vwap_scalp", "BUY")
    assert not ok, "VWAP BUY should be blocked in extreme vol"
    assert "Extreme volatility" in reason
    ok2, _ = sel.should_trade("vwap_scalp", "SELL")
    assert ok2, "VWAP SELL should be allowed in extreme vol"
    ok3, _ = sel.should_trade("orb", "BUY")
    assert ok3, "ORB BUY should be allowed in extreme vol"
    print("[OK] Extreme vol blocking works")
    
    print("\n=== ALL HARDENING CHECKS PASSED ===")
    
except Exception as e:
    print(f"[FAIL] {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
