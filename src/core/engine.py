"""Trading engine — main execution loop for AtoBot Trading."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from loguru import logger

from src.config.settings import Settings
from src.data.market_data import MarketDataProvider
from src.exchange.base_client import BaseExchangeClient
from src.intelligence.ai_advisor import AITradeAdvisor
from src.models.order import Order, OrderSide, OrderStatus
from src.models.trade import Trade
from src.notifications.base_notifier import BaseNotifier
from src.persistence.repository import TradingRepository
from src.risk.position_sizer import PositionSizer
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy
from src.strategies.strategy_selector import AdaptiveStrategySelector
from src.utils.helpers import calculate_pnl, decimal_from_str, format_usd


class TradingEngine:
    """Main trading loop that orchestrates strategy execution."""

    def __init__(
        self,
        exchange: BaseExchangeClient,
        risk_manager: RiskManager,
        market_data: MarketDataProvider,
        repository: TradingRepository,
        notifier: BaseNotifier | None,
        settings: Settings,
        # Multi-strategy (preferred)
        strategies: list[BaseStrategy] | None = None,
        # Legacy single-strategy support (for backward compat / tests)
        strategy: BaseStrategy | None = None,
        # AI advisor (optional)
        ai_advisor: AITradeAdvisor | None = None,
        # Scanner suite (optional)
        scanner: "MarketScanner | None" = None,
        news_intel: "NewsIntelligence | None" = None,
        regime_detector: "MarketRegimeDetector | None" = None,
    ) -> None:
        self.exchange = exchange
        # Support both single strategy (legacy/tests) and multi-strategy
        if strategies:
            self.strategies = strategies
        elif strategy:
            self.strategies = [strategy]
        else:
            self.strategies = []
        # Keep .strategy for backward compat (points to first strategy)
        self.strategy = self.strategies[0] if self.strategies else None
        self.risk = risk_manager
        self.market_data = market_data
        self.repository = repository
        self.notifier = notifier
        self.settings = settings
        self.ai_advisor = ai_advisor
        self.scanner = scanner
        self.news_intel = news_intel
        self.regime_detector = regime_detector
        self._stop_event = asyncio.Event()

        # ── Adaptive strategy selector (auto-created if regime detector exists) ──
        self.strategy_selector: AdaptiveStrategySelector | None = None
        if self.regime_detector:
            self.strategy_selector = AdaptiveStrategySelector()
            for strat in self.strategies:
                self.strategy_selector.register_strategy(strat.name)
            logger.info("Strategy selector enabled with {} strategies", len(self.strategies))

        # ── Position Sizer (v5: Kelly + 2% risk + portfolio heat) ─────────
        self.position_sizer = PositionSizer(
            account_size=100_000.0,  # Updated on first balance fetch
            max_risk_per_trade=getattr(settings, "MAX_RISK_PER_TRADE", 0.02),
            max_portfolio_heat=getattr(settings, "MAX_PORTFOLIO_HEAT", 0.06),
            max_position_pct=getattr(settings, "MAX_POSITION_PCT", 0.10),
            kelly_fraction=getattr(settings, "KELLY_FRACTION", 0.5),
            min_trade_history=getattr(settings, "MIN_KELLY_TRADES", 20),
            max_sector_concentration=getattr(settings, "MAX_SECTOR_CONCENTRATION", 0.25),
        )
        # Inject into all strategies
        for strat in self.strategies:
            strat.set_position_sizer(self.position_sizer)
        if self.strategies:
            logger.info(
                "PositionSizer injected | Kelly={} | max_risk={:.0%} | heat_cap={:.0%}",
                settings.KELLY_SIZING_ENABLED,
                self.position_sizer.max_risk_per_trade,
                self.position_sizer.max_portfolio_heat,
            )

        self._consecutive_errors = 0
        self._max_consecutive_errors = 3
        self._tick_count = 0
        self._heartbeat_interval = 60  # ticks
        self._last_scan_tick = 0  # Track last intraday scan

    async def run(self) -> None:
        """Run the main trading loop.

        Loop every ``POLL_INTERVAL_SECONDS``:
        1. Fetch current price for all trading pairs.
        2. Check risk manager status.
        3. Call ``strategy.on_tick()`` for each pair.
        4. Validate each proposed order with risk manager.
        5. Place approved orders on exchange (or log in dry-run).
        6. Check status of existing open orders.
        7. For filled orders call ``strategy.on_order_filled()``.
        8. Check stop-loss for open positions.
        9. Update database.
        10. Periodically send heartbeat / status.

        Exceptions never crash the loop — they are caught, logged, and
        counted. Three consecutive critical errors trigger an emergency
        shutdown.
        """
        logger.info(
            "Trading engine starting | strategies={} | symbols={} | dry_run={} | ai_advisor={}",
            [s.name for s in self.strategies],
            self.settings.SYMBOLS,
            self.settings.DRY_RUN,
            bool(self.ai_advisor and self.ai_advisor._enabled),
        )

        while not self._stop_event.is_set():
            try:
                await self._tick()
                self._consecutive_errors = 0  # Reset on success
            except Exception as exc:
                self._consecutive_errors += 1
                logger.error(
                    "Engine tick error ({}/{}): {}",
                    self._consecutive_errors,
                    self._max_consecutive_errors,
                    exc,
                )
                if self.notifier:
                    await self.notifier.send_error_alert(str(exc))

                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.critical("Too many consecutive errors — emergency shutdown")
                    await self.risk.emergency_shutdown(
                        f"Too many consecutive errors: {exc}",
                        exchange=self.exchange,
                    )
                    if self.notifier:
                        await self.notifier.send_error_alert(
                            "🚨 EMERGENCY SHUTDOWN — too many consecutive errors"
                        )
                    break

            # Wait for next poll interval (or until stopped)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.settings.POLL_INTERVAL_SECONDS,
                )
                break  # stop event was set
            except asyncio.TimeoutError:
                pass  # Normal: timeout means it's time for the next tick

        logger.info("Trading engine stopped")

    async def stop(self) -> None:
        """Signal the engine to stop after the current tick."""
        logger.info("Trading engine stop requested")
        self._stop_event.set()

    # ── Single tick ───────────────────────────────────────────────────────────

    async def _tick(self) -> None:
        """Execute one iteration of the trading loop."""
        self._tick_count += 1

        # 1 & 2 — Check risk status
        if self.risk.is_halted:
            logger.warning("Trading halted: {}", self.risk.halt_reason)
            return

        # Update balance for risk tracking
        try:
            balances = await self.exchange.get_account_balance()
            usd_balance = balances.get("USD", Decimal("0"))
            await self.risk.update_balance(usd_balance)
            # Keep PositionSizer account size in sync (v5)
            if float(usd_balance) > 0:
                self.position_sizer.update_account_size(float(usd_balance))
        except Exception as exc:
            logger.warning("Could not fetch balance: {}", exc)
            balances = {}

        # ── Process streaming trade updates (if available) ────────────────
        await self._process_trade_updates()

        # ── Process streaming news events (if available) ──────────────────
        await self._process_news_events()

        # ── Intraday scanner re-scan (if enabled) ────────────────────────
        await self._run_intraday_scan()

        # ── Regime detection update ───────────────────────────────────────
        await self._update_regime()

        # ── Market hours check ────────────────────────────────────────────
        if self.settings.MARKET_HOURS_ONLY and hasattr(self.exchange, 'is_market_open'):
            try:
                market_open = await self.exchange.is_market_open()
                if not market_open:
                    if self._tick_count % 60 == 1:  # Log every ~5 min at 5s interval
                        # Show next open from calendar if available
                        next_open_str = ""
                        if hasattr(self.exchange, 'get_next_market_open'):
                            try:
                                next_open = await self.exchange.get_next_market_open()
                                if next_open:
                                    next_open_str = f" | next open: {next_open.strftime('%Y-%m-%d %H:%M ET')}"
                            except Exception:
                                pass
                        logger.info(
                            "Market closed — waiting for market hours (tick #{}){}",
                            self._tick_count, next_open_str,
                        )
                    return
            except Exception as exc:
                logger.warning("Could not check market hours: {}", exc)

        # ── EOD flatten check ─────────────────────────────────────────────
        if self.settings.FLATTEN_EOD and hasattr(self.exchange, 'get_market_clock'):
            try:
                clock = await self.exchange.get_market_clock()
                if clock.get("is_open"):
                    from datetime import datetime
                    next_close = datetime.fromisoformat(str(clock["next_close"]).replace("Z", "+00:00"))
                    now = datetime.fromisoformat(str(clock["timestamp"]).replace("Z", "+00:00"))
                    minutes_left = (next_close - now).total_seconds() / 60
                    if minutes_left <= self.settings.FLATTEN_MINUTES_BEFORE_CLOSE:
                        logger.info("EOD flatten: {} min until close — closing all positions", int(minutes_left))
                        if not self.settings.DRY_RUN and hasattr(self.exchange, 'close_all_positions'):
                            await self.exchange.close_all_positions()
                        if self.notifier:
                            await self.notifier.send_message(
                                f"🏠 EOD flatten — closing all positions ({int(minutes_left)} min to close)"
                            )
                        return
            except Exception as exc:
                logger.warning("EOD flatten check error: {}", exc)

        for symbol in self.settings.SYMBOLS:
            for strat in self.strategies:
                try:
                    await self._process_symbol(symbol, balances, strat)
                except Exception as exc:
                    logger.error("Error processing {} [{}]: {}", symbol, strat.name, exc)

        # Stale order cleanup (cancel limit orders older than max age)
        await self._cancel_stale_orders()

        # 10 — Heartbeat
        if self._tick_count % self._heartbeat_interval == 0:
            await self._send_heartbeat()

    async def _process_symbol(
        self, symbol: str, balances: dict[str, Decimal], strategy: BaseStrategy | None = None,
    ) -> None:
        """Process a single trading pair with a specific strategy in one tick."""
        # Use provided strategy or fall back to first (legacy)
        strat = strategy or self.strategy
        if strat is None:
            return

        # ── Halt detection (from streaming trading statuses) ──────────────
        if hasattr(self.exchange, "is_symbol_halted") and self.exchange.is_symbol_halted(symbol):
            logger.debug("Symbol {} is halted — skipping [{}]", symbol, strat.name)
            return

        # ── Scanner edge-score gate (skip low-score symbols) ─────────────
        if self.scanner:
            min_score = getattr(self.settings, "SCANNER_MIN_EDGE_SCORE", 0)
            scan_result = self.scanner.get_symbol_scan(symbol)
            if scan_result and scan_result.edge_score < min_score:
                logger.debug(
                    "Scanner: {} score={:.0f} < min={:.0f} — skipping [{}]",
                    symbol, scan_result.edge_score, min_score, strat.name,
                )
                return

        # ── News avoidance check ──────────────────────────────────────────
        if self.news_intel:
            should_avoid, reason = self.news_intel.should_avoid(symbol)
            if should_avoid:
                logger.info("News avoid: {} — {} [{}]", symbol, reason, strat.name)
                return

        # ── Risk-off regime check (reduce size or skip) ───────────────────
        # Strategy selector gate (regime-aware enable/disable)
        if self.strategy_selector:
            allowed, reason = self.strategy_selector.should_trade(strat.name)
            if not allowed:
                logger.debug("Strategy selector: {} — {} [{}]", symbol, reason, strat.name)
                return

        # 1. Fetch current price
        current_price = await self.market_data.get_current_price(symbol)
        logger.debug("Tick {} | {} [{}] = {}", self._tick_count, symbol, strat.name, current_price)

        # 6. Check existing open orders
        await self._check_open_orders(symbol, strat)

        # 8. Check stop-loss
        pos = strat.positions.get(symbol)
        if pos and not pos.is_closed:
            pos.update_price(current_price)
            if await self.risk.check_stop_loss(pos):
                logger.warning("Stop-loss triggered for {} [{}]", symbol, strat.name)
                await strat.cancel_all(symbol)
                if self.notifier:
                    await self.notifier.send_error_alert(
                        f"Stop-loss triggered for {symbol} [{strat.name}] at {current_price}"
                    )

        # 3. Call strategy.on_tick()
        proposed_orders = await strat.on_tick(symbol, current_price)

        # 4 & 5. Validate and place orders
        for order in proposed_orders:
            # Apply regime size multiplier to BUY orders
            if self.strategy_selector and str(order.side).upper() == "BUY":
                adjusted_qty = float(order.quantity) * self.strategy_selector.get_size_multiplier()
                weight = self.strategy_selector.get_strategy_weight(strat.name)
                adjusted_qty *= weight
                adjusted_qty = max(1.0, adjusted_qty)
                order.quantity = Decimal(str(int(adjusted_qty)))
                if weight != 1.0 or self.strategy_selector.get_size_multiplier() != 1.0:
                    logger.debug(
                        "Size adjusted: {} qty={} (regime={:.2f}x, weight={:.2f}) [{}]",
                        order.symbol, order.quantity,
                        self.strategy_selector.get_size_multiplier(), weight, strat.name,
                    )

            allowed, reason = await self.risk.can_place_order(order, balances)
            if not allowed:
                logger.info(
                    "Order rejected by risk manager: {} | {} [{}]", reason, order.symbol, strat.name
                )
                continue

            # AI advisor evaluation (if enabled)
            if (
                self.ai_advisor
                and self.ai_advisor._enabled
                and str(order.side).upper() == "BUY"
            ):
                try:
                    indicators_data = {}
                    bars = None
                    try:
                        bars_raw = await self.exchange.get_klines(order.symbol, "5m", 40)
                        if bars_raw and len(bars_raw) >= 35:
                            import pandas as pd
                            from src.data import indicators as ind
                            df = pd.DataFrame(bars_raw)
                            for col in ("open", "high", "low", "close", "volume"):
                                df[col] = df[col].astype(float)
                            macd_info = ind.macd_signal(df)
                            rsi_info = ind.rsi_bounce(df)
                            indicators_data = {
                                "macd": macd_info,
                                "rsi": rsi_info,
                                "strategy": strat.name,
                            }
                            bars = bars_raw[-5:]
                    except Exception:
                        pass

                    ai_result = await self.ai_advisor.evaluate_entry(
                        symbol=order.symbol,
                        strategy=strat.name,
                        current_price=float(current_price),
                        indicators_data=indicators_data,
                        recent_bars=bars,
                    )
                    if not ai_result.get("allow", True):
                        logger.info(
                            "Order blocked by AI advisor: {} | {} [{}] | confidence={:.0%} | {}",
                            order.symbol, strat.name,
                            ai_result.get("confidence", 0),
                            ai_result.get("confidence", 0),
                            ai_result.get("reason", "no reason"),
                        )
                        continue
                    elif ai_result.get("confidence", 0.5) < self.settings.AI_MIN_CONFIDENCE:
                        logger.info(
                            "Order blocked by AI low confidence: {} [{}] | confidence={:.0%} (min={:.0%})",
                            order.symbol, strat.name,
                            ai_result.get("confidence", 0),
                            self.settings.AI_MIN_CONFIDENCE,
                        )
                        continue
                except Exception as exc:
                    logger.debug("AI advisor error (allowing trade): {}", exc)

            if self.settings.DRY_RUN:
                # Simulate placement
                order.status = OrderStatus.OPEN
                order.id = f"DRY-{order.internal_id[:8]}"
                logger.info(
                    "[DRY RUN] Would place {} {} {} @ {} qty={} [{}]",
                    order.order_type,
                    order.side,
                    order.symbol,
                    order.price,
                    order.quantity,
                    strat.name,
                )
            else:
                try:
                    if order.order_type == "LIMIT":
                        resp = await self.exchange.place_limit_order(
                            order.symbol, order.side, order.price, order.quantity
                        )
                    else:
                        resp = await self.exchange.place_market_order(
                            order.symbol, order.side, order.quantity
                        )
                    order.id = str(resp.get("orderId", ""))
                    order.status = OrderStatus.OPEN
                    order.exchange_response = resp
                    logger.info(
                        "Order placed | {} {} {} @ {} qty={} | id={} [{}]",
                        order.order_type,
                        order.side,
                        order.symbol,
                        order.price,
                        order.quantity,
                        order.id,
                        strat.name,
                    )
                except Exception as exc:
                    order.mark_failed({"error": str(exc)})
                    logger.error("Failed to place order: {}", exc)
                    continue

            strat.active_orders.append(order)

            # 9. Save to DB
            try:
                await self.repository.save_order(order)
            except Exception as exc:
                logger.error("Failed to save order to DB: {}", exc)

        # Update risk manager open order count (across all strategies)
        total_active = sum(
            len([o for o in s.active_orders if o.is_active])
            for s in self.strategies
        )
        self.risk.set_open_order_count(total_active)

    async def _check_open_orders(self, symbol: str, strategy: BaseStrategy | None = None) -> None:
        """Check exchange for filled/cancelled orders and handle them."""
        strat = strategy or self.strategy
        if strat is None:
            return

        active_orders = [
            o
            for o in strat.active_orders
            if o.symbol == symbol and o.is_active and o.id
        ]

        for order in active_orders:
            try:
                if self.settings.DRY_RUN:
                    # Simulate fills: market orders fill immediately,
                    # limit orders fill when price reaches limit price.
                    current_price = await self.market_data.get_current_price(symbol)
                    should_fill = False
                    fill_price = current_price

                    # Market orders always fill immediately
                    if str(order.order_type).upper() == "MARKET":
                        should_fill = True
                    # Limit buys fill when price <= limit, sells when price >= limit
                    elif str(order.order_type).upper() == "LIMIT":
                        if str(order.side).upper() == "BUY" and current_price <= order.price:
                            should_fill = True
                            fill_price = order.price  # Limit fills at limit price
                        elif str(order.side).upper() == "SELL" and current_price >= order.price:
                            should_fill = True
                            fill_price = order.price

                    if not should_fill:
                        continue

                    # Simulate the fill
                    order.mark_filled(order.quantity)
                    logger.info(
                        "[DRY RUN] Simulated FILL | {} {} {} @ {} qty={} [{}]",
                        order.side, order.symbol, order.order_type,
                        fill_price, order.quantity, strat.name,
                    )

                    # Compute PnL for sell orders
                    trade_pnl = None
                    pos = strat.positions.get(symbol)
                    if str(order.side).upper() == "SELL" and pos and not pos.is_closed:
                        trade_pnl = calculate_pnl(
                            pos.entry_price, fill_price, order.quantity, "BUY"
                        )

                    trade = Trade(
                        symbol=order.symbol,
                        side=order.side,
                        price=fill_price,
                        quantity=order.quantity,
                        fee=Decimal("0"),
                        fee_asset="USD",
                        pnl=trade_pnl,
                        strategy=order.strategy,
                        order_id=order.internal_id,
                    )

                    # Strategy callback
                    follow_up = await strat.on_order_filled(order)
                    for fo in follow_up:
                        strat.active_orders.append(fo)

                    # Track PnL and trade count
                    self.risk.record_trade(trade.pnl)
                    if trade.pnl is not None:
                        await self.risk.update_daily_pnl(trade.pnl)

                    continue  # Done with this dry-run order

                status_resp = await self.exchange.get_order_status(symbol, order.id)
                exchange_status = status_resp.get("status", "")

                if exchange_status == "FILLED":
                    filled_qty = decimal_from_str(
                        status_resp.get("executedQty", str(order.quantity))
                    )
                    # Use actual fill price from exchange (Fix #5)
                    raw_fill_price = status_resp.get("filledAvgPrice")
                    fill_price = (
                        decimal_from_str(raw_fill_price)
                        if raw_fill_price
                        else order.price  # fallback to proposed price
                    )
                    order.mark_filled(filled_qty)
                    order.exchange_response = status_resp
                    logger.info(
                        "Order FILLED | {} {} @ {} (proposed {}) qty={}",
                        order.side,
                        order.symbol,
                        fill_price,
                        order.price,
                        filled_qty,
                    )

                    # Compute PnL for sell orders (Fix #2)
                    trade_pnl = None
                    pos = strat.positions.get(order.symbol)
                    if str(order.side).upper() == "SELL" and pos and not pos.is_closed:
                        trade_pnl = calculate_pnl(
                            pos.entry_price, fill_price, filled_qty, "BUY"
                        )

                    # Create trade record with actual fill price
                    trade = Trade(
                        symbol=order.symbol,
                        side=order.side,
                        price=fill_price,
                        quantity=filled_qty,
                        fee=decimal_from_str(status_resp.get("commission", "0")),
                        fee_asset=status_resp.get("commissionAsset", "USD"),
                        pnl=trade_pnl,
                        strategy=order.strategy,
                        order_id=order.internal_id,
                    )

                    # Notify
                    if self.notifier:
                        await self.notifier.send_trade_alert(trade)

                    # Save trade
                    try:
                        await self.repository.save_trade(trade)
                    except Exception as exc:
                        logger.error("Failed to save trade: {}", exc)

                    # Strategy callback
                    follow_up = await strat.on_order_filled(order)
                    for fo in follow_up:
                        strat.active_orders.append(fo)

                    # Update order in DB
                    try:
                        await self.repository.update_order(order)
                    except Exception as exc:
                        logger.error("Failed to update order in DB: {}", exc)

                    # Track PnL and trade count (Fix #2 & #3)
                    self.risk.record_trade(trade.pnl)
                    if trade.pnl is not None:
                        await self.risk.update_daily_pnl(trade.pnl)

                elif exchange_status == "CANCELED":
                    order.mark_cancelled()
                    await strat.on_order_cancelled(order)
                    try:
                        await self.repository.update_order(order)
                    except Exception as exc:
                        logger.error("Failed to update cancelled order: {}", exc)

                elif exchange_status == "PARTIALLY_FILLED":
                    filled_qty = decimal_from_str(
                        status_resp.get("executedQty", "0")
                    )
                    prev_filled = order.filled_quantity
                    new_fill = filled_qty - prev_filled
                    order.filled_quantity = filled_qty
                    order.status = OrderStatus.PARTIALLY_FILLED

                    # Record a trade for the new partial fill slice
                    if new_fill > Decimal("0"):
                        raw_fill_price = status_resp.get("filledAvgPrice")
                        fill_price = (
                            decimal_from_str(raw_fill_price)
                            if raw_fill_price
                            else order.price
                        )

                        trade_pnl = None
                        pos = strat.positions.get(order.symbol)
                        if str(order.side).upper() == "SELL" and pos and not pos.is_closed:
                            trade_pnl = calculate_pnl(
                                pos.entry_price, fill_price, new_fill, "BUY"
                            )

                        trade = Trade(
                            symbol=order.symbol,
                            side=order.side,
                            price=fill_price,
                            quantity=new_fill,
                            fee=Decimal("0"),
                            fee_asset="USD",
                            pnl=trade_pnl,
                            strategy=order.strategy,
                            order_id=order.internal_id,
                        )
                        try:
                            await self.repository.save_trade(trade)
                        except Exception as exc:
                            logger.error("Failed to save partial trade: {}", exc)

                        self.risk.record_trade(trade.pnl)
                        if trade.pnl is not None:
                            await self.risk.update_daily_pnl(trade.pnl)

                        logger.info(
                            "Partial fill | {} {} @ {} new_qty={} total={}/{} [{}]",
                            order.side, order.symbol, fill_price,
                            new_fill, filled_qty, order.quantity,
                            strat.name,
                        )

            except Exception as exc:
                logger.warning(
                    "Error checking order {} for {}: {}",
                    order.internal_id[:8],
                    symbol,
                    exc,
                )

    # ── Streaming helpers ────────────────────────────────────────────────────

    async def _process_trade_updates(self) -> None:
        """Drain queued trade-update events pushed by the TradingStream.

        Each event dict has keys: event (fill/partial_fill/canceled/rejected/…),
        order (raw SDK order object dict), timestamp.
        """
        if not hasattr(self.exchange, "drain_trade_updates"):
            return

        events = await self.exchange.drain_trade_updates()
        if not events:
            return

        for evt in events:
            event_type = evt.get("event", "")
            order_id = str(evt.get("order_id", ""))
            symbol = evt.get("symbol", "")

            logger.debug("Stream trade update: {} {} {}", event_type, symbol, order_id)

            # Find the matching local order across all strategies
            matched_order = None
            matched_strat = None
            for strat in self.strategies:
                for o in strat.active_orders:
                    if o.id == order_id and o.is_active:
                        matched_order = o
                        matched_strat = strat
                        break
                if matched_order:
                    break

            if not matched_order:
                logger.debug("Stream update for unknown order {} — skipping", order_id)
                continue

            try:
                if event_type == "fill":
                    filled_qty = decimal_from_str(
                        evt.get("filled_qty", str(matched_order.quantity))
                    )
                    raw_fill_price = evt.get("filled_avg_price")
                    fill_price = (
                        decimal_from_str(raw_fill_price)
                        if raw_fill_price
                        else matched_order.price
                    )
                    matched_order.mark_filled(filled_qty)

                    logger.info(
                        "Stream FILL | {} {} @ {} qty={} [{}]",
                        matched_order.side, symbol, fill_price,
                        filled_qty, matched_strat.name,
                    )

                    trade_pnl = None
                    pos = matched_strat.positions.get(symbol)
                    if str(matched_order.side).upper() == "SELL" and pos and not pos.is_closed:
                        trade_pnl = calculate_pnl(
                            pos.entry_price, fill_price, filled_qty, "BUY"
                        )

                    trade = Trade(
                        symbol=symbol,
                        side=matched_order.side,
                        price=fill_price,
                        quantity=filled_qty,
                        fee=decimal_from_str(evt.get("commission", "0")),
                        fee_asset="USD",
                        pnl=trade_pnl,
                        strategy=matched_order.strategy,
                        order_id=matched_order.internal_id,
                    )

                    if self.notifier:
                        await self.notifier.send_trade_alert(trade)
                    try:
                        await self.repository.save_trade(trade)
                    except Exception as exc:
                        logger.error("Failed to save stream trade: {}", exc)

                    follow_up = await matched_strat.on_order_filled(matched_order)
                    for fo in follow_up:
                        matched_strat.active_orders.append(fo)

                    try:
                        await self.repository.update_order(matched_order)
                    except Exception as exc:
                        logger.error("Failed to update stream order: {}", exc)

                    self.risk.record_trade(trade.pnl)
                    if trade.pnl is not None:
                        await self.risk.update_daily_pnl(trade.pnl)

                elif event_type == "partial_fill":
                    filled_qty = decimal_from_str(
                        evt.get("filled_qty", "0")
                    )
                    prev_filled = matched_order.filled_quantity
                    new_fill = filled_qty - prev_filled
                    if new_fill > Decimal("0"):
                        matched_order.filled_quantity = filled_qty
                        matched_order.status = OrderStatus.PARTIALLY_FILLED

                        raw_fill_price = evt.get("filled_avg_price")
                        fill_price = (
                            decimal_from_str(raw_fill_price)
                            if raw_fill_price
                            else matched_order.price
                        )

                        trade_pnl = None
                        pos = matched_strat.positions.get(symbol)
                        if str(matched_order.side).upper() == "SELL" and pos and not pos.is_closed:
                            trade_pnl = calculate_pnl(
                                pos.entry_price, fill_price, new_fill, "BUY"
                            )

                        trade = Trade(
                            symbol=symbol,
                            side=matched_order.side,
                            price=fill_price,
                            quantity=new_fill,
                            fee=Decimal("0"),
                            fee_asset="USD",
                            pnl=trade_pnl,
                            strategy=matched_order.strategy,
                            order_id=matched_order.internal_id,
                        )
                        try:
                            await self.repository.save_trade(trade)
                        except Exception as exc:
                            logger.error("Failed to save partial stream trade: {}", exc)

                        self.risk.record_trade(trade.pnl)
                        if trade.pnl is not None:
                            await self.risk.update_daily_pnl(trade.pnl)

                        logger.info(
                            "Stream partial fill | {} {} @ {} new={} total={}/{} [{}]",
                            matched_order.side, symbol, fill_price,
                            new_fill, filled_qty, matched_order.quantity,
                            matched_strat.name,
                        )

                elif event_type in ("canceled", "expired", "replaced"):
                    matched_order.mark_cancelled()
                    await matched_strat.on_order_cancelled(matched_order)
                    try:
                        await self.repository.update_order(matched_order)
                    except Exception as exc:
                        logger.error("Failed to update cancelled stream order: {}", exc)
                    logger.info(
                        "Stream {} | {} {} [{}]",
                        event_type, symbol, order_id, matched_strat.name,
                    )

                elif event_type == "rejected":
                    matched_order.mark_failed({"reason": "rejected_by_exchange"})
                    logger.warning(
                        "Stream REJECTED | {} {} [{}]",
                        symbol, order_id, matched_strat.name,
                    )

            except Exception as exc:
                logger.error("Error processing stream trade update: {}", exc)

    async def _process_news_events(self) -> None:
        """Drain queued news events and classify via News Intelligence."""
        if not hasattr(self.exchange, "drain_news_events"):
            return

        events = await self.exchange.drain_news_events()
        if not events:
            return

        for evt in events:
            try:
                headline = evt.get("headline", "")
                symbols = evt.get("symbols", [])
                # Only process news for symbols we are actually trading
                relevant = [s for s in symbols if s in self.settings.SYMBOLS]
                if not relevant:
                    continue

                logger.info("News event for {}: {}", relevant, headline[:80])

                # Classify via News Intelligence (instant, no API calls)
                if self.news_intel:
                    news_event = await self.news_intel.classify(evt)
                    if news_event.actionable and self.notifier:
                        await self.notifier.send_message(
                            f"🚨 News T{news_event.tier}: {', '.join(relevant)} — "
                            f"{headline[:120]} | {news_event.suggested_action}"
                        )

                # Legacy AI advisor analysis (if available and no news_intel)
                elif self.ai_advisor and self.ai_advisor._enabled:
                    if hasattr(self.ai_advisor, "analyze_news"):
                        result = await self.ai_advisor.analyze_news(
                            headline=headline,
                            symbols=relevant,
                            source=evt.get("source", ""),
                            summary=evt.get("summary", ""),
                        )
                        if result and result.get("risk_flag"):
                            logger.warning(
                                "AI news risk flag — {} | {}",
                                relevant, result.get("reason", ""),
                            )
                            if self.notifier:
                                await self.notifier.send_message(
                                    f"📰 News risk: {', '.join(relevant)} — {headline[:120]}"
                                )
            except Exception as exc:
                logger.debug("Error processing news event: {}", exc)

    async def _run_intraday_scan(self) -> None:
        """Periodically re-scan the market for new setups."""
        if not self.scanner:
            return

        scan_interval = getattr(self.settings, "SCANNER_INTERVAL_SECONDS", 60)
        poll_interval = self.settings.POLL_INTERVAL_SECONDS
        ticks_between_scans = max(1, int(scan_interval / poll_interval))

        if (self._tick_count - self._last_scan_tick) < ticks_between_scans:
            return

        self._last_scan_tick = self._tick_count
        try:
            results = await self.scanner.intraday_scan()
            if results:
                top = results[:3]
                logger.info(
                    "Intraday scan: {} results | top: {}",
                    len(results),
                    [(r.symbol, r.signal.value, f"{r.edge_score:.0f}") for r in top],
                )
        except Exception as exc:
            logger.debug("Intraday scan error: {}", exc)

    async def _update_regime(self) -> None:
        """Update market regime detection every ~60 seconds."""
        if not self.regime_detector:
            return

        # Only update every 12 ticks (~60s at 5s poll)
        if self._tick_count % 12 != 0:
            return

        try:
            # Try to get SPY bars for trend detection
            spy_bars = None
            vix_level = None
            try:
                import pandas as pd
                bars_raw = await self.exchange.get_klines("SPY", "5m", 30)
                if bars_raw and len(bars_raw) >= 20:
                    spy_bars = pd.DataFrame(bars_raw)
                    for col in ("open", "high", "low", "close", "volume"):
                        if col in spy_bars.columns:
                            spy_bars[col] = spy_bars[col].astype(float)
            except Exception:
                pass

            regime = await self.regime_detector.update(
                spy_bars=spy_bars,
                vix_level=vix_level,
            )

            # Update strategy selector with new regime weights
            if self.strategy_selector:
                self.strategy_selector.update_from_regime(self.regime_detector)

            # Sync regime multiplier to PositionSizer (v5)
            if hasattr(regime, "size_multiplier"):
                self.position_sizer.update_regime_multiplier(regime.size_multiplier)

            if not regime.is_favorable():
                logger.info("Regime unfavorable: {} — trading cautiously", regime.summary())
        except Exception as exc:
            logger.debug("Regime update error: {}", exc)

    async def _send_heartbeat(self) -> None:
        """Send a periodic status update."""
        try:
            all_status = {}
            for strat in self.strategies:
                all_status[strat.name] = await strat.get_status()

            # Include regime info in heartbeat
            regime_str = ""
            if self.regime_detector:
                r = self.regime_detector.current
                regime_str = f" | regime={r.trend.value}/{r.volatility.value} size={r.size_multiplier:.1f}x"

            logger.info(
                "Heartbeat | tick={} | strategies={}{}", self._tick_count,
                list(all_status.keys()), regime_str,
            )
            if self.notifier:
                lines = [f"💓 Heartbeat | Tick #{self._tick_count}"]
                for name, status in all_status.items():
                    lines.append(
                        f"  [{name}] orders={status.get('active_orders')} "
                        f"PnL={status.get('unrealized_pnl', 'N/A')}"
                    )
                await self.notifier.send_message("\n".join(lines))
        except Exception as exc:
            logger.warning("Heartbeat failed: {}", exc)

    # ── Stale order cleanup ───────────────────────────────────────────────────

    async def _cancel_stale_orders(self) -> None:
        """Cancel limit orders that have been open longer than max age."""
        max_age = getattr(self.settings, "STALE_ORDER_MAX_AGE_SECONDS", 1800)
        if max_age <= 0:
            return

        now = datetime.now(timezone.utc)
        for strat in self.strategies:
            stale = [
                o for o in strat.active_orders
                if o.is_active
                and o.id
                and str(o.order_type).upper() == "LIMIT"
                and (now - o.created_at).total_seconds() > max_age
            ]
            for order in stale:
                try:
                    if not self.settings.DRY_RUN:
                        await self.exchange.cancel_order(order.symbol, order.id)
                    order.mark_cancelled()
                    logger.info(
                        "Stale order cancelled | {} {} {} age={}s [{}]",
                        order.side, order.symbol, order.id,
                        int((now - order.created_at).total_seconds()),
                        strat.name,
                    )
                    await strat.on_order_cancelled(order)
                    try:
                        await self.repository.update_order(order)
                    except Exception:
                        pass
                except Exception as exc:
                    logger.warning("Failed to cancel stale order {}: {}", order.id, exc)

    # ── Daily summary ─────────────────────────────────────────────────────────

    async def send_daily_summary(self) -> None:
        """Build and send an end-of-day summary to the notifier."""
        if not self.notifier:
            return
        try:
            total_positions = sum(
                len([p for p in s.positions.values() if not p.is_closed])
                for s in self.strategies
            )
            wins = self.risk._daily_wins if hasattr(self.risk, "_daily_wins") else 0
            trades = self.risk._daily_trade_count
            win_rate = round((wins / trades) * 100, 1) if trades > 0 else 0.0
            summary = {
                "pnl": format_usd(self.risk.daily_pnl),
                "trades": trades,
                "win_rate": win_rate,
                "open_positions": total_positions,
                "balance": format_usd(self.risk.current_balance),
            }
            if hasattr(self.notifier, "send_daily_summary"):
                await self.notifier.send_daily_summary(summary)
            else:
                await self.notifier.send_message(
                    f"📊 Daily Summary | PnL: {summary['pnl']} | "
                    f"Trades: {trades} | Win: {win_rate}% | "
                    f"Balance: {summary['balance']}"
                )
            logger.info("Daily summary sent: {}", summary)
        except Exception as exc:
            logger.warning("Failed to send daily summary: {}", exc)
