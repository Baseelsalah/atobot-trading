"""Trading engine — main execution loop for AtoBot Trading."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from loguru import logger

from src.config.settings import Settings
from src.data.market_data import MarketDataProvider
from src.exchange.base_client import BaseExchangeClient
from src.models.order import Order, OrderSide, OrderStatus
from src.models.trade import Trade
from src.notifications.base_notifier import BaseNotifier
from src.persistence.repository import TradingRepository
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy
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
        self._stop_event = asyncio.Event()
        self._consecutive_errors = 0
        self._max_consecutive_errors = 3
        self._tick_count = 0
        self._heartbeat_interval = 60  # ticks

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
            "Trading engine starting | strategies={} | symbols={} | dry_run={}",
            [s.name for s in self.strategies],
            self.settings.SYMBOLS,
            self.settings.DRY_RUN,
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
        except Exception as exc:
            logger.warning("Could not fetch balance: {}", exc)
            balances = {}

        # ── Market hours check ────────────────────────────────────────────
        if self.settings.MARKET_HOURS_ONLY and hasattr(self.exchange, 'is_market_open'):
            try:
                market_open = await self.exchange.is_market_open()
                if not market_open:
                    logger.debug("Market closed — skipping tick")
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
            allowed, reason = await self.risk.can_place_order(order, balances)
            if not allowed:
                logger.info(
                    "Order rejected by risk manager: {} | {} [{}]", reason, order.symbol, strat.name
                )
                continue

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

    async def _send_heartbeat(self) -> None:
        """Send a periodic status update."""
        try:
            all_status = {}
            for strat in self.strategies:
                all_status[strat.name] = await strat.get_status()
            logger.info("Heartbeat | tick={} | strategies={}", self._tick_count, list(all_status.keys()))
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
