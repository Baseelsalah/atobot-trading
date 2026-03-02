"""Main bot orchestrator for AtoBot Trading."""

from __future__ import annotations

import asyncio

from loguru import logger

from src.config.settings import Settings
from src.core.engine import TradingEngine
from src.data.market_data import MarketDataProvider
from src.exchange.base_client import BaseExchangeClient
from src.intelligence.ai_advisor import AITradeAdvisor
from src.notifications.base_notifier import BaseNotifier
from src.notifications.telegram_notifier import TelegramNotifier
from src.persistence.database import close_database, init_database
from src.persistence.repository import TradingRepository
from src.risk.risk_manager import RiskManager
from src.scanner.market_scanner import MarketScanner
from src.scanner.news_intel import NewsIntelligence
from src.scanner.regime_detector import MarketRegimeDetector
from src.strategies.base_strategy import BaseStrategy
from src.strategies.crypto_strategy import CryptoSwingStrategy
from src.strategies.ema_pullback_strategy import EMAPullbackStrategy
from src.strategies.momentum_strategy import MomentumStrategy
from src.strategies.pairs_strategy import PairsTradingStrategy
from src.strategies.swing_strategy import SwingStrategy
from src.strategies.vwap_strategy import VWAPScalpStrategy
from src.utils.logger import setup_logger


class AtoBot:
    """Top-level bot that wires every component together."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.exchange: BaseExchangeClient | None = None
        self.strategies: list[BaseStrategy] = []
        self.risk_manager: RiskManager | None = None
        self.market_data: MarketDataProvider | None = None
        self.notifier: BaseNotifier | None = None
        self.repository: TradingRepository | None = None
        self.ai_advisor: AITradeAdvisor | None = None
        self.scanner: MarketScanner | None = None
        self.news_intel: NewsIntelligence | None = None
        self.regime_detector: MarketRegimeDetector | None = None
        self.engine: TradingEngine | None = None
        self._shutdown_event = asyncio.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise all components and start the trading engine.

        Steps:
        1. Set up logging.
        2. Validate settings.
        3. Initialise database.
        4. Connect to exchange.
        5. Create risk manager, market data provider, strategy, notifier.
        6. Load previous state from database.
        7. Initialise strategy for each trading pair.
        8. Start the engine loop.
        9. Send startup notification.
        """
        # 1. Logging
        setup_logger(self.settings.LOG_LEVEL, self.settings.LOG_FILE)
        logger.info("AtoBot starting …")
        logger.info(
            "Config: exchange={} strategies={} symbols={} dry_run={}",
            self.settings.EXCHANGE,
            self.settings.STRATEGIES,
            self.settings.SYMBOLS,
            self.settings.DRY_RUN,
        )

        # 3. Database
        session_factory = await init_database(self.settings.DATABASE_URL)
        self.repository = TradingRepository(session_factory)

        # 4. Exchange – select based on config
        self.exchange = self._create_exchange()
        await self.exchange.connect()
        logger.info("Exchange connected")

        # 5. Components
        self.risk_manager = RiskManager(self.settings)
        self.market_data = MarketDataProvider(self.exchange)

        # Strategy (day-trading) — supports multiple simultaneous strategies
        self.strategies = self._create_strategies()
        strategy_names = [s.name for s in self.strategies]
        logger.info("Active strategies: {}", strategy_names)

        # ── Merge crypto symbols into SYMBOLS list if crypto_swing is active ──
        if getattr(self.settings, 'CRYPTO_ENABLED', False) and "crypto_swing" in self.settings.STRATEGIES:
            crypto_syms = [s.strip() for s in self.settings.CRYPTO_SYMBOLS.split(",") if s.strip()]
            existing = set(self.settings.SYMBOLS)
            added = [s for s in crypto_syms if s not in existing]
            if added:
                self.settings.SYMBOLS = list(self.settings.SYMBOLS) + added
                logger.info("Crypto symbols merged: {} (total symbols: {})", added, len(self.settings.SYMBOLS))

        # Notifier
        if self.settings.NOTIFICATIONS_ENABLED:
            self.notifier = TelegramNotifier(
                self.settings.TELEGRAM_BOT_TOKEN,
                self.settings.TELEGRAM_CHAT_ID,
            )
            logger.info("Telegram notifications enabled")
        else:
            self.notifier = None
            logger.info("Notifications disabled")

        # AI Advisor (OpenAI-powered trade evaluation)
        if self.settings.AI_ADVISOR_ENABLED and self.settings.OPENAI_API_KEY:
            self.ai_advisor = AITradeAdvisor(
                api_key=self.settings.OPENAI_API_KEY,
                model=self.settings.OPENAI_MODEL,
            )
            await self.ai_advisor.initialize()
            if self.ai_advisor._enabled:
                logger.info("AI Trade Advisor enabled (model={})", self.settings.OPENAI_MODEL)
                # Generate pre-market briefing
                try:
                    briefing = await self.ai_advisor.generate_market_briefing(self.settings.SYMBOLS)
                    logger.info("AI Market Briefing:\n{}", briefing)
                    if self.notifier:
                        await self.notifier.send_message(f"🤖 AI Market Briefing:\n{briefing}")
                except Exception as exc:
                    logger.warning("AI briefing failed: {}", exc)
        else:
            self.ai_advisor = None
            logger.info("AI Advisor disabled (set OPENAI_API_KEY to enable)")

        # 6. Load previous state
        try:
            state = await self.repository.load_bot_state()
            if state:
                logger.info("Loaded previous bot state: {} keys", len(state))
        except Exception as exc:
            logger.warning("Could not load bot state: {}", exc)

        # 7. Initialise strategies for each symbol
        for strategy in self.strategies:
            for symbol in self.settings.SYMBOLS:
                try:
                    await strategy.initialize(symbol)
                    logger.info("{} initialised for {}", strategy.name, symbol)
                except Exception as exc:
                    logger.error("Failed to initialise {} for {}: {}", strategy.name, symbol, exc)

        # 7b. Reconcile positions — sync in-memory state with exchange
        await self._reconcile_positions()

        # 7c. Start WebSocket streams (if enabled and supported)
        await self._start_streams()

        # 7d. Check corporate announcements (earnings/splits risk)
        await self._check_corporate_announcements()

        # 7e. Initialize Market Scanner, News Intelligence, Regime Detector
        await self._init_scanner_suite()

        # 8. Create and run engine
        self.engine = TradingEngine(
            exchange=self.exchange,
            strategies=self.strategies,
            risk_manager=self.risk_manager,
            market_data=self.market_data,
            repository=self.repository,
            notifier=self.notifier,
            settings=self.settings,
            ai_advisor=self.ai_advisor,
            scanner=self.scanner,
            news_intel=self.news_intel,
            regime_detector=self.regime_detector,
        )

        # 9. Startup notification
        if self.notifier:
            strat_names = ", ".join(s.name for s in self.strategies)
            await self.notifier.send_message(
                "🚀 *AtoBot Started — Day Trading Mode*\n"
                f"Strategies: {strat_names}\n"
                f"Symbols: {', '.join(self.settings.SYMBOLS)}\n"
                f"Paper: {self.settings.ALPACA_PAPER}\n"
                f"Dry Run: {self.settings.DRY_RUN}"
            )

        logger.info("AtoBot is running!")
        await self.engine.run()

    async def stop(self) -> None:
        """Gracefully shut down all components.

        Steps:
        1. Signal engine to stop.
        2. Cancel all open orders (if not dry-run).
        3. Save current state.
        4. Send shutdown notification.
        5. Disconnect from exchange.
        6. Close database.
        """
        logger.info("AtoBot shutting down …")

        # 0. Send daily summary before teardown
        if self.engine:
            try:
                await self.engine.send_daily_summary()
            except Exception as exc:
                logger.warning("Could not send daily summary: {}", exc)

        # 1. Stop engine
        if self.engine:
            await self.engine.stop()

        # 2. Cancel open orders & flatten positions
        if self.strategies and not self.settings.DRY_RUN:
            for strategy in self.strategies:
                for symbol in self.settings.SYMBOLS:
                    try:
                        await strategy.cancel_all(symbol)
                    except Exception as exc:
                        logger.error("Error cancelling orders for {} ({}): {}", symbol, strategy.name, exc)
            # Flatten all positions on shutdown (day trading)
            # Skip swing positions (they hold overnight by design)
            if hasattr(self.exchange, 'close_all_positions'):
                try:
                    # Check if any strategy is swing (exempt from flatten)
                    swing_symbols = set()
                    for strategy in self.strategies:
                        if getattr(strategy, 'exempt_eod_flatten', False):
                            for sym, pos in strategy.positions.items():
                                if not pos.is_closed:
                                    swing_symbols.add(sym)

                    if swing_symbols:
                        # Close only non-swing positions
                        positions = await self.exchange.get_positions()
                        for p in positions:
                            if p['symbol'] not in swing_symbols:
                                await self.exchange.close_position(p['symbol'])
                        logger.info(
                            "Day-trade positions flattened on shutdown (kept {} swing positions)",
                            len(swing_symbols),
                        )
                    else:
                        await self.exchange.close_all_positions()
                        logger.info("All positions flattened on shutdown")
                except Exception as exc:
                    logger.error("Error flattening positions: {}", exc)

        # 3. Save state
        if self.repository and self.strategies:
            try:
                all_status = {}
                for strategy in self.strategies:
                    status = await strategy.get_status()
                    all_status[strategy.name] = status
                await self.repository.save_bot_state({"strategies_status": all_status})
            except Exception as exc:
                logger.error("Failed to save bot state: {}", exc)

        # 4. Notify
        if self.notifier:
            try:
                await self.notifier.send_message("🛑 *AtoBot Stopped*")
            except Exception:
                pass

        # 5. Exchange
        if self.exchange:
            try:
                await self.exchange.disconnect()
            except Exception as exc:
                logger.error("Error disconnecting from exchange: {}", exc)

        # 6. Database
        await close_database()
        logger.info("AtoBot shutdown complete")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _create_exchange(self) -> BaseExchangeClient:
        """Return the correct exchange client based on config."""
        exchange = self.settings.EXCHANGE.lower()
        if exchange == "alpaca":
            from src.exchange.alpaca_client import AlpacaClient
            return AlpacaClient(self.settings)
        elif exchange == "binance":
            from src.exchange.binance_client import BinanceClient
            return BinanceClient(self.settings)
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")

    def _create_strategies(self) -> list[BaseStrategy]:
        """Create strategy instances based on STRATEGIES config."""
        strategy_map = {
            "momentum": MomentumStrategy,
            "vwap_scalp": VWAPScalpStrategy,
            "ema_pullback": EMAPullbackStrategy,
            "pairs": PairsTradingStrategy,
            "swing": SwingStrategy,
            "crypto_swing": CryptoSwingStrategy,
        }
        strategies = []
        for name in self.settings.STRATEGIES:
            cls = strategy_map.get(name)
            if cls is None:
                raise ValueError(f"Unknown strategy: {name}")
            strategies.append(cls(self.exchange, self.risk_manager, self.settings))
        if not strategies:
            raise ValueError("No strategies configured")
        return strategies

    async def _start_streams(self) -> None:
        """Launch WebSocket streaming for prices, order updates, and news."""
        if not self.exchange or not hasattr(self.exchange, "start_streams"):
            return

        streaming = getattr(self.settings, "STREAMING_ENABLED", False)
        trade_stream = getattr(self.settings, "TRADE_STREAM_ENABLED", False)
        news_stream = getattr(self.settings, "NEWS_STREAM_ENABLED", False)

        if not any([streaming, trade_stream, news_stream]):
            logger.info("All streaming disabled — using REST polling only")
            return

        try:
            await self.exchange.start_streams(self.settings.SYMBOLS)

            status_parts = []
            if streaming and hasattr(self.exchange, "is_streaming") and self.exchange.is_streaming():
                status_parts.append("prices")
            if trade_stream and hasattr(self.exchange, "is_trade_streaming") and self.exchange.is_trade_streaming():
                status_parts.append("trades")
            if news_stream:
                status_parts.append("news")

            if status_parts:
                logger.info("WebSocket streams active: {}", ", ".join(status_parts))
                # Fetch data feed info
                data_feed = getattr(self.settings, "DATA_FEED", "iex")
                logger.info("Data feed: {} ({})", data_feed.upper(), "free" if data_feed == "iex" else "paid SIP")
            else:
                logger.info("WebSocket streams configured but not yet connected")
        except Exception as exc:
            logger.warning("Failed to start streams (falling back to REST): {}", exc)

    async def _check_corporate_announcements(self) -> None:
        """Check for upcoming earnings/splits that could affect our symbols."""
        if not self.exchange or not hasattr(self.exchange, "get_corporate_announcements"):
            return

        try:
            from datetime import datetime, timedelta, timezone

            now = datetime.now(timezone.utc)
            since = now - timedelta(days=1)
            until = now + timedelta(days=2)

            announcements = await self.exchange.get_corporate_announcements(
                ca_types=["Dividend", "Merger", "Spinoff", "Split"],
                since=since,
                until=until,
            )
            if announcements:
                for ann in announcements[:10]:  # Cap at 10 announcements
                    symbol = getattr(ann, "symbol", "?")
                    ca_type = getattr(ann, "ca_type", "?")
                    ca_sub_type = getattr(ann, "ca_sub_type", "")
                    if symbol in self.settings.SYMBOLS:
                        logger.warning(
                            "⚠️ Corporate action for {}: {} {} — trade with caution",
                            symbol, ca_type, ca_sub_type,
                        )
                        if self.notifier:
                            await self.notifier.send_message(
                                f"⚠️ Corporate action: {symbol} — {ca_type} {ca_sub_type}"
                            )
                logger.info(
                    "Corporate announcements check: {} found in date range",
                    len(announcements),
                )
            else:
                logger.debug("No corporate announcements found for date range")
        except Exception as exc:
            logger.debug("Corporate announcements check skipped: {}", exc)

    async def _init_scanner_suite(self) -> None:
        """Initialize the Market Scanner, News Intelligence, and Regime Detector."""
        # Market Scanner
        if getattr(self.settings, "SCANNER_ENABLED", False):
            try:
                self.scanner = MarketScanner(
                    exchange=self.exchange,
                    settings=self.settings,
                )
                logger.info("Market Scanner enabled")

                # Run pre-market scan
                try:
                    results = await self.scanner.pre_market_scan()
                    ctx = self.scanner.get_market_context()
                    logger.info(
                        "Pre-market scan: {} results | SPY={:.2f} ({:+.1f}%) | risk_off={}",
                        len(results),
                        ctx.spy_price if ctx else 0,
                        ctx.spy_change_pct if ctx else 0,
                        ctx.risk_off if ctx else False,
                    )
                    if results and self.notifier:
                        top = results[:5]
                        lines = ["🔍 Pre-Market Scanner:"]
                        for r in top:
                            lines.append(
                                f"  {r.symbol} | {r.signal.value} | "
                                f"score={r.edge_score:.0f} gap={r.gap_percent:+.1f}% "
                                f"rvol={r.relative_volume:.1f}x"
                            )
                        await self.notifier.send_message("\n".join(lines))
                except Exception as exc:
                    logger.warning("Pre-market scan failed: {}", exc)
            except Exception as exc:
                logger.warning("Scanner init failed: {}", exc)

        # News Intelligence
        if getattr(self.settings, "NEWS_INTEL_ENABLED", False):
            try:
                self.news_intel = NewsIntelligence(ai_advisor=self.ai_advisor)
                logger.info("News Intelligence enabled")
            except Exception as exc:
                logger.warning("News Intelligence init failed: {}", exc)

        # Market Regime Detector
        if getattr(self.settings, "REGIME_DETECTION_ENABLED", False):
            try:
                self.regime_detector = MarketRegimeDetector(
                    exchange=self.exchange,
                    settings=self.settings,
                )
                logger.info("Market Regime Detector enabled")
            except Exception as exc:
                logger.warning("Regime Detector init failed: {}", exc)

    async def _reconcile_positions(self) -> None:
        """Sync in-memory strategy positions with actual exchange positions.

        On restart the bot has empty position dicts. If there are real
        open positions on the exchange (e.g. from a previous session that
        crashed), we handle them:
        - LONG positions in our SYMBOLS list → track in first strategy
        - SHORT positions or unknown symbols → close immediately (we are long-only)
        """
        if not self.exchange or not hasattr(self.exchange, "get_positions"):
            return

        try:
            exchange_positions = await self.exchange.get_positions()
        except Exception as exc:
            logger.warning("Position reconciliation failed: {}", exc)
            return

        if not exchange_positions:
            logger.info("Position reconciliation: no open positions on exchange")
            return

        # Build a set of symbols already tracked by any strategy
        tracked: set[str] = set()
        for strat in self.strategies:
            tracked.update(strat.positions.keys())

        # Assign untracked positions to the first strategy
        primary = self.strategies[0] if self.strategies else None
        if primary is None:
            return

        from src.models.position import Position

        managed_symbols = {s.upper() for s in self.settings.SYMBOLS}
        newly_tracked = 0
        closed_orphans = 0

        for pos_data in exchange_positions:
            symbol = pos_data["symbol"]
            if symbol in tracked:
                logger.debug("Reconcile: {} already tracked — skipping", symbol)
                continue

            side = "LONG" if str(pos_data.get("side", "long")).lower() == "long" else "SHORT"

            # Close SHORT positions and positions not in our SYMBOLS list
            # We are a long-only day-trading bot
            if side == "SHORT" or symbol not in managed_symbols:
                reason = "SHORT position (we are long-only)" if side == "SHORT" else f"{symbol} not in managed SYMBOLS"
                logger.warning(
                    "Closing orphaned position: {} {} qty={} | Reason: {}",
                    side, symbol, abs(pos_data["qty"]), reason,
                )
                if not self.settings.DRY_RUN and hasattr(self.exchange, 'close_position'):
                    try:
                        await self.exchange.close_position(symbol)
                        closed_orphans += 1
                        logger.info("Closed orphaned position: {} {}", side, symbol)
                    except Exception as exc:
                        logger.error("Failed to close orphaned position {}: {}", symbol, exc)
                continue

            # Track LONG positions in our symbol list
            position = Position(
                symbol=symbol,
                side=side,
                entry_price=pos_data["avg_entry_price"],
                current_price=pos_data["current_price"],
                quantity=abs(pos_data["qty"]),
                strategy=primary.name,
            )
            primary.positions[symbol] = position
            # Seed trailing high for trailing stop
            primary._trailing_highs[symbol] = pos_data["current_price"]
            newly_tracked += 1
            logger.info(
                "Reconciled position: {} {} {} @ {} qty={} → [{}]",
                side, symbol, position.entry_price,
                position.current_price, position.quantity, primary.name,
            )

        logger.info(
            "Position reconciliation complete: {} exchange positions, {} tracked, {} orphans closed",
            len(exchange_positions), newly_tracked, closed_orphans,
        )
