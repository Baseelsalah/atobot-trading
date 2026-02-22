"""Main bot orchestrator for AtoBot Trading."""

from __future__ import annotations

import asyncio

from loguru import logger

from src.config.settings import Settings
from src.core.engine import TradingEngine
from src.data.market_data import MarketDataProvider
from src.exchange.base_client import BaseExchangeClient
from src.notifications.base_notifier import BaseNotifier
from src.notifications.telegram_notifier import TelegramNotifier
from src.persistence.database import close_database, init_database
from src.persistence.repository import TradingRepository
from src.risk.risk_manager import RiskManager
from src.strategies.base_strategy import BaseStrategy
from src.strategies.momentum_strategy import MomentumStrategy
from src.strategies.orb_strategy import ORBStrategy
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

        # 8. Create and run engine
        self.engine = TradingEngine(
            exchange=self.exchange,
            strategies=self.strategies,
            risk_manager=self.risk_manager,
            market_data=self.market_data,
            repository=self.repository,
            notifier=self.notifier,
            settings=self.settings,
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
            if hasattr(self.exchange, 'close_all_positions'):
                try:
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
            "orb": ORBStrategy,
            "vwap_scalp": VWAPScalpStrategy,
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

    async def _reconcile_positions(self) -> None:
        """Sync in-memory strategy positions with actual exchange positions.

        On restart the bot has empty position dicts. If there are real
        open positions on the exchange (e.g. from a previous session that
        crashed), we load them into the *first* strategy so they get
        managed (stop-loss, exit logic) rather than being orphaned.
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

        for pos_data in exchange_positions:
            symbol = pos_data["symbol"]
            if symbol in tracked:
                logger.debug("Reconcile: {} already tracked — skipping", symbol)
                continue

            side = "LONG" if str(pos_data.get("side", "long")).lower() == "long" else "SHORT"
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
            logger.info(
                "Reconciled position: {} {} {} @ {} qty={} → [{}]",
                side, symbol, position.entry_price,
                position.current_price, position.quantity, primary.name,
            )

        logger.info(
            "Position reconciliation complete: {} exchange positions, {} newly tracked",
            len(exchange_positions),
            len(exchange_positions) - len(tracked & {p["symbol"] for p in exchange_positions}),
        )
