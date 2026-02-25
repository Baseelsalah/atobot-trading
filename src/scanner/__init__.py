"""Market scanner & intelligence package for AtoBot.

Provides:
- ``MarketScanner`` — pre-market gapper / intraday momentum scanner
- ``MarketRegimeDetector`` — bull/bear/chop regime classification
- ``NewsIntelligence`` — real-time news sentiment & catalyst classification
"""

from src.scanner.market_scanner import MarketScanner
from src.scanner.regime_detector import MarketRegimeDetector
from src.scanner.news_intel import NewsIntelligence

__all__ = ["MarketScanner", "MarketRegimeDetector", "NewsIntelligence"]
