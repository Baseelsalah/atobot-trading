"""Smart Order Manager — bracket orders, partial profit-taking, scale-in.

See ``src/orders/__init__.py`` for the full SmartOrderManager and BracketOrder classes.
Re-export for convenience.
"""

from src.orders import BracketOrder, BracketState, SmartOrderManager

__all__ = ["SmartOrderManager", "BracketOrder", "BracketState"]
