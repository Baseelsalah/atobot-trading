"""Utility / helper functions for AtoBot Trading."""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal


def round_price(price: Decimal, tick_size: Decimal) -> Decimal:
    """Round a price down to the nearest tick_size increment.

    Args:
        price: The raw price value.
        tick_size: Exchange tick size (e.g. Decimal("0.01")).

    Returns:
        Price rounded to the allowed precision.
    """
    if tick_size <= Decimal("0"):
        return price
    return (price / tick_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * tick_size


def round_quantity(quantity: Decimal, step_size: Decimal) -> Decimal:
    """Round a quantity down to the nearest step_size increment.

    Args:
        quantity: The raw quantity value.
        step_size: Exchange lot/step size (e.g. Decimal("0.001")).

    Returns:
        Quantity rounded to the allowed precision.
    """
    if step_size <= Decimal("0"):
        return quantity
    return (quantity / step_size).quantize(Decimal("1"), rounding=ROUND_DOWN) * step_size


def calculate_pnl(
    entry_price: Decimal,
    exit_price: Decimal,
    quantity: Decimal,
    side: str,
) -> Decimal:
    """Calculate realised PnL for a closed or partially-closed position.

    Args:
        entry_price: Average entry price.
        exit_price: Exit / fill price.
        quantity: Quantity being closed.
        side: ``"BUY"`` (long) or ``"SELL"`` (short).

    Returns:
        Signed PnL in quote currency.
    """
    if side.upper() in ("BUY", "LONG"):
        return (exit_price - entry_price) * quantity
    else:
        return (entry_price - exit_price) * quantity


def format_usd(amount: Decimal) -> str:
    """Format a USD amount with a dollar sign and commas.

    Examples::

        format_usd(Decimal("1234.5")) -> "$1,234.50"
        format_usd(Decimal("-50"))     -> "-$50.00"
    """
    negative = amount < Decimal("0")
    abs_val = abs(amount)
    formatted = f"${abs_val:,.2f}"
    return f"-{formatted}" if negative else formatted


def format_shares(amount: Decimal, symbol: str) -> str:
    """Format a share quantity with its symbol.

    Examples::

        format_shares(Decimal("10.5"), "AAPL") -> "10.5 AAPL"
    """
    # Strip trailing zeros but keep at least 2 decimal places
    normalised = amount.normalize()
    # Ensure at least "0.00" style if very small
    if normalised == Decimal("0"):
        return f"0.00 {symbol}"
    return f"{normalised} {symbol}"


def decimal_from_str(value: str | float | int) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: String, float, or int to convert.

    Returns:
        Decimal representation.
    """
    return Decimal(str(value))
