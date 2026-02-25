"""Technical indicators for AtoBot Trading."""

from __future__ import annotations

import pandas as pd


def _validate_df(df: pd.DataFrame, required_col: str, min_rows: int) -> None:
    """Validate that the DataFrame has the required column and sufficient rows."""
    if required_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain a '{required_col}' column. "
            f"Available columns: {list(df.columns)}"
        )
    if len(df) < min_rows:
        raise ValueError(
            f"Insufficient data: need at least {min_rows} rows, got {len(df)}"
        )


def sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        df: OHLCV DataFrame with a 'close' column.
        period: Look-back period.

    Returns:
        pandas Series of SMA values.
    """
    _validate_df(df, "close", period)
    return df["close"].rolling(window=period).mean()


def ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Exponential Moving Average.

    Args:
        df: OHLCV DataFrame with a 'close' column.
        period: Look-back period.

    Returns:
        pandas Series of EMA values.
    """
    _validate_df(df, "close", period)
    return df["close"].ewm(span=period, adjust=False).mean()


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        df: OHLCV DataFrame with a 'close' column.
        period: Look-back period.

    Returns:
        pandas Series of RSI values (0-100).
    """
    _validate_df(df, "close", period + 1)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("inf"))
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    return rsi_series


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence).

    Args:
        df: OHLCV DataFrame with a 'close' column.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line EMA period.

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    _validate_df(df, "close", slow + signal)
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Args:
        df: OHLCV DataFrame with a 'close' column.
        period: SMA look-back period.
        std_dev: Number of standard deviations for upper/lower bands.

    Returns:
        Tuple of (upper_band, middle_band, lower_band).
    """
    _validate_df(df, "close", period)
    middle = df["close"].rolling(window=period).mean()
    rolling_std = df["close"].rolling(window=period).std()
    upper = middle + (rolling_std * std_dev)
    lower = middle - (rolling_std * std_dev)
    return upper, middle, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range.

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close' columns.
        period: Look-back period.

    Returns:
        pandas Series of ATR values.
    """
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column.")
    if len(df) < period + 1:
        raise ValueError(
            f"Insufficient data: need at least {period + 1} rows, got {len(df)}"
        )

    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.rolling(window=period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume-Weighted Average Price.

    Uses the typical price (high + low + close) / 3 weighted by volume.
    Assumes the DataFrame represents a single intraday session.

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close', and 'volume' columns.

    Returns:
        pandas Series of cumulative VWAP values.
    """
    for col in ("high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column.")
    if len(df) < 1:
        raise ValueError("DataFrame must have at least 1 row.")

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return cum_tp_vol / cum_vol.replace(0, float("nan"))


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average of volume.

    Args:
        df: OHLCV DataFrame with a 'volume' column.
        period: Look-back period.

    Returns:
        pandas Series of volume SMA values.
    """
    _validate_df(df, "volume", period)
    return df["volume"].rolling(window=period).mean()


# ── Advanced indicators (from Alpaca ChatGPT example) ────────────────────────


def macd_signal(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> dict:
    """Return MACD signal analysis: golden cross, death cross, and values.

    Args:
        df: OHLCV DataFrame with a 'close' column.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal_period: Signal line EMA period.

    Returns:
        Dict with keys: macd, signal, histogram, golden_cross, death_cross,
        bullish (True if MACD > signal).
    """
    _validate_df(df, "close", slow + signal_period)
    macd_line, signal_line, hist = macd(df, fast, slow, signal_period)

    macd_now = macd_line.iloc[-1]
    macd_prev = macd_line.iloc[-2]
    sig_now = signal_line.iloc[-1]
    sig_prev = signal_line.iloc[-2]

    golden_cross = (macd_prev < sig_prev) and (macd_now > sig_now)
    death_cross = (macd_prev > sig_prev) and (macd_now < sig_now)

    return {
        "macd": macd_now,
        "signal": sig_now,
        "histogram": hist.iloc[-1],
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "bullish": macd_now > sig_now,
        "above_zero": macd_now > 0,
    }


def multi_ma_trend(
    df: pd.DataFrame,
    fast: int = 50,
    mid: int = 100,
    slow: int = 200,
) -> dict:
    """Multi-timeframe moving average trend filter (from Alpaca example).

    Checks if fast MA > mid MA > slow MA (strong uptrend).

    Args:
        df: OHLCV DataFrame with a 'close' column.
        fast: Fast MA period.
        mid: Mid MA period.
        slow: Slow MA period.

    Returns:
        Dict with: in_uptrend, in_downtrend, ma_fast, ma_mid, ma_slow values.
    """
    if len(df) < slow + 1:
        return {
            "in_uptrend": False,
            "in_downtrend": False,
            "ma_fast": None,
            "ma_mid": None,
            "ma_slow": None,
            "insufficient_data": True,
        }

    ma_fast = df["close"].rolling(fast).mean()
    ma_mid = df["close"].rolling(mid).mean()
    ma_slow = df["close"].rolling(slow).mean()

    f = ma_fast.iloc[-1]
    m = ma_mid.iloc[-1]
    s = ma_slow.iloc[-1]

    in_uptrend = (f > m) and (m > s)
    in_downtrend = (f < m) and (m < s)

    return {
        "in_uptrend": in_uptrend,
        "in_downtrend": in_downtrend,
        "ma_fast": f,
        "ma_mid": m,
        "ma_slow": s,
        "insufficient_data": False,
    }


def rsi_bounce(df: pd.DataFrame, period: int = 14, oversold: float = 30.0) -> dict:
    """Detect RSI oversold bounce signal (from Alpaca example).

    Args:
        df: OHLCV DataFrame with a 'close' column.
        period: RSI period.
        oversold: Oversold threshold.

    Returns:
        Dict with: rsi_now, rsi_prev, is_oversold, bounce (prev below, now above),
        retreating (prev above 70, now below 65).
    """
    _validate_df(df, "close", period + 2)
    rsi_series = rsi(df, period)
    rsi_now = rsi_series.iloc[-1]
    rsi_prev = rsi_series.iloc[-2]

    return {
        "rsi_now": rsi_now,
        "rsi_prev": rsi_prev,
        "is_oversold": rsi_now < oversold,
        "bounce": (rsi_prev < oversold) and (rsi_now > oversold),
        "retreating": (rsi_prev > 70) and (rsi_now < 65),
        "overbought": rsi_now > 70,
    }
