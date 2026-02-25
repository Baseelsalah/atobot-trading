"""Advanced technical indicators for Ultra AtoBot Trading.

Inspired by top trading bots (freqtrade, jesse-ai, FinRL) and professional
day-trading indicator libraries (TA-Lib, WorldQuant 101 Alphas).

Covers: Fibonacci, Pivot Points, Stochastic, ADX, OBV, MFI, CMF,
Ichimoku, Keltner, Donchian, SuperTrend, Parabolic SAR, Williams %R,
CCI, Hull MA, Heikin-Ashi, and composite order-flow signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require(df: pd.DataFrame, cols: list[str], min_rows: int = 1) -> None:
    """Validate DataFrame has required columns and minimum rows."""
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"DataFrame must contain '{c}'. Got: {list(df.columns)}")
    if len(df) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows, got {len(df)}")


# ═══════════════════════════════════════════════════════════════════════════════
# TREND INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════


def adx(df: pd.DataFrame, period: int = 14) -> dict:
    """Average Directional Index — trend strength (0-100).

    Returns dict with adx, plus_di, minus_di, trending (ADX>25), strong (ADX>50).
    """
    _require(df, ["high", "low", "close"], period + 2)
    high, low, close = df["high"], df["low"], df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1))
    adx_val = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    current_adx = float(adx_val.iloc[-1])
    return {
        "adx": current_adx,
        "plus_di": float(plus_di.iloc[-1]),
        "minus_di": float(minus_di.iloc[-1]),
        "trending": current_adx > 25,
        "strong_trend": current_adx > 50,
        "bullish_trend": float(plus_di.iloc[-1]) > float(minus_di.iloc[-1]),
        "series": adx_val,
    }


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> dict:
    """SuperTrend indicator — trend direction with dynamic support/resistance.

    Returns dict with direction (1=up, -1=down), value, changed (flip detected).
    """
    _require(df, ["high", "low", "close"], period + 2)
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    n = len(df)

    # ATR
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(period).mean().values

    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * np.nan_to_num(atr)
    lower_band = hl2 - multiplier * np.nan_to_num(atr)

    final_upper = np.copy(upper_band)
    final_lower = np.copy(lower_band)
    direction = np.ones(n)

    for i in range(1, n):
        if upper_band[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if lower_band[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i - 1]

        if direction[i - 1] == 1:
            if close[i] < final_lower[i]:
                direction[i] = -1
            else:
                direction[i] = 1
        else:
            if close[i] > final_upper[i]:
                direction[i] = 1
            else:
                direction[i] = -1

    st_value = np.where(direction == 1, final_lower, final_upper)

    return {
        "direction": int(direction[-1]),  # 1=bullish, -1=bearish
        "value": float(st_value[-1]),
        "changed": direction[-1] != direction[-2] if n > 1 else False,
        "bullish": direction[-1] == 1,
        "series": pd.Series(st_value, index=df.index),
    }


def parabolic_sar(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02,
                  af_max: float = 0.2) -> dict:
    """Parabolic SAR — trailing stop and trend reversal.

    Returns dict with value, bullish, reversed.
    """
    _require(df, ["high", "low", "close"], 3)
    high, low = df["high"].values, df["low"].values
    n = len(df)
    psar = np.copy(low)
    af = af_start
    bull = True
    ep = high[0]
    hp = high[0]
    lp = low[0]

    for i in range(1, n):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            psar[i] = min(psar[i], low[i - 1], low[max(0, i - 2)])
            if low[i] < psar[i]:
                bull = False
                psar[i] = hp
                af = af_start
                lp = low[i]
            else:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + af_step, af_max)
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
            psar[i] = max(psar[i], high[i - 1], high[max(0, i - 2)])
            if high[i] > psar[i]:
                bull = True
                psar[i] = lp
                af = af_start
                hp = high[i]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + af_step, af_max)

    prev_above = high[-2] > psar[-2] if n > 1 else False
    curr_above = high[-1] > psar[-1]
    return {
        "value": float(psar[-1]),
        "bullish": bull,
        "reversed": prev_above != curr_above,
        "series": pd.Series(psar, index=df.index),
    }


def hull_ma(df: pd.DataFrame, period: int = 9) -> pd.Series:
    """Hull Moving Average — faster, smoother MA with less lag.

    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    _require(df, ["close"], period + 2)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma_half = df["close"].rolling(half).apply(
        lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True
    )
    wma_full = df["close"].rolling(period).apply(
        lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True
    )
    diff = 2 * wma_half - wma_full
    hma = diff.rolling(sqrt_period).apply(
        lambda x: np.average(x, weights=range(1, len(x) + 1)), raw=True
    )
    return hma


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26,
             senkou_b: int = 52) -> dict:
    """Ichimoku Cloud — comprehensive trend system.

    Returns dict with tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou,
    price_above_cloud, bullish_cross, cloud_bullish.
    """
    _require(df, ["high", "low", "close"], senkou_b + 1)
    high, low, close = df["high"], df["low"], df["close"]

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_line = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    chikou = close.shift(-kijun)

    # Current values
    t = float(tenkan_sen.iloc[-1])
    k = float(kijun_sen.iloc[-1])
    price = float(close.iloc[-1])

    # Cloud at current position
    sa = float(senkou_a.iloc[-1]) if not pd.isna(senkou_a.iloc[-1]) else t
    sb = float(senkou_b_line.iloc[-1]) if not pd.isna(senkou_b_line.iloc[-1]) else k
    cloud_top = max(sa, sb)
    cloud_bottom = min(sa, sb)

    return {
        "tenkan_sen": t,
        "kijun_sen": k,
        "senkou_a": sa,
        "senkou_b": sb,
        "price_above_cloud": price > cloud_top,
        "price_below_cloud": price < cloud_bottom,
        "price_in_cloud": cloud_bottom <= price <= cloud_top,
        "bullish_cross": t > k,
        "cloud_bullish": sa > sb,
        "cloud_thickness": abs(sa - sb),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MOMENTUM / OSCILLATOR INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
               smooth: int = 3) -> dict:
    """Stochastic Oscillator — momentum with oversold/overbought zones.

    Returns dict with k, d, oversold, overbought, bullish_cross, bearish_cross.
    """
    _require(df, ["high", "low", "close"], k_period + d_period)
    high_roll = df["high"].rolling(k_period).max()
    low_roll = df["low"].rolling(k_period).min()
    raw_k = 100 * (df["close"] - low_roll) / (high_roll - low_roll).replace(0, 1)
    k = raw_k.rolling(smooth).mean()
    d = k.rolling(d_period).mean()

    k_now, d_now = float(k.iloc[-1]), float(d.iloc[-1])
    k_prev, d_prev = float(k.iloc[-2]), float(d.iloc[-2])

    return {
        "k": k_now,
        "d": d_now,
        "oversold": k_now < 20,
        "overbought": k_now > 80,
        "bullish_cross": k_prev <= d_prev and k_now > d_now,
        "bearish_cross": k_prev >= d_prev and k_now < d_now,
        "series_k": k,
        "series_d": d,
    }


def williams_r(df: pd.DataFrame, period: int = 14) -> dict:
    """Williams %R — momentum oscillator (0 to -100).

    Returns dict with value, oversold (<-80), overbought (>-20).
    """
    _require(df, ["high", "low", "close"], period)
    highest = df["high"].rolling(period).max()
    lowest = df["low"].rolling(period).min()
    wr = -100 * (highest - df["close"]) / (highest - lowest).replace(0, 1)
    val = float(wr.iloc[-1])
    return {
        "value": val,
        "oversold": val < -80,
        "overbought": val > -20,
        "series": wr,
    }


def cci(df: pd.DataFrame, period: int = 20) -> dict:
    """Commodity Channel Index — identifies cyclical trends.

    Returns dict with value, overbought (>100), oversold (<-100).
    """
    _require(df, ["high", "low", "close"], period)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci_val = (tp - sma) / (0.015 * mad.replace(0, 1))
    val = float(cci_val.iloc[-1])
    return {
        "value": val,
        "overbought": val > 100,
        "oversold": val < -100,
        "extreme_high": val > 200,
        "extreme_low": val < -200,
        "series": cci_val,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VOLUME INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════


def obv(df: pd.DataFrame) -> dict:
    """On-Balance Volume — cumulative volume flow.

    Returns dict with value, trend (rising/falling), divergence signals.
    """
    _require(df, ["close", "volume"], 3)
    direction = np.sign(df["close"].diff())
    obv_series = (direction * df["volume"]).cumsum()
    obv_now = float(obv_series.iloc[-1])
    obv_sma = float(obv_series.rolling(20).mean().iloc[-1]) if len(df) >= 20 else obv_now

    # Price-OBV divergence
    price_up = float(df["close"].iloc[-1]) > float(df["close"].iloc[-5]) if len(df) >= 5 else False
    obv_up = obv_now > float(obv_series.iloc[-5]) if len(df) >= 5 else False

    return {
        "value": obv_now,
        "trend": "rising" if obv_now > obv_sma else "falling",
        "bullish_divergence": not price_up and obv_up,  # price down, OBV up
        "bearish_divergence": price_up and not obv_up,  # price up, OBV down
        "series": obv_series,
    }


def mfi(df: pd.DataFrame, period: int = 14) -> dict:
    """Money Flow Index — volume-weighted RSI (0-100).

    Returns dict with value, overbought (>80), oversold (<20).
    """
    _require(df, ["high", "low", "close", "volume"], period + 1)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw_money_flow = tp * df["volume"]
    positive = raw_money_flow.where(tp > tp.shift(1), 0)
    negative = raw_money_flow.where(tp < tp.shift(1), 0)
    pos_sum = positive.rolling(period).sum()
    neg_sum = negative.rolling(period).sum()
    mfi_ratio = pos_sum / neg_sum.replace(0, 1e-10)
    mfi_val = 100 - (100 / (1 + mfi_ratio))
    val = float(mfi_val.iloc[-1])
    return {
        "value": val,
        "overbought": val > 80,
        "oversold": val < 20,
        "series": mfi_val,
    }


def cmf(df: pd.DataFrame, period: int = 20) -> dict:
    """Chaikin Money Flow — accumulation/distribution pressure.

    Returns dict with value (>0 = buying, <0 = selling), strong buy/sell.
    """
    _require(df, ["high", "low", "close", "volume"], period)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / \
          (df["high"] - df["low"]).replace(0, 1)
    mfv = clv * df["volume"]
    cmf_val = mfv.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, 1)
    val = float(cmf_val.iloc[-1])
    return {
        "value": val,
        "buying_pressure": val > 0,
        "selling_pressure": val < 0,
        "strong_buying": val > 0.25,
        "strong_selling": val < -0.25,
        "series": cmf_val,
    }


def vwap_bands(df: pd.DataFrame, std_devs: tuple[float, ...] = (1.0, 2.0)) -> dict:
    """VWAP with standard deviation bands — institutional support/resistance.

    Returns dict with vwap, upper/lower bands at each std dev.
    """
    _require(df, ["high", "low", "close", "volume"], 2)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (tp * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    vwap_line = cum_tp_vol / cum_vol.replace(0, float("nan"))

    # Rolling VWAP std dev
    cum_tp2_vol = (tp**2 * df["volume"]).cumsum()
    variance = (cum_tp2_vol / cum_vol) - vwap_line**2
    std = np.sqrt(variance.clip(lower=0))

    result = {"vwap": float(vwap_line.iloc[-1])}
    for sd in std_devs:
        result[f"upper_{sd}"] = float(vwap_line.iloc[-1] + sd * std.iloc[-1])
        result[f"lower_{sd}"] = float(vwap_line.iloc[-1] - sd * std.iloc[-1])
    result["series"] = vwap_line
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════


def keltner_channels(df: pd.DataFrame, ema_period: int = 20,
                     atr_period: int = 14, multiplier: float = 2.0) -> dict:
    """Keltner Channels — EMA-based volatility channel.

    Returns dict with upper, middle (EMA), lower, squeeze (BB inside Keltner).
    """
    _require(df, ["high", "low", "close"], max(ema_period, atr_period) + 1)
    middle = df["close"].ewm(span=ema_period, adjust=False).mean()

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(atr_period).mean()

    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val

    return {
        "upper": float(upper.iloc[-1]),
        "middle": float(middle.iloc[-1]),
        "lower": float(lower.iloc[-1]),
        "width": float((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1] * 100),
        "price_above_upper": float(df["close"].iloc[-1]) > float(upper.iloc[-1]),
        "price_below_lower": float(df["close"].iloc[-1]) < float(lower.iloc[-1]),
    }


def donchian_channels(df: pd.DataFrame, period: int = 20) -> dict:
    """Donchian Channels — highest high / lowest low breakout system.

    Returns dict with upper, lower, middle, breakout_high, breakout_low.
    """
    _require(df, ["high", "low", "close"], period)
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    middle = (upper + lower) / 2
    price = float(df["close"].iloc[-1])
    return {
        "upper": float(upper.iloc[-1]),
        "lower": float(lower.iloc[-1]),
        "middle": float(middle.iloc[-1]),
        "breakout_high": price >= float(upper.iloc[-1]),
        "breakout_low": price <= float(lower.iloc[-1]),
        "width_pct": float((upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1] * 100),
    }


def squeeze_momentum(df: pd.DataFrame, bb_period: int = 20, bb_mult: float = 2.0,
                      kc_period: int = 20, kc_mult: float = 1.5) -> dict:
    """Squeeze Momentum — Bollinger Band inside Keltner Channel detection.

    When BB is narrower than KC, volatility is compressed (squeeze).
    Breakout from squeeze = high-probability move.

    Returns dict with squeeze_on, momentum value, momentum direction.
    """
    _require(df, ["high", "low", "close"], max(bb_period, kc_period) + 1)

    # Bollinger Bands
    bb_mid = df["close"].rolling(bb_period).mean()
    bb_std = df["close"].rolling(bb_period).std()
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    # Keltner Channels
    kc_mid = df["close"].ewm(span=kc_period, adjust=False).mean()
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    kc_atr = tr.rolling(kc_period).mean()
    kc_upper = kc_mid + kc_mult * kc_atr
    kc_lower = kc_mid - kc_mult * kc_atr

    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    # Momentum (linear regression of price - midline)
    delta = df["close"] - (df["high"].rolling(kc_period).max() + df["low"].rolling(kc_period).min()) / 2
    mom = delta.rolling(kc_period).mean()

    return {
        "squeeze_on": bool(squeeze_on.iloc[-1]),
        "squeeze_off": not bool(squeeze_on.iloc[-1]),
        "momentum": float(mom.iloc[-1]),
        "momentum_rising": float(mom.iloc[-1]) > float(mom.iloc[-2]) if len(df) > 1 else False,
        "firing_long": not bool(squeeze_on.iloc[-1]) and float(mom.iloc[-1]) > 0,
        "firing_short": not bool(squeeze_on.iloc[-1]) and float(mom.iloc[-1]) < 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE PATTERN INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════


def pivot_points(df: pd.DataFrame) -> dict:
    """Classic Pivot Points — institutional support/resistance levels.

    Uses previous bar's H/L/C to compute today's pivots.
    Returns dict with pp, r1-r3, s1-s3.
    """
    _require(df, ["high", "low", "close"], 2)
    h = float(df["high"].iloc[-2])
    l = float(df["low"].iloc[-2])
    c = float(df["close"].iloc[-2])
    pp = (h + l + c) / 3
    return {
        "pp": pp,
        "r1": 2 * pp - l,
        "r2": pp + (h - l),
        "r3": h + 2 * (pp - l),
        "s1": 2 * pp - h,
        "s2": pp - (h - l),
        "s3": l - 2 * (h - pp),
        "current_price": float(df["close"].iloc[-1]),
        "above_pivot": float(df["close"].iloc[-1]) > pp,
    }


def fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> dict:
    """Fibonacci Retracement Levels from recent swing high/low.

    Returns dict with fib levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
    and nearest support/resistance.
    """
    _require(df, ["high", "low", "close"], lookback)
    window = df.iloc[-lookback:]
    swing_high = float(window["high"].max())
    swing_low = float(window["low"].min())
    diff = swing_high - swing_low
    price = float(df["close"].iloc[-1])

    levels = {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "fib_236": swing_high - 0.236 * diff,
        "fib_382": swing_high - 0.382 * diff,
        "fib_500": swing_high - 0.500 * diff,
        "fib_618": swing_high - 0.618 * diff,
        "fib_786": swing_high - 0.786 * diff,
    }

    # Find nearest support and resistance
    all_levels = sorted(levels.values())
    supports = [l for l in all_levels if l < price]
    resistances = [l for l in all_levels if l > price]
    levels["nearest_support"] = supports[-1] if supports else swing_low
    levels["nearest_resistance"] = resistances[0] if resistances else swing_high
    levels["current_price"] = price
    return levels


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Heikin-Ashi candles — smoothed candle representation.

    Returns a new DataFrame with ha_open, ha_high, ha_low, ha_close, trend.
    """
    _require(df, ["open", "high", "low", "close"], 2)
    ha = pd.DataFrame(index=df.index)
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha["ha_open"] = 0.0
    ha["ha_open"].iloc[0] = (float(df["open"].iloc[0]) + float(df["close"].iloc[0])) / 2
    for i in range(1, len(df)):
        ha["ha_open"].iloc[i] = (ha["ha_open"].iloc[i - 1] + ha["ha_close"].iloc[i - 1]) / 2
    ha["ha_high"] = pd.concat([df["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(axis=1)
    ha["ha_low"] = pd.concat([df["low"], ha["ha_open"], ha["ha_close"]], axis=1).min(axis=1)
    ha["trend"] = np.where(ha["ha_close"] > ha["ha_open"], 1, -1)
    return ha


# ═══════════════════════════════════════════════════════════════════════════════
# ORDER FLOW / TAPE READING SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════


def volume_profile(df: pd.DataFrame, bins: int = 20) -> dict:
    """Volume Profile — identify high-volume price levels (POC, VAH, VAL).

    POC = Point of Control (highest volume price)
    VAH/VAL = Value Area High/Low (70% of volume)
    """
    _require(df, ["high", "low", "close", "volume"], 5)
    prices = (df["high"] + df["low"] + df["close"]) / 3
    price_min, price_max = float(prices.min()), float(prices.max())
    if price_max == price_min:
        return {"poc": price_min, "vah": price_min, "val": price_min}

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    vol_profile = np.zeros(bins)

    for i in range(len(df)):
        p = float(prices.iloc[i])
        v = float(df["volume"].iloc[i])
        bin_idx = min(int((p - price_min) / (price_max - price_min) * bins), bins - 1)
        vol_profile[bin_idx] += v

    poc_idx = int(np.argmax(vol_profile))
    poc = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)

    # Value area (70% of total volume)
    total_vol = vol_profile.sum()
    target = total_vol * 0.70
    sorted_bins = np.argsort(vol_profile)[::-1]
    cumul = 0.0
    va_bins = set()
    for idx in sorted_bins:
        va_bins.add(idx)
        cumul += vol_profile[idx]
        if cumul >= target:
            break

    va_indices = sorted(va_bins)
    vah = float(bin_edges[max(va_indices) + 1])
    val_ = float(bin_edges[min(va_indices)])

    current = float(df["close"].iloc[-1])
    return {
        "poc": poc,
        "vah": vah,
        "val": val_,
        "above_poc": current > poc,
        "in_value_area": val_ <= current <= vah,
        "current_price": current,
    }


def relative_volume(df: pd.DataFrame, period: int = 20) -> dict:
    """Relative Volume (RVOL) — compare current vs historical volume.

    Returns rvol ratio, volume_surge flag, and volume_dry flag.
    """
    _require(df, ["volume"], period + 1)
    avg_vol = float(df["volume"].iloc[:-1].tail(period).mean())
    current_vol = float(df["volume"].iloc[-1])
    rvol = current_vol / avg_vol if avg_vol > 0 else 0
    return {
        "rvol": round(rvol, 2),
        "current_volume": current_vol,
        "avg_volume": avg_vol,
        "volume_surge": rvol >= 2.0,
        "above_average": rvol >= 1.0,
        "volume_dry": rvol < 0.5,
    }


def tape_reading_signals(df: pd.DataFrame) -> dict:
    """Synthetic tape reading from OHLCV data — infer order flow.

    Analyzes candle structure to detect:
    - Aggressive buying (close near high, high volume)
    - Aggressive selling (close near low, high volume)
    - Absorption (high volume, small body = institutional defense)
    - Exhaustion (volume climax with reversal candle)
    """
    _require(df, ["open", "high", "low", "close", "volume"], 5)
    last = df.iloc[-1]
    prev = df.iloc[-2]
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    v = float(last["volume"])
    body = abs(c - o)
    rng = h - l if h != l else 0.001
    body_ratio = body / rng

    # Close position within range
    close_pos = (c - l) / rng  # 0=at low, 1=at high

    # Volume relative to recent
    avg_vol = float(df["volume"].iloc[-6:-1].mean())
    vol_ratio = v / avg_vol if avg_vol > 0 else 1

    # Hammer / Shooting Star detection
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    hammer = lower_wick > 2 * body and upper_wick < body and body_ratio < 0.4
    shooting_star = upper_wick > 2 * body and lower_wick < body and body_ratio < 0.4

    return {
        "aggressive_buying": close_pos > 0.75 and vol_ratio > 1.5,
        "aggressive_selling": close_pos < 0.25 and vol_ratio > 1.5,
        "absorption": vol_ratio > 2.0 and body_ratio < 0.3,
        "exhaustion_top": shooting_star and vol_ratio > 2.0,
        "exhaustion_bottom": hammer and vol_ratio > 2.0,
        "hammer": hammer,
        "shooting_star": shooting_star,
        "close_position": round(close_pos, 3),
        "body_ratio": round(body_ratio, 3),
        "volume_ratio": round(vol_ratio, 2),
        "bullish_candle": c > o,
        "doji": body_ratio < 0.1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SIGNALS (Multi-indicator confluence)
# ═══════════════════════════════════════════════════════════════════════════════


def confluence_score(df: pd.DataFrame) -> dict:
    """Multi-indicator confluence score (0-100) for trade quality.

    Combines: RSI, MACD, Stoch, ADX, OBV, SuperTrend, Squeeze.
    Higher score = stronger signal confluence.
    """
    _require(df, ["high", "low", "close", "volume"], 55)

    score = 0
    signals = {}
    max_score = 0

    # RSI (weight: 15)
    try:
        from src.data.indicators import rsi as calc_rsi
        rsi_val = float(calc_rsi(df, 14).iloc[-1])
        signals["rsi"] = rsi_val
        max_score += 15
        if 30 < rsi_val < 70:
            score += 10  # neutral zone
        elif rsi_val <= 30:
            score += 15  # oversold (buy opportunity)
        # overbought = 0 points (risky)
    except Exception:
        pass

    # MACD (weight: 15)
    try:
        from src.data.indicators import macd_signal
        m = macd_signal(df)
        signals["macd_bullish"] = m["bullish"]
        max_score += 15
        if m["bullish"]:
            score += 10
        if m["golden_cross"]:
            score += 15
        elif m["above_zero"]:
            score += 5
    except Exception:
        pass

    # Stochastic (weight: 10)
    try:
        s = stochastic(df)
        signals["stoch_k"] = s["k"]
        max_score += 10
        if s["bullish_cross"]:
            score += 10
        elif s["oversold"]:
            score += 8
    except Exception:
        pass

    # ADX (weight: 15)
    try:
        a = adx(df)
        signals["adx"] = a["adx"]
        max_score += 15
        if a["trending"] and a["bullish_trend"]:
            score += 15
        elif a["trending"]:
            score += 5
    except Exception:
        pass

    # OBV trend (weight: 10)
    try:
        o = obv(df)
        signals["obv_trend"] = o["trend"]
        max_score += 10
        if o["trend"] == "rising":
            score += 10
        if o["bullish_divergence"]:
            score += 10  # bonus
    except Exception:
        pass

    # SuperTrend (weight: 15)
    try:
        st = supertrend(df)
        signals["supertrend"] = st["direction"]
        max_score += 15
        if st["bullish"]:
            score += 15
        if st["changed"] and st["bullish"]:
            score += 5  # bonus for fresh flip
    except Exception:
        pass

    # Squeeze Momentum (weight: 10)
    try:
        sq = squeeze_momentum(df)
        signals["squeeze"] = sq["squeeze_on"]
        max_score += 10
        if sq["firing_long"]:
            score += 10
        elif sq["squeeze_on"]:
            score += 5  # building energy
    except Exception:
        pass

    # Volume (weight: 10)
    try:
        rv = relative_volume(df)
        signals["rvol"] = rv["rvol"]
        max_score += 10
        if rv["volume_surge"]:
            score += 10
        elif rv["above_average"]:
            score += 5
    except Exception:
        pass

    # Normalize to 0-100
    normalized = min(100, int((score / max_score * 100) if max_score > 0 else 0))
    return {
        "score": normalized,
        "raw_score": score,
        "max_possible": max_score,
        "signals": signals,
        "strong": normalized >= 70,
        "moderate": 40 <= normalized < 70,
        "weak": normalized < 40,
    }
