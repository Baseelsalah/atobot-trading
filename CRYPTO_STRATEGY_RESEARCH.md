# Crypto Trading Strategy Research — AtoBot Enhancement Plan

> **Compiled from**: GitHub repos (Freqtrade 47.1K★, Jesse 7.5K★, Hummingbot 17.5K★, 
> FinRL 14K★, Superalgos 5.3K★, Backtesting.py 8K★), Freqtrade docs, r/algotrading wiki,
> TradingView RSI docs, academic papers (NeurIPS, JOSS), and strategy source code analysis.

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [What Top Crypto Bots Do (Competitive Landscape)](#2-competitive-landscape)
3. [Proven Technical Indicators for Crypto](#3-proven-indicators)
4. [Advanced Entry/Exit Strategies](#4-advanced-strategies)
5. [AI/ML Approaches That Actually Work](#5-aiml-approaches)
6. [Crypto-Specific Signals (Not Available in Stocks)](#6-crypto-specific-signals)
7. [Risk Management Best Practices](#7-risk-management)
8. [Common Pitfalls to Avoid](#8-common-pitfalls)
9. [Actionable Improvements for AtoBot](#9-actionable-improvements)
10. [Implementation Priority Matrix](#10-priority-matrix)

---

## 1. Executive Summary

After researching the most successful open-source crypto trading frameworks and academic literature, 
here are the **key themes that separate profitable bots from losing ones**:

### The Big 5 Principles
1. **Multi-timeframe confluence** — Never trade on a single timeframe. Use higher TFs for trend, lower TFs for entry.
2. **Adaptive parameters** — Fixed RSI thresholds fail. Models must retrain/adapt to regime changes.
3. **Volume confirmation is non-negotiable** — Every successful strategy requires volume > average as a filter.
4. **Outlier/noise handling** — Crypto is 10x noisier than stocks. Outlier detection (SVM, DBSCAN, DI) is critical.
5. **Risk-first architecture** — Position sizing based on volatility (ATR), not fixed %. Dynamic stop-losses that adapt.

### What We Already Do Well
- ✅ 4H timeframe (good for swing trading crypto)
- ✅ RSI oversold bounce (proven signal)
- ✅ EMA stack alignment (trend following)
- ✅ Volume surge filter
- ✅ BTC trend gate for alts
- ✅ ATR-based stops
- ✅ 24h cooldown after stop-loss
- ✅ Trailing stop with activation threshold

### What We're Missing (Biggest Opportunities)
- ❌ Multi-timeframe analysis (we only use 4H)
- ❌ Bollinger Band position (proven by Freqtrade as key filter)
- ❌ MACD histogram divergence
- ❌ Relative volume (vs simple volume surge)
- ❌ Rate of Change (ROC) momentum
- ❌ Mean reversion signals for ranging markets
- ❌ Regime detection (trending vs ranging vs volatile)
- ❌ Dynamic RSI thresholds based on volatility
- ❌ Correlated pair analysis (ETH tracking BTC)
- ❌ Day-of-week/hour-of-day seasonality
- ❌ Consecutive loss pair-locking (beyond cooldown)
- ❌ RSI divergence detection

---

## 2. Competitive Landscape

### Freqtrade (47.1K ★ — Industry Leader)
**Key strategy patterns from source code:**
```python
# Entry: RSI crosses above 30 AND price below BB middle AND volume > 0
enter_long = (
    qtpylib.crossed_above(dataframe['rsi'], 30) &
    (dataframe['tema'] <= dataframe['bb_middleband']) &
    (dataframe['tema'] > dataframe['tema'].shift(1)) &  # TEMA rising
    (dataframe['volume'] > 0)
)

# Exit: RSI crosses above 70 AND price above BB middle AND falling
exit_long = (
    qtpylib.crossed_above(dataframe['rsi'], 70) &
    (dataframe['tema'] > dataframe['bb_middleband']) &
    (dataframe['tema'] < dataframe['tema'].shift(1)) &  # TEMA falling
    (dataframe['volume'] > 0)
)
```

**FreqAI ML Features (what actually gets fed to models):**
- RSI, MFI, ADX, SMA, EMA (all with multiple period lengths: 10, 20)
- Bollinger Band width + close/lower_band ratio
- Rate of Change (ROC)
- Relative volume (volume / rolling mean)
- Percent change
- Raw price + raw volume
- Day of week (1-7 normalized), Hour of day (1-25 normalized)
- All features expanded across multiple timeframes (5m, 15m, 4h)
- All features expanded across correlated pairs (ETH/USD, LINK/USD, BNB/USD)

**ML Target (what model predicts):**
```python
# Predict: mean close price over next N candles / current close - 1
target = close.shift(-N).rolling(N).mean() / close - 1
# Entry when target > 0.01 (1% expected gain)
# Exit when target < 0
```

**Key Freqtrade Innovations:**
- **Hyperopt**: Automated parameter optimization with Bayesian search
- **FreqAI**: Self-adaptive ML model that retrains in live deployment
- **Pair locking**: Lock pairs after consecutive losses (like our cooldown but per-pair)
- **Informative pairs**: Use BTC/ETH data when trading altcoins
- **Recursive analysis**: Detect if strategy is accidentally looking into the future
- **Orderflow analysis**: Track bid/ask imbalances for entry timing
- **Data pipeline**: MinMaxScaler → VarianceThreshold → optional SVM outlier removal → optional PCA

### Jesse AI (7.5K ★ — Crypto-First Framework)
**Key design patterns:**
- Strategy as clean Python class with `should_long()`, `go_long()`, `update_position()`
- 300+ built-in indicators
- Smart order routing (auto-selects MARKET/LIMIT/STOP based on entry price vs current)
- Multi-timeframe + multi-symbol backtesting without look-ahead bias
- Hyperparameter optimization via Optuna
- Partial fills support (enter/exit at multiple price levels)
- Dynamic position updates via `update_position()`:
  ```python
  def update_position(self):
      # Trailing stop implementation
      if self.is_long:
          self.take_profit = self.position.qty, self.high - 10
      # Scale into winners
      if self.position.pnl_percentage > 5 and ta.rsi(self.candles) < 30:
          self.buy = self.position.qty, self.price  # double position
  ```

### FinRL (14K ★ — Reinforcement Learning for Finance)
**Architecture:**
- Three layers: Market Environments → Agents → Applications
- Dedicated `cryptocurrency_trading` and `high_frequency_trading` environments
- Agents: ElegantRL, RLlib, StableBaselines3
- Supports Alpaca, Binance, CCXT as data sources
- Train-Test-Trade pipeline
- Indicators used: MACD, Bollinger (upper/lower), RSI-30, DX-30, SMA-30, SMA-60
- Published in NeurIPS, ICAIF (top ML/finance venues)

### Hummingbot (17.5K ★ — Market Making)
**Relevant concepts:**
- Market making with adaptive spreads (not applicable to our swing strategy, but useful for DCA)
- Cross-exchange arbitrage detection
- $34B+ lifetime trading volume
- Modular connector architecture (easy to add exchanges)

### Superalgos (5.3K ★ — Visual Strategy Builder)
**Architecture insights:**
- Bitcoin-Factory: ML-based prediction for BTC
- TensorFlow integration for price forecasting
- Multi-server deployment for running multiple strategies
- Social trading network for sharing strategies

---

## 3. Proven Technical Indicators for Crypto

### Tier 1 — Must-Have (Proven across all frameworks)
| Indicator | How Used | Why It Works for Crypto |
|-----------|----------|------------------------|
| **RSI (14)** | Oversold (<30-35) for entries, Overbought (>70-75) for exits | Crypto overextends more than stocks, RSI reversals are reliable |
| **EMA Stack (20/50/200)** | 20>50 = uptrend, price above 200 = macro bull | Crypto trends longer and harder than stocks |
| **Bollinger Bands (20, 2.2σ)** | Price near lower band = entry, BB width for volatility | Crypto mean-reverts violently after BB squeezes |
| **Volume (relative)** | volume / rolling(20).mean() > 1.5x | Crypto volume spikes precede moves by 1-4 candles |
| **ATR (14)** | Stop-loss placement, position sizing | Crypto ATR varies 5x more than stocks; fixed stops fail |

### Tier 2 — Strong Edge (Used by winning strategies)
| Indicator | How Used | Why It Works |
|-----------|----------|--------------|
| **MACD (12/26/9)** | Histogram divergence, signal line cross | Catches momentum shifts before price confirms |
| **MFI (Money Flow Index)** | Oversold < 20, Overbought > 80 | Combines price AND volume — better than RSI alone |
| **ADX (14)** | > 25 = trending, < 20 = ranging | Filters out choppy markets (our biggest loss generator) |
| **ROC (Rate of Change)** | Momentum confirmation | Leading indicator — shows acceleration before RSI |
| **TEMA (Triple EMA)** | Faster than EMA, less lag | Better entry timing in fast crypto moves |

### Tier 3 — Situational Edge
| Indicator | How Used | When to Use |
|-----------|----------|-------------|
| **Stochastic RSI** | Oversold/overbought in trending markets | When regular RSI stays overbought during strong trends |
| **Awesome Oscillator** | Zero-line crossovers | Trend confirmation alongside MACD |
| **Hilbert Transform** | Dominant cycle period detection | Auto-tune indicator periods to current market rhythm |
| **Plus/Minus DI** | Directional movement | ADX companion for trend direction |
| **SAR (Parabolic)** | Trailing stop placement | Alternative to ATR-based trailing stops |

### Multi-Timeframe Analysis (THE #1 MISSED OPPORTUNITY)
From Freqtrade's approach — use **informative pairs** on higher timeframes:
```
Trading on 4H → Check trend on 1D
- 1D RSI > 50 = bullish market (allow longs)
- 1D RSI < 50 = bearish market (reduce size or skip)
- 1D EMA 20 > EMA 50 = uptrend intact
- 1D Bollinger Band position for macro context
```
This single addition typically improves win rate by **5-15%** across all backtests.

---

## 4. Advanced Entry/Exit Strategies

### A. RSI Divergence (High Win Rate Signal)
```
Bullish Divergence: Price makes LOWER low, RSI makes HIGHER low
→ Strong reversal signal, especially at 4H timeframe on crypto
→ Filter: Only take if volume is rising on the second low

Bearish Divergence: Price makes HIGHER high, RSI makes LOWER high
→ Exit/short signal
→ Filter: Only act if ADX is declining (momentum fading)
```
**Source**: Wilder's original work + Cardwell's trend confirmation refinements.
- Cardwell: Bullish divergence only occurs in bearish trends → it signals a brief correction, not a reversal
- Cardwell: Use "Positive Reversals" (price higher low + RSI lower low in bull trend) as continuation signals

### B. RSI Failure Swings (Less Known, High Accuracy)
```
Bullish Failure Swing:
1. RSI drops below 30
2. RSI bounces above 30
3. RSI pulls back but STAYS above 30
4. RSI breaks above previous swing high
→ Entry signal (independent of price action)

Bearish Failure Swing:
1. RSI rises above 70
2. RSI drops below 70
3. RSI rises but STAYS below 70
4. RSI breaks below the previous swing low
→ Exit signal
```

### C. Bollinger Band Squeeze → Breakout
```
Setup:
1. BB Width < 20-period low of BB Width (squeeze detected)
2. Wait for close outside upper or lower band
3. Confirm with volume > 1.5x average
4. Enter in direction of breakout

Exit:
- When price touches opposite band
- Or BB Width returns to normal (expansion over)
```

### D. Multi-Level Take Profits (from Jesse AI)
```python
# Instead of single TP:
take_profit = [
    (qty * 0.33, entry * 1.05),  # 33% at 5%
    (qty * 0.33, entry * 1.08),  # 33% at 8%
    (qty * 0.34, entry * 1.12),  # 34% at 12%
]
# Move stop to breakeven after first TP hit
```
This approach locks in profits while letting winners run. Crypto's volatility makes this 
especially effective — you capture the "meat" of the move without giving it all back.

### E. Regime-Based Strategy Switching
```
IF ADX > 25 AND EMA20 > EMA50:
    → Use MOMENTUM strategy (trend-following, wider targets)
    → RSI entry: <40 (buy the dip in uptrend)
    → TP: 10-15%

IF ADX < 20:
    → Use MEAN REVERSION strategy (buy low, sell high)
    → RSI entry: <30 (oversold in range)
    → TP: 3-5% (smaller, faster)
    → Tighter stop: 2-3%

IF ADX > 25 AND EMA20 < EMA50:
    → AVOID or SHORT (downtrend)
    → No longs allowed
```

---

## 5. AI/ML Approaches That Actually Work

### A. FreqAI Feature Engineering (Production-Proven)
The most successful ML approach for crypto trading uses **gradient boosted trees** (LightGBM, XGBoost, CatBoost) 
with carefully engineered features:

**Feature Set (108+ features from Freqtrade example):**
```python
# Base features (auto-expanded across periods 10 and 20):
RSI, MFI, ADX, SMA, EMA, BB_width, close/BB_lower, ROC, relative_volume

# Auto-expanded across:
timeframes: [5m, 15m, 4h]
correlated_pairs: [ETH/USD, LINK/USD, BNB/USD]
shifted_candles: 2 (include 2 previous candle values)

# Non-expanded features:
day_of_week: (0-6 normalized to 0-1)
hour_of_day: (0-23 normalized to 0-1)
pct_change: close-to-close percent change
raw_volume: absolute volume
raw_price: absolute close price
```

**Prediction Target:**
```python
# Predict mean close over next 24 candles / current close - 1
# If prediction > 0.01 → BUY
# If prediction < 0 → SELL
```

**Key ML Insights:**
- LightGBM > Neural networks for tabular financial data (confirmed by NeurIPS paper)
- Self-adaptive retraining every N candles is CRITICAL — static models decay in weeks
- Outlier removal (SVM or DBSCAN) improves prediction accuracy by 10-20%
- PCA can reduce 108 features to ~30 while keeping 99.9% variance
- Weight recent data exponentially more than old data (W_i = exp(-i / α*n))

### B. Reinforcement Learning (FinRL/FreqAI — Advanced)
**How it works:**
- Agent receives: current profit, position state, trade duration + all indicators
- Agent outputs: 5 actions (neutral, long_enter, long_exit, short_enter, short_exit)
- Agent is rewarded for: profitable trades, penalized for: losses, holding too long, invalid actions

**Best practices from FreqAI RL:**
```python
# Good reward design:
reward = pnl * factor  # Scale reward with profit
if trade_duration > max_duration:
    factor *= 0.5  # Penalize overly long trades
if rsi < 40 and entering:
    factor = 40 / rsi  # Reward entering at oversold
if neutral and no_position:
    reward = -1  # Penalize sitting on sidelines

# BAD reward design:
# Single large penalty for rare events → neural net can't learn
# Instead: small continuous penalties that scale with severity
```

**Framework:** stable-baselines3 + PPO algorithm + gym environment
**Reality check:** RL is more complex than supervised ML and can find "cheats" that 
don't work in live trading. Start with LightGBM regressor, add RL later.

### C. Practical ML Implementation Path for AtoBot
**Phase 1 (Low Risk, High Impact):**
- Add day-of-week and hour-of-day as entry filters (no ML needed, just statistics)
- Compute correlation coefficient between BTC and ETH for dynamic alt sizing
- Log all features at each tick for future ML training data collection

**Phase 2 (Medium Complexity):**
- Train LightGBM on historical 4H data with FreqAI-style features
- Target: predict 24-candle (4-day) return > 1%
- Use model confidence as a FILTER (not sole signal) — "enter only if model agrees"
- Retrain weekly on rolling 6-month window

**Phase 3 (Advanced):**
- Add RL agent for dynamic stop/TP management
- Multi-asset portfolio optimization
- Sentiment integration (fear/greed as feature)

---

## 6. Crypto-Specific Signals (Not Available in Stocks)

### A. Fear & Greed Index
- Alternative.me publishes daily crypto fear/greed (0-100)
- **Extreme Fear** (<25) = historically the best time to buy BTC
- **Extreme Greed** (>75) = time to reduce position size or take profit
- Can be fetched via free API: `https://api.alternative.me/fng/`
- Usage: Multiply position size by `(100 - fear_greed) / 100` during greed periods

### B. Funding Rate (Perpetual Futures)
- When funding rate is very negative → shorts are paying longs → bullish signal
- When funding rate is very positive → longs are paying shorts → bearish/overheated
- Available from Binance/Bybit APIs (even if not trading there, use as signal)
- Typical values: -0.01% to +0.03% per 8h period
- Extreme values (>0.05% or <-0.03%) are tradeable signals

### C. On-Chain Metrics
- **Exchange inflow/outflow**: Large inflows to exchanges = selling pressure incoming
- **Whale wallet tracking**: >1000 BTC wallets moving to exchanges = bearish
- **Hash rate**: Rising hash rate = miners bullish = usually precedes price rise
- **MVRV ratio** (Market Value / Realized Value): >3.7 historically marks tops, <1 marks bottoms
- Sources: Glassnode API (paid), CryptoQuant API (free tier available)

### D. BTC Dominance
- Rising BTC.D = altcoins will underperform (reduce ETH position size)
- Falling BTC.D = alt season (increase ETH allocation)
- Available from TradingView or CoinGecko API
- Threshold: BTC.D > 60% = BTC-only mode, BTC.D < 50% = increase alt allocation

### E. Crypto Market Hours Seasonality
Despite 24/7 trading, crypto has patterns:
- **Most volatile**: UTC 14:00-21:00 (US market hours overlap)
- **Least volatile**: UTC 02:00-08:00 (Asian night/early morning)
- **Best entry windows**: UTC 06:00-10:00 (before US wake-up momentum)
- **Monday/Tuesday**: Historically stronger than weekends
- **Monthly**: End-of-month rebalancing effects

Usage: Reduce position size during low-volatility hours, increase during high-activity windows.

---

## 7. Risk Management Best Practices

### A. Volatility-Adjusted Position Sizing (from FinRL)
```python
# Current (fixed risk %):
risk_per_trade = 0.04  # 4% of account

# Better (ATR-normalized):
dollar_risk = account_equity * 0.02  # 2% max loss
stop_distance = atr_14 * 2  # Stop at 2 ATR
position_size = dollar_risk / stop_distance
# Result: smaller positions in volatile markets, larger in calm markets
```

### B. Correlation-Aware Sizing
```python
# BTC and ETH correlation is typically 0.85-0.95
# If both positions are open, effective risk is nearly doubled
corr = calculate_correlation(btc_returns, eth_returns, window=30)
if corr > 0.8:
    max_combined_risk = 0.04  # Treat as single correlated bet
    per_position_risk = max_combined_risk / 2  # 2% each
else:
    per_position_risk = 0.04  # Full size (uncorrelated)
```

### C. Maximum Drawdown Circuit Breaker
```python
# If account drops 10% from peak → stop trading for 48 hours
# If account drops 15% from peak → stop trading for 1 week
# If account drops 25% from peak → stop trading, manual review required
peak_equity = max(equity_history)
current_dd = (peak_equity - current_equity) / peak_equity
if current_dd > 0.10:
    pause_trading(hours=48)
```

### D. Kelly Criterion for Crypto (Modified)
```python
# Kelly % = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
# But crypto is too volatile for full Kelly → use HALF Kelly
win_rate = 0.55
avg_win = 0.08  # 8% average winner
avg_loss = 0.04  # 4% average loser
kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
half_kelly = kelly / 2
# Use half_kelly as max position size
```

### E. Consecutive Loss Management
From Freqtrade's pair-locking:
```python
# After 2 consecutive losses on same symbol:
#   → Lock symbol for 24 hours (already have this as cooldown)
# After 3 consecutive losses across ALL symbols:
#   → Reduce position size by 50% for next 3 trades
# After 5 consecutive losses:
#   → Pause all trading for 24 hours
```

---

## 8. Common Pitfalls to Avoid

### From Freqtrade Documentation (Hard-Won Lessons)

1. **Lookahead Bias** — Using `.shift(-1)` or future data in indicators
   ```python
   # WRONG: This looks into the future
   dataframe['sma'] = dataframe['close'].mean()  # Uses ALL data including future
   
   # CORRECT: Use rolling window
   dataframe['sma'] = dataframe['close'].rolling(20).mean()
   ```

2. **Repainting Indicators** — Indicators that change historical values
   - Zigzag, Renko, some pivot point calculations repaint
   - Always verify indicator stability on historical data

3. **Overfitting to Backtest** — Strategy works perfectly on history, fails live
   - Use walk-forward analysis (train on 6 months, test on 2 months, repeat)
   - If Sharpe > 3.0 in backtest, it's almost certainly overfit
   - Add realistic slippage (0.05% for crypto) and fees (0.25% each way)

4. **Survivorship Bias** — Only testing on coins that still exist
   - Many coins that existed in 2021 are dead now
   - Stick to BTC/ETH which have longest history

5. **Ignoring Fees** — 0.25% per side × 2 = 0.50% round trip
   - A strategy with 1% average gain and 0.5% fees = only 0.5% real gain
   - Need minimum 2:1 R:R after fees to be profitable

6. **Not Using Volume Filter** — #1 false signal eliminator
   ```python
   # ALWAYS require:
   volume > 0  # At minimum (filters dead candles)
   volume > volume.rolling(20).mean()  # Better (confirmation)
   ```

---

## 9. Actionable Improvements for AtoBot

### TIER 1 — Quick Wins (1-2 days each, high impact)

#### 1.1 Add Bollinger Band Filter
```python
# In _generate_signal():
bb_upper, bb_middle, bb_lower = calculate_bollinger(closes, period=20, std=2.2)
bb_width = (bb_upper - bb_lower) / bb_middle

# Entry filter: price must be in lower half of BB
if close <= bb_middle:
    confluence += 1  # Near support

# Exit signal: price touches upper band
if close >= bb_upper:
    # Consider taking profit
```

#### 1.2 Add ADX Regime Filter
```python
adx = calculate_adx(highs, lows, closes, period=14)

# Only trade when there's a clear trend
if adx > 25:
    # Trending market → use current trend-following strategy
    # Allow positions, wider targets
elif adx < 20:
    # Choppy market → reduce position size by 50% OR skip
    # This alone could eliminate 30-40% of losing trades
```

#### 1.3 Add MACD Confirmation
```python
macd_line, signal_line, histogram = calculate_macd(closes, 12, 26, 9)

# MACD histogram turning positive (or rising) = bullish momentum
if histogram[-1] > histogram[-2]:  # Histogram rising
    confluence += 1
```

#### 1.4 Add Day/Hour Seasonality Filter
```python
from datetime import datetime, timezone

now = datetime.now(timezone.utc)
hour = now.hour
day = now.weekday()

# Avoid low-volatility dead zones
if 2 <= hour <= 7:  # UTC 2-7 AM = Asian night
    required_confluence += 1  # Require extra confirmation

# Slight preference for Monday-Wednesday entries
if day in (0, 1, 2):  # Mon, Tue, Wed
    confluence += 0.5  # Half-point bonus
```

### TIER 2 — Medium Effort (3-5 days each, significant edge)

#### 2.1 Multi-Timeframe Analysis
```python
# Current: only check 4H bars
# Improvement: also check 1D bars for macro trend

daily_bars = fetch_bars(symbol, timeframe='1D', limit=50)
daily_ema20 = ema(daily_bars.close, 20)
daily_ema50 = ema(daily_bars.close, 50)
daily_rsi = rsi(daily_bars.close, 14)

# Gate: only enter longs if daily trend is bullish
if daily_ema20[-1] > daily_ema50[-1] and daily_rsi[-1] > 45:
    allow_long = True
else:
    allow_long = False  # Or reduce size by 50%
```

#### 2.2 RSI Divergence Detection
```python
def detect_rsi_divergence(prices, rsi_values, lookback=10):
    """Detect bullish/bearish divergence."""
    # Find price swing lows in last N bars
    price_lows = find_swing_lows(prices, lookback)
    rsi_at_lows = [rsi_values[i] for i in price_lows]
    
    # Bullish divergence: price lower low, RSI higher low
    if len(price_lows) >= 2:
        if prices[price_lows[-1]] < prices[price_lows[-2]] and \
           rsi_at_lows[-1] > rsi_at_lows[-2]:
            return 'bullish_divergence'
    
    return None
```

#### 2.3 Relative Volume Instead of Absolute
```python
# Current: volume > 1.5x average
# Better: relative volume with decay
relative_volume = volume / volume.rolling(20).mean()

# Tier the signal:
if relative_volume > 3.0:
    confluence += 2  # Massive volume = strong signal
elif relative_volume > 1.5:
    confluence += 1  # Normal surge
# No bonus if volume is average or below
```

#### 2.4 Dynamic Stop-Loss (Volatility-Adaptive)
```python
# Current: fixed 5% stop
# Better: ATR-based stop
atr = calculate_atr(highs, lows, closes, 14)
volatility_ratio = atr / close  # Normalize by price

# In calm markets (low ATR): tighter stops (3%)
# In volatile markets (high ATR): wider stops (7%)
stop_pct = max(0.03, min(0.07, volatility_ratio * 2))
```

#### 2.5 Multi-Level Take Profit
```python
# Current: single TP at 10%
# Better: staged exits
tp_levels = [
    (0.33, 0.05),   # Take 1/3 at 5%
    (0.33, 0.08),   # Take 1/3 at 8%
    (0.34, 0.12),   # Take 1/3 at 12%
]
# Move stop to breakeven after first TP hit
# This locks in profit while letting winners run
```

### TIER 3 — Major Features (1-2 weeks, game-changing)

#### 3.1 Fear & Greed Index Integration
```python
import requests

def get_fear_greed():
    """Fetch daily crypto fear & greed index."""
    resp = requests.get('https://api.alternative.me/fng/?limit=1')
    data = resp.json()
    value = int(data['data'][0]['value'])  # 0-100
    return value

# Usage in position sizing:
fg = get_fear_greed()
if fg < 25:  # Extreme Fear = buy opportunity
    size_multiplier = 1.25  # Increase size 25%
elif fg > 75:  # Extreme Greed = reduce exposure
    size_multiplier = 0.50  # Cut size in half
else:
    size_multiplier = 1.0
```

#### 3.2 Correlation-Aware Dual Position Management
```python
def btc_eth_correlation(btc_returns, eth_returns, window=30):
    """Calculate rolling correlation."""
    return btc_returns.rolling(window).corr(eth_returns).iloc[-1]

# If correlation > 0.85: treat as single bet
# Reduce combined position size to avoid double-risk
corr = btc_eth_correlation(btc_rets, eth_rets)
if corr > 0.85:
    max_per_position = 0.02  # 2% risk each (4% total)
else:
    max_per_position = 0.04  # Full 4% each (less correlated)
```

#### 3.3 ML Signal Confidence Filter
```python
# Phase 1: Collect training data (log at each tick)
features = {
    'rsi': rsi_value,
    'adx': adx_value,
    'bb_position': (close - bb_lower) / (bb_upper - bb_lower),
    'relative_volume': rel_vol,
    'macd_hist': hist[-1],
    'ema_slope': (ema20[-1] - ema20[-5]) / ema20[-5],
    'hour_of_day': now.hour / 24,
    'day_of_week': now.weekday() / 7,
}

# Phase 2: Train LightGBM (after collecting 3+ months of data)
# Target: 1 if next 4 days return > 2%, else 0
# Use model.predict_proba() as confidence score
# Only enter trade if confidence > 0.6

# Phase 3: Retrain weekly on rolling 6-month window
```

#### 3.4 Regime Detection Engine
```python
def detect_regime(closes, volumes, period=50):
    """Classify current market regime."""
    adx = calculate_adx(period=14)
    bb_width = calculate_bb_width(period=20)
    vol_ratio = np.std(closes[-period:]) / np.mean(closes[-period:])
    
    if adx > 25 and vol_ratio < 0.05:
        return 'TRENDING'      # Strong trend, moderate vol
    elif adx < 20 and bb_width < bb_width_20_period_low:
        return 'SQUEEZE'       # Low vol, breakout coming
    elif vol_ratio > 0.08:
        return 'VOLATILE'      # High vol, reduce size
    else:
        return 'RANGING'       # Choppy, mean-revert only

# Adjust strategy based on regime:
regime_config = {
    'TRENDING':  {'sl': 0.07, 'tp': 0.12, 'confluence': 2, 'size_mult': 1.0},
    'SQUEEZE':   {'sl': 0.04, 'tp': 0.08, 'confluence': 3, 'size_mult': 0.8},
    'VOLATILE':  {'sl': 0.08, 'tp': 0.15, 'confluence': 3, 'size_mult': 0.5},
    'RANGING':   {'sl': 0.03, 'tp': 0.05, 'confluence': 2, 'size_mult': 0.6},
}
```

---

## 10. Implementation Priority Matrix

| # | Enhancement | Effort | Impact | Risk | Priority |
|---|-------------|--------|--------|------|----------|
| 1 | ADX regime filter (skip choppy) | 1 day | ★★★★★ | Low | **DO FIRST** |
| 2 | Bollinger Band filter | 1 day | ★★★★☆ | Low | **DO FIRST** |
| 3 | MACD confirmation signal | 1 day | ★★★★☆ | Low | **DO FIRST** |
| 4 | Multi-timeframe (add 1D check) | 2 days | ★★★★★ | Low | **DO SECOND** |
| 5 | Relative volume (tiered) | 1 day | ★★★☆☆ | Low | **DO SECOND** |
| 6 | Day/hour seasonality filter | 1 day | ★★★☆☆ | Low | **DO SECOND** |
| 7 | RSI divergence detection | 3 days | ★★★★☆ | Med | **WEEK 2** |
| 8 | Multi-level take profit | 2 days | ★★★★☆ | Med | **WEEK 2** |
| 9 | Dynamic ATR-based stops | 2 days | ★★★★☆ | Med | **WEEK 2** |
| 10 | Fear & Greed index | 1 day | ★★★☆☆ | Low | **WEEK 2** |
| 11 | Correlation-aware sizing | 2 days | ★★★☆☆ | Med | **WEEK 3** |
| 12 | Regime detection engine | 3 days | ★★★★★ | Med | **WEEK 3** |
| 13 | Consecutive loss management | 1 day | ★★★☆☆ | Low | **WEEK 3** |
| 14 | ML feature logging | 2 days | ★★★☆☆ | Low | **WEEK 3** |
| 15 | ML confidence filter | 1-2 wk | ★★★★★ | High | **MONTH 2** |
| 16 | RL-based exit management | 2-3 wk | ★★★★☆ | High | **MONTH 3** |

### Estimated Impact (Conservative)
- **Tier 1 Quick Wins** (items 1-3): +15-25% improvement in win rate
- **Tier 2 Medium** (items 4-10): +20-35% improvement in overall profitability
- **Tier 3 Advanced** (items 11-16): +30-50% improvement but requires more careful implementation

### Implementation Order Recommendation
1. **Week 1**: ADX filter + Bollinger Bands + MACD = immediately better entry quality
2. **Week 2**: Multi-timeframe + RSI divergence + Multi-level TP = catch bigger moves, reduce giveback
3. **Week 3**: Regime detection + Fear/Greed + Correlation sizing = smart market adaptation
4. **Month 2+**: ML pipeline setup, data collection, model training

---

## Sources & References

1. **Freqtrade** — github.com/freqtrade/freqtrade (47.1K★, MIT License)
2. **Jesse AI** — github.com/jesse-ai/jesse (7.5K★, MIT License)
3. **FinRL** — github.com/AI4Finance-Foundation/FinRL (14K★, MIT License)
4. **Hummingbot** — github.com/hummingbot/hummingbot (17.5K★, Apache 2.0)
5. **Superalgos** — github.com/Superalgos/Superalgos (5.3K★, Apache 2.0)
6. **Backtesting.py** — github.com/kernc/backtesting.py (8K★, AGPL-3.0)
7. **FreqAI Docs** — docs.freqtrade.io/en/stable/freqai/
8. **FreqAI Feature Engineering** — docs.freqtrade.io/en/stable/freqai-feature-engineering/
9. **FreqAI RL** — docs.freqtrade.io/en/stable/freqai-reinforcement-learning/
10. **r/algotrading Wiki** — reddit.com/r/algotrading/wiki/index
11. **TradingView RSI Guide** — tradingview.com/support/solutions (Wilder + Cardwell)
12. **Caulk et al. (2022)** — "FreqAI: generalizing adaptive modeling for chaotic time-series market forecasts" JOSS
13. **Liu et al. (2021)** — "FinRL: Deep RL framework to automate trading" ACM ICAIF
14. **Liu et al. (2020)** — "Deep RL for automated stock trading: An ensemble strategy" NeurIPS Deep RL Workshop
15. **Gorishniy et al. (2021)** — "Revisiting Deep Learning Models for Tabular Data" NeurIPS (LightGBM > neural nets for tabular)

---

*Document generated for AtoBot crypto strategy enhancement. Not financial advice.*
