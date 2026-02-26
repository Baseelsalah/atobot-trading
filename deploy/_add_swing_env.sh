#!/bin/bash
# Add swing strategy config to VPS .env

ENV_FILE="/opt/atobot/.env"

# Update STRATEGIES to include swing
sed -i 's/STRATEGIES=\["vwap_scalp","momentum"\]/STRATEGIES=["vwap_scalp","momentum","swing"]/' "$ENV_FILE"

# Append swing config if not already present
if ! grep -q "SWING_RSI_OVERSOLD" "$ENV_FILE"; then
cat >> "$ENV_FILE" << 'EOF'

# ── Swing Strategy (v2 tuned: 67.9% WR, 1.63 PF, $106/mo) ──────────────────
SWING_RSI_OVERSOLD=38.0
SWING_RSI_OVERBOUGHT=70.0
SWING_VOLUME_SURGE=1.3
SWING_MIN_CONFLUENCE=2
SWING_TAKE_PROFIT_PCT=3.0
SWING_STOP_LOSS_PCT=1.5
SWING_TRAILING_ACTIVATION_PCT=1.5
SWING_TRAILING_OFFSET_PCT=0.75
SWING_MAX_HOLD_DAYS=5
SWING_MAX_POSITIONS=3
SWING_RISK_PER_TRADE_PCT=3.0
SWING_ORDER_SIZE_USD=250.0
SWING_EQUITY_CAP=500.0
SWING_MAX_GAP_PCT=5.0
SWING_SYMBOLS=AAPL,MSFT,NVDA,TSLA,AMD,META,GOOGL,AMZN
EOF
echo "Swing config added to .env"
else
echo "Swing config already exists in .env"
fi

# Show result
echo "---"
grep -E "STRATEGIES|SWING" "$ENV_FILE"
