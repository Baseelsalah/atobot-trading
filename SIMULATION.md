# 🎯 Running a Trading Bot Simulation

This guide shows you exactly how to run a simulation of the trading bot in DRY_RUN mode.

## What is a Simulation?

A simulation runs the bot with real market data but **does not execute actual trades**. It's perfect for:
- Testing your bot configuration
- Watching how the bot analyzes the market
- Verifying signal generation (BUY/SELL/HOLD)
- Checking PnL calculations
- Ensuring no crashes or errors

## Prerequisites

You need:
- **Node.js v18+** and **npm**
- **Alpaca API credentials** (paper trading or live account)
  - Even in DRY_RUN mode, the bot needs API access to fetch market data
  - Get free paper trading credentials at [alpaca.markets](https://alpaca.markets/)

## Quick Start (5 Steps)

### 1. Install Dependencies
```bash
npm install
```

### 2. Create Environment File
Create a `.env` file in the project root:
```bash
# Required for simulation
DRY_RUN=1

# Your Alpaca API credentials (paper trading recommended for testing)
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_API_SECRET=your_alpaca_secret_key_here

# Server configuration
PORT=5000
```

### 3. Start the Bot
```bash
# Development mode (recommended for testing)
npm run dev

# OR production mode
npm start
```

### 4. Watch the Simulation
You should see output like:
```
[SIM] DRY_RUN active - orders will be logged but not executed
[BOOT] bootTimeET=09:30 pid=12345 version=1.0.0
[PREFLIGHT] BOOT PREFLIGHT CHECK
[PREFLIGHT] Alpaca Auth: OK
[PREFLIGHT] Clock Status: OK
[CONTROL] Analysis Loop: RUNNING | DRY_RUN: ON
[ANALYSIS] Fetching market data for SPY...
[SIGNAL] SPY: BUY signal detected
[DRY_RUN] BUY 1 share of SPY at $450.25
[PnL] Simulated profit: +$12.50
```

### 5. Verify Success
**The simulation is working correctly if you see:**
- ✅ `[SIM] DRY_RUN active` at startup
- ✅ Regular analysis cycles (every few minutes)
- ✅ Market data being fetched
- ✅ Signal generation (BUY/SELL/HOLD)
- ✅ `[DRY_RUN]` prefix on trade logs
- ✅ PnL calculations
- ✅ No crashes or freezes

**Let it run for 5-10 minutes to fully test it.**

## Running on a VPS (24/7 Operation)

To run the simulation continuously on a remote server:

### 1. SSH into Your VPS
```bash
ssh youruser@your-vps-ip
```

### 2. Install Prerequisites
```bash
# Update package manager
sudo apt update

# Install Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installation
node --version
npm --version
```

### 3. Clone and Setup
```bash
# Clone repository
git clone https://github.com/Baseelsalah/atobot-trading.git
cd atobot-trading

# Install dependencies
npm install

# Create .env file
nano .env
# Add your configuration (see step 2 above)
```

### 4. Use tmux for Persistent Sessions
```bash
# Install tmux
sudo apt install tmux

# Create a new tmux session
tmux new -s tradingbot

# Start the bot inside tmux
npm run dev

# Detach from tmux (bot keeps running)
# Press: Ctrl + B, then D

# Later, reattach to see logs
tmux attach -t tradingbot
```

### 5. Verify Continuous Operation
```bash
# Disconnect from VPS completely
exit

# Reconnect later
ssh youruser@your-vps-ip

# Reattach to bot session
tmux attach -t tradingbot

# If logs continued during your absence, simulation is persistent!
```

## Troubleshooting

### "Preflight check failed - Trading BLOCKED"
**Cause:** Invalid or missing Alpaca API credentials

**Fix:**
- Verify your `.env` file has correct `ALPACA_API_KEY` and `ALPACA_API_SECRET`
- Test credentials at [app.alpaca.markets](https://app.alpaca.markets)
- Make sure you're using paper trading keys for testing

### "NETWORK_ERROR: fetch failed"
**Cause:** Cannot connect to Alpaca API

**Fix:**
- Check your internet connection
- Verify firewall isn't blocking outbound HTTPS
- Confirm Alpaca API is operational: [status.alpaca.markets](https://status.alpaca.markets)

### "Failed to acquire leader lock"
**Cause:** Cannot write to cloud storage (Google Cloud or AWS)

**Impact:** Bot will still run and analyze markets, but won't execute trades (even simulated ones)

**Fix:** This is expected in local development without cloud storage configured. The bot will still perform analysis and generate signals.

### Bot stops when I close my laptop
**Cause:** Running locally, not on a persistent server

**Fix:** 
- Use a VPS (DigitalOcean, AWS, Linode) for 24/7 operation
- OR use a process manager like PM2 (see README.md)
- OR keep your computer awake while testing

## What's Happening During a Simulation?

The bot performs these steps continuously:

1. **Preflight Checks** - Validates API credentials and market hours
2. **Market Data Fetch** - Gets real-time price data from Alpaca
3. **Indicator Calculation** - Computes technical indicators (VWAP, SMA, volume)
4. **Signal Generation** - Analyzes data and generates trade signals
5. **Risk Assessment** - Checks position limits and risk rules
6. **Trade Logging** - Logs what it *would* trade (with `[DRY_RUN]` prefix)
7. **PnL Simulation** - Calculates theoretical profit/loss
8. **Wait & Repeat** - Sleeps for configured interval, then repeats

**Key Difference:** In DRY_RUN mode, step 6 only *logs* trades instead of sending orders to Alpaca.

## Next Steps

Once your simulation runs successfully:

1. ✅ **Tune parameters** - Adjust risk limits, position sizes, indicators
2. ✅ **Test different symbols** - Modify watchlist in configuration
3. ✅ **Run for longer periods** - Let it run for days to test stability
4. ✅ **Review logs** - Check `reports/` directory for detailed analysis
5. ⚠️ **Go live carefully** - When ready, remove `DRY_RUN=1` from `.env`

---

**Remember:** Always test thoroughly in simulation mode before running with real money!
