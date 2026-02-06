# AtoBot Trading System

AI-powered automated stock trading bot with full hands-free operation.

## 🚀 Quick Start

### Full Automation (Zero Daily Involvement)

```bash
# 1. Set up cron jobs for automated operation
crontab -e

# Add these lines:
30 13 * * 1-5 cd /workspaces/atobot-trading && npm run daily:premarket
*/30 14-20 * * 1-5 cd /workspaces/atobot-trading && npm run daily:monitor
15 21 * * 1-5 cd /workspaces/atobot-trading && npm run daily:postmarket

# 2. That's it! Bot now runs fully automated.
```

**See AUTOMATION_COMPLETE.md for full setup guide**

### Manual Operation (Testing/Development)

```bash
# Start bot in testing mode (safe, no real orders)
npm run dev

# Or start with PM2 for auto-restart
npm run pm2:start

# Switch to live trading (requires confirmation)
npm run mode:live
npm run pm2:restart
```

---

## ✅ What This Bot Does

**Fully Automated Daily Routine:**
1. ✅ Wakes up at 8:30 AM ET
2. ✅ Runs pre-market safety checks
3. ✅ Auto-switches to live trading mode
4. ✅ Starts trading at 9:35 AM ET
5. ✅ Monitors itself every 30 minutes
6. ✅ Auto-restarts if crashes
7. ✅ Closes all positions by 3:45 PM ET
8. ✅ Auto-switches to testing mode overnight
9. ✅ Alerts you ONLY if something goes wrong

**Trading Features:**
- Two algorithmic strategies (VWAP Reversion, Opening Range Breakout)
- AI-powered market analysis with OpenAI GPT-4
- Regime detection (trending vs choppy markets)
- Dynamic position sizing based on ATR volatility
- Multi-layer risk management (stop loss, take profit, daily limits)
- Day trading rules (all positions closed by 3:45 PM ET)

**Reliability Features:**
- PM2 process manager (auto-restart on crash)
- API retry logic (3x with exponential backoff)
- Persistent state (survives mid-day restarts)
- Health monitoring endpoints
- Smoke testing before market open
- Alpaca clock synchronization (no delays)

---

## 📚 Documentation

| Guide | Purpose |
|-------|---------|
| **AUTOMATION_COMPLETE.md** | 🔥 Full automation setup - START HERE |
| **AUTOMATION_GUIDE.md** | Complete automation documentation |
| **MODE_SWITCHER_GUIDE.md** | Manual mode switching (testing ↔ live) |
| **TESTING_GUIDE.md** | Daily operations and monitoring |
| **QUICK_REFERENCE.md** | Command cheat sheet |
| **crontab-template.txt** | Cron job examples |

---

## 🤖 Automation Commands

```bash
# Pre-Market Routine (8:30 AM ET)
npm run daily:premarket
# - Runs smoke test
# - Auto-switches to live mode
# - Starts bot
# - Verifies health

# Trading Hours Monitor (Every 30 min)
npm run daily:monitor
# - Checks bot is running
# - Auto-restarts if crashed
# - Detects stalls
# - Checks for alerts

# Post-Market Routine (4:15 PM ET)
npm run daily:postmarket
# - Verifies positions closed
# - Reviews daily report
# - Auto-switches to test mode
```

---

## 🎮 Manual Control Commands

```bash
# Mode Switching
npm run mode:status      # Check current mode
npm run mode:test        # Switch to testing mode
npm run mode:live        # Switch to live mode (requires confirmation)
npm run mode:rollback    # Undo last mode change

# Bot Management
npm run pm2:start        # Start bot with PM2
npm run pm2:stop         # Stop bot
npm run pm2:restart      # Restart bot
npm run pm2:status       # Check status
npm run pm2:logs         # View logs

# Testing
npm run smoke-test       # Pre-market validation
npm run dev              # Run in development mode

# Production
npm run build            # Build production bundle
npm run start            # Run production build
```

---

## 🔧 Configuration

### Required Environment Variables

```bash
# Alpaca API (get from alpaca.markets)
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret

# OpenAI API (optional if using deterministic strategies only)
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4

# Trading Mode
DRY_RUN=0                # 0=live trading, 1=testing mode
TIME_GUARD_OVERRIDE=0    # 0=FORT KNOX active, 1=disabled (testing only)
```

### Optional: Alert Configuration

```bash
# Email Alerts (via SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@email.com
SMTP_PASS=your_app_password
ALERT_EMAIL=your@email.com

# SMS Alerts (via Twilio)
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+1...
TWILIO_TO_NUMBER=+1...
```

---

## 🛡️ Safety Features

**Time Guards (FORT KNOX):**
- Entry window: 9:35 AM - 11:35 AM ET only (2 hours)
- Force close: 3:45 PM ET (15 min before market close)
- Early close detection: Automatic adjustment

**Daily Limits:**
- Max 10 new entries per day
- Max -$500 daily loss (stops new entries)
- Max +$500 daily profit (stops new entries)
- Max 5 concurrent positions

**Risk Per Trade:**
- 0.5% account risk per trade
- 1% stop loss, 2.5% take profit (1:2.5 R:R)
- ATR-based position sizing

---

## 📊 Monitoring

### Web Dashboard
```
http://localhost:5000
```

Shows:
- Current positions
- Trade history
- Performance metrics
- Bot status

### Health Endpoints

```bash
# Overall health
curl http://localhost:5000/health

# Trading readiness
curl http://localhost:5000/readiness

# Current positions
curl http://localhost:5000/api/trading/positions
```

### Log Files

```bash
# PM2 logs
npm run pm2:logs --follow

# Automation logs
cat reports/automation.jsonl | tail -20

# Daily reports
cat daily_reports/$(date +%Y-%m-%d).json | jq

# Alerts
ls -ltr reports/alerts/
```

---

## 🚨 Alert System

**Alerts created in:** `reports/alerts/`

**Alert Types:**
- **INFO** - Normal operations (pre-market complete, etc.)
- **WARNING** - Non-critical issues (stall detected, etc.)
- **CRITICAL** - Immediate attention (smoke test failed, positions open overnight)

**How you're notified:**
1. File system alerts
2. Cron email (if `MAILTO` set in crontab)
3. SMTP email (if configured)
4. SMS (if Twilio configured)

---

## 🧪 Testing Before Going Live

### Week 1: Manual Testing

```bash
# Test each automation script
npm run daily:premarket
npm run daily:monitor
npm run daily:postmarket

# Verify mode switching
npm run mode:test
npm run mode:live
npm run mode:status
```

### Week 2: Automated Testing (Paper Trading)

```bash
# Install cron jobs
crontab -e
# Add the 3 automation lines

# Keep DRY_RUN=1 for safety
# Let automation run for a week
# Monitor logs daily
```

### Week 3+: Full Automation (Live Trading)

```bash
# Remove DRY_RUN from .env
# Let pre-market routine auto-switch to live
# Monitor first week manually
# Then... forget about it! 🎉
```

---

## ⚠️ Important Notes

**This bot trades with REAL MONEY when in live mode.**

Before enabling full automation:
- ✅ Test in paper trading for 2+ weeks
- ✅ Verify smoke tests pass consistently
- ✅ Understand the strategies being used
- ✅ Set appropriate daily loss limits
- ✅ Have alerts configured
- ✅ Monitor first week manually

**Safety limits are configured in:** `server/dayTraderConfig.ts`

---

## 📁 Project Structure

```
atobot-trading/
├── server/
│   ├── index.ts                 # Main entry point
│   ├── tradingBot.ts            # Core trading logic
│   ├── strategyEngine.ts        # Trading strategies
│   ├── tradingTimeGuard.ts      # FORT KNOX time guards
│   ├── riskManager.ts           # Risk management
│   ├── persistentState.ts       # State persistence
│   └── ...
├── scripts/
│   ├── daily-automation.ts      # 🔥 Automation agent
│   ├── mode-switcher.ts         # Mode switching tool
│   ├── smoke-test.ts            # Pre-market validation
│   └── build.ts                 # Production build
├── reports/
│   ├── alerts/                  # Alert files
│   ├── automation.jsonl         # Automation log
│   └── state/                   # Persistent state
├── daily_reports/               # Daily performance reports
├── ecosystem.config.js          # PM2 configuration
├── crontab-template.txt         # Cron job examples
└── Documentation files...
```

---

## 🆘 Troubleshooting

### Bot Not Trading

```bash
# Check mode
npm run mode:status
# Should show: LIVE TRADING

# Check time guard
npm run pm2:logs --lines 20 | grep "TIME GUARD"
# Should show entry window active

# Check health
curl http://localhost:5000/health | jq
```

### Automation Not Running

```bash
# Check cron jobs installed
crontab -l

# Check cron logs
tail -f /var/log/syslog | grep CRON

# Test automation manually
npm run daily:premarket
```

### Positions Not Closing

```bash
# Check force close logs
npm run pm2:logs --lines 100 | grep "FORCE"

# Manually close all positions
# Via Alpaca dashboard or API
```

---

## 🎯 Support

**Documentation:**
- Read AUTOMATION_COMPLETE.md first
- Check TESTING_GUIDE.md for daily operations
- Use QUICK_REFERENCE.md as command cheat sheet

**Logs:**
- PM2 logs: `npm run pm2:logs`
- Automation log: `cat reports/automation.jsonl`
- Alerts: `ls -ltr reports/alerts/`

**Emergency Stop:**
```bash
npm run pm2:stop
# Then close positions manually via Alpaca dashboard
```

---

## 📜 License

MIT

---

**Your self-sufficient AI trading agent is ready!** 🤖📈

See AUTOMATION_COMPLETE.md to get started with full hands-free operation.

	- `tmux attach -t tradingbot`

Result: You can close your laptop, disconnect, or shut VS Code—the bot keeps running.

## Verify it keeps running after you close your laptop

The bot only keeps running if it’s on a machine that stays awake. Use this quick test on a remote server:

1. Start your bot inside tmux:
	- `tmux new -s tradingbot`
	- `DRY_RUN=1 npm run dev`
	- or `DRY_RUN=1 npm start`
2. Detach from tmux:
	- Press `Ctrl + B`, then `D`
3. Close your laptop (sleep/disconnect is fine).
4. Reopen your laptop and SSH back into the server.
5. Reattach:
	- `tmux attach -t tradingbot`
6. If logs are still running, the bot is persistent.

Note: If you run the bot on your laptop, closing the lid stops it. tmux only helps if the machine itself stays awake (VPS, cloud server, Raspberry Pi, mini-PC).

## Deploy on a VPS (24/7 uptime)

A VPS is a remote computer that stays on 24/7, perfect for running the bot even when your laptop is closed.

1. Create a VPS (DigitalOcean, Linode, AWS, etc.).
2. SSH into the server.
3. Install prerequisites:
	- Node.js (version required by this repo)
	- Git
4. Clone the repo and install dependencies.
5. Run the bot inside tmux (see the tmux section above).

Optional: For production, use a process manager (systemd or pm2) to auto-restart the bot if it crashes.

### Auto-restart with systemd (recommended on Linux)

1. Create a service file at `/etc/systemd/system/atobot.service`:
	```
	[Unit]
	Description=AtoBot Trading Bot
	After=network.target

	[Service]
	Type=simple
	User=YOUR_USER
	WorkingDirectory=/path/to/atobot-trading
	Environment=DRY_RUN=1
	ExecStart=/usr/bin/npm run dev
	Restart=always
	RestartSec=5

	[Install]
	WantedBy=multi-user.target
	```
2. Reload and start:
	- `sudo systemctl daemon-reload`
	- `sudo systemctl enable atobot`
	- `sudo systemctl start atobot`
3. Check status:
	- `sudo systemctl status atobot`

### Auto-restart with pm2 (Node-based process manager)

1. Install pm2:
	- `npm install -g pm2`
2. Start the bot:
	- `pm2 start "npm run dev" --name tradingbot --env DRY_RUN=1`
3. Save and enable startup:
	- `pm2 save`
	- `pm2 startup`

## Health check (verify uptime and persistence)

Use this checklist to confirm the bot is running and persistent.

1. Check tmux sessions:
	- `tmux ls`
2. Attach to the bot session:
	- `tmux attach -t tradingbot`
	- If logs are printing, the bot is alive.
3. Detach, disconnect, and reattach later:
	- Press `Ctrl + B`, then `D`
	- Reconnect and run `tmux attach -t tradingbot`
	- If logs continued during your absence, persistence is confirmed.

### If using pm2

1. Check status:
	- `pm2 status`
2. Reboot test:
	- `sudo reboot`
	- After reconnecting, run `pm2 status`
	- If the bot is **online**, auto-restart works.

### If using systemd

1. Check status:
	- `systemctl status tradingbot`
2. Reboot test:
	- `sudo reboot`
	- After reconnecting, run `systemctl status tradingbot`
	- If it’s **active (running)**, auto-restart works.

## Quick status check (fast verification)

Use these commands to confirm the bot is running without digging through logs.

1. Check if the tmux session is alive:
	- `tmux ls`
	- If you see `tradingbot`, the session is running.
2. Reattach to see live logs:
	- `tmux attach -t tradingbot`
	- If logs are printing, the bot is active.
3. Check pm2 status (if using pm2):
	- `pm2 status`
	- The bot should show as **online**.
4. Check systemd status (if using systemd):
	- `systemctl status tradingbot`
	- Look for **active (running)**.
5. Confirm the process is alive:
	- `ps aux | grep tsx`
	- If you see a tsx/node process for the bot, it’s running.

## Daily operator checklist (before market open)

Use this quick routine each morning to confirm the bot is healthy and ready for the trading day.

1. Log into your VPS:
	- `ssh youruser@your-vps-ip`
2. Check the tmux session:
	- `tmux ls`
	- You should see `tradingbot: 1 windows`.
3. Confirm the bot is running:
	- `tmux attach -t tradingbot`
	- Look for new log entries, analysis cycles, DRY_RUN trades, and no frozen output.
	- Detach: Press `Ctrl + B`, then `D`.
4. If using pm2, check status:
	- `pm2 status`
	- The bot should show **online**.
5. If using systemd, check service health:
	- `systemctl status tradingbot`
	- Look for **active (running)**.
6. Confirm the process exists:
	- `ps aux | grep tsx`
	- If you see a tsx/node process, it’s alive.
7. Optional: Check logs:
	- pm2: `pm2 logs tradingbot`
	- systemd: `journalctl -u tradingbot -f`

## Smoke-screen test (fast readiness check)

Run these steps in order. If all pass, the bot is ready for the next trading day.

1. SSH into your VPS:
	- `ssh youruser@your-vps-ip`
2. Check if the tmux session exists:
	- `tmux ls`
	- You should see `tradingbot: 1 windows`.
3. Attach to tmux and check for live logs:
	- `tmux attach -t tradingbot`
	- Look for new timestamps, analysis cycles, heartbeat logs, DRY_RUN checks, and no frozen output.
	- Detach: Press `Ctrl + B`, then `D`.
4. Check the auto-restart layer:
	- pm2: `pm2 status` (bot should be **online**)
	- systemd: `systemctl status tradingbot` (should be **active (running)**)
5. Confirm the Python process exists:
	- `ps aux | grep tsx`
	- If you see a tsx/node process, the bot is running at the OS level.
6. Optional: Check logs for errors:
	- pm2: `pm2 logs tradingbot`
	- systemd: `journalctl -u tradingbot -f`
