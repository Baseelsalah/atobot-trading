# 🚀 DEPLOY ATOBOT TO YOUR SERVER - EASY GUIDE

## Option 1: One-Command Deployment (Easiest!)

### Step 1: Upload Code to Your Server

**Method A: Using Git (Recommended)**
```bash
# On your server, clone the repository
git clone <your-repo-url> atobot-trading
cd atobot-trading
```

**Method B: Using SCP (if no git)**
```bash
# On your local machine, upload the entire directory
scp -r /path/to/atobot-trading user@your-server-ip:/home/user/
```

**Method C: Using SFTP**
```bash
# Use FileZilla, WinSCP, or Cyberduck to upload
# Upload entire atobot-trading folder to your server
```

---

### Step 2: Run One-Command Deployment

```bash
# SSH into your server
ssh user@your-server-ip

# Navigate to bot directory
cd atobot-trading

# Run deployment script (does EVERYTHING for you!)
bash scripts/deploy.sh
```

**That's it!** The script will:
- ✅ Install Node.js (if needed)
- ✅ Install PM2
- ✅ Install dependencies
- ✅ Build production bundle
- ✅ Set up your .env file
- ✅ Install cron jobs
- ✅ Start the bot
- ✅ Verify everything works

---

## Option 2: Manual Step-by-Step (If You Prefer)

### Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Node.js 20.x
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node -v  # Should show v20.x
npm -v   # Should show 10.x
```

---

### Step 1: Upload Your Code

Choose one method from above (Git, SCP, or SFTP)

---

### Step 2: Install PM2

```bash
sudo npm install -g pm2
pm2 -v  # Verify installation
```

---

### Step 3: Install Dependencies

```bash
cd atobot-trading
npm install
```

---

### Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your API keys
nano .env
```

**Required Variables:**
```bash
# Alpaca API
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_API_SECRET=your_alpaca_secret_here

# OpenAI API (optional)
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4

# Trading Mode (start with paper trading!)
DRY_RUN=1
TIME_GUARD_OVERRIDE=1
SIM_CLOCK_OPEN=1
SIM_TIME_ET=2026-02-06 10:00
```

Press `Ctrl+X`, then `Y`, then `Enter` to save.

---

### Step 5: Build Production Bundle

```bash
npm run build
```

---

### Step 6: Install Cron Jobs

```bash
bash scripts/install-cron.sh
# Type "yes" when prompted
```

---

### Step 7: Start the Bot

```bash
npm run pm2:start

# Save PM2 process list
pm2 save

# Set PM2 to start on boot
pm2 startup
# Copy and run the command it shows
```

---

### Step 8: Verify Everything Works

```bash
# Check PM2 status
npm run pm2:status

# Check cron jobs
crontab -l

# Check bot health
curl http://localhost:5000/health | jq

# View live logs
npm run pm2:logs --lines 50
```

---

## Troubleshooting

### Issue: "PM2 not found"

```bash
# Install PM2 globally
sudo npm install -g pm2

# Verify
which pm2
pm2 -v
```

### Issue: "Port 5000 already in use"

```bash
# Find what's using port 5000
sudo lsof -i :5000

# Kill the process or change port in .env
PORT=5001
```

### Issue: "npm install fails"

```bash
# Clear npm cache
npm cache clean --force

# Try again
npm install
```

### Issue: "Cron jobs not running"

```bash
# Check cron service is running
sudo systemctl status cron

# Start cron if stopped
sudo systemctl start cron

# View cron logs
tail -f /tmp/atobot-premarket.log
tail -f /tmp/atobot-monitor.log
```

### Issue: "Bot not trading"

```bash
# Check mode
npm run mode:status
# Should show: TESTING or LIVE TRADING

# Check time guard
npm run pm2:logs --lines 50 | grep "TIME GUARD"

# Check health
curl http://localhost:5000/health | jq
```

---

## Accessing Your Server

### SSH Access

```bash
# Basic SSH
ssh username@your-server-ip

# With key file
ssh -i /path/to/key.pem username@your-server-ip

# Jump host (if behind firewall)
ssh -J jump-host username@your-server-ip
```

### Common Server Providers

**AWS EC2:**
```bash
ssh -i "your-key.pem" ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com
```

**DigitalOcean:**
```bash
ssh root@your-droplet-ip
```

**Linode:**
```bash
ssh root@your-linode-ip
```

**Vultr:**
```bash
ssh root@your-vultr-ip
```

**Local VPS:**
```bash
ssh your-username@your-local-ip
```

---

## Post-Deployment Checklist

- [ ] Bot running: `npm run pm2:status` shows "online"
- [ ] Cron jobs installed: `crontab -l` shows 6 jobs
- [ ] Health endpoint working: `curl http://localhost:5000/health`
- [ ] Logs accessible: `npm run pm2:logs`
- [ ] Mode correct: `npm run mode:status`
- [ ] Alerts directory exists: `ls reports/alerts/`
- [ ] Automation log exists: `cat reports/automation.jsonl`

---

## What Happens After Deployment

### Automatic Schedule

| Time | What Happens |
|------|--------------|
| **8:30 AM ET** | Pre-market routine runs → Bot validated → Switches to live mode |
| **9:35 AM ET** | Trading window opens → Bot starts analyzing |
| **Every 30 min** | Health check → Auto-restart if crashed |
| **3:45 PM ET** | Force close all positions (day trading rule) |
| **4:15 PM ET** | Post-market routine → Verify close → Switch to test mode |
| **Sunday 6 PM** | Weekly performance report generated |
| **11:55 PM Daily** | All data backed up |
| **1st of Month** | Strategy analysis → Auto-disable losers |

---

## Monitoring Your Bot

### Daily (Optional)

```bash
# Quick health check
curl http://localhost:5000/health | jq

# View recent logs
npm run pm2:logs --lines 100

# Check automation events
cat reports/automation.jsonl | tail -20
```

### Weekly (Recommended)

```bash
# View weekly report
cat weekly_reports/weekly_*.txt | tail -1

# Check for alerts
ls -ltr reports/alerts/ | tail -10

# Review performance
cat daily_reports/$(date +%Y-%m-%d).json | jq
```

### Monthly (Required)

```bash
# View monthly analysis
cat monthly_reports/monthly_*.txt | tail -1

# Check disabled strategies
cat server/strategyConfig.json | jq
```

---

## Useful Commands

```bash
# Bot Management
npm run pm2:start          # Start bot
npm run pm2:stop           # Stop bot
npm run pm2:restart        # Restart bot
npm run pm2:delete         # Remove from PM2
npm run pm2:logs           # View logs
npm run pm2:status         # Check status

# Mode Management
npm run mode:status        # Check current mode
npm run mode:test          # Switch to testing
npm run mode:live          # Switch to live (requires confirmation)

# Testing
npm run smoke-test         # Pre-market validation
npm run daily:monitor      # Test monitoring

# Automation Logs
cat reports/automation.jsonl | tail -20
tail -f /tmp/atobot-premarket.log
tail -f /tmp/atobot-monitor.log
```

---

## Need Help?

1. **Read the docs:**
   - `AUTOMATION_COMPLETE.md` - Quick start
   - `INSTALLATION_COMPLETE.md` - Full guide
   - `QUICK_REFERENCE.md` - Commands

2. **Check logs:**
   - PM2: `npm run pm2:logs`
   - Automation: `cat reports/automation.jsonl`
   - Cron: `tail -f /tmp/atobot-*.log`

3. **Verify setup:**
   - Bot status: `npm run pm2:status`
   - Health: `curl http://localhost:5000/health | jq`
   - Mode: `npm run mode:status`

---

## 🎉 You're Done!

Your bot is now running **100% hands-free** on your server!

**It will:**
- ✅ Wake itself up every morning
- ✅ Validate before trading
- ✅ Trade automatically (9:35 AM - 3:45 PM ET)
- ✅ Monitor and heal itself
- ✅ Close all positions daily
- ✅ Analyze and tune strategies
- ✅ Back up all data
- ✅ Alert you only when needed

**You just check in when you want to!** 🤖📈
