# 🤖 Complete Automation Guide - Hands-Free Trading

## Overview

This guide shows you how to run AtoBot with **ZERO daily involvement**. The bot will:
- ✅ Automatically switch to live mode at 8:30 AM ET
- ✅ Run pre-market validation tests
- ✅ Start trading at 9:35 AM ET
- ✅ Monitor itself during trading hours
- ✅ Close all positions by 3:45 PM ET
- ✅ Switch back to testing mode overnight
- ✅ Alert you ONLY if something goes wrong

---

## Quick Start: Full Automation in 3 Steps

### Step 1: Test the Automation Scripts

```bash
# Test pre-market routine (safe - won't actually trade)
npm run daily:premarket

# Test monitoring
npm run daily:monitor

# Test post-market routine
npm run daily:postmarket
```

### Step 2: Set Up Cron Jobs

```bash
# Edit crontab
crontab -e

# Add these lines (adjust /path/to/atobot):
30 13 * * 1-5 cd /path/to/atobot && npm run daily:premarket >> /tmp/atobot-premarket.log 2>&1
*/30 14-20 * * 1-5 cd /path/to/atobot && npm run daily:monitor >> /tmp/atobot-monitor.log 2>&1
15 21 * * 1-5 cd /path/to/atobot && npm run daily:postmarket >> /tmp/atobot-postmarket.log 2>&1
```

### Step 3: Enable and Forget

```bash
# Verify cron jobs
crontab -l

# That's it! Bot now runs fully automated.
```

---

## What Gets Automated

### 1. Pre-Market Routine (8:30 AM ET)

**What it does:**
1. ✅ Runs smoke test (Alpaca connection, clock sync, positions check)
2. ✅ **Automatically switches to LIVE MODE** (bypasses confirmation)
3. ✅ Starts or restarts bot with live configuration
4. ✅ Waits for bot initialization
5. ✅ Verifies bot health and live mode active
6. ✅ Sends INFO alert that bot is ready

**How it works:**
- Direct `.env` modification (sets `DRY_RUN=0`, removes simulation vars)
- Restarts bot to apply new config
- Verifies via health endpoint and logs

**If something fails:**
- Creates CRITICAL alert in `reports/alerts/`
- Logs failure to `reports/automation.jsonl`
- Does NOT start trading
- You get notified (if email/SMS configured)

**Command:**
```bash
npm run daily:premarket
```

---

### 2. Trading Hours Monitoring (Every 30 Min)

**What it does:**
1. ✅ Checks if bot process is running (auto-restart if crashed)
2. ✅ Runs health check endpoint
3. ✅ Detects stalls (no tick in 10+ minutes)
4. ✅ Checks for critical alerts
5. ✅ Auto-restarts bot if health check fails

**When it runs:**
- Every 30 minutes from 9:30 AM to 4:00 PM ET
- Only on weekdays (Monday-Friday)

**What it catches:**
- Bot crashes during trading hours
- Health endpoint failures
- Trading stalls (no activity)
- Critical alerts pending

**Command:**
```bash
npm run daily:monitor
```

---

### 3. Post-Market Routine (4:15 PM ET)

**What it does:**
1. ✅ Verifies all positions closed (day trading compliance)
2. ✅ Reviews daily performance report
3. ✅ **Automatically switches to TESTING MODE** (safe overnight)
4. ✅ Optionally restarts bot in test mode

**Day trading compliance check:**
- If positions still open → Creates CRITICAL alert
- Logs violation for review
- You must manually close positions

**Safety:**
- Switches back to test mode automatically
- Prevents accidental overnight trading
- Bot ready for next day's pre-market routine

**Command:**
```bash
npm run daily:postmarket
```

---

## Cron Schedule Explanation

### Times in Different Formats

**Pre-Market (8:30 AM ET):**
```cron
30 13 * * 1-5    # UTC Standard Time (Nov-Mar)
30 12 * * 1-5    # UTC Daylight Time (Mar-Nov)
```

**Monitoring (Every 30 min, 9:30 AM - 4:00 PM ET):**
```cron
*/30 14-20 * * 1-5    # Standard Time
*/30 13-19 * * 1-5    # Daylight Time
```

**Post-Market (4:15 PM ET):**
```cron
15 21 * * 1-5    # Standard Time
15 20 * * 1-5    # Daylight Time
```

**What the numbers mean:**
```
30 13 * * 1-5
│  │  │ │ └─── Day of week (1-5 = Mon-Fri)
│  │  │ └───── Month (1-12, * = all)
│  │  └─────── Day of month (1-31, * = all)
│  └────────── Hour (0-23, UTC)
└──────────── Minute (0-59)
```

---

## Alert System

### Alert Severities

**INFO** - Normal operations
- Pre-market routine complete
- Post-market routine complete
- Daily report generated

**WARNING** - Non-critical issues
- Smoke test passed with warnings
- Bot may be stalled
- Daily report not found

**CRITICAL** - Requires immediate attention
- Smoke test FAILED
- Health check FAILED
- Bot not running during trading hours
- Positions still open after market close
- Automation routine crashed

### Where Alerts Go

1. **File System**
   ```bash
   reports/alerts/CRITICAL_2026-02-06T14-30-00-000Z.txt
   reports/alerts/WARNING_2026-02-06T14-30-00-000Z.txt
   reports/alerts/INFO_2026-02-06T14-30-00-000Z.txt
   ```

2. **Automation Log**
   ```bash
   reports/automation.jsonl
   ```

3. **Email** (if configured)
   - Add `MAILTO=your-email@example.com` to crontab
   - Receives all cron job output

4. **SMS** (if configured)
   - Add Twilio credentials to `.env`
   - Critical alerts sent via SMS

---

## Configuration Options

### Environment Variables

**Required:**
```bash
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
OPENAI_API_KEY=your_key  # Optional if using deterministic strategies only
```

**Automation Control:**
```bash
# Auto-restart bot after post-market routine (default: no)
AUTO_RESTART_POSTMARKET=1

# Auto-confirm live mode (dangerous - bypasses safety)
# Not recommended - automation already handles this
AUTO_CONFIRM_LIVE=1
```

**Alert Configuration:**
```bash
# Email alerts via SMTP
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
ALERT_EMAIL=your-email@example.com

# SMS alerts via Twilio (optional)
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+1...
TWILIO_TO_NUMBER=+1...
```

---

## Safety Mechanisms

### 1. Built-In Trading Limits

**Time Guards (FORT KNOX):**
- Entry window: 9:35 AM - 11:35 AM ET only (2 hours)
- Force close: 3:45 PM ET (15 min before market close)
- Early close detection: Adjusts automatically

**Daily Limits:**
- Max 10 new entries per day
- Max $500 daily loss (stops new entries)
- Max $500 daily profit (stops new entries Paper trading safety)
- Max 5 concurrent positions

**Risk Per Trade:**
- 0.5% of account risk per trade
- ATR-based position sizing
- 1% stop loss, 2.5% take profit (1:2.5 R:R)

### 2. Persistent State

**Survives restarts:**
- Trade counters (newEntriesToday)
- Daily P/L
- Reset completion flag

**Location:**
```bash
reports/state/daily_state.json
```

### 3. API Retry Logic

**Prevents single failures from stopping trading:**
- 3 retries with exponential backoff
- Base delay: 1000ms
- Max delay: 10000ms

### 4. Health Monitoring

**Endpoints checked by automation:**
- `/health` - Overall bot status
- `/readiness` - Ready to trade
- `/api/trading/positions` - Position verification

---

## Monitoring While Automated

### Check Current Status

```bash
# View automation log (last 10 events)
cat reports/automation.jsonl | tail -10

# Check if bot is running
npm run pm2:status

# View live logs
npm run pm2:logs --follow

# Check health
curl http://localhost:5000/health | jq

# Check for alerts
ls -ltr reports/alerts/
```

### Dashboard Access

```bash
# Open web dashboard
http://your-server:5000

# Shows:
# - Current positions
# - Trade history
# - Performance metrics
# - Bot status
```

### Weekly Review

```bash
# Review last week's automation logs
cat reports/automation.jsonl | grep "routine complete"

# Check alert summary
ls reports/alerts/ | wc -l

# View daily reports
ls -ltr daily_reports/
```

---

## Testing Before Going Live

### Phase 1: Manual Testing (Week 1)

```bash
# Run automation scripts manually
npm run daily:premarket
npm run daily:monitor
npm run daily:postmarket

# Verify behavior
# Check logs, alerts, mode switching
```

### Phase 2: Cron with DRY_RUN (Week 2)

1. Set up cron jobs
2. Keep `DRY_RUN=1` in `.env`
3. Let automation run for a week
4. Verify:
   - Cron jobs execute on schedule
   - Smoke tests pass
   - Bot starts/stops correctly
   - Alerts are sent

### Phase 3: Full Automation (Week 3+)

1. Remove `DRY_RUN=1` from `.env`
2. Let pre-market routine auto-switch to live
3. Monitor first week manually
4. Then let it run hands-free

---

## What You Should Still Monitor

### Daily (Optional - Only If You Want To)

- **Morning (9:40 AM):** Check first trade executed successfully
- **Mid-Day (12:00 PM):** Glance at dashboard to see P/L
- **Evening (4:30 PM):** Review daily report

### Weekly (Recommended)

- **Performance Review:** Check win rate, profit factor
- **Strategy Analysis:** See which strategies are working
- **Alert Summary:** Review any warnings or criticals
- **Log Cleanup:** Archive old logs

### Monthly (Required)

- **Account Review:** Verify Alpaca account balance
- **Strategy Tuning:** Adjust based on performance data
- **System Updates:** Update dependencies, security patches
- **Backup Reports:** Archive reports/ and daily_reports/

---

## Troubleshooting Automation

### Cron Jobs Not Running

**Check if cron service is active:**
```bash
systemctl status cron
# or: service cron status
```

**View cron logs:**
```bash
tail -f /var/log/syslog | grep CRON
# or: tail -f /var/log/cron
```

**Test command manually:**
```bash
cd /path/to/atobot && npm run daily:premarket
```

### Pre-Market Routine Fails

**Check automation log:**
```bash
cat reports/automation.jsonl | grep premarket | tail -5
```

**Common causes:**
- Alpaca API down → Retry manually
- Smoke test failed → Fix issues before 9:35 AM
- .env file missing → Restore from backup
- npm/node not in PATH → Add to crontab

### Bot Not Trading Despite Automation

**Verify mode:**
```bash
npm run mode:status
# Should show: LIVE TRADING
```

**Check logs:**
```bash
npm run pm2:logs --lines 50 | grep "DRY_RUN"
# Should show: DRY_RUN (env): OFF
```

**Verify time guard:**
```bash
npm run pm2:logs --lines 50 | grep "TIME GUARD"
# Should show entry window active
```

### Positions Not Closing at 3:45 PM

**Check force close logs:**
```bash
npm run pm2:logs --lines 100 | grep "FORCE"
```

**Manually close positions:**
```bash
# Via Alpaca dashboard or API
curl -X DELETE https://paper-api.alpaca.markets/v2/positions \
  -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
  -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET"
```

**Investigate why force close failed:**
- Check logs for errors
- Verify time guard is active
- Check if bot was running at 3:45 PM

---

## Advanced Automation Ideas

### 1. Automated Strategy Tuning

**Monthly cron job to analyze performance:**
```cron
0 1 1 * * cd /path/to/atobot && npm run analyze:strategies
```

**What it could do:**
- Calculate win rate per strategy
- Disable strategies with <40% win rate
- Adjust position sizing based on performance
- Generate monthly report

### 2. Automated Backup

**Daily backup of critical data:**
```cron
0 23 * * * cd /path/to/atobot && npm run backup:daily
```

**What it backs up:**
- Reports directory
- Daily reports
- .env configuration
- Automation logs

### 3. Automated Alerts Integration

**Slack/Discord webhook alerts:**
```bash
# Add to .env
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

**Send alerts to team chat instead of email**

### 4. Automated Performance Email

**Weekly performance digest:**
```cron
0 18 * * 5 cd /path/to/atobot && npm run report:weekly
```

**Email includes:**
- Total P/L for the week
- Win rate and profit factor
- Best/worst trades
- Strategy performance breakdown

---

## Full Automation Checklist

Before enabling full hands-free automation:

### Pre-Flight Checklist

- [ ] Tested in paper trading for 2+ weeks
- [ ] Win rate >45% established
- [ ] Profit factor >1.2 confirmed
- [ ] Smoke tests passing consistently
- [ ] Daily loss limits configured
- [ ] Time guards verified working
- [ ] Force close tested successfully
- [ ] PM2 auto-restart working
- [ ] Persistent state tested (survived restart)
- [ ] API retry logic tested
- [ ] Health endpoints responding

### Safety Checklist

- [ ] Email alerts configured
- [ ] SMS alerts configured (optional)
- [ ] Daily max loss set appropriately
- [ ] Daily max profit set (paper safety)
- [ ] Max positions limit set
- [ ] Account has sufficient buying power
- [ ] Using correct Alpaca account (paper vs live)
- [ ] Backup .env file created
- [ ] Know how to manually stop trading

### Automation Checklist

- [ ] Cron jobs tested manually
- [ ] Crontab installed and verified
- [ ] Automation logs being written
- [ ] Pre-market routine tested
- [ ] Monitoring routine tested
- [ ] Post-market routine tested
- [ ] Alert system tested (create fake CRITICAL alert)
- [ ] Dashboard accessible remotely

### Monitoring Checklist

- [ ] Can access server remotely (SSH)
- [ ] Can access dashboard remotely (HTTP)
- [ ] Know where automation logs are
- [ ] Know where PM2 logs are
- [ ] Know where alerts are created
- [ ] Have mobile access to server
- [ ] Alerts come to phone/email
- [ ] Can manually stop bot remotely

---

## Emergency Stop Procedures

### Stop All Trading Immediately

```bash
# SSH into server
ssh your-server

# Stop bot
cd /path/to/atobot
npm run pm2:stop

# Close all positions manually
# Via Alpaca dashboard or API

# Disable cron jobs
crontab -e
# Comment out all atobot lines with #
```

### Rollback to Testing Mode

```bash
npm run mode:rollback
# or manually:
npm run mode:test
npm run pm2:restart
```

### Disable Automation Temporarily

```bash
# Disable cron jobs
crontab -e
# Add # to beginning of each atobot line

# Or remove crontab entirely
crontab -r
```

---

## Summary: What Can Be Fully Automated

| Task | Can Automate? | How |
|------|---------------|-----|
| Pre-market smoke test | ✅ Yes | daily:premarket |
| Switch to live mode | ✅ Yes | daily:premarket (auto) |
| Start bot | ✅ Yes | daily:premarket |
| Monitor health | ✅ Yes | daily:monitor |
| Restart if crashed | ✅ Yes | daily:monitor |
| Detect stalls | ✅ Yes | daily:monitor |
| Trade execution | ✅ Yes | Bot handles automatically |
| Position management | ✅ Yes | Bot handles automatically |
| Force close positions | ✅ Yes | Bot handles at 3:45 PM |
| Verify positions closed | ✅ Yes | daily:postmarket |
| Switch to test mode | ✅ Yes | daily:postmarket |
| Send alerts | ✅ Yes | Built into automation scripts |
| Generate daily reports | ✅ Yes | Bot handles automatically |

**Bottom Line:** Everything can be automated. You only need to check in if alerts are sent.

---

## Next Steps

1. **Test automation scripts manually**
   ```bash
   npm run daily:premarket
   npm run daily:monitor
   npm run daily:postmarket
   ```

2. **Set up cron jobs using template**
   ```bash
   cp crontab-template.txt mycron.txt
   # Edit paths in mycron.txt
   crontab mycron.txt
   ```

3. **Monitor first automated trading day**
   - Check logs at 9:00 AM (pre-market routine)
   - Check logs at 9:45 AM (first trades)
   - Check logs at 12:00 PM (mid-day)
   - Check logs at 4:30 PM (post-market routine)

4. **After 1 week, review performance**
   ```bash
   cat reports/automation.jsonl | jq
   ls -ltr daily_reports/
   ```

5. **Go fully hands-free!**

---

**Your bot is now a self-sufficient trading machine.** 🤖📈
