# 🚀 AtoBot Quick Reference Card

## Daily Commands (Run These Every Day)

### Before Market Open (8:30-9:20 AM ET)
```bash
# 1. Run smoke test
npm run smoke-test

# 2. Check bot status
npm run pm2:status

# 3. Start bot if needed
npm run pm2:start
```

### During Trading (9:35 AM - 3:45 PM ET)
```bash
# Check health
curl http://localhost:5000/health | jq

# Watch live logs
npm run pm2:logs --follow

# Check positions
curl http://localhost:5000/api/trading/positions | jq
```

### After Market Close (4:00+ PM ET)
```bash
# Verify all positions closed
curl http://localhost:5000/api/trading/positions

# View daily report
cat daily_reports/$(date +%Y-%m-%d).json | jq

# Optional: Stop bot for the night
npm run pm2:stop
```

---

## 🤖 AI Mode Switcher (NEW!)

```bash
# Check current mode (Testing vs Live)
npm run mode:status

# Switch to testing mode (safe, no real orders)
npm run mode:test

# Switch to live trading mode (requires confirmation)
npm run mode:live

# Rollback last mode change
npm run mode:rollback
```

**See MODE_SWITCHER_GUIDE.md for full documentation**

---

## Emergency Commands

```bash
# Bot crashed - restart it
npm run pm2:restart

# Bot won't start - fresh start
npm run pm2:delete
npm run build
npm run pm2:start

# View error logs only
npm run pm2:logs --err

# View last 100 log lines
npm run pm2:logs --lines 100
```

---

## Monitoring URLs

- **Web Dashboard:** http://localhost:5000
- **Health Check:** http://localhost:5000/health
- **Readiness:** http://localhost:5000/readiness

---

## Key Time Windows (Eastern Time)

| Time | Event | Description |
|------|-------|-------------|
| 9:30 AM | Market Opens | Bot waits 5 minutes |
| **9:35 AM** | **Entry Window Opens** | Bot starts trading |
| 11:35 AM | Entry Cutoff | No new positions |
| **3:45 PM** | **Force Close** | All positions liquidated |
| 4:00 PM | Market Closes | Review day's trades |

---

## File Locations

```
reports/
  ├── state/daily_state.json          # Persistent state (survives restarts)
  ├── runtime/
  │   ├── heartbeat_latest.json       # Current status
  │   ├── boots_YYYY-MM-DD.jsonl      # Boot/shutdown log
  │   └── did_run_YYYY-MM-DD.json     # Proof bot ran today
  ├── alerts/
  │   └── CRITICAL_*.txt              # Critical alerts (check these!)
  └── ...

daily_reports/
  └── YYYY-MM-DD.json                 # Daily performance report

logs/
  ├── out.log                         # PM2 stdout
  └── err.log                         # PM2 stderr
```

---

## Troubleshooting Quick Fixes

### Bot won't trade
```bash
# 1. Check time (must be 9:35 AM - 11:35 AM ET for new entries)
date -u +"%H:%M %Z"

# 2. Check health endpoint
curl http://localhost:5000/health | jq '.tradingState, .entryAllowed'

# 3. Check TIME_GUARD_OVERRIDE isn't blocking
grep TIME_GUARD_OVERRIDE .env
```

### Bot has overnight positions (BAD!)
```bash
# Close them manually via Alpaca dashboard
# Then investigate why force close failed

# Check logs for force close event
npm run pm2:logs --lines 200 | grep "FORCE"
```

### Daily counters reset mid-day
```bash
# Check persistent state
cat reports/state/daily_state.json

# If corrupted, bot will recreate from trades
npm run pm2:restart
```

---

## PM2 Cheat Sheet

```bash
npm run pm2:start      # Start the bot
npm run pm2:stop       # Stop the bot
npm run pm2:restart    # Restart the bot
npm run pm2:delete     # Delete and remove from PM2
npm run pm2:status     # Check status
npm run pm2:logs       # View logs (live)
npm run pm2:monit      # Interactive monitor
```

---

## Success Indicators

✅ **Bot is healthy when:**
- PM2 status shows "online"
- Health endpoint returns `"status": "ok"`
- Readiness returns HTTP 200
- Logs show regular `[ANALYSIS]` cycles
- No CRITICAL_*.txt files in reports/alerts/

❌ **Bot needs attention when:**
- PM2 status shows "errored" or "stopped"
- Health endpoint returns HTTP 500
- Readiness returns HTTP 503
- Logs show repeated errors
- CRITICAL_*.txt files appear
- Positions remain at 4:00 PM close

---

## Phase 1 Complete! 🎉

You now have:
- ✅ PM2 auto-restart on crash
- ✅ API retry logic (3x with backoff)
- ✅ Persistent state (survives restarts)
- ✅ Health check endpoints
- ✅ Smoke test for daily validation
- ✅ Alpaca clock synchronization
- ✅ **AI Mode Switcher (Testing ↔ Live)** 🆕
- ✅ Production build script

**Next:** Run for 2-4 weeks to establish baseline performance data, then implement Phase 2 (alerts, pre-market automation, analytics).

---

Print this card or keep it open in a terminal window!
