# AtoBot Testing & Daily Operations Guide

## ✅ Alpaca Connection & Clock Sync - VERIFIED

Your bot is **already** properly synced with Alpaca:

- ✅ Uses `alpaca.getClock()` API for real-time market status
- ✅ Refreshes every 60 seconds to stay synchronized
- ✅ Detects early close days automatically (Christmas Eve, Black Friday, etc.)
- ✅ Adjusts entry/exit windows dynamically based on market schedule
- ✅ No delays - uses Alpaca's authoritative market clock

**How it works:**
1. Bot calls `alpaca.getClock()` to get current market status
2. Caches clock data for 60 seconds to prevent API spam
3. Uses `next_close` timestamp to detect early closes
4. Automatically calculates dynamic cutoffs (entry at next_close - 5min, force close at next_close - 2min)

---

## 🧪 Pre-Market Testing (Run Daily)

### 1. Smoke Test (NEW - Just Created!)

**Run between 8:30-9:20 AM ET before market open:**

```bash
npm run smoke-test
```

**What it checks:**
- ✅ Alpaca API connection & authentication
- ✅ Account status & buying power
- ✅ Clock synchronization with Alpaca
- ✅ No overnight positions (day trading rule)
- ✅ Daily state persistence
- ✅ Environment configuration
- ✅ System resources (memory, disk space)

**Expected output:**
```
🟢 SMOKE TEST PASSED - READY FOR TRADING
   You can safely start the bot with: npm run pm2:start
```

### 2. Manual System Check

**Check bot status:**
```bash
npm run pm2:status
```

**Expected:** Bot should be "online" with 0 restarts (or low restart count)

**Check recent logs:**
```bash
npm run pm2:logs --lines 50
```

**Look for:**
- `[BOOT]` - Bot startup log with timestamp
- `Trading bot auto-started successfully`
- No error messages

### 3. Health Check Endpoints

**Test health endpoint:**
```bash
curl http://localhost:5000/health
```

**Expected response:**
```json
{
  "status": "ok",
  "uptime": 3600,
  "botStatus": "active",
  "marketStatus": "CLOSED",
  "entryAllowed": false,
  "tradingState": "ACTIVE_TRADING",
  "lastTickET": null,
  "ticksSinceBoot": 0,
  "pid": 12345,
  "memoryMB": 150
}
```

**Test readiness endpoint:**
```bash
curl http://localhost:5000/readiness
```

---

## 📅 Daily Trading Day Routine

### Morning (8:30 - 9:30 AM ET)

#### 8:30 AM - Initial Setup
1. **Run smoke test:**
   ```bash
   npm run smoke-test
   ```

2. **If bot isn't running, start it:**
   ```bash
   npm run build         # Build production bundle
   npm run pm2:start      # Start with PM2
   ```

3. **Verify bot started:**
   ```bash
   npm run pm2:status
   ```

#### 9:00 AM - Pre-Market Validation
1. **Check for overnight positions (shouldn't be any):**
   ```bash
   curl http://localhost:5000/api/trading/positions
   ```

2. **Verify account status:**
   - Check buying power > $1000
   - Verify it's the correct account (paper vs live)

3. **Check persistent state:**
   ```bash
   cat reports/state/daily_state.json
   ```

   Should show today's date with 0 entries and $0 P/L

#### 9:20 AM - Final Checks
1. **Monitor logs for market open prep:**
   ```bash
   npm run pm2:logs --lines 100
   ```

2. **Look for these logs:**
   - `[CONTROL] Market open watcher started`
   - `[TIME GUARD]` messages showing schedule
   - No error messages

### Market Open (9:30 - 9:35 AM ET)

**What happens automatically:**
- 9:30 AM: Market opens
- 9:30-9:35 AM: Bot waits (5-minute warmup period)
- 9:35 AM: **Entry window opens** - Bot starts trading

**Watch the logs:**
```bash
npm run pm2:logs --follow
```

**Look for:**
```
[CONTROL] Entry Window Open at 09:35 ET
[ANALYSIS] Running analysis cycle...
[ACTION=TRADE] symbol=SPY side=buy qty=10 price=450.25 strategy=VWAP_REVERSION
```

### During Trading Hours (9:35 AM - 3:45 PM ET)

**Check status every 30-60 minutes:**
```bash
curl http://localhost:5000/health | jq
```

**Monitor trade activity:**
```bash
# Check positions
curl http://localhost:5000/api/trading/positions | jq

# Check recent trades (via dashboard or API)
# Visit: http://localhost:5000
```

**Watch for alerts:**
```bash
ls -ltr reports/alerts/
```

If you see `CRITICAL_*.txt` files, investigate immediately.

### Entry Window Closes (11:35 AM ET)

**What happens:**
- Bot stops opening NEW positions
- Continues managing existing positions
- Applies trailing stops, take profits, stop losses

**What to verify:**
```bash
npm run pm2:logs --lines 50
```

Look for:
```
[TIME GUARD] Entry window closed - manage only
```

### Force Close (3:45 PM ET)

**What happens automatically:**
- Bot force-liquidates ALL positions
- No positions held overnight (day trading rule)

**What to verify:**
```bash
# Check positions (should be empty)
curl http://localhost:5000/api/trading/positions

# Check logs
npm run pm2:logs --lines 50
```

Look for:
```
[TIME GUARD] Force close triggered
[ACTION=EXIT] symbol=SPY trade_id=ato_xxx reason=time_guard_force_close
```

### Market Close (4:00 PM ET)

**What to check:**
1. **All positions closed:**
   ```bash
   curl http://localhost:5000/api/trading/positions
   ```

   Should return `[]` (empty array)

2. **Daily P/L summary:**
   ```bash
   cat daily_reports/$(date +%Y-%m-%d).json | jq
   ```

3. **Did-run proof file:**
   ```bash
   cat reports/runtime/did_run_$(date +%Y-%m-%d).json | jq
   ```

### Evening (After 4:00 PM ET)

**Review performance:**
1. Check daily report in `daily_reports/` directory
2. Review trade log for patterns
3. Check win rate and expectancy

**Optional - Stop bot:**
```bash
npm run pm2:stop
```

**Bot will auto-restart at 4 AM ET Monday-Friday** (configured in PM2)

---

## 🚨 Troubleshooting

### Bot Not Starting

**Check PM2 status:**
```bash
npm run pm2:status
```

**If status shows "errored":**
```bash
# View error logs
npm run pm2:logs --err --lines 50

# Delete and restart
npm run pm2:delete
npm run pm2:start
```

### Bot Not Trading

**Check trading state:**
```bash
curl http://localhost:5000/health | jq '.tradingState'
```

**Possible states:**
- `ACTIVE_TRADING` - Normal, should trade
- `MANAGE_ONLY` - Recovery mode, positions only
- `STOPPED` - Bot stopped trading

**Check time guard:**
```bash
npm run pm2:logs --lines 20 | grep "TIME GUARD"
```

### API Connection Issues

**Test Alpaca connection:**
```bash
curl -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_API_SECRET" \
     https://paper-api.alpaca.markets/v2/account
```

**If fails:**
- Verify API keys in `.env` file
- Check Alpaca dashboard for API status
- Ensure not hitting rate limits

### Lost Daily State

**If trade counters reset mid-day:**
```bash
# Check persistent state
cat reports/state/daily_state.json

# Manually restore (if needed)
# Edit the file and restart bot
```

---

## 📊 Monitoring Dashboard

**Access web dashboard:**
```
http://localhost:5000
```

**Shows:**
- Current positions
- Trade history
- Performance metrics
- Bot status

---

## 🔔 Alert System (Future - Phase 2)

**Will notify you via:**
- Email for critical failures
- SMS for urgent issues (optional)
- File-based alerts in `reports/alerts/`

**Alert types:**
- Process crash during market hours
- Stall detection (no tick for 5+ min)
- Daily loss limit hit
- Positions remaining at 4:00 PM
- API failures after retries

---

## 📝 Daily Checklist

### Pre-Market (8:30-9:30 AM ET)
- [ ] Run `npm run smoke-test`
- [ ] Verify bot status with `npm run pm2:status`
- [ ] Check no overnight positions
- [ ] Review previous day's performance
- [ ] Ensure sufficient buying power

### Market Open (9:35 AM ET)
- [ ] Watch logs for first trade
- [ ] Verify entry window opened
- [ ] Check positions are being opened

### Mid-Day Check (12:00 PM ET)
- [ ] Check health endpoint
- [ ] Review active positions
- [ ] Verify no critical alerts

### Pre-Close (3:30 PM ET)
- [ ] Check upcoming force close at 3:45 PM
- [ ] Review day's P/L
- [ ] Ensure positions will close properly

### Post-Market (4:15 PM ET)
- [ ] Verify all positions closed
- [ ] Review daily report
- [ ] Check for any alerts
- [ ] Review trade quality

---

## 🎯 Success Metrics to Track

**Daily:**
- Number of trades executed
- Win rate %
- Realized P/L
- Positions held at close (should be 0)

**Weekly:**
- Uptime during market hours
- Market open capture rate (should be 100%)
- Average trade quality
- Strategy performance (VWAP vs ORB)

**Monthly:**
- Total profit/loss
- Sharpe ratio
- Max drawdown
- System reliability (crashes, restarts)

---

## Next Steps

After running successfully for 2-4 weeks:
1. Analyze baseline performance metrics
2. Identify which strategies are winning
3. Implement Phase 2 (email alerts, pre-market script automation)
4. Optimize underperforming strategies
5. Consider expanding trading window or adding strategies
