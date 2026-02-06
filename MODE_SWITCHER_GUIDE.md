# 🤖 Mode Switcher Guide

The AI agent for automated mode switching between testing and live trading.

## Quick Commands

```bash
# Check current mode
npm run mode:status

# Switch to testing mode (safe, no real orders)
npm run mode:test

# Switch to live trading mode (REQUIRES CONFIRMATION)
npm run mode:live

# Rollback last mode change
npm run mode:rollback
```

## Features

### ✅ Safety Features
- **Automatic backup** - Creates `.env.backup` before any changes
- **Smoke test validation** - Runs full smoke test in target mode
- **Confirmation required** - Live mode requires typing "CONFIRM LIVE MODE"
- **Audit logging** - All mode changes logged to `reports/mode_changes.jsonl`
- **Rollback capability** - Can instantly revert last change

### 🧪 Testing Mode
When you run `npm run mode:test`, it sets:
- `DRY_RUN=1` - No real orders placed
- `TIME_GUARD_OVERRIDE=1` - Can trade outside market hours
- `SIM_CLOCK_OPEN=1` - Simulates market being open
- `SIM_TIME_ET=2026-02-04 10:00` - Fixed simulation time

**Use for:**
- Strategy testing
- Code changes verification
- Pre-market validation
- Integration testing

### 🔴 Live Trading Mode
When you run `npm run mode:live`, it sets:
- `DRY_RUN=0` - Real orders will be placed
- `TIME_GUARD_OVERRIDE=0` - FORT KNOX time guard active
- Removes all simulation variables

**Requirements before switching to live:**
1. ✅ Smoke test passes
2. ✅ Verified Alpaca API keys (live account)
3. ✅ Checked account buying power > $1000
4. ✅ Reviewed strategy performance in testing
5. ✅ Confirmed time guard schedule is correct

## Usage Examples

### Example 1: Pre-Market Testing to Live

```bash
# Morning routine (8:30 AM ET)
# 1. Check current mode
npm run mode:status

# 2. Run smoke test in testing mode
npm run smoke-test

# 3. Switch to live mode
npm run mode:live
# Type: CONFIRM LIVE MODE

# 4. Restart bot with new config
npm run pm2:restart

# 5. Verify bot is in live mode
npm run pm2:logs --lines 50 | grep "DRY_RUN"
```

### Example 2: Emergency Rollback to Testing

```bash
# If something goes wrong in live mode
# 1. Immediately rollback
npm run mode:rollback

# 2. Restart bot
npm run pm2:restart

# 3. Verify back in testing mode
npm run mode:status
```

### Example 3: End-of-Day Switch to Testing

```bash
# After market close (4:15 PM ET)
# 1. Switch to testing mode for tomorrow's prep
npm run mode:test

# 2. Bot auto-restarts or you can do it manually
npm run pm2:restart

# 3. Test any strategy changes in safe mode
npm run smoke-test
```

## How It Works

### 1. Reading Current Configuration
- Parses `.env` file
- Preserves comments and structure
- Detects current mode (Testing vs Live)

### 2. Mode Switch Process
1. **Backup** - Copies `.env` to `.env.backup`
2. **Validation** - Checks if already in target mode
3. **Confirmation** - Prompts for "CONFIRM LIVE MODE" (live only)
4. **Update** - Modifies `.env` with new configuration
5. **Smoke Test** - Runs full validation in target mode
6. **Audit Log** - Records change to `reports/mode_changes.jsonl`
7. **Instructions** - Displays next steps (restart bot)

### 3. Rollback Process
1. Checks if `.env.backup` exists
2. Copies backup to `.env`
3. Confirms rollback complete

### 4. Audit Logging
Every mode change is logged with:
- Timestamp
- From mode → To mode
- User who made the change
- Smoke test result (pass/fail)

**Log location:** `reports/mode_changes.jsonl`

**Example log entry:**
```json
{
  "timestamp": "2026-02-06T14:30:00.000Z",
  "from": "TESTING (Full Simulation)",
  "to": "LIVE TRADING",
  "user": "trader",
  "validated": true,
  "smokeTestPassed": true
}
```

## Safety Checks

### Before Switching to Live Mode
The script will:
1. ✅ Prompt for explicit confirmation
2. ✅ Display warning about real money
3. ✅ Run smoke test in live configuration
4. ✅ Verify all required environment variables
5. ✅ Check Alpaca connection
6. ✅ Ensure no overnight positions

### If Smoke Test Fails
- Mode is still switched (config updated)
- Warning displayed: "Fix issues before starting bot"
- Suggests rollback command
- Does NOT auto-restart bot

## Integration with Daily Workflow

### Morning Routine (8:30 AM ET)
```bash
# 1. Check mode
npm run mode:status

# 2. If in testing, switch to live
npm run mode:live

# 3. Run smoke test
npm run smoke-test

# 4. Start/restart bot
npm run pm2:restart

# 5. Watch logs for first trade
npm run pm2:logs --follow
```

### Evening Routine (4:15 PM ET)
```bash
# 1. Verify all positions closed
curl http://localhost:5000/api/trading/positions

# 2. Optional: Switch back to testing for safety
npm run mode:test

# 3. Review audit log
cat reports/mode_changes.jsonl | tail -5
```

## Troubleshooting

### "No backup found. Cannot rollback."
- You haven't switched modes yet (no backup created)
- Or you manually deleted `.env.backup`
- **Fix:** Manually edit `.env` or re-run mode switcher

### Smoke test fails after switch
- **Don't panic** - config is updated, bot not restarted yet
- **Fix issues** identified in smoke test
- **Then restart:** `npm run pm2:restart`
- **Or rollback:** `npm run mode:rollback`

### Bot still in old mode after switch
- **Cause:** Bot reads `.env` on startup only
- **Fix:** Restart bot with `npm run pm2:restart`
- **Verify:** `npm run pm2:logs --lines 20 | grep "DRY_RUN"`

### "CONFIRM LIVE MODE" required but testing automation
- **For automation:** Don't use `mode:live` directly
- **Instead:** Use `mode:test` which doesn't require confirmation
- **For live:** Manual intervention required (safety feature)

## Advanced Usage

### Programmatic Mode Check (Bash Scripts)
```bash
#!/bin/bash
# Check if in live mode
if grep -q "^DRY_RUN=0" .env; then
  echo "Live mode active"
else
  echo "Testing mode active"
fi
```

### Pre-Commit Hook (Prevent Live Config Commits)
```bash
# .git/hooks/pre-commit
#!/bin/bash
if grep -q "^DRY_RUN=0" .env; then
  echo "ERROR: .env is in LIVE mode. Do not commit!"
  exit 1
fi
```

### Automated Daily Schedule (Cron)
```cron
# Switch to live mode at 8:45 AM ET (13:45 UTC) on weekdays
45 13 * * 1-5 cd /path/to/atobot && npm run mode:live

# Switch back to testing at 4:15 PM ET (21:15 UTC)
15 21 * * 1-5 cd /path/to/atobot && npm run mode:test
```

**WARNING:** Automated live mode switching bypasses confirmation. Use with extreme caution.

## File Locations

```
.env                              # Current configuration
.env.backup                       # Last backup (for rollback)
reports/mode_changes.jsonl        # Audit log of all switches
scripts/mode-switcher.ts          # Mode switcher source code
```

## Security Notes

1. **Never commit `.env`** - Contains API keys
2. **Never commit `.env.backup`** - Contains API keys
3. **Review `mode_changes.jsonl`** - May contain usernames
4. **Limit live mode access** - Only authorized users
5. **Use PM2 for production** - Automatic restarts

## Next Steps

Now that you have the mode switcher:
1. ✅ Test switching between modes
2. ✅ Integrate into daily routine
3. ✅ Review audit logs weekly
4. ✅ Set up alerts for mode changes (optional)
5. ✅ Document your specific workflow

---

**Remember:** Live mode = real money. Always run smoke test first!
