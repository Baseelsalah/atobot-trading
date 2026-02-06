#!/bin/bash
# Demo Automation - Start AI Agent in Paper Trading Mode
# This shows you how the automation works without risking real money

echo "🤖 Starting AtoBot AI Agent Demo"
echo "=========================================="
echo ""
echo "This demo will:"
echo "  1. Run pre-market automation"
echo "  2. Show you how monitoring works"
echo "  3. Run post-market automation"
echo ""
echo "⚠️  SAFE: Running in PAPER TRADING mode (DRY_RUN=1)"
echo ""
read -p "Press ENTER to start demo..."
echo ""

# Ensure we're in paper trading mode
if ! grep -q "^DRY_RUN=1" .env 2>/dev/null; then
    echo "⚠️  Setting DRY_RUN=1 for safety..."
    if [ -f .env ]; then
        # Update existing .env
        sed -i 's/^DRY_RUN=0/DRY_RUN=1/' .env 2>/dev/null || true
        if ! grep -q "^DRY_RUN=" .env; then
            echo "DRY_RUN=1" >> .env
        fi
    else
        echo "DRY_RUN=1" > .env
        echo "TIME_GUARD_OVERRIDE=1" >> .env
        echo "SIM_CLOCK_OPEN=1" >> .env
        echo "SIM_TIME_ET=$(date +%Y-%m-%d) 10:00" >> .env
    fi
    echo "✅ Paper trading mode enabled"
    echo ""
fi

# Demo 1: Pre-Market Automation
echo "=========================================="
echo "DEMO 1: Pre-Market Automation"
echo "=========================================="
echo ""
echo "This runs automatically at 8:30 AM ET every trading day."
echo "It will:"
echo "  - Run smoke test"
echo "  - Switch to live mode (but we'll keep it in paper mode for demo)"
echo "  - Start the bot"
echo "  - Verify health"
echo ""
read -p "Press ENTER to run pre-market automation..."
echo ""

npm run daily:premarket

echo ""
echo "✅ Pre-market automation complete!"
echo ""
read -p "Press ENTER for next demo..."
echo ""

# Demo 2: Monitoring
echo "=========================================="
echo "DEMO 2: Trading Hours Monitoring"
echo "=========================================="
echo ""
echo "This runs every 30 minutes during trading hours."
echo "It will:"
echo "  - Check bot is running"
echo "  - Auto-restart if crashed"
echo "  - Check health"
echo "  - Detect stalls"
echo ""
read -p "Press ENTER to run monitoring check..."
echo ""

npm run daily:monitor

echo ""
echo "✅ Monitoring check complete!"
echo ""
read -p "Press ENTER for next demo..."
echo ""

# Demo 3: Post-Market
echo "=========================================="
echo "DEMO 3: Post-Market Automation"
echo "=========================================="
echo ""
echo "This runs automatically at 4:15 PM ET every trading day."
echo "It will:"
echo "  - Verify positions closed"
echo "  - Review daily report"
echo "  - Switch to test mode"
echo ""
read -p "Press ENTER to run post-market automation..."
echo ""

npm run daily:postmarket

echo ""
echo "✅ Post-market automation complete!"
echo ""

# Demo 4: Show other features
echo "=========================================="
echo "BONUS: Other Automation Features"
echo "=========================================="
echo ""
echo "Available automation scripts:"
echo ""
echo "  npm run weekly:report      - Weekly performance analysis"
echo "  npm run monthly:analyze    - Strategy auto-tuning"
echo "  npm run daily:backup       - Backup all trading data"
echo ""
echo "These run automatically via cron jobs when you install them."
echo ""
read -p "Press ENTER to see weekly report demo..."
echo ""

npm run weekly:report

echo ""
echo "=========================================="
echo "✅ DEMO COMPLETE!"
echo "=========================================="
echo ""
echo "What you just saw will happen AUTOMATICALLY every day when you install cron jobs."
echo ""
echo "Next steps:"
echo "  1. Review the automation logs created"
echo "  2. Check reports/automation.jsonl for all events"
echo "  3. Install cron jobs: bash scripts/install-cron.sh"
echo ""
echo "Your bot will then run 100% hands-free! 🤖"
echo ""
