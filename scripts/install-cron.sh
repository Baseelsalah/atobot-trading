#!/bin/bash
# AtoBot Cron Job Installer
# This script installs all automation cron jobs for hands-free operation

echo "🤖 AtoBot Cron Job Installer"
echo "===================================="
echo ""

# Get the current directory
ATOBOT_DIR=$(pwd)
echo "📁 AtoBot Directory: $ATOBOT_DIR"
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found."
    echo "   Please run this script from the atobot-trading directory."
    exit 1
fi

# Create temporary crontab file
TEMP_CRON=$(mktemp)

# Preserve existing crontab (if any)
crontab -l > "$TEMP_CRON" 2>/dev/null || true

# Add header
cat >> "$TEMP_CRON" << 'EOF'

# ========================================
# AtoBot Trading Automation
# Auto-generated cron jobs
# ========================================

# Set PATH for npm/node
PATH=/usr/local/bin:/usr/bin:/bin
SHELL=/bin/bash

EOF

# Add pre-market routine (8:30 AM ET = 1:30 PM UTC standard time)
echo "# Pre-Market Routine - 8:30 AM ET (1:30 PM UTC)" >> "$TEMP_CRON"
echo "30 13 * * 1-5 cd $ATOBOT_DIR && npm run daily:premarket >> /tmp/atobot-premarket.log 2>&1" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Add monitoring (every 30 min during trading hours)
echo "# Trading Hours Monitor - Every 30 min (9:30 AM - 4:00 PM ET)" >> "$TEMP_CRON"
echo "*/30 14-20 * * 1-5 cd $ATOBOT_DIR && npm run daily:monitor >> /tmp/atobot-monitor.log 2>&1" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Add post-market routine (4:15 PM ET = 9:15 PM UTC)
echo "# Post-Market Routine - 4:15 PM ET (9:15 PM UTC)" >> "$TEMP_CRON"
echo "15 21 * * 1-5 cd $ATOBOT_DIR && npm run daily:postmarket >> /tmp/atobot-postmarket.log 2>&1" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Add weekly performance report (Sunday 6:00 PM)
echo "# Weekly Performance Report - Sunday 6:00 PM" >> "$TEMP_CRON"
echo "0 18 * * 0 cd $ATOBOT_DIR && npm run weekly:report >> /tmp/atobot-weekly.log 2>&1" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Add daily backup (11:55 PM)
echo "# Daily Backup - 11:55 PM" >> "$TEMP_CRON"
echo "55 23 * * * cd $ATOBOT_DIR && npm run daily:backup >> /tmp/atobot-backup.log 2>&1" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Add strategy analysis (Monthly, 1st day at 2:00 AM)
echo "# Monthly Strategy Analysis - 1st of month at 2:00 AM" >> "$TEMP_CRON"
echo "0 2 1 * * cd $ATOBOT_DIR && npm run monthly:analyze >> /tmp/atobot-monthly.log 2>&1" >> "$TEMP_CRON"
echo "" >> "$TEMP_CRON"

# Show what will be installed
echo "📋 Cron jobs to be installed:"
echo "================================"
cat "$TEMP_CRON" | grep -v "^#" | grep -v "^$"
echo ""

# Ask for confirmation
read -p "Install these cron jobs? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "❌ Installation cancelled."
    rm "$TEMP_CRON"
    exit 0
fi

# Install crontab
crontab "$TEMP_CRON"
rm "$TEMP_CRON"

echo ""
echo "✅ Cron jobs installed successfully!"
echo ""
echo "📊 Verify installation:"
echo "   crontab -l"
echo ""
echo "📝 View logs:"
echo "   tail -f /tmp/atobot-premarket.log"
echo "   tail -f /tmp/atobot-monitor.log"
echo "   tail -f /tmp/atobot-postmarket.log"
echo ""
echo "🧪 Test automation manually:"
echo "   npm run daily:premarket"
echo "   npm run daily:monitor"
echo "   npm run daily:postmarket"
echo ""
echo "🚀 Automation is now active!"
