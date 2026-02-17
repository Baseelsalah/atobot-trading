#!/bin/bash
# Codespace Keep-Alive - prevents idle timeout by pinging local server
# Runs 24/7 to ensure Codespace never idles out and misses market open

LOG_FILE="/workspaces/atobot-trading/logs/keepalive.log"

# Trim log file if it gets too large (>5MB)
trim_log() {
  if [ -f "$LOG_FILE" ] && [ $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null) -gt 5242880 ]; then
    tail -1000 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Log trimmed to last 1000 lines" >> "$LOG_FILE"
  fi
}

while true; do
  ET_HOUR=$(TZ="America/New_York" date +%H | sed 's/^0//')
  ET_TIME=$(TZ="America/New_York" date +"%H:%M ET")

  # Ping local server to register activity (keeps Codespace alive)
  RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/api/health 2>/dev/null || echo "000")

  if [ "$ET_HOUR" -ge 6 ] && [ "$ET_HOUR" -lt 17 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] keepalive ping $ET_TIME status=$RESPONSE (trading hours)" >> "$LOG_FILE"
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] keepalive ping $ET_TIME status=$RESPONSE (off-hours)" >> "$LOG_FILE"
  fi

  # Trim log periodically
  trim_log

  # Ping every 5 minutes (well under 30-min idle timeout)
  sleep 300
done
