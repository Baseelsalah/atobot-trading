#!/usr/bin/env bash
# ── AtoBot Quick Deploy ──────────────────────────────────────────────────────
# Run on the VPS to manually pull latest & restart (Watchtower does this
# automatically, but this is handy for immediate deploys).
#
# Usage: bash deploy/deploy.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

GREEN='\033[0;32m'
NC='\033[0m'
log() { echo -e "${GREEN}[AtoBot]${NC} $*"; }

cd /opt/atobot

log "Pulling latest code..."
git pull origin main

log "Pulling latest Docker image..."
docker compose pull bot

log "Restarting containers..."
docker compose up -d --remove-orphans

log "Cleaning up old images..."
docker image prune -f

log "Current status:"
docker compose ps

log "Deploy complete! Tailing logs (Ctrl+C to stop)..."
docker compose logs -f --tail=20 bot
