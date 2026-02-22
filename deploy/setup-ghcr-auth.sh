#!/usr/bin/env bash
# ── Configure GHCR Authentication for Watchtower ─────────────────────────────
# Run this ONCE on the VPS after setup to let Watchtower pull private images.
#
# Usage: bash deploy/setup-ghcr-auth.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
log()  { echo -e "${GREEN}[AtoBot]${NC} $*"; }
warn() { echo -e "${YELLOW}[AtoBot]${NC} $*"; }

log "This script configures Docker to pull from GitHub Container Registry."
log ""
log "You'll need a GitHub Personal Access Token (classic) with 'read:packages' scope."
log "Create one at: https://github.com/settings/tokens/new"
log ""

read -rp "GitHub username: " GITHUB_USER
read -rsp "Personal Access Token: " GITHUB_TOKEN
echo ""

log "Logging in to ghcr.io..."
echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin

log ""
log "Docker config saved. Watchtower will auto-pull new images."
log ""
warn "If you're running as 'atobot' user, make sure the token is configured there:"
warn "  su - atobot"
warn "  bash /opt/atobot/deploy/setup-ghcr-auth.sh"
