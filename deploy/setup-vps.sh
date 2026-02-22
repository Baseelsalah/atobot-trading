#!/usr/bin/env bash
# ── AtoBot VPS Setup Script ──────────────────────────────────────────────────
# Run this on a fresh Ubuntu 22.04+ DigitalOcean droplet:
#   curl -sSL https://raw.githubusercontent.com/<YOU>/atobot-trading/main/deploy/setup-vps.sh | bash
# Or after cloning: bash deploy/setup-vps.sh
#
# What it does:
#   1. Updates the system
#   2. Installs Docker + Docker Compose
#   3. Creates the atobot user & directories
#   4. Configures UFW firewall (SSH + dashboard only)
#   5. Sets up automatic security updates
#   6. Creates a systemd service so the bot starts on boot
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[AtoBot]${NC} $*"; }
warn() { echo -e "${YELLOW}[AtoBot]${NC} $*"; }
err()  { echo -e "${RED}[AtoBot]${NC} $*"; exit 1; }

# ── Must run as root ─────────────────────────────────────────────────────────
[[ $EUID -eq 0 ]] || err "Run this script as root: sudo bash deploy/setup-vps.sh"

# ── 1. System update ─────────────────────────────────────────────────────────
log "Updating system packages..."
apt-get update -qq && apt-get upgrade -y -qq

# ── 2. Install Docker ────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    log "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
else
    log "Docker already installed."
fi

# ── 3. Install Docker Compose v2 (plugin) ────────────────────────────────────
if ! docker compose version &>/dev/null; then
    log "Installing Docker Compose plugin..."
    apt-get install -y docker-compose-plugin
else
    log "Docker Compose already installed."
fi

# ── 4. Create atobot user ────────────────────────────────────────────────────
if ! id -u atobot &>/dev/null; then
    log "Creating 'atobot' user..."
    useradd -m -s /bin/bash -G docker atobot
else
    log "User 'atobot' already exists."
    usermod -aG docker atobot
fi

# ── 5. Create app directory ──────────────────────────────────────────────────
APP_DIR="/opt/atobot"
mkdir -p "$APP_DIR"
chown atobot:atobot "$APP_DIR"

# ── 6. Firewall ──────────────────────────────────────────────────────────────
log "Configuring UFW firewall..."
apt-get install -y -qq ufw
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw allow 8501/tcp comment "AtoBot Dashboard"
echo "y" | ufw enable
ufw status verbose

# ── 7. Automatic security updates ────────────────────────────────────────────
log "Enabling unattended-upgrades..."
apt-get install -y -qq unattended-upgrades
dpkg-reconfigure -f noninteractive unattended-upgrades

# ── 8. Swap (tiny droplets need it) ──────────────────────────────────────────
if [ ! -f /swapfile ]; then
    log "Creating 1 GB swap..."
    fallocate -l 1G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
else
    log "Swap already exists."
fi

# ── 9. Systemd service ───────────────────────────────────────────────────────
log "Creating systemd service..."
cat > /etc/systemd/system/atobot.service << 'EOF'
[Unit]
Description=AtoBot Day-Trading Bot
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
User=atobot
Group=docker
WorkingDirectory=/opt/atobot
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
ExecReload=/usr/bin/docker compose restart
Restart=on-failure
RestartSec=10
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable atobot.service

# ── 10. Docker log rotation ──────────────────────────────────────────────────
log "Configuring Docker log rotation..."
mkdir -p /etc/docker
cat > /etc/docker/daemon.json << 'EOF'
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    }
}
EOF
systemctl restart docker

# ── Done ──────────────────────────────────────────────────────────────────────
log ""
log "╔══════════════════════════════════════════════════════════╗"
log "║  VPS setup complete!                                    ║"
log "║                                                         ║"
log "║  Next steps:                                            ║"
log "║  1. Clone your repo:                                    ║"
log "║     su - atobot                                         ║"
log "║     cd /opt/atobot                                      ║"
log "║     git clone https://github.com/YOU/atobot-trading .   ║"
log "║                                                         ║"
log "║  2. Create .env:                                        ║"
log "║     cp .env.example .env                                ║"
log "║     nano .env   # fill in your Alpaca keys              ║"
log "║                                                         ║"
log "║  3. Start the bot:                                      ║"
log "║     sudo systemctl start atobot                         ║"
log "║                                                         ║"
log "║  4. Check logs:                                         ║"
log "║     docker compose logs -f bot                          ║"
log "╚══════════════════════════════════════════════════════════╝"
log ""
