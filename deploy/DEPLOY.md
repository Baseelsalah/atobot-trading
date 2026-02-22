# AtoBot Deployment Guide — DigitalOcean

Deploy AtoBot to a $4/mo DigitalOcean droplet so it runs 24/7, survives reboots,
and auto-updates when you push code to GitHub.

---

## Architecture

```
Your laptop                    DigitalOcean droplet ($4/mo)
┌──────────┐   git push    ┌─────────────────────────────────────┐
│ VS Code  │──────────────→│ GitHub Actions (CI/CD)              │
│ AtoBot   │               │   ├── Run tests                     │
└──────────┘               │   └── Build & push Docker image     │
                           │         ↓                            │
                           │ ghcr.io (container registry)         │
                           │         ↓  (auto-pull every 5 min)  │
                           │ ┌─────────────────────────────────┐ │
                           │ │ Docker                          │ │
                           │ │   ├── atobot (trading bot)      │ │
                           │ │   ├── atobot-dashboard (:8501)  │ │
                           │ │   └── watchtower (auto-updater) │ │
                           │ └─────────────────────────────────┘ │
                           └─────────────────────────────────────┘
```

**Flow:** Push code → GitHub tests & builds image → Watchtower on VPS pulls it → bot restarts with new version. Zero manual SSH needed for updates.

---

## Step 1: Create a GitHub Repository

1. Go to https://github.com/new
2. Name: `atobot-trading` (private recommended)
3. Don't add README (we already have one)
4. Click **Create repository**
5. Run these commands in VS Code terminal:

```powershell
cd c:\Users\basee\OneDrive\Desktop\AtoBot\atobot-trading
git init
git add .
git commit -m "Initial commit: AtoBot stock day-trading bot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/atobot-trading.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

---

## Step 2: Create a DigitalOcean Droplet

1. Sign up at https://www.digitalocean.com (use any promo code for $200 credit)
2. Click **Create → Droplet**
3. Settings:
   - **Region:** New York (NYC1) — closest to NYSE
   - **Image:** Ubuntu 24.04 LTS
   - **Plan:** Basic → Regular → **$4/mo** (512 MB / 1 vCPU / 10 GB)
   - **Authentication:** SSH key (recommended) or password
   - **Hostname:** `atobot`
4. Click **Create Droplet**
5. Copy the IP address (e.g., `164.90.xxx.xxx`)

---

## Step 3: Set Up the VPS

SSH into your new droplet:

```bash
ssh root@YOUR_DROPLET_IP
```

Download and run the setup script:

```bash
curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/atobot-trading/main/deploy/setup-vps.sh | bash
```

This installs Docker, creates the `atobot` user, configures the firewall, creates swap,
and sets up a systemd service.

---

## Step 4: Clone & Configure

Switch to the atobot user and clone your repo:

```bash
su - atobot
cd /opt/atobot
git clone https://github.com/YOUR_USERNAME/atobot-trading.git .
```

Create your environment file:

```bash
cp .env.example .env
nano .env
```

**Critical settings to change:**
- `ALPACA_API_KEY` — your real Alpaca API key
- `ALPACA_API_SECRET` — your real Alpaca API secret
- `ALPACA_PAPER=true` — keep this true until you're confident!
- `DRY_RUN=false` — set to false to actually place orders
- `SYMBOLS` — your target stock symbols
- `STRATEGIES` — recommended: `["vwap_scalp","orb"]`

Save and exit (Ctrl+X, Y, Enter).

---

## Step 5: Set Up GHCR Authentication

So Watchtower can pull your private Docker images:

```bash
bash deploy/setup-ghcr-auth.sh
```

It will ask for your GitHub username and a Personal Access Token.
Create a token at https://github.com/settings/tokens/new with the `read:packages` scope.

---

## Step 6: Start the Bot

```bash
# Remove the override file (it's for local dev only)
rm -f docker-compose.override.yml

# Start everything
sudo systemctl start atobot
```

Verify it's running:

```bash
docker compose ps
docker compose logs -f bot
```

You should see AtoBot starting up, connecting to Alpaca, and waiting for market hours.

---

## Step 7: Verify Auto-Deploy

Back on your laptop, make any small change and push:

```powershell
git add . && git commit -m "test deploy" && git push
```

1. Go to https://github.com/YOUR_USERNAME/atobot-trading/actions to see CI/CD running
2. Within 5 minutes, Watchtower on the VPS will pull the new image and restart the bot
3. Verify: `ssh root@YOUR_DROPLET_IP "docker compose -f /opt/atobot/docker-compose.yml logs --tail=5 bot"`

---

## Useful Commands (SSH into VPS)

```bash
# Check bot status
docker compose ps

# View live logs
docker compose logs -f bot

# View last 100 lines of bot logs
docker compose logs --tail=100 bot

# Restart bot
docker compose restart bot

# Stop everything
sudo systemctl stop atobot

# Start everything
sudo systemctl start atobot

# Manual deploy (immediate, don't wait for Watchtower)
bash deploy/deploy.sh

# Check disk usage
df -h

# Check memory
free -m

# View bot's database
sqlite3 /opt/atobot/data/atobot.db "SELECT * FROM trades ORDER BY id DESC LIMIT 10;"
```

---

## Dashboard Access

The Streamlit dashboard is available at:

```
http://YOUR_DROPLET_IP:8501
```

> **Security note:** This is exposed to the internet. For production, add
> Nginx + password auth or restrict to your IP in UFW:
> ```bash
> ufw delete allow 8501/tcp
> ufw allow from YOUR_HOME_IP to any port 8501
> ```

---

## Monitoring & Alerts

AtoBot sends notifications via Telegram (if configured in `.env`):
- Trade executed alerts
- Daily performance summary (end of day)
- Error alerts

To enable:
1. Message [@BotFather](https://t.me/BotFather) on Telegram → `/newbot`
2. Copy the bot token → `TELEGRAM_BOT_TOKEN` in `.env`
3. Get your chat ID from [@userinfobot](https://t.me/userinfobot) → `TELEGRAM_CHAT_ID`
4. Set `NOTIFICATIONS_ENABLED=true`
5. Restart: `docker compose restart bot`

---

## Cost Breakdown

| Item | Monthly Cost |
|------|-------------|
| DigitalOcean droplet (512 MB) | $4.00 |
| Domain name (optional) | ~$1.00 |
| **Total** | **$4.00–$5.00** |

The bot uses ~150 MB RAM and minimal CPU. The $4 droplet handles it easily.

---

## Troubleshooting

**Bot won't start:**
```bash
docker compose logs bot   # Check error messages
cat .env                  # Verify API keys are set
```

**"Permission denied" errors:**
```bash
sudo usermod -aG docker atobot
newgrp docker
```

**Out of memory:**
```bash
free -m                   # Check memory
sudo fallocate -l 2G /swapfile  # Increase swap
```

**Can't pull images:**
```bash
docker login ghcr.io     # Re-authenticate
bash deploy/setup-ghcr-auth.sh
```

**Bot stopped trading mid-day:**
```bash
docker compose ps         # Check if container restarted
docker compose logs --tail=50 bot  # Look for errors
sudo systemctl status atobot      # Check systemd service
```
