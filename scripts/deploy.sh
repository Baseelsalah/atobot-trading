#!/bin/bash
# AtoBot Deployment Script
# Run this on your VPS/server to install everything

set -e  # Exit on error

echo "🤖 AtoBot Deployment Script"
echo "============================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}⚠️  Please do not run as root. Run as your normal user with sudo access.${NC}"
    exit 1
fi

echo "This script will:"
echo "  1. Install Node.js (if needed)"
echo "  2. Install PM2 globally"
echo "  3. Install project dependencies"
echo "  4. Build production bundle"
echo "  5. Install cron jobs"
echo "  6. Start the bot"
echo ""
read -p "Continue? (yes/no): " CONTINUE

if [ "$CONTINUE" != "yes" ]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "=========================================="
echo "Step 1: Checking Node.js"
echo "=========================================="

if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v)
    echo -e "${GREEN}✅ Node.js already installed: $NODE_VERSION${NC}"
else
    echo -e "${YELLOW}📦 Installing Node.js...${NC}"
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
    echo -e "${GREEN}✅ Node.js installed${NC}"
fi

echo ""
echo "=========================================="
echo "Step 2: Installing PM2"
echo "=========================================="

if command -v pm2 &> /dev/null; then
    PM2_VERSION=$(pm2 -v)
    echo -e "${GREEN}✅ PM2 already installed: $PM2_VERSION${NC}"
else
    echo -e "${YELLOW}📦 Installing PM2...${NC}"
    sudo npm install -g pm2
    echo -e "${GREEN}✅ PM2 installed${NC}"
fi

echo ""
echo "=========================================="
echo "Step 3: Installing Project Dependencies"
echo "=========================================="

if [ ! -f "package.json" ]; then
    echo -e "${RED}❌ Error: package.json not found${NC}"
    echo "Please run this script from the atobot-trading directory."
    exit 1
fi

echo -e "${YELLOW}📦 Running npm install...${NC}"
npm install
echo -e "${GREEN}✅ Dependencies installed${NC}"

echo ""
echo "=========================================="
echo "Step 4: Building Production Bundle"
echo "=========================================="

echo -e "${YELLOW}🔨 Building...${NC}"
npm run build
echo -e "${GREEN}✅ Production build complete${NC}"

echo ""
echo "=========================================="
echo "Step 5: Environment Configuration"
echo "=========================================="

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  No .env file found${NC}"
    echo "Creating .env from example..."

    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}📝 Please edit .env and add your API keys:${NC}"
        echo "   nano .env"
        echo ""
        read -p "Press ENTER after you've configured .env..."
    else
        echo -e "${RED}❌ .env.example not found${NC}"
        echo "Please create .env manually with your API keys."
        exit 1
    fi
fi

echo -e "${GREEN}✅ Environment configuration ready${NC}"

echo ""
echo "=========================================="
echo "Step 6: Installing Cron Jobs"
echo "=========================================="

echo -e "${YELLOW}📅 Installing automation cron jobs...${NC}"
bash scripts/install-cron.sh <<< "yes"
echo -e "${GREEN}✅ Cron jobs installed${NC}"

echo ""
echo "=========================================="
echo "Step 7: Starting the Bot"
echo "=========================================="

# Stop any existing instance
pm2 delete atobot 2>/dev/null || true

echo -e "${YELLOW}▶️  Starting bot with PM2...${NC}"
npm run pm2:start

# Save PM2 process list
pm2 save

# Setup PM2 to start on boot
echo -e "${YELLOW}🔧 Setting up PM2 startup script...${NC}"
PM2_STARTUP=$(pm2 startup | tail -n 1)
if [[ $PM2_STARTUP == sudo* ]]; then
    echo "Please run this command manually:"
    echo "$PM2_STARTUP"
    read -p "Press ENTER after running the command..."
fi

echo -e "${GREEN}✅ Bot started with PM2${NC}"

echo ""
echo "=========================================="
echo "Step 8: Verification"
echo "=========================================="

echo -e "${YELLOW}🔍 Checking bot status...${NC}"
npm run pm2:status

echo ""
echo -e "${YELLOW}🔍 Checking cron jobs...${NC}"
crontab -l | grep atobot | wc -l | xargs echo "Cron jobs installed:"

echo ""
echo -e "${YELLOW}🔍 Checking health endpoint...${NC}"
sleep 5
curl -s http://localhost:5000/health | jq . || echo "Health check pending..."

echo ""
echo "=========================================="
echo "🎉 DEPLOYMENT COMPLETE!"
echo "=========================================="
echo ""
echo "Your bot is now running! Here's what happens automatically:"
echo ""
echo "  🌅 8:30 AM ET  - Pre-market routine (validates & starts trading)"
echo "  👀 Every 30min - Health monitoring (auto-restart if needed)"
echo "  🌙 4:15 PM ET  - Post-market routine (close & switch to test mode)"
echo "  📊 Sunday 6PM  - Weekly performance report"
echo "  💾 11:55 PM    - Daily backup"
echo "  📈 1st of Mon  - Monthly strategy analysis"
echo ""
echo "Useful Commands:"
echo "  npm run pm2:status         # Check bot status"
echo "  npm run pm2:logs           # View live logs"
echo "  npm run pm2:monit          # Interactive monitoring"
echo "  npm run mode:status        # Check trading mode"
echo "  curl http://localhost:5000/health | jq  # Health check"
echo ""
echo "Logs:"
echo "  PM2 logs:        npm run pm2:logs"
echo "  Automation logs: cat reports/automation.jsonl"
echo "  Cron logs:       tail -f /tmp/atobot-*.log"
echo ""
echo "Documentation:"
echo "  - AUTOMATION_COMPLETE.md    (Quick start)"
echo "  - INSTALLATION_COMPLETE.md  (Full guide)"
echo "  - QUICK_REFERENCE.md        (Command cheat sheet)"
echo ""
echo -e "${GREEN}✅ Your AI trading agent is now fully operational! 🤖📈${NC}"
echo ""
