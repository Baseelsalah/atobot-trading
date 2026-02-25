#!/usr/bin/env bash
# ── AtoBot Paper-Trading Quick Start ─────────────────────────────────────────
# Sets up paper trading locally with Docker. No VPS needed.
#
# Prerequisites:
#   - Docker Desktop installed and running
#   - .env file with ALPACA_API_KEY and ALPACA_API_SECRET set
#   - ALPACA_PAPER=true in .env
#
# Usage:
#   chmod +x deploy/paper_trade.sh
#   ./deploy/paper_trade.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  AtoBot v2 — Paper-Trading Launcher"
echo "  LONG + SHORT · PAIRS · ML · 5 STRATEGIES"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Check prerequisites ──────────────────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Install Docker Desktop first."
    exit 1
fi

if ! docker info &> /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Start Docker Desktop first."
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo ""
    echo "IMPORTANT: Edit .env and set these values:"
    echo "  ALPACA_API_KEY=your_paper_key"
    echo "  ALPACA_API_SECRET=your_paper_secret"
    echo "  ALPACA_PAPER=true"
    echo ""
    echo "Get paper-trading keys from: https://app.alpaca.markets/ → Paper Trading → API Keys"
    echo ""
    exit 1
fi

# ── Validate .env ─────────────────────────────────────────────────────────────
if grep -q "your_alpaca_key_here" .env 2>/dev/null; then
    echo "ERROR: .env still has placeholder API keys. Edit .env first."
    exit 1
fi

# Check paper mode
if ! grep -qi "ALPACA_PAPER=true" .env 2>/dev/null; then
    echo ""
    echo "⚠️  WARNING: ALPACA_PAPER is not set to 'true' in .env"
    echo "   This script is for PAPER TRADING only."
    echo ""
    read -p "Continue anyway? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        exit 1
    fi
fi

# ── Show config ───────────────────────────────────────────────────────────────
echo "Configuration:"
echo "  Paper Mode:    $(grep ALPACA_PAPER .env | head -1)"
echo "  Strategies:    $(grep STRATEGIES .env | head -1)"
echo "  Symbols:       $(grep '^SYMBOLS=' .env | head -1)"
echo "  Short Selling: $(grep SHORT_SELLING_ENABLED .env 2>/dev/null | head -1 || echo 'true (default)')"
echo "  Pairs Trading: $(grep PAIRS_TRADING_ENABLED .env 2>/dev/null | head -1 || echo 'true (default)')"
echo "  ML Model:      $(grep ML_FEATURES_ENABLED .env 2>/dev/null | head -1 || echo 'true (default)')"
echo ""

# ── Build & Start ─────────────────────────────────────────────────────────────
echo "Building Docker image..."
docker compose build --no-cache bot

echo ""
echo "Starting AtoBot (bot + dashboard)..."
docker compose up -d bot dashboard

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ AtoBot is running in paper-trading mode!"
echo ""
echo "  Dashboard:  http://localhost:8501"
echo "  Bot logs:   docker compose logs -f bot"
echo "  Stop:       docker compose down"
echo ""
echo "  Monitor:"
echo "    docker compose logs -f bot     # Live trading log"
echo "    docker compose logs dashboard  # Dashboard log"
echo "    docker compose ps              # Container status"
echo "═══════════════════════════════════════════════════════════════"
echo ""
