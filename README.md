# AtoBot Trading Dashboard

AI-powered stock trading dashboard with Alpaca API integration.

## 🎯 Running a Simulation (DRY_RUN Mode)

The bot supports simulation mode where it analyzes the market and generates signals WITHOUT executing real trades.

### How to run a simulation:

1. **Set up your environment:**
   ```bash
   # Create .env file with DRY_RUN enabled
   echo "DRY_RUN=1" > .env
   echo "ALPACA_API_KEY=your_alpaca_key" >> .env
   echo "ALPACA_API_SECRET=your_alpaca_secret" >> .env
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the bot:**
   ```bash
   npm run dev    # For development mode
   # or
   npm start      # For production mode
   ```

4. **What you should see:**
   - `[SIM] DRY_RUN active - orders will be logged but not executed`
   - Market data fetching
   - Signal analysis (BUY/SELL/HOLD)
   - `[DRY_RUN]` prefixed trade logs
   - PnL calculations
   - No actual orders sent to broker

5. **Monitor the simulation:**
   - Let it run for 5-10 minutes
   - Watch for analysis cycles
   - Verify no crashes or freezes
   - Check that signals are being generated

**Note:** Even in DRY_RUN mode, you need valid Alpaca API credentials because the bot fetches real market data and checks market hours. Only trade execution is disabled.

## Keep the bot running with tmux

Use tmux so the bot keeps running after you disconnect.

1. Install tmux:
	- `sudo apt install tmux`
2. Start a tmux session:
	- `tmux new -s tradingbot`
3. Run your bot inside tmux:
	- `npm run dev` (for development) or `npm start` (for production)
4. Detach (bot keeps running):
	- Press `Ctrl + B`, then `D`
5. Reattach later:
	- `tmux attach -t tradingbot`

Result: You can close your laptop, disconnect, or shut VS Code—the bot keeps running.

## Verify it keeps running after you close your laptop

The bot only keeps running if it’s on a machine that stays awake. Use this quick test on a remote server:

1. Start your bot inside tmux:
	- `tmux new -s tradingbot`
	- `npm run dev` (for development) or `npm start` (for production)
2. Detach from tmux:
	- Press `Ctrl + B`, then `D`
3. Close your laptop (sleep/disconnect is fine).
4. Reopen your laptop and SSH back into the server.
5. Reattach:
	- `tmux attach -t tradingbot`
6. If logs are still running, the bot is persistent.

Note: If you run the bot on your laptop, closing the lid stops it. tmux only helps if the machine itself stays awake (VPS, cloud server, Raspberry Pi, mini-PC).

## Deploy on a VPS (24/7 uptime)

A VPS is a remote computer that stays on 24/7, perfect for running the bot even when your laptop is closed.

1. Create a VPS (DigitalOcean, Linode, AWS, etc.).
2. SSH into the server.
3. Install prerequisites:
	- Node.js (v18 or later)
	- npm
	- Git
4. Clone the repo and install dependencies:
	- `git clone <repo-url>`
	- `cd atobot-trading`
	- `npm install`
5. Set up environment variables:
	- Create a `.env` file with required API keys (see `.env.example` if available)
	- Set `DRY_RUN=1` for simulation mode
6. Run the bot inside tmux (see the tmux section above).

Optional: For production, use a process manager (systemd or pm2) to auto-restart the bot if it crashes.

### Auto-restart with systemd (recommended on Linux)

1. Create a service file at `/etc/systemd/system/atobot.service`:
	```
	[Unit]
	Description=AtoBot Trading Bot
	After=network.target

	[Service]
	Type=simple
	User=YOUR_USER
	WorkingDirectory=/path/to/atobot-trading
	ExecStart=/usr/bin/npm start
	Restart=always
	RestartSec=5

	[Install]
	WantedBy=multi-user.target
	```
2. Reload and start:
	- `sudo systemctl daemon-reload`
	- `sudo systemctl enable atobot`
	- `sudo systemctl start atobot`
3. Check status:
	- `sudo systemctl status atobot`

### Auto-restart with pm2 (Node-based process manager)

1. Install pm2:
	- `npm install -g pm2`
2. Start the bot:
	- `pm2 start npm --name atobot -- start`
3. Save and enable startup:
	- `pm2 save`
	- `pm2 startup`

## Health check (verify uptime and persistence)

Use this checklist to confirm the bot is running and persistent.

1. Check tmux sessions:
	- `tmux ls`
2. Attach to the bot session:
	- `tmux attach -t tradingbot`
	- If logs are printing, the bot is alive.
3. Detach, disconnect, and reattach later:
	- Press `Ctrl + B`, then `D`
	- Reconnect and run `tmux attach -t tradingbot`
	- If logs continued during your absence, persistence is confirmed.

### If using pm2

1. Check status:
	- `pm2 status`
2. Reboot test:
	- `sudo reboot`
	- After reconnecting, run `pm2 status`
	- If the bot is **online**, auto-restart works.

### If using systemd

1. Check status:
	- `systemctl status tradingbot`
2. Reboot test:
	- `sudo reboot`
	- After reconnecting, run `systemctl status tradingbot`
	- If it’s **active (running)**, auto-restart works.

## Quick status check (fast verification)

Use these commands to confirm the bot is running without digging through logs.

1. Check if the tmux session is alive:
	- `tmux ls`
	- If you see `tradingbot`, the session is running.
2. Reattach to see live logs:
	- `tmux attach -t tradingbot`
	- If logs are printing, the bot is active.
3. Check pm2 status (if using pm2):
	- `pm2 status`
	- The bot should show as **online**.
4. Check systemd status (if using systemd):
	- `systemctl status tradingbot`
	- Look for **active (running)**.
5. Confirm the process is alive:
	- `ps aux | grep node`
	- If you see a Node.js process for the bot, it’s running.

## Daily operator checklist (before market open)

Use this quick routine each morning to confirm the bot is healthy and ready for the trading day.

1. Log into your VPS:
	- `ssh youruser@your-vps-ip`
2. Check the tmux session:
	- `tmux ls`
	- You should see `tradingbot: 1 windows`.
3. Confirm the bot is running:
	- `tmux attach -t tradingbot`
	- Look for new log entries, analysis cycles, DRY_RUN trades, and no frozen output.
	- Detach: Press `Ctrl + B`, then `D`.
4. If using pm2, check status:
	- `pm2 status`
	- The bot should show **online**.
5. If using systemd, check service health:
	- `systemctl status tradingbot`
	- Look for **active (running)**.
6. Confirm the process exists:
	- `ps aux | grep node`
	- If you see a Node.js process for the bot, it’s alive.
7. Optional: Check logs:
	- pm2: `pm2 logs tradingbot`
	- systemd: `journalctl -u tradingbot -f`

## Smoke-screen test (fast readiness check)

Run these steps in order. If all pass, the bot is ready for the next trading day.

1. SSH into your VPS:
	- `ssh youruser@your-vps-ip`
2. Check if the tmux session exists:
	- `tmux ls`
	- You should see `tradingbot: 1 windows`.
3. Attach to tmux and check for live logs:
	- `tmux attach -t tradingbot`
	- Look for new timestamps, analysis cycles, heartbeat logs, DRY_RUN checks, and no frozen output.
	- Detach: Press `Ctrl + B`, then `D`.
4. Check the auto-restart layer:
	- pm2: `pm2 status` (bot should be **online**)
	- systemd: `systemctl status tradingbot` (should be **active (running)**)
5. Confirm the Node.js process exists:
	- `ps aux | grep node`
	- If you see a Node.js process, the bot is running at the OS level.
6. Optional: Check logs for errors:
	- pm2: `pm2 logs tradingbot`
	- systemd: `journalctl -u tradingbot -f`
