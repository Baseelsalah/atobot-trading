# 🚀 QUICK DEPLOYMENT CHEAT SHEET

## You Need:
- [ ] A server (VPS, AWS EC2, DigitalOcean, local server, etc.)
- [ ] SSH access to that server
- [ ] Your Alpaca API keys
- [ ] (Optional) OpenAI API key

---

## Super Simple Deployment (2 Steps)

### Step 1: Get code to your server

**If you pushed to GitHub:**
```bash
ssh user@your-server
git clone https://github.com/yourusername/atobot-trading
cd atobot-trading
```

**If uploading from local machine:**
```bash
# Option A: SCP (from your computer)
scp -r /path/to/atobot-trading user@your-server:/home/user/

# Option B: Use FileZilla/WinSCP/Cyberduck
# Just drag-and-drop the entire atobot-trading folder
```

### Step 2: Run deployment script

```bash
ssh user@your-server
cd atobot-trading
bash scripts/deploy.sh
```

**Answer "yes" when prompted. Script will:**
- Install everything needed
- Set up your .env file (you'll add API keys)
- Install cron jobs
- Start the bot
- Verify it works

**DONE!** 🎉

---

## What The Script Does

1. ✅ Installs Node.js 20.x (if not installed)
2. ✅ Installs PM2 (process manager)
3. ✅ Installs dependencies (`npm install`)
4. ✅ Builds production bundle
5. ✅ Creates .env from .env.example
6. ✅ Prompts you to add API keys
7. ✅ Installs 6 cron jobs (automation)
8. ✅ Starts bot with PM2
9. ✅ Verifies everything works

---

## After Deployment

### Verify it's working:

```bash
# Check bot status
npm run pm2:status
# Should show "online"

# Check health
curl http://localhost:5000/health | jq
# Should return JSON with "status": "ok"

# View live logs
npm run pm2:logs --follow
# Should see analysis cycles every 5 minutes

# Check cron jobs
crontab -l
# Should show 6 automation jobs
```

### Your bot now runs automatically:

- **8:30 AM ET** - Wakes up, validates, starts trading
- **Every 30 min** - Health check, auto-restart if needed
- **4:15 PM ET** - Closes positions, switches to safe mode
- **Sunday 6 PM** - Weekly performance report
- **11:55 PM** - Daily backup
- **1st of month** - Strategy analysis & auto-tuning

---

## Common Server Types

### AWS EC2
```bash
ssh -i "your-key.pem" ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com
```

### DigitalOcean Droplet
```bash
ssh root@your-droplet-ip
```

### Linode
```bash
ssh root@your-linode-ip
```

### Vultr
```bash
ssh root@your-vultr-ip
```

### Home Server / Local VPS
```bash
ssh yourusername@192.168.1.xxx
```

---

## If Something Goes Wrong

### Bot won't start:
```bash
npm run pm2:logs --err
# Check error logs
```

### Cron jobs not installing:
```bash
# Install manually
crontab atobot-crontab.txt

# Verify
crontab -l
```

### Health endpoint fails:
```bash
# Check if bot is running
npm run pm2:status

# Check port
netstat -tulpn | grep 5000

# Restart bot
npm run pm2:restart
```

### PM2 command not found:
```bash
sudo npm install -g pm2
which pm2
```

---

## Need Help?

1. **Read full guide:** `DEPLOYMENT_GUIDE.md`
2. **Check logs:** `npm run pm2:logs`
3. **View automation log:** `cat reports/automation.jsonl`
4. **Read docs:** `AUTOMATION_COMPLETE.md`

---

## Emergency Commands

```bash
# Stop bot
npm run pm2:stop

# Delete from PM2
npm run pm2:delete

# Remove cron jobs
crontab -r

# Fresh start
npm run pm2:delete
npm run build
npm run pm2:start
```

---

## ⚡ Complete Command Reference

```bash
# One-time deployment
bash scripts/deploy.sh

# Check status
npm run pm2:status
curl http://localhost:5000/health | jq
crontab -l

# View logs
npm run pm2:logs --follow
tail -f /tmp/atobot-premarket.log
cat reports/automation.jsonl | tail -20

# Control bot
npm run pm2:start
npm run pm2:stop
npm run pm2:restart

# Mode switching
npm run mode:status
npm run mode:test
npm run mode:live

# Testing
npm run smoke-test
npm run daily:monitor
```

---

## 🎉 That's It!

**Two steps:**
1. Get code to server
2. Run `bash scripts/deploy.sh`

**Then forget about it!** Your bot runs 100% automatically. 🤖📈
