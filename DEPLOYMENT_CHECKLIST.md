# ✅ DEPLOYMENT CHECKLIST - Print This Out!

Check off each step as you complete it!

---

## BEFORE YOU START

- [ ] I have my Alpaca API keys ready (from alpaca.markets)
- [ ] I know what a "server" is (read BEGINNER_DEPLOYMENT.md if not!)
- [ ] I have 30 minutes of uninterrupted time

---

## STEP 1: GET A SERVER (5-10 minutes)

**Choose ONE option:**

### Option A: DigitalOcean (Easiest for beginners)
- [ ] Go to DigitalOcean.com
- [ ] Sign up for account
- [ ] Click "Create" → "Droplets"
- [ ] Select Ubuntu 22.04
- [ ] Select $6/month plan
- [ ] Create a password (WRITE IT DOWN!)
- [ ] Click "Create Droplet"
- [ ] Wait 1 minute
- [ ] **WRITE DOWN:** IP Address: ________________
- [ ] **WRITE DOWN:** Password: ________________

### Option B: AWS EC2
- [ ] Go to AWS.amazon.com
- [ ] Create free account
- [ ] Launch EC2 instance (Ubuntu)
- [ ] Download .pem key file
- [ ] **WRITE DOWN:** IP Address: ________________

### Option C: Use Your Home Computer
- [ ] Make sure it has Ubuntu or can run Linux
- [ ] **WRITE DOWN:** IP Address: ________________

---

## STEP 2: CONNECT TO YOUR SERVER (2-5 minutes)

**On Windows:**
- [ ] Download PuTTY
- [ ] Open PuTTY
- [ ] Enter your IP address
- [ ] Click "Open"
- [ ] Login as: `root`
- [ ] Enter your password
- [ ] **SUCCESS!** You see a command prompt

**On Mac/Linux:**
- [ ] Open Terminal
- [ ] Type: `ssh root@YOUR-IP-ADDRESS`
- [ ] Type "yes" to fingerprint question
- [ ] Enter password
- [ ] **SUCCESS!** You see a command prompt

---

## STEP 3: GET YOUR BOT TO THE SERVER (5-15 minutes)

**Choose ONE method:**

### Method 1: GitHub (If you have Git)
- [ ] Created GitHub account
- [ ] Created repository named "atobot-trading"
- [ ] Uploaded your code to GitHub
- [ ] On server, typed: `git clone YOUR-REPO-URL`
- [ ] Typed: `cd atobot-trading`
- [ ] **SUCCESS!** Folder is on server

### Method 2: FileZilla (Easiest!)
- [ ] Downloaded FileZilla
- [ ] Connected to your server (IP, root, password)
- [ ] Dragged atobot-trading folder to server
- [ ] Waited for upload to finish
- [ ] On server, typed: `cd atobot-trading`
- [ ] **SUCCESS!** Folder is on server

### Method 3: ZIP Download
- [ ] Asked AI to create ZIP file
- [ ] Downloaded ZIP
- [ ] Uploaded ZIP to server via FileZilla
- [ ] On server, typed: `unzip atobot-trading.zip`
- [ ] Typed: `cd atobot-trading`
- [ ] **SUCCESS!** Folder is on server

---

## STEP 4: RUN THE INSTALLER (3-5 minutes)

- [ ] On server, typed: `bash scripts/deploy.sh`
- [ ] Typed "yes" when asked to continue
- [ ] Waited while it installed Node.js
- [ ] Waited while it installed PM2
- [ ] Waited while it installed dependencies
- [ ] Waited while it built the bot
- [ ] **PAUSED** at "edit .env" message

---

## STEP 5: ADD YOUR TRADING KEYS (2 minutes)

- [ ] On server, typed: `nano .env`
- [ ] Found line: `ALPACA_API_KEY=`
- [ ] Pasted my Alpaca API Key after the `=`
- [ ] Found line: `ALPACA_API_SECRET=`
- [ ] Pasted my Alpaca Secret after the `=`
- [ ] Found line: `DRY_RUN=0`
- [ ] Changed it to: `DRY_RUN=1` (IMPORTANT!)
- [ ] Pressed Ctrl+X
- [ ] Pressed Y
- [ ] Pressed Enter
- [ ] **SUCCESS!** File saved

---

## STEP 6: CONTINUE INSTALLATION (1-2 minutes)

- [ ] Pressed Enter to continue
- [ ] Waited while cron jobs installed
- [ ] Waited while bot started
- [ ] Saw "DEPLOYMENT COMPLETE!" message
- [ ] **SUCCESS!** Installation finished

---

## STEP 7: VERIFY IT'S WORKING (2 minutes)

- [ ] Typed: `npm run pm2:status`
- [ ] Saw `atobot | online` (SUCCESS!)
- [ ] Typed: `curl http://localhost:5000/health`
- [ ] Saw `"status":"ok"` somewhere (SUCCESS!)
- [ ] Typed: `npm run pm2:logs`
- [ ] Saw logs scrolling (CTRL+C to stop)
- [ ] **SUCCESS!** Bot is running!

---

## STEP 8: CHECK AUTOMATION (1 minute)

- [ ] Typed: `crontab -l`
- [ ] Saw 6 lines with "atobot" (SUCCESS!)
- [ ] **SUCCESS!** Automation installed

---

## 🎉 YOU'RE DONE!

### Your bot is now:
- ✅ Running 24/7on your server
- ✅ In practice mode (DRY_RUN=1 - no real money!)
- ✅ Will wake up at 8:30 AM ET every day
- ✅ Will trade automatically
- ✅ Will monitor itself
- ✅ Will close positions at 3:45 PM ET

### What to do now:
- [ ] Bookmark your server IP
- [ ] Save your passwords securely
- [ ] Check the dashboard: http://YOUR-SERVER-IP:5000
- [ ] Read BEGINNER_DEPLOYMENT.md for more info

---

## 📱 DAILY MAINTENANCE (Optional!)

**You don't HAVE to do this, but you CAN if you want:**

### Morning (Optional):
- [ ] SSH to server
- [ ] Type: `npm run pm2:logs --lines 50`
- [ ] See what trades it made

### Anytime:
- [ ] Open browser: http://YOUR-SERVER-IP:5000
- [ ] View dashboard
- [ ] See performance

### Weekly (Recommended):
- [ ] SSH to server
- [ ] Type: `cat weekly_reports/weekly_*.txt | tail -50`
- [ ] Review performance

---

## 🆘 TROUBLESHOOTING

If something's not working, check the boxes that apply:

**Bot not running?**
- [ ] Typed: `npm run pm2:restart`
- [ ] Waited 10 seconds
- [ ] Typed: `npm run pm2:status` again

**Can't connect to server?**
- [ ] Checked IP address is correct
- [ ] Checked password is correct
- [ ] Tried again

**FileZilla won't connect?**
- [ ] Used `sftp://YOUR-IP` (not just the IP)
- [ ] Port is set to 22
- [ ] Username is `root`

**Still stuck?**
- [ ] Took a screenshot
- [ ] Noted exactly what step I'm on
- [ ] Asked for help with specific details

---

## 💾 SAVE THIS INFO

**Server Details:**
- IP Address: ____________________
- Username: root
- Password: ____________________
- Server Provider: ____________________

**Alpaca Details:**
- API Key: (saved in .env file)
- API Secret: (saved in .env file)
- Mode: Paper Trading (DRY_RUN=1)

**Important URLs:**
- Dashboard: http://____________________:5000
- DigitalOcean: https://cloud.digitalocean.com

---

## 📞 NEED HELP?

**Tell me:**
1. Which step number you're on
2. What you see on your screen
3. What error message (if any)
4. Screenshot (if possible)

**I'll help you through it!** 🤖

---

**Print this out and check boxes as you go!** ✅
