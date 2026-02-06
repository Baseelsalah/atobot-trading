# 📺 VISUAL STEP-BY-STEP GUIDE
## Copy-Paste This Guide - No Thinking Required!

---

## 🎬 STEP 1: Get a Server (Choose ONE Option)

### OPTION A: DigitalOcean (Recommended for Beginners)

**What you'll do:**
1. Open your web browser
2. Go to: **www.digitalocean.com**
3. Click the blue **"Sign Up"** button (top right)
4. Enter your email and create a password
5. Click **"Get Started"**

**Create your server:**
1. Look for a green button that says **"Create"** (top right)
2. Click it
3. Click **"Droplets"**
4. You'll see a bunch of options. Here's what to click:

**Choose an image:**
- Click **"Ubuntu"**
- Click **"22.04 (LTS) x64"**

**Choose a plan:**
- Click **"Basic"** (should already be selected)
- Scroll down and click the **$6/mo** box
- It says "1 GB / 1 CPU, 25 GB SSD" - that's perfect!

**Choose a datacenter:**
- Click one close to where you live
- If in USA: **"New York"** or **"San Francisco"**
- If in Europe: **"London"** or **"Amsterdam"**

**Authentication:**
- Click **"Password"** (not SSH)
- Create a strong password
- **WRITE THIS PASSWORD DOWN ON PAPER!** ✍️

**Finalize:**
- Hostname: Type **"atobot"** (so you remember what it is)
- Leave everything else as default
- Click the big green **"Create Droplet"** button at bottom

**Wait 60 seconds...**

**Success! You'll see your droplet!**
- You'll see a number like **123.45.67.89** - this is your IP address
- **WRITE DOWN YOUR IP ADDRESS:** ________________

---

### OPTION B: "I Want to Use My Own Computer at Home"

**Requirements:**
- Computer must be always on (or you set to never sleep)
- Must have Ubuntu Linux or be able to install it
- Must be connected to internet

**Get your IP address:**
1. Open Terminal
2. Type: `ip addr show`
3. Look for a number like 192.168.1.X
4. **WRITE IT DOWN:** ________________

---

## 🔌 STEP 2: Connect to Your Server

### On WINDOWS:

**Download PuTTY:**
1. Google: **"putty download"**
2. Click the first link (putty.org)
3. Download **"putty.exe"** (64-bit)
4. Open the downloaded file

**Connect:**
1. You'll see a window with a box labeled **"Host Name"**
2. Type your IP address: **123.45.67.89** (use YOUR IP!)
3. Make sure Port says **22**
4. Click **"Open"** at the bottom

**A black window appears! Don't panic!**
1. It asks **"login as:"**
2. Type: **root**
3. Press **Enter**
4. It asks for **"password:"**
5. Type your password (YOU WON'T SEE IT TYPING - that's normal for security!)
6. Press **Enter**

**SUCCESS! You should see:**
```
Welcome to Ubuntu 22.04...
root@atobot:~#
```

---

### On MAC or LINUX:

**Open Terminal:**
1. Press **Cmd + Space** (Mac) or **Ctrl + Alt + T** (Linux)
2. Type: **terminal**
3. Press **Enter**

**Connect:**
1. In the terminal, type this (replace with YOUR IP):
   ```
   ssh root@123.45.67.89
   ```
2. Press **Enter**
3. It asks about **"fingerprint"** - Type: **yes**
4. Press **Enter**
5. Type your **password**
6. Press **Enter**

**SUCCESS! You see:**
```
root@atobot:~#
```

---

## 📦 STEP 3: Get Your Bot to the Server

### METHOD 1: Using FileZilla (EASIEST - RECOMMENDED!)

**Download FileZilla:**
1. Google: **"filezilla download"**
2. Download **FileZilla Client** (NOT Server)
3. Install it (click Next, Next, Finish)
4. Open FileZilla

**You see a window with two sides:**
- Left = Your Computer
- Right = Your Server (empty for now)

**Connect to your server:**
1. Look at the TOP of FileZilla
2. You'll see boxes labeled: **Host, Username, Password, Port**

**Fill them in:**
- **Host:** Type `sftp://123.45.67.89` (use YOUR IP!)
- **Username:** Type `root`
- **Password:** Type your password
- **Port:** Type `22`

3. Click **"Quickconnect"**

**You're connected! The right side now shows your server's files!**

**Upload your bot:**
1. On the **LEFT side** (your computer), find your **atobot-trading** folder
   - It's probably in Downloads or Documents
2. **RIGHT-CLICK** on the **atobot-trading** folder
3. Click **"Upload"**
4. Wait 2-5 minutes (you'll see a progress bar at bottom)

**SUCCESS! The folder is now on the right side (your server)!**

**Close FileZilla - you're done with it!**

---

### METHOD 2: Using GitHub (If You Know Git)

**On your computer:**
1. Create GitHub account at github.com
2. Create new repository: "atobot-trading"
3. Push your code (GitHub shows you how)

**On your server (in PuTTY/Terminal):**
```
git clone https://github.com/YOURUSERNAME/atobot-trading.git
```

---

## ⚙️ STEP 4: Install Everything (ONE COMMAND!)

**In your server terminal (PuTTY or Mac Terminal), copy and paste these EXACTLY:**

**1. Go into the bot folder:**
```bash
cd atobot-trading
```
Press **Enter**

*(cd means "change directory" - like double-clicking a folder)*

**2. Run the installer:**
```bash
bash scripts/deploy.sh
```
Press **Enter**

**What happens now:**
- It says "Continue? (yes/no)"
- Type: **yes**
- Press **Enter**

**Now sit back and watch! It will:**
- ✅ Install Node.js (30 seconds)
- ✅ Install PM2 (20 seconds)
- ✅ Install dependencies (60 seconds)
- ✅ Build your bot (30 seconds)
- ⏸️  **PAUSE and ask you to edit .env**

**Don't close anything! We'll do the .env part next!**

---

## 🔑 STEP 5: Add Your Trading Keys

**The installer just said: "Please edit .env and add your API keys"**

**Here's what to do:**

**1. Open the .env file for editing:**
```bash
nano .env
```
Press **Enter**

**You now see a text file! You can edit it!**

**2. Find the Alpaca keys:**
Use arrow keys to scroll down until you see:
```
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
```

**3. Get your Alpaca keys:**
- Open a new browser tab
- Go to **alpaca.markets**
- Log in
- Click your name (top right) → **"Paper Trading"** or **"API Keys"**
- You'll see your **API Key** and **Secret Key**

**4. Replace the text:**
- In the terminal (nano), use arrows to move to `your_key_here`
- **Delete that text**
- **Type (or paste) your ACTUAL API key**
- Do the same for the secret key

**5. IMPORTANT - Set to Practice Mode:**
Scroll down and find:
```
DRY_RUN=0
```

**Change it to:**
```
DRY_RUN=1
```

*(This means PRACTICE MODE - no real money!)*

**6. Save and exit:**
- Press **Ctrl + X** (bottom of screen shows this)
- Press **Y** (for "yes, save")
- Press **Enter**

**You're back at the command prompt!**

---

## ✅ STEP 6: Finish Installation

**The installer is waiting! Let's continue:**

```bash
bash scripts/deploy.sh
```
Press **Enter**

**It will:**
- ✅ Install cron jobs (automatic scheduling)
- ✅ Start your bot with PM2
- ✅ Verify everything works
- ✅ Show you "DEPLOYMENT COMPLETE!"

**Wait 1-2 minutes... done!**

---

## 🎉 STEP 7: Verify It's Working!

**Type these commands to check (one at a time):**

**1. Check if bot is running:**
```bash
npm run pm2:status
```

**You should see:**
```
atobot     │ online    │ 0    │ ...
```

**"online" = IT'S WORKING! ✅**

---

**2. Check health:**
```bash
curl http://localhost:5000/health
```

**You'll see a bunch of text. Look for:**
```
"status":"ok"
```

**If you see that = IT'S HEALTHY! ✅**

---

**3. Watch it think (optional but cool!):**
```bash
npm run pm2:logs
```

**You'll see LIVE logs scrolling! Things like:**
```
[ANALYSIS] Tick @ 10:05 ET
[TIME GUARD] Entry window active
MARKET_CLOCK is_open=true
```

**That's your bot analyzing the market in real-time! 🤖**

**Press Ctrl + C to stop watching**

---

## 🌐 STEP 8: Check the Web Dashboard

**Open your web browser and go to:**
```
http://YOUR-SERVER-IP:5000
```

*(Replace YOUR-SERVER-IP with your actual IP, like: http://123.45.67.89:5000)*

**You should see a dashboard with:**
- Your bot's status
- Trade history
- Performance charts

---

## 🎊 YOU DID IT!

**Your bot is now:**
- ✅ Running 24/7 on your server
- ✅ In practice mode (no real money!)
- ✅ Will automatically:
  - Wake up at 8:30 AM ET
  - Trade during market hours
  - Close everything at 3:45 PM
  - Go to sleep overnight
  - Monitor itself and auto-restart if it crashes

**You don't have to do ANYTHING else!**

---

## 📱 How to Check on It Anytime

**From your computer:**

1. **Open PuTTY (Windows) or Terminal (Mac)**
2. **Connect to your server** (same as before)
3. **Type:**
   ```bash
   cd atobot-trading
   npm run pm2:status
   ```
4. **See your bot status!**

**Or use the web dashboard:**
- Just open: `http://YOUR-IP:5000`

---

## 🆘 "HELP! Something's Wrong!"

### "It says 'command not found'"
**→ Make sure you're in the right folder:**
```bash
cd atobot-trading
```

### "The bot shows 'stopped' not 'online'"
**→ Restart it:**
```bash
npm run pm2:restart
```
Wait 10 seconds, check again:
```bash
npm run pm2:status
```

### "I can't connect to my server"
**→ Check:**
- Is your IP address correct?
- Is your password correct?
- Did you type `sftp://` before IP in FileZilla?

### "FileZilla upload failed"
**→ Check:**
- Port is `22`
- Protocol is `SFTP` (not FTP)
- Username is `root`

### "I'm completely stuck!"
**→ Tell me:**
1. What step number you're on
2. Exactly what you see on screen (or screenshot)
3. Any error messages

**I'll help you through it!** 🤝

---

## 📝 WRITE THIS DOWN (Important Info!)

```
MY SERVER DETAILS:
━━━━━━━━━━━━━━━━━
IP Address: ___________________
Username: root
Password: ___________________
Provider: ___________________

ACCESS DASHBOARD:
━━━━━━━━━━━━━━━━━
http://___________________:5000

ALPACA ACCOUNT:
━━━━━━━━━━━━━━━━━
Mode: Paper Trading (Practice)
API Keys: Saved in .env file
```

---

## 🎯 WHAT HAPPENS NEXT?

**Today:**
- Your bot is running in practice mode
- It's learning the market
- Making fake trades
- No real money involved

**This Week:**
- Check the dashboard once a day (optional!)
- Watch it make trades
- See the profit/loss

**Sunday (This Week):**
- Bot automatically creates weekly performance report
- Shows you how well it did

**After 2-4 Weeks:**
- Review all the practice trades
- See if it's profitable
- Decide if you want to try real money

**Important:**
- Keep it in **practice mode** (DRY_RUN=1) for at least 2-4 weeks!
- Don't switch to real money until you're comfortable
- The bot will keep running 24/7 automatically

---

## ✅ YOU'RE ALL SET!

**You successfully deployed a fully automated trading bot!**

**That's actually really impressive for someone new to coding!** 🎉

Your bot will now:
- Wake up every morning
- Trade during market hours(practice mode)
- Close positions safely
- Monitor itself
- Generate reports

**You just check in when you want to see how it's doing!**

---

**Any questions? Just ask! No question is too basic!** 😊
