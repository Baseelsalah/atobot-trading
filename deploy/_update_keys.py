"""Update VPS .env with new Alpaca keys and restart bot.

Reads all credentials from local .env — never hardcode secrets in this file.

Required in .env:
    ALPACA_API_KEY, ALPACA_API_SECRET  — new keys to push to VPS
    VPS_HOST, VPS_USER, VPS_PASS       — server SSH credentials
"""
import os
import time

import paramiko
from dotenv import load_dotenv

load_dotenv()

HOST = os.environ["VPS_HOST"]
USER = os.environ["VPS_USER"]
PASS = os.environ["VPS_PASS"]

NEW_KEY    = os.environ["ALPACA_API_KEY"]
NEW_SECRET = os.environ["ALPACA_API_SECRET"]


def run(client, cmd, timeout=60):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode()
    err = stderr.read().decode()
    if out.strip():
        print(out.strip())
    if exit_code != 0 and err.strip():
        print(f"  [ERR] {err.strip()}")
    return out, exit_code


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=15)
print("Connected to VPS\n")

# 1. Update keys
print("[1/4] Updating Alpaca API keys...")
run(client, f"sed -i 's/ALPACA_API_KEY=.*/ALPACA_API_KEY={NEW_KEY}/' /opt/atobot/.env")
run(client, f"sed -i 's/ALPACA_API_SECRET=.*/ALPACA_API_SECRET={NEW_SECRET}/' /opt/atobot/.env")

print("\n  Verifying keys in .env:")
run(client, "grep ALPACA_API /opt/atobot/.env")

# 2. Restart bot
print("\n[2/4] Restarting bot with new keys...")
run(client, "cd /opt/atobot && docker compose restart bot", timeout=90)
time.sleep(8)

# 3. Check status
print("\n[3/4] Container status:")
run(client, "cd /opt/atobot && docker compose ps")

# 4. Check logs
print("\n[4/4] Recent bot logs:")
run(client, "cd /opt/atobot && docker compose logs --tail=25 bot")

client.close()
print("\nDone!")
