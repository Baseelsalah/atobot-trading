"""Update VPS .env with new Alpaca keys and restart bot."""
import paramiko
import time

HOST = "165.232.55.24"
USER = "root"
PASS = "eddy587SFbs"

NEW_KEY = "PKSI4TAWFCMN2GZL7ELQNQFLMR"
NEW_SECRET = "BH1oF94L5tUfodCjqsJnzwdh7LvFvECqPY7SDsMpVL5V"


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

# Verify
print("\n  Verifying keys in .env:")
run(client, "grep ALPACA_API /opt/atobot/.env")

# 2. Restart bot
print("\n[2/4] Restarting bot with new keys...")
run(client, "cd /opt/atobot && docker compose restart bot", timeout=60)

time.sleep(8)

# 3. Check status
print("\n[3/4] Container status:")
run(client, "cd /opt/atobot && docker compose ps")

# 4. Check logs
print("\n[4/4] Recent bot logs:")
run(client, "cd /opt/atobot && docker compose logs --tail=25 bot")

client.close()
print("\nDone!")
