#!/usr/bin/env python3
"""Deploy hardening v5 changes to VPS — upload modified files, rebuild, restart."""
import paramiko, os, time

VPS_IP = "167.172.207.247"
SSH_KEY = os.path.expanduser("~/.ssh/atobot_key")
PROJECT = r"c:\Users\basee\OneDrive\Desktop\AtoBot\atobot-trading"

# Files modified in v5 hardening
FILES_TO_UPLOAD = [
    # (local_relative_path, remote_path)
    ("src/config/settings.py",          "/opt/atobot/src/config/settings.py"),
    ("src/risk/risk_manager.py",        "/opt/atobot/src/risk/risk_manager.py"),
    ("src/strategies/strategy_selector.py", "/opt/atobot/src/strategies/strategy_selector.py"),
    ("src/scanner/regime_detector.py",  "/opt/atobot/src/scanner/regime_detector.py"),
    ("src/core/engine.py",             "/opt/atobot/src/core/engine.py"),
    ("deploy/.env.production",         "/opt/atobot/deploy/.env.production"),
]

def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key = paramiko.Ed25519Key.from_private_key_file(SSH_KEY)
    client.connect(VPS_IP, username="root", pkey=key, timeout=30)
    return client

def run(client, cmd, timeout=600):
    print(f"\n>>> {cmd[:120]}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    rc = stdout.channel.recv_exit_status()
    for line in out.strip().split("\n")[-30:]:
        if line.strip():
            print(f"  {line}")
    if err.strip():
        for line in err.strip().split("\n")[-10:]:
            if line.strip():
                print(f"  [err] {line}")
    print(f"  [exit: {rc}]")
    return out, err, rc

client = connect()
print("Connected to VPS!")

# 1. Upload all modified files
print("\n=== UPLOADING HARDENING V5 FILES ===")
sftp = client.open_sftp()
for local_rel, remote in FILES_TO_UPLOAD:
    local_path = os.path.join(PROJECT, local_rel.replace("/", os.sep))
    if not os.path.exists(local_path):
        print(f"  SKIP (missing): {local_rel}")
        continue
    # Ensure remote directory exists
    remote_dir = os.path.dirname(remote)
    try:
        sftp.stat(remote_dir)
    except FileNotFoundError:
        run(client, f"mkdir -p {remote_dir}")
    sftp.put(local_path, remote)
    print(f"  OK: {local_rel}")

# Also upload .env.production to the active location
sftp.put(
    os.path.join(PROJECT, "deploy", ".env.production"),
    "/opt/atobot/.env.production",
)
print("  OK: .env.production -> /opt/atobot/.env.production")
sftp.close()

# 2. Stop current containers
print("\n=== STOPPING CONTAINERS ===")
run(client, "cd /opt/atobot && docker compose down 2>&1")

# 3. Rebuild with cache
print("\n=== REBUILDING (with cache) ===")
run(client, "cd /opt/atobot && docker compose build 2>&1 | tail -25", timeout=600)

# 4. Start containers
print("\n=== STARTING CONTAINERS ===")
run(client, "cd /opt/atobot && docker compose up -d 2>&1")

# 5. Wait for startup
print("\n=== WAITING 15s FOR STARTUP ===")
time.sleep(15)

# 6. Verify
print("\n=== CONTAINER STATUS ===")
run(client, "cd /opt/atobot && docker compose ps 2>&1")

print("\n=== BOT LOGS (last 40 lines) ===")
run(client, "cd /opt/atobot && docker compose logs --tail=40 bot 2>&1")

# 7. Verify hardening features in config
print("\n=== VERIFY HARDENING CONFIG ===")
run(client, "grep -E 'CIRCUIT_BREAKER|GAP_FILTER|ATR_ADAPTIVE|FORCE_EOD|EOD_BLOCK' /opt/atobot/.env.production 2>&1")

client.close()
print("\n=== DEPLOYMENT COMPLETE ===")
print("Hardening v5 deployed: circuit breaker, gap filter, ATR stops, crisis sizing, zero overnight")
