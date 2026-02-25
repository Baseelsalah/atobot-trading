#!/usr/bin/env python3
"""Quick restart: stop and start bot after files were already uploaded."""
import paramiko, os, time

VPS_IP = "167.172.207.247"
SSH_KEY = os.path.expanduser("~/.ssh/atobot_key")

def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key = paramiko.Ed25519Key.from_private_key_file(SSH_KEY)
    client.connect(VPS_IP, username="root", pkey=key, timeout=30)
    return client

def run(client, cmd, timeout=120):
    print(f"\n>>> {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    rc = stdout.channel.recv_exit_status()
    for line in out.strip().split("\n")[-40:]:
        if line.strip():
            print(f"  {line}")
    if err.strip():
        for line in err.strip().split("\n")[-10:]:
            if line.strip():
                print(f"  [err] {line}")
    print(f"  [exit: {rc}]")
    return out, err, rc

client = connect()
print("Connected!")

# Check current state
print("\n=== Current container state ===")
run(client, "cd /opt/atobot && docker compose ps 2>&1")

# Full rebuild + restart (source is baked into Docker image via COPY)
print("\n=== Rebuilding image with uploaded fixes ===")
run(client, "cd /opt/atobot && docker compose build 2>&1 | tail -15", timeout=600)

print("\n=== Stopping containers ===")
run(client, "cd /opt/atobot && docker compose down 2>&1", timeout=120)

print("\n=== Starting containers ===")
run(client, "cd /opt/atobot && docker compose up -d 2>&1", timeout=120)

time.sleep(12)

# Check status
print("\n=== Container status ===")
run(client, "cd /opt/atobot && docker compose ps 2>&1")

# Verify config was loaded
print("\n=== Verify config in container ===")
run(client, "docker exec atobot grep -E 'LIMIT_ENTRY|DAILY_LOSS' /app/.env 2>/dev/null || echo 'checking env...'")
run(client, "docker exec atobot env 2>/dev/null | grep -E 'LIMIT_ENTRY|DAILY_LOSS' || echo 'env vars not found directly'")

# Show recent logs
print("\n=== Bot logs (last 40 lines) ===")
run(client, "cd /opt/atobot && docker compose logs --tail=40 bot 2>&1")

client.close()
print("\n✓ Done!")
