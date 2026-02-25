#!/usr/bin/env python3
"""Deploy limit-order fix + strategy selector fix + logging improvements.

Changes deployed:
1. .env.production: LIMIT_ENTRY_ENABLED=false (market orders for reliable fills)
2. .env.production: DAILY_LOSS_LIMIT_USD raised to $2000
3. strategy_selector.py: Fixed preferred_direction attribute bug + VWAP choppy exemption
4. engine.py: Upgraded gate rejection logging to INFO level
"""
import paramiko
import os
import time

VPS_IP = "167.172.207.247"
SSH_KEY = os.path.expanduser("~/.ssh/atobot_key")
PROJECT = r"c:\Users\basee\OneDrive\Desktop\AtoBot\atobot-trading"

# Files to upload (local_path_relative, remote_path)
FILES_TO_UPLOAD = [
    ("deploy/.env.production", "/opt/atobot/.env"),
    ("src/strategies/strategy_selector.py", "/opt/atobot/src/strategies/strategy_selector.py"),
    ("src/core/engine.py", "/opt/atobot/src/core/engine.py"),
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


def main():
    client = connect()
    print("Connected to VPS!")

    # Upload changed files
    print("\n=== Uploading changed files ===")
    sftp = client.open_sftp()
    for local_rel, remote_path in FILES_TO_UPLOAD:
        local_path = os.path.join(PROJECT, local_rel)
        print(f"  {local_rel} -> {remote_path}")
        sftp.put(local_path, remote_path)
    sftp.close()
    print("  All files uploaded!")

    # Show key config changes
    print("\n=== Verifying config ===")
    run(client, "grep -E 'LIMIT_ENTRY|DAILY_LOSS_LIMIT|LIMIT_OFFSET' /opt/atobot/.env")

    # Rebuild with cache
    print("\n=== Rebuilding container ===")
    run(client, "cd /opt/atobot && docker compose build 2>&1 | tail -25", timeout=600)

    # Restart
    print("\n=== Restarting bot ===")
    run(client, "cd /opt/atobot && docker compose down 2>&1")
    run(client, "cd /opt/atobot && docker compose up -d 2>&1")

    # Wait for startup
    time.sleep(15)

    # Check status
    print("\n=== Container status ===")
    run(client, "cd /opt/atobot && docker compose ps 2>&1")

    # Show recent logs
    print("\n=== Bot logs (last 40 lines) ===")
    run(client, "cd /opt/atobot && docker compose logs --tail=40 bot 2>&1")

    client.close()
    print("\n✓ Deployment complete!")
    print("\nChanges deployed:")
    print("  • LIMIT_ENTRY_ENABLED=false (market orders for reliable fills)")
    print("  • LIMIT_OFFSET_PCT=0.10 (safer fallback if re-enabled)")
    print("  • DAILY_LOSS_LIMIT_USD=2000 (was $500, too tight for $17K positions)")
    print("  • Fixed preferred_direction attribute bug in strategy_selector")
    print("  • VWAP allowed in choppy markets (mean-reversion strategy)")
    print("  • Gate rejection logging upgraded to INFO for visibility")


if __name__ == "__main__":
    main()
