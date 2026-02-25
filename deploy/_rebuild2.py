#!/usr/bin/env python3
"""Kill old build, rebuild WITH cache, start bot."""
import paramiko, os, time

VPS_IP = "167.172.207.247"
SSH_KEY = os.path.expanduser("~/.ssh/atobot_key")
PROJECT = r"c:\Users\basee\OneDrive\Desktop\AtoBot\atobot-trading"

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
print("Connected!")

# Upload fixed Dockerfile
print("\n=== Upload fixed Dockerfile ===")
sftp = client.open_sftp()
sftp.put(os.path.join(PROJECT, "Dockerfile"), "/opt/atobot/Dockerfile")
print("  Done")
sftp.close()

# Kill old build processes
print("\n=== Kill old build ===")
run(client, "pkill -f 'docker.*build' 2>/dev/null; sleep 2; echo killed")

# Rebuild WITH cache (much faster)
print("\n=== Rebuild with cache ===")
run(client, "cd /opt/atobot && docker compose build 2>&1 | tail -25", timeout=600)

# Start
print("\n=== Start containers ===")
run(client, "cd /opt/atobot && docker compose up -d 2>&1")

time.sleep(12)

print("\n=== Status ===")
run(client, "cd /opt/atobot && docker compose ps 2>&1")

print("\n=== Bot logs ===")
run(client, "cd /opt/atobot && docker compose logs --tail=30 bot 2>&1")

client.close()
print("\nDone!")
