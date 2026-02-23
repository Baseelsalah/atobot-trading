#!/usr/bin/env python3
"""Upload guardian files + updated configs, rebuild, start guardian container."""
import paramiko, os, time, stat

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

def sftp_mkdir_p(sftp, remote_dir):
    """Create remote directory tree if it doesn't exist."""
    dirs_to_create = []
    current = remote_dir
    while current and current != "/":
        try:
            sftp.stat(current)
            break
        except FileNotFoundError:
            dirs_to_create.append(current)
            current = os.path.dirname(current)
    for d in reversed(dirs_to_create):
        try:
            sftp.mkdir(d)
            print(f"  mkdir {d}")
        except Exception:
            pass

client = connect()
print("Connected!")

sftp = client.open_sftp()

# 1. Upload guardian module files
print("\n=== Upload Guardian Files ===")
guardian_dir = os.path.join(PROJECT, "src", "guardian")
remote_guardian = "/opt/atobot/src/guardian"
sftp_mkdir_p(sftp, remote_guardian)

guardian_files = [
    "__init__.py",
    "__main__.py",
    "health_monitor.py",
    "self_healer.py",
    "performance_analyzer.py",
    "auto_tuner.py",
    "agent.py",
]
for f in guardian_files:
    local_path = os.path.join(guardian_dir, f)
    remote_path = f"{remote_guardian}/{f}"
    sftp.put(local_path, remote_path)
    print(f"  Uploaded {f}")

# 2. Upload updated docker-compose.yml, Dockerfile, and requirements.txt
print("\n=== Upload Docker Configs ===")
for f in ["docker-compose.yml", "Dockerfile", "requirements.txt"]:
    sftp.put(os.path.join(PROJECT, f), f"/opt/atobot/{f}")
    print(f"  Uploaded {f}")

sftp.close()

# 3. Kill any old build processes
print("\n=== Kill old builds ===")
run(client, "pkill -f 'docker.*build' 2>/dev/null; sleep 2; echo done")

# 4. Rebuild with cache (only new guardian layer should rebuild)
print("\n=== Rebuild Docker image (with cache) ===")
out, err, rc = run(client, "cd /opt/atobot && docker compose build 2>&1 | tail -30", timeout=600)
if rc != 0:
    print("\n!!! Build failed — trying without cache !!!")
    run(client, "cd /opt/atobot && docker compose build --no-cache 2>&1 | tail -30", timeout=900)

# 5. Start all containers (including guardian)
print("\n=== Start all containers ===")
run(client, "cd /opt/atobot && docker compose up -d 2>&1")

time.sleep(15)

# 6. Check status
print("\n=== Container Status ===")
run(client, "cd /opt/atobot && docker compose ps 2>&1")

# 7. Check guardian logs
print("\n=== Guardian logs ===")
run(client, "cd /opt/atobot && docker compose logs --tail=30 guardian 2>&1")

# 8. Check bot still healthy
print("\n=== Bot logs (last 10) ===")
run(client, "cd /opt/atobot && docker compose logs --tail=10 bot 2>&1")

client.close()
print("\nDone! Guardian deployed.")
