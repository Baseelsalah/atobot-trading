#!/usr/bin/env python3
"""Provision a fresh DigitalOcean droplet and deploy AtoBot."""
import paramiko, time, os, stat, glob

VPS_IP = "167.172.207.247"
SSH_KEY = os.path.expanduser("~/.ssh/atobot_key")
PROJECT = r"c:\Users\basee\OneDrive\Desktop\AtoBot\atobot-trading"

def ssh_connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key = paramiko.Ed25519Key.from_private_key_file(SSH_KEY)
    client.connect(VPS_IP, username="root", pkey=key, timeout=30)
    return client

def run(client, cmd, timeout=300):
    print(f"\n>>> {cmd[:120]}...")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    rc = stdout.channel.recv_exit_status()
    if out.strip():
        # Print last 30 lines
        lines = out.strip().split("\n")
        for l in lines[-30:]:
            print(f"  {l}")
    if err.strip():
        for l in err.strip().split("\n")[-10:]:
            print(f"  [err] {l}")
    if rc != 0:
        print(f"  [exit code: {rc}]")
    return out, err, rc

def wait_for_apt(client, max_wait=180):
    """Wait until apt locks are free."""
    print("\n>>> Waiting for apt locks to clear...")
    start = time.time()
    while time.time() - start < max_wait:
        _, _, rc = run(client, "fuser /var/lib/apt/lists/lock /var/lib/dpkg/lock /var/lib/dpkg/lock-frontend 2>/dev/null; echo done")
        # fuser exits 0 if processes found, 1 if none
        out2, _, rc2 = run(client, "fuser /var/lib/dpkg/lock 2>/dev/null; echo $?")
        lines = out2.strip().split("\n")
        last = lines[-1].strip()
        if last == "1":
            print("  Apt locks are free!")
            return
        print(f"  Still locked, waiting 10s... ({int(time.time()-start)}s elapsed)")
        time.sleep(10)
    print("  WARNING: Timed out waiting for apt locks, trying anyway...")

def provision(client):
    """Install Docker, firewall, swap, etc."""
    wait_for_apt(client)
    
    print("\n=== STEP 1: Update & install packages ===")
    run(client, "export DEBIAN_FRONTEND=noninteractive && apt-get update -qq", timeout=120)
    run(client, "export DEBIAN_FRONTEND=noninteractive && apt-get upgrade -y -qq", timeout=300)
    run(client, "export DEBIAN_FRONTEND=noninteractive && apt-get install -y -qq docker.io docker-compose-v2 ufw fail2ban curl", timeout=300)
    
    print("\n=== STEP 2: Enable Docker ===")
    run(client, "systemctl enable docker && systemctl start docker")
    run(client, "docker --version && docker compose version")
    
    print("\n=== STEP 3: Create swap (1GB) ===")
    run(client, """
        if [ ! -f /swapfile ]; then
            fallocate -l 1G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
            echo '/swapfile none swap sw 0 0' >> /etc/fstab
            echo 'Swap created'
        else
            echo 'Swap already exists'
        fi
    """)
    
    print("\n=== STEP 4: Firewall ===")
    run(client, "ufw allow OpenSSH && ufw allow 8501/tcp && yes | ufw enable 2>/dev/null; ufw status")
    
    print("\n=== STEP 5: Create app directory ===")
    run(client, "mkdir -p /opt/atobot/src /opt/atobot/deploy /opt/atobot/data /opt/atobot/logs")
    
    print("\n=== STEP 6: Create atobot user ===")
    run(client, """
        id atobot 2>/dev/null || useradd -r -s /bin/false atobot
        usermod -aG docker atobot 2>/dev/null
        echo 'atobot user ready'
    """)

def upload_files(client):
    """Upload project files via SFTP."""
    print("\n=== UPLOADING FILES ===")
    sftp = client.open_sftp()
    
    SKIP = {".venv", "__pycache__", ".git", "node_modules", ".mypy_cache", 
            ".pytest_cache", "deploy", ".github", "atobot.db"}
    
    def ensure_remote_dir(remote_path):
        """Create remote directory if it doesn't exist."""
        try:
            sftp.stat(remote_path)
        except FileNotFoundError:
            ensure_remote_dir(os.path.dirname(remote_path))
            sftp.mkdir(remote_path)
    
    uploaded = 0
    base = PROJECT
    remote_base = "/opt/atobot"
    
    for root, dirs, files in os.walk(base):
        # Filter out skipped directories
        dirs[:] = [d for d in dirs if d not in SKIP]
        
        rel_root = os.path.relpath(root, base).replace("\\", "/")
        if rel_root == ".":
            rel_root = ""
        
        for f in files:
            # Skip binary/irrelevant files
            if f.endswith((".pyc", ".pyo", ".db", ".db-journal")):
                continue
            if f.startswith(".") and f not in (".env", ".dockerignore"):
                continue
                
            local_path = os.path.join(root, f)
            if rel_root:
                remote_path = f"{remote_base}/{rel_root}/{f}"
            else:
                remote_path = f"{remote_base}/{f}"
            
            # Ensure directory exists
            remote_dir = os.path.dirname(remote_path)
            ensure_remote_dir(remote_dir)
            
            try:
                sftp.put(local_path, remote_path)
                uploaded += 1
                print(f"  [{uploaded}] {rel_root}/{f}" if rel_root else f"  [{uploaded}] {f}")
            except Exception as e:
                print(f"  SKIP {remote_path}: {e}")
    
    sftp.close()
    print(f"\n  Total files uploaded: {uploaded}")

def setup_systemd(client):
    """Create systemd service for auto-start."""
    print("\n=== SYSTEMD SERVICE ===")
    run(client, r"""cat > /etc/systemd/system/atobot.service << 'EOF'
[Unit]
Description=AtoBot Trading Bot
After=docker.service
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=/opt/atobot
ExecStart=/usr/bin/docker compose up
ExecStop=/usr/bin/docker compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable atobot
echo 'systemd service created and enabled'
""")

def build_and_start(client):
    """Build Docker image and start the bot."""
    print("\n=== BUILDING DOCKER IMAGE ===")
    run(client, "cd /opt/atobot && docker compose build --no-cache 2>&1 | tail -20", timeout=600)
    
    print("\n=== STARTING BOT ===")
    run(client, "cd /opt/atobot && docker compose up -d 2>&1")
    
    # Wait for containers to start
    time.sleep(10)
    
    print("\n=== CONTAINER STATUS ===")
    run(client, "cd /opt/atobot && docker compose ps")
    
    print("\n=== BOT LOGS (last 30 lines) ===")
    run(client, "cd /opt/atobot && docker compose logs --tail=30 bot 2>&1")

def main():
    import sys
    print(f"Connecting to {VPS_IP}...")
    client = ssh_connect()
    print("Connected!\n")
    
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    try:
        if step == "all":
            provision(client)
            upload_files(client)
            setup_systemd(client)
            build_and_start(client)
        elif step == "upload":
            upload_files(client)
        elif step == "systemd":
            setup_systemd(client)
        elif step == "build":
            build_and_start(client)
        elif step == "provision":
            provision(client)
        
        print("\n" + "="*60)
        print(f"STEP '{step}' COMPLETE!")
        print(f"  VPS: {VPS_IP}")
        print(f"  Dashboard: http://{VPS_IP}:8501")
        print("="*60)
    finally:
        client.close()

if __name__ == "__main__":
    main()
