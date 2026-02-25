#!/usr/bin/env python3
"""Check build status on VPS."""
import paramiko, os, sys

VPS_IP = "167.172.207.247"
SSH_KEY = os.path.expanduser("~/.ssh/atobot_key")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
key = paramiko.Ed25519Key.from_private_key_file(SSH_KEY)
client.connect(VPS_IP, username="root", pkey=key, timeout=30)

cmd = sys.argv[1] if len(sys.argv) > 1 else "tail -20 /tmp/build.log"
print(f"Running: {cmd}")
stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
print(stdout.read().decode())
err = stderr.read().decode()
if err:
    print(f"STDERR: {err}")
print(f"Exit: {stdout.channel.recv_exit_status()}")
client.close()
