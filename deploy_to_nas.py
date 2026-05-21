#!/usr/bin/env python3
"""
Deploy hybrid_simple.py to Synology NAS via SSH + base64-pipe.
Adds daily Cron job: 22:00 UTC (every day)

Setup:
  - Read hybrid_simple.py locally
  - base64 encode
  - SSH to NAS, pipe to file
  - Add to /etc/crontab
  - Restart crond
"""
import paramiko
import base64
from pathlib import Path

# Credentials
HOST = "192.168.178.32"
PORT = 22
USER = "benutzername"
PW = "f/hGT5%4$fh"

# Paths
LOCAL_SCRIPT = Path("hybrid_simple.py")
NAS_PATH = "/var/services/homes/benutzername/boersenbot/hybrid_daily.py"
NAS_HOME = "/var/services/homes/benutzername/boersenbot"

# Cron entry (22:00 UTC daily)
CRON_LINE = "0 22 * * * benutzername python3 /var/services/homes/benutzername/boersenbot/hybrid_daily.py >> /var/services/homes/benutzername/boersenbot/hybrid.log 2>&1"

print("="*80)
print("DEPLOY HYBRID SYSTEM TO SYNOLOGY NAS")
print("="*80)

# 1) Read local script
print(f"\n[1/4] Reading {LOCAL_SCRIPT} ...", flush=True)
if not LOCAL_SCRIPT.exists():
    print(f"  ERROR: {LOCAL_SCRIPT} not found")
    exit(1)

with open(LOCAL_SCRIPT, "rb") as f:
    script_bytes = f.read()

print(f"  {len(script_bytes)} bytes")

# 2) SSH connect
print("\n[2/4] Connecting to NAS ...", flush=True)
try:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PW, timeout=10)
    print(f"  Connected to {HOST}")
except Exception as e:
    print(f"  ERROR: {e}")
    exit(1)

# 3) Create directory + upload via base64
print("\n[3/4] Uploading script via base64-pipe ...", flush=True)
try:
    # Ensure directory exists
    client.exec_command(f"mkdir -p {NAS_HOME}")

    # base64 encode
    b64 = base64.b64encode(script_bytes).decode('ascii')

    # Upload
    cmd = f"echo '{b64}' | base64 -d > {NAS_PATH}"
    stdin, stdout, stderr = client.exec_command(cmd)
    err = stderr.read().decode()
    if err:
        print(f"  WARNING: {err}")
    else:
        print(f"  Uploaded to {NAS_PATH}")

    # Verify
    stdin, stdout, stderr = client.exec_command(f"wc -l {NAS_PATH}")
    result = stdout.read().decode().strip()
    print(f"  Verify: {result}")

except Exception as e:
    print(f"  ERROR: {e}")
    client.close()
    exit(1)

# 4) Add cron entry
print("\n[4/4] Adding cron entry ...", flush=True)
try:
    # Check if already exists
    stdin, stdout, stderr = client.exec_command(f"grep -q '{NAS_PATH}' /etc/crontab && echo 'exists' || echo 'new'")
    exists = stdout.read().decode().strip()

    if exists == "exists":
        print(f"  Cron entry already exists, skipping")
    else:
        # Append cron entry (need sudo)
        cmd = f"echo '{PW}' | sudo -S sh -c \"echo '{CRON_LINE}' >> /etc/crontab\""
        stdin, stdout, stderr = client.exec_command(cmd)
        err = stderr.read().decode()
        if err and "sorry" not in err.lower():
            print(f"  Cron entry added")
        elif "sorry" in err.lower():
            print(f"  WARNING: sudo password failed, manual cron setup needed")

    # Restart crond
    cmd = f"echo '{PW}' | sudo -S kill -HUP $(pidof crond) 2>/dev/null || true"
    stdin, stdout, stderr = client.exec_command(cmd)
    print(f"  Crond reloaded")

except Exception as e:
    print(f"  WARNING: Could not add cron: {e}")

client.close()

print("\n" + "="*80)
print("DEPLOYMENT COMPLETE")
print("="*80)
print(f"\nScript deployed to: {NAS_PATH}")
print(f"Cron: Daily 22:00 UTC")
print(f"Log: {NAS_HOME}/hybrid.log")
print(f"\nVerify: ssh {USER}@{HOST} 'tail -f {NAS_HOME}/hybrid.log'")
