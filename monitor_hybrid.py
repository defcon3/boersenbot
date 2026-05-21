#!/usr/bin/env python3
"""
HYBRID SYSTEM MONITOR

Watch NAS hybrid.log für:
- Tägliche Executions (mindestens 1x täglich)
- Errors/Exceptions (traceback, ValueError, etc)
- Slippage vs Backtest (wenn verfügbar)

Alerts via Email (SMTP)
"""
import paramiko
import time
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText

# Credentials
NAS_HOST = "192.168.178.32"
NAS_USER = "benutzername"
NAS_PW = "f/hGT5%4$fh"
NAS_LOG = "/var/services/homes/benutzername/boersenbot/hybrid.log"

# Email config
EMAIL_FROM = "claude-code-monitor@localhost"
EMAIL_TO = "veit.luther@gmx.de"
SMTP_SERVER = "localhost"
SMTP_PORT = 25

def fetch_log(client, lines=50):
    """Fetch last N lines from NAS log."""
    try:
        stdin, stdout, stderr = client.exec_command(f"tail -{lines} {NAS_LOG}")
        return stdout.read().decode()
    except Exception as e:
        return f"ERROR: {e}"

def check_errors(log_content):
    """Check for errors in log."""
    errors = []
    keywords = ["Traceback", "Error", "Exception", "FAIL", "WARNING"]

    for line in log_content.split('\n'):
        for kw in keywords:
            if kw.lower() in line.lower():
                errors.append(line.strip())

    return errors

def check_execution(log_content):
    """Check if ran today."""
    today = datetime.now().date()
    today_str = today.isoformat()

    if today_str in log_content:
        return True, f"Executed today ({today})"
    else:
        return False, f"NOT executed today (last log might be from {today - timedelta(days=1)})"

def send_alert(subject, body):
    """Send email alert."""
    try:
        msg = MIMEText(body)
        msg["Subject"] = f"[HYBRID-MONITOR] {subject}"
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())

        print(f"  Alert sent: {subject}")
    except Exception as e:
        print(f"  ERROR sending email: {e}")

print("="*80)
print("HYBRID SYSTEM MONITOR")
print("="*80)
print(f"Connecting to {NAS_HOST} ...\n")

try:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(NAS_HOST, username=NAS_USER, password=NAS_PW, timeout=10)

    # Fetch log
    print("[1/3] Fetching hybrid.log ...", flush=True)
    log_content = fetch_log(client, lines=100)
    print(f"  Fetched {len(log_content)} bytes\n")

    # Check execution
    print("[2/3] Checking daily execution ...", flush=True)
    executed, msg = check_execution(log_content)
    print(f"  {msg}")

    if not executed:
        send_alert("HYBRID NOT EXECUTED TODAY", f"Last log:\n{log_content[-500:]}")

    # Check errors
    print("\n[3/3] Checking for errors ...", flush=True)
    errors = check_errors(log_content)

    if errors:
        print(f"  Found {len(errors)} error-like lines:")
        for err in errors[:5]:  # Show first 5
            print(f"    - {err}")

        send_alert("HYBRID ERRORS DETECTED", f"Errors:\n" + "\n".join(errors[:10]))
    else:
        print(f"  No errors detected")

    client.close()

    print("\n" + "="*80)
    print(f"Monitor check completed at {datetime.now().isoformat()}")
    print("="*80)

except Exception as e:
    print(f"\nERROR: {e}")
    send_alert("MONITOR CONNECTION FAILED", f"Could not connect to NAS: {e}")

