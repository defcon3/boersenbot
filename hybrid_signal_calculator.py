#!/usr/bin/env python3
"""
HYBRID SIGNAL CALCULATOR (Local)
Runs locally, calculates today's signal, saves to NAS via SSH.
No external dependencies except yfinance (already available).
"""
import json
import paramiko
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    import numpy as np
    import pandas as pd
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    exit(1)

# NAS credentials
NAS_HOST = "192.168.178.32"
NAS_USER = "benutzername"
NAS_PW = "f/hGT5%4$fh"
NAS_SIGNAL_PATH = "/var/services/homes/benutzername/boersenbot/today_signal.json"
NAS_LOG_PATH = "/var/services/homes/benutzername/boersenbot/hybrid.log"

print("="*80)
print("HYBRID SIGNAL CALCULATOR")
print("="*80)

try:
    # 1) FETCH DATA (need full history for MA200)
    print("\n[1/4] Fetching SPY and VIX data...", flush=True)
    spy = yf.download("SPY", start="2014-01-01", progress=False)
    vix = yf.download("^VIX", start="2014-01-01", progress=False)["Close"]

    # 2) CALCULATE MA50, MA200
    print("[2/4] Calculating moving averages...", flush=True)
    close_prices = np.asarray(spy['Close']).flatten()
    ma50 = float(np.mean(close_prices[-50:]))
    ma200 = float(np.mean(close_prices[-200:]))

    current_price = float(close_prices[-1])
    uptrend_signal = 1 if ma50 > ma200 else 0

    # 3) CALCULATE VIX NORMALIZATION
    print("[3/4] Calculating VIX normalization...", flush=True)
    vix_vals = np.asarray(vix).flatten()
    vix_mean = float(np.mean(vix_vals[-60:]))
    vix_std = float(np.std(vix_vals[-60:]))
    current_vix = float(vix_vals[-1])

    vix_norm = ((current_vix - vix_mean) / (vix_std + 1e-6)) * 0.1
    vix_norm = np.clip(vix_norm, -0.5, 0.5)

    # Position sizing
    position_size = uptrend_signal * np.clip(1 - vix_norm, 0.2, 1.0)

    # 4) CREATE SIGNAL JSON
    print("[4/4] Creating signal JSON...", flush=True)
    signal_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "current_price": float(current_price),
        "ma50": float(ma50),
        "ma200": float(ma200),
        "uptrend_signal": int(uptrend_signal),
        "current_vix": float(current_vix),
        "vix_mean_60d": float(vix_mean),
        "vix_norm": float(vix_norm),
        "position_size": float(position_size),
        "action": "LONG" if position_size > 0.2 else "FLAT"
    }

    signal_json = json.dumps(signal_data, indent=2)

    print("\n" + "="*80)
    print("TODAY'S SIGNAL")
    print("="*80)
    print(signal_json)

    # UPLOAD TO NAS
    print("\n[UPLOAD] Sending to NAS...", flush=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(NAS_HOST, username=NAS_USER, password=NAS_PW, timeout=10)

    # Upload signal JSON
    b64_signal = base64.b64encode(signal_json.encode()).decode()
    cmd = f"echo '{b64_signal}' | base64 -d > {NAS_SIGNAL_PATH}"
    stdin, stdout, stderr = client.exec_command(cmd)
    stdout.read()

    # Append to log
    log_entry = f"[{datetime.utcnow().isoformat()}] Signal={signal_data['action']}, PosSize={position_size:.3f}, VIX={current_vix:.2f}, MA50/200={'UP' if uptrend_signal else 'DN'}\n"
    b64_log = base64.b64encode(log_entry.encode()).decode()
    cmd = f"echo '{b64_log}' | base64 -d >> {NAS_LOG_PATH}"
    stdin, stdout, stderr = client.exec_command(cmd)
    stdout.read()

    client.close()

    print(f"[OK] Signal uploaded to NAS")
    print(f"\n[OK] HYBRID SIGNAL CALCULATED SUCCESSFULLY")

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    exit(1)
