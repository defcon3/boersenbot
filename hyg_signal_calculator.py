#!/usr/bin/env python3
"""
HYG SIGNAL CALCULATOR (Local)

Berechnet täglich das HYG-Position-Sizing-Signal basierend auf STLFSI4.

Strategy S1 (Sizing 50/100):
  - Normal: 50% HYG, 50% Cash
  - Bei Stress (STLFSI4 > 0.2909): 100% HYG für 20 Tage
  - Stress-Schwelle: Train-Q75 aus 2003-2018 (siehe hyg_stress_buy_edge memory)

Lädt Signal-JSON auf NAS via SSH + base64-pipe.
"""
import json
import paramiko
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from fred_helper import get_series
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    exit(1)

# Config (validiert durch Backtest)
STRESS_THRESHOLD = 0.2909  # Train-Q75 STLFSI4 2003-2018
HOLDING_DAYS = 20  # Forward-Window aus Pre-Reg
NORMAL_EXPOSURE = 0.5
STRESS_EXPOSURE = 1.0

# NAS
NAS_HOST = "192.168.178.32"
NAS_USER = "benutzername"
NAS_PW = "f/hGT5%4$fh"
NAS_SIGNAL_PATH = "/var/services/homes/benutzername/boersenbot/hyg_today_signal.json"
NAS_LOG_PATH = "/var/services/homes/benutzername/boersenbot/hyg.log"

print("="*80)
print("HYG SIGNAL CALCULATOR")
print("="*80)

try:
    # 1) FRED Daten
    print("\n[1/5] Lade STLFSI4 (Stress-Index)...", flush=True)
    stlfsi = get_series("STLFSI4", start="2024-01-01", force_refresh=True)
    print(f"  STLFSI4: {len(stlfsi)} Werte, latest = {stlfsi.iloc[-1]:.4f}")

    current_stlfsi = float(stlfsi.iloc[-1])
    last_update = stlfsi.index[-1].strftime("%Y-%m-%d")

    # 2) Holding-Period-Tracking
    print("\n[2/5] Stress-History für Holding-Period...", flush=True)
    # Letzte HOLDING_DAYS Werte (ca. 4 Wochen für wöchentliche STLFSI)
    recent_stress = (stlfsi.tail(HOLDING_DAYS) > STRESS_THRESHOLD).any()
    days_since_stress = 0
    if recent_stress:
        last_stress_date = stlfsi[stlfsi > STRESS_THRESHOLD].index[-1]
        days_since_stress = (datetime.utcnow().date() - last_stress_date.date()).days
    print(f"  Stress in last {HOLDING_DAYS} Tage: {recent_stress}, days since stress: {days_since_stress}")

    # 3) Position-Size bestimmen
    print("\n[3/5] Position-Sizing...", flush=True)
    in_stress_window = recent_stress and (days_since_stress <= HOLDING_DAYS)
    exposure = STRESS_EXPOSURE if in_stress_window else NORMAL_EXPOSURE
    action = "STRESS_BUY (100%)" if in_stress_window else "NORMAL (50%)"
    print(f"  Action: {action}")
    print(f"  Exposure: {exposure*100:.0f}% HYG")

    # 4) HYG Preis (Sanity-Check)
    print("\n[4/5] HYG aktueller Preis...", flush=True)
    hyg = yf.download("HYG", period="5d", progress=False)
    hyg_price = float(np.asarray(hyg['Close']).flatten()[-1])
    print(f"  HYG: ${hyg_price:.2f}")

    # 5) Signal-JSON erstellen
    signal_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "stlfsi4_value": current_stlfsi,
        "stlfsi4_last_update": last_update,
        "stress_threshold": STRESS_THRESHOLD,
        "in_stress_window": bool(in_stress_window),
        "days_since_stress": days_since_stress,
        "holding_days": HOLDING_DAYS,
        "exposure_pct": exposure * 100,
        "action": action,
        "hyg_price": hyg_price,
        "strategy": "S1_sizing_50_100"
    }

    signal_json = json.dumps(signal_data, indent=2)

    print("\n" + "="*80)
    print("TODAY'S HYG SIGNAL")
    print("="*80)
    print(signal_json)

    # 6) Upload zu NAS
    print("\n[5/5] Upload zu NAS...", flush=True)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(NAS_HOST, username=NAS_USER, password=NAS_PW, timeout=10)

    b64_signal = base64.b64encode(signal_json.encode()).decode()
    cmd = f"echo '{b64_signal}' | base64 -d > {NAS_SIGNAL_PATH}"
    stdin, stdout, stderr = client.exec_command(cmd)
    stdout.read()

    # Log
    log_entry = (
        f"[{datetime.utcnow().isoformat()}] "
        f"STLFSI4={current_stlfsi:.4f} "
        f"({'STRESS' if in_stress_window else 'NORMAL'}) "
        f"Exposure={exposure*100:.0f}% "
        f"DaysInStress={days_since_stress}/{HOLDING_DAYS}\n"
    )
    b64_log = base64.b64encode(log_entry.encode()).decode()
    cmd = f"echo '{b64_log}' | base64 -d >> {NAS_LOG_PATH}"
    stdin, stdout, stderr = client.exec_command(cmd)
    stdout.read()

    client.close()
    print("[OK] Signal auf NAS hochgeladen")
    print("\n[OK] HYG SIGNAL CALCULATED")

except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    exit(1)
