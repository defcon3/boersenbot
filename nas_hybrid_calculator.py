#!/usr/bin/env python3
"""
HYBRID SIGNAL CALCULATOR (NAS Native)

Pure Python - keine numpy/pandas/yfinance Dependencies.
Nur requests (auf NAS installiert).

Laeuft direkt auf der NAS, schreibt JSON + Log lokal.

Strategy: MA50 > MA200 + VIX-normalized Position-Sizing
"""
import json
import os
import sys
import requests
import statistics
from datetime import datetime, timezone

# Config (Pfade auf NAS)
BASE_DIR = "/var/services/homes/benutzername/boersenbot"
SIGNAL_PATH = os.path.join(BASE_DIR, "today_signal.json")
LOG_PATH = os.path.join(BASE_DIR, "hybrid.log")

YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; HybridBot/1.0)"}


def fetch_yahoo_close(ticker, days=300):
    """Hole letzte N Tage Close-Preise von Yahoo Finance."""
    url = f"{YF_BASE}/{ticker}"
    params = {
        "interval": "1d",
        "range": f"{days}d",
        "includePrePost": "false"
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]

    # Filtere None-Werte (Holidays, etc.)
    filtered = [(t, c) for t, c in zip(timestamps, closes) if c is not None]
    return filtered  # list of (unix_ts, close_price)


def log_entry(message):
    """Append Log-Eintrag."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{ts}] {message}\n")


print("="*80)
print("HYBRID SIGNAL CALCULATOR (NAS Native)")
print("="*80)

try:
    # 1) SPY Close
    print("\n[1/4] Lade SPY Close-Preise...")
    spy_data = fetch_yahoo_close("SPY", days=300)
    spy_closes = [c for _, c in spy_data]
    print(f"  SPY: {len(spy_closes)} Werte, latest={spy_closes[-1]:.2f}")

    if len(spy_closes) < 200:
        raise RuntimeError(f"Zu wenig SPY-Daten: {len(spy_closes)} < 200")

    # 2) VIX Close
    print("[2/4] Lade VIX Close-Preise...")
    vix_data = fetch_yahoo_close("^VIX", days=80)
    vix_closes = [c for _, c in vix_data]
    print(f"  VIX: {len(vix_closes)} Werte, latest={vix_closes[-1]:.2f}")

    if len(vix_closes) < 60:
        raise RuntimeError(f"Zu wenig VIX-Daten: {len(vix_closes)} < 60")

    # 3) Compute MAs + VIX Normalization
    print("[3/4] Berechne MAs + VIX-Norm...")
    ma50 = statistics.mean(spy_closes[-50:])
    ma200 = statistics.mean(spy_closes[-200:])
    current_price = spy_closes[-1]
    uptrend_signal = 1 if ma50 > ma200 else 0

    vix_recent = vix_closes[-60:]
    vix_mean = statistics.mean(vix_recent)
    vix_std = statistics.stdev(vix_recent)
    current_vix = vix_closes[-1]
    vix_norm_raw = ((current_vix - vix_mean) / (vix_std + 1e-6)) * 0.1
    vix_norm = max(-0.5, min(0.5, vix_norm_raw))  # Clip

    # Position-Sizing
    size_raw = 1.0 - vix_norm
    size_clipped = max(0.2, min(1.0, size_raw))
    position_size = uptrend_signal * size_clipped
    action = "LONG" if position_size > 0.2 else "FLAT"

    print(f"  MA50={ma50:.2f}, MA200={ma200:.2f}, Uptrend={uptrend_signal}")
    print(f"  VIX={current_vix:.2f}, Norm={vix_norm:+.3f}")
    print(f"  Position-Size: {position_size:.3f} ({action})")

    # 4) Write Signal-JSON
    print("[4/4] Schreibe Signal-JSON...")
    signal = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "current_price": current_price,
        "ma50": ma50,
        "ma200": ma200,
        "uptrend_signal": uptrend_signal,
        "current_vix": current_vix,
        "vix_mean_60d": vix_mean,
        "vix_norm": vix_norm,
        "position_size": position_size,
        "action": action,
        "calculator": "nas_native_v1"
    }

    with open(SIGNAL_PATH, "w") as f:
        json.dump(signal, f, indent=2)

    log_entry(f"Hybrid: Signal={action}, PosSize={position_size:.3f}, "
              f"VIX={current_vix:.2f}, MA50/200={'UP' if uptrend_signal else 'DN'}")

    print(f"\n[OK] Signal: {SIGNAL_PATH}")
    print(f"[OK] Log: {LOG_PATH}")

except Exception as e:
    err_msg = f"Hybrid-Calculator ERROR: {e}"
    print(f"\n[FAIL] {err_msg}", file=sys.stderr)
    try:
        log_entry(f"[ERROR] {err_msg}")
    except Exception:
        pass
    sys.exit(1)
