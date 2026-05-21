#!/usr/bin/env python3
"""
HYG SIGNAL CALCULATOR (NAS Native)

Pure Python - keine numpy/pandas/yfinance Dependencies.
Nur requests + Standard-Lib.

Liest STLFSI4 (wöchentlich, Friday) direkt von FRED, schreibt Signal lokal.

Strategy S1 Sizing 50/100:
  Normal: 50% HYG, 50% Cash
  Bei Stress (STLFSI4 > 0.2909): 100% HYG für 20 Tage
"""
import json
import os
import sys
import requests
from datetime import datetime, timezone

# Config
BASE_DIR = "/var/services/homes/benutzername/boersenbot"
SIGNAL_PATH = os.path.join(BASE_DIR, "hyg_today_signal.json")
LOG_PATH = os.path.join(BASE_DIR, "hyg.log")
FRED_KEY_PATH = os.path.join(BASE_DIR, ".fred_key")

# Strategy-Parameter (validated by backtest)
STRESS_THRESHOLD = 0.2909  # Train-Q75 STLFSI4 2003-2018
HOLDING_DAYS = 20
NORMAL_EXPOSURE = 0.50
STRESS_EXPOSURE = 1.00

YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; HygBot/1.0)"}


def load_fred_key():
    if not os.path.exists(FRED_KEY_PATH):
        raise RuntimeError(f"FRED-Key fehlt: {FRED_KEY_PATH}")
    with open(FRED_KEY_PATH) as f:
        return f.read().strip()


def fetch_fred_series(series_id, start="2024-01-01"):
    """FRED-Daten via API."""
    api_key = load_fred_key()
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json()["observations"]
    # Returns list of (date_str, value)
    result = []
    for o in obs:
        try:
            v = float(o["value"])
            result.append((o["date"], v))
        except ValueError:
            continue  # Skip "."-Werte
    return result


def fetch_yahoo_close(ticker, days=10):
    url = f"{YF_BASE}/{ticker}"
    params = {"interval": "1d", "range": f"{days}d"}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    return [c for c in closes if c is not None]


def log_entry(message):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{ts}] {message}\n")


print("="*80)
print("HYG SIGNAL CALCULATOR (NAS Native)")
print("="*80)

try:
    # 1) STLFSI4 von FRED
    print("\n[1/4] Lade STLFSI4 von FRED...")
    stlfsi_data = fetch_fred_series("STLFSI4", start="2024-01-01")
    if not stlfsi_data:
        raise RuntimeError("Keine STLFSI4-Daten erhalten")

    last_date, last_value = stlfsi_data[-1]
    print(f"  STLFSI4: {len(stlfsi_data)} Werte, latest = {last_value:.4f} ({last_date})")

    # 2) Holding-Period-Tracking
    print("[2/4] Stress-History pruefen...")
    # Letzte ~4 Wochen (für 20-Tage-Holding mit weekly STLFSI = ~4 Werte)
    # Suche letzten Stress-Eintrag
    recent_stress_dates = [d for d, v in stlfsi_data[-30:] if v > STRESS_THRESHOLD]

    today = datetime.now(timezone.utc).date()
    if recent_stress_dates:
        last_stress = datetime.strptime(recent_stress_dates[-1], "%Y-%m-%d").date()
        days_since_stress = (today - last_stress).days
        in_stress_window = days_since_stress <= HOLDING_DAYS
    else:
        days_since_stress = 999
        in_stress_window = False

    print(f"  Letzter Stress: {recent_stress_dates[-1] if recent_stress_dates else 'keiner'}")
    print(f"  Tage seit Stress: {days_since_stress}")
    print(f"  In Stress-Window: {in_stress_window}")

    # 3) Position-Sizing
    print("[3/4] Position-Sizing...")
    exposure = STRESS_EXPOSURE if in_stress_window else NORMAL_EXPOSURE
    action = "STRESS_BUY (100%)" if in_stress_window else "NORMAL (50%)"
    print(f"  Action: {action}")

    # 4) HYG-Preis (Sanity-Check)
    print("[4/4] HYG-Preis...")
    hyg_prices = fetch_yahoo_close("HYG", days=5)
    hyg_price = hyg_prices[-1] if hyg_prices else None
    print(f"  HYG: ${hyg_price:.2f}" if hyg_price else "  HYG: N/A")

    # Signal-JSON
    signal = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "stlfsi4_value": last_value,
        "stlfsi4_last_update": last_date,
        "stress_threshold": STRESS_THRESHOLD,
        "in_stress_window": in_stress_window,
        "days_since_stress": days_since_stress,
        "holding_days": HOLDING_DAYS,
        "exposure_pct": exposure * 100,
        "action": action,
        "hyg_price": hyg_price,
        "strategy": "S1_sizing_50_100",
        "calculator": "nas_native_v1"
    }

    with open(SIGNAL_PATH, "w") as f:
        json.dump(signal, f, indent=2)

    log_entry(f"HYG: STLFSI4={last_value:.4f} {'STRESS' if in_stress_window else 'NORMAL'} "
              f"Exposure={exposure*100:.0f}% HYG=${hyg_price:.2f} "
              f"DaysSinceStress={days_since_stress}/{HOLDING_DAYS}")

    print(f"\n[OK] Signal: {SIGNAL_PATH}")
    print(f"[OK] Log: {LOG_PATH}")

except Exception as e:
    err_msg = f"HYG-Calculator ERROR: {e}"
    print(f"\n[FAIL] {err_msg}", file=sys.stderr)
    try:
        log_entry(f"[ERROR] {err_msg}")
    except Exception:
        pass
    sys.exit(1)
