# -*- coding: utf-8 -*-
"""weather_outlier_screen_low.py — Lay-Kandidaten fuer Jupiter-Wetter-Buckets
("Lowest temperature in X on <Datum>?").

Schwester zu weather_outlier_screen.py, aber fuer Tagestiefstand statt Tageshoch.
Methodik identisch zu weather_outlier_screen.py.

ACHTUNG (Stand 09.07.2026): Bias/Sigma aus preregs/weather_source_calib_*.csv sind
aus TAGESHOCH-Fehlern kalibriert und werden hier unvalidiert auf das Tagestief
angewandt. Nachtminima haben andere Fehlercharakteristik (Ausstrahlung, Nebel,
Inversionen) -> Kandidaten nur mit extra Sicherheitsabstand handeln, bis eine
eigene Min-Kalibrierung (weather_source_compare.py auf Tagesminima) existiert.

Aufruf:
  python weather_outlier_screen_low.py                  # Zieltag = morgen (UTC)
  python weather_outlier_screen_low.py --date 2026-07-10
"""
import argparse
import csv
import glob
import math
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import airportsdata
import requests

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

API = "https://api.jup.ag/prediction/v1"
OM = "https://api.open-meteo.com/v1/forecast"
CALIB_GLOB = r"preregs/weather_source_calib_*.csv"

MODELS = ["gfs_seamless", "icon_seamless", "ukmo_seamless", "jma_seamless", "ecmwf_ifs025"]
SHORT = {"gfs_seamless": "GFS", "icon_seamless": "ICON", "ukmo_seamless": "UKMO",
         "jma_seamless": "JMA", "ecmwf_ifs025": "ECMWF", "ensemble_mean": "ENS"}

STATIONS = {
    "Wellington": "NZWN", "Tokyo": "RJTT", "Seoul": "RKSI", "Shanghai": "ZSPD",
    "Beijing": "ZBAA", "Kuala Lumpur": "WMKK", "Shenzhen": "ZGSZ", "Chengdu": "ZUUU",
    "Karachi": "OPKC", "Jeddah": "OEJN", "Ankara": "LTAC", "Helsinki": "EFHK",
    "London": "EGLC", "Paris": "LFPB", "Madrid": "LEMD", "Milan": "LIMC",
    "Munich": "EDDM", "Amsterdam": "EHAM", "Warsaw": "EPWA", "Cape Town": "FACT",
    "Mexico City": "MMMX", "Buenos Aires": "SAEZ",
    "Sao Paulo": "SBGR", "Taipei": "RCSS", "Tel Aviv": "LLBG",
    "Toronto": "CYYZ", "Wuhan": "ZHHH", "Panama City": "MPMG",
}

MIN_DIST = 2.0
MAX_PMODEL = 0.10
MIN_YES = 0.025

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"


def ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bucket_prob(kind, k, mu, sigma):
    if kind == "le":
        return ncdf((k + 0.5 - mu) / sigma)
    if kind == "ge":
        return 1.0 - ncdf((k - 0.5 - mu) / sigma)
    return ncdf((k + 0.5 - mu) / sigma) - ncdf((k - 0.5 - mu) / sigma)


def dist_deg(kind, k, mu):
    if kind == "le":
        return mu - (k + 0.5)
    if kind == "ge":
        return (k - 0.5) - mu
    return abs(k - mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="Zieltag YYYY-MM-DD (default: morgen UTC)")
    args = ap.parse_args()
    if args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    target_day = target.isoformat()
    title_day = f"{MONTHS[target.month - 1]} {target.day}"
    title_re = re.compile(rf"^Lowest temperature in (.+) on {re.escape(title_day)}\?")

    calib = {}
    for path in sorted(glob.glob(CALIB_GLOB)):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                calib[(row["city"], row["model"])] = (float(row["bias"]), float(row["sigma"]))
    if not calib:
        sys.exit(f"Keine Kalibrierung unter {CALIB_GLOB} gefunden (im Repo-Root ausfuehren).")

    print(f"Ziel: 'Lowest temperature in ... on {title_day}?' ({target_day})")
    print("Lade Jupiter-Wetter-Events ...", flush=True)
    events = []
    for s in range(0, 120, 10):
        page = None
        for attempt in range(4):
            r = S.get(f"{API}/events", params={"category": "weather", "start": s, "end": s + 10}, timeout=30)
            if r.status_code == 429:
                wait = 6 * (attempt + 1)
                print(f"  429 bei page {s}, warte {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            page = r.json().get("data", [])
            break
        if page is None:
            print(f"  page {s}: aufgegeben (429)")
            break
        events += page
        if len(page) < 10:
            break
        time.sleep(1.5)
    print(f"  {len(events)} Events geladen.")

    targets = {}
    for e in events:
        t = e.get("metadata", {}).get("title", "")
        m = title_re.match(t)
        if not m:
            continue
        city = m.group(1)
        mks, celsius = [], True
        for mk in e.get("markets", []):
            ti = mk.get("title", "")
            if "°C" not in ti:
                celsius = False
                break
            num = re.search(r"(-?\d+)", ti)
            if not num:
                continue
            kind = "le" if "below" in ti else ("ge" if "higher" in ti else "eq")
            pr = mk.get("pricing") or {}
            mks.append({
                "kind": kind, "k": int(num.group(1)), "title": ti,
                "marketId": mk.get("marketId"), "status": mk.get("status"),
                "buyYes": (pr.get("buyYesPriceUsd") or 0) / 1e6,
                "buyNo": (pr.get("buyNoPriceUsd") or 0) / 1e6,
            })
        if celsius and mks:
            targets[city] = sorted(mks, key=lambda x: x["k"])
    print(f"  Staedte in °C: {sorted(targets)}")

    AP = airportsdata.load("ICAO")
    rows = []
    city_info = {}
    for city, mks in sorted(targets.items()):
        icao = STATIONS.get(city)
        if not icao or icao not in AP:
            print(f"  {city}: keine Station -> skip")
            continue
        if (city, "ensemble_mean") not in calib:
            print(f"  {city}: keine Kalibrierung -> skip")
            continue
        lat, lon = AP[icao]["lat"], AP[icao]["lon"]
        try:
            r = S.get(OM, params={
                "latitude": lat, "longitude": lon, "hourly": "temperature_2m",
                "models": ",".join(MODELS), "timezone": "auto", "forecast_days": 5,
            }, timeout=30)
            r.raise_for_status()
            j = r.json()
        except Exception as ex:
            print(f"  {city}: Open-Meteo-Fehler {ex} -> skip")
            continue
        hh = j.get("hourly", {})
        times = hh.get("time", [])
        raw = {}
        for mdl in MODELS:
            vals = hh.get(f"temperature_2m_{mdl}", [])
            mn = min((v for t, v in zip(times, vals)
                      if v is not None and t.startswith(target_day)), default=None)
            raw[mdl] = mn
        have = [m for m in MODELS if raw[m] is not None]
        if len(have) < 3:
            print(f"  {city}: nur {len(have)} Modelle -> skip")
            continue
        mu = {m: raw[m] - calib[(city, m)][0] for m in have if (city, m) in calib}
        ens_raw = sum(raw[m] for m in have) / len(have)
        b_e, s_e = calib[(city, "ensemble_mean")]
        mu_ens, sig_ens = ens_raw - b_e, s_e

        open_mks = [x for x in mks if x["status"] == "open"]
        fav = max(open_mks, key=lambda x: x["buyYes"]) if open_mks else None
        city_info[city] = {"mu": mu, "mu_ens": mu_ens, "sig_ens": sig_ens,
                           "fav": fav, "mks": mks, "icao": icao}

        for x in open_mks:
            probs = {}
            for m, mm in mu.items():
                probs[m] = bucket_prob(x["kind"], x["k"], mm, calib[(city, m)][1])
            probs["ensemble_mean"] = bucket_prob(x["kind"], x["k"], mu_ens, sig_ens)
            pmax_m = max(probs, key=probs.get)
            d = dist_deg(x["kind"], x["k"], mu_ens)
            rows.append({
                "city": city, **x,
                "p_ens": probs["ensemble_mean"], "p_max": probs[pmax_m], "p_max_src": SHORT[pmax_m],
                "dist": d, "dist_sig": d / sig_ens if sig_ens else 0.0,
            })
        time.sleep(0.5)

    print("\n" + "=" * 100)
    print(f"KANDIDATEN-FILTER: dist>={MIN_DIST}°C, alle Modelle P<={MAX_PMODEL:.0%}, buyYes>={MIN_YES:.0%}, NO kaufbar")
    print("=" * 100)
    cand = [r for r in rows
            if r["dist"] >= MIN_DIST and r["p_max"] <= MAX_PMODEL
            and r["buyYes"] >= MIN_YES and 0 < r["buyNo"] < 1]
    cand.sort(key=lambda r: (1 - r["buyNo"]) / r["buyNo"], reverse=True)

    hdr = f"{'Stadt':13} {'Bucket':>15} {'YES':>5} {'NO':>6} {'Rend%':>6} {'P_ens':>6} {'P_max':>10} {'dist':>6} {'d/σ':>5}  Markt-ID"
    print(hdr)
    print("-" * len(hdr))
    for r in cand[:18]:
        rend = (1 - r["buyNo"]) / r["buyNo"] * 100
        print(f"{r['city']:13} {r['title']:>15} {r['buyYes']:5.2f} {r['buyNo']:6.3f} {rend:6.1f} "
              f"{r['p_ens']*100:5.1f}% {r['p_max']*100:5.1f}%/{r['p_max_src']:4} {r['dist']:5.1f}° {r['dist_sig']:5.1f}  {r['marketId']}")

    near = [r for r in rows if r not in cand and r["buyYes"] >= 0.06 and r["p_ens"] <= 0.10
            and 0 < r["buyNo"] < 1 and r["dist"] >= 1.0]
    near.sort(key=lambda r: r["buyYes"], reverse=True)
    if near:
        print("\nKNAPP VERFEHLT (Info, kein Kandidat):")
        for r in near[:8]:
            why = []
            if r["dist"] < MIN_DIST:
                why.append(f"dist {r['dist']:.1f}°<{MIN_DIST}")
            if r["p_max"] > MAX_PMODEL:
                why.append(f"P_max {r['p_max']:.0%} ({r['p_max_src']})")
            print(f"  {r['city']:13} {r['title']:>15} YES {r['buyYes']:.2f} NO {r['buyNo']:.3f} -> {', '.join(why)}")

    for city in sorted(city_info):
        ci = city_info[city]
        mu_s = "  ".join(f"{SHORT[m]} {v:.1f}" for m, v in ci["mu"].items())
        print(f"\n--- {city} ({ci['icao']})  korr. Forecasts: {mu_s} | ENS {ci['mu_ens']:.1f}±{ci['sig_ens']:.1f} ---")
        for x in ci["mks"]:
            if x["buyYes"] < 0.02 and not (ci["fav"] and x["marketId"] == ci["fav"]["marketId"]):
                continue
            p_e = bucket_prob(x["kind"], x["k"], ci["mu_ens"], ci["sig_ens"])
            mark = " <FAV" if ci["fav"] and x["marketId"] == ci["fav"]["marketId"] else ""
            inc = " *" if any(r["city"] == city and r["marketId"] == x["marketId"] for r in cand) else ""
            print(f"   {x['title']:>15}  YES {x['buyYes']:.2f}  NO {x['buyNo']:.3f}  P_ens {p_e*100:5.1f}%  {x['status']}{mark}{inc}")

    print(f"\n(Stand {datetime.now(timezone.utc).strftime('%d.%m. %H:%M UTC')}; "
          f"P aus Normal-Annahme, Kalibrierung 700d Lead-24h — ACHTUNG: Tageshoch-Kalibrierung auf Tief angewandt)")


if __name__ == "__main__":
    main()
