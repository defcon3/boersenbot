# -*- coding: utf-8 -*-
"""weather_outlier_screen_low.py — Lay-Kandidaten fuer Jupiter-Wetter-Buckets
("Lowest temperature in X on <Datum>?").

Schwester zu weather_outlier_screen.py, aber fuer Tagestiefstand statt Tageshoch.
Methodik identisch zu weather_outlier_screen.py.

Seit 10.07.2026 mit EIGENER Min-Kalibrierung (weather_source_compare.py --var min,
700d Lead-24h auf Tagesminima -> preregs/weather_source_calib_min_*.csv). Die
zuvor benutzte Tageshoch-Kalibrierung lag auf Minima nachweislich daneben
(Shanghai: High-Korrektur hebt an, Min-Wahrheit ist bias +0,34 -> ~1,5C Fehler
in mu; Beinahe-Fehltrade "Lowest 24C" am 10.07.). Staedte ohne Min-Kalibrierung
werden automatisch geskippt -- bei Bedarf nachkalibrieren.

Seit 14.07.2026 werden Filter-Logik und Schwellen aus weather_outlier_screen.py
IMPORTIERT statt kopiert. Der Beijing-33-Verlust
(preregs/weather_lay_postmortem_2026_07_14_beijing.md) entstand an einer nicht-
ausreisser-robusten Ensemble-Mittelung, die BEIDE Screens hatten — ein Fix, der
nur in einer Kopie landet, ist kein Fix. Damit gelten hier automatisch:
Ausreisser-Bereinigung, Spannen-Veto, Doppel-Kalibrierung (700d + 40d); seit
16.07. die Wetterfrosch-Doktrin (markt-blind, P_pess-Ranking, MIN_EV entfiel)
und seit 17.07. Debias-vor-Mittelung + sigma(s) fuer Einzelmodelle + Lead-
Autowahl (build_views/model_sigma aus dem High-Screen).

Aufruf:
  python weather_outlier_screen_low.py                  # Zieltag = morgen (UTC)
  python weather_outlier_screen_low.py --date 2026-07-10
"""
import argparse
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import airportsdata

from weather_stations import station_info
import requests

from weather_outlier_screen import (MIN_DIST, MAX_PMODEL, MIN_YES, MAX_SPREAD,
                                    bucket_prob, dist_deg, robust_mean, load_calib,
                                    reject_reasons, ens_sigma, model_sigma, build_views)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

API = "https://api.jup.ag/prediction/v1"
OM = "https://api.open-meteo.com/v1/forecast"
CALIB_GLOB = r"preregs/weather_source_calib_min_*.csv"        # 700d, Ganzjahr, Tagestief
CALIB40_GLOB = r"preregs/weather_source_calib40d_min_*.csv"   # 40d, Sommer, Tagestief

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
    # Moskau (25.07.) hier NACHGETRAGEN am 26.07.: der urspruengliche Commit
    # erwischte nur drei der vier STATIONS-Maps, diese blieb zurueck.
    # Settlement ueber NOAA (weather.gov/wrh/timeseries?site=UUWW).
    "Moscow": "UUWW",
    # Hong Kong nachgeruestet 26.07.: Pseudo-Station "HKO" (weather_stations.
    # SPECIAL_STATIONS). Settelt auf die HKO-Klimareihe, NICHT auf METAR/
    # Wunderground — Settle-Pfade muessen die Station ueberspringen.
    "Hong Kong": "HKO",
    # NYC nachgeruestet 26.07.: Settlement-Station laut Marktregel ist
    # LaGuardia (KLGA), NICHT Central Park — "the lowest temperature recorded at
    # the LaGuardia Airport Station", Quelle wunderground.com/history/daily/us/
    # ny/new-york-city/KLGA. ACHTUNG: Die NYC-Bretter laufen in FAHRENHEIT mit
    # 2-Grad-Buckets ("70-71F"); der Ladder-Logger verwirft sie deshalb weiter
    # (celsius-Pruefung), die Kalibrierung hier ist einheitenunabhaengig.
    "NYC": "KLGA",
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="Zieltag YYYY-MM-DD (default: morgen UTC)")
    ap.add_argument("--force-lead1", action="store_true",
                    help="bei Zieltag >24h trotzdem mit der Lead-24h-Kalibrierung rechnen "
                         "(nur zur Orientierung — nie darauf setzen, Madrid-Lehre 13.07.)")
    args = ap.parse_args()
    if args.date:
        target = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    target_day = target.isoformat()
    title_day = f"{MONTHS[target.month - 1]} {target.day}"
    title_re = re.compile(rf"^Lowest temperature in (.+) on {re.escape(title_day)}\?")

    # Lead-Autowahl analog weather_outlier_screen.py (17.07., Backlog Prio 1).
    lead_days = max(1, (target - datetime.now(timezone.utc).date()).days)
    use_lead = 1 if args.force_lead1 else lead_days
    if use_lead > 1:
        cg = CALIB_GLOB.replace("weather_source_calib_min_", f"weather_source_calib_min_lead{use_lead}_")
        cg40 = CALIB40_GLOB.replace("weather_source_calib40d_min_", f"weather_source_calib40d_min_lead{use_lead}_")
        calib = load_calib(cg, exclude=("calib40",))
        calib40 = load_calib(cg40)
        if not calib or not calib40:
            sys.exit(f"Zieltag ist {lead_days} Tage weg, aber keine Min-Lead-{use_lead}-Kalibrierung "
                     f"gefunden ({cg} / {cg40}). Erzeugen mit:\n"
                     f"  python weather_source_compare.py --var min --days 700 --lead {use_lead} "
                     f"--calib-csv preregs/weather_source_calib_min_lead{use_lead}_YYYY_MM_DD.csv\n"
                     f"  python weather_source_compare.py --var min --days 40 --lead {use_lead} "
                     f"--calib-csv preregs/weather_source_calib40d_min_lead{use_lead}_YYYY_MM_DD.csv\n"
                     f"(Notbehelf --force-lead1: nur Orientierung, nie darauf setzen.)")
        print(f"Min-Lead-{use_lead}-Kalibrierung geladen.")
    else:
        calib = load_calib(CALIB_GLOB, exclude=("calib40", "_lead"))
        calib40 = load_calib(CALIB40_GLOB, exclude=("_lead",))
        if not calib:
            sys.exit(f"Keine Kalibrierung unter {CALIB_GLOB} gefunden (im Repo-Root ausfuehren).")
        if not calib40:
            sys.exit(f"Keine 40d-Sommer-Kalibrierung unter {CALIB40_GLOB} gefunden. Erzeugen mit:\n"
                     f"  python weather_source_compare.py --var min --days 40 "
                     f"--calib-csv preregs/weather_source_calib40d_min_YYYY_MM_DD.csv")
        if lead_days > 1:
            print(f"\n!! ACHTUNG (--force-lead1): Zieltag ist {lead_days} Tage weg, gerechnet wird "
                  f"mit der Lead-24h-Kalibrierung — nur Orientierung, nie darauf setzen.\n")

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

    rows = []
    city_info = {}
    for city, mks in sorted(targets.items()):
        icao = STATIONS.get(city)
        st_info = station_info(icao)     # deckt auch Sonderstationen wie HKO ab
        if not st_info:
            print(f"  {city}: keine Station -> skip")
            continue
        if (city, "ensemble_mean") not in calib:
            print(f"  {city}: keine Kalibrierung -> skip")
            continue
        lat, lon = st_info["lat"], st_info["lon"]
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
        raw = {m: raw[m] for m in have}
        _, dropped = robust_mean(raw)   # Drop-Menge (auf Rohwerten, s. build_views)
        spread = max(raw.values()) - min(raw.values())
        has40 = (city, "ensemble_mean") in calib40

        # Debias-vor-Mittelung (17.07.), identische Logik wie im High-Screen.
        views = build_views(city, raw, calib, calib40, spread, dropped)
        if not views:
            print(f"  {city}: <3 kalibrierte Modelle in jeder Familie -> skip")
            continue

        open_mks = [x for x in mks if x["status"] == "open"]
        fav = max(open_mks, key=lambda x: x["buyYes"]) if open_mks else None
        city_info[city] = {"raw": raw, "views": views, "dropped": dropped, "spread": spread,
                           "fav": fav, "mks": mks, "icao": icao}

        for x in open_mks:
            probs = {}
            for cal in (calib, calib40):
                for m in raw:
                    if (city, m) not in cal:
                        continue
                    b = cal[(city, m)][0]
                    s = model_sigma(cal, city, m, spread)  # sigma(s) seit 17.07.
                    probs[m] = max(probs.get(m, 0.0),
                                   bucket_prob(x["kind"], x["k"], raw[m] - b, s))
            pmax_m = max(probs, key=probs.get) if probs else None

            pv = [(bucket_prob(x["kind"], x["k"], mu, s), lbl, s) for lbl, mu, s in views]
            p_use, p_src, sig_use = max(pv)
            d = min(dist_deg(x["kind"], x["k"], mu) for _, mu, _ in views)
            be = 1.0 - x["buyNo"]

            rows.append({
                "city": city, **x,
                "p_ens": p_use, "p_src": p_src,
                "p_max": probs[pmax_m] if pmax_m else 1.0,
                "p_max_src": SHORT[pmax_m] if pmax_m else "?",
                "dist": d, "dist_sig": d / sig_use if sig_use else 0.0,
                "spread": spread, "has40": has40, "be": be, "ev": be - p_use,
            })
        time.sleep(0.5)

    print("\n" + "=" * 112)
    print(f"KANDIDATEN-FILTER (markt-blind bis auf die Profitschwelle): dist>={MIN_DIST}°C | "
          f"Modellspanne<={MAX_SPREAD}°C | jedes Modell P<={MAX_PMODEL:.0%} (700d UND 40d) | buyYes>={MIN_YES:.0%}")
    print("=" * 112)
    cand = [r for r in rows if not reject_reasons(r) and 0 < r["buyNo"] < 1]
    # Wetterfrosch-Doktrin 16.07. (wie High-Screen): sicherste zuerst, markt-blind.
    cand.sort(key=lambda r: (r["p_ens"], -r["dist_sig"]))

    hdr = (f"{'Stadt':13} {'Bucket':>15} {'YES':>5} {'NO':>6} {'Rend%':>6} {'BE':>5} "
           f"{'P_pess':>15} {'EV':>8} {'P_max':>11} {'Span':>6} {'dist':>6}  Markt-ID")
    print(hdr)
    print("-" * len(hdr))
    for r in cand[:18]:
        rend = (1 - r["buyNo"]) / r["buyNo"] * 100
        print(f"{r['city']:13} {r['title']:>15} {r['buyYes']:5.2f} {r['buyNo']:6.3f} {rend:6.1f} "
              f"{r['be']*100:4.1f}% {r['p_ens']*100:6.1f}%/{r['p_src']:<7} {r['ev']*100:+6.1f}pp "
              f"{r['p_max']*100:5.1f}%/{r['p_max_src']:5} {r['spread']:5.1f}° {r['dist']:5.1f}°  {r['marketId']}")
    if not cand:
        print("(keiner — das ist das haeufige Ergebnis, kein Fehler)")

    cand_ids = {r["marketId"] for r in cand}
    near = [r for r in rows if r["marketId"] not in cand_ids and r["buyYes"] >= 0.06
            and 0 < r["buyNo"] < 1 and r["dist"] >= 1.0 and r["ev"] > -0.15]
    near.sort(key=lambda r: r["ev"], reverse=True)
    if near:
        print("\nVERWORFEN (Info, kein Kandidat):")
        for r in near[:10]:
            rend = (1 - r["buyNo"]) / r["buyNo"] * 100
            print(f"  {r['city']:13} {r['title']:>15} NO {r['buyNo']:.3f} ({rend:5.1f} %) "
                  f"EV {r['ev']*100:+6.1f}pp -> {', '.join(reject_reasons(r))}")

    for city in sorted(city_info):
        ci = city_info[city]
        raw_s = "  ".join(f"{SHORT[m]} {v:.1f}{'*' if m in ci['dropped'] else ''}"
                          for m, v in ci["raw"].items())
        flag = "  (* = Ausreisser, aus dem bereinigten Mittel entfernt)" if ci["dropped"] else ""
        veto = "  << SPANNEN-VETO" if ci["spread"] > MAX_SPREAD else ""
        print(f"\n--- {city} ({ci['icao']})  roh: {raw_s}  | Spanne {ci['spread']:.1f}°{veto}{flag} ---")
        print("    korr. Sichten: " + "   ".join(f"{lbl} {mu:.1f}±{s:.1f}" for lbl, mu, s in ci["views"]))
        for x in ci["mks"]:
            if x["buyYes"] < 0.02 and not (ci["fav"] and x["marketId"] == ci["fav"]["marketId"]):
                continue
            p_e = max(bucket_prob(x["kind"], x["k"], mu, s) for _, mu, s in ci["views"])
            mark = " <FAV" if ci["fav"] and x["marketId"] == ci["fav"]["marketId"] else ""
            inc = " *" if x["marketId"] in cand_ids else ""
            print(f"   {x['title']:>15}  YES {x['buyYes']:.2f}  NO {x['buyNo']:.3f}  "
                  f"P_pess {p_e*100:5.1f}%  {x['status']}{mark}{inc}")

    print(f"\n(Stand {datetime.now(timezone.utc).strftime('%d.%m. %H:%M UTC')}; P aus Normal-Annahme, "
          f"Min-Doppel-Kalibrierung 700d + 40d, Lead-{use_lead * 24}h, Debias-vor-Mittelung, "
          f"sigma(s) fuer ENS + Einzelmodelle.)")


if __name__ == "__main__":
    main()
