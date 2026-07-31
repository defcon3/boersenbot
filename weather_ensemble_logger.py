# -*- coding: utf-8 -*-
"""weather_ensemble_logger.py — Forward-Test-Logger zur Pre-Reg
`preregs/weather_gfs_ensemble_mu_2026_07_31.md` (Commit 747d41b).

## Warum es diesen Logger gibt

H1 fragt, ob der Median der 31 GFS-Member ein robusteres mu liefert als das
Mittel der fuenf Punktmodelle — besonders an zerklueffteten Tagen, die der
Spannen-Veto heute sperrt. Rueckwirkend ist das **nicht** pruefbar: fuer
Ensemble-Laeufe existiert kein Archiv (`previous_dayN` liefert auf dem
Ensemble-Endpoint leere Spalten, `historical-forecast-api/v1/ensemble` und
`previous-runs-api/v1/ensemble` gibt es nicht). Also wird vorwaerts gemessen.

Der Logger schreibt **beide Seiten im selben Lauf** — waeren die Punktmodelle
spaeter aus dem Archiv nachgeladen, haetten sie einen anderen Modelllauf und
damit einen unfairen Vorteil oder Nachteil.

Der Ist-Wert wird hier NICHT geschrieben. Er kommt beim Auswerten aus der
settelnden Quelle (WU-Tabelle, `fetch_actual_daily_extreme_wu`) — dieselbe, auf
die der Markt aufloest.

## Betrieb

Einmal taeglich, moeglichst zur **gleichen Uhrzeit**. Der tatsaechliche Vorlauf
wird je Zeile als `lead_h` mitgeschrieben.

**Der Vorlauf ist NICHT ueber die Staedte hinweg gleich** — gemessen 16 bis 38 h
bei einem einzigen Lauf. Grund: Zieltag ist „morgen in der lokalen Zeit der
Stadt", und das ist von einem festen Laufzeitpunkt aus je nach Zeitzone
unterschiedlich weit weg (Ostasien hat schon den uebernaechsten Kalendertag).
Die lokale Definition ist zwingend, weil die Bretter auf lokale Kalendertage
aufloesen — die Streuung ist also inhaerent, nicht reparierbar. **Das Eval muss
auf ein `lead_h`-Band filtern**, sonst vergleicht es 16-h- mit 38-h-Prognosen.

    python weather_ensemble_logger.py            # Zieltag = morgen (lokal je Stadt)
    python weather_ensemble_logger.py --dry-run  # nichts schreiben, nur zeigen

Idempotent: eine (Stadt, Zieltag)-Kombination wird nie doppelt geschrieben.
~31 Aufrufe je Lauf und Endpoint, weit innerhalb der freien Open-Meteo-Stufe.
"""
import argparse
import csv
import datetime as dt
import os
import statistics
import sys
import time
from zoneinfo import ZoneInfo

import requests

from weather_source_compare import STATIONS, MODELS
# station_info statt airportsdata direkt: nur so faellt Hong Kong nicht raus —
# HKO ist eine Pseudo-Station (Hong Kong Observatory) und steht in keinem
# Flughafen-Datensatz, ist aber eine handelbare Stadt.
from weather_stations import station_info

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

ENS_API  = "https://ensemble-api.open-meteo.com/v1/ensemble"
FC_API   = "https://api.open-meteo.com/v1/forecast"
ENS_MODEL = "gfs025"
OUT      = "weather_ensemble_log.csv"

FIELDS = [
    "run_utc", "city", "icao", "target_day", "tz", "lead_h",
    "mu5", "spread5", "m_gfs", "m_icon", "m_ukmo", "m_jma", "m_ecmwf",
    "ens_n", "ens_median", "ens_mean", "ens_sd", "ens_min", "ens_max",
    "ens_members",
]


def get(url, params, tries=4, timeout=60):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException:
            time.sleep(2 * (i + 1)); continue
        if r.status_code == 429:
            time.sleep(5 * (i + 1)); continue
        if r.status_code >= 500:
            time.sleep(2 * (i + 1)); continue
        r.raise_for_status()
        j = r.json()
        h = j.get("hourly", {})
        # Die Previous-Runs-/Ensemble-Backends liefern gelegentlich HTTP 200 mit
        # leeren Spalten (bekannter Cache-Effekt) — das ist KEIN gueltiges Ergebnis.
        if h.get("time") and any(v is not None for k in h if k != "time" for v in h[k]):
            return j
        time.sleep(2 * (i + 1))
    return None


def day_max(times, values, target_day):
    """Maximum der Stundenwerte, die auf den Zieltag fallen (lokale Zeit der API)."""
    vals = [v for t, v in zip(times, values) if v is not None and t[:10] == target_day]
    return max(vals) if vals else None


def fetch_points(lat, lon):
    """Rohe Stundenreihen der fuenf Punktmodelle + aufgeloeste Zeitzone.

    Bewusst EIN Aufruf je Stadt: der Zieltag wird erst aus der zurueckgegebenen
    Zeitzone bestimmt, nicht vorher geraten.
    """
    j = get(FC_API, {"latitude": lat, "longitude": lon, "hourly": "temperature_2m",
                     "models": ",".join(MODELS), "timezone": "auto", "forecast_days": 3})
    if not j:
        return None, None
    return j["hourly"], j.get("timezone", "UTC")


def points_for_day(hourly, target_day):
    out = {}
    for m in MODELS:
        v = day_max(hourly["time"], hourly.get(f"temperature_2m_{m}", []), target_day)
        if v is not None:
            out[m] = v
    return out


def fetch_members(lat, lon, target_day):
    """Tagesmax je Ensemble-Member fuer den Zieltag, aus dem aktuellen Lauf."""
    j = get(ENS_API, {"latitude": lat, "longitude": lon, "hourly": "temperature_2m",
                      "models": ENS_MODEL, "timezone": "auto", "forecast_days": 3})
    if not j:
        return []
    h = j["hourly"]
    vals = []
    for k in h:
        if "member" not in k:
            continue
        v = day_max(h["time"], h[k], target_day)
        if v is not None:
            vals.append(v)
    return vals


def existing_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["city"], r["target_day"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    run_utc = dt.datetime.now(dt.timezone.utc)
    done = existing_keys(args.out)
    new_file = not os.path.exists(args.out)
    rows = []

    for city, icao in STATIONS.items():
        try:
            info = station_info(icao)
        except Exception as e:
            print(f"  {city:14s} uebersprungen ({icao}: {type(e).__name__})")
            continue
        if not info:
            print(f"  {city:14s} uebersprungen ({icao} unbekannt)")
            continue
        lat, lon = info["lat"], info["lon"]

        hourly, tz = fetch_points(lat, lon)
        if hourly is None:
            print(f"  {city:14s} keine Modelldaten — uebersprungen")
            continue
        tzname = tz or "UTC"
        # Zieltag = morgen in der LOKALEN Zeit der Stadt, nicht in UTC.
        local_now = run_utc.astimezone(ZoneInfo(tzname))
        target = (local_now.date() + dt.timedelta(days=1)).isoformat()

        if (city, target) in done:
            print(f"  {city:14s} {target} schon geloggt")
            continue

        pts = points_for_day(hourly, target)
        mem = fetch_members(lat, lon, target)
        if len(pts) < len(MODELS) or len(mem) < 10:
            print(f"  {city:14s} unvollstaendig (Modelle {len(pts)}/{len(MODELS)}, "
                  f"Member {len(mem)}) — Zeile verworfen")
            continue

        vals5 = [pts[m] for m in MODELS]
        mu5 = sum(vals5) / len(vals5)
        # Vorlauf bis zur ueblichen Zeit des Tagesmaximums (~15 h lokal).
        peak = dt.datetime.fromisoformat(f"{target}T15:00").replace(tzinfo=ZoneInfo(tzname))
        lead_h = round((peak - run_utc).total_seconds() / 3600, 1)

        rows.append({
            "run_utc": run_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "city": city, "icao": icao, "target_day": target, "tz": tzname,
            "lead_h": lead_h,
            "mu5": round(mu5, 3), "spread5": round(max(vals5) - min(vals5), 3),
            "m_gfs": pts[MODELS[0]], "m_icon": pts[MODELS[1]], "m_ukmo": pts[MODELS[2]],
            "m_jma": pts[MODELS[3]], "m_ecmwf": pts[MODELS[4]],
            "ens_n": len(mem),
            "ens_median": round(statistics.median(mem), 3),
            "ens_mean": round(statistics.fmean(mem), 3),
            "ens_sd": round(statistics.stdev(mem), 3) if len(mem) > 1 else 0.0,
            "ens_min": round(min(mem), 3), "ens_max": round(max(mem), 3),
            # Rohwerte mitschreiben: sonst sind spaeter keine anderen Statistiken
            # als die hier vorberechneten mehr rechenbar.
            "ens_members": ";".join(f"{v:.2f}" for v in sorted(mem)),
        })
        print(f"  {city:14s} {target}  mu5={mu5:5.2f} (Spanne {max(vals5)-min(vals5):4.2f})  "
              f"ens_med={statistics.median(mem):5.2f} sd={statistics.stdev(mem):4.2f} "
              f"n={len(mem)}  Lead {lead_h:.0f} h")
        time.sleep(0.5)

    if args.dry_run:
        print(f"\n--dry-run: {len(rows)} Zeilen NICHT geschrieben")
        return 0
    if not rows:
        print("\nnichts Neues zu schreiben")
        return 0
    with open(args.out, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} Zeilen an {args.out} angehaengt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
