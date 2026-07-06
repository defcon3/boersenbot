#!/usr/bin/env python3
"""
weather_source_compare.py — Welche Wettervorhersagequelle prognostiziert das
Tageshoch am besten? (Session 2026-07-06, Wiedereinstieg nach der falsifizierten
Open-Meteo-Bias-These — diesmal Quellenvergleich statt Einzelquelle.)

Vergleicht mehrere NWP-Modelle (alle ueber Open-Meteos kostenlose, keylose
Previous-Runs-API, die den ORIGINAL-Forecast mit fixer Lead-Time archiviert,
NICHT nachtraeglich rekonstruiert wie die normale Historical-Forecast-API)
gegen die tatsaechliche METAR-Messung (IEM ASOS-Archiv) je Stadt.

Modelle: GFS (NOAA/USA), ICON (DWD/Deutschland), UKMO (UK Met Office),
JMA (Japan), ECMWF (Europa) — 5 unabhaengige nationale/internationale
Wetterdienste, alle ueber denselben Open-Meteo-Endpunkt, Lead-Time exakt
24h (temperature_2m_previous_day1).

Metrik je Stadt x Modell: Bias (Forecast - Ist) und MAE des Tageshochs,
taegliche Aggregation aus stuendlichen previous_day1-Werten in lokaler Zeit.

Staedte + Stations-ICAOs wiederverwendet aus weather_latency_logger.py
(STATION_FALLBACK + CITIES), Koordinaten via airportsdata (offline, keine API).

Aufruf:
  python weather_source_compare.py --days 20
  python weather_source_compare.py --days 20 --city Wellington,London
"""

import argparse
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

# Alle 22 Jupiter-Wetter-Markt-Staedte (weather_latency_logger.CITIES), Stationen per
# resolve_station() aufgeloest (2026-07-06). Hong Kong bleibt unaufloesbar (wie schon
# beim Latenz-Logger vermerkt) -- hier bewusst weggelassen statt mit None zu arbeiten.
STATIONS = {
    "Wellington": "NZWN", "Tokyo": "RJTT", "Seoul": "RKSI", "Shanghai": "ZSPD",
    "Beijing": "ZBAA", "Kuala Lumpur": "WMKK", "Shenzhen": "ZGSZ", "Chengdu": "ZUUU",
    "Karachi": "OPKC", "Jeddah": "OEJN", "Ankara": "LTAC", "Helsinki": "EFHK",
    "London": "EGLC", "Paris": "LFPB", "Madrid": "LEMD", "Milan": "LIMC",
    "Munich": "EDDM", "Amsterdam": "EHAM", "Warsaw": "EPWA", "Cape Town": "FACT",
    "Mexico City": "MMMX", "Buenos Aires": "SAEZ",
}

MODELS = ["gfs_seamless", "icon_seamless", "ukmo_seamless", "jma_seamless", "ecmwf_ifs025"]
MODEL_LABEL = {
    "gfs_seamless": "GFS (NOAA)", "icon_seamless": "ICON (DWD)",
    "ukmo_seamless": "UKMO (UK Met Office)", "jma_seamless": "JMA (Japan)",
    "ecmwf_ifs025": "ECMWF",
}

PREVRUN = "https://previous-runs-api.open-meteo.com/v1/forecast"
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

_airports = airportsdata.load("ICAO")


def fetch_model_daily_max(icao, lat, lon, start, end, retries=4):
    """Pro Modell: taegliches Hoch aus den previous_day1-Stundenwerten (lokale Zeit).
    Gibt zusaetzlich die von Open-Meteo aufgeloeste IANA-Zeitzone zurueck, damit der
    METAR-Abgleich dieselben Tagesgrenzen verwendet (sonst Fehlausrichtung bei Staedten
    fern von UTC, z. B. Seoul +9h -- entdeckt 2026-07-06 an Seoul/JMA-UKMO-Ausreissern).

    RETRY: die kostenlose Previous-Runs-API liefert bei IDENTISCHEN Parametern manchmal
    eine leere Antwort zurueck (Backend-Cache-Inkonsistenz, 2026-07-06 beobachtet) --
    ein zweiter Versuch bringt fast immer die vollen Daten."""
    for attempt in range(retries):
        r = requests.get(PREVRUN, params={
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "hourly": "temperature_2m_previous_day1",
            "models": ",".join(MODELS),
            "timezone": "auto",
        }, timeout=30)
        r.raise_for_status()
        j = r.json()
        hourly = j.get("hourly", {})
        times = hourly.get("time", [])
        if times and any(v is not None for k in hourly if k != "time" for v in hourly[k]):
            break
        time.sleep(2 * (attempt + 1))
    tz_name = j.get("timezone", "UTC")
    per_model_per_day = {m: {} for m in MODELS}
    for m in MODELS:
        key = f"temperature_2m_previous_day1_{m}"
        vals = hourly.get(key, [])
        for t, v in zip(times, vals):
            if v is None:
                continue
            day = t[:10]
            per_model_per_day[m].setdefault(day, []).append(v)
    return {m: {d: max(vs) for d, vs in days.items()} for m, days in per_model_per_day.items()}, tz_name


def fetch_actual_daily_max(icao, start, end, tz_name):
    """Echtes Tageshoch aus IEM-METAR-Archiv, in DERSELBEN lokalen Zeitzone wie die
    Modell-Tagesgrenzen (tz_name von Open-Meteo) -- sonst Fehlausrichtung bei Staedten
    fern von UTC (z. B. Seoul +9h -- Ursache der -4C-Ausreisser vor diesem Fix)."""
    r = requests.get(IEM, params={
        "station": icao, "data": "tmpc",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": tz_name, "format": "onlycomma", "latlon": "no", "elev": "no",
        "missing": "M", "trace": "T", "direct": "no", "report_type": 3,
    }, timeout=30)
    r.raise_for_status()
    daily_max = {}
    for line in r.text.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        _, valid, tmpc = parts
        if tmpc == "M":
            continue
        day = valid[:10]
        try:
            t = float(tmpc)
        except ValueError:
            continue
        daily_max[day] = max(daily_max.get(day, t), t)
    return daily_max


def analyze_city(city, icao, days):
    station = _airports.get(icao)
    if not station:
        print(f"{city} ({icao}): Station nicht in airportsdata gefunden -- uebersprungen.")
        return None
    lat, lon = station["lat"], station["lon"]

    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=days)

    model_days, tz_name = fetch_model_daily_max(icao, lat, lon, start.isoformat(), end.isoformat())
    actual_days = fetch_actual_daily_max(icao, start, end, tz_name)

    results = {}
    for m in MODELS:
        diffs = []
        for day, fc in model_days.get(m, {}).items():
            act = actual_days.get(day)
            if act is None:
                continue
            diffs.append(fc - act)
        if diffs:
            n = len(diffs)
            bias = sum(diffs) / n
            mae = sum(abs(d) for d in diffs) / n
            var = sum((d - bias) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
            results[m] = {"n": n, "bias": bias, "mae": mae, "sigma": var ** 0.5}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=700,
                    help="Rueckblick in Tagen (default 700 -- Previous-Runs-Archiv "
                         "reicht ca. bis Anfang 2024 zurueck, laenger liefert nichts mehr)")
    ap.add_argument("--city", default=None, help="Kommagetrennte Teilmenge, sonst alle STATIONS")
    ap.add_argument("--calib-csv", default=None,
                    help="Pfad: schreibt Stadt,Modell,n,bias,sigma je Stadt/Modell (Kalibrierungs-"
                         "Grundlage fuer eine Wahrscheinlichkeits-Pre-Reg, z.B. preregs/)")
    args = ap.parse_args()

    cities = STATIONS if not args.city else {c: STATIONS[c] for c in args.city.split(",") if c in STATIONS}

    agg = {m: {"n": 0, "bias_sum": 0.0, "mae_sum": 0.0} for m in MODELS}
    calib_rows = []

    for city, icao in cities.items():
        print(f"\n=== {city} ({icao}) ===")
        res = analyze_city(city, icao, args.days)
        if not res:
            continue
        for m in MODELS:
            r = res.get(m)
            if not r:
                print(f"  {MODEL_LABEL[m]:24} keine Daten")
                continue
            print(f"  {MODEL_LABEL[m]:24} n={r['n']:3d}  Bias={r['bias']:+.2f}C  "
                 f"MAE={r['mae']:.2f}C  Sigma={r['sigma']:.2f}C")
            agg[m]["n"] += r["n"]
            agg[m]["bias_sum"] += r["bias"] * r["n"]
            agg[m]["mae_sum"] += r["mae"] * r["n"]
            calib_rows.append((city, m, r["n"], r["bias"], r["sigma"]))
        time.sleep(1)  # freundlich zu den freien APIs

    if args.calib_csv:
        with open(args.calib_csv, "w", encoding="utf-8") as f:
            f.write("city,model,n,bias,sigma\n")
            for city, m, n, bias, sigma in calib_rows:
                f.write(f"{city},{m},{n},{bias:.3f},{sigma:.3f}\n")
        print(f"\nKalibrierungs-CSV geschrieben: {args.calib_csv}")

    print("\n=== GESAMT (alle Staedte gepoolt) ===")
    ranked = []
    for m in MODELS:
        n = agg[m]["n"]
        if n == 0:
            continue
        bias = agg[m]["bias_sum"] / n
        mae = agg[m]["mae_sum"] / n
        ranked.append((mae, m, n, bias))
    ranked.sort()
    for mae, m, n, bias in ranked:
        print(f"  {MODEL_LABEL[m]:24} n={n:3d}  Bias={bias:+.2f}C  MAE={mae:.2f}C")
    if ranked:
        best = ranked[0]
        print(f"\nBeste Quelle nach MAE: {MODEL_LABEL[best[1]]} ({best[0]:.2f}C)")


if __name__ == "__main__":
    main()
