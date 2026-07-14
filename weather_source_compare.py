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
  python weather_source_compare.py --var min --days 700 --city Shanghai,Seoul,London,Paris \
      --calib-csv preregs/weather_source_calib_min_YYYY_MM_DD.csv
  python weather_source_compare.py --var min --days 700 --city London --lead 2

--lead N kalibriert previous_dayN statt previous_day1 (Lead N*24h). Noetig
geworden am 11.07.2026: "Lowest London July 13" wurde ~31h vor dem Tief
gehandelt; der 48h-Check (lead 2) zeigte London-Min sigma 0,79->0,90 (700d)
bzw. 0,59->0,72 (40d) -- moderat, Trade blieb tragfaehig.

--var min kalibriert das TAGESTIEF statt des Tageshochs (fuer die "Lowest
temperature"-Maerkte / weather_outlier_screen_low.py). Noetig geworden am
10.07.2026: die High-Kalibrierung lag auf Shanghai-Minima ~3-4C zu warm
(Beinahe-Fehltrade "Lowest 24C" -- Ist-Minimum lag exakt im Bucket, waehrend
die high-basierte Rechnung 0,1 % Risiko behauptete).
"""

import argparse
import math
import sys
import time
from datetime import datetime, timedelta, timezone

import airportsdata
import numpy as np
import requests
from scipy.optimize import minimize_scalar

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
    "Sao Paulo": "SBGR", "Taipei": "RCSS", "Tel Aviv": "LLBG",
    "Toronto": "CYYZ", "Wuhan": "ZHHH", "Panama City": "MPMG",
}

MODELS = ["gfs_seamless", "icon_seamless", "ukmo_seamless", "jma_seamless", "ecmwf_ifs025"]
MODEL_LABEL = {
    "gfs_seamless": "GFS (NOAA)", "icon_seamless": "ICON (DWD)",
    "ukmo_seamless": "UKMO (UK Met Office)", "jma_seamless": "JMA (Japan)",
    "ecmwf_ifs025": "ECMWF", "ensemble_mean": "Ensemble-Mittel (5 Modelle)",
}
ALL_SOURCES = MODELS + ["ensemble_mean"]

PREVRUN = "https://previous-runs-api.open-meteo.com/v1/forecast"
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

_airports = airportsdata.load("ICAO")


def fetch_model_daily_extreme(icao, lat, lon, start, end, agg, retries=4, lead=1):
    """Pro Modell: taegliches Extrem (agg = max fuer Tageshoch, min fuer Tagestief)
    aus den previous_day{lead}-Stundenwerten (lokale Zeit), Lead = lead*24h.
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
            "hourly": f"temperature_2m_previous_day{lead}",
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
        key = f"temperature_2m_previous_day{lead}_{m}"
        vals = hourly.get(key, [])
        for t, v in zip(times, vals):
            if v is None:
                continue
            day = t[:10]
            per_model_per_day[m].setdefault(day, []).append(v)
    daily = {m: {d: agg(vs) for d, vs in days.items()} for m, days in per_model_per_day.items()}
    # Ensemble-Mittel als 6. "Quelle": gemittelt nur an Tagen, an denen ALLE 5 Modelle
    # einen Wert liefern (fairer Vergleich, kein Bias durch Teilausfaelle einzelner Modelle).
    all_days = set.intersection(*(set(d.keys()) for d in daily.values())) if daily else set()
    daily["ensemble_mean"] = {d: sum(daily[m][d] for m in MODELS) / len(MODELS) for d in all_days}
    return daily, tz_name


def fetch_actual_daily_extreme(icao, start, end, tz_name, agg):
    """Echtes Tagesextrem (agg wie oben) aus IEM-METAR-Archiv, in DERSELBEN lokalen Zeitzone wie die
    Modell-Tagesgrenzen (tz_name von Open-Meteo) -- sonst Fehlausrichtung bei Staedten
    fern von UTC (z. B. Seoul +9h -- Ursache der -4C-Ausreisser vor diesem Fix)."""
    r = requests.get(IEM, params={
        "station": icao, "data": "tmpc",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "tz": tz_name, "format": "onlycomma", "latlon": "no", "elev": "no",
        "missing": "M", "trace": "T", "direct": "no",
        # 3+4 = Routine-METARs + SPECI-Zwischenmeldungen; nur 3 verpasst Extrema
        # zwischen den vollen Stunden (Fix 12.07., analog weather_ladder_logger).
        "report_type": [3, 4],
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
        daily_max[day] = agg(daily_max.get(day, t), t)
    return daily_max


# Sigma ist nicht konstant, sondern haengt von der Tagesspanne der 5 Modelle ab:
#   sigma_city(s) = a_city + b * s
# Belegt in preregs/weather_spread_sigma_2026_07_14.md (15.371 Stadt-Tage, 28
# Staedte, Block-Bootstrap t = 21,4; empirisches Sigma steigt monoton 0,93 -> 1,73).
# Das feste Sigma ist ein Mittel ueber ALLE Spannen und deshalb auf RUHIGEN Tagen
# viel zu breit (Sommer-OOS: sagt 3,3 %, liefert 1,4 %) -- genau das korrigiert a+b*s.
#
# b wird je Kalibrierfenster NEU gefittet und mit in die CSV geschrieben, statt als
# Konstante im Code zu leben: Die Steigung ist saisonabhaengig (gemessen 14.07.:
# Sommer 0,107 / Winter 0,177 / Ganzjahr 0,140). Wuerde man dem 40d-Sommerfenster
# die Ganzjahres-Steigung aufzwingen, kaeme sigma auf ruhigen Sommertagen ~5 % zu
# ENG heraus -- also in die gefaehrliche Richtung.
SIGMA_FLOOR = 0.3


def _fit_sigma_model(pairs_by_city, floor=SIGMA_FLOOR):
    """MLE fuer sigma = a_city + b*spread (Gauss, zentrierte Residuen).
    b gemeinsam ueber alle Staedte dieses Fensters, a je Stadt.
    Rueckgabe: (b, {city: a})."""
    cities = sorted(pairs_by_city)
    R = {c: np.array([p[0] for p in pairs_by_city[c]]) for c in cities}
    S = {c: np.array([p[1] for p in pairs_by_city[c]]) for c in cities}

    def a_for(c, b):
        r, s = R[c], S[c]

        def nll(a):
            sig = np.maximum(a + b * s, floor)
            return float(np.sum(np.log(sig) + r ** 2 / (2.0 * sig ** 2)))
        return float(minimize_scalar(nll, bounds=(0.01, 6.0), method="bounded").x)

    def nll_b(b):
        tot = 0.0
        for c in cities:
            sig = np.maximum(a_for(c, b) + b * S[c], floor)
            tot += float(np.sum(np.log(sig) + R[c] ** 2 / (2.0 * sig ** 2)))
        return tot

    b = float(minimize_scalar(nll_b, bounds=(-0.05, 0.60), method="bounded").x)
    return b, {c: a_for(c, b) for c in cities}


def analyze_city(city, icao, days, agg, lead=1):
    station = _airports.get(icao)
    if not station:
        print(f"{city} ({icao}): Station nicht in airportsdata gefunden -- uebersprungen.")
        return None
    lat, lon = station["lat"], station["lon"]

    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=days)

    model_days, tz_name = fetch_model_daily_extreme(icao, lat, lon, start.isoformat(), end.isoformat(), agg, lead=lead)
    actual_days = fetch_actual_daily_extreme(icao, start, end, tz_name, agg)

    # Tagesspanne der 5 Modelle -- nur an Tagen, an denen ALLE fuenf liefern
    # (dieselbe Menge, auf der auch das Ensemble-Mittel gebildet wird).
    spread_by_day = {}
    for day in model_days.get("ensemble_mean", {}):
        vals = [model_days[m][day] for m in MODELS if day in model_days.get(m, {})]
        if len(vals) == len(MODELS):
            spread_by_day[day] = max(vals) - min(vals)

    results = {}
    for m in ALL_SOURCES:
        diffs, with_spread = [], []
        for day, fc in model_days.get(m, {}).items():
            act = actual_days.get(day)
            if act is None:
                continue
            diffs.append(fc - act)
            if day in spread_by_day:
                with_spread.append((fc - act, spread_by_day[day]))
        if diffs:
            n = len(diffs)
            bias = sum(diffs) / n
            mae = sum(abs(d) for d in diffs) / n
            var = sum((d - bias) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
            results[m] = {"n": n, "bias": bias, "mae": mae, "sigma": var ** 0.5,
                          # zentrierte Residuen + Tagesspanne -- Rohstoff fuer den
                          # sigma(s)-Fit, der in main() ueber ALLE Staedte gepoolt wird
                          "pairs": [(d - bias, s) for d, s in with_spread]}
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
    ap.add_argument("--var", choices=["max", "min"], default="max",
                    help="max = Tageshoch (default), min = Tagestief (fuer Lowest-Maerkte)")
    ap.add_argument("--lead", type=int, default=1, choices=range(1, 8), metavar="N",
                    help="previous_dayN = Lead N*24h (default 1; 2 fuer Maerkte, die >24h "
                         "vor dem Extremum gehandelt werden)")
    args = ap.parse_args()
    agg_fn = {"max": max, "min": min}[args.var]
    print(f"Zielgroesse: Tages{'hoch' if args.var == 'max' else 'tief'} "
          f"(previous_day{args.lead}, Lead {args.lead * 24}h)")

    cities = STATIONS if not args.city else {c: STATIONS[c] for c in args.city.split(",") if c in STATIONS}

    agg = {m: {"n": 0, "bias_sum": 0.0, "mae_sum": 0.0} for m in ALL_SOURCES}
    calib_rows = []
    ens_pairs = {}   # city -> [(zentriertes Residuum, Tagesspanne), ...] fuer den sigma(s)-Fit

    for city, icao in cities.items():
        print(f"\n=== {city} ({icao}) ===")
        try:
            res = analyze_city(city, icao, args.days, agg_fn, lead=args.lead)
        except Exception as ex:
            # Netzabbruch bei EINER Stadt darf nicht den ganzen Lauf killen (14.07.:
            # ConnectionReset bei Munich hat eine 28-Staedte-Kalibrierung verworfen).
            print(f"  Netz-/Datenfehler: {str(ex)[:90]} -> Stadt uebersprungen")
            continue
        if not res:
            continue
        for m in ALL_SOURCES:
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
        ens_pairs[city] = res["ensemble_mean"]["pairs"] if "ensemble_mean" in res else []
        time.sleep(1)  # freundlich zu den freien APIs

    # sigma(s) = a_city + b*s -- b gemeinsam ueber alle Staedte DIESES Fensters
    # gefittet (saisonabhaengig, siehe Kopf), a je Stadt. Nur fuer das Ensemble:
    # die Screens benutzen sigma(s) ausschliesslich fuer die ENS-Sicht, die
    # Einzelmodelle behalten ihr festes Sigma als Veto-Grundlage.
    usable = {c: p for c, p in ens_pairs.items() if len(p) >= 25}
    b_slope, a_city = (None, {})
    if len(usable) >= 3:
        b_slope, a_city = _fit_sigma_model(usable)
        print(f"\n=== sigma(s) = a_city + b*Spanne ===")
        print(f"  b (gemeinsam, aus diesem Fenster gefittet) = {b_slope:+.3f}")
        print(f"  a_city: min {min(a_city.values()):.2f}  median "
              f"{sorted(a_city.values())[len(a_city)//2]:.2f}  max {max(a_city.values()):.2f}")
    else:
        print("\n  (zu wenige Staedte mit Spannen-Daten -> kein sigma(s); "
              "Verbraucher fallen auf das feste Sigma zurueck)")

    if args.calib_csv:
        with open(args.calib_csv, "w", encoding="utf-8") as f:
            # a/b: sigma(s) = a + b*Spanne (nur ensemble_mean). Leer -> Verbraucher
            # faellt auf die Spalte sigma (fest) zurueck.
            f.write("city,model,n,bias,sigma,a,b\n")
            for city, m, n, bias, sigma in calib_rows:
                a = a_city.get(city) if m == "ensemble_mean" else None
                bb = b_slope if a is not None else None
                f.write(f"{city},{m},{n},{bias:.3f},{sigma:.3f},"
                        f"{'' if a is None else format(a, '.3f')},"
                        f"{'' if bb is None else format(bb, '.4f')}\n")
        print(f"\nKalibrierungs-CSV geschrieben: {args.calib_csv}")

    print("\n=== GESAMT (alle Staedte gepoolt) ===")
    ranked = []
    for m in ALL_SOURCES:
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
