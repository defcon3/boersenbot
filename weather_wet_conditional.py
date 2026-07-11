# -*- coding: utf-8 -*-
"""weather_wet_conditional.py — Forecast-Fehler eines Tagesextrems KONDITIONIERT
auf die vom Modell selbst vorhergesagte Regenmenge (Wetterlagen-Veto fuer Lays).

Entstanden 2026-07-11 (generalisiert aus Scratchpad shanghai_rain_conditional.py):
Der High-Screen lieferte Shanghai "28C" NO @0,94 fuer den 12.07. — formal bestand
er die Doppel-Kalibrierung (P700 1,2 % / P40 2,5 % vs BE 5,6 %), aber alle fuenf
Modelle sahen 6-16 mm Regen + Sturmboeen. Dieser Drilldown (477 Tage ZSPD) zeigte,
dass die Schoenwetter-Kalibrierung an solchen Tagen NICHT gilt:

    Split (fc-Regen ENS)   n    Bias    Sigma   P(err<-1.5C)
    trocken  (<1mm)       344   +1.46   1.11        5,5 %
    nass     (>=5mm)       75   +0.76   1.20       21,3 %
    sehr nass(>=9mm)       41   +0.76   1.40       26,8 %

Der Zu-kuehl-Bias halbiert sich (die pauschale Korrektur hebt mu ~0,6C zu hoch),
Sigma steigt, die kalte Flanke wird 4-6x fetter -> P(28er-Fenster) real 2,5-5 %
statt 1,2 %, EV ~0 -> Lay verworfen (Markt preiste 6-8 %, plausibel korrekt;
08.07. blieb Shanghai nach Regenfront real bei max 26C).

REGEL-KANDIDAT: Lay-Kandidat + Modelle sehen fuer den Zieltag nennenswert Regen
-> diesen Drilldown fuer die Stadt laufen lassen und P gegen den passenden
Nass-Split statt gegen die Gesamt-Kalibrierung halten.

Methodik: Open-Meteo Previous-Runs previous_day1 (Temp + Precip, echter
24h-Lead), IEM-METAR als Ist (wie weather_source_compare.py); err = Ist -
roh-ENS; "korr. err" = err - Gesamt-Bias des Datensatzes (so rechnet auch
weather_outlier_screen*.py, nur mit der CSV-Kalibrierung).

Aufruf:
  python weather_wet_conditional.py --city Shanghai                 # Tageshoch
  python weather_wet_conditional.py --city Paris --var min --days 700
  python weather_wet_conditional.py --city Shanghai --thresholds 1,5,9
"""
import argparse
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

from weather_source_compare import (MODELS, PREVRUN, STATIONS, _airports,
                                    fetch_actual_daily_extreme)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

SUMMER_MONTHS = ("06", "07", "08", "09")


def fetch_temp_precip_day1(icao, days):
    """previous_day1-Stundenwerte Temperatur+Niederschlag je Modell -> Tages-Extrem/-Summe."""
    st = _airports[icao]
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=days)
    for attempt in range(4):
        r = requests.get(PREVRUN, params={
            "latitude": st["lat"], "longitude": st["lon"],
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "hourly": "temperature_2m_previous_day1,precipitation_previous_day1",
            "models": ",".join(MODELS),
            "timezone": "auto",
        }, timeout=60)
        r.raise_for_status()
        j = r.json()
        hourly = j.get("hourly", {})
        times = hourly.get("time", [])
        if times and any(v is not None for k in hourly if k != "time" for v in hourly[k]):
            break
        time.sleep(2 * (attempt + 1))
    return j, start, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, help=f"eine aus: {', '.join(sorted(STATIONS))}")
    ap.add_argument("--var", choices=["max", "min"], default="max")
    ap.add_argument("--days", type=int, default=700)
    ap.add_argument("--thresholds", default="1,5,9",
                    help="Regen-Splits in mm (Default 1,5,9: trocken <t1, Baender, nass >=letzte)")
    args = ap.parse_args()
    icao = STATIONS.get(args.city)
    if not icao or icao not in _airports:
        sys.exit(f"Stadt {args.city!r} nicht in STATIONS/airportsdata.")
    agg = {"max": max, "min": min}[args.var]
    thr = sorted(float(x) for x in args.thresholds.split(","))

    j, start, end = fetch_temp_precip_day1(icao, args.days)
    tz_name = j.get("timezone", "UTC")
    hourly = j.get("hourly", {})
    times = hourly.get("time", [])

    text = {m: {} for m in MODELS}   # Tages-Extrem Temperatur je Modell
    psum = {m: {} for m in MODELS}   # Tages-Regensumme je Modell
    for m in MODELS:
        for t, v in zip(times, hourly.get(f"temperature_2m_previous_day1_{m}", [])):
            if v is None:
                continue
            d = t[:10]
            text[m][d] = agg(text[m].get(d, v), v)
        for t, v in zip(times, hourly.get(f"precipitation_previous_day1_{m}", [])):
            if v is None:
                continue
            d = t[:10]
            psum[m][d] = psum[m].get(d, 0.0) + v

    days_t = set.intersection(*(set(text[m]) for m in MODELS))
    days_p = set.intersection(*(set(psum[m]) for m in MODELS))
    ens_t = {d: sum(text[m][d] for m in MODELS) / len(MODELS) for d in days_t}
    ens_p = {d: sum(psum[m][d] for m in MODELS) / len(MODELS) for d in days_p}
    actual = fetch_actual_daily_extreme(icao, start, end, tz_name, agg)

    rows = [(d, actual[d] - ens_t[d], ens_p[d])
            for d in sorted(days_t & days_p & set(actual))]
    if len(rows) < 30:
        sys.exit(f"Nur {len(rows)} auswertbare Tage — zu duenn.")
    bias_all = sum(e for _, e, _ in rows) / len(rows)
    print(f"{args.city} ({icao}) Tages{'hoch' if args.var == 'max' else 'tief'}, "
          f"{len(rows)} Tage, Gesamt-Bias (Ist-ENS) {bias_all:+.2f}C "
          f"(Kalibrier-Konvention: {-bias_all:+.2f})")

    def stats(sub, label):
        k = len(sub)
        if k < 15:
            print(f"{label:26} n={k:3d}  (zu duenn)")
            return
        b = sum(e for e, _ in sub) / k
        sig = (sum((e - b) ** 2 for e, _ in sub) / (k - 1)) ** 0.5
        corr = [e - bias_all for e, _ in sub]
        cold15 = sum(1 for c in corr if c <= -1.5) / k
        cold25 = sum(1 for c in corr if c <= -2.5) / k
        warm15 = sum(1 for c in corr if c >= 1.5) / k
        warm25 = sum(1 for c in corr if c >= 2.5) / k
        print(f"{label:26} n={k:3d}  Bias {b:+.2f}  Sigma {sig:.2f}  "
              f"P(korr.err<=-1.5/-2.5) {cold15*100:4.1f}/{cold25*100:4.1f}%  "
              f"P(>=+1.5/+2.5) {warm15*100:4.1f}/{warm25*100:4.1f}%")

    def splits(pairs, prefix=""):
        stats(pairs, prefix + "alle")
        stats([(e, p) for e, p in pairs if p < thr[0]], prefix + f"trocken (<{thr[0]:g}mm)")
        for lo, hi in zip(thr, thr[1:]):
            stats([(e, p) for e, p in pairs if lo <= p < hi], prefix + f"{lo:g}-{hi:g}mm")
        stats([(e, p) for e, p in pairs if p >= thr[-1]], prefix + f"nass (>={thr[-1]:g}mm)")

    print("\nSplit nach ENS-Forecast-Regensumme (previous_day1):")
    splits([(e, p) for _, e, p in rows])
    print("\nNur Sommer (Jun-Sep):")
    splits([(e, p) for d, e, p in rows if d[5:7] in SUMMER_MONTHS], "Sommer ")

    print("\nDie 12 nassesten Tage (Forecast-Regen) und was real geschah:")
    for d, e, p in sorted(rows, key=lambda x: -x[2])[:12]:
        print(f"  {d}  fc-Regen {p:5.1f}mm  Ist-ENS {e:+.2f}C  (korr. err {e - bias_all:+.2f})")


if __name__ == "__main__":
    main()
