#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""weather_eps_logger.py — taeglicher Forward-Logger fuer echte EPS-Member
(Backlog Prio 4, Pre-Reg preregs/weather_eps_sigma_prereg_2026_07_18.md).

WARUM EIN LOGGER: Die Machbarkeits-Checks vom 18.07. haben beide Abkuerzungen
geschlossen: (a) `past_days` der Ensemble-API liefert fuer vergangene Tage den
jeweils JUENGSTEN Lauf (Member-SD ~0,3-0,5 vs 0,6-0,9 fuer Zukunftstage) —
das ist Lead ~0-6h, nicht der 24h-Forecast; (b) die Previous-Runs-API kennt
`ecmwf_ifs025_ensemble` nur dem Schema nach, alle Werte null (4 Stichfenster
2024-06 .. 2026-07, mit Retries). Lead-24h-Member-Historie existiert also
nirgends rueckwirkend — sie muss ab jetzt selbst gesammelt werden.

WAS GELOGGT WIRD: je Stadt x EPS-Modell die Member-TAGESMAXIMA des morgigen
lokalen Tages (Zieltag), aus stuendlichen Werten der Ensemble-API. Ein Lauf
pro Tag genuegt (zur Screen-Zeit, ~06-10 UTC — die Pre-Reg vergleicht die
EPS-Verteilung mit der sigma(s)-Normal-P DESSELBEN Zeitpunkts).

Aufruf (idempotent pro Tag — vorhandene (run_day, city, model, target)-Zeilen
werden nicht dupliziert):
  python weather_eps_logger.py            # alle 28 Staedte
  python weather_eps_logger.py --city Seoul,London
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import airportsdata
import requests

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from weather_source_compare import STATIONS  # dieselben 28 Staedte/Stationen

ENS_API = "https://ensemble-api.open-meteo.com/v1/ensemble"
EPS_MODELS = ["ecmwf_ifs025_ensemble", "ncep_gefs025", "icon_seamless_eps"]
LOG = "preregs/weather_eps_log.csv"

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default=None, help="Kommagetrennte Teilmenge, sonst alle")
    ap.add_argument("--log", default=LOG)
    args = ap.parse_args()
    cities = STATIONS if not args.city else {
        c: STATIONS[c] for c in args.city.split(",") if c in STATIONS}
    AP = airportsdata.load("ICAO")
    run_utc = datetime.now(timezone.utc)
    run_day = run_utc.date().isoformat()

    seen = set()
    if os.path.exists(args.log):
        with open(args.log, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add((row["run_day"], row["city"], row["model"], row["target_day"]))

    new_rows = []
    for city, icao in cities.items():
        st = AP.get(icao)
        if not st:
            continue
        try:
            r = S.get(ENS_API, params={
                "latitude": st["lat"], "longitude": st["lon"],
                "hourly": "temperature_2m", "models": ",".join(EPS_MODELS),
                "forecast_days": 3, "timezone": "auto"}, timeout=60)
            r.raise_for_status()
            j = r.json()
        except Exception as ex:
            print(f"  {city}: Fehler {str(ex)[:80]} -> skip")
            continue
        hh = j.get("hourly", {})
        times = hh.get("time", [])
        # Zieltag = morgen in LOKALER Stadt-Zeit (timezone=auto liefert lokale Stempel)
        target = None
        for t in times:
            if t[:10] > times[0][:10]:
                target = t[:10]
                break
        if target is None:
            continue
        # key -> (model, member); Basisreihe "temperature_2m_<model>" = Member 00
        daymax = defaultdict(dict)
        for k, vals in hh.items():
            if not k.startswith("temperature_2m"):
                continue
            for t, v in zip(times, vals):
                if v is None or t[:10] != target:
                    continue
                daymax[k][t[:10]] = max(daymax[k].get(t[:10], v), v)
        per_model = defaultdict(list)
        for k, dd in daymax.items():
            model = next((m for m in EPS_MODELS if m in k), None)
            if model and target in dd:
                per_model[model].append(dd[target])
        for model, members in sorted(per_model.items()):
            key = (run_day, city, model, target)
            if key in seen or len(members) < 5:
                continue
            new_rows.append({
                "run_utc": run_utc.strftime("%Y-%m-%dT%H:%MZ"), "run_day": run_day,
                "city": city, "model": model, "target_day": target,
                "n_member": len(members),
                "members": ";".join(f"{v:.1f}" for v in sorted(members))})
            print(f"  {city:13} {model:22} target {target}  n={len(members)}")
        time.sleep(0.6)

    if new_rows:
        exists = os.path.exists(args.log)
        with open(args.log, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run_utc", "run_day", "city", "model",
                                              "target_day", "n_member", "members"])
            if not exists:
                w.writeheader()
            w.writerows(new_rows)
    print(f"{len(new_rows)} neue Zeilen -> {args.log}")


if __name__ == "__main__":
    main()
