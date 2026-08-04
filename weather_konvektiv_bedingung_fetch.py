#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""weather_konvektiv_bedingung_fetch.py — zieht die Konvektionsbedingung je
Stadttag fuer `preregs/weather_konvektiv_sigma_2026_08_03.md`.

Die Bedingung muss HANDELBAR sein, also am Vortag feststehen. Sie kommt deshalb
aus der Open-Meteo Previous-Runs-API, Feld `*_previous_day1` — das ist der Lauf
von vor 24 h, derselbe Stand, auf dem der Autobuy entscheidet. Definition
identisch zur Warm-Bias-Pre-Reg vom 03.08., bewusst NICHT neu justiert:

    regen_mm   = Summe Niederschlag   06-18 h LOKAL
    wolken_tag = Mittel Bewoelkung    09-18 h LOKAL
    konvektiv  = regen_mm >= 1 UND wolken_tag >= 60

ARCHIVTIEFE: die previous_day1-Felder reichen bis ~Juni 2024 zurueck (geprueft
03.08.: 2024-06-01 liefert, 2024-01-01 nicht). Das Fenster des Residuen-Dumps
(ab 2024-09-01) liegt vollstaendig darin — aber knapp. Ein laengeres Fenster ist
mit dieser Quelle nicht zu haben.

CHUNKING nach Jahresscheiben: 700 Tage x 24 h x 2 Felder in einer Antwort ist
gross genug, dass Abbrueche mitten im Stream vorkommen (die IncompleteRead-Falle
aus dem Residuen-Zug vom 03.08.). Kleinere Scheiben + Wiederholung sind billiger
als ein Neustart.

Schreibt: preregs/weather_konvektiv_sigma_bedingung_2026_08_04.csv.gz

Aufruf:
  python weather_konvektiv_bedingung_fetch.py
"""
import gzip
import csv
import sys
import time
from collections import defaultdict

import airportsdata
import pandas as pd
import requests

from weather_outlier_screen import STATIONS

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

PREVRUN = "https://previous-runs-api.open-meteo.com/v1/forecast"
DUMP = "preregs/weather_konvektiv_sigma_residuen_2026_08_03.csv.gz"
OUT = "preregs/weather_konvektiv_sigma_bedingung_2026_08_04.csv.gz"
SCHEIBE = 200          # Tage je Anfrage
VERSUCHE = 4


def hole(lat, lon, von, bis):
    """Stundenwerte einer Scheibe. Leere Antwort bei identischen Parametern ist
    eine bekannte Cache-Macke der API — deshalb wiederholen, nicht aufgeben."""
    letzter = None
    for versuch in range(VERSUCHE):
        try:
            j = requests.get(PREVRUN, params={
                "latitude": lat, "longitude": lon,
                "start_date": von, "end_date": bis,
                "hourly": "precipitation_previous_day1,cloud_cover_previous_day1",
                "timezone": "auto"}, timeout=90).json()
            if j.get("hourly", {}).get("time"):
                return j["hourly"]
            letzter = j.get("reason", "leere Antwort")
        except Exception as exc:
            letzter = repr(exc)
        time.sleep(3 * (versuch + 1))
    print(f"      Scheibe {von}..{bis} fehlgeschlagen: {letzter}")
    return None


def main():
    df = pd.read_csv(DUMP, usecols=["city", "date"])
    spanne = df.groupby("city").date.agg(["min", "max"])
    ap = airportsdata.load("ICAO")

    print(f"{len(spanne)} Staedte, {spanne['min'].min()} .. {spanne['max'].max()}")
    zeilen = []
    for city, (von, bis) in spanne.iterrows():
        icao = STATIONS.get(city)
        a = ap.get(icao or "")
        if not a:
            print(f"  uebersprungen (keine Koordinate): {city} ({icao})")
            continue
        tage = defaultdict(lambda: {"p": [], "c": []})
        scheiben = pd.date_range(von, bis, freq=f"{SCHEIBE}D").strftime("%Y-%m-%d")
        for i, start in enumerate(scheiben):
            ende = (min(pd.Timestamp(start) + pd.Timedelta(days=SCHEIBE - 1),
                        pd.Timestamp(bis))).strftime("%Y-%m-%d")
            h = hole(a["lat"], a["lon"], start, ende)
            if not h:
                continue
            for ts, p, cc in zip(h["time"], h["precipitation_previous_day1"],
                                 h["cloud_cover_previous_day1"]):
                d, hh = ts.split("T")
                hh = int(hh[:2])
                if p is not None and 6 <= hh <= 18:
                    tage[d]["p"].append(p)
                if cc is not None and 9 <= hh <= 18:
                    tage[d]["c"].append(cc)
            time.sleep(1)
        n = 0
        for d, k in sorted(tage.items()):
            if not k["c"]:
                continue
            zeilen.append({"city": city, "date": d,
                           "regen_mm": round(sum(k["p"]), 2),
                           "wolken_tag": round(sum(k["c"]) / len(k["c"]), 1)})
            n += 1
        print(f"  {city:<15} {n:>4} Tage")

    with gzip.open(OUT, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["city", "date", "regen_mm", "wolken_tag"])
        w.writeheader()
        w.writerows(zeilen)
    konv = sum(1 for z in zeilen if z["regen_mm"] >= 1 and z["wolken_tag"] >= 60)
    print(f"\n{len(zeilen)} Stadttage -> {OUT}")
    print(f"davon konvektiv (>=1 mm UND >=60 %): {konv} ({konv/max(len(zeilen),1):.1%})")


if __name__ == "__main__":
    main()
