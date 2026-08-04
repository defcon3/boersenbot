# -*- coding: utf-8 -*-
"""weather_icon_source_bound.py — Wie viel Anker-Qualitaet steckt ueberhaupt in
der Quellen- und Extraktionswahl?

## Die Frage

Am 04.08.2026 bewarb metar.ws einen neuen Kanal: Tages-Max/Min direkt aus dem
Modelllauf, per WebSocket auf die Flughafenstation gepusht. Sandbox (frei) gibt
GFS 0.25, ICON Global und ICON-EU, die Bezahlstufen zusaetzlich ICON-D2, HRRR
und AROME. Die Frage war nicht "kostet das etwas", sondern "**bringt eine
andere Quelle desselben Modells unseren Anker voran**".

Dieses Skript misst die **Schranke** dafuer. Es beantwortet nicht, welche
Extraktion besser ist — es misst, wie viel Spielraum zwischen den moeglichen
Extraktionen ueberhaupt liegt. Liegt der Spielraum unter dem, was unsere
Kalibrierung ohnehin entfernt, ist die Frage erledigt, ohne dass eine einzige
fremde Zeile Daten geholt werden muss.

## Drei Sonden

* **A — Modellwahl.** ICON-Varianten (`seamless` / `global` / `eu` / `d2`) am
  exakt gleichen Stationspunkt. Wie weit liegen die angebotenen Varianten
  auseinander?
* **B — Extraktionsstelle.** Dasselbe Modell, Abgriffspunkt um 0,05 / 0,10 /
  0,25 Grad verschoben (rund 5 / 11 / 28 km). Wie stark bewegt der Ort das
  Tageshoch?
* **C — Ist dieser Versatz konstant?** Sonde B ueber `--days` Tage. Ein
  **konstanter** Versatz ist kein Gewinn: genau Konstanten entfernt unsere
  stadtweise Bias-Kalibrierung (`bias_700d` / `bias_40d`, s.
  `weather_outlier_screen.py`) per Konstruktion. Handelbar waere nur der
  *zustandsabhaengige* Rest, also sd(Versatz).

## Ergebnis vom 04.08.2026 (10 Staedte, Lead d+1, 60 Tage)

    A  Median-Spanne der ICON-Varianten      0,35 K
       -> und `icon_seamless` IST ICON-D2, wo D2 existiert (Munich, Paris,
          London, Milan: bis auf die Nachkommastelle identisch). Das
          hochaufloesende Modell der Bezahlstufe steckt bei Lead 24 h laengst
          in unserem Ensemble.
    B  Median max|Diff| ueber Extraktionsstellen   1,00 K  (Madrid 2,1)
       -> der Ort schlaegt die Modellwahl um rund das Dreifache.
    C  Median sd(Versatz)                     0,32 K
       -> bei mittleren Versaetzen bis 1,28 K (Milan). Der Ortseffekt ist
          also fast reine Konstante je Stadt — und damit bereits kalibriert.

**Schluss:** Nach Abzug dessen, was die Kalibrierung ohnehin holt, bleiben aus
der gesamten Quellen- und Extraktionsfrage rund 0,3 K zustandsabhaengige
Variabilitaet. Der Restfehler des Ankers liegt bei 0,79 Bucket. Die Quellenwahl
kann ihn nicht erklaeren; der Fehler sitzt in der zustandsabhaengigen
Stadt-Verschiebung (Tel Aviv, Hong-Kong-Nassbias) und in der sigma-
Fehlkalibrierung mit Vorzeichen.

**Ehrlich zur Reichweite:** Das ist Modell gegen Modell, nicht gegen Settlement
— eine Schranke, kein Gate. Sie sagt "hier ist nicht genug Spielraum, als dass
es sich lohnte zu messen", nicht "Quelle X ist schlechter als Quelle Y".

Aufruf:
    python weather_icon_source_bound.py                  # A + B + C, 10 Staedte
    python weather_icon_source_bound.py --probe c --days 90
    python weather_icon_source_bound.py --cities Madrid,Tokyo --probe b
"""
import argparse
import statistics as st
import sys
import time

import requests

from weather_outlier_screen import OM, STATIONS
from weather_stations import canonical_city, station_info

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

# Die Varianten, die metar.ws anbietet, plus das seamless, das wir fahren.
ICON_VARIANTS = ["icon_seamless", "icon_global", "icon_eu", "icon_d2"]
OFFSETS = [0.05, 0.10, 0.25]          # Grad; rund 5 / 11 / 28 km
DEFAULT_CITIES = ["Munich", "Paris", "London", "Milan", "Warsaw", "Tokyo",
                  "Beijing", "Madrid", "Tel Aviv", "Hong Kong"]


def coords(city):
    """Stationskoordinate wie im Screen — NICHT der Stadtmittelpunkt."""
    icao = STATIONS.get(canonical_city(city))
    st_ = station_info(icao) if icao else None
    return (st_["lat"], st_["lon"]) if st_ else None


def fetch(lat, lon, models, past_days=0, forecast_days=3, retries=3):
    """Tageshoch-Reihe je Modell. Gibt {modell: [werte...]} zurueck."""
    p = {"latitude": lat, "longitude": lon, "daily": "temperature_2m_max",
         "models": ",".join(models), "timezone": "auto",
         "forecast_days": forecast_days}
    if past_days:
        p["past_days"] = past_days
    for n in range(retries):
        try:
            j = requests.get(OM, params=p, timeout=40).json()
        except Exception as ex:
            if n == retries - 1:
                return {}
            time.sleep(2 * (n + 1))
            continue
        d = j.get("daily") or {}
        if not d:                      # z. B. "No data available for this location"
            return {}
        # Bei genau einem Modell haengt Open-Meteo das Suffix nicht an.
        if len(models) == 1:
            return {models[0]: d.get("temperature_2m_max") or []}
        return {m: (d.get("temperature_2m_max_" + m) or []) for m in models}
    return {}


def probe_a(cities):
    """Wie weit liegen die ICON-Varianten am selben Stationspunkt auseinander?"""
    print("\n=== A) ICON-Varianten am selben Stationspunkt (Tageshoch d+1) ===")
    print(f"{'Stadt':12s} " + " ".join(f"{v.replace('icon_',''):>9s}"
                                       for v in ICON_VARIANTS) + f" {'Spanne':>7s}")
    spans = []
    for c in cities:
        ll = coords(c)
        if not ll:
            print(f"{c:12s} keine Station"); continue
        got = fetch(*ll, ICON_VARIANTS)
        vals = {}
        for m in ICON_VARIANTS:
            s = got.get(m) or []
            vals[m] = s[1] if len(s) > 1 and s[1] is not None else None
        have = [v for v in vals.values() if v is not None]
        sp = (max(have) - min(have)) if len(have) > 1 else None
        if sp is not None:
            spans.append(sp)
        cells = " ".join(f"{vals[m]:9.1f}" if vals[m] is not None else f"{'-':>9s}"
                         for m in ICON_VARIANTS)
        print(f"{c:12s} {cells} {sp:7.2f}" if sp is not None
              else f"{c:12s} {cells} {'-':>7s}")
    if spans:
        print(f"Median-Spanne ICON-Varianten: {st.median(spans):.2f} K")
    return spans


def probe_b(cities):
    """Wie stark bewegt eine andere Extraktionsstelle das Tageshoch?"""
    print("\n=== B) Extraktionsstelle verschoben, gleiches Modell (ICON, d+1) ===")
    print(f"{'Stadt':12s} {'Station':>8s} " +
          " ".join(f"{'+'+str(o)+'deg':>9s}" for o in OFFSETS) + f" {'max|Diff|':>10s}")
    worst = []
    for c in cities:
        ll = coords(c)
        if not ll:
            print(f"{c:12s} keine Station"); continue
        lat, lon = ll

        def one(la, lo):
            s = (fetch(la, lo, ["icon_seamless"]) or {}).get("icon_seamless") or []
            return s[1] if len(s) > 1 else None

        base = one(lat, lon)
        off = [one(lat + o, lon + o) for o in OFFSETS]
        diffs = [abs(o - base) for o in off if o is not None and base is not None]
        md = max(diffs) if diffs else None
        if md is not None:
            worst.append(md)
        f = lambda v: f"{v:9.1f}" if v is not None else f"{'-':>9s}"
        print(f"{c:12s} {base:8.1f} " + " ".join(f(v) for v in off) +
              (f" {md:10.2f}" if md is not None else f" {'-':>10s}"))
    if worst:
        print(f"Median max|Diff| ueber Extraktionsstellen: {st.median(worst):.2f} K")
    return worst


def probe_c(cities, days, offset):
    """Ist der Versatz einer anderen Extraktionsstelle konstant?

    Nur sd(Versatz) waere handelbar — der Mittelwert steckt bereits in
    bias_700d/bias_40d.
    """
    print(f"\n=== C) Ist der Versatz konstant? (ICON, {days} Tage, +{offset} Grad) ===")
    print(f"{'Stadt':12s} {'n':>4s} {'mittl.Versatz':>14s} {'sd(Versatz)':>12s}")
    sds = []
    for c in cities:
        ll = coords(c)
        if not ll:
            print(f"{c:12s} keine Station"); continue
        lat, lon = ll
        a = (fetch(lat, lon, ["icon_seamless"], past_days=days,
                   forecast_days=1) or {}).get("icon_seamless") or []
        b = (fetch(lat + offset, lon + offset, ["icon_seamless"], past_days=days,
                   forecast_days=1) or {}).get("icon_seamless") or []
        d = [x - y for x, y in zip(a, b) if x is not None and y is not None]
        if not d:
            print(f"{c:12s} keine Daten"); continue
        s = st.pstdev(d)
        sds.append(s)
        print(f"{c:12s} {len(d):4d} {st.mean(d):14.2f} {s:12.2f}")
    if sds:
        print(f"Median sd(Versatz): {st.median(sds):.2f} K")
        print("Zur Einordnung: die stadtweise Bias-Kalibrierung entfernt den "
              "Mittelwert vollstaendig.\nHandelbar waere nur die sd-Spalte.")
    return sds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cities", default=",".join(DEFAULT_CITIES),
                    help="Komma-Liste; 'all' nimmt alle Screen-Staedte")
    ap.add_argument("--probe", default="abc", help="Teilmenge von a,b,c")
    ap.add_argument("--days", type=int, default=60, help="Sonde C: Ruecklaufzeit")
    ap.add_argument("--offset", type=float, default=0.10,
                    help="Sonde C: Verschiebung in Grad")
    a = ap.parse_args()

    cities = (sorted(STATIONS) if a.cities.strip().lower() == "all"
              else [c.strip() for c in a.cities.split(",") if c.strip()])
    p = a.probe.lower()
    if "a" in p: probe_a(cities)
    if "b" in p: probe_b(cities)
    if "c" in p: probe_c(cities, a.days, a.offset)


if __name__ == "__main__":
    main()
