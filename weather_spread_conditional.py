# -*- coding: utf-8 -*-
"""weather_spread_conditional.py — Ist der Ensemble-Fehler groesser, wenn die
Modelle sich uneinig sind? (Empirische Rechtfertigung des Spannen-Vetos.)

Entstanden 14.07.2026 als Gegenprobe zum eigenen Fix: Nach dem Beijing-33-Verlust
(preregs/weather_lay_postmortem_2026_07_14_beijing.md) bekamen beide Screens einen
harten Spannen-Veto (Modellspanne > 3 Grad -> kein Kandidat). Der war zunaechst
nur plausibel, nicht gemessen — und er kostet Kandidaten (am 16.07. fielen 10 von
15 Staedten darunter). Also die faire Frage: filtert er echtes Risiko weg oder
nur Rendite?

Methodik (analog weather_wet_conditional.py, nur konditioniert auf die
Modellspanne statt auf Regen): Previous-Runs-Archiv (echter 24h-Lead), 5 Modelle,
je Tag Spanne = max-min der rohen Tageshochs. Ist = METAR (IEM, report_type 3+4).
Fehler = Ist - korrigiertes ENS-Mittel (Bias je Stadt aus dem Datensatz selbst,
damit der Vergleich zwischen den Spannen-Klassen bias-frei ist).

Kernfrage: Waechst |Fehler| und die Trefferwahrscheinlichkeit weit entfernter
Buckets mit der Spanne? Wenn ja, ist der Veto empirisch gedeckt.

Aufruf:
  python weather_spread_conditional.py --days 700
  python weather_spread_conditional.py --days 700 --city Beijing,Madrid,Tokyo
"""
import argparse
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone

import airportsdata

from weather_source_compare import (STATIONS, MODELS, fetch_model_daily_extreme,
                                    fetch_actual_daily_extreme)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

BINS = [(0.0, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 99.0)]
AP = airportsdata.load("ICAO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=700)
    ap.add_argument("--city", default=None, help="Kommagetrennt, sonst alle")
    ap.add_argument("--lead", type=int, default=1, choices=range(1, 8), metavar="N")
    args = ap.parse_args()

    cities = STATIONS if not args.city else {c: STATIONS[c] for c in args.city.split(",") if c in STATIONS}
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    start = end - timedelta(days=args.days)

    # (spanne, fehler) ueber alle Staedte gepoolt; Bias wird je Stadt entfernt
    pool = []
    for city, icao in cities.items():
        st = AP.get(icao)
        if not st:
            continue
        try:
            daily, tz = fetch_model_daily_extreme(icao, st["lat"], st["lon"],
                                                  start.isoformat(), end.isoformat(),
                                                  max, lead=args.lead)
            actual = fetch_actual_daily_extreme(icao, start, end, tz, max)
        except Exception as ex:
            print(f"  {city}: {str(ex)[:60]} -> skip")
            continue

        rows = []
        for day, ist in actual.items():
            vals = [daily[m][day] for m in MODELS if day in daily.get(m, {})]
            if len(vals) < 5:
                continue
            rows.append((max(vals) - min(vals), sum(vals) / len(vals), ist))
        if len(rows) < 50:
            print(f"  {city}: nur {len(rows)} Tage -> skip")
            continue

        bias = statistics.mean(ens - ist for _, ens, ist in rows)  # Forecast - Ist
        for spread, ens, ist in rows:
            pool.append((spread, ist - (ens - bias)))  # bias-bereinigter Fehler
        print(f"  {city:14} n={len(rows):3d}  Bias={bias:+.2f}  "
              f"Median-Spanne={statistics.median(r[0] for r in rows):.1f}°")
        time.sleep(1)

    if not pool:
        sys.exit("Keine Daten.")

    print(f"\n{'='*82}\nENSEMBLE-FEHLER KONDITIONIERT AUF DIE MODELLSPANNE "
          f"(Lead {args.lead*24}h, {len(cities)} Staedte gepoolt)\n{'='*82}")
    print(f"{'Spanne':>12} {'n':>5} {'MAE':>7} {'Sigma':>7} {'P(|err|>1.5)':>13} {'P(|err|>2.5)':>13}")
    print("-" * 82)
    for lo, hi in BINS:
        sub = [e for s, e in pool if lo <= s < hi]
        if not sub:
            continue
        mae = statistics.mean(abs(e) for e in sub)
        sig = statistics.pstdev(sub)
        p15 = sum(1 for e in sub if abs(e) > 1.5) / len(sub)
        p25 = sum(1 for e in sub if abs(e) > 2.5) / len(sub)
        label = f"{lo:.1f}-{hi:.1f}°" if hi < 90 else f">{lo:.1f}°"
        print(f"{label:>12} {len(sub):5d} {mae:6.2f}° {sig:6.2f}° {p15*100:12.1f}% {p25*100:12.1f}%")

    lo_grp = [e for s, e in pool if s < 3.0]
    hi_grp = [e for s, e in pool if s >= 3.0]
    if lo_grp and hi_grp:
        mae_lo = statistics.mean(abs(e) for e in lo_grp)
        mae_hi = statistics.mean(abs(e) for e in hi_grp)
        p_lo = sum(1 for e in lo_grp if abs(e) > 2.5) / len(lo_grp)
        p_hi = sum(1 for e in hi_grp if abs(e) > 2.5) / len(hi_grp)
        print(f"\nVETO-SCHWELLE 3,0°:  unter -> MAE {mae_lo:.2f}°, P(|err|>2.5°) {p_lo*100:.1f}% (n={len(lo_grp)})")
        print(f"                     ueber -> MAE {mae_hi:.2f}°, P(|err|>2.5°) {p_hi*100:.1f}% (n={len(hi_grp)})")
        print(f"                     Faktor MAE {mae_hi/mae_lo:.2f}x, Tail-Risiko {p_hi/max(p_lo,1e-9):.2f}x")
        print(f"\n  Anteil der Tage ueber der Schwelle: {len(hi_grp)/len(pool)*100:.0f} %")


if __name__ == "__main__":
    main()
