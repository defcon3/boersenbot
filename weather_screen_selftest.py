# -*- coding: utf-8 -*-
"""weather_screen_selftest.py — Regressions-Check der Screen-Filter.

Faengt die aktuelle Filter-Logik den bekannten Verlierer? Der Beijing-33-NO-Lay
(Zieltag 14.07.2026) ging verloren — Ist-Hoch exakt 33,0 °C, mitten im
Verlustfenster; nur der Autopilot-Take-Profit rettete ihn. Post-Mortem:
preregs/weather_lay_postmortem_2026_07_14_beijing.md

Dieser Test friert die Forecasts ein, die damals live waren (aus dem Previous-Runs-
Archiv, echter 24h-Lead), und verlangt, dass der Screen sie ABLEHNT. Er ist absichtlich
kein Unit-Test der Einzelfunktionen, sondern des Urteils: "haetten wir den Trade
heute noch gemacht?"

Aufruf:  python weather_screen_selftest.py     (Exit 0 = alle Faelle korrekt beurteilt)
"""
import sys

from weather_outlier_screen import (robust_mean, bucket_prob, dist_deg, reject_reasons,
                                    ens_sigma, load_calib, CALIB_GLOB, CALIB40_GLOB, SHORT)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

M = {"GFS": "gfs_seamless", "ICON": "icon_seamless", "UKMO": "ukmo_seamless",
     "JMA": "jma_seamless", "ECMWF": "ecmwf_ifs025"}

# (Name, Stadt, ROHE Forecasts wie sie live waren, Bucket, NO-Preis, YES-Preis, Soll-Urteil)
FAELLE = [
    ("Beijing 33° (14.07.) — GESETZT, Ist 33,0 -> VERLOREN", "Beijing",
     {"GFS": 34.7, "ICON": 34.4, "UKMO": 34.0, "JMA": 37.1, "ECMWF": 33.5},
     33, 0.79, 0.20, "ABLEHNEN"),
    ("Beijing 32° (16.07.) — gleiche JMA-Signatur, nicht gesetzt", "Beijing",
     {"GFS": 34.5, "ICON": 33.8, "UKMO": 34.4, "JMA": 38.4, "ECMWF": 33.2},
     32, 0.82, 0.18, "ABLEHNEN"),
]


def urteil(city, raw_named, k, no_px, yes_px):
    raw = {M[n]: v for n, v in raw_named.items()}
    calib = load_calib(CALIB_GLOB, exclude=("_min_", "calib40"))
    calib40 = load_calib(CALIB40_GLOB, exclude=("_min_",))
    ens = sum(raw.values()) / len(raw)
    rob, dropped = robust_mean(raw)
    spread = max(raw.values()) - min(raw.values())

    views = []
    for cname, cal in (("700d", calib), ("40d", calib40)):
        if (city, "ensemble_mean") not in cal:
            continue
        b = cal[(city, "ensemble_mean")][0]
        s = ens_sigma(cal, city, spread)
        views.append((cname, ens - b, s))
        if dropped:
            views.append((f"{cname}/rob", rob - b, s))

    probs = {}
    for cal in (calib, calib40):
        for m in raw:
            if (city, m) in cal:
                b, s = cal[(city, m)][:2]
                probs[m] = max(probs.get(m, 0.0), bucket_prob("eq", k, raw[m] - b, s))
    pm = max(probs, key=probs.get)
    p_use, p_src, _ = max((bucket_prob("eq", k, mu, s), lbl, s) for lbl, mu, s in views)
    d = min(dist_deg("eq", k, mu) for _, mu, _ in views)
    be = 1.0 - no_px
    r = {"dist": d, "spread": spread, "p_max": probs[pm], "p_max_src": SHORT[pm],
         "ev": be - p_use, "has40": True, "buyYes": yes_px}
    return r, reject_reasons(r), p_use, p_src, be, dropped, views


fehler = 0
for name, city, raw_named, k, no_px, yes_px, soll in FAELLE:
    r, why, p_use, p_src, be, dropped, views = urteil(city, raw_named, k, no_px, yes_px)
    ist = "ABLEHNEN" if why else "KANDIDAT"
    ok = ist == soll
    fehler += not ok
    print("=" * 86)
    print(f"{name}\n{'=' * 86}")
    print("  roh: " + "  ".join(f"{n} {v}{'*' if M[n] in dropped else ''}" for n, v in raw_named.items())
          + f"   Spanne {r['spread']:.1f}°")
    print("  Sichten: " + "   ".join(f"{l} {mu:.2f}±{s:.2f}" for l, mu, s in views))
    print(f"  P_pess {p_use*100:.1f} % ({p_src})  BE {be*100:.1f} %  EV {r['ev']*100:+.1f}pp  "
          f"P_max {r['p_max']*100:.1f} % ({r['p_max_src']})")
    print(f"\n  Soll {soll} / Ist {ist}  ->  {'OK' if ok else 'FEHLGESCHLAGEN !!!'}")
    if why:
        print(f"  Gruende: {', '.join(why)}")
    print()

print("=" * 86)
if fehler:
    print(f"{fehler} Fall/Faelle NICHT korrekt beurteilt — die Filter greifen nicht mehr!")
    sys.exit(1)
print("Alle Faelle korrekt beurteilt. Der bekannte Verlierer wird abgelehnt.")
