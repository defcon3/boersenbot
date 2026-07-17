# -*- coding: utf-8 -*-
"""weather_screen_selftest.py — Regressions-Check der Screen-Filter.

Faengt die aktuelle Filter-Logik den bekannten Verlierer? Der Beijing-33-NO-Lay
(Zieltag 14.07.2026) ging verloren — Ist-Hoch exakt 33,0 °C, mitten im
Verlustfenster; nur der Autopilot-Take-Profit rettete ihn. Post-Mortem:
preregs/weather_lay_postmortem_2026_07_14_beijing.md

Dieser Test friert die Forecasts ein, die damals live waren (aus dem Previous-Runs-
Archiv, echter 24h-Lead), und verlangt, dass der Screen sie ABLEHNT. Er ist absichtlich
kein Unit-Test der Einzelfunktionen, sondern des Urteils: "haetten wir den Trade
heute noch gemacht?" Die View-Konstruktion kommt seit 17.07. aus build_views
(Debias-vor-Mittelung) — der Test benutzt sie IMPORTIERT, damit er immer die
echte Screen-Logik beurteilt, nicht eine Kopie. Zusaetzlich ein Fake-Kalibrier-
Fall fuer die Debias-Konsistenz bei Modell-Ausfall/Drop (Backlog Prio 1).

Aufruf:  python weather_screen_selftest.py     (Exit 0 = alle Faelle korrekt beurteilt)
"""
import sys

from weather_outlier_screen import (robust_mean, bucket_prob, dist_deg, reject_reasons,
                                    model_sigma, build_views, load_calib,
                                    CALIB_GLOB, CALIB40_GLOB, SHORT)

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
    calib = load_calib(CALIB_GLOB, exclude=("_min_", "calib40", "_lead"))
    calib40 = load_calib(CALIB40_GLOB, exclude=("_min_", "_lead"))
    _, dropped = robust_mean(raw)
    spread = max(raw.values()) - min(raw.values())
    views = build_views(city, raw, calib, calib40, spread, dropped)

    probs = {}
    for cal in (calib, calib40):
        for m in raw:
            if (city, m) in cal:
                b = cal[(city, m)][0]
                s = model_sigma(cal, city, m, spread)
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

# --- Debias-Konsistenz (17.07., Backlog Prio 1): Fake-Kalibrierung, Wahrheit 30,0 —
# jedes Modell zeigt exakt seinen eigenen Bias. Das korrigierte Mittel muss in ALLEN
# Sichten (voll / Modell-Ausfall / Ausreisser-Drop) exakt 30,0 bleiben. Die alte
# Logik (rohes Mittel − ENS-Bias) lag bei Ausfall/Drop daneben (hier −0,25/−0,25;
# real gemessen bis +0,98°, Jeddah ohne JMA).
FAKE = {("Testcity", "ensemble_mean"): (0.25, 1.0, None, None),
        ("Testcity", "gfs_seamless"): (1.0, 1.0, None, None),
        ("Testcity", "icon_seamless"): (0.0, 1.0, None, None),
        ("Testcity", "ukmo_seamless"): (0.5, 1.0, None, None),
        ("Testcity", "jma_seamless"): (-0.5, 1.0, None, None)}
RAW_VOLL = {"gfs_seamless": 31.0, "icon_seamless": 30.0, "ukmo_seamless": 30.5,
            "jma_seamless": 29.5}
print("=" * 86)
print("Debias-Konsistenz (Fake-Kalibrierung, Wahrheit 30,0)")
print("=" * 86)
for name, raw_t, dropped_t in [
        ("volle Modellmenge", RAW_VOLL, {}),
        ("Modell-Ausfall (GFS fehlt)", {m: v for m, v in RAW_VOLL.items() if m != "gfs_seamless"}, {}),
        ("Ausreisser-Drop (GFS geflaggt)", RAW_VOLL, {"gfs_seamless": 31.0})]:
    spread_t = max(raw_t.values()) - min(raw_t.values())
    vs = build_views("Testcity", raw_t, FAKE, {}, spread_t, dropped_t)
    schlecht = [(lbl, mu) for lbl, mu, _ in vs if abs(mu - 30.0) > 1e-9]
    ok = vs and not schlecht
    fehler += not ok
    zeig = "  ".join(f"{lbl}={mu:.2f}" for lbl, mu, _ in vs) or "(keine Views!)"
    print(f"  {name:32} {zeig}  ->  {'OK' if ok else 'FEHLGESCHLAGEN !!!'}")
print()

print("=" * 86)
if fehler:
    print(f"{fehler} Fall/Faelle NICHT korrekt beurteilt — die Filter greifen nicht mehr!")
    sys.exit(1)
print("Alle Faelle korrekt beurteilt. Der bekannte Verlierer wird abgelehnt; "
      "Debias-vor-Mittelung ist ausfall-konsistent.")
