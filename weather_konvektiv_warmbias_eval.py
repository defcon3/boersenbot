# -*- coding: utf-8 -*-
"""weather_konvektiv_warmbias_eval.py — Preist der Markt an konvektiv gedeckelten
Tropentagen zu warm?

Pre-Reg: preregs/weather_konvektiv_warmbias_2026_08_03.md (Gates G1-G5 dort
festgelegt, BEVOR die Zielgroesse angesehen wurde).

Zielgroesse je Stadttag: d = market_fav_k - settle_k (Buckets, 1 Bucket = 1 K).
d > 0 = Markt war zu warm.

Quelle: bb_WeatherLadders (var='max'), nur Lead-1-Snapshots — der Logger laeuft
12:30 UTC, ein Lead-0-Snapshot liegt fuer Asien NACH dem Tagesmaximum und der
Markt kennt das Ergebnis dann bereits. Bedingung aus der Vortagesprognose
(Open-Meteo previous-runs, previous_day1) und damit handelbar.

Aufruf:
  python weather_konvektiv_warmbias_eval.py            # zieht Daten frisch
  python weather_konvektiv_warmbias_eval.py --cache scratch_konvektiv.csv
"""
import argparse
import csv
import math
import statistics as st
import sys
import time
from collections import defaultdict

import airportsdata
import pymssql
import requests

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_CONFIG = {"server": "158.181.48.77", "database": "dbdata",
             "user": "326773", "password": "Extaler11!"}
PREVRUN = "https://previous-runs-api.open-meteo.com/v1/forecast"

TROPEN_LAT = 23.5
REGEN_MIN = 1.0      # mm, Summe 06-18 h lokal, Vortagesprognose
WOLKEN_MIN = 60.0    # %, Mittel 09-18 h lokal, Vortagesprognose

# Hong Kong laeuft ueber die HKO-Pseudostation und hat keine ICAO-Koordinate.
COORD_EXTRA = {"Hong Kong": (22.302, 114.174)}


def lade_stadttage():
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT target_date, city, MAX(icao), MAX(market_fav_k), MAX(settle_k), AVG(mu_ens)
        FROM bb_WeatherLadders
        WHERE var='max' AND market_fav_k IS NOT NULL AND settle_k IS NOT NULL
          AND DATEDIFF(day, CAST(snapshot_utc AS DATE), target_date) = 1
        GROUP BY target_date, city""")
    rows = cur.fetchall()
    conn.close()
    return rows


def hole_bedingung(rows):
    """Vortagesprognose fuer Regen/Bewoelkung je Stadttag (ein Call je Stadt)."""
    ap = airportsdata.load("ICAO")
    je_stadt = defaultdict(list)
    for d, city, icao, fav, settle, mu in rows:
        je_stadt[city].append((d, icao, fav, settle, mu))

    out = []
    for city in sorted(je_stadt):
        tage = je_stadt[city]
        a = ap.get(tage[0][1] or "")
        lat, lon = (a["lat"], a["lon"]) if a else COORD_EXTRA.get(city, (None, None))
        if lat is None:
            print(f"  uebersprungen (keine Koordinate): {city}")
            continue
        ds = sorted(t[0] for t in tage)
        j = {}
        for _ in range(3):
            j = requests.get(PREVRUN, params={
                "latitude": lat, "longitude": lon,
                "start_date": str(ds[0]), "end_date": str(ds[-1]),
                "hourly": "precipitation_previous_day1,cloud_cover_previous_day1",
                "timezone": "auto"}, timeout=40).json()
            if j.get("hourly", {}).get("time"):
                break
            time.sleep(3)   # leere Antwort bei identischen Parametern = Cache-Macke
        h = j.get("hourly", {})
        if not h.get("time"):
            print(f"  uebersprungen (keine Prognose): {city}")
            continue
        tag = defaultdict(lambda: {"p": [], "c": []})
        for ts, p, cc in zip(h["time"], h["precipitation_previous_day1"],
                             h["cloud_cover_previous_day1"]):
            d, hh = ts.split("T")
            hh = int(hh[:2])
            if p is not None and 6 <= hh <= 18:
                tag[d]["p"].append(p)
            if cc is not None and 9 <= hh <= 18:
                tag[d]["c"].append(cc)
        for d, _, fav, settle, mu in tage:
            k = tag.get(str(d))
            if not k or not k["c"]:
                continue
            out.append({"city": city, "lat": round(lat, 2), "date": str(d),
                        "fav": fav, "settle": settle,
                        "mu": round(mu, 2) if mu is not None else "",
                        "regen_mm": round(sum(k["p"]), 2),
                        "wolken_tag": round(sum(k["c"]) / len(k["c"]), 1)})
    return out


def t_ein(werte):
    """t gegen 0. Gibt (mean, t, n)."""
    n = len(werte)
    if n < 3:
        return (float("nan"), float("nan"), n)
    m = st.mean(werte)
    s = st.stdev(werte)
    return (m, m / (s / math.sqrt(n)) if s > 0 else float("inf"), n)


def t_zwei(a, b):
    """Welch-t fuer den Mittelwertunterschied a - b."""
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    if va + vb == 0:
        return float("inf")
    return (st.mean(a) - st.mean(b)) / math.sqrt(va + vb)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--cache", help="CSV mit bereits gezogenen Bedingungsdaten")
    args = ap_.parse_args()

    if args.cache:
        rows = list(csv.DictReader(open(args.cache, encoding="utf-8")))
        for r in rows:
            r["lat"] = float(r["lat"]); r["fav"] = int(r["fav"])
            r["settle"] = int(r["settle"]); r["regen_mm"] = float(r["regen_mm"])
            r["wolken_tag"] = float(r["wolken_tag"])
            r["mu"] = float(r["mu"]) if r["mu"] not in ("", None) else None
    else:
        roh = lade_stadttage()
        print(f"Stadttage (Lead 1): {len(roh)}")
        rows = hole_bedingung(roh)

    for r in rows:
        r["d"] = r["fav"] - r["settle"]

    trop = [r for r in rows if abs(r["lat"]) <= TROPEN_LAT]
    rest = [r for r in rows if abs(r["lat"]) > TROPEN_LAT]
    konv = lambda r: r["regen_mm"] >= REGEN_MIN and r["wolken_tag"] >= WOLKEN_MIN

    K = [r for r in trop if konv(r)]              # Primaergruppe
    T_trocken = [r for r in trop if not konv(r)]
    R = [r for r in rest if konv(r)]              # Kontrolle ausserhalb der Tropen

    print("=" * 74)
    print("KONVEKTIVER WARM-BIAS — d = market_fav_k - settle_k  (d>0 = Markt zu warm)")
    print("=" * 74)
    print(f"Stadttage gesamt {len(rows)} | Tropen {len(trop)} | uebrige {len(rest)}")
    print(f"Bedingung: Regen >= {REGEN_MIN} mm UND Wolken >= {WOLKEN_MIN} % (Vortagesprognose)\n")

    for name, grp in [("K  Tropen + konvektiv", K), ("   Tropen trocken", T_trocken),
                      ("R  Rest + konvektiv", R),
                      ("   Rest trocken", [r for r in rest if not konv(r)])]:
        m, t, n = t_ein([r["d"] for r in grp])
        staedte = len({r["city"] for r in grp})
        print(f"{name:<24} n={n:>3} Staedte={staedte:>2}  mean(d)={m:+.3f}  t={t:+.2f}")

    mK, tK, nK = t_ein([r["d"] for r in K])
    print("\n" + "-" * 74)
    print("GATES")
    print("-" * 74)
    g1 = mK > 0 and tK > 2.0
    print(f"G1 Signal          mean(d)={mK:+.3f}  t={tK:+.2f} > 2,0 ...... {'PASS' if g1 else 'FAIL'}")

    t2 = t_zwei([r["d"] for r in K], [r["d"] for r in T_trocken])
    g2 = t2 > 1.5
    print(f"G2 Spezifitaet     K vs Tropen-trocken  t={t2:+.2f} > 1,5 ... {'PASS' if g2 else 'FAIL'}")

    mR = st.mean([r["d"] for r in R]) if len(R) >= 3 else float("nan")
    g3 = mK > mR
    print(f"G3 Tropen > Rest   {mK:+.3f} vs {mR:+.3f} ................... {'PASS' if g3 else 'FAIL'}")

    g4 = nK >= 30 and len({r["city"] for r in K}) >= 4
    print(f"G4 Menge           n={nK}, Staedte={len({r['city'] for r in K})} ....................... {'PASS' if g4 else 'FAIL'}")

    print("G5 Leave-one-city-out:")
    g5 = True
    for c in sorted({r["city"] for r in K}):
        m, t, n = t_ein([r["d"] for r in K if r["city"] != c])
        ok = (m > 0) == (mK > 0) and t > 1.0
        g5 &= ok
        print(f"     ohne {c:<14} n={n:>3}  mean={m:+.3f}  t={t:+.2f}  {'ok' if ok else 'RISS'}")
    print(f"G5 Robustheit ............................................ {'PASS' if g5 else 'FAIL'}")

    print("\n" + "=" * 74)
    print(f"URTEIL: {'GRUEN' if all([g1, g2, g3, g4, g5]) else 'ROT'}")
    print("=" * 74)

    # --- beschreibend, kein Gate ---------------------------------------------
    print("\nSensitivitaet (kein Gate — nicht als zweiter Primaertest lesen):")
    for name, f in [("Regen >= 1 mm", lambda r: r["regen_mm"] >= 1),
                    ("Regen >= 3 mm", lambda r: r["regen_mm"] >= 3),
                    ("Regen >= 5 mm", lambda r: r["regen_mm"] >= 5),
                    ("Wolken >= 60 %", lambda r: r["wolken_tag"] >= 60),
                    ("Wolken >= 75 %", lambda r: r["wolken_tag"] >= 75)]:
        m, t, n = t_ein([r["d"] for r in trop if f(r)])
        print(f"   Tropen ∧ {name:<16} n={n:>3}  mean(d)={m:+.3f}  t={t:+.2f}")

    print("\nMarkt gegen unser Modell in der Primaergruppe K:")
    paare = [(r["d"], round(r["mu"]) - r["settle"]) for r in K if r.get("mu") is not None]
    if paare:
        mm, tm, nm = t_ein([p[1] for p in paare])
        md, td, nd = t_ein([p[0] for p in paare])
        tref = len([1 for p in paare if p[0] == 0]) / len(paare)
        mref = len([1 for p in paare if p[1] == 0]) / len(paare)
        print(f"   n={nm}  Markt mean(d)={md:+.3f}  Modell mean={mm:+.3f}")
        print(f"   exakter Treffer: Markt {tref:.1%}  Modell {mref:.1%}")

    print("\nJe Stadt in K:")
    for c in sorted({r["city"] for r in K}):
        g = [r["d"] for r in K if r["city"] == c]
        print(f"   {c:<15} n={len(g):>2}  mean(d)={st.mean(g):+.2f}")


if __name__ == "__main__":
    main()
