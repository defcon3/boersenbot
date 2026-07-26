# -*- coding: utf-8 -*-
"""weather_classb_eval.py — Pre-Reg-Auswertung „Klasse B": Lay der ±1/±2-Offset-
Fenster relativ zum bias-korrigierten Modell-Favoriten.

Pre-Reg (Gates, Definition, Konsequenz): preregs/weather_classb_lay_2026_07_18.md.
Anlass: Nutzer-Entscheid 18.07. — gibt es zwischen den Screen-Kandidaten
(dist >= 2°) und dem Zentrum eine handelbare Rendite-Klasse, oder ist das Band
Beijing-33-Land?

Drei Teile (Default --part all = b -> a -> c):
  b  Preisseite + Forward aus bb_WeatherLadders (echte Snapshots seit 11.07.;
     NUR Vortags-Snapshots, Zieltag-Snapshots sind intraday) -> classb_market_side.json
  a  700d-Klassenraten (Methodik weather_error_quantiles: previous_day1-ENS,
     LOO-Bias, IEM-Ist) fuer das handelbare Staedte-Universum aus Teil b
     -> classb_700d_offsets.json  (langsam: ~1 IEM+1 OM-Fetch je Stadt)
  c  Kombination, ROI = (1-P_emp)/cost - 1, t = (BE_net - P_emp)/SE_P, Gates.
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pymssql

from weather_error_quantiles import fetch_city_both, offsets_for
from weather_source_compare import STATIONS
from weather_ladder_logger import DB_CONFIG

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

FEE = 0.07
A_JSON = "classb_700d_offsets.json"
B_JSON = "classb_market_side.json"
CLASSES = [-2, -1, 1, 2]


def art(basis, var):
    """Artefakt-Name je Variable. 'max' behaelt die historischen Dateinamen, damit
    die Ergebnisse der Pre-Reg vom 18.07. unveraendert gueltig bleiben; 'min'
    bekommt ein Suffix, sonst ueberschreibt ein Min-Lauf den Max-Cache."""
    return basis if var == "max" else basis.replace(".json", "_min.json")


def cost_of(no):
    return no + FEE * min(no, 1.0 - no)


def part_b(var="max"):
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor(as_dict=True)
    cur.execute("""
        SELECT city, k, buy_no, offset_fav, settle_result, target_date, snapshot_utc
        FROM bb_WeatherLadders
        WHERE var=%s AND kind='eq' AND mu_ens IS NOT NULL
          AND buy_no IS NOT NULL AND buy_no > 0 AND buy_no < 1
          AND status='open' AND offset_fav IS NOT NULL
          AND CAST(snapshot_utc AS date) < target_date
    """, (var,))
    rows = cur.fetchall()
    conn.close()
    best = {}
    for r in rows:
        key = (str(r["target_date"]), r["city"], r["k"])
        if key not in best or r["snapshot_utc"] > best[key]["snapshot_utc"]:
            best[key] = r
    price = defaultdict(list)
    fwd = defaultdict(list)
    for r in best.values():
        off = r["offset_fav"]
        if off not in CLASSES:
            continue
        price[off].append(r["buy_no"])
        if r["settle_result"] is not None:
            c = cost_of(r["buy_no"])
            pnl = -c if r["settle_result"] else (1.0 - c)
            fwd[off].append({"pnl": pnl, "cost": c, "hit": int(bool(r["settle_result"])),
                             "city": r["city"], "target": str(r["target_date"]), "k": r["k"]})
    out = {
        "cities": sorted({r["city"] for r in rows}),
        "n_windows": len(best),
        "price": {str(k): v for k, v in price.items()},
        "fwd": {str(k): v for k, v in fwd.items()},
    }
    json.dump(out, open(art(B_JSON, var), "w"))
    print(f"[b] {len(rows)} Zeilen, {len(best)} Fenster (Vortags-Snapshot), "
          f"{len(out['cities'])} Staedte -> {art(B_JSON, var)}")
    for cls in CLASSES:
        p = price.get(cls, [])
        f = fwd.get(cls, [])
        if p:
            print(f"[b]  Klasse {cls:+d}: n_preis={len(p)}  Ø NO {sum(p)/len(p):.3f}  "
                  f"| gesettelt n={len(f)} hits={sum(x['hit'] for x in f)}")
    return out


def part_a(cities, workers=2, var="max"):
    try:
        res = json.load(open(art(A_JSON, var)))
        print(f"[a] merge: {len(res)} Staedte bereits vorhanden")
    except Exception:
        res = {}
    cities = [c for c in cities if c not in res]
    todo = [(c, STATIONS[c]) for c in cities if c in STATIONS]
    skipped = [c for c in cities if c not in STATIONS]
    if skipped:
        print(f"[a] ohne Station, uebersprungen: {skipped}")

    def one(city, icao):
        time.sleep(2)  # IEM-Schonung (429-Lehre vom Erstlauf mit 3 Workern)
        # fetch_city_both liefert (ens_min, ens_max, act) — je nach
        # Zielvariable die passende ENS-Reihe waehlen.
        ens_min, ens_max, act = fetch_city_both(icao, 700)
        r = offsets_for(ens_min if var == "min" else ens_max, act, var)
        if not r:
            return city, None
        offs, sigma, _model_p, n = r
        return city, {"offs": offs, "sigma": sigma, "n": n}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, c, i): c for c, i in todo}
        for fut in as_completed(futs):
            city = futs[fut]
            try:
                city, payload = fut.result()
                if payload:
                    res[city] = payload
                    print(f"[a] {city}: n={payload['n']} sigma={payload['sigma']:.2f}", flush=True)
                else:
                    print(f"[a] {city}: zu wenig Daten", flush=True)
            except Exception as ex_:
                print(f"[a] {city}: FEHLER {ex_}", flush=True)
    json.dump(res, open(art(A_JSON, var), "w"))
    print(f"[a] {len(res)}/{len(todo)} Staedte -> {art(A_JSON, var)}")
    return res


def part_c(var="max"):
    A = json.load(open(art(A_JSON, var)))
    B = json.load(open(art(B_JSON, var)))
    pooled = []
    for city, d in A.items():
        pooled += [(city, o) for o in d["offs"]]
    n700 = len(pooled)
    print(f"\n=== KLASSE-B-REPORT (Pre-Reg preregs/weather_classb_lay_2026_07_18.md) ===")
    print(f"700d-Basis: {n700} Stadt-Tage, {len(A)} Staedte | "
          f"Preisseite: {B['n_windows']} Fenster seit 11.07.\n")
    print("Klasse |  P_emp (700d)      | Ø NO (n)      | cost  | BE_net | ROI netto |   t   | Forward (n, hits, ROI)")
    print("-" * 112)
    gates = {}
    for cls in CLASSES:
        hits = sum(1 for _, o in pooled if o == cls)
        P = hits / n700
        SE = math.sqrt(P * (1 - P) / n700)
        prices = B["price"].get(str(cls), [])
        if not prices:
            print(f"  {cls:+d}   | {P*100:5.2f} % ({hits})   | (keine Preise)")
            continue
        mean_cost = sum(cost_of(x) for x in prices) / len(prices)
        mean_no = sum(prices) / len(prices)
        be_net = 1.0 - mean_cost
        roi = (1.0 - P) / mean_cost - 1.0
        t = (be_net - P) / SE if SE > 0 else float("nan")
        f = B["fwd"].get(str(cls), [])
        froi = (sum(x["pnl"] for x in f) / sum(x["cost"] for x in f)) if f else None
        fs = f"n={len(f)} hits={sum(x['hit'] for x in f)} ROI {froi*100:+.1f} %" if f else "n=0"
        print(f"  {cls:+d}   | {P*100:5.2f} % ± {SE*100:.2f} | {mean_no:.3f} ({len(prices):3d}) | "
              f"{mean_cost:.3f} | {be_net*100:5.1f} % | {roi*100:+7.2f} % | {t:5.2f} | {fs}")
        gates[cls] = {"P": P, "SE": SE, "roi": roi, "t": t, "cost": mean_cost,
                      "fwd_n": len(f), "fwd_roi": froi}

    print("\n— Robustheit (G-B4): Pooled-ROI je Klasse ohne die jeweils guenstigste Stadt / ohne Seoul —")
    for cls, g in gates.items():
        worst_roi, worst_city = None, None
        for drop in A.keys():
            sub = [(c, o) for c, o in pooled if c != drop]
            Ps = sum(1 for _, o in sub if o == cls) / len(sub)
            r = (1.0 - Ps) / g["cost"] - 1.0
            if worst_roi is None or r < worst_roi:
                worst_roi, worst_city = r, drop
        sub = [(c, o) for c, o in pooled if c != "Seoul"]
        Pseoul = sum(1 for _, o in sub if o == cls) / len(sub)
        r_seoul = (1.0 - Pseoul) / g["cost"] - 1.0
        g["worst_roi"] = worst_roi
        g["roi_no_seoul"] = r_seoul
        print(f"  {cls:+d}: worst-drop {worst_city}: ROI {worst_roi*100:+.2f} % | ohne Seoul: {r_seoul*100:+.2f} %")

    print("\n— Gates (G-B1 ROI>3 %, G-B2 t>1,5, G-B3 Forward, G-B4 Robustheit) —")
    for cls, g in gates.items():
        g1 = g["roi"] > 0.03
        g2 = g["t"] > 1.5
        g3 = True if g["fwd_n"] < 30 else (g["fwd_roi"] is not None and g["fwd_roi"] > -0.05)
        g3s = "nachrichtlich" if g["fwd_n"] < 30 else ("PASS" if g3 else "FAIL")
        g4 = g["worst_roi"] > 0 and (g["roi_no_seoul"] > 0) == (g["roi"] > 0)
        verdict = "GRUEN" if (g1 and g2 and g3 and g4) else "ROT"
        print(f"  Klasse {cls:+d}: G-B1 {'PASS' if g1 else 'FAIL'} | G-B2 {'PASS' if g2 else 'FAIL'} "
              f"| G-B3 {g3s} | G-B4 {'PASS' if g4 else 'FAIL'}  => {verdict}")

    print("\n— je Stadt (P je Klasse in %, n, sigma) —")
    print(f"{'Stadt':14s}" + "".join(f"{c:+7d}" for c in CLASSES) + "      n  sigma")
    for city in sorted(A):
        offs = A[city]["offs"]
        n = len(offs)
        row = "".join(f"{sum(1 for o in offs if o == c)/n*100:6.1f}%" for c in CLASSES)
        print(f"{city:14s}{row} {n:6d}  {A[city]['sigma']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "c", "all"], default="all")
    ap.add_argument("--cities", default=None,
                    help="Kommaliste fuer Teil a (Default: Staedte aus Teil-b-JSON)")
    ap.add_argument("--workers", type=int, default=2, help="parallele Fetches in Teil a")
    ap.add_argument("--var", choices=["max", "min"], default="max",
                    help="Tageshoch (max, Pre-Reg 18.07.) oder Tagestief (min, in der Pre-Reg ausgeklammert: 'Min-Bretter existieren kaum')")
    args = ap.parse_args()
    if args.part in ("b", "all"):
        part_b(args.var)
    if args.part in ("a", "all"):
        if args.cities:
            cities = [c.strip() for c in args.cities.split(",")]
        else:
            cities = json.load(open(art(B_JSON, args.var)))["cities"]
        part_a(cities, workers=args.workers, var=args.var)
    if args.part in ("c", "all"):
        part_c(args.var)


if __name__ == "__main__":
    main()
