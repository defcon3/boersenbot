#!/usr/bin/env python3
"""
weather_band_breite_eval.py — Auswertung zu `preregs/weather_band_breite_2026_08_02.md`.

FRAGE: Traegt das Preisband 0,70-0,90 auf Daten, die es nicht gebaut haben — und
schraenken die beiden Auswahlkriterien des Autobuy (Abstands-Rang, Spannen-Veto)
den Bot ein, ohne etwas dafuer zu liefern?

HOLDOUT: Fenster A = Zieltage 10.-19.07.2026. Der Ladder-Logger schrieb dort
bereits, waehrend es das Preisband noch nicht gab (gebaut am 27.07. auf den
Zieltagen 20.-26.07.) und der Autobuy noch nicht lief (ab 20.07.). Das ist kein
nachtraeglich gezogener Split.

ENTDUPLIZIEREN — die Falle vom 01.08.: bb_WeatherLadders schreibt die Leiter je
Zieltag MEHRFACH (Lead 0/1/2, dazu vier Testlaeufe am 26.07.). Zeilen sind keine
Positionen. Geschluesselt wird auf (target_date, city, k); 1058 Rohzeilen werden
so zu 542 Positionen. Wer das vergisst, meldet die dreifache Menge.

SIGNIFIKANZ: gepaarter t-Test ueber TAGES-Mittel, nie ueber Einzelpositionen.
Staedte desselben Zieltags haengen an derselben Wetterlage; positionsweise
gerechnet waere das t etwa um den Faktor sqrt(Kandidaten je Tag) zu gross.

WAS GEWERTET WIRD: `settle_result`, also die Markt-Aufloesung — das ist das Geld.
Nicht `wu_settle_k`. Am 02.08. wurde gemessen, dass beide an 5 von 319 Stadt-Tagen
auseinanderliegen und der Markt jedes Mal METAR folgte, nie WU.

Aufruf:
  python weather_band_breite_eval.py
  python weather_band_breite_eval.py --fenster 2026-07-10 2026-07-19
  python weather_band_breite_eval.py --grenzen 0.69 0.91     # Stabilitaetsprobe
"""

import argparse
import math
import statistics
import sys
from collections import defaultdict

import pymssql

from weather_ladder_logger import DB_CONFIG
from weather_stations import bucket_grenzen

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

FEE = 0.07
USD = 5.0
BAND_LO, BAND_HI = 0.70, 0.90


def pnl_lay(no, verloren):
    """Identisch zu weather_minus1_shadow / _ppess_filter, damit vergleichbar."""
    n = USD / no
    fee = FEE * n * min(no, 1 - no)
    return (-USD - fee) if verloren else (n - USD - fee)


def lade(von, bis):
    """Entduplizierte -1-Kandidaten mit bekanntem Markt-Settlement."""
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "SELECT target_date, city, k, buy_no, settle_result, mu_ens "
        "FROM bb_WeatherLadders WHERE var='max' AND kind='eq' AND offset_fav=-1 "
        "AND buy_no IS NOT NULL AND target_date BETWEEN %s AND %s", (von, bis))
    roh = cur.fetchall()
    conn.close()

    je = {}
    for td, city, k, no, res, mu in roh:
        d = je.setdefault((str(td), city, int(k)),
                          {"nos": [], "res": None, "mus": []})
        d["nos"].append(float(no))
        if res is not None:
            d["res"] = bool(res)
        if mu is not None:
            d["mus"].append(float(mu))

    posten = []
    for (td, city, k), d in je.items():
        if d["res"] is None or not d["mus"]:
            continue
        no = sum(d["nos"]) / len(d["nos"])
        mu = sum(d["mus"]) / len(d["mus"])
        oben = bucket_grenzen(k, city)[1]
        posten.append({
            "tag": td, "city": city, "k": k, "no": no, "mu": mu,
            "abstand": mu - oben,
            "verloren": d["res"],
            "pnl": pnl_lay(no, d["res"]),
        })
    return sorted(posten, key=lambda p: (p["tag"], p["city"])), len(roh), len(je)


def kennzahl(menge):
    if not menge:
        return {"n": 0, "pnl": 0.0, "roi": 0.0, "verl": 0, "quote": 0.0}
    p = sum(x["pnl"] for x in menge)
    v = sum(1 for x in menge if x["verloren"])
    return {"n": len(menge), "pnl": p, "roi": 100 * p / (USD * len(menge)),
            "verl": v, "quote": 100 * v / len(menge)}


def tages_roi(menge):
    """ROI je Zieltag — die Einheit, ueber die das t gerechnet wird."""
    je = defaultdict(list)
    for x in menge:
        je[x["tag"]].append(x)
    return {t: 100 * sum(y["pnl"] for y in xs) / (USD * len(xs))
            for t, xs in je.items()}


def gepaartes_t(a, b):
    """t ueber die Tagesdifferenzen zweier Regeln auf denselben Zieltagen."""
    tage = sorted(set(a) & set(b))
    diff = [a[t] - b[t] for t in tage]
    if len(diff) < 3:
        return float("nan"), len(diff), float("nan")
    m = statistics.mean(diff)
    sd = statistics.stdev(diff)
    if sd == 0:
        return float("inf"), len(diff), m
    return m / (sd / math.sqrt(len(diff))), len(diff), m


def zeile(name, menge, breite=34):
    k = kennzahl(menge)
    if not k["n"]:
        print(f"  {name:<{breite}} —")
        return
    print(f"  {name:<{breite}} {k['n']:3d} Lays  {k['pnl']:+7.2f} $  "
          f"ROI {k['roi']:+7.2f} %  Verlierer {k['verl']:2d}/{k['n']:<3d} "
          f"({k['quote']:4.1f} %)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fenster", nargs=2, default=["2026-07-10", "2026-07-19"],
                    metavar=("VON", "BIS"), help="Holdout-Fenster A")
    ap.add_argument("--grenzen", nargs=2, type=float, default=[BAND_LO, BAND_HI],
                    metavar=("LO", "HI"), help="Preisband (Stabilitaetsprobe)")
    ap.add_argument("--cap", type=int, default=8, help="Cap des breiten Buchs")
    ap.add_argument("--eng", type=int, default=3,
                    help="Positionen je Tag im engen Buch (heutige Praxis)")
    a = ap.parse_args()
    lo, hi = a.grenzen

    posten, n_roh, n_dedup = lade(*a.fenster)
    band = [p for p in posten if lo <= p["no"] < hi]
    tage = sorted({p["tag"] for p in posten})

    print("=" * 78)
    print(f"FENSTER A   {a.fenster[0]} .. {a.fenster[1]}   Preisband {lo:.2f}-{hi:.2f}")
    print("=" * 78)
    print(f"Rohzeilen {n_roh} -> entdupliziert {n_dedup} -> mit Settlement und mu "
          f"{len(posten)}")
    print(f"Zieltage {len(tage)} | im Band {len(band)}\n")

    # ---------------------------------------------------------------- G1
    print("G1  FUNDAMENT — traegt das Band auf ungesehenen Daten?")
    zeile("Preisband", band)
    zeile("alle -1-Kandidaten", posten)
    t, nt, m = gepaartes_t(tages_roi(band), tages_roi(posten))
    k_band, k_alle = kennzahl(band), kennzahl(posten)
    vorteil = k_band["roi"] - k_alle["roi"]
    print(f"\n  Vorteil {vorteil:+.2f} pp   (Tagesmittel {m:+.2f} pp, t = {t:+.2f} "
          f"ueber {nt} Tage)")
    g1 = k_band["roi"] > 0 and vorteil >= 8.0 and t > 1.5
    print(f"  Verlangt: ROI > 0 UND Vorteil >= 8 pp UND t > 1,5   ->  "
          f"{'BESTANDEN' if g1 else 'GERISSEN'}")

    # ---------------------------------------------------------------- G2
    print("\nG2  ROBUSTHEIT — haelt es ohne den besten Tag und die stuetzende Stadt?")
    tr = tages_roi(band)
    if tr:
        best_tag = max(tr, key=tr.get)
        ohne_tag = [p for p in band if p["tag"] != best_tag]
        zeile(f"ohne besten Tag ({best_tag})", ohne_tag)
        je_stadt = defaultdict(float)
        for p in band:
            je_stadt[p["city"]] += p["pnl"]
        best_stadt = max(je_stadt, key=je_stadt.get) if je_stadt else None
        ohne_stadt = [p for p in band if p["city"] != best_stadt]
        zeile(f"ohne staerkste Stadt ({best_stadt})", ohne_stadt)
        ges = sum(p["pnl"] for p in band)
        anteil = 100 * (ges - sum(p["pnl"] for p in ohne_tag)) / ges if ges else 0
        print(f"\n  Bester Zieltag traegt {anteil:.1f} % des Effekts "
              f"(Grenze 35 %)")
        g2 = (kennzahl(ohne_tag)["roi"] > 0 and kennzahl(ohne_stadt)["roi"] > 0
              and abs(anteil) <= 35.0)
        print(f"  ->  {'BESTANDEN' if g2 else 'GERISSEN'}")

    # ---------------------------------------------------------------- G3
    print("\nG3  RANG — traegt der Temperaturabstand INNERHALB des Bandes?")
    if len(band) >= 6:
        med = statistics.median(p["abstand"] for p in band)
        nah = [p for p in band if p["abstand"] < med]
        weit = [p for p in band if p["abstand"] >= med]
        print(f"  Median-Abstand {med:+.2f} K   (genau EIN Schnitt, keine "
              f"Schwellensuche)")
        zeile("mu NAH an der Kante", nah)
        zeile("mu WEIT von der Kante", weit)
        t3, nt3, m3 = gepaartes_t(tages_roi(weit), tages_roi(nah))
        unterschied = kennzahl(weit)["roi"] - kennzahl(nah)["roi"]
        print(f"\n  Unterschied {unterschied:+.2f} pp   (Tagesmittel {m3:+.2f} pp, "
              f"t = {t3:+.2f} ueber {nt3} Tage)")
        belegt = unterschied >= 8.0 and t3 > 2.0
        print(f"  Verlangt fuer 'Rang traegt': >= 8 pp UND t > 2,0   ->  "
              f"{'RANG BELEGT' if belegt else 'RANG NICHT BELEGT'}")

    # ---------------------------------------------------------------- H3
    print("\nH3  GELDTEST — breites gegen enges Buch (dieselben Zieltage)")
    je_tag = defaultdict(list)
    for p in band:
        je_tag[p["tag"]].append(p)
    breit, eng = [], []
    for t_, xs in je_tag.items():
        nach_abstand = sorted(xs, key=lambda p: -p["abstand"])
        breit += nach_abstand[:a.cap]
        eng += nach_abstand[:a.eng]
    zeile(f"breit (alle im Band, Cap {a.cap})", breit)
    zeile(f"eng (Top {a.eng} nach Abstand)", eng)
    th, nth, mh = gepaartes_t(tages_roi(breit), tages_roi(eng))
    print(f"\n  Differenz {kennzahl(breit)['roi'] - kennzahl(eng)['roi']:+.2f} pp   "
          f"(Tagesmittel {mh:+.2f} pp, t = {th:+.2f} ueber {nth} Tage)")
    print("  HINWEIS: das enge Buch ist hier OHNE Spannen-Veto nachgebildet — die "
          "Modellspanne\n  existiert erst ab dem 20.07. im Autobuy-Log. Das enge "
          "Buch ist damit eher zu gut\n  dargestellt, nicht zu schlecht.")

    # ---------------------------------------------------------------- G5
    print("\nG5  KAPITAL — ist die breite Menge ueberhaupt finanzierbar?")
    je_anzahl = sorted(len(xs) for xs in je_tag.values())
    med_tag = je_anzahl[len(je_anzahl) // 2] if je_anzahl else 0
    max_tag = je_anzahl[-1] if je_anzahl else 0
    print(f"  Kandidaten je Zieltag: Median {med_tag}, Maximum {max_tag}")
    for einsatz in (5.0, 10.0):
        print(f"    bei {einsatz:.0f} $/Lay und zwei ueberlappenden Tagen: "
              f"Median {2*med_tag*einsatz:.0f} $ gebunden, "
              f"Spitze {2*max_tag*einsatz:.0f} $")

    print("\n" + "=" * 78)
    print("Die Gate-Entscheidungen oben sind die aus der Pre-Reg vom 02.08.,")
    print("unveraendert. Reisst G1, wird NICHT geweitet und NICHT nach einem")
    print("anderen Band gesucht — so vorregistriert.")


if __name__ == "__main__":
    main()
