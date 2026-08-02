#!/usr/bin/env python3
"""weather_markt_anker_eval.py — Auswertung zu
`preregs/weather_markt_anker_2026_08_02.md`.

FRAGE: Liefert ein Lay auf `market_fav_k - 1` mehr als eines auf `k0 - 1`?

DER ANLASS (H6 der Ursachen-Messung, ohne Gate): der Bucket unter dem
Markt-Favoriten trifft 19,7 %, der unter unserem eigenen 23,7 % — bei einem
Break-even von 22,6 %. Die eine Zahl liegt darunter, die andere darueber.

WAS H6 NICHT GEMESSEN HAT: die Oekonomie. Ein Bucket, der seltener trifft, ist
teurer — der Markt weiss, wo sein Favorit liegt. Deshalb misst G1 den ROI und
nicht die Trefferquote, und deshalb fuehrt dieses Skript die Preise
POSITIONSWEISE mit.

DIE ZENTRALE KONTROLLE IST G3: Das Preisband 0,70-0,90 waehlt Buckets bereits
nach dem Preis, also nach der Marktmeinung. Der Markt-Anker koennte dasselbe
messen. Erst wenn der Vorteil INNERHALB des Bandes bestehen bleibt — wo beide
Buecher dieselbe Preisklasse handeln — ist er etwas Eigenes.

FENSTER R (12.07.-01.08.) traegt KEIN Gate: die Trefferquoten sind aus der
Ursachen-Messung bekannt. Der Beleg ist Fenster F ab 03.08.

DOKTRIN: Besteht das hier, kommt die Bucket-Wahl aus dem Preis statt aus mu_ens.
Unser Modell traegt dann noch Spannen-Veto, P_pess und Doppel-Kalibrierung — die
Punktprognose nicht mehr. Ein Preis-Anker ist ausserdem nicht diversifizierend:
laeuft der Markt kollektiv falsch, laufen wir mit. Das ist eine Entscheidung des
Betreibers, keine der Messung.

Aufruf:
  python weather_markt_anker_eval.py                # Fenster F (Gates)
  python weather_markt_anker_eval.py --fenster R    # Bezifferung, kein Gate
"""

import argparse
import math
import statistics
import sys
from collections import defaultdict

import pymssql

from weather_ladder_logger import DB_CONFIG

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

USD, FEE = 5.0, 0.07
BAND_LO, BAND_HI = 0.70, 0.90
FENSTER = {"R": ("2026-07-12", "2026-08-01"), "F": ("2026-08-03", "2099-12-31")}
MIN_TAGE, MIN_POS, MIN_STAEDTE, MIN_ABW = 40, 250, 20, 100


def pnl_lay(no, verloren):
    n = USD / no
    fee = FEE * n * min(no, 1 - no)
    return (-USD - fee) if verloren else (n - USD - fee)


def breakeven_q(no):
    n = USD / no
    fee = FEE * n * min(no, 1 - no)
    g, v = n - USD - fee, USD + fee
    return g / (g + v)


def lade(von, bis):
    """Ein Datensatz je Stadt-Tag: beide Anker, volle Leiter, Settlement."""
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor(as_dict=True)
    cur.execute(
        "SELECT snapshot_utc, target_date, city, k, buy_no, offset_fav, "
        "market_fav_k, settle_k FROM bb_WeatherLadders "
        "WHERE var='max' AND kind='eq' AND target_date BETWEEN %s AND %s",
        (von, bis))
    roh = cur.fetchall()
    conn.close()

    neueste = {}
    for r in roh:
        if (r["target_date"] - r["snapshot_utc"].date()).days != 1:
            continue
        key = (str(r["target_date"]), r["city"])
        if key not in neueste or r["snapshot_utc"] > neueste[key]:
            neueste[key] = r["snapshot_utc"]

    je = {}
    for r in roh:
        key = (str(r["target_date"]), r["city"])
        if ((r["target_date"] - r["snapshot_utc"].date()).days != 1
                or r["snapshot_utc"] != neueste.get(key)):
            continue
        d = je.setdefault(key, {"tag": key[0], "city": key[1], "k0": None,
                                "mkt": None, "settle_k": None, "preise": {}})
        if r["settle_k"] is not None:
            d["settle_k"] = int(r["settle_k"])
        if r["market_fav_k"] is not None:
            d["mkt"] = int(r["market_fav_k"])
        if r["offset_fav"] is not None:
            d["k0"] = int(r["k"]) - int(r["offset_fav"])
        if r["buy_no"]:
            d["preise"][int(r["k"])] = float(r["buy_no"])
    return {k: v for k, v in je.items()
            if v["k0"] is not None and v["mkt"] is not None
            and v["settle_k"] is not None}


def buch(saetze, anker, nur_band=False):
    """anker: 'k0' oder 'mkt'. Gelayt wird jeweils der Bucket DARUNTER."""
    posten = []
    for s in saetze.values():
        ziel = s[anker] - 1
        no = s["preise"].get(ziel)
        if not no or (nur_band and not (BAND_LO <= no < BAND_HI)):
            continue
        verloren = (s["settle_k"] == ziel)
        posten.append({"tag": s["tag"], "city": s["city"], "ziel": ziel, "no": no,
                       "verloren": verloren, "pnl": pnl_lay(no, verloren),
                       "be": breakeven_q(no)})
    return posten


def kennzahl(m):
    if not m:
        return None
    p = sum(x["pnl"] for x in m)
    return {"n": len(m), "pnl": p, "roi": 100 * p / (USD * len(m)),
            "quote": 100 * sum(1 for x in m if x["verloren"]) / len(m),
            "be": 100 * statistics.mean(x["be"] for x in m),
            "no": statistics.mean(x["no"] for x in m)}


def tages_roi(m):
    e = defaultdict(list)
    for x in m:
        e[x["tag"]].append(x["pnl"])
    return {t: 100 * sum(v) / (USD * len(v)) for t, v in e.items()}


def gepaart(a, b):
    """t der Tagesdifferenz ROI(a) - ROI(b) auf gemeinsamen Zieltagen."""
    ra, rb = tages_roi(a), tages_roi(b)
    tage = sorted(set(ra) & set(rb))
    diff = [ra[t] - rb[t] for t in tage]
    if len(diff) < 3:
        return float("nan"), len(diff), float("nan")
    m, sd = statistics.mean(diff), statistics.stdev(diff)
    if sd == 0:
        return float("inf"), len(diff), m
    return m / (sd / math.sqrt(len(diff))), len(diff), m


def zeile(name, k):
    if not k:
        print(f"  {name:<26} —")
        return
    print(f"  {name:<26}{k['n']:4d} Lays  Ø NO {k['no']:.3f}  "
          f"Treffer {k['quote']:5.1f} %  be {k['be']:5.1f} %  "
          f"ROI {k['roi']:+7.2f} %")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fenster", choices=["R", "F"], default="F")
    a = ap.parse_args()
    von, bis = FENSTER[a.fenster]

    saetze = lade(von, bis)
    tage = sorted({s["tag"] for s in saetze.values()})
    staedte = sorted({s["city"] for s in saetze.values()})

    print("=" * 78)
    print(f"MARKT-ANKER   Fenster {a.fenster}   {von} .. "
          f"{bis if a.fenster == 'R' else 'offen'}")
    print("=" * 78)
    if not saetze:
        print("Noch keine gesettelten Stadt-Tage mit beiden Ankern.")
        return
    abw = [s for s in saetze.values() if s["k0"] != s["mkt"]]
    print(f"Stadt-Tage {len(saetze)} | Zieltage {len(tage)} | Staedte {len(staedte)}")
    print(f"Anker weichen ab: {len(abw)} ({100*len(abw)/len(saetze):.1f} %)")
    if a.fenster == "R":
        print("\n*** FENSTER R TRAEGT KEIN GATE — die Trefferquoten sind aus der")
        print("*** Ursachen-Messung bekannt. Der Beleg ist Fenster F.")

    A, B = buch(saetze, "k0"), buch(saetze, "mkt")
    kA, kB = kennzahl(A), kennzahl(B)

    # ---------------------------------------------------------------- G0
    if a.fenster == "F":
        g0 = (len(tage) >= MIN_TAGE and min(len(A), len(B)) >= MIN_POS
              and len(staedte) >= MIN_STAEDTE and len(abw) >= MIN_ABW)
        print(f"\nG0  BASIS   Verlangt: >= {MIN_TAGE} Zieltage, >= {MIN_POS} "
              f"Positionen je Buch,\n    >= {MIN_STAEDTE} Staedte, >= {MIN_ABW} "
              f"Abweichungsfaelle  ->  {'BESTANDEN' if g0 else 'GERISSEN'}")
        if not g0:
            print(f"    Fehlen: {max(0, MIN_TAGE-len(tage))} Zieltage, "
                  f"{max(0, MIN_POS-min(len(A), len(B)))} Positionen, "
                  f"{max(0, MIN_ABW-len(abw))} Abweichungsfaelle.")
            print("    Vor G0 wird nicht ausgewertet — so vorregistriert.")
            return

    # ------------------------------------------------------------- G1/G2
    print(f"\nG1  GELD — beide Buecher, identisches Regelwerk ausser dem Anker")
    zeile("A  Anker mu_ens (heute)", kA)
    zeile("B  Anker Markt-Favorit", kB)
    t1, n1, m1 = gepaart(B, A)
    print(f"\n  ROI-Differenz {kB['roi']-kA['roi']:+.2f} pp   "
          f"(Tagesmittel {m1:+.2f} pp, t = {t1:+.2f} ueber {n1} Tage)")
    if a.fenster == "F":
        g1 = (kB["roi"] - kA["roi"]) >= 4.0 and t1 > 2.0
        print(f"  Verlangt: >= 4 pp UND t > 2,0  ->  "
              f"{'BELEGT' if g1 else 'NICHT BELEGT'}")

    print(f"\nG2  MECHANISMUS — kommt der Vorteil aus der Trefferquote?")
    print(f"  Trefferquote B {kB['quote']:.1f} % gegen A {kA['quote']:.1f} %   "
          f"(B unter eigenem Break-even {kB['be']:.1f} %? "
          f"{'ja' if kB['quote'] < kB['be'] else 'nein'})")
    print(f"  Preisaufschlag: Ø NO {kB['no']:.3f} gegen {kA['no']:.3f} "
          f"({kB['no']-kA['no']:+.3f})")
    if a.fenster == "F":
        g2 = kB["quote"] < kA["quote"] and kB["quote"] < kB["be"]
        print(f"  ->  {'BELEGT' if g2 else 'NICHT BELEGT'}")

    # ---------------------------------------------------------------- G3
    print(f"\nG3  EIGENSTAENDIGKEIT — haelt der Vorteil INNERHALB des Preisbands?")
    Ab, Bb = buch(saetze, "k0", True), buch(saetze, "mkt", True)
    kAb, kBb = kennzahl(Ab), kennzahl(Bb)
    zeile(f"A  Band {BAND_LO:.2f}-{BAND_HI:.2f}", kAb)
    zeile(f"B  Band {BAND_LO:.2f}-{BAND_HI:.2f}", kBb)
    if kAb and kBb:
        t3, n3, m3 = gepaart(Bb, Ab)
        print(f"\n  Differenz {kBb['roi']-kAb['roi']:+.2f} pp   "
              f"(Tagesmittel {m3:+.2f} pp, t = {t3:+.2f} ueber {n3} Tage)")
        if a.fenster == "F":
            g3 = (kBb["roi"] - kAb["roi"]) >= 4.0 and t3 > 2.0
            print(f"  Verlangt: >= 4 pp UND t > 2,0  ->  "
                  f"{'BELEGT' if g3 else 'NICHT BELEGT'}")
            print("  Reisst nur G3, ist der Markt-Anker eine Umschreibung des")
            print("  Preisfilters und bringt nichts Eigenes.")

    # ---------------------------------------------------------------- H4
    print(f"\nH4  UNEINIGKEIT (diagnostisch) — dort MUSS der Effekt sitzen")
    keys_abw = {(s["tag"], s["city"]) for s in abw}
    Aa = [x for x in A if (x["tag"], x["city"]) in keys_abw]
    Ba = [x for x in B if (x["tag"], x["city"]) in keys_abw]
    zeile("A  nur Abweichungsfaelle", kennzahl(Aa))
    zeile("B  nur Abweichungsfaelle", kennzahl(Ba))
    if Aa and Ba:
        t4, _, m4 = gepaart(Ba, Aa)
        print(f"  Differenz {kennzahl(Ba)['roi']-kennzahl(Aa)['roi']:+.2f} pp "
              f"(t = {t4:+.2f})")

    # ---------------------------------------------------------------- G4
    if a.fenster == "F":
        print(f"\nG4  ROBUSTHEIT")
        h = len(tage) // 2
        for lbl, tg in (("1. Haelfte", tage[:h]), ("2. Haelfte", tage[h:])):
            ka = kennzahl([x for x in A if x["tag"] in tg])
            kb = kennzahl([x for x in B if x["tag"] in tg])
            if ka and kb:
                print(f"  {lbl}: A {ka['roi']:+.2f} %  B {kb['roi']:+.2f} %  "
                      f"({kb['roi']-ka['roi']:+.2f} pp)")
        je_c = defaultdict(float)
        for x in B:
            je_c[x["city"]] += x["pnl"]
        if je_c:
            stark = max(je_c, key=je_c.get)
            ka = kennzahl([x for x in A if x["city"] != stark])
            kb = kennzahl([x for x in B if x["city"] != stark])
            if ka and kb:
                print(f"  ohne staerkste Stadt ({stark}): "
                      f"{kb['roi']-ka['roi']:+.2f} pp")
        tr = tages_roi(B)
        if tr:
            best = max(tr, key=tr.get)
            ka = kennzahl([x for x in A if x["tag"] != best])
            kb = kennzahl([x for x in B if x["tag"] != best])
            if ka and kb:
                print(f"  ohne besten Zieltag ({best}): "
                      f"{kb['roi']-ka['roi']:+.2f} pp")

    print("\n" + "=" * 78)
    if a.fenster == "R":
        print("Bezifferung, kein Beleg. Die Entscheidung faellt in Fenster F.")
    else:
        print("Bestehen G1-G4, geht NICHTS sofort live: es folgt ein Schattenbuch")
        print("ueber ein zweites Fenster. Die Entscheidung ist dann auch eine")
        print("Doktrin-Entscheidung und gehoert dem Betreiber, nicht der Messung.")


if __name__ == "__main__":
    main()
