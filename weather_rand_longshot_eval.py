#!/usr/bin/env python3
"""weather_rand_longshot_eval.py — Auswertung zu
`preregs/weather_rand_longshot_2026_08_02.md`.

FRAGE: Trifft der Rand der Bucket-Leiter (|offset_fav| >= 2) seltener, als er
bepreist wird — und reicht die Differenz nach Gebuehr?

FORWARD. Das retrospektive Fenster 12.07.-01.08. ist fuer diese Frage verbraucht
(die -2-Zahlen wurden am 02.08. vollstaendig gesehen). Gewertet wird ab Zieltag
03.08.2026. Wer dieses Skript auf ein frueheres Fenster richtet, misst die Daten,
die die These erzeugt haben.

WARUM DER GANZE RAND UND NICHT NUR -2: Teststaerke. Bei einer bepreisten Quote
von 7,3 % gegen eine realisierte von 5,7 % braucht z = 2 rund 1.050 Positionen.
Die -2-Klasse liefert 15 je Zieltag (70 Zieltage), der ganze Rand 56 (19
Zieltage, mit Aufschlag fuer Tageskorrelation 30). Der Haupttest ist deshalb der
Mechanismus ueber den ganzen Rand; -2 allein ist der nachgeordnete Spezialfall.

DER FEHLER, DEN DIESES SKRIPT NICHT WIEDERHOLT: G5 der Vorgaenger-Pre-Reg zog
seine Schwelle aus dem MEDIAN-Preis 0,960, waehrend der gehandelte Mittelpreis
bei 0,922 lag — bei linksschiefer Preisverteilung beschreibt ein Median nichts.
Der Break-even wird hier IMMER positionsweise aus dem echten Preis gerechnet.
Und G5 hatte keine t-Bedingung; der Befund stand bei t ~ 1,1.

SIGNIFIKANZ: t ueber TAGES-Mittel. Bei 56 Positionen je Zieltag ist das der
Unterschied zwischen einem belastbaren und einem grob ueberhoehten t.

Aufruf:
  python weather_rand_longshot_eval.py                 # Hauptlauf ab 03.08.
  python weather_rand_longshot_eval.py --zwischenschau # Sicherheitscheck, nur nach unten
  python weather_rand_longshot_eval.py --ohne Shenzhen "Hong Kong"
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

USD = 5.0
FEE_HAUPT = 0.07
FEE_ALT = 0.04
RAND = (-3, -2, 2, 3)
START = "2026-08-03"
MIN_TAGE, MIN_POS, MIN_STAEDTE = 30, 1400, 20
ZWISCHEN_TAGE, ZWISCHEN_STOPP = 15, -5.0


def pnl_lay(no, verloren, fee_rate):
    n = USD / no
    fee = fee_rate * n * min(no, 1 - no)
    return (-USD - fee) if verloren else (n - USD - fee)


def breakeven_q(no, fee_rate):
    """Positionsweise — nie aus einem Median. Das war der Fehler vom 02.08."""
    n = USD / no
    fee = fee_rate * n * min(no, 1 - no)
    gewinn, verlust = n - USD - fee, USD + fee
    return gewinn / (gewinn + verlust)


def ein_stichproben_t(werte):
    xs = [x for x in werte if x is not None and not math.isnan(x)]
    if len(xs) < 3:
        return float("nan"), len(xs), float("nan")
    m, sd = statistics.mean(xs), statistics.stdev(xs)
    if sd == 0:
        return float("inf"), len(xs), m
    return m / (sd / math.sqrt(len(xs))), len(xs), m


def lade(start, lead, ohne):
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor(as_dict=True)
    cur.execute(
        "SELECT snapshot_utc, target_date, city, k, buy_no, buy_yes, mu_ens, "
        "sigma_ens, offset_fav, settle_k FROM bb_WeatherLadders "
        "WHERE var='max' AND kind='eq' AND target_date >= %s", (start,))
    roh = cur.fetchall()
    conn.close()

    def lead_of(r):
        return (r["target_date"] - r["snapshot_utc"].date()).days

    neueste = {}
    for r in roh:
        if lead_of(r) != lead:
            continue
        key = (str(r["target_date"]), r["city"])
        if key not in neueste or r["snapshot_utc"] > neueste[key]:
            neueste[key] = r["snapshot_utc"]

    settle = {}
    for r in roh:
        if r["settle_k"] is not None:
            settle[(str(r["target_date"]), r["city"])] = int(r["settle_k"])

    posten = []
    for r in roh:
        key = (str(r["target_date"]), r["city"])
        if (lead_of(r) != lead or r["snapshot_utc"] != neueste.get(key)
                or r["city"] in ohne or r["offset_fav"] is None
                or int(r["offset_fav"]) not in RAND or not r["buy_no"]
                or key not in settle):
            continue
        o = int(r["offset_fav"])
        k0 = int(r["k"]) - o
        no = float(r["buy_no"])
        verloren = (settle[key] - k0 == o)
        posten.append({
            "tag": key[0], "city": key[1], "off": o, "no": no,
            "yes": float(r["buy_yes"]) if r["buy_yes"] else None,
            "verloren": verloren,
            "pnl": pnl_lay(no, verloren, FEE_HAUPT),
            "pnl_alt": pnl_lay(no, verloren, FEE_ALT),
            "be": breakeven_q(no, FEE_HAUPT),
        })
    return sorted(posten, key=lambda p: (p["tag"], p["city"], p["off"]))


def kennzahl(menge, feld="pnl"):
    if not menge:
        return None
    p = sum(x[feld] for x in menge)
    v = sum(1 for x in menge if x["verloren"])
    mkt = [((1 - x["no"]) + x["yes"]) / 2 for x in menge if x["yes"]]
    return {"n": len(menge), "pnl": p, "roi": 100 * p / (USD * len(menge)),
            "ist": 100 * v / len(menge),
            "markt": 100 * statistics.mean(mkt) if mkt else float("nan"),
            "be": 100 * statistics.mean(x["be"] for x in menge),
            "no": statistics.mean(x["no"] for x in menge)}


def tages(menge, f):
    eimer = defaultdict(list)
    for x in menge:
        v = f(x)
        if v is not None:
            eimer[x["tag"]].append(v)
    return {t: statistics.mean(v) for t, v in eimer.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--ohne", nargs="*", default=[], metavar="STADT")
    ap.add_argument("--zwischenschau", action="store_true",
                    help="Sicherheitscheck nach 15 Zieltagen — darf NUR nach unten entscheiden")
    a = ap.parse_args()

    posten = lade(a.start, a.lead, set(a.ohne))
    tage = sorted({p["tag"] for p in posten})
    staedte = sorted({p["city"] for p in posten})

    print("=" * 78)
    print(f"RAND-LONGSHOT   ab {a.start}   Lead {a.lead}"
          + (f"   ohne {', '.join(sorted(a.ohne))}" if a.ohne else ""))
    print("=" * 78)
    if not posten:
        print("Noch keine gesettelten Randpositionen im Forward-Fenster.")
        return
    print(f"Randpositionen {len(posten)} | Zieltage {len(tage)} "
          f"({tage[0]} .. {tage[-1]}) | Staedte {len(staedte)}")

    ges = kennzahl(posten)

    # ------------------------------------------------- Zwischenschau
    if a.zwischenschau:
        print(f"\nSICHERHEITS-ZWISCHENSCHAU (Sequenzregel 1 der Pre-Reg)")
        print(f"  Zieltage {len(tage)} von {ZWISCHEN_TAGE} | ROI {ges['roi']:+.2f} %")
        if len(tage) < ZWISCHEN_TAGE:
            print("  Noch nicht faellig.")
        elif ges["roi"] < ZWISCHEN_STOPP:
            print(f"  ROI unter {ZWISCHEN_STOPP:+.0f} %  ->  ABBRUCH laut Pre-Reg.")
        else:
            print("  Kein Abbruch. Ein gutes Zwischenergebnis fuehrt zu NICHTS —")
            print("  insbesondere nicht zu einem frueheren Live-Gang.")
        return

    # ---------------------------------------------------------- G0
    print(f"\nG0  LAUFZEIT UND BASIS")
    g0 = (len(tage) >= MIN_TAGE and len(posten) >= MIN_POS
          and len(staedte) >= MIN_STAEDTE)
    print(f"  Verlangt: >= {MIN_TAGE} Zieltage, >= {MIN_POS} Positionen, "
          f">= {MIN_STAEDTE} Staedte  ->  {'BESTANDEN' if g0 else 'GERISSEN'}")
    if not g0:
        print(f"  Noch nicht auswertbar. Fehlen: {max(0, MIN_TAGE-len(tage))} Zieltage, "
              f"{max(0, MIN_POS-len(posten))} Positionen.")
        print("  Vor G0 wird nicht ausgewertet — so vorregistriert.")
        return

    # ---------------------------------------------------------- G1
    print(f"\nG1  MECHANISMUS — trifft der Rand seltener, als er bepreist wird?")
    print(f"  P_ist {ges['ist']:.2f} %   P_markt {ges['markt']:.2f} %   "
          f"Break-even {ges['be']:.2f} %   (Ø NO {ges['no']:.3f})")
    d1 = tages(posten, lambda x: ((1.0 if x["verloren"] else 0.0)
                                  - (((1 - x["no"]) + x["yes"]) / 2 if x["yes"] else None))
               if x["yes"] else None)
    t1, n1, m1 = ein_stichproben_t(list(d1.values()))
    print(f"  P_markt - P_ist = {ges['markt']-ges['ist']:+.2f} pp   "
          f"(Tagesmittel {-100*m1:+.2f} pp, t = {-t1:+.2f} ueber {n1} Tage)")
    g1 = (ges["markt"] - ges["ist"]) >= 1.5 and -t1 > 2.0
    print(f"  Verlangt: >= 1,5 pp UND t > 2,0  ->  {'BELEGT' if g1 else 'NICHT BELEGT'}")

    print("\n  je Randklasse (diagnostisch):")
    for o in RAND:
        k = kennzahl([p for p in posten if p["off"] == o])
        if k:
            print(f"    {o:+d}: {k['n']:4d} Lays  Ø NO {k['no']:.3f}  "
                  f"ist {k['ist']:5.2f} %  markt {k['markt']:5.2f} %  "
                  f"be {k['be']:5.2f} %  ROI {k['roi']:+6.2f} %")

    # ---------------------------------------------------------- G2
    print(f"\nG2  GELD — ueberlebt der Mechanismus die Gebuehr?")
    k_alt = kennzahl(posten, "pnl_alt")
    print(f"  ROI Fee 0,07 {ges['roi']:+.2f} %  ({ges['pnl']:+.2f} $ auf "
          f"{USD*len(posten):.0f} $)   |   Fee 0,04 {k_alt['roi']:+.2f} %")
    t2, n2, m2 = ein_stichproben_t(list(tages(posten, lambda x: x["pnl"]).values()))
    print(f"  Tagesmittel {m2:+.4f} $/Position, t = {t2:+.2f} ueber {n2} Tage")
    g2 = ges["roi"] > 0 and t2 > 2.0
    print(f"  Verlangt: ROI > 0 UND t > 2,0  ->  {'BELEGT' if g2 else 'NICHT BELEGT'}")

    # ---------------------------------------------------------- G3
    print(f"\nG3  ROBUSTHEIT")
    mitte = len(tage) // 2
    h1 = [p for p in posten if p["tag"] in tage[:mitte]]
    h2 = [p for p in posten if p["tag"] in tage[mitte:]]
    r1, r2 = kennzahl(h1), kennzahl(h2)
    print(f"  1. Haelfte ({tage[0]}..{tage[mitte-1]}): ROI {r1['roi']:+.2f} % (n={r1['n']})")
    print(f"  2. Haelfte ({tage[mitte]}..{tage[-1]}): ROI {r2['roi']:+.2f} % (n={r2['n']})")
    tr = tages(posten, lambda x: x["pnl"])
    best_tag = max(tr, key=tr.get)
    ohne_tag = kennzahl([p for p in posten if p["tag"] != best_tag])
    je_stadt = defaultdict(float)
    for p in posten:
        je_stadt[p["city"]] += p["pnl"]
    best_stadt = max(je_stadt, key=je_stadt.get)
    ohne_stadt = kennzahl([p for p in posten if p["city"] != best_stadt])
    print(f"  ohne besten Tag ({best_tag}): ROI {ohne_tag['roi']:+.2f} %")
    print(f"  ohne staerkste Stadt ({best_stadt}): ROI {ohne_stadt['roi']:+.2f} %")
    anteil = (100 * (ges["pnl"] - ohne_tag["pnl"]) / ges["pnl"]) if ges["pnl"] else float("nan")
    print(f"  bester Zieltag traegt {anteil:.1f} % des Effekts (Grenze 35 %)")
    g3 = ((r1["roi"] > 0) == (r2["roi"] > 0) and ohne_tag["roi"] > 0
          and ohne_stadt["roi"] > 0 and abs(anteil) <= 35.0)
    print(f"  ->  {'BESTANDEN' if g3 else 'GERISSEN'}")

    # ---------------------------------------------------------- G4
    print(f"\nG4  DIE -2-KLASSE (nachgeordnet, schwaechere Schwelle)")
    m2menge = [p for p in posten if p["off"] == -2]
    k2 = kennzahl(m2menge)
    if k2:
        t4, n4, _ = ein_stichproben_t(list(tages(m2menge, lambda x: x["pnl"]).values()))
        print(f"  {k2['n']} Lays  Ø NO {k2['no']:.3f}  Treffer {k2['ist']:.2f} %  "
              f"gegen Break-even {k2['be']:.2f} %  ROI {k2['roi']:+.2f} %  t = {t4:+.2f}")
        g4 = k2["roi"] > 0 and t4 > 1.5
        print(f"  Verlangt: ROI > 0 UND t > 1,5  ->  "
              f"{'BELEGT' if g4 else 'NICHT BELEGT'}")
        print(f"  Erwartet wurde: nach {MIN_TAGE} Zieltagen nicht entscheidbar "
              f"(noetig ~1.050 Positionen).")

    # ---------------------------------------------------------- H4
    print(f"\nH4  PREISRICHTUNG (diagnostisch, EIN Schnitt am Median-NO)")
    med = statistics.median(p["no"] for p in posten)
    billig = kennzahl([p for p in posten if p["no"] < med])
    teuer = kennzahl([p for p in posten if p["no"] >= med])
    print(f"  Median-NO {med:.3f}")
    print(f"    billiger: {billig['n']:4d} Lays  ROI {billig['roi']:+6.2f} %  "
          f"Treffer {billig['ist']:.2f} % gegen be {billig['be']:.2f} %")
    print(f"    teurer  : {teuer['n']:4d} Lays  ROI {teuer['roi']:+6.2f} %  "
          f"Treffer {teuer['ist']:.2f} % gegen be {teuer['be']:.2f} %")
    print("  These war: die teuren tragen (Umkehrung der -1-Doktrin).")

    # ------------------------------------------------------ Sequenz
    print("\n" + "=" * 78)
    if g1 and g2 and g3:
        print("G1-G3 BELEGT. Es geht NICHTS live: es folgt ein zweites Fenster als")
        print("Schattenbuch mit echten Fill-Preisen aus der Polymarket-API — ein")
        print("2-%-Effekt ist mit Snapshot-Preisen nicht zu bestaetigen (Designfalle 5).")
    elif 1.0 < -t1 <= 2.0 or (g1 and 1.0 < t2 <= 2.0):
        print("t zwischen 1,0 und 2,0  ->  EINMALIGE Verlaengerung auf 60 Zieltage")
        print("(Sequenzregel 2, vorab festgelegt — keine nachtraegliche Rettung).")
    else:
        print("Mechanismus nicht belegt. Es wird NICHT nach einer Randklasse, einer")
        print("anderen Offset-Grenze oder einem Preisband gesucht, in dem es traegt.")
    print("\nG5 (Buchtiefe bei NO >= 0,93, Kapitalbindung) ist NICHT Teil dieses")
    print("Skripts — er verlangt echte Buecher aus der Polymarket-Public-Data-API")
    print("und entscheidet ueber Handelbarkeit, nicht ueber Wahrheit.")


if __name__ == "__main__":
    main()
