#!/usr/bin/env python3
"""
weather_minus1_ppess_filter.py — traegt die eigene Prognose als Auswahlkriterium?

ANLASS (Betreiber, 27.07.2026): Der Autobuy setzte fuer Zieltag 28.07. Warschau 21
zu NO 0,77. Der Markt preist damit 23 % Bucket-Chance — unser eigenes Modell sagt
28 %. Die Wette war schon beim Kauf negativ, und zwar nach UNSERER Rechnung.

URSACHE: Der Autobuy rankt die -1-Kandidaten nach NO-PREIS, also nach dem Markt.
Der Screen rankt seit dem 16.07. nach P_pess aufsteigend (Wetterfrosch-Doktrin:
die eigene Prognose ist die Referenz, der Markt-Favorit ist egal). Die Doktrin ist
nie am Bot angekommen. Solange hoher NO-Preis und grosser Abstand zwischen mu und
Bucket zusammenfallen, faellt das nicht auf; an Tagen wie dem 27.07. faellt es auf.

FRAGE: Haette ein Filter auf den RAND — eingepreiste Bucket-Chance minus eigene
Bucket-Chance — die Auswahl verbessert? Und traegt der Rand ueberhaupt Information
ueber den Ausgang, oder sortiert er nur zufaellig?

  rand = (1 - NO)  -  P_modell        [Prozentpunkte]

  rand > 0  Markt zahlt mehr fuer das Risiko, als unser Modell dafuer verlangt
            = das ist die Lage, in der ein Lay Sinn ergibt
  rand < 0  wir layen etwas, das wir selbst fuer wahrscheinlicher halten, als der
            Markt es bezahlt = negativer Erwartungswert nach eigener Rechnung

DATENBASIS — bewusst das Schattenbuch, nicht die 25 Live-Lays. Bei 25 Positionen
mit EINEM Verlierer kann jeder Filter, der genau diesen einen trifft, brillant
aussehen; das waere Kurvenanpassung an ein einzelnes Ereignis. bb_WeatherLadders
enthaelt jeden -1-Kandidaten mit Vortagspreis und Ausgang, auch die nie gekauften.

  mu_ens/sigma_ens sind die 700d-Lead-24h-kalibrierte Ensemble-Sicht — EINE der
  vier Sichten, aus denen der Screen sein P_pess als MAXIMUM bildet. P_modell hier
  ist also systematisch etwas kleiner als das P_pess des Screens, der Rand
  entsprechend etwas guenstiger. Wer die Schwelle spaeter vorregistriert, muss sie
  gegen dieselbe Groesse definieren, die der Bot zur Laufzeit berechnen kann.

RECHNUNG identisch zu weather_minus1_shadow (5 $/Lay, Fee 0,07*n*min(NO,1-NO)),
damit die Zahlen direkt vergleichbar bleiben.

SIGNIFIKANZ: t ueber TAGES-Mittel, nicht ueber Einzelpositionen. Staedte desselben
Tages haengen an derselben Wetterlage (Moskau und Warschau lagen am 28.07. in
einer Luftmasse) — positionsweise gerechnet waere das t zu gross.

Aufruf:
  python weather_minus1_ppess_filter.py
  python weather_minus1_ppess_filter.py --von 2026-07-20 --schwellen -10,-5,0,5,10
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


def pnl_lay(no, verloren):
    """PnL eines 5-$-Lays. verloren=True heisst: der Bucket ist eingetroffen."""
    n = USD / no
    fee = FEE * n * min(no, 1.0 - no)
    return (-USD if verloren else n - USD - fee)


def p_bucket(mu, sigma, k, city):
    """P(Tageshoch faellt in Bucket k) unter Normal(mu, sigma).

    Die Grenzen kommen aus weather_stations.bucket_grenzen — half_up fuer die
    meisten Staedte, [k, k+1) fuer die BUCKET_FLOOR-Staedte (Hong Kong). Wer hier
    die falschen Grenzen nimmt, liegt bei sigma~1 um ein halbes Sigma daneben.
    """
    if not sigma or sigma <= 0:
        return None
    ub, ob = bucket_grenzen(k, city)
    phi = lambda x: 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))
    return phi(ob) - phi(ub)


def lade_kandidaten(von):
    """Alle -1-Fenster mit Vortags-Snapshot, mu/sigma und bekanntem Ausgang.

    Deduplizierung ueber den SPAETESTEN Vortags-Snapshot je (Zieltag, Stadt, k) —
    noetig, weil der Ladder-Logger am 26.07. dreimal lief (2552 statt ~680 Zeilen).
    """
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor(as_dict=True)
    cur.execute("""
        SELECT city, k, buy_no, target_date, settle_result, snapshot_utc,
               mu_ens, sigma_ens
        FROM bb_WeatherLadders
        WHERE var='max' AND kind='eq' AND offset_fav=-1 AND status='open'
          AND buy_no IS NOT NULL AND buy_no > 0 AND buy_no < 1
          AND settle_result IS NOT NULL
          AND mu_ens IS NOT NULL AND sigma_ens IS NOT NULL
          AND CAST(snapshot_utc AS date) < target_date
          AND target_date >= %s
    """, (von,))
    rows = cur.fetchall()
    conn.close()

    best = {}
    for r in rows:
        key = (str(r["target_date"]), r["city"], r["k"])
        if key not in best or r["snapshot_utc"] > best[key]["snapshot_utc"]:
            best[key] = r

    posten = []
    for (tag, city, k), r in best.items():
        p = p_bucket(r["mu_ens"], r["sigma_ens"], k, city)
        if p is None:
            continue
        no = float(r["buy_no"])
        verloren = bool(r["settle_result"])   # settle_result=1: Bucket eingetroffen
        posten.append(dict(
            tag=tag, city=city, k=k, no=no, mu=r["mu_ens"], sigma=r["sigma_ens"],
            eingepreist=1.0 - no,
            p_modell=p,
            rand=(1.0 - no) - p,              # in Anteilen, Ausgabe in Punkten
            verloren=verloren,
            pnl=pnl_lay(no, verloren),
        ))
    return posten


def kennzahlen(posten):
    if not posten:
        return None
    n = len(posten)
    pnl = sum(p["pnl"] for p in posten)
    verl = sum(1 for p in posten if p["verloren"])
    return dict(n=n, pnl=pnl, einsatz=USD * n, roi=pnl / (USD * n),
                verl=verl, verlq=verl / n)


def t_tagesweise(posten):
    """t-Wert ueber Tages-Mittel des ROI je Lay. None bei < 2 Tagen."""
    je_tag = defaultdict(list)
    for p in posten:
        je_tag[p["tag"]].append(p["pnl"] / USD)
    mittel = [statistics.fmean(v) for v in je_tag.values()]
    if len(mittel) < 2:
        return None, len(mittel)
    sd = statistics.stdev(mittel)
    if sd == 0:
        return None, len(mittel)
    return statistics.fmean(mittel) / (sd / math.sqrt(len(mittel))), len(mittel)


def zeile(titel, posten, basis_roi=None):
    kz = kennzahlen(posten)
    if not kz:
        print(f"  {titel:<24}      keine Kandidaten")
        return
    t, ntage = t_tagesweise(posten)
    tt = f"{t:5.2f}" if t is not None else "  —  "
    delta = ""
    if basis_roi is not None:
        delta = f"  ({(kz['roi'] - basis_roi) * 100:+5.2f} pp)"
    print(f"  {titel:<24} {kz['n']:4d} Lays  {kz['pnl']:+8.2f} $  "
          f"ROI {kz['roi'] * 100:+6.2f} %  Verl {kz['verl']:3d}/{kz['n']:<3d} "
          f"({kz['verlq'] * 100:4.1f} %)  t {tt} ({ntage}d){delta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--von", default="2026-07-20", help="fruehester Zieltag")
    ap.add_argument("--schwellen", default="-10,-5,0,2,5,8,10",
                    help="Rand-Schwellen in Prozentpunkten, kommasepariert")
    a = ap.parse_args()

    posten = lade_kandidaten(a.von)
    if not posten:
        print("Keine Kandidaten mit mu/sigma und Settlement gefunden.")
        return

    tage = sorted({p["tag"] for p in posten})
    basis = kennzahlen(posten)
    print(f"Schattenbuch der -1-Klasse ab {a.von}: {basis['n']} Kandidaten, "
          f"{len(tage)} Zieltage ({tage[0]} .. {tage[-1]})")
    print(f"P_modell aus mu_ens/sigma_ens (700d-Lead-24h-Ensemble) — eine Sicht, "
          f"nicht das P_pess-Maximum des Screens.\n")

    print("RAND-VERTEILUNG (eingepreist minus eigenes Modell, in Punkten)")
    raender = sorted(p["rand"] * 100 for p in posten)
    q = lambda f: raender[min(int(f * len(raender)), len(raender) - 1)]
    print(f"  min {raender[0]:+6.1f} | q25 {q(.25):+6.1f} | median {q(.5):+6.1f} | "
          f"q75 {q(.75):+6.1f} | max {raender[-1]:+6.1f}")
    neg = sum(1 for p in posten if p["rand"] < 0)
    print(f"  {neg} von {len(posten)} Kandidaten ({neg / len(posten) * 100:.0f} %) haben "
          f"NEGATIVEN Rand — nach eigener Rechnung schlechte Wetten.\n")

    print("TRENNSCHAERFE — traegt der Rand Information ueber den Ausgang?")
    gew = [p["rand"] * 100 for p in posten if not p["verloren"]]
    verl = [p["rand"] * 100 for p in posten if p["verloren"]]
    if gew and verl:
        print(f"  Rand bei Gewinnern:  {statistics.fmean(gew):+6.2f} pp  (n={len(gew)})")
        print(f"  Rand bei Verlierern: {statistics.fmean(verl):+6.2f} pp  (n={len(verl)})")
        print(f"  Differenz:           {statistics.fmean(gew) - statistics.fmean(verl):+6.2f} pp"
              f"   {'(Rand zeigt in die erwartete Richtung)' if statistics.fmean(gew) > statistics.fmean(verl) else '(FALSCHE Richtung!)'}")
    else:
        print("  (nur eine Ausgangsklasse vorhanden)")

    print("\nFILTER-SWEEP — nur layen, wenn rand >= Schwelle")
    zeile("ohne Filter", posten)
    print()
    for s in [float(x) for x in a.schwellen.split(",")]:
        gefiltert = [p for p in posten if p["rand"] * 100 >= s]
        zeile(f"rand >= {s:+.0f} pp", gefiltert, basis["roi"])

    print("\nZUM VERGLEICH — Ranking nach NO-Preis (das, was der Bot heute tut)")
    nach_no = sorted(posten, key=lambda p: -p["no"])
    for anteil in (0.2, 0.4, 0.6):
        m = max(1, int(len(posten) * anteil))
        zeile(f"beste {int(anteil * 100)} % nach NO", nach_no[:m], basis["roi"])
    nach_rand = sorted(posten, key=lambda p: -p["rand"])
    print()
    for anteil in (0.2, 0.4, 0.6):
        m = max(1, int(len(posten) * anteil))
        zeile(f"beste {int(anteil * 100)} % nach Rand", nach_rand[:m], basis["roi"])

    # Der Rand ist per Konstruktion an den NO-Preis gekoppelt (rand = (1-NO) - P).
    # Ein Sweep ueber den Rand kann deshalb bloss den Preis rueckwaerts messen. Die
    # eigentliche Frage ist, ob P_modell ZUSAETZLICH zum Preis etwas weiss: teile
    # in Preisbaender und vergleiche INNERHALB des Bandes die Haelfte mit hohem
    # gegen die mit niedrigem P_modell. Traegt das Modell Information, muessen die
    # niedrigen P haeufiger gewinnen.
    print("\nKONDITIONAL — weiss P_modell etwas, das im NO-Preis noch nicht steckt?")
    baender = [(0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.70, 0.85), (0.0, 0.70)]
    for lo, hi in baender:
        grp = [p for p in posten if lo <= p["no"] < hi]
        if len(grp) < 6:
            print(f"  NO {lo:.2f}-{hi:.2f}: nur {len(grp)} Kandidaten — zu duenn")
            continue
        med = statistics.median(p["p_modell"] for p in grp)
        tief = [p for p in grp if p["p_modell"] <= med]
        hoch = [p for p in grp if p["p_modell"] > med]
        kt, kh = kennzahlen(tief), kennzahlen(hoch)
        print(f"  NO {lo:.2f}-{hi:.2f}  (n={len(grp):3d}, Median P_modell {med * 100:4.1f} %)")
        print(f"     P_modell TIEF : {kt['n']:3d} Lays  ROI {kt['roi'] * 100:+7.2f} %  "
              f"Verl {kt['verl']}/{kt['n']}")
        print(f"     P_modell HOCH : {kh['n']:3d} Lays  ROI {kh['roi'] * 100:+7.2f} %  "
              f"Verl {kh['verl']}/{kh['n']}"
              f"   -> {'Modell hilft' if kt['roi'] > kh['roi'] else 'Modell schadet'}")

    # Der eigentliche Test: WER ist besser kalibriert? Beide Seiten geben eine
    # Wahrscheinlichkeit ab (Modell: P_bucket unter Normal; Markt: 1-NO). Bins
    # bilden, in jedem Bin die tatsaechliche Trefferquote dagegenhalten. Ein
    # kalibrierter Schaetzer liegt in jedem Bin auf der Diagonalen.
    # Wo im Preisspektrum sitzt der Edge? Der Bot nimmt heute die HOECHSTEN
    # NO-Preise ("die konservativsten"). Ob das die ertragreichsten sind, war nie
    # gemessen — hier steht es.
    # DIE DOKTRIN-KONFORME GROESSE (Betreiber-Einwand 27.07.): nicht P_bucket,
    # sondern der rohe TEMPERATURABSTAND aus der eigenen Prognose. P vermischt
    # Abstand und Sigma; ist Sigma schlecht geschaetzt, verdirbt es den Abstand
    # mit. Der Abstand allein braucht kein Sigma und keinen Preis.
    #
    # Bei einem -1-Lay liegt mu immer im Bucket UEBER dem gelayten, der Abstand
    # zur oberen Grenze des gelayten Buckets liegt also konstruktionsbedingt in
    # [0, 1) K. Er misst genau, wo mu innerhalb des Favoriten-Buckets sitzt:
    # knapp ueber der Kante (gefaehrlich) oder weit oben (sicher).
    print("\nTEMPERATURABSTAND — mu ueber der Oberkante des gelayten Buckets")
    for p in posten:
        _ub, ob = bucket_grenzen(p["k"], p["city"])
        p["abstand"] = p["mu"] - ob
    absts = sorted(p["abstand"] for p in posten)
    qa = lambda f: absts[min(int(f * len(absts)), len(absts) - 1)]
    print(f"  min {absts[0]:+.2f} K | q25 {qa(.25):+.2f} | median {qa(.5):+.2f} | "
          f"q75 {qa(.75):+.2f} | max {absts[-1]:+.2f} K")
    gew_a = [p["abstand"] for p in posten if not p["verloren"]]
    verl_a = [p["abstand"] for p in posten if p["verloren"]]
    if gew_a and verl_a:
        d = statistics.fmean(gew_a) - statistics.fmean(verl_a)
        print(f"  Abstand bei Gewinnern:  {statistics.fmean(gew_a):+.2f} K  (n={len(gew_a)})")
        print(f"  Abstand bei Verlierern: {statistics.fmean(verl_a):+.2f} K  (n={len(verl_a)})")
        print(f"  Differenz:              {d:+.2f} K   "
              f"{'(weiter weg = sicherer, wie erwartet)' if d > 0 else '(FALSCHE Richtung)'}")

    print("\n  Filter auf den Abstand — nur layen, wenn mu weit genug ueber der Kante:")
    for s in (0.20, 0.35, 0.50, 0.65, 0.80):
        zeile(f"abstand >= {s:.2f} K", [p for p in posten if p["abstand"] >= s], basis["roi"])

    print("\n  Konditional: traegt der Abstand INNERHALB der Preisbaender?")
    for lo, hi in ((0.85, 1.01), (0.70, 0.85), (0.0, 0.70)):
        grp = [p for p in posten if lo <= p["no"] < hi]
        if len(grp) < 6:
            continue
        med = statistics.median(p["abstand"] for p in grp)
        nah = [p for p in grp if p["abstand"] <= med]
        weit = [p for p in grp if p["abstand"] > med]
        kn, kw = kennzahlen(nah), kennzahlen(weit)
        print(f"     NO {lo:.2f}-{hi:.2f} (n={len(grp):3d}, Median-Abstand {med:+.2f} K)")
        print(f"        mu NAH an der Kante : {kn['n']:3d} Lays  ROI {kn['roi'] * 100:+7.2f} %  "
              f"Verl {kn['verl']}/{kn['n']}")
        print(f"        mu WEIT von Kante   : {kw['n']:3d} Lays  ROI {kw['roi'] * 100:+7.2f} %  "
              f"Verl {kw['verl']}/{kw['n']}"
              f"   -> {'Abstand traegt' if kw['roi'] > kn['roi'] else 'Abstand traegt NICHT'}")

    print("\nROI JE PREISBAND — wo sitzt der Edge wirklich?")
    for lo, hi in ((0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.75, 0.85),
                   (0.70, 0.75), (0.0, 0.70)):
        grp = [p for p in posten if lo <= p["no"] < hi]
        if grp:
            zeile(f"NO {lo:.2f}-{hi:.2f}", grp, basis["roi"])

    # Ergaenzen sich Prognose und Preis, oder messen sie dasselbe? EXPLORATIV —
    # auf 7 Tagen ist jede Kombination schnell ueberangepasst. Die Zahlen taugen
    # als Richtungsangabe fuer eine Pre-Reg, nicht als Freigabe.
    print("\nKOMBINATION (explorativ!) — eigene Prognose UND Preisband")
    zeile("alles", posten, basis["roi"])
    band = [p for p in posten if 0.70 <= p["no"] < 0.90]
    zeile("nur Band 0,70-0,90", band, basis["roi"])
    absf = [p for p in posten if p["abstand"] >= 0.50]
    zeile("nur abstand >= 0,50 K", absf, basis["roi"])
    beides = [p for p in posten if 0.70 <= p["no"] < 0.90 and p["abstand"] >= 0.50]
    zeile("beides", beides, basis["roi"])
    print(f"  Ueberschneidung: {len(beides)} von {len(band)} im Band bzw. "
          f"{len(absf)} mit Abstand — die Kriterien sind {'weitgehend deckungsgleich' if len(beides) > 0.7 * min(len(band), len(absf)) else 'komplementaer'}.")

    print("\nKALIBRIERUNG — Modell gegen Markt, je Wahrscheinlichkeits-Bin")
    for name, key in (("EIGENES MODELL", "p_modell"), ("MARKT (1-NO)", "eingepreist")):
        print(f"  {name}")
        for lo, hi in ((0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 1.0)):
            grp = [p for p in posten if lo <= p[key] < hi]
            if not grp:
                continue
            gesagt = statistics.fmean(p[key] for p in grp)
            real = sum(1 for p in grp if p["verloren"]) / len(grp)
            ab = (real - gesagt) * 100
            print(f"     {lo * 100:3.0f}-{hi * 100:3.0f} %: n={len(grp):3d}  "
                  f"gesagt {gesagt * 100:5.1f} %  eingetreten {real * 100:5.1f} %  "
                  f"Abweichung {ab:+6.1f} pp")

    print("\nDIE SCHLECHTESTEN KANDIDATEN NACH EIGENEM MODELL (rand aufsteigend)")
    for p in sorted(posten, key=lambda p: p["rand"])[:8]:
        print(f"  {p['tag']}  {p['city']:<14} {p['k']:3d}°C  NO {p['no']:.2f}  "
              f"eingepreist {p['eingepreist'] * 100:4.1f} %  eigenes {p['p_modell'] * 100:4.1f} %  "
              f"rand {p['rand'] * 100:+5.1f} pp  {'VERLOREN' if p['verloren'] else 'gewonnen'}")


if __name__ == "__main__":
    main()
