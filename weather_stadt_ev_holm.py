#!/usr/bin/env python3
"""Holm-Bonferroni ueber alle Stadt-x-Offset-Zellen.

FRAGE DES BETREIBERS (06.08.2026): "rechne mal holm-bonferroni".

WARUM HOLM: Die einfache Bonferroni-Korrektur (alpha/m fuer jeden Test) ist
konservativ — sie kontrolliert die FWER, verschenkt aber Macht. Holm (1979)
kontrolliert dieselbe Fehlerrate und ist dabei gleichmaessig maechtiger: die
p-Werte werden aufsteigend sortiert und gegen alpha/(m-i+1) geprueft, die
Schwelle wird also mit jedem verworfenen Test milder. Was Bonferroni findet,
findet Holm immer auch; umgekehrt nicht.

WAS SICH GEGENUEBER weather_stadt_ev.py AUSSERDEM AENDERT — der Test selbst:
Dort lief ein Binomialtest gegen den MITTELPREIS der Zelle. Das ist aus
demselben Grund unsauber, aus dem der EV aus dem Mittelpreis unsauber war: es
behandelt 15 Wetten zu unterschiedlichen Preisen, als waeren sie eine Wette zum
Durchschnittspreis. Hier stattdessen ein exakter Monte-Carlo-Test:

    H0: der Markt ist fair — Bucket i trifft mit genau seiner Wahrschein-
        lichkeit p_i (= sein Preis).
    Statistik: der positionsweise Gesamt-ROI der Zelle.
    p-Wert: Anteil der Simulationen unter H0 mit ROI >= beobachtet.

Das ist eine Poisson-Binomial-Situation (jede Position hat ihr eigenes p) und
analytisch unhandlich, per Simulation aber exakt bis auf den MC-Fehler.

Der p-Wert ist EINSEITIG (nur "besser als fair" ist interessant).
"""
import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pymssql

DB_CONFIG = {"server": "158.181.48.77", "database": "dbdata",
             "user": "326773", "password": "Extaler11!"}
FEE = 0.07


def lade(conn, var, von=None, bis=None):
    cur = conn.cursor()
    q = ("SELECT city, target_date, k, buy_yes, offset_fav, market_fav_k, "
         "settle_k, wu_settle_k, snapshot_utc FROM bb_WeatherLadders "
         "WHERE var=%s AND kind='eq' AND (settle_k IS NOT NULL OR wu_settle_k IS NOT NULL) "
         "AND offset_fav IS NOT NULL AND buy_yes > 0")
    p = [var]
    if von:
        q += " AND target_date >= %s"
        p.append(von)
    if bis:
        q += " AND target_date <= %s"
        p.append(bis)
    cur.execute(q, tuple(p))
    best = {}
    for city, td, k, yes, off, mfav, sk, wu, snap in cur.fetchall():
        if isinstance(td, datetime):
            td = td.date()
        if snap.date() != td - timedelta(days=1):      # Lead 1
            continue
        key = (city, str(td), int(k))
        if key not in best or snap > best[key][-1]:
            best[key] = (float(yes), int(off), mfav,
                         int(wu if wu is not None else sk), snap)
    return best


def roi_vec(preise, treffer):
    """Positionsweiser ROI je Wette, vektorisiert. treffer: bool-Array."""
    n = 1.0 / preise
    gewinn = n - 1.0 - FEE * n * np.minimum(preise, 1.0 - preise)
    return np.where(treffer, gewinn, -1.0)


def mc_pwert(preise, treffer, sims, rng):
    """P(ROI unter H0 >= beobachtet). H0: Bucket i trifft mit p_i."""
    beob = roi_vec(preise, treffer).mean()
    zufall = rng.random((sims, preise.size)) < preise      # Bernoulli(p_i)
    n = 1.0 / preise
    gewinn = n - 1.0 - FEE * n * np.minimum(preise, 1.0 - preise)
    sim = np.where(zufall, gewinn, -1.0).mean(axis=1)
    # +1 im Zaehler und Nenner: kein p-Wert von exakt 0 (MC-Konvention).
    return beob, (int((sim >= beob).sum()) + 1) / (sims + 1)


def holm(zellen, alpha=0.05):
    """Holm-Bonferroni. Gibt die Liste mit Schwelle und Entscheid zurueck.

    Sortiert aufsteigend nach p. Der i-te Test (1-basiert) wird gegen
    alpha/(m-i+1) geprueft. Beim ERSTEN Test, der reisst, stoppt das Verfahren —
    alle folgenden gelten automatisch als nicht signifikant, unabhaengig von
    ihrem p-Wert. Das ist der Unterschied zu einem einfachen Schwellenvergleich.
    """
    m = len(zellen)
    sortiert = sorted(zellen, key=lambda z: z["p"])
    gestoppt = False
    for i, z in enumerate(sortiert, start=1):
        z["schwelle"] = alpha / (m - i + 1)
        z["rang"] = i
        if gestoppt or z["p"] > z["schwelle"]:
            gestoppt = True
            z["signifikant"] = False
        else:
            z["signifikant"] = True
    return sortiert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", default="max", choices=("max", "min"))
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--von", default=None, help="Zieltag ab (Forward-Auswertung)")
    ap.add_argument("--bis", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(20260806)
    conn = pymssql.connect(**DB_CONFIG)
    best = lade(conn, args.var, args.von, args.bis)
    tage = {t for _, t, _ in best}
    if not tage:
        print(f"Keine gesettelten Lead-1-Snapshots im Fenster "
              f"{args.von or 'Anfang'} bis {args.bis or 'Ende'} (var={args.var}). "
              "Vor dem 07.08.2026 ist das normal.")
        return 0
    print(f"Lead-1-Snapshots: {len(best)} Buckets, {len(tage)} Zieltage "
          f"({min(tage)} bis {max(tage)}), var={args.var}")
    print(f"Monte-Carlo unter H0 'Markt ist fair': {args.sims:,} Simulationen "
          f"je Zelle, einseitig.\n")

    # Beide Anker in EINE Testfamilie. Das ist die ehrliche Zaehlung: es wurde
    # in beiden Sichten gesucht, also muessen beide in die Korrektur.
    zellen = []
    for anker in ("fav", "mkt"):
        grp = defaultdict(list)
        for (city, td, k), (yes, off, mfav, ist, _) in best.items():
            if anker == "fav":
                o = off
            else:
                if mfav is None:
                    continue
                o = k - int(mfav)
            if abs(o) > 2:
                continue
            grp[(city, o)].append((yes, ist == k))
        for (city, o), rows in grp.items():
            if len(rows) < args.min_n:
                continue
            preise = np.array([r[0] for r in rows])
            treffer = np.array([r[1] for r in rows])
            beob, pw = mc_pwert(preise, treffer, args.sims, rng)
            zellen.append({"anker": anker, "city": city, "off": o,
                           "n": len(rows), "roi": beob, "p": pw,
                           "q": treffer.mean(), "preis": preise.mean()})

    m = len(zellen)
    print(f"Testfamilie: {m} Zellen (beide Anker zusammen).")
    print(f"  Bonferroni  : jede Zelle gegen {args.alpha/m:.2e}")
    print(f"  Holm        : Rang 1 gegen {args.alpha/m:.2e}, "
          f"Rang 2 gegen {args.alpha/(m-1):.2e}, ... , Rang {m} gegen {args.alpha:.3f}\n")

    sortiert = holm(zellen, args.alpha)
    print("DIE ZWOELF KLEINSTEN p-WERTE")
    print(f"  {'Rg':>3s} {'Anker':5s} {'Stadt':14s} {'Off':>4s} {'n':>4s} "
          f"{'trifft':>7s} {'Preis':>6s} {'ROI':>9s} {'p (MC)':>9s} "
          f"{'Holm-Schwelle':>14s}  Entscheid")
    for z in sortiert[:12]:
        ent = "SIGNIFIKANT" if z["signifikant"] else "nicht signifikant"
        print(f"  {z['rang']:3d} {z['anker']:5s} {z['city']:14s} {z['off']:+4d} "
              f"{z['n']:4d} {100*z['q']:6.1f} % {z['preis']:6.3f} "
              f"{100*z['roi']:+8.1f} % {z['p']:9.4f} {z['schwelle']:14.2e}  {ent}")

    n_sig = sum(1 for z in sortiert if z["signifikant"])
    n_bonf = sum(1 for z in sortiert if z["p"] < args.alpha / m)
    n_roh = sum(1 for z in sortiert if z["p"] < args.alpha)
    print(f"\nERGEBNIS")
    print(f"  unkorrigiert (p < {args.alpha}) : {n_roh:3d} von {m} Zellen")
    print(f"  Bonferroni                : {n_bonf:3d}")
    print(f"  Holm-Bonferroni           : {n_sig:3d}")
    erwartet = args.alpha * m
    print(f"\n  Zufallserwartung fuer unkorrigierte Treffer: {erwartet:.1f} "
          f"({args.alpha:.0%} von {m}) — beobachtet {n_roh}.")
    if n_roh <= erwartet:
        print("  ==> Die Zahl der nominal signifikanten Zellen liegt NICHT ueber "
              "dem Zufall.\n      Es gibt nichts zu korrigieren, weil es nichts "
              "zu finden gibt.")

    # Gegenprobe: was muesste eine Zelle liefern, um Holm zu ueberleben?
    print(f"\n  Damit Rang 1 ueberlebt, braeuchte es p < {args.alpha/m:.2e}. "
          f"Der kleinste beobachtete p-Wert ist {sortiert[0]['p']:.4f} "
          f"— Faktor {sortiert[0]['p']/(args.alpha/m):.0f} zu gross.")


if __name__ == "__main__":
    sys.exit(main())
