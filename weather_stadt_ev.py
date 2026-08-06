#!/usr/bin/env python3
"""Wo trifft ein Bucket oefter, als sein Preis verlangt? — je Stadt, je Offset.

FRAGE DES BETREIBERS (06.08.2026), sinngemaess: "Ich bin angetreten, um eine
konsistente Prognosequelle zu finden. Wenn die Vorhersage fuer Tokio immer
einen Bucket danebenliegt, ist das der Gelddrucker. Ob 10 Cent oder 1 Dollar
Gewinn ist egal, Hauptsache mehr als vorher. Warum wurde das verworfen?"

Antwort: es wurde nie gerechnet. Gemessen wurde bisher, wie oft ein Bucket
TRIFFT (weather_bucket_abstand_eval.py, 03.08.) — nie, was er dabei KOSTET.
Das ist der Unterschied zwischen einer Modellschwaeche und einem Edge. Tel Aviv
trifft die -1-Klasse in 14 von 18 Faellen; ob das Geld wert ist, haengt allein
daran, ob der Markt dort unter 78 % bepreist.

WAS HIER GERECHNET WIRD, je Stadt und je Offset zum jeweiligen Anker:
    Trefferquote q   — wie oft settelt der Bucket
    Mittelpreis  p   — was YES auf ihn im Schnitt kostet (buy_yes, Lead-1)
    EV               — q/p - 1 - Fee, also der ROI eines YES-Kaufs
Ein Offset ist nur dann Geld, wenn q > p. Alles andere ist Modellkritik.

ZWEI ANKER, bewusst getrennt:
  * offset_fav  — relativ zum EIGENEN Favoriten (mu_ens). Findet Staedte, in
    denen UNSERE Prognose konsistent danebenliegt.
  * offset_mkt  — relativ zum MARKT-Favoriten (market_fav_k). Kontrolle: liegt
    der Markt anderswo, ist die Verschiebung unser Problem und meist schon
    eingepreist.

FALLEN, die hier vermieden werden:
  * Favorit nur ueber offset_fav aus der DB (dort steckt favorit_k(mu, city)
    inkl. BUCKET_FLOOR fuer Hong Kong) — nie selbst half_up runden.
  * Lead 1, also der letzte Snapshot VOR dem Zieltag. Lead 0 liegt fuer Asien
    nach dem Tagesmaximum und rechnet den Fehler klein.
  * wu_settle_k hat Vorrang vor settle_k (Settlement-Quelle des Marktes).
  * n je Stadt ist klein (11-26 Tage). Bonferroni ueber 30 Staedte ist Pflicht,
    sonst baut man 29 Zufallsmuster — genau die Lehre vom 03.08.
"""
import argparse
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import pymssql

DB_CONFIG = {"server": "158.181.48.77", "database": "dbdata",
             "user": "326773", "password": "Extaler11!"}
FEE = 0.07


def roi_yes(p, getroffen):
    """ROI EINES YES-Kaufs zu genau diesem Preis. Einsatz 1, Fee wie im Projekt.

    ⚠️ Bewusst positionsweise. Die erste Fassung rechnete EV = q/p̄ - 1 aus der
    Trefferquote und dem MITTELPREIS — das verdeckt, ob die Treffer bei teuren
    oder bei billigen Preisen lagen. Beijing +0 kam so auf +133 % EV, waehrend
    die positionsweise Rechnung -57 % ergibt (t -1,98). Genau die Lehre vom
    02.08.2026: Break-even gehoert positionsweise aus dem echten Preis.
    """
    if p <= 0 or p >= 1:
        return None
    n = 1.0 / p
    return n - 1 - FEE * n * min(p, 1 - p) if getroffen else -1.0


def kennzahlen(rows):
    """(n, Trefferquote, Mittelpreis, mittlerer ROI, t) fuer eine Zelle."""
    import statistics as stat
    r = [roi_yes(p, t) for p, t in rows]
    r = [x for x in r if x is not None]
    n = len(r)
    if n < 2:
        return None
    q = sum(1 for _, t in rows if t) / len(rows)
    p = sum(p for p, _ in rows) / len(rows)
    sd = stat.stdev(r)
    t = stat.mean(r) / (sd / n ** 0.5) if sd else 0.0
    return n, q, p, stat.mean(r), t


def binom_p(k, n, p):
    """Einseitiger Binomialtest: P(X >= k) bei n Versuchen und Rate p."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def lade(conn, var, von=None, bis=None):
    """Lead-1-Snapshot je (Stadt, Zieltag, k): der letzte VOR dem Zieltag.

    --von/--bis grenzen das Fenster ab. Fuer die Auswertung der Pre-Reg
    preregs/weather_stadt_zellen_2026_08_06.md ist es 2026-08-07 bis
    2026-09-03; die Grenze steht dort fest und wird nicht verschoben.
    """
    cur = conn.cursor()
    q = ("SELECT city, target_date, k, buy_yes, offset_fav, market_fav_k, "
         "       settle_k, wu_settle_k, snapshot_utc "
         "FROM bb_WeatherLadders "
         "WHERE var=%s AND kind='eq' AND (settle_k IS NOT NULL OR wu_settle_k IS NOT NULL) "
         "  AND offset_fav IS NOT NULL AND buy_yes > 0")
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
        # Lead 1: Snapshot am Vortag des Zieltags.
        if snap.date() != td - timedelta(days=1):
            continue
        key = (city, str(td), int(k))
        if key not in best or snap > best[key][-1]:
            ist = wu if wu is not None else sk
            best[key] = (float(yes), int(off), mfav, int(ist), snap)
    return best


def auswerten(best, anker, min_n, label):
    """Je (Stadt, Offset) Trefferquote gegen Mittelpreis."""
    grp = defaultdict(list)
    for (city, td, k), (yes, off, mfav, ist, _) in best.items():
        if anker == "fav":
            o = off
        else:
            if mfav is None:
                continue
            o = k - int(mfav)
        grp[(city, o)].append((yes, ist == k))

    print(f"\n{'='*100}")
    print(f"{label}")
    print(f"{'='*100}")
    print(f"  {'Stadt':14s} {'Off':>4s} {'n':>4s} {'trifft':>8s} {'Preis':>7s} "
          f"{'ROI':>9s} {'t':>7s} {'p-Wert':>10s}")
    treffer = []
    for (city, o), rows in sorted(grp.items()):
        if len(rows) < min_n or abs(o) > 2:
            continue
        kz = kennzahlen(rows)
        if not kz:
            continue
        n, q, p, roi, t = kz
        pw = binom_p(sum(1 for _, tr in rows if tr), n, p)
        treffer.append((roi, city, o, n, q, p, pw, t))
    for roi, city, o, n, q, p, pw, t in sorted(treffer, reverse=True):
        mark = ""
        if roi > 0 and pw < 0.05 / 30:
            mark = "  <== BONFERRONI-FEST"
        elif roi > 0 and t >= 1.5:
            mark = "  <- t >= 1,5"
        print(f"  {city:14s} {o:+4d} {n:4d} {100*q:7.1f} % {p:7.3f} "
              f"{100*roi:+8.1f} % {t:+7.2f} {pw:10.2e}{mark}")
    return treffer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", default="max", choices=("max", "min"))
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--von", default=None, help="Zieltag ab (Fenster der Pre-Reg)")
    ap.add_argument("--bis", default=None)
    args = ap.parse_args()

    conn = pymssql.connect(**DB_CONFIG)
    best = lade(conn, args.var, args.von, args.bis)
    staedte = {c for c, _, _ in best}
    tage = {t for _, t, _ in best}
    if not tage:
        print(f"Keine gesettelten Lead-1-Snapshots im Fenster "
              f"{args.von or 'Anfang'} bis {args.bis or 'Ende'} (var={args.var}).")
        print("Das ist vor dem 07.08.2026 normal — das Fenster der Pre-Reg "
              "preregs/weather_stadt_zellen_2026_08_06.md beginnt erst dann.")
        return 0
    print(f"Lead-1-Snapshots: {len(best)} Buckets, {len(staedte)} Staedte, "
          f"{len(tage)} Zieltage ({min(tage)} bis {max(tage)}), var={args.var}")
    print(f"Bonferroni-Schwelle bei 30 Staedten: p < {0.05/30:.2e}")

    a = auswerten(best, "fav", args.min_n,
                  "ANKER: EIGENER FAVORIT  (Offset 0 = unser mu-Favorit)")
    b = auswerten(best, "mkt", args.min_n,
                  "ANKER: MARKT-FAVORIT  (Offset 0 = teuerster YES-Bucket)")

    # --- Der Test mit grossem n: alle Staedte zusammen, je Offset ---
    # Einzelne Staedte haben n = 8..19 und ueberleben Bonferroni nie. Der
    # Longshot-Bias, falls es einer ist, muesste sich aber ueber ALLE Bretter
    # zeigen — dort ist n = 390 statt 19. Anker ist der MARKT-Favorit, weil der
    # eigene mu die Stadt-Verschiebung mitschleppt.
    for anker, label in (("mkt", "MARKT-FAVORIT"), ("fav", "EIGENER FAVORIT")):
        gesamt = defaultdict(list)
        for (city, td, k), (yes, off, mfav, ist, _) in best.items():
            if anker == "fav":
                o = off
            else:
                if mfav is None:
                    continue
                o = k - int(mfav)
            gesamt[o].append((yes, ist == k))
        print(f"\n{'='*100}")
        print(f"ALLE STAEDTE ZUSAMMEN — Anker {label}")
        print(f"{'='*100}")
        print(f"  {'Off':>4s} {'n':>5s} {'trifft':>8s} {'Preis':>7s} "
              f"{'ROI':>9s} {'t':>7s} {'p-Wert':>11s}")
        for o in sorted(gesamt):
            if abs(o) > 3:
                continue
            kz = kennzahlen(gesamt[o])
            if not kz:
                continue
            n, q, p, roi, t = kz
            pw = binom_p(sum(1 for _, tr in gesamt[o] if tr), n, p)
            mark = "  <== TRAEGT" if roi > 0 and t >= 2.0 else ""
            print(f"  {o:+4d} {n:5d} {100*q:7.1f} % {p:7.3f} "
                  f"{100*roi:+8.1f} % {t:+7.2f} {pw:11.2e}{mark}")

    print(f"\n{'='*100}")
    print("FAZIT")
    print(f"{'='*100}")
    for label, t in (("eigener Favorit", a), ("Markt-Favorit", b)):
        pos = [x for x in t if x[0] > 0]
        stark = [x for x in pos if x[7] >= 1.5]
        bon = [x for x in pos if x[6] < 0.05 / 30]
        print(f"  {label:16s}: {len(t):3d} Zellen, {len(pos):3d} mit ROI > 0, "
              f"{len(stark)} davon mit t >= 1,5, {len(bon)} Bonferroni-fest")
        for roi, city, o, n, q, p, pw, tt in sorted(stark, reverse=True)[:8]:
            print(f"      {city} {o:+d}: n={n}, trifft {100*q:.0f} % zu Preis "
                  f"{p:.2f}  ->  ROI {100*roi:+.0f} %, t {tt:+.2f}")


if __name__ == "__main__":
    sys.exit(main())
