#!/usr/bin/env python3
"""
crypto_flip_persistence.py — Haelt der Favorit aus der Fenster-Halbzeit bis zum Schluss?

THESE (Nutzer, 2026-07-26, aus Beobachtung der 5-Min-Flip-Maerkte): "Wenn die
Richtung nach 2,5 Minuten bekannt ist (Favorit), dann endet es auch so." Beobachtet
an BTC am Sonntag. Anschluss an [[crypto-weekday-vol]]: sonntags/sonnabends ist die
5-Min-Spanne 18-31 % kleiner — weniger Spanne = weniger Gelegenheit zur Umkehr,
die Persistenz SOLLTE also am Wochenende hoeher sein.

GEMESSEN wird die reine Spot-Persistenz, NICHT die Handelbarkeit:
    Fenster = [t, t+5min) auf dem 5-Min-Raster (so liegen die Jupiter-Fenster)
    Strike  = Open der ersten Minute (= Null-Linie des Marktes)
    Halbzeit= Stand nach 2 bzw. 3 Minuten (klammert die 2,5 des Nutzers ein;
              1-Min-Kerzen sind das feinste Raster mit brauchbarer Historie)
    Ende    = Close der fuenften Minute
    Persistenz = P(Endrichtung = Halbzeitrichtung | Halbzeit != Strike)
Settlement-Konvention wie bei Jupiter: "Up" gilt bei >= Strike, ein exakter
Gleichstand am Ende zaehlt also als Up.

WARUM DAS NOCH KEIN EDGE IST: entscheidend ist nicht, ob der Favorit haelt,
sondern was er zur Halbzeit KOSTET. Am 09.07. wurde die verwandte Frueh-Favorit-
These auf den 15-Min-Maerkten falsifiziert — die Dreh-Raten waren zwar niedrig
(42 % bis 6 % je nach Vorsprung), aber die Gewinnrate lag in JEDEM Bucket
praktisch auf dem Ask (netto -7,4 %/Trade, Cluster-t -5,0). Diese Auswertung
liefert die linke Haelfte der Rechnung; die rechte braucht CLOB-Preise zur
Halbzeit (bb_BtcFlip, derzeit zu duenn).

METHODIK: Aufschluesselung nach Vorsprung in bps ist Pflicht — ein Favorit mit
1 bps Vorsprung ist etwas anderes als einer mit 30. Standardfehler werden auf
TAGESEBENE geclustert, weil Fenster desselben Tages im selben Vol-Regime liegen
und nicht als unabhaengige Beobachtungen zaehlen duerfen.

Aufruf:
  python crypto_flip_persistence.py --days 730 --assets BTC
  python crypto_flip_persistence.py --days 365
"""

import argparse
import math
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")
from crypto_weekday_vol import ASSETS, klines          # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

WD = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Sonnabend", "Sonntag"]
WIN_MIN = 5
BPS = [(0, 2, "0-2 bps"), (2, 5, "2-5"), (5, 10, "5-10"),
       (10, 20, "10-20"), (20, 1e9, ">20")]


def fenster_bauen(rows, halbzeit_min):
    """rows = 1-Min-Kerzen -> Liste der 5-Min-Fenster mit Halbzeit- und Endstand.
    Nur Fenster mit allen 5 Minuten luekenlos auf dem 5-Min-Raster."""
    per_min = {int(ts / 60000): (o, h, l, c) for ts, o, h, l, c in rows}
    out = []
    for m0 in sorted(per_min):
        if m0 % WIN_MIN != 0:
            continue
        teile = [per_min.get(m0 + i) for i in range(WIN_MIN)]
        if any(t is None for t in teile):
            continue
        strike = teile[0][0]
        if strike <= 0:
            continue
        halb = teile[halbzeit_min - 1][3]
        ende = teile[WIN_MIN - 1][3]
        d_halb = (halb - strike) / strike * 1e4
        if d_halb == 0:
            continue                       # kein Favorit zur Halbzeit
        up_halb = d_halb > 0
        up_ende = ende >= strike           # Jupiter-Konvention: >= zaehlt als Up
        dt = datetime.fromtimestamp(m0 * 60, timezone.utc)
        out.append({"tag": dt.date(), "wd": dt.weekday(), "stunde": dt.hour,
                    "vorsprung": abs(d_halb), "haelt": up_halb == up_ende})
    return out


def cluster_t(fenster, p0=0.5):
    """t-Wert fuer 'Persistenz > p0', Standardfehler ueber TAGE geclustert."""
    per_tag = defaultdict(list)
    for f in fenster:
        per_tag[f["tag"]].append(1.0 if f["haelt"] else 0.0)
    raten = [st.mean(v) for v in per_tag.values() if len(v) >= 5]
    if len(raten) < 3:
        return float("nan"), len(raten)
    m, sd = st.mean(raten), st.stdev(raten)
    if sd == 0:
        return float("nan"), len(raten)
    return (m - p0) / (sd / math.sqrt(len(raten))), len(raten)


def quote(fenster):
    if not fenster:
        return 0.0, 0
    return sum(1 for f in fenster if f["haelt"]) / len(fenster), len(fenster)


def tabelle(titel, fenster):
    p, n = quote(fenster)
    t, ntage = cluster_t(fenster)
    print(f"\n   {titel}: {p * 100:.2f} % halten  (n = {n:,} Fenster, "
          f"{ntage} Tage, Cluster-t vs. 50 % = {t:.2f})")
    print(f"      {'Vorsprung':<12} {'n':>8} {'haelt':>8} {'Cluster-t':>10}")
    for lo, hi, lab in BPS:
        grp = [f for f in fenster if lo <= f["vorsprung"] < hi]
        if len(grp) < 50:
            continue
        pg, ng = quote(grp)
        tg, _ = cluster_t(grp)
        print(f"      {lab:<12} {ng:8,} {pg * 100:7.2f} % {tg:10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--assets", default="BTC,ETH,SOL,XRP,BNB,DOGE")
    ap.add_argument("--halbzeit", type=int, default=3,
                    help="Halbzeit-Minute (2 und 3 klammern die 2,5 des Nutzers ein)")
    args = ap.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    print("Persistenz des Halbzeit-Favoriten in 5-Minuten-Flip-Fenstern")
    print(f"Zeitraum {start.date()} bis {end.date()} | Halbzeit = Minute "
          f"{args.halbzeit} von {WIN_MIN} | Tagesgrenzen UTC")
    print("Lesehilfe: 50 % = Muenzwurf. Die These verlangt deutlich mehr, "
          "und zwar am Wochenende mehr als werktags.")

    gesamt = defaultdict(list)
    for a in [x.strip().upper() for x in args.assets.split(",")]:
        sym = ASSETS.get(a)
        if not sym:
            print(f"\n{a}: unbekannt"); continue
        rows = klines(sym, int(start.timestamp() * 1000),
                      int(end.timestamp() * 1000), "1m")
        if not rows:
            print(f"\n{a} ({sym}): keine Daten"); continue
        fenster = fenster_bauen(rows, args.halbzeit)
        print(f"\n{'=' * 74}\n{a} ({sym}) — {len(rows):,} Minuten, "
              f"{len(fenster):,} verwertbare Fenster")
        tabelle("ALLE Tage", fenster)
        for wd, lab in ((6, "Sonntag"), (5, "Sonnabend")):
            tabelle(lab, [f for f in fenster if f["wd"] == wd])
        tabelle("Montag-Freitag", [f for f in fenster if f["wd"] < 5])
        print(f"\n   Persistenz je Wochentag:")
        for wd in range(7):
            grp = [f for f in fenster if f["wd"] == wd]
            if not grp:
                continue
            p, n = quote(grp)
            t, _ = cluster_t(grp)
            print(f"      {WD[wd]:<11} {n:7,} Fenster  {p * 100:6.2f} %  t={t:6.2f}")
        for f in fenster:
            gesamt[a].append(f)

    if len(gesamt) > 1:
        alle = [f for v in gesamt.values() for f in v]
        print(f"\n{'=' * 74}\nGEPOOLT ueber {len(gesamt)} Assets ({len(alle):,} Fenster)")
        print("ACHTUNG: die Coins laufen synchron (Krypto-Beta) — das effektive N "
              "ist naeher an dem eines einzelnen Assets als an der Summe.")
        tabelle("ALLE Tage", alle)
        for wd, lab in ((6, "Sonntag"), (5, "Sonnabend")):
            tabelle(lab, [f for f in alle if f["wd"] == wd])
        tabelle("Montag-Freitag", [f for f in alle if f["wd"] < 5])
    return 0


if __name__ == "__main__":
    sys.exit(main())
