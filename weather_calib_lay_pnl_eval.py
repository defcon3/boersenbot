# -*- coding: utf-8 -*-
"""weather_calib_lay_pnl_eval.py — Bringt die 40d-Sicht dem LAY-BUCH etwas?

VORGESCHICHTE: Der Divergenz-Eval (27./28.07.) hat gezeigt, dass die 40d-Sommer-
Sicht die bessere Favoriten-Prognose liefert — max OOS 45 % vs 37 % Treffer,
min OOS 61 % vs 35 %. Der Ladder-Logger loggt mu_ens aber aus der reinen
700d-Sicht, und der Autobuy filtert hart auf offset_fav = -1. Die gesamte
Kandidatenmenge haengt also an der schlechteren der beiden Sichten.

Umgestellt wurde trotzdem nichts, und zwar mit Absicht: eine bessere
Favoriten-Trefferquote ist NICHT dasselbe wie eine bessere Lay-PnL. Das ist die
Lehre aus dem -1-Waechter (t = 5,46 auf der breiten Menge, schadet dem engen
Live-Buch). Was dem Buch nuetzt, entscheidet die Rendite, nicht die Guete.

WARUM DAS UEBERHAUPT GEGENEINANDER LAUFEN KANN: Gelayt wird nicht der Favorit,
sondern der Bucket EIN GRAD DARUNTER. Eine Sicht, die den Favoriten oefter
trifft, verschiebt damit auch das Lay-Ziel. Trifft die 40d-Sicht besser, weil
sie waermer liegt, dann rutscht ihr -1-Bucket auf das Feld, das die 700d-Sicht
fuer den Favoriten hielt — ein Feld mit hoher Eintrittswahrscheinlichkeit ist
aber das SCHLECHTESTE Lay-Ziel. Besser prognostizieren und schlechter verdienen
ist hier kein Widerspruch, sondern der Normalfall, den es zu messen gilt.

MESSUNG — gleiche Tage, gleiche Preise, nur die Sicht wechselt:
  mu = roh - bias, also unterscheiden sich beide Sichten NUR um eine Konstante
  je Stadt:   mu_40 = mu_700 + (bias_700 - bias_40)
  Damit ist die 40d-Sicht exakt aus dem geloggten mu_ens rekonstruierbar, ohne
  einen einzigen Forecast neu zu ziehen — keine Nachlade-Unschaerfe, kein
  Zeitversatz. Beide Buecher kaufen aus DERSELBEN Ladder-Zeile, es unterscheidet
  sich allein, welches k das Lay-Ziel ist.

RECHNUNG identisch zum Bestand (weather_minus1_shadow / classb_eval):
  Einsatz 5 $, Kontrakte n = 5/NO, Fee 0,07*n*min(NO, 1-NO)
  Lay gewinnt (Bucket trifft NICHT ein): n*1,00 - Fee ;  verliert: 0
  Nur VORTAGS-Snapshots (Zieltag-Snapshots waeren intraday und im Rueckblick zu
  guenstig), nur der spaeteste je Zieltag, nur gesettelte Leitern.

DREI BUECHER, weil nur das dritte ueber die Live-Umstellung entscheidet:
  roh   alle -1-Kandidaten, ohne Preisfilter — die breite Menge
  band  V2-Regel: 0,70 <= NO < 0,90, Rang nach Temperaturabstand, Cap 8/Tag
        (das Spannen-Veto fehlt: es braucht Modell-Rohdaten von damals, die
        rueckwirkend nicht rekonstruierbar sind. Es wirkt auf BEIDE Buecher
        gleich, verschiebt den Vergleich also nicht systematisch.)
  gate  wie band, aber nur Staedte mit D < 0,7 K — die Menge, die das seit
        heute scharfe Einigkeits-Gate ueberhaupt noch durchlaesst

DER ENTSCHEIDENDE SPLIT ist der letzte: wo beide Sichten sich einig sind, ist
das Lay identisch und der Vergleich leer. Der Unterschied entsteht nur auf den
DIVERGENTEN Leitern (k0_700 != k0_40) — und genau die sperrt das neue Gate zum
grossen Teil schon weg. Ein 40d-Vorteil, der ausschliesslich aus gesperrten
Staedten kommt, ist fuer das Live-Buch ohne Wert.

Aufruf:  python weather_calib_lay_pnl_eval.py [--var max|min|beide] [--von TAG]
"""
import argparse
import math
import statistics
import sys
from collections import defaultdict

import pymssql

from weather_calib_divergence_eval import DB_CONFIG, divergenzen
from weather_stations import bucket_grenzen, favorit_k

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

FEE = 0.07
USD = 5.0
BAND_LO, BAND_HI = 0.70, 0.90
CAP = 8
GATE_D = 0.7        # MAX_DIVERGENZ aus reject_reasons(), beide Screens


def lay_pnl(no, gewonnen):
    """Netto-PnL eines 5-$-Lays. gewonnen = der gelayte Bucket trat NICHT ein."""
    n = USD / no
    fee = FEE * n * min(no, 1.0 - no)
    return (n - fee - USD) if gewonnen else -USD


def hole_leitern(var, von):
    """(city, target_date) -> {mu_ens, settle_k, fenster: {k: buy_no}}

    Nur der spaeteste VORTAGS-Snapshot je Zieltag. Ein Zieltag-Snapshot kennt
    schon Teile des Tagesverlaufs und waere im Rueckblick zu guenstig."""
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
    WITH letzte AS (
      SELECT city, target_date, MAX(snapshot_utc) AS snap
      FROM bb_WeatherLadders
      WHERE var=%s AND mu_ens IS NOT NULL AND settle_k IS NOT NULL
        AND CAST(snapshot_utc AS date) < target_date
        AND target_date >= %s
      GROUP BY city, target_date)
    SELECT w.city, w.target_date, w.k, w.buy_no, w.mu_ens, w.settle_k
    FROM letzte l JOIN bb_WeatherLadders w
      ON w.city=l.city AND w.target_date=l.target_date
     AND w.var=%s AND w.snapshot_utc=l.snap
    WHERE w.kind='eq' AND w.status='open' AND w.buy_no > 0
    """, (var, von, var))
    leitern = {}
    for city, target, k, no, mu, sk in cur.fetchall():
        e = leitern.setdefault((city, target), {"mu": float(mu), "settle_k": int(sk),
                                                "fenster": {}})
        e["fenster"][int(k)] = float(no)
    conn.close()
    return leitern


def kandidat(leiter, city, mu):
    """Das -1-Lay unter der Sicht mu: (k, NO, abstand, gewonnen) oder None.

    abstand = wie weit die eigene Prognose ueber der Bucket-OBERKANTE sitzt —
    dieselbe Groesse, nach der der Autobuy rangiert."""
    k = favorit_k(mu, city) - 1
    no = leiter["fenster"].get(k)
    if no is None:
        return None
    return (k, no, mu - bucket_grenzen(k, city)[1], leiter["settle_k"] != k)


def buch(leitern, div, sicht, modus):
    """Ein Buch simulieren. sicht: '700d'|'40d'. modus: 'roh'|'band'|'gate'.

    Rueckgabe: (trades, pnl_je_leiter) — pnl_je_leiter fuer den gepaarten Test,
    Schluessel (city, target_date)."""
    je_tag = defaultdict(list)
    for (city, target), leiter in leitern.items():
        if city not in div:
            continue                      # keine 40d-Kalibrierung -> kein Vergleich
        if modus == "gate" and abs(div[city]) >= GATE_D:
            continue
        mu = leiter["mu"] + (div[city] if sicht == "40d" else 0.0)
        c = kandidat(leiter, city, mu)
        if not c:
            continue
        k, no, abstand, gewonnen = c
        if modus in ("band", "gate") and not (BAND_LO <= no < BAND_HI):
            continue
        je_tag[target].append((abstand, city, k, no, gewonnen))

    trades, pnl = [], {}
    for target, kand in je_tag.items():
        kand.sort(key=lambda x: -x[0])    # Rang nach Temperaturabstand, wie live
        for abstand, city, k, no, gewonnen in (kand[:CAP] if modus != "roh" else kand):
            p = lay_pnl(no, gewonnen)
            trades.append((city, target, k, no, gewonnen, p))
            pnl[(city, target)] = p
    return trades, pnl


def kennzahl(trades):
    if not trades:
        return None
    p = [t[5] for t in trades]
    gew = sum(1 for t in trades if t[4])
    return {"n": len(p), "pnl": sum(p), "rendite": 100 * sum(p) / (USD * len(p)),
            "quote": 100 * gew / len(p), "no": statistics.mean(t[3] for t in trades)}


def zeile(name, k):
    if not k:
        print(f"    {name:<10} —  keine Trades")
        return
    print(f"    {name:<10} n={k['n']:>3}  Lay-Quote {k['quote']:>5.1f} %  "
          f"NO {k['no']:.2f}  PnL {k['pnl']:>+7.2f} $  Rendite {k['rendite']:>+6.2f} %")


def gepaart(pnl_a, pnl_b):
    """t-Test auf den Differenzen der Leitern, die BEIDE Buecher gehandelt haben.

    Gepaart, weil beide Buecher denselben Tag, dieselbe Stadt und dieselbe
    Ladder-Zeile benutzen — nur das Lay-Ziel unterscheidet sich. Alles, was
    beiden gemeinsam ist (Wetterlage, Marktniveau), faellt damit heraus."""
    gemeinsam = set(pnl_a) & set(pnl_b)
    d = [pnl_b[s] - pnl_a[s] for s in gemeinsam]
    d_ungleich = [x for x in d if abs(x) > 1e-9]
    if len(d) < 2:
        return None
    sd = statistics.stdev(d)
    t = (statistics.mean(d) / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0
    return {"n": len(d), "n_div": len(d_ungleich), "mittel": statistics.mean(d),
            "summe": sum(d), "t": t}


def auswerten(var, von):
    leitern = hole_leitern(var, von)
    div = divergenzen(var, signiert=True)
    ohne = {c for (c, _) in leitern} - set(div)
    print(f"\n=== {var.upper()} — {len(leitern)} gesettelte Leitern ab {von} ===")
    if ohne:
        print(f"  ohne 40d-Kalibrierung, nicht vergleichbar: {', '.join(sorted(ohne))}")

    # Wie oft weichen die Sichten ueberhaupt im Lay-Ziel ab?
    n_div = n_ver = 0
    for (city, target), leiter in leitern.items():
        if city not in div:
            continue
        n_ver += 1
        if favorit_k(leiter["mu"], city) != favorit_k(leiter["mu"] + div[city], city):
            n_div += 1
    print(f"  vergleichbar {n_ver} Leitern, davon {n_div} mit ABWEICHENDEM Lay-Ziel "
          f"({100 * n_div / n_ver:.0f} %)" if n_ver else "  nichts vergleichbar")
    if not n_ver:
        return

    for modus, titel in (("roh", "ROH — alle -1-Kandidaten, kein Preisfilter"),
                         ("band", "BAND — V2-Regel 0,70-0,90, Cap 8"),
                         ("gate", f"GATE — V2-Regel NUR Staedte mit D < {GATE_D} K (Live-Menge)")):
        print(f"\n  {titel}")
        t700, p700 = buch(leitern, div, "700d", modus)
        t40, p40 = buch(leitern, div, "40d", modus)
        zeile("700d", kennzahl(t700))
        zeile("40d", kennzahl(t40))
        g = gepaart(p700, p40)
        if g:
            print(f"    gepaart: {g['n']} gemeinsame Leitern, davon {g['n_div']} mit "
                  f"unterschiedlichem Lay | Delta {g['summe']:+.2f} $ "
                  f"({g['mittel']:+.3f} $/Lay), t = {g['t']:+.2f}")
        else:
            print("    gepaart: zu wenige gemeinsame Leitern")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", choices=["max", "min", "beide"], default="beide")
    ap.add_argument("--von", default="2026-07-10", help="frueheste target_date")
    a = ap.parse_args()
    for var in (["max", "min"] if a.var == "beide" else [a.var]):
        auswerten(var, a.von)
    print("\nLesart: entscheidend ist der GATE-Block — nur diese Menge laesst das "
          "Einigkeits-Gate heute noch durch.")


if __name__ == "__main__":
    main()
