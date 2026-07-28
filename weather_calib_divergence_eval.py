# -*- coding: utf-8 -*-
"""weather_calib_divergence_eval.py — Traegt die 700d/40d-Uneinigkeit ueberhaupt?

FRAGE: Die Doppel-Kalibrierungs-Regel verlangt, dass 700d-Ganzjahr und
40d-Sommer sich einig sind. Im Code steht davon bisher nur die halbe Regel —
beide Sichten MUESSEN bestehen (seit 14.07.), aber die DIFFERENZ ihrer mu ist
kein eigenes Kriterium. Bei Hong Kong min liegen die beiden 1,07 K auseinander,
bei Seoul lagen sie am 12.07. 1,1 K auseinander (damals von Hand aussortiert).

Bevor daraus ein Gate wird, muss die Vorhersagekraft gemessen sein. Wenn die
Divergenz nicht mit dem realisierten Prognosefehler zusammenhaengt, ist ein
Gate reine Kosmetik, das nur Kandidaten wegnimmt.

MESSUNG: bb_WeatherLadders fuehrt mu_ens (die reine 700d-Sicht — der Ladder-Logger
laedt bewusst nur die 700d-Basis) und seit dem Settle-Backfill settle_k, den
tatsaechlichen Ist-Bucket. Damit ist je Stadt/var messbar:

    Fehler e = settle_k - favorit_k(mu_ens, city)

und je Stadt/var aus den Kalibrier-CSVs:

    Divergenz D = |bias_700d(ensemble_mean) - bias_40d(ensemble_mean)|

Wenn die Regel etwas taugt, muessen Staedte mit hohem D ein schlechteres |e|
zeigen — und zwar systematisch, nicht als Einzelfall.

VORBEHALT, der ins Ergebnis gehoert: D ist ein STADT-Merkmal, der Vergleich also
ein Between-City-Vergleich. Staedte unterscheiden sich auch sonst (Kuestenlage,
Konvektion, Stationsqualitaet). Ein Zusammenhang belegt hier Tauglichkeit als
FILTER, nicht Kausalitaet.

Aufruf:  python weather_calib_divergence_eval.py [--var max|min|beide]
"""
import argparse
import csv
import glob
import math
import sys

import pymssql

from weather_stations import favorit_k

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_CONFIG = {
    "server": "158.181.48.77",
    "database": "dbdata",
    "user": "326773",
    "password": "Extaler11!",
}

GLOBS = {
    "max": ("preregs/weather_source_calib_*.csv", "preregs/weather_source_calib40d_*.csv"),
    "min": ("preregs/weather_source_calib_min_*.csv", "preregs/weather_source_calib40d_min_*.csv"),
}


def bias_map(pattern, var):
    """city -> bias(ensemble_mean) aus allen passenden CSVs (spaetere gewinnen)."""
    out = {}
    for path in sorted(glob.glob(pattern)):
        if "_lead" in path:
            continue
        ist_min = "_min_" in path
        if ist_min != (var == "min"):
            continue          # _min_-Dateien gehoeren zur min-Familie und nur dorthin
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["model"] == "ensemble_mean":
                    out[row["city"]] = float(row["bias"])
    return out


def divergenzen(var, signiert=False):
    """|D| je Stadt — oder die SIGNIERTE Differenz b700 - b40.

    Das Vorzeichen ist die interessantere Groesse: mu = roh - bias, also ist
    mu_40 - mu_700 = b700 - b40. Ein positives S heisst 'die 40d-Sicht liegt
    waermer als die geloggte 700d-Sicht'. Faellt der realisierte Fehler e dann
    ebenfalls positiv aus (Ist waermer als prognostiziert), hat die 40d-Sicht
    recht behalten — und die Divergenz sagt nicht nur, DASS es schiefgeht,
    sondern WOHIN."""
    g700, g40 = GLOBS[var]
    b700, b40 = bias_map(g700, var), bias_map(g40, var)
    d = {c: b700[c] - b40[c] for c in b700 if c in b40}
    return d if signiert else {c: abs(v) for c, v in d.items()}


def hole_tage(var):
    """Ein Datensatz je (city, target_date) mit mu_ens UND settle_k.

    mu_ens/settle_k stehen redundant in jeder Fensterzeile der Leiter — MAX()
    greift also einen beliebigen, aber identischen Wert. Nur der SPAETESTE
    Snapshot je Zieltag zaehlt: frueher am Tag ist der Forecast aelter."""
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
    WITH letzte AS (
      SELECT city, target_date, MAX(snapshot_utc) AS snap
      FROM bb_WeatherLadders
      WHERE var=%s AND mu_ens IS NOT NULL AND settle_k IS NOT NULL
      GROUP BY city, target_date)
    SELECT l.city, l.target_date, MAX(w.mu_ens), MAX(w.settle_k)
    FROM letzte l JOIN bb_WeatherLadders w
      ON w.city=l.city AND w.target_date=l.target_date
     AND w.var=%s AND w.snapshot_utc=l.snap
    GROUP BY l.city, l.target_date
    """, (var, var))
    rows = cur.fetchall()
    conn.close()
    return rows


def auswerten(var):
    div = divergenzen(var)
    rows = hole_tage(var)
    if not rows:
        print(f"  {var}: keine gesettelten Zeilen mit mu_ens -> nichts zu messen")
        return
    je_stadt = {}
    for city, _tag, mu, settle_k in rows:
        if city not in div:
            continue          # keine der beiden Kalibrierungen -> kein D
        e = int(settle_k) - favorit_k(float(mu), city)
        je_stadt.setdefault(city, []).append(e)

    print(f"\n{'='*78}\n{var.upper()} — Divergenz D vs. realisierter Fehler\n{'='*78}")
    print(f"{'Stadt':14} {'D (K)':>7} {'n':>4} {'Treffer':>8} {'|e| Mittel':>11} {'Bias e':>7}")
    zeilen = []
    for city, es in sorted(je_stadt.items(), key=lambda kv: -div[kv[0]]):
        if len(es) < 5:
            continue          # zu duenn fuer eine Quote
        treffer = sum(1 for e in es if e == 0) / len(es)
        abs_m = sum(abs(e) for e in es) / len(es)
        bias = sum(es) / len(es)
        zeilen.append((div[city], len(es), treffer, abs_m, bias, city))
        print(f"{city:14} {div[city]:7.2f} {len(es):4d} {treffer:7.0%} {abs_m:11.2f} {bias:+7.2f}")

    if len(zeilen) < 6:
        print("\n  Zu wenige Staedte mit n>=5 fuer eine Korrelation.")
        return

    # Pearson ueber Staedte: D gegen |e|. Gewichtung bewusst NICHT nach n —
    # gefragt ist der Zusammenhang zwischen Stadt-Merkmalen, nicht zwischen Tagen.
    xs = [z[0] for z in zeilen]
    ys = [z[3] for z in zeilen]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    r = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else float("inf")
    print(f"\n  Korrelation D vs |e| ueber {n} Staedte: r = {r:+.3f}, t = {t:+.2f}")

    # Zweite, schaerfere Frage: sagt das VORZEICHEN der Divergenz die Richtung
    # des Fehlers voraus? Wenn ja, ist ein Gate die falsche Antwort — dann muss
    # die 40d-Sicht in mu einfliessen, statt den Kandidaten nur wegzuwerfen.
    sig = divergenzen(var, signiert=True)
    xs2 = [sig[z[5]] for z in zeilen]
    ys2 = [z[4] for z in zeilen]          # Bias von e, nicht |e|
    mx2, my2 = sum(xs2) / n, sum(ys2) / n
    sxy2 = sum((x - mx2) * (y - my2) for x, y in zip(xs2, ys2))
    sxx2 = sum((x - mx2) ** 2 for x in xs2)
    syy2 = sum((y - my2) ** 2 for y in ys2)
    r2 = sxy2 / math.sqrt(sxx2 * syy2) if sxx2 > 0 and syy2 > 0 else 0.0
    t2 = r2 * math.sqrt((n - 2) / (1 - r2 * r2)) if abs(r2) < 1 else float("inf")
    print(f"  Korrelation SIGNIERT (b700-b40) vs Fehler-Bias:  r = {r2:+.3f}, t = {t2:+.2f}")

    # Split an der Doktrin-Schwelle, die bei Hong Kong min gegriffen hat.
    for schwelle in (0.3, 0.5, 0.7, 1.0):
        lo = [z for z in zeilen if z[0] < schwelle]
        hi = [z for z in zeilen if z[0] >= schwelle]
        if not lo or not hi:
            continue
        f_lo = sum(z[2] * z[1] for z in lo) / sum(z[1] for z in lo)
        f_hi = sum(z[2] * z[1] for z in hi) / sum(z[1] for z in hi)
        a_lo = sum(z[3] * z[1] for z in lo) / sum(z[1] for z in lo)
        a_hi = sum(z[3] * z[1] for z in hi) / sum(z[1] for z in hi)
        print(f"  D < {schwelle:.1f}: {len(lo):2d} Staedte, Treffer {f_lo:.0%}, |e| {a_lo:.2f}   |   "
              f"D >= {schwelle:.1f}: {len(hi):2d} Staedte, Treffer {f_hi:.0%}, |e| {a_hi:.2f}")


def gegenrechnung(var, ab_datum=None):
    """Waere die 40d-Sicht der bessere Favorit gewesen als die geloggte 700d-Sicht?

    mu = ens_raw - bias mit dem ENS-Bias (weather_ladder_logger.forecast_mu), und
    ens_raw ist fuer beide Sichten dasselbe. Also ist die 40d-Sicht exakt
    rekonstruierbar: mu_40 = mu_700 + (b700 - b40). Kein Nachrechnen der
    Forecasts noetig, keine Naeherung.

    ab_datum: nur Zieltage NACH dem Stand der 40d-CSVs bewerten. Die aktuellen
    40d-Dateien sind vom 17.07. und ihr Fenster enthaelt die Zieltage bis dahin —
    wer die mitbewertet, misst teilweise sich selbst."""
    sig = divergenzen(var, signiert=True)
    rows = hole_tage(var)
    tref = {"700d": 0, "40d": 0}
    absf = {"700d": 0.0, "40d": 0.0}
    n = 0
    for city, tag, mu, settle_k in rows:
        if city not in sig:
            continue
        if ab_datum and str(tag) <= ab_datum:
            continue
        mu700 = float(mu)
        mu40 = mu700 + sig[city]
        e700 = int(settle_k) - favorit_k(mu700, city)
        e40 = int(settle_k) - favorit_k(mu40, city)
        tref["700d"] += (e700 == 0)
        tref["40d"] += (e40 == 0)
        absf["700d"] += abs(e700)
        absf["40d"] += abs(e40)
        n += 1
    if not n:
        print(f"  {var}: keine Tage im Fenster")
        return
    fenster = f"nur Zieltage nach {ab_datum}" if ab_datum else "alle gesettelten Zieltage"
    print(f"\n  Favorit-Guete {var} ({fenster}, n={n} Stadt-Tage):")
    for sicht in ("700d", "40d"):
        print(f"    {sicht:5} geloggt{'  <- heute in mu_ens' if sicht == '700d' else '        '}"
              f"  Treffer {tref[sicht]/n:5.0%}   |e| {absf[sicht]/n:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", choices=["max", "min", "beide"], default="beide")
    ap.add_argument("--oos-ab", default="2026-07-17", metavar="YYYY-MM-DD",
                    help="Gegenrechnung zusaetzlich nur auf Zieltage NACH diesem Datum "
                         "(default 17.07. = Stand der aktuellen 40d-CSVs; alles davor "
                         "steckt in deren Kalibrierfenster)")
    args = ap.parse_args()
    for var in (["max", "min"] if args.var == "beide" else [args.var]):
        auswerten(var)
        gegenrechnung(var)
        gegenrechnung(var, ab_datum=args.oos_ab)


if __name__ == "__main__":
    main()
