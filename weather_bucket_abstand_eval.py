# -*- coding: utf-8 -*-
"""weather_bucket_abstand_eval.py — Wie weit daneben landet der Ist-Bucket?

Auftrag des Betreibers (03.08.2026): nicht nur die Trefferquote, sondern die
VOLLE, VORZEICHENBEHAFTETE Verteilung des Abstands — 0, +-1, +-2, +-3, +-4 —
je Stadt, max und min getrennt, und zwar gegen BEIDE Anker:

    unser Anker:  settle_k - favorit_k(mu_ens, city)
    Markt-Anker:  settle_k - market_fav_k

Das Vorzeichen bleibt erhalten: zu warm ist nicht zu kalt. Ein Betragsmittel
verdeckt genau das, was handlungsrelevant ist — landet der Ist in einer Stadt
systematisch bei -2 statt gestreut, ist das eine Auswahlregel.

RUNDUNGSFALLE (am 03.08. selbst getreten): den Favoriten NICHT mit int(mu+0.5)
bilden. Hong Kong ist BUCKET_FLOOR ("28C" = [28,0..28,9] statt [27,5..28,5)) und
erscheint mit half_up faelschlich mit null Treffern. Deshalb kommt der Favorit
ausschliesslich aus weather_stations.favorit_k().

LEAD: gewertet wird der Snapshot vom VORTAG (Lead 1) — das ist der Stand, auf
dem der Autobuy entscheidet. Lead 0 waere fuer die asiatischen Staedte bereits
NACH dem Tagesmaximum und wuerde den Fehler kuenstlich kleinrechnen.

Aufruf:
  python weather_bucket_abstand_eval.py
  python weather_bucket_abstand_eval.py --lead 2 --von 2026-07-01
"""
import argparse
import statistics as st
import sys
from collections import Counter, defaultdict

import pymssql

from weather_stations import favorit_k

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB_CONFIG = {"server": "158.181.48.77", "database": "dbdata",
             "user": "326773", "password": "Extaler11!"}


def lade(lead, von):
    """Ein Stadttag = eine Zeile. mu_ens/market_fav_k sind je Snapshot konstant,
    MAX() ist hier nur der Aggregat-Zwang von SQL, kein Auswahlkriterium."""
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT target_date, var, city, AVG(mu_ens), MAX(market_fav_k), MAX(settle_k)
        FROM bb_WeatherLadders
        WHERE settle_k IS NOT NULL
          AND DATEDIFF(day, CAST(snapshot_utc AS DATE), target_date) = %s
          AND target_date >= %s
        GROUP BY target_date, var, city
        ORDER BY var, city, target_date""", (lead, von))
    rows = cur.fetchall()
    conn.close()
    return rows


def verteilung(werte, lo=-4, hi=4):
    """Counter -> Zeile '. 1 3 9 4 1 . .' ueber den festen Bereich lo..hi,
    Ausreisser jenseits davon in Klammern angehaengt."""
    c = Counter(werte)
    zellen = []
    for k in range(lo, hi + 1):
        n = c.get(k, 0)
        zellen.append(f"{n:>3}" if n else "  .")
    weiter = sum(n for k, n in c.items() if k < lo or k > hi)
    return "".join(zellen) + (f"  (+{weiter} weiter draussen)" if weiter else "")


def kennzahlen(werte):
    n = len(werte)
    treffer = sum(1 for d in werte if d == 0) / n
    mittel = st.mean(werte)
    mae = st.mean([abs(d) for d in werte])
    return n, treffer, mittel, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=1,
                    help="Snapshot-Vorlauf in Tagen (1 = Vortag, der Entscheidungsstand)")
    ap.add_argument("--von", default="2026-07-01")
    args = ap.parse_args()

    rows = lade(args.lead, args.von)
    print("=" * 96)
    print(f"BUCKET-ABSTANDSVERTEILUNG — settle_k minus Favorit, vorzeichenbehaftet")
    print(f"Lead {args.lead} (Snapshot am Vortag) | ab {args.von} | "
          f"negativ = Ist KAELTER als der Favorit")
    print("=" * 96)

    for var in ("max", "min"):
        daten = defaultdict(lambda: {"eigen": [], "markt": [], "tage": []})
        for d, v, city, mu, mfav, settle in rows:
            if v != var:
                continue
            g = daten[city]
            if mu is not None:
                g["eigen"].append(settle - favorit_k(mu, city))
                g["tage"].append((d, settle - favorit_k(mu, city),
                                  settle - mfav if mfav is not None else None))
            if mfav is not None:
                g["markt"].append(settle - mfav)

        alle_e = [x for g in daten.values() for x in g["eigen"]]
        alle_m = [x for g in daten.values() for x in g["markt"]]
        if not alle_e and not alle_m:
            print(f"\n### {var}: keine Daten")
            continue

        print(f"\n\n### Bretter '{var}' — {len(daten)} Staedte")
        print("-" * 96)
        kopf = "  ".join(f"{k:>3}" for k in range(-4, 5))
        print(f"{'':>34}{'-4  -3  -2  -1   0  +1  +2  +3  +4':>38}")

        for name, werte in (("unser Favorit favorit_k(mu_ens)", alle_e),
                            ("Markt-Favorit market_fav_k     ", alle_m)):
            if not werte:
                continue
            n, tq, m, mae = kennzahlen(werte)
            print(f"{name}  n={n:>4}{verteilung(werte)}")
            print(f"{'':>34}   Treffer {tq:5.1%} | Mittel {m:+.2f} | MAE {mae:.2f} Bucket")

        # --- je Stadt -------------------------------------------------------
        print(f"\n{'Stadt':<15}{'n':>4}  {'-4  -3  -2  -1   0  +1  +2  +3  +4':^30}"
              f"{'Treffer':>9}{'Mittel':>8}{'MAE':>6}   Markt-TQ")
        print("-" * 96)
        for city in sorted(daten, key=lambda c: -len(daten[c]["eigen"])):
            g = daten[city]
            if not g["eigen"]:
                continue
            n, tq, m, mae = kennzahlen(g["eigen"])
            mtq = (sum(1 for d in g["markt"] if d == 0) / len(g["markt"])
                   if g["markt"] else float("nan"))
            zeichen = "  <<<" if abs(m) >= 0.8 and n >= 8 else ""
            print(f"{city:<15}{n:>4} {verteilung(g['eigen'])}"
                  f"{tq:>8.0%}{m:>+8.2f}{mae:>6.2f}   {mtq:>6.0%}{zeichen}")

        # --- Verlustschwanz: jeder Einzelfall ab 3 Buckets -------------------
        rand = [(c, d, off, moff) for c, g in daten.items()
                for d, off, moff in g["tage"] if abs(off) >= 3]
        print(f"\nVerlustschwanz — alle Stadttage mit |Abstand| >= 3 ({len(rand)} von {len(alle_e)}"
              f" = {len(rand)/len(alle_e):.1%}):" if alle_e else "")
        for c, d, off, moff in sorted(rand, key=lambda x: -abs(x[2])):
            mm = f"Markt {moff:+d}" if moff is not None else "Markt --"
            print(f"   {d}  {c:<15} unser {off:+d}   {mm}")

    print("\n" + "=" * 96)
    print("Lesart: Mittel < 0 = unser Favorit ist zu WARM (der Ist landet darunter).")
    print("'<<<' markiert Staedte mit |Mittel| >= 0,8 Bucket bei n >= 8 — dort ist die")
    print("Verschiebung systematisch, nicht Streuung.")
    print("=" * 96)


if __name__ == "__main__":
    main()
