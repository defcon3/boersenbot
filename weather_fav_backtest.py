# -*- coding: utf-8 -*-
"""weather_fav_backtest.py — Was haette "immer den eigenen Favoriten backen"
gebracht? Forward-Messung auf den geloggten Preisleitern.

Frage (Nutzer 28.07.2026): Statt zu layen einfach jeden Tag in jeder Stadt den
Favoriten des eigenen Modells (offset_fav == 0) mit festem Einsatz YES kaufen —
wie waere das gelaufen? Der Test misst zusaetzlich den MARKT-Favoriten
(market_fav_k) als Vergleichsmassstab, weil sich beide in weniger als der
Haelfte der Faelle decken.

Datenbasis: bb_WeatherLadders (weather_ladder_logger.py), eine Zeile je
(Snapshot, Zieltag, var, Stadt, Fenster). Gewertet wird der Snapshot mit dem
gewuenschten LEAD (Default 1 = Vortag ~12:30 UTC, derselbe Zeitpunkt, zu dem
auch weather_minus1_autobuy.py kauft). Lead 0 ist bewusst NICHT der Default:
dort laeuft der Zieltag in vielen Zeitzonen schon, die Preise stehen dann teils
auf 0,001/1,0 und der Test wuerde sich selbst betruegen.

Settlement gegen wu_settle_k (Wunderground) — die Quelle, auf die Polymarket
tatsaechlich settelt, nicht die METAR-Spalte (BA-20-Fall, siehe Logger-Docstring).

FEE: Das Modell aus autopilot.py/crypto_updown_backtest.py lautet
fee = 0,07 * Stueck * min(p, 1-p). Nachgemessen am 28.07. an fuenf echten
Positionen liegt es konsistent zu HOCH (ist/modell = 0,51 .. 0,69, also eine
effektive Rate um 0,04). Der Report gibt deshalb brutto / 0,04 / 0,07 aus,
statt sich auf eine Zahl festzulegen — die Richtung des Ergebnisses haengt
nicht daran.

Aufruf:
  python weather_fav_backtest.py                 # Lead 1, 5 $ je Wette
  python weather_fav_backtest.py --lead 2
  python weather_fav_backtest.py --stake 25
"""

import argparse
import collections
import statistics
import sys

import pymssql

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

FEE_RATE = 0.07          # Repo-Modell (autopilot.py); siehe Docstring
FEE_VARIANTEN = (0.0, 0.04, 0.07)
BAND_LO, BAND_HI = 0.70, 0.90   # Preisband des -1-Autobuy V2, nur als Kontrast


def bet(stake, price, won, rate):
    """Fester Einsatz zum Ask, PnL netto nach Fee."""
    n = stake / price
    return (n if won else 0.0) - stake - rate * n * min(price, 1.0 - price)


def lade_leitern(lead):
    """Je (Zieltag, var, Stadt) die Leiter des spaetesten Snapshots mit dem
    gewuenschten Lead. Nur Zieltage, fuer die ein WU-Settlement vorliegt."""
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor(as_dict=True)
    cur.execute("""
        SELECT target_date, var, city, k, buy_yes, buy_no, offset_fav,
               market_fav_k, wu_settle_k, snapshot_utc
        FROM bb_WeatherLadders
        WHERE wu_settle_k IS NOT NULL AND buy_yes IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()

    def lead_of(r):
        return (r["target_date"] - r["snapshot_utc"].date()).days

    neueste = {}
    for r in rows:
        if lead_of(r) != lead:
            continue
        key = (r["target_date"], r["var"], r["city"])
        if key not in neueste or r["snapshot_utc"] > neueste[key]:
            neueste[key] = r["snapshot_utc"]
    leitern = collections.defaultdict(list)
    for r in rows:
        key = (r["target_date"], r["var"], r["city"])
        if lead_of(r) == lead and r["snapshot_utc"] == neueste.get(key):
            leitern[key].append(r)
    return leitern


def handelbar(preis):
    """0,001 und 1,0 sind Platzhalter fuer "kein Buch" bzw. "entschieden"."""
    return preis is not None and 0.001 < preis < 1.0


def report(name, pnls, stake, extra=""):
    if not pnls:
        print(f"{name:<38} keine Wetten")
        return
    n, tot = len(pnls), sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    sd = statistics.stdev(pnls) if n > 1 else 0.0
    t = (tot / n) / (sd / n ** 0.5) if sd else float("nan")
    print(f"{name:<38} N={n:<4} Treffer={wins:<4} ({wins / n * 100:4.1f}%) "
          f"PnL={tot:+8.2f}$  ROI={tot / (n * stake) * 100:+6.2f}%  t={t:+5.2f}  {extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=1,
                    help="Snapshot-Abstand zum Zieltag in Tagen (Default 1)")
    ap.add_argument("--stake", type=float, default=5.0, help="Einsatz je Wette (Default 5)")
    ap.add_argument("--fee-rate", type=float, default=FEE_RATE,
                    help=f"Fee = rate*Stueck*min(p,1-p) (Default {FEE_RATE})")
    args = ap.parse_args()

    leitern = lade_leitern(args.lead)
    print(f"=== Lead {args.lead} | Einsatz {args.stake:.0f}$ | "
          f"{len(leitern)} gesettelte Stadt-Tage ===\n")

    fav, mfav, lay = [], [], []
    fav_var = collections.defaultdict(list)
    fav_fee = {r: [] for r in FEE_VARIANTEN}
    lay_fee = {r: [] for r in FEE_VARIANTEN}
    preise_fav, preise_mkt = [], []
    paare = []          # (unser_k, markt_k, settle_k) fuer den paarigen Vergleich
    tage = collections.defaultdict(list)

    for key, leiter in sorted(leitern.items()):
        settle = leiter[0]["wu_settle_k"]
        eigen = [r for r in leiter if r["offset_fav"] == 0 and handelbar(r["buy_yes"])]
        markt = [r for r in leiter if r["market_fav_k"] is not None
                 and r["k"] == r["market_fav_k"] and handelbar(r["buy_yes"])]

        if eigen:
            r = eigen[0]
            traf = r["k"] == settle
            pnl = bet(args.stake, r["buy_yes"], traf, args.fee_rate)
            fav.append(pnl)
            fav_var[r["var"]].append(pnl)
            preise_fav.append(r["buy_yes"])
            tage[key[0]].append(pnl)
            for rate in FEE_VARIANTEN:
                fav_fee[rate].append(bet(args.stake, r["buy_yes"], traf, rate))
        if markt:
            r = markt[0]
            mfav.append(bet(args.stake, r["buy_yes"], r["k"] == settle, args.fee_rate))
            preise_mkt.append(r["buy_yes"])
        if eigen and markt:
            paare.append((eigen[0]["k"], markt[0]["k"], settle))

        # Kontrast: -1-Lay im Preisband, OHNE die Gates des Autobuy
        # (Doppel-Kalibrierung, Spannen-Veto, Abstand, P_pess). Das ist
        # ABSICHTLICH nicht das Live-Buch, sondern dessen ungefilterte Huelle.
        cand = [r for r in leiter if r["offset_fav"] is not None
                and abs(r["offset_fav"]) >= 1 and r["buy_no"] is not None
                and BAND_LO <= r["buy_no"] <= BAND_HI]
        if cand:
            r = max(cand, key=lambda r: abs(r["offset_fav"]))
            lay.append(bet(args.stake, r["buy_no"], r["k"] != settle, args.fee_rate))
            for rate in FEE_VARIANTEN:
                lay_fee[rate].append(bet(args.stake, r["buy_no"], r["k"] != settle, rate))

    p_fav = statistics.mean(preise_fav) if preise_fav else 0.0
    report("A) EIGENER Favorit YES", fav, args.stake, f"O-Preis={p_fav:.3f}")
    for v in ("max", "min"):
        report(f"   davon {v}", fav_var[v], args.stake)
    report("B) MARKT-Favorit YES", mfav, args.stake,
           f"O-Preis={statistics.mean(preise_mkt):.3f}" if preise_mkt else "")
    report(f"C) Lay-Band {BAND_LO:.2f}-{BAND_HI:.2f} (ohne Gates)", lay, args.stake)

    print("\n--- Fee-Sensitivitaet (dieselben Wetten) ---")
    labels = {0.0: "brutto", 0.04: "0,04 (live gemessen)", 0.07: "0,07 (Repo-Modell)"}
    for rate in FEE_VARIANTEN:
        report(f"  Favorit  Fee {labels[rate]}", fav_fee[rate], args.stake)
    for rate in FEE_VARIANTEN:
        report(f"  Lay-Band Fee {labels[rate]}", lay_fee[rate], args.stake)

    if paare:
        n = len(paare)
        gleich = sum(1 for o, m, _ in paare if o == m)
        traf_o = sum(1 for o, _, s in paare if o == s)
        traf_m = sum(1 for _, m, s in paare if m == s)
        nur_o = sum(1 for o, m, s in paare if o == s and m != s)
        nur_m = sum(1 for o, m, s in paare if m == s and o != s)
        print(f"\n--- eigener vs. Markt-Favorit (paarig, N={n}) ---")
        print(f"gleicher Bucket        : {gleich} ({gleich / n * 100:.1f} %)")
        print(f"Trefferquote eigener   : {traf_o / n * 100:.1f} %")
        print(f"Trefferquote Markt     : {traf_m / n * 100:.1f} %")
        print(f"nur eigener trifft {nur_o}, nur Markt trifft {nur_m}")
        if nur_o + nur_m:
            chi = (abs(nur_o - nur_m) - 1) ** 2 / (nur_o + nur_m)
            sig = "p < 0,01" if chi > 6.63 else ("p < 0,05" if chi > 3.84 else "n. s.")
            print(f"McNemar chi^2 = {chi:.2f}  ({sig})")
        uneinig = [(o, m, s) for o, m, s in paare if o != m]
        if uneinig:
            u = len(uneinig)
            print(f"nur bei Uneinigkeit (N={u}): eigener trifft "
                  f"{sum(1 for o, _, s in uneinig if o == s) / u * 100:.1f} %, "
                  f"Markt {sum(1 for _, m, s in uneinig if m == s) / u * 100:.1f} %")

    print("\n--- Favorit je Zieltag (kumuliert) ---")
    kum = 0.0
    for d in sorted(tage):
        kum += sum(tage[d])
        print(f"  {d}  N={len(tage[d]):<3} {sum(tage[d]):+7.2f}$   kum {kum:+8.2f}$")


if __name__ == "__main__":
    main()
