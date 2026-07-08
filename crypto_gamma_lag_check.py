#!/usr/bin/env python3
"""
crypto_gamma_lag_check.py — Erste Sichtung der Gamma-Lag-These (2026-07-06):
Hinkt der Jupiter-Prediction-Preis dem Binance-Spot beim Über-/Unterschreiten
des price_to_beat hinterher, insbesondere in den letzten Sekunden vor Close?

Pro Event (bb_CryptoUpDown15m):
  1. spot_cross_secs  = secs_to_close beim ERSTEN Tick, an dem spot dauerhaft
                        auf der Seite des finalen Ergebnisses liegt (letzter
                        Vorzeichenwechsel von spot - price_to_beat)
  2. market_cross_secs = secs_to_close beim ERSTEN Tick, an dem der Preis der
                        GEWINNSEITE (up_buy falls result='Up', sonst down_buy)
                        dauerhaft > 0.5 steht (letzter Wechsel < 0.5 -> > 0.5)
  3. lag_secs = spot_cross_secs - market_cross_secs
               (positiv = Markt zieht NACH, wie in der Session-These vermutet;
                negativ = Markt-Preis bewegt sich VOR dem Spot-Cross, z. B. weil
                Binance nicht exakt die Chainlink-Quelle ist)

Nur explorative Sichtung (Rohverteilung von lag_secs), KEIN Backtest/Pre-Reg.

BEFUND 2026-07-08 (btc, min-ticks=20): These NICHT bestaetigt.
  n=649 (von 774 Events; 125 Rand-/Lueckenfaelle verworfen)
  lag_secs: median 0.0s, mean 3.4s, stdev 82.1s, p10/p90 -20s/+30s
  Markt nach Spot: 129 | Markt vor Spot: 119 | gleicher Tick: 401 (62 %)
  -> Markt preist den price_to_beat-Cross praktisch zeitgleich mit dem Spot
     ein; die Verteilung ist symmetrisch um 0, kein ausnutzbares Lag.
  Vorbehalt: Tick-Aufloesung des Loggers (~10s+) kann ein Lag von wenigen
  Sekunden in der Schlussphase nicht aufloesen; dafuer braeuchte es einen
  Sekunden-Logger. Angesichts des klaren Nullbefunds nicht weiterverfolgt.

Aufruf:
  python crypto_gamma_lag_check.py --asset btc
  python crypto_gamma_lag_check.py --asset btc --min-ticks 30
"""

import argparse
import statistics
import sys

import pymssql

DB_CONFIG = {
    "server": "158.181.48.77",
    "database": "dbdata",
    "user": "326773",
    "password": "Extaler11!",
}

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def fetch_events(asset, min_ticks):
    conn = pymssql.connect(**DB_CONFIG, autocommit=True)
    cur = conn.cursor(as_dict=True)
    cur.execute("""
        SELECT event_id
        FROM bb_CryptoUpDown15m
        WHERE asset=%s AND settled=1 AND result IS NOT NULL
        GROUP BY event_id
        HAVING COUNT(*) >= %s
    """, (asset, min_ticks))
    event_ids = [r["event_id"] for r in cur.fetchall()]

    events = {}
    for eid in event_ids:
        cur.execute("""
            SELECT ts_utc, secs_to_close, up_buy, down_buy, spot, price_to_beat, result
            FROM bb_CryptoUpDown15m
            WHERE event_id=%s
            ORDER BY ts_utc ASC
        """, (eid,))
        rows = cur.fetchall()
        events[eid] = rows
    conn.close()
    return events


def find_last_flip_secs(rows, side_positive_fn):
    """Letzter Tick VOR dem finalen, dauerhaften Wechsel auf die 'positive' Seite ->
    gibt secs_to_close DES ERSTEN Ticks NACH dem letzten Flip zurueck (0 falls immer
    positiv, None falls nie positiv wird)."""
    if not rows:
        return None
    flags = [side_positive_fn(r) for r in rows]
    if flags[-1] is not True:
        return None  # endet nicht auf der erwarteten Seite -> Datenluecke/Rand, verwerfen
    # rueckwaerts suchen: letzter Index, an dem flags False ist
    last_false = None
    for i in range(len(flags) - 1, -1, -1):
        if flags[i] is False:
            last_false = i
            break
    if last_false is None:
        # war die ganze Zeit schon auf der Gewinnerseite
        return rows[0]["secs_to_close"]
    if last_false + 1 >= len(rows):
        return None
    return rows[last_false + 1]["secs_to_close"]


def analyze(rows):
    if not rows or rows[0]["price_to_beat"] is None:
        return None
    result = rows[0]["result"]
    if result not in ("Up", "Down"):
        return None
    ptb = rows[0]["price_to_beat"]

    def spot_side_positive(r):
        if r["spot"] is None:
            return None
        is_up = r["spot"] >= ptb
        return is_up if result == "Up" else (not is_up)

    def market_side_positive(r):
        price = r["up_buy"] if result == "Up" else r["down_buy"]
        if price is None:
            return None
        return price > 0.5

    spot_secs = find_last_flip_secs(rows, spot_side_positive)
    market_secs = find_last_flip_secs(rows, market_side_positive)
    if spot_secs is None or market_secs is None:
        return None
    return {
        "spot_cross_secs": spot_secs,
        "market_cross_secs": market_secs,
        "lag_secs": spot_secs - market_secs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="btc")
    ap.add_argument("--min-ticks", type=int, default=20,
                    help="Mindest-Tickzahl je Event, sonst zu lueckenhaft (default 20)")
    args = ap.parse_args()

    print(f"Lade Events fuer {args.asset} (>= {args.min_ticks} Ticks, settled)...")
    events = fetch_events(args.asset, args.min_ticks)
    print(f"{len(events)} Events gefunden.")

    lags = []
    skipped = 0
    for eid, rows in events.items():
        r = analyze(rows)
        if r is None:
            skipped += 1
            continue
        lags.append(r["lag_secs"])

    print(f"Ausgewertet: {len(lags)}  |  Uebersprungen (Rand-/Luecken-Faelle): {skipped}")
    if not lags:
        print("Keine auswertbaren Events -- evtl. min-ticks senken oder mehr Logging-Zeit abwarten.")
        return

    lags_sorted = sorted(lags)
    print(f"\nlag_secs = spot_cross_secs - market_cross_secs (positiv = Markt hinkt hinterher)")
    print(f"  n         = {len(lags)}")
    print(f"  mean      = {statistics.mean(lags):.1f}s")
    print(f"  median    = {statistics.median(lags):.1f}s")
    print(f"  stdev     = {statistics.stdev(lags):.1f}s" if len(lags) > 1 else "  stdev     = n/a")
    print(f"  min/max   = {min(lags):.1f}s / {max(lags):.1f}s")
    n = len(lags_sorted)
    print(f"  p10/p90   = {lags_sorted[int(n*0.1)]:.1f}s / {lags_sorted[int(n*0.9)]:.1f}s")
    pos = sum(1 for x in lags if x > 0)
    neg = sum(1 for x in lags if x < 0)
    zero = len(lags) - pos - neg
    print(f"  Markt hinkt nach (>0): {pos}  |  Markt vorher (<0): {neg}  |  gleichzeitig (0): {zero}")


if __name__ == "__main__":
    main()
