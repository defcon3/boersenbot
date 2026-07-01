#!/usr/bin/env python3
"""
analyze_weather_latency.py — Reprice-Lag-Auswertung von bb_WeatherLatency (These C).

Kernfrage: Wenn das gemessene Tages-Hoch (Auflösungs-Station) in einen Bucket
steigt, der ÜBER dem aktuellen Markt-Favoriten liegt (Markt „hinkt hinterher"),
wie viele Minuten braucht der Markt, bis sein Favorit nachzieht? Und zu welchem
Preis handelt der bereits erreichte (wahre) Bucket in diesem Fenster?

- Lag > 0 und Bucket-Preis niedrig  -> handelbares Latenz-Fenster (Edge-Kandidat)
- Lag ~ 0 (Markt antizipiert/führt)  -> These C tot

„Markt-hinterher-Episode" = zusammenhängende Messungen mit obs_bucket > fav_bucket.
Lag = Zeit von Episodenbeginn bis fav_bucket >= obs_bucket (bzw. obs_bucket_price>=0.5).
Innerhalb eines Tages ist obs_bucket monoton (laufendes Max) — Episoden sind sauber.

Aufruf:  python analyze_weather_latency.py [--min-price 0.5]
"""
import argparse
from collections import defaultdict

import pymssql

DB_CONFIG = {
    "server": "158.181.48.77",
    "database": "dbdata",
    "user": "326773",
    "password": "Extaler11!",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catch-thresh", type=float, default=0.5,
                    help="obs_bucket_price-Schwelle, ab der der Markt als 'nachgezogen' gilt")
    a = ap.parse_args()

    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor(as_dict=True)
    cur.execute(
        "SELECT city, market_date, ts_utc, obs_bucket, fav_bucket, fav_price, obs_bucket_price "
        "FROM bb_WeatherLatency ORDER BY city, market_date, ts_utc")
    rows = cur.fetchall()
    print(f"Zeilen gesamt: {len(rows)}")

    series = defaultdict(list)
    for r in rows:
        series[(r["city"], r["market_date"])].append(r)
    print(f"Stadt/Tag-Serien: {len(series)}\n")

    episodes = []   # (city, day, bucket, start_ts, lag_min, entry_price, caught)
    for (city, day), rs in series.items():
        i = 0
        n = len(rs)
        while i < n:
            r = rs[i]
            ob, fb = r["obs_bucket"], r["fav_bucket"]
            # Episodenbeginn: Station-Hoch liegt ÜBER Markt-Favorit
            if ob is not None and fb is not None and ob > fb:
                start = r["ts_utc"]
                entry = r["obs_bucket_price"]   # Preis des wahren Buckets bei Episodenbeginn
                lag = None
                caught = False
                j = i
                while j < n:
                    rj = rs[j]
                    obp = rj["obs_bucket_price"]
                    fbj = rj["fav_bucket"]
                    if (fbj is not None and rj["obs_bucket"] is not None and fbj >= rj["obs_bucket"]) \
                       or (obp is not None and obp >= a.catch_thresh):
                        lag = (rj["ts_utc"] - start).total_seconds() / 60.0
                        caught = True
                        break
                    j += 1
                episodes.append((city, day, ob, start, lag, entry, caught))
                i = j + 1 if caught else n   # nach dem Nachziehen weitersuchen
            else:
                i += 1

    if not episodes:
        print("Noch KEINE 'Markt-hinterher'-Episoden (obs_bucket > fav_bucket) in den Daten.")
        print("=> Entweder Markt antizipiert immer (These C tot) ODER noch zu wenig Peak-Daten.")
        return

    caught = [e for e in episodes if e[6] and e[4] is not None]
    open_ep = [e for e in episodes if not e[6]]
    print(f"Markt-hinterher-Episoden: {len(episodes)}  (nachgezogen: {len(caught)}, offen/Tagesende: {len(open_ep)})\n")

    if caught:
        lags = sorted(e[4] for e in caught)
        entries = [e[5] for e in caught if e[5] is not None]
        med = lags[len(lags) // 2]
        print(f"Reprice-Lag (Minuten bis Markt nachzieht): "
              f"Median {med:.1f} | Mittel {sum(lags)/len(lags):.1f} | Min {lags[0]:.1f} | Max {lags[-1]:.1f}")
        big = [l for l in lags if l >= 5]
        print(f"  Episoden mit Lag >= 5 min (handelbar?): {len(big)}/{len(lags)}")
        if entries:
            print(f"  Preis des wahren Buckets bei Episodenbeginn: "
                  f"Median {sorted(entries)[len(entries)//2]:.2f} | Mittel {sum(entries)/len(entries):.2f}")
        print("\n  Top-Episoden nach Lag:")
        for city, day, ob, start, lag, entry, _ in sorted(caught, key=lambda e: -(e[4] or 0))[:10]:
            print(f"    {city:12} {day:8} Bucket {ob}C  Lag {lag:5.1f}min  Einstiegspreis "
                  f"{entry if entry is not None else '-'}  @ {start:%m-%d %H:%M}Z")

    print("\nFAZIT-Leitfaden: Median-Lag >= ~5 min UND Einstiegspreis niedrig (<0.4) "
          "-> handelbares Fenster, Trading-Daemon lohnt Pruefung (G1-G5 netto nach Fee).\n"
          "Median-Lag ~0 -> Markt reagiert sofort/antizipiert -> These C tot, Wetter als FAIL committen.")


if __name__ == "__main__":
    main()
