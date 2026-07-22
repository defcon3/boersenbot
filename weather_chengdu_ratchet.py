#!/usr/bin/env python3
"""
weather_chengdu_ratchet.py — Chengdu Ratchet-Edge-Test gegen echte Marktpreise.

ERGEBNIS 2026-07-22: FAIL (Details preregs/weather_chengdu_ratchet_2026_07_22.md).
Der Nowcast-Ratchet hat in Chengdu (Settlement ZUUU) keinen Edge:
- Buckets strikt unter dem laufenden Tages-Max (garantierte Verlierer) tragen
  nie mehr als ~0.01 Restpreis; Event-Study ueber 63 Ratchet-Schritte -> 0/63
  mit >0.02 toter Masse, kein Reprice-Lag (t0 = +10min = +20min = 0.0024).
- Markt ist brutal antizipatorisch: vormittags (6-11h lokal) liegt der Favorit
  +6.6 Buckets ueber dem bisherigen Max -> untere Buckets tot, bevor die Station
  sie erreicht. Overshoot (realisiert > Vormittags-Favorit) nur 2/11 Tagen, klein.

Datenquelle: bb_WeatherLatency (Centron), Chengdu, Logger-Historie ab 01.07.2026.
"""
import sys, json
import pymssql, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
DB = dict(server="158.181.48.77", database="dbdata", user="326773", password="Extaler11!")


def load():
    con = pymssql.connect(**DB)
    df = pd.read_sql(
        "SELECT market_date,ts_utc,obs_max,fav_bucket,all_prices FROM bb_WeatherLatency "
        "WHERE city='Chengdu' AND all_prices IS NOT NULL AND obs_max IS NOT NULL "
        "ORDER BY ts_utc", con)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"])
    df["local"] = df["ts_utc"] + pd.Timedelta(hours=8)
    df["rmax"] = df["obs_max"].round()
    df["dead"] = df.apply(_dead_mass, axis=1)
    return df


def _dead_mass(row):
    """Summe Marktpreis auf Buckets strikt < round(obs_max) = garantierte Verlierer."""
    try:
        p = json.loads(row["all_prices"])
    except Exception:
        return None
    cur = round(row["obs_max"])
    return sum(v for k, v in p.items() if int(k) < cur)


def ratchet_events(df):
    """Tote Masse beim Ratchet-Schritt (rmax steigt) und 10/20 min danach."""
    rows = []
    for md, g in df.groupby("market_date"):
        g = g.sort_values("ts_utc").reset_index(drop=True)
        for i in g.index[g["rmax"].diff() > 0]:
            t0 = g.loc[i, "ts_utc"]

            def at(mins):
                w = g[(g["ts_utc"] >= t0 + pd.Timedelta(minutes=mins - 2)) &
                      (g["ts_utc"] <= t0 + pd.Timedelta(minutes=mins + 2))]
                return w["dead"].iloc[0] if len(w) else None
            rows.append(dict(day=md, dead_t0=g.loc[i, "dead"], dead_10=at(10), dead_20=at(20)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load()
    ev = ratchet_events(df)
    print(f"Tote Masse (Buckets < laufendes Max): mean {df['dead'].mean():.4f} max {df['dead'].max():.3f}")
    print(f"Ratchet-Schritte N={len(ev)}: dead t0 {ev['dead_t0'].mean():.4f} "
          f"+10min {ev['dead_10'].mean():.4f} +20min {ev['dead_20'].mean():.4f}")
    print(f"Schritte mit toter Masse >0.02: {(ev['dead_t0'] > 0.02).sum()}/{len(ev)}")
    df["fav_ahead"] = df["fav_bucket"] - df["rmax"]
    morn = df[df["local"].dt.hour.between(6, 11)]
    print(f"Vormittag: Favorit liegt {morn['fav_ahead'].mean():+.2f} Buckets ueber dem bisherigen Max")
