#!/usr/bin/env python3
"""
weather_ratchet_test.py — Nowcast-Ratchet-Edge-Test gegen echte Jupiter-Preise.

Verallgemeinerung von weather_chengdu_ratchet.py auf beliebige Staedte.
These: Buckets strikt UNTER dem laufenden Tages-Max sind garantiert unmoeglich
(Max faellt nie) -> Restpreis darauf = Lay-Gratisgeld, falls der Markt nachhinkt.

ERGEBNIS 2026-07-22: FAIL fuer alle bisher getesteten Staedte (Chengdu, Seoul,
Muenchen, Paris, Madrid, London). Details in preregs/weather_ratchet_eu_2026_07_22.md.
Markt antizipatorisch (Vormittag Favorit +6.7..+7.8 Buckets ueber Max), unmoegliche
Buckets sofort genullt, kein systematischer Reprice-Lag. Einzige Instanz eines
echten Lags: Madrid 03.07. 14:02 (Bucket 34 @0.40 bei obs_max 35, ~2-4 min) --
zu selten/fluechtig fuer 5$-Handel mit 2xFee.

Datenquelle: bb_WeatherLatency (Centron), Logger-Historie ab 01.07.2026.
"""
import sys, json
import pymssql, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
DB = dict(server="158.181.48.77", database="dbdata", user="326773", password="Extaler11!")
# Stadt -> UTC-Offset (Juli/DST)
CITIES = {"Chengdu": 8, "Seoul": 9, "Munich": 2, "Paris": 2, "Madrid": 2, "London": 1}


def dead_mass(row):
    try:
        p = json.loads(row["all_prices"])
    except Exception:
        return None
    cur = round(row["obs_max"])
    return sum(v for k, v in p.items() if int(k) < cur)


def test_city(con, city, off):
    df = pd.read_sql(
        "SELECT market_date,ts_utc,obs_max,fav_bucket,all_prices FROM bb_WeatherLatency "
        f"WHERE city='{city}' AND all_prices IS NOT NULL AND obs_max IS NOT NULL "
        "ORDER BY ts_utc", con)
    if df.empty:
        print(f"{city}: keine Daten")
        return
    df["ts_utc"] = pd.to_datetime(df["ts_utc"])
    df["local"] = df["ts_utc"] + pd.Timedelta(hours=off)
    df["rmax"] = df["obs_max"].round()
    df["dead"] = df.apply(dead_mass, axis=1)

    rows = []
    for _, g in df.groupby("market_date"):
        g = g.sort_values("ts_utc").reset_index(drop=True)
        for i in g.index[g["rmax"].diff() > 0]:
            t0 = g.loc[i, "ts_utc"]
            w = g[(g["ts_utc"] >= t0 + pd.Timedelta(minutes=8)) &
                  (g["ts_utc"] <= t0 + pd.Timedelta(minutes=12))]
            rows.append((g.loc[i, "dead"], w["dead"].iloc[0] if len(w) else None))
    ev = pd.DataFrame(rows, columns=["t0", "t10"])
    df["fa"] = df["fav_bucket"] - df["rmax"]
    morn = df[df["local"].dt.hour.between(6, 11)]
    print(f"{city:8} {df['market_date'].nunique():2}d | dead mean {df['dead'].mean():.4f} "
          f"max {df['dead'].max():.3f} | Ratchet {len(ev):3} Schritte, >0.02: "
          f"{(ev['t0'] > 0.02).sum()}/{len(ev)} | Fav-Vorlauf {morn['fa'].mean():+.1f}")


if __name__ == "__main__":
    con = pymssql.connect(**DB)
    for c, o in CITIES.items():
        test_city(con, c, o)
