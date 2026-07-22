#!/usr/bin/env python3
"""
weather_seoul_seabreeze_market.py — Markt-Test des Seoul-Seebrise-Deckels.

ERGEBNIS 2026-07-22: FAIL (Details in preregs/weather_seoul_seabreeze_2026_07_22.md).
Der Markt preist die Seebrise-Deckelung bereits am fruehen Nachmittag ein
(fav_1400 ~= obs_1400 -> Markt trackt live das Stationshoch, kein Forecast-Spalt).

Prueft je Seoul-Markttag: Markt-Favorit-Bucket zum Entscheidungs-Snapshot
(lokal <=14h) minus realisierter Settlement-Bucket, aufgeteilt nach RKSI-
Nachmittags-Windregime (OST=Landluft, WEST=Seebrise). Datenquelle Markt:
bb_WeatherLatency (Centron). Windregime: IEM ASOS via weather_seoul_seabreeze.
"""
import sys, io
import pymssql, pandas as pd, requests
from weather_seoul_seabreeze import sector, EAST, WEST

sys.stdout.reconfigure(encoding="utf-8")
DB = dict(server="158.181.48.77", database="dbdata", user="326773", password="Extaler11!")
MON = {m: i for i, m in enumerate(
    ["", "January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"])}


def wind_regime(year=2026, m1=7, d1=1, m2=7, d2=22):
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=RKSI&"
           "data=drct&data=sknt"
           f"&year1={year}&month1={m1}&day1={d1}&year2={year}&month2={m2}&day2={d2}"
           "&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T")
    w = pd.read_csv(io.StringIO(requests.get(url, timeout=120).text))
    w["valid"] = pd.to_datetime(w["valid"])
    w["drct"] = pd.to_numeric(w["drct"], errors="coerce")
    w["local"] = w["valid"] + pd.Timedelta(hours=9)
    w["d"] = w["local"].dt.date
    aft = w[w.local.dt.hour.between(12, 16)]
    wind = aft.groupby("d").agg(drct=("drct", "median"))
    wind["sec"] = wind["drct"].apply(sector)
    wind["regime"] = wind["sec"].apply(
        lambda s: "OST" if s in EAST else ("WEST" if s in WEST else "N/S"))
    return wind


def main():
    wind = wind_regime()
    con = pymssql.connect(**DB)
    df = pd.read_sql("SELECT market_date,ts_utc,obs_max,fav_bucket "
                     "FROM bb_WeatherLatency WHERE city='Seoul' ORDER BY ts_utc", con)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"])
    df["local"] = df["ts_utc"] + pd.Timedelta(hours=9)
    df["d"] = df["local"].dt.date
    df["mkt_d"] = df["market_date"].apply(
        lambda s: pd.Timestamp(2026, MON[s.split()[0]], int(s.split()[1])).date())
    df = df[df["d"] == df["mkt_d"]]  # nur Messungen am Zieltag

    rows = []
    for md, g in df.groupby("mkt_d"):
        dec = g[g["local"].dt.hour <= 14]
        if dec.empty:
            continue
        snap = dec.iloc[-1]
        rows.append(dict(date=md, regime=wind["regime"].get(md), sec=wind["sec"].get(md),
                         fav_1400=snap["fav_bucket"], realized=round(g["obs_max"].max()),
                         mkt_minus_real=snap["fav_bucket"] - round(g["obs_max"].max())))
    r = pd.DataFrame(rows).sort_values("date")
    print(r.to_string(index=False))
    print("\nMarkt-Favorit(14h) - realisiert, nach Windregime (>0 = Markt zu heiss):")
    for reg in ["OST", "WEST", "N/S"]:
        g = r[r.regime == reg]
        if len(g):
            print(f"  {reg:4} n={len(g):2}: mean {g['mkt_minus_real'].mean():+.2f}")


if __name__ == "__main__":
    main()
