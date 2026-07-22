#!/usr/bin/env python3
"""
weather_seoul_seabreeze.py — Seoul (RKSI) Seebrise-Deckel-Analyse.

BEFUND (2026-07-22, 8 Sommer Jun-Aug 2018-2026, IEM ASOS, N=780 Tage):
Die Settlement-Station Seoul=RKSI (Incheon) ist eine KUeSTENstation und liegt
strukturell ~2 C unter den staadtnahen/inlaendischen Nachbarn (RKSS Gimpo NO 32km,
RKSM Seoul-AB O 59km, RKSO Osan SO 66km) — an 49/52 Tagen die kaelteste.

Der Offset RKSS-RKSI ist MONOTON in RKSIs eigener Nachmittags-Windrichtung
(lokale 12-16h): Ostsektor (Landluft, Nachbarn liegen im Osten) +0.5 C, Westsektor
(Seebrise vom Gelben Meer) +2.4 C. D.h.:
  - Ostwind-Tage: RKSS ~= RKSI (Bucket ±1: 85%), RKSS taugt als Echtzeit-Proxy.
  - Westwind-Tage (72% der Sommertage!): RKSI wird gedeckelt, Nachbarn sind
    Fehlsignal (Bucket exakt nur 7% roh, 36% selbst mit -2C-Korrektur).
Zeitliches Lead-Lag RKSS->RKSI ist ~0 (kein Fruehwarn-Vorsprung); nur RKSM/RKSO
sind ausserdem zu verrauscht (std 1.4-2.1 vs RKSS 0.94) -> nur RKSS nutzen.

FOLGERUNG (tradbar): Nicht die Nachbarn sind der Edge, sondern RKSIs Windrichtung
selbst. Dreht der Nachmittagswind auf West/Seebrise, ist RKSIs Settlement-Max
gedeckelt -> Downside-Signal ggue. einem forecast-heiss gepreisten Markt.
Getestet gegen echte Jupiter-Preise in weather_seoul_seabreeze_market.py.

Quelle: IEM ASOS-Archiv (mesonet.agron.iastate.edu), kostenlos, kein Key.
"""
import io, sys, time
import requests, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

STATIONS = ["RKSI", "RKSS", "RKSM", "RKSO"]
TZ = 9  # Seoul local = UTC+9
EAST = {"NE", "E", "SE"}
WEST = {"W", "NW", "SW"}


def load(y1=2018, y2=2026):
    frames = []
    for yr in range(y1, y2 + 1):
        url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
               + "".join(f"station={s}&" for s in STATIONS)
               + "data=tmpc&data=drct&data=sknt"
               + f"&year1={yr}&month1=6&day1=1&year2={yr}&month2=8&day2=31"
               + "&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T")
        for _ in range(3):
            try:
                t = requests.get(url, timeout=180).text
                if t.startswith("station"):
                    frames.append(pd.read_csv(io.StringIO(t)))
                    break
            except Exception:
                time.sleep(3)
    df = pd.concat(frames, ignore_index=True)
    for c in ("tmpc", "drct", "sknt"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["valid"] = pd.to_datetime(df["valid"])
    df = df.dropna(subset=["tmpc"])
    df["local"] = df["valid"] + pd.Timedelta(hours=TZ)
    df["date"] = df["local"].dt.date
    return df


def sector(d):
    if pd.isna(d):
        return None
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((d + 22.5) % 360) // 45)]


def daily_table(df):
    """Pro Tag: Stations-Maxima + RKSI-Nachmittagswind (Sektor, Speed)."""
    piv = df.pivot_table(index="date", columns="station", values="tmpc",
                         aggfunc="max").dropna(subset=["RKSI"])
    aft = df[(df.station == "RKSI") & (df.local.dt.hour.between(12, 16))]
    wind = aft.groupby("date").agg(drct=("drct", "median"), sknt=("sknt", "median"))
    wind["sec"] = wind["drct"].apply(sector)
    j = piv.join(wind, how="inner").dropna(subset=["sec"])
    j["regime"] = j["sec"].apply(lambda s: "OST" if s in EAST else ("WEST" if s in WEST else "N/S"))
    return j


if __name__ == "__main__":
    df = load()
    j = daily_table(df)
    print(f"N={len(j)} Tage (Jun-Aug 2018-2026)\n")
    print("Offset RKSS-RKSI nach RKSI-Nachmittagswind:")
    for sec in ["E", "NE", "SE", "N", "S", "SW", "NW", "W"]:
        g = j[j.sec == sec]
        if len(g) < 3:
            continue
        off = g["RKSS"] - g["RKSI"]
        print(f"  {sec:2} n={len(g):3}: {off.mean():+.2f}+-{off.std():.2f}")
    for name, secs in [("OSTSEKTOR", EAST), ("WESTSEKTOR", WEST)]:
        g = j[j.sec.isin(secs)]
        off = (g["RKSS"] - g["RKSI"])
        rb, sb = g["RKSI"].round(), g["RKSS"].round()
        print(f"\n{name} n={len(g)}: RKSS-RKSI {off.mean():+.2f}+-{off.std():.2f} | "
              f"Bucket exakt {(sb == rb).mean() * 100:.0f}% |diff|<=1 {(abs(sb - rb) <= 1).mean() * 100:.0f}%")
