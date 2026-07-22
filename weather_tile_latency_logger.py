#!/usr/bin/env python3
"""
weather_tile_latency_logger.py — Kachel-vs-History-Tabelle-Latenz-Messer (PRIO 1).

HYPOTHESE (Nutzer, 2026-07-22): Auf wunderground.com (= Polymarket/Jupiter-
Settlement-Quelle) updatet die grosse **Current-Kachel** (v3-Current, TWC/PWS,
quasi-live) SCHNELLER als die **History-Tabelle** (nur stuendlicher Eintrag).
An Kipp-Tagen mit X,9 Grad zeigt die Kachel schon X+1, waehrend Tabelle+METAR
+Markt noch X sehen -> waehrend dieses Fensters ist der X-Bucket ueberbewertet
(Lay = Value) bzw. X+1 zu billig. Beleg 22.07. Chengdu: Kachel 102 F (=38,9 ->
39), Tabelle/METAR 38, Markt settlete 38 = NO.

WICHTIG — WU speichert intern GANZE GRAD FAHRENHEIT. Die C-Buckets entstehen
durch F->C-Umrechnung + Rundung; genau an der X,9-Grenze divergieren Kachel und
Tabelle. Deshalb loggen wir BEIDE in F UND C.

Dieser Logger schreibt NUR die Wetter-Seite (Kachel/Tabelle/METAR). Die
Marktpreise (fav_bucket/all_prices) schreibt weather_latency_logger.py bereits
alle 2 min nach bb_WeatherLatency -> Analyse joint beide per (city, ts_utc).

Aufruf:
  python weather_tile_latency_logger.py --dry --once
  python weather_tile_latency_logger.py --once
  python weather_tile_latency_logger.py --loop --interval 120
  python weather_tile_latency_logger.py --loop --cities Chengdu,Shenzhen,Madrid
"""
import argparse
import logging
import sys
import time
from datetime import datetime, timezone

import requests

try:
    import pymssql
except ImportError:
    pymssql = None
import airportsdata

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Hardcodierte Centron-Creds — bewusste Projektentscheidung (siehe CLAUDE.md).
DB_CONFIG = {
    "server": "158.181.48.77", "database": "dbdata",
    "user": "326773", "password": "Extaler11!",
}

WU_KEY = "e1f10a1e78da46f5b10a1e78da96f525"  # oeffentlicher Web-Key der wunderground.com-Seite
WU_TILE = "https://api.weather.com/v3/wx/observations/current"
WU_TABLE = "https://api.weather.com/v1/location/{icao}:9:{cc}/observations/historical.json"
METAR = "https://aviationweather.gov/api/data/metar"
UA = {"User-Agent": "boersenbot-tile-latency/1.0"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("tile_latency")

_ap = airportsdata.load("ICAO")


def f2c(f):
    return None if f is None else (f - 32) * 5.0 / 9.0


def db():
    if pymssql is None:
        raise RuntimeError("pymssql fehlt")
    return pymssql.connect(**DB_CONFIG, autocommit=True)


DDL = [
    """
    IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name='bb_WeatherTileLatency')
    CREATE TABLE bb_WeatherTileLatency (
        id            BIGINT IDENTITY(1,1) PRIMARY KEY,
        city          NVARCHAR(32)  NOT NULL,
        station       NVARCHAR(8)   NULL,
        cc            NVARCHAR(4)   NULL,
        ts_utc        DATETIME      NOT NULL,     -- Messzeitpunkt (Poll)
        local_time    NVARCHAR(20)  NULL,
        -- KACHEL (v3 current)
        tile_f        INT           NULL,         -- Grad Fahrenheit (WU-intern)
        tile_c        FLOAT         NULL,         -- daraus konvertiert
        tile_bucket   INT           NULL,         -- round(tile_c)
        tile_valid_utc DATETIME     NULL,         -- validTimeUtc der Kachel
        -- HISTORY-TABELLE (settlementrelevant)
        tbl_last_f    INT           NULL,         -- letzter stuendlicher Tabelleneintrag (F)
        tbl_last_c    FLOAT         NULL,
        tbl_last_utc  DATETIME      NULL,         -- valid_time_gmt des letzten Eintrags
        tbl_max_f     INT           NULL,         -- Tages-Max bisher aus der Tabelle (F)
        tbl_max_c     FLOAT         NULL,
        tbl_bucket    INT           NULL,         -- round(tbl_max_c) = aktueller Settlement-Stand
        -- METAR (Referenz)
        metar_max_c   FLOAT         NULL,
        metar_bucket  INT           NULL,
        -- abgeleitet: zeigt die Kachel einen HOEHEREN Bucket als die Tabelle?
        tile_ahead    INT           NULL,         -- tile_bucket - tbl_bucket
        logged_utc    DATETIME      NOT NULL DEFAULT GETUTCDATE()
    )
    """,
    """
    IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name='ix_wtl_city')
    CREATE INDEX ix_wtl_city ON bb_WeatherTileLatency(city, ts_utc)
    """,
]


def ensure_tables(conn):
    cur = conn.cursor()
    for stmt in DDL:
        cur.execute(stmt)
    log.info("Tabelle bb_WeatherTileLatency sichergestellt.")


def _get(url, params=None):
    for a in range(3):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 429:
                time.sleep(3 * (a + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if a == 2:
                raise
            time.sleep(2 * (a + 1))
    return None


def city_stations(conn):
    """Autoritative City->Station-Zuordnung aus bb_WeatherLatency (haeufigste je Stadt)."""
    cur = conn.cursor()
    cur.execute("""SELECT city, station, COUNT(*) n FROM bb_WeatherLatency
                   WHERE station IS NOT NULL GROUP BY city, station""")
    best = {}
    for city, station, n in cur.fetchall():
        if city not in best or n > best[city][1]:
            best[city] = (station, n)
    out = {}
    for city, (station, _) in best.items():
        ap = _ap.get(station)
        if ap:
            out[city] = (station, ap["lat"], ap["lon"], ap["country"])
        else:
            log.warning(f"{city}: Station {station} nicht in airportsdata — uebersprungen")
    return out


def fetch_tile(lat, lon):
    """v3-Current in imperial (F, feinste Aufloesung an der Grenze)."""
    d = _get(WU_TILE, {"geocode": f"{lat},{lon}", "units": "e",
                       "language": "en-US", "format": "json", "apiKey": WU_KEY})
    if not d:
        return None
    tf = d.get("temperature")
    vt = d.get("validTimeUtc")
    return {"tile_f": tf, "tile_valid_utc": datetime.fromtimestamp(vt, timezone.utc) if vt else None}


def fetch_table(icao, cc):
    """Heutige History-Tabelle (imperial): letzter Eintrag + Tages-Max bisher."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    d = _get(WU_TABLE.format(icao=icao, cc=cc),
             {"apiKey": WU_KEY, "units": "e", "startDate": today})
    if not d:
        return None
    obs = [o for o in (d.get("observations") or []) if o.get("temp") is not None]
    if not obs:
        return {"tbl_last_f": None, "tbl_last_utc": None, "tbl_max_f": None}
    obs.sort(key=lambda o: o["valid_time_gmt"])
    last = obs[-1]
    return {"tbl_last_f": last["temp"],
            "tbl_last_utc": datetime.fromtimestamp(last["valid_time_gmt"], timezone.utc),
            "tbl_max_f": max(o["temp"] for o in obs)}


def fetch_metar_max(icao):
    """Tages-Max aus rohem METAR (aviationweather), Referenz."""
    try:
        d = _get(METAR, {"ids": icao, "format": "json", "hours": 18})
    except Exception:
        return None
    if not d:
        return None
    today = datetime.now(timezone.utc).date()
    temps = [x["temp"] for x in d if x.get("temp") is not None
             and datetime.fromtimestamp(x["obsTime"], timezone.utc).date() == today]
    return max(temps) if temps else None


def poll_city(conn, city, icao, lat, lon, cc, dry):
    now = datetime.now(timezone.utc)
    tile = fetch_tile(lat, lon) or {}
    tbl = fetch_table(icao, cc) or {}
    metar_c = fetch_metar_max(icao)

    tile_c = f2c(tile.get("tile_f"))
    tbl_max_c = f2c(tbl.get("tbl_max_f"))
    tbl_last_c = f2c(tbl.get("tbl_last_f"))
    tile_bucket = round(tile_c) if tile_c is not None else None
    tbl_bucket = round(tbl_max_c) if tbl_max_c is not None else None
    metar_bucket = round(metar_c) if metar_c is not None else None
    tile_ahead = (tile_bucket - tbl_bucket) if (tile_bucket is not None and tbl_bucket is not None) else None

    row = dict(city=city, station=icao, cc=cc, ts_utc=now,
               local_time=None,
               tile_f=tile.get("tile_f"), tile_c=tile_c, tile_bucket=tile_bucket,
               tile_valid_utc=tile.get("tile_valid_utc"),
               tbl_last_f=tbl.get("tbl_last_f"), tbl_last_c=tbl_last_c,
               tbl_last_utc=tbl.get("tbl_last_utc"),
               tbl_max_f=tbl.get("tbl_max_f"), tbl_max_c=tbl_max_c, tbl_bucket=tbl_bucket,
               metar_max_c=metar_c, metar_bucket=metar_bucket, tile_ahead=tile_ahead)

    flag = "  <-- KACHEL VORAUS!" if tile_ahead and tile_ahead > 0 else ""
    log.info(f"{city:12} tile {row['tile_f']}F/{tile_bucket}C | "
             f"table max {row['tbl_max_f']}F/{tbl_bucket}C | metar {metar_bucket}C{flag}")
    if dry:
        return
    cur = conn.cursor()
    cols = ",".join(row.keys())
    ph = ",".join(["%s"] * len(row))
    cur.execute(f"INSERT INTO bb_WeatherTileLatency ({cols}) VALUES ({ph})", tuple(row.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--cities", default="", help="Kommaliste; leer = alle aus bb_WeatherLatency")
    args = ap.parse_args()

    conn = db()
    if not args.dry:
        ensure_tables(conn)
    stations = city_stations(conn)
    if args.cities:
        want = {c.strip() for c in args.cities.split(",")}
        stations = {c: v for c, v in stations.items() if c in want}
    log.info(f"{len(stations)} Staedte: {', '.join(sorted(stations))}")

    def cycle():
        for city, (icao, lat, lon, cc) in sorted(stations.items()):
            try:
                poll_city(conn, city, icao, lat, lon, cc, args.dry)
            except Exception as e:
                log.warning(f"{city}: {e}")
            time.sleep(0.5)

    if args.loop:
        while True:
            cycle()
            time.sleep(args.interval)
    else:
        cycle()


if __name__ == "__main__":
    main()
