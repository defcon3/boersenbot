#!/usr/bin/env python3
"""
weather_latency_logger.py — Reprice-Latenz-Messer für Jupiters Wetter-Märkte
("Highest temperature in {Stadt} on {Datum}?", 1-°C-Buckets, Polymarket-gespiegelt).

HYPOTHESE (These C, NICHT verifiziert): In dünnen Wetter-Märkten hinkt der Preis
dem tatsächlichen Stationshoch nach — steigt das gemessene Tages-Hoch in einen
höheren Bucket, braucht der Markt evtl. Minuten, bis sein Favorit nachzieht. Diese
Zeitreihe misst genau diesen Lag; die eigentliche Edge-Frage (Fenster groß genug?)
wird SPÄTER per SQL geprüft. Kein Echtgeld in diesem Skript.

Vorgeschichte (Session 2026-07-01): statischer Forecast-Edge widerlegt (Open-Meteo
~0,5°C zu kühl, aber Markt preist es ein; Markt scharf). Kernlektion: die Auflösung
ist die SPEZIFISCHE Wunderground-Flughafenstation (Seoul=Incheon RKSI, Shanghai=
Pudong ZSPD, London=City EGLC, Paris=Le Bourget LFPB — NICHT die Haupt-Airports).
Deshalb: Station je Stadt automatisch aus der Polymarket-Auflösungsbeschreibung
ziehen; Ist-Hoch aus ECHTZEIT-METAR (aviationweather.gov, nicht IEM — das lagt live).

Speicher: Centron SQL Server (dbdata), Tabelle bb_WeatherLatency.

Aufruf:
  python weather_latency_logger.py --dry --once
  python weather_latency_logger.py --once
  python weather_latency_logger.py --loop --interval 120
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

try:
    import pymssql
except ImportError:
    pymssql = None

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Hardcodierte Centron-Creds — bewusste Projektentscheidung (siehe CLAUDE.md), Muster aus app.py.
DB_CONFIG = {
    "server": "158.181.48.77",
    "database": "dbdata",
    "user": "326773",
    "password": "Extaler11!",
}

JUP = "https://prediction-market-api.jup.ag/api/v1"
GAMMA = "https://gamma-api.polymarket.com/public-search"
METAR = "https://aviationweather.gov/api/data/metar"

# °C-Städte (US-Fahrenheit-Märkte bewusst erst mal ausgelassen). Stadt -> UTC-Offset (Juli/DST).
# Abdeckung über 24h: Ozeanien -> Asien -> Europa -> (Nacht) Amerika-LatAm.
CITIES = {
    "Wellington": 12,
    "Tokyo": 9, "Seoul": 9, "Shanghai": 8, "Hong Kong": 8, "Beijing": 8,
    "Kuala Lumpur": 8, "Shenzhen": 8, "Chengdu": 8, "Karachi": 5,
    "Jeddah": 3, "Ankara": 3, "Helsinki": 3,
    "London": 1, "Paris": 2, "Madrid": 2, "Milan": 2, "Munich": 2,
    "Amsterdam": 2, "Warsaw": 2, "Cape Town": 2,
    "Mexico City": -6, "Buenos Aires": -3,
}

# Bestätigte Auflösungs-Stationen (Fallback, falls Polymarket-Lookup mal hakt). Session 2026-07-01 verifiziert.
STATION_FALLBACK = {
    "Wellington": "NZWN", "London": "EGLC", "Paris": "LFPB", "Madrid": "LEMD",
    "Seoul": "RKSI", "Shanghai": "ZSPD",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("weather_latency")

UA = {"User-Agent": "Mozilla/5.0"}


# ---------------------------------------------------------------- DB

def get_conn():
    if pymssql is None:
        raise RuntimeError("pymssql nicht installiert (pip install pymssql)")
    return pymssql.connect(**DB_CONFIG, autocommit=True)


DDL = [
    """
    IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name='bb_WeatherLatency')
    CREATE TABLE bb_WeatherLatency (
        id               BIGINT IDENTITY(1,1) PRIMARY KEY,
        city             NVARCHAR(32)  NOT NULL,
        station          NVARCHAR(8)   NULL,        -- Auflösungs-Station (ICAO)
        market_date      NVARCHAR(16)  NULL,        -- z.B. 'July 1'
        event_id         NVARCHAR(64)  NULL,
        ts_utc           DATETIME      NOT NULL,     -- Messzeitpunkt
        local_time       NVARCHAR(20)  NULL,        -- lokale Stadtzeit
        obs_max          FLOAT         NULL,         -- gemessenes Tages-Hoch bisher (°C)
        obs_bucket       INT           NULL,         -- round(obs_max)
        last_ob_utc      DATETIME      NULL,         -- Zeit der letzten METAR-Messung
        last_temp        FLOAT         NULL,
        fav_bucket       INT           NULL,         -- aktueller Markt-Favorit-Bucket
        fav_price        FLOAT         NULL,
        obs_bucket_price FLOAT         NULL,         -- Markt-Preis auf dem erreichten Bucket
        all_prices       NVARCHAR(MAX) NULL,         -- JSON bucket->price
        logged_utc       DATETIME      NOT NULL DEFAULT GETUTCDATE()
    )
    """,
    """
    IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name='ix_wlat_city')
    CREATE INDEX ix_wlat_city ON bb_WeatherLatency(city, market_date, ts_utc)
    """,
]


def ensure_tables(conn):
    cur = conn.cursor()
    for stmt in DDL:
        cur.execute(stmt)
    log.info("Tabelle bb_WeatherLatency sichergestellt.")


# ---------------------------------------------------------------- HTTP

def _get(url, params=None, retries=3):
    for a in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", "") or 3 * (a + 1))
                log.warning(f"429 {url} — warte {wait:.0f}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            log.warning(f"GET {url} Fehler ({e}), Versuch {a + 1}")
            time.sleep(2)
    return None


# ---------------------------------------------------------------- Stations-Auflösung (Polymarket)

_station_cache = {}


def resolve_station(city):
    if city in _station_cache:
        return _station_cache[city]
    icao = STATION_FALLBACK.get(city)
    data = _get(GAMMA, {"q": f"highest temperature {city}", "limit_per_type": 5})
    if data:
        for e in data.get("events", []):
            if e.get("title", "").lower().startswith(f"highest temperature in {city.lower()}"):
                ms = e.get("markets", [])
                desc = (ms[0].get("description", "") if ms else "") or e.get("description", "")
                m = re.search(r"wunderground\.com/history/daily/[a-z]{2}/[^/]+/([A-Z]{4})", desc)
                if m:
                    icao = m.group(1)
                break
    _station_cache[city] = icao
    if icao:
        log.info(f"Station {city} -> {icao}")
    else:
        log.warning(f"Station {city} nicht auflösbar (übersprungen)")
    return icao


# ---------------------------------------------------------------- Jupiter-Markt

def all_weather_events():
    evs = []
    for s in range(0, 60, 10):
        page = _get(f"{JUP}/events", {"category": "weather", "start": s, "end": s + 10})
        if not page:
            continue
        evs += page.get("data", [])
    return evs


def market_buckets(events, city, day_str):
    want = f"Highest temperature in {city} on {day_str}?"
    for e in events:
        if e.get("metadata", {}).get("title", "") == want:
            out = []
            for mk in e.get("markets", []):
                if mk.get("status") != "open":
                    continue
                t = mk.get("title", "")
                if "°C" not in t:
                    return [], e.get("eventId")  # °F-Stadt -> überspringen
                m = re.search(r"(-?\d+)", t)
                if not m:
                    continue
                out.append((int(m.group(1)), (mk.get("pricing", {}).get("buyYesPriceUsd") or 0) / 1e6))
            return out, e.get("eventId")
    return [], None


# ---------------------------------------------------------------- METAR (Echtzeit, Batch)

def metar_batch(stations):
    """Ein Request für alle Stationen -> {icao: [(dt_utc, temp), ...]} letzte 20h."""
    ids = ",".join(sorted(set(s for s in stations if s)))
    if not ids:
        return {}
    data = _get(METAR, {"ids": ids, "format": "json", "hours": 20})
    out = {}
    for m in (data or []):
        icao = m.get("icaoId"); ot = m.get("obsTime"); t = m.get("temp")
        if not icao or ot is None or t is None:
            continue
        out.setdefault(icao, []).append((datetime.fromtimestamp(ot, timezone.utc), float(t)))
    return out


def day_max(obs, off):
    """Tages-Hoch (lokaler Kalendertag) + letzte Messung aus METAR-Liste."""
    now_local = datetime.now(timezone.utc) + timedelta(hours=off)
    ld = now_local.date()
    mx = None; last = None
    for dt_utc, t in obs:
        loc = dt_utc + timedelta(hours=off)
        if loc.date() == ld:
            if mx is None or t > mx:
                mx = t
            if last is None or dt_utc > last[0]:
                last = (dt_utc, t)
    return mx, now_local, last


# ---------------------------------------------------------------- Zyklus

def cycle(conn, dry=False):
    ts = datetime.now(timezone.utc)
    stations = {c: resolve_station(c) for c in CITIES}
    metars = metar_batch(stations.values())
    events = all_weather_events()
    rows = 0
    for city, off in CITIES.items():
        icao = stations.get(city)
        if not icao:
            continue
        omax, local, last = day_max(metars.get(icao, []), off)
        day_str = f"{local.strftime('%B')} {local.day}"
        bk, eid = market_buckets(events, city, day_str)
        if not bk:
            continue
        fav_v, fav_p = max(bk, key=lambda r: r[1])
        obk = round(omax) if omax is not None else None
        obk_price = next((p for v, p in bk if v == obk), None)
        prices = {v: round(p, 3) for v, p in bk}
        peak = "PEAK" if 14 <= local.hour <= 17 else ("post" if (local.hour > 17 or local.hour < 5) else "pre")
        log.info(f"{city:12} {local:%H:%M}L[{peak}] {icao} | Hoch {omax if omax is None else round(omax,1)}->{obk}C "
                 f"| Fav {fav_v}C@{fav_p:.2f} | {obk}C@{obk_price if obk_price is not None else '-'}")
        if not dry:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO bb_WeatherLatency (city,station,market_date,event_id,ts_utc,local_time,"
                "obs_max,obs_bucket,last_ob_utc,last_temp,fav_bucket,fav_price,obs_bucket_price,all_prices) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (city, icao, day_str, eid, ts.strftime("%Y-%m-%d %H:%M:%S"),
                 local.strftime("%Y-%m-%d %H:%M"),
                 None if omax is None else float(omax), obk,
                 None if last is None else last[0].strftime("%Y-%m-%d %H:%M:%S"),
                 None if last is None else float(last[1]),
                 fav_v, float(fav_p),
                 None if obk_price is None else float(obk_price),
                 json.dumps(prices)))
            rows += 1
    if not dry:
        log.info(f"Zyklus: {rows} Zeilen -> bb_WeatherLatency")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry", action="store_true", help="nur loggen, nicht in DB schreiben")
    a = ap.parse_args()

    conn = None
    if not a.dry:
        conn = get_conn()
        ensure_tables(conn)

    log.info("Zyklus-Start")
    cycle(conn, dry=a.dry)
    while a.loop:
        time.sleep(a.interval)
        try:
            cycle(conn, dry=a.dry)
        except Exception as e:
            log.error(f"Zyklus-Fehler: {e}")
            if conn is not None:
                try:
                    conn = get_conn()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
