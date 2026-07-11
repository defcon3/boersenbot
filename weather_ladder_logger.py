# -*- coding: utf-8 -*-
"""weather_ladder_logger.py — Taeglicher Snapshot ALLER Jupiter-Wetter-Preisleitern
(Highest/Lowest, alle Grad-Fenster mit YES/NO-Ask) -> Centron bb_WeatherLadders,
plus METAR-Settle-Backfill fuer vergangene Zieltage.

Zweck (Nutzer-Entscheid 11.07.2026, Storage-Wahl Centron): Forward-Messung der
Lay-Klassen-Frage — welche Fensterklassen (Offset zum Favoriten) sind am Markt
systematisch ueber-/unterbepreist? Die 700d-Empirie (weather_error_quantiles.py)
liefert die Ist-Klassenraten relativ zu UNSEREM mu; hier kommen die echten
Marktpreise + Markt-Favoriten dazu, damit auch markt-fav-relative Klassen und
realisierte Lay-PnL je Klasse auswertbar werden (Ziel N≈1500 Fenster in 2 Wochen).

Eine Zeile je (Snapshot, Zieltag, var, Stadt, Fenster). mu_ens/sigma_ens aus der
700d-Kalibrierung (max: weather_source_calib_*.csv ohne _min_, min: _min_-CSV);
Staedte ohne Station/Kalibrierung werden trotzdem geloggt (mu NULL — Preise sind
auch ohne Modell wertvoll). Settle: IEM-METAR-Extrem des lokalen Kalendertags,
half-up gerundet, in ALLE Zeilen des Zieltags (settle_k, settle_result je Fenster).

Aufruf (Default: Snapshot + Settle-Backfill, fuer den taeglichen Timer):
  python weather_ladder_logger.py
  python weather_ladder_logger.py --create-table     # einmalig
  python weather_ladder_logger.py --settle-only      # nur Backfill
  python weather_ladder_logger.py --snapshot-only
VPS: deploy/boersenbot_weather_ladder.{service,timer} (taeglich 12:30 UTC).
"""
import argparse
import csv
import glob
import math
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone

import airportsdata
import pymssql
import requests

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

API = "https://api.jup.ag/prediction/v1"
OM = "https://api.open-meteo.com/v1/forecast"
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
CALIB_GLOB = "preregs/weather_source_calib_*.csv"

MODELS = ["gfs_seamless", "icon_seamless", "ukmo_seamless", "jma_seamless", "ecmwf_ifs025"]

STATIONS = {
    "Wellington": "NZWN", "Tokyo": "RJTT", "Seoul": "RKSI", "Shanghai": "ZSPD",
    "Beijing": "ZBAA", "Kuala Lumpur": "WMKK", "Shenzhen": "ZGSZ", "Chengdu": "ZUUU",
    "Karachi": "OPKC", "Jeddah": "OEJN", "Ankara": "LTAC", "Helsinki": "EFHK",
    "London": "EGLC", "Paris": "LFPB", "Madrid": "LEMD", "Milan": "LIMC",
    "Munich": "EDDM", "Amsterdam": "EHAM", "Warsaw": "EPWA", "Cape Town": "FACT",
    "Mexico City": "MMMX", "Buenos Aires": "SAEZ",
    "Sao Paulo": "SBGR", "Taipei": "RCSS", "Tel Aviv": "LLBG",
    "Toronto": "CYYZ", "Wuhan": "ZHHH", "Panama City": "MPMG",
}

MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}
TITLE_RE = re.compile(r"^(Highest|Lowest) temperature in (.+) on ([A-Z][a-z]+) (\d{1,2})\?")

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"
AP = airportsdata.load("ICAO")


def half_up(x):
    return math.floor(x + 0.5)


def load_calib():
    calib = {"max": {}, "min": {}}
    for path in sorted(glob.glob(CALIB_GLOB)):
        var = "min" if "_min_" in path else "max"
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                calib[var][(row["city"], row["model"])] = (float(row["bias"]), float(row["sigma"]))
    return calib


def fetch_events():
    events = []
    for s in range(0, 200, 10):
        page = None
        for attempt in range(4):
            r = S.get(f"{API}/events", params={"category": "weather", "start": s, "end": s + 10}, timeout=30)
            if r.status_code == 429:
                time.sleep(6 * (attempt + 1))
                continue
            r.raise_for_status()
            page = r.json().get("data", [])
            break
        if page is None:
            print(f"  Jupiter page {s}: aufgegeben (429)")
            break
        events += page
        if len(page) < 10:
            break
        time.sleep(1.5)
    return events


def title_target_date(month_name, day, today):
    m = MONTHS.get(month_name)
    if not m:
        return None
    year = today.year
    # Jahreswechsel: "January 2" im Dezember gelistet -> naechstes Jahr
    if m < today.month - 6:
        year += 1
    elif m > today.month + 6:
        year -= 1
    return date(year, m, int(day))


def forecast_mu(icao, var, target_day, calib_city):
    """roh-ENS des Zieltags (lokale Zeit) aus 5-Modell-Forecast -> mu/sigma via
    Kalibrierung. None wenn <3 Modelle oder keine ENS-Kalibrierung."""
    st = AP.get(icao)
    if not st:
        return None, None
    try:
        r = S.get(OM, params={
            "latitude": st["lat"], "longitude": st["lon"], "hourly": "temperature_2m",
            "models": ",".join(MODELS), "timezone": "auto", "forecast_days": 7,
        }, timeout=30)
        r.raise_for_status()
        j = r.json()
    except Exception as ex:
        print(f"    Open-Meteo-Fehler {icao}: {ex}")
        return None, None
    hh = j.get("hourly", {})
    times = hh.get("time", [])
    agg = max if var == "max" else min
    prefix = target_day.isoformat()
    raw = []
    for mdl in MODELS:
        vals = [v for t, v in zip(times, hh.get(f"temperature_2m_{mdl}", []))
                if v is not None and t.startswith(prefix)]
        if vals:
            raw.append(agg(vals))
    if len(raw) < 3 or not calib_city:
        return None, None
    ens_raw = sum(raw) / len(raw)
    bias, sigma = calib_city
    return ens_raw - bias, sigma


def snapshot(conn):
    now = datetime.now(timezone.utc)
    today = now.date()
    calib = load_calib()
    events = fetch_events()
    print(f"{len(events)} Weather-Events geladen ({now.strftime('%Y-%m-%d %H:%M UTC')})")

    mu_cache = {}
    rows = []
    for e in events:
        t = e.get("metadata", {}).get("title", "")
        m = TITLE_RE.match(t)
        if not m:
            continue
        var = "max" if m.group(1) == "Highest" else "min"
        city = m.group(2)
        target = title_target_date(m.group(3), m.group(4), today)
        if not target:
            continue
        mks = []
        celsius = True
        for mk in e.get("markets", []):
            ti = mk.get("title", "")
            if "°C" not in ti:
                celsius = False
                break
            num = re.search(r"(-?\d+)", ti)
            if not num:
                continue
            kind = "le" if "below" in ti else ("ge" if "higher" in ti else "eq")
            pr = mk.get("pricing") or {}
            mks.append({
                "kind": kind, "k": int(num.group(1)), "marketId": mk.get("marketId"),
                "status": mk.get("status"),
                "buyYes": (pr.get("buyYesPriceUsd") or 0) / 1e6,
                "buyNo": (pr.get("buyNoPriceUsd") or 0) / 1e6,
            })
        if not celsius or not mks:
            continue
        icao = STATIONS.get(city)
        key = (city, var, target)
        if key not in mu_cache:
            calib_city = calib[var].get((city, "ensemble_mean")) if icao else None
            mu_cache[key] = forecast_mu(icao, var, target, calib_city) if icao else (None, None)
            time.sleep(0.4)
        mu_ens, sig_ens = mu_cache[key]
        k0 = half_up(mu_ens) if mu_ens is not None else None
        eq_open = [x for x in mks if x["kind"] == "eq" and x["status"] == "open" and x["buyYes"] > 0]
        market_fav_k = max(eq_open, key=lambda x: x["buyYes"])["k"] if eq_open else None
        for x in mks:
            rows.append((
                now, target, var, city, icao, x["marketId"], x["kind"], x["k"],
                x["buyYes"], x["buyNo"], x["status"],
                mu_ens, sig_ens,
                (x["k"] - k0) if (k0 is not None and x["kind"] == "eq") else None,
                market_fav_k,
            ))

    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO bb_WeatherLadders (snapshot_utc, target_date, var, city, icao, market_id, "
        "kind, k, buy_yes, buy_no, status, mu_ens, sigma_ens, offset_fav, market_fav_k) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)
    conn.commit()
    n_cities = len({(r[3], r[2], r[1]) for r in rows})
    print(f"Snapshot: {len(rows)} Fenster-Zeilen ({n_cities} Stadt/var/Zieltag-Leitern) eingefuegt.")


def settle(conn):
    """METAR-Ist fuer vergangene Zieltage nachtragen (lokaler Kalendertag komplett)."""
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT target_date, var, city, icao FROM bb_WeatherLadders "
                "WHERE settle_k IS NULL AND icao IS NOT NULL")
    todo = cur.fetchall()
    n_done = 0
    for target, var, city, icao in todo:
        if isinstance(target, datetime):
            target = target.date()
        st = AP.get(icao)
        tz_name = st.get("tz") if st else None
        if not tz_name:
            continue
        # lokaler Tag muss vorbei sein: konservativ erst ab Folgetag 12:00 UTC settlen
        if datetime.now(timezone.utc) < datetime(target.year, target.month, target.day,
                                                 tzinfo=timezone.utc) + timedelta(days=1, hours=12):
            continue
        try:
            r = S.get(IEM, params={
                "station": icao, "data": "tmpc",
                "year1": target.year, "month1": target.month, "day1": target.day,
                "year2": target.year, "month2": target.month, "day2": target.day,
                "tz": tz_name, "format": "onlycomma", "latlon": "no", "elev": "no",
                "missing": "M", "trace": "T", "direct": "no", "report_type": 3,
            }, timeout=60)
            r.raise_for_status()
        except Exception as ex:
            print(f"    IEM-Fehler {icao} {target}: {ex}")
            continue
        temps = []
        for line in r.text.splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 3 or parts[2] == "M":
                continue
            try:
                temps.append(float(parts[2]))
            except ValueError:
                continue
        if len(temps) < 12:      # zu wenige Obs -> lieber offen lassen
            continue
        settle_k = half_up(min(temps) if var == "min" else max(temps))
        cur.execute(
            "UPDATE bb_WeatherLadders SET settle_k=%s, "
            "settle_result=CASE WHEN (kind='eq' AND k=%s) OR (kind='le' AND %s<=k) "
            "OR (kind='ge' AND %s>=k) THEN 1 ELSE 0 END "
            "WHERE target_date=%s AND var=%s AND city=%s AND settle_k IS NULL",
            (settle_k, settle_k, settle_k, settle_k, target, var, city))
        n_done += 1
        time.sleep(0.5)
    conn.commit()
    print(f"Settle-Backfill: {n_done} Stadt/var/Zieltage aufgeloest ({len(todo)} offen gewesen).")


def create_table(conn):
    cur = conn.cursor()
    cur.execute("""
    IF OBJECT_ID('bb_WeatherLadders', 'U') IS NULL
    CREATE TABLE bb_WeatherLadders (
        id INT IDENTITY PRIMARY KEY,
        snapshot_utc DATETIME2 NOT NULL,
        target_date DATE NOT NULL,
        var CHAR(3) NOT NULL,
        city NVARCHAR(50) NOT NULL,
        icao CHAR(4) NULL,
        market_id NVARCHAR(30) NULL,
        kind CHAR(2) NOT NULL,
        k INT NOT NULL,
        buy_yes FLOAT NULL,
        buy_no FLOAT NULL,
        status NVARCHAR(20) NULL,
        mu_ens FLOAT NULL,
        sigma_ens FLOAT NULL,
        offset_fav INT NULL,
        market_fav_k INT NULL,
        settle_k INT NULL,
        settle_result BIT NULL
    )""")
    cur.execute("""
    IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name='ix_bb_WeatherLadders_target')
    CREATE INDEX ix_bb_WeatherLadders_target ON bb_WeatherLadders (target_date, var, city)""")
    conn.commit()
    print("Tabelle bb_WeatherLadders bereit.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--create-table", action="store_true")
    ap.add_argument("--snapshot-only", action="store_true")
    ap.add_argument("--settle-only", action="store_true")
    args = ap.parse_args()
    conn = pymssql.connect(**DB_CONFIG)
    if args.create_table:
        create_table(conn)
        return
    if not args.settle_only:
        snapshot(conn)
    if not args.snapshot_only:
        settle(conn)
    conn.close()


if __name__ == "__main__":
    main()
