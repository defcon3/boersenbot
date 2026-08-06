#!/usr/bin/env python3
"""
weather_wu_forecast_logger.py — Wundergrounds EIGENE Prognose mitschreiben.

ANLASS (06.08.2026, Frage des Betreibers): "Ich wollte eigentlich wissen, ob bei
den Quellen fuer die Prognose auch Wunderground selbst mit dabei ist."

Antwort war nein — und das ist eine Luecke. Das Modell (mu_ens) besteht aus
fuenf NWP-Laeufen ueber Open-Meteo (GFS/ICON/UKMO/JMA/ECMWF). Wunderground kam
im ganzen Repo bisher NUR als Beobachtungsquelle vor (observations/historical,
pws/history) — also fuer Settlement, nie fuer die Vorhersage.

WARUM DAS ZAEHLT: Der Markt settelt gegen Wunderground; der Markttext verlinkt
wunderground.com/history/daily/... Wer diese Bretter handelt, schaut also mit
hoher Wahrscheinlichkeit auf WUs eigene Vorhersage — und die ist GANZZAHLIG,
zeigt also direkt auf einen Bucket. Zudem ist sie auf genau die Station
kalibriert, gegen die abgerechnet wird.

Das ist NICHT die am 04.08. abgesagte Fremdquelle. Damals wurde die Schranke
zwischen NWP-Modellen gemessen (~0,3 K nach Kalibrierung, alle rechnen dieselbe
Physik auf demselben Gitter). WU ist eine kommerzielle Prognose mit eigener
Nachbearbeitung — und sie kostet nichts (derselbe Web-Key wie im Ladder-Logger).

ERSTMESSUNG 06.08. fuer Zieltag 07.08., 16 Staedte: WU deckt sich zu 31 % mit
dem Markt-Favoriten und zu 38 % mit unserem. In zwei Dritteln der Staedte sagt
WU etwas anderes als beide — eine echte dritte Meinung. Auffaellig: Tel Aviv
WU 32, Markt 32, unser Modell 33; genau die Stadt, deren Favorit gemessen
1,11 Bucket zu hoch liegt (weather-stadt-verschiebung-telaviv).

WAS DIESER LOGGER TUT: taeglich je Stadt die WU-Tagesextreme fuer Lead 1 und
Lead 2 nach bb_WeatherWuForecast schreiben. Mehr nicht. Er greift in nichts ein
und wird bewusst NICHT in weather_ladder_logger.py eingebaut — der ist der
kritische Pfad fuer Autobuy und Settlement und wird fuer ein Experiment nicht
angefasst.

LEAD ist relativ zum LOKALEN Kalendertag der Stadt, nicht zu UTC. Um 12:20 UTC
ist in Tokio bereits der Folgetag; ein Lead ueber UTC gerechnet waere fuer Asien
systematisch um einen Tag verschoben. WU liefert validTimeLocal mit — genau
dagegen wird gematcht.

Auswertung fruehestens nach 3-4 Wochen, dann gegen bb_WeatherLadders:
  1. Liegt WU naeher am Settlement als mu_ens? (Bucket-MAE, gepaart je Stadt-Tag)
  2. Ist die Abweichung JE STADT konstant? (die Ausgangsfrage: "in Tokio immer
     einen drunter")
  3. Wenn WU und mu_ens auseinanderlaufen — wer behaelt recht, und zahlt der
     Markt dafuer?

Aufruf:  python weather_wu_forecast_logger.py [--dry-run] [--leads 1 2]
"""
import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pymssql
import requests

sys.path.insert(0, "/home/veit/boersenbot")
sys.path.insert(0, ".")
from weather_stations import station_info  # noqa: E402

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
WU_FC = "https://api.weather.com/v3/wx/forecast/daily/5day"
WU_KEY = "e1f10a1e78da46f5b10a1e78da96f525"   # oeffentlicher Web-Key, wie im Ladder-Logger
TABELLE = "bb_WeatherWuForecast"


def tabelle_anlegen(conn):
    """Eigene Tabelle. Der Platz ist unkritisch: ~30 Staedte x 2 Leads x 1 Zeile
    am Tag sind rund 2 MB im Jahr — die DB hat ein 400-MB-Hartlimit, das am
    05.08. schon einmal den Handel stillgelegt hat, deshalb der Hinweis."""
    cur = conn.cursor()
    cur.execute(f"""
    IF OBJECT_ID('{TABELLE}', 'U') IS NULL
    CREATE TABLE {TABELLE} (
        id           INT IDENTITY(1,1) PRIMARY KEY,
        fetched_utc  DATETIME     NOT NULL,
        target_date  DATE         NOT NULL,
        lead         TINYINT      NOT NULL,
        city         NVARCHAR(64) NOT NULL,
        icao         CHAR(4)      NULL,
        wu_max       INT          NULL,
        wu_min       INT          NULL,
        CONSTRAINT UQ_wufc UNIQUE (target_date, lead, city)
    )""")
    conn.commit()


def staedte(conn):
    """Aktive Bretter der letzten 7 Tage — waechst automatisch mit neuen Staedten.

    Bewusst aus der DB statt aus einer festen Liste: kommt ein Brett dazu, wird
    es ohne Codeaenderung mitgeloggt. Staedte ohne ICAO (die acht stationslosen,
    siehe MU_PENDING) fallen raus — ohne Station keine Koordinate.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT city, icao FROM bb_WeatherLadders "
        "WHERE icao IS NOT NULL AND snapshot_utc >= %s",
        (datetime.now(timezone.utc) - timedelta(days=7),))
    return [(c, (i or "").strip()) for c, i in cur.fetchall()]


def wu_forecast(lat, lon, tries=3):
    """5-Tage-Prognose. Gibt {lokales Datum: (max, min)} zurueck."""
    for versuch, pause in ((1, 4), (2, 12), (3, 0)):
        try:
            r = requests.get(WU_FC, params={
                "geocode": f"{lat:.4f},{lon:.4f}", "format": "json",
                "units": "m", "language": "en-US", "apiKey": WU_KEY}, timeout=25)
            if r.status_code == 429:
                time.sleep(pause)
                continue
            r.raise_for_status()
            j = r.json()
            break
        except Exception as e:
            if not pause:
                raise
            time.sleep(pause)
    else:
        return {}
    tage = j.get("validTimeLocal") or []
    mx = j.get("calendarDayTemperatureMax") or []
    mn = j.get("calendarDayTemperatureMin") or []
    out = {}
    for i, t in enumerate(tage):
        tag = t[:10]
        out[tag] = (mx[i] if i < len(mx) else None,
                    mn[i] if i < len(mn) else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="nichts schreiben")
    ap.add_argument("--leads", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()

    run = datetime.now(timezone.utc)
    print(f"WU-Forecast-Logger {run:%Y-%m-%d %H:%M} UTC, Leads {args.leads}")
    conn = pymssql.connect(**DB_CONFIG)
    if not args.dry_run:
        tabelle_anlegen(conn)
    orte = staedte(conn)
    print(f"{len(orte)} Staedte aus bb_WeatherLadders (letzte 7 Tage).\n")

    cur = conn.cursor()
    n_ok = n_fehl = 0
    for city, icao in sorted(orte):
        st = station_info(icao)
        if not st or st.get("lat") is None:
            print(f"  {city:16s} keine Koordinate zu '{icao}' — uebersprungen")
            n_fehl += 1
            continue
        try:
            fc = wu_forecast(st["lat"], st["lon"])
        except Exception as e:
            print(f"  {city:16s} WU-Fehler: {type(e).__name__} {e}")
            n_fehl += 1
            continue
        if not fc:
            print(f"  {city:16s} keine Prognose erhalten")
            n_fehl += 1
            continue

        # Lokales Heute der STADT — nicht UTC. In Tokio ist um 12:20 UTC schon
        # der Folgetag; ueber UTC gerechnet waere jeder Lead in Asien um einen
        # Tag verschoben.
        tz = ZoneInfo(st["tz"]) if st.get("tz") else timezone.utc
        heute_lokal = datetime.now(tz).date()

        teile = []
        for lead in args.leads:
            ziel = heute_lokal + timedelta(days=lead)
            werte = fc.get(ziel.isoformat())
            if not werte or werte[0] is None:
                teile.append(f"+{lead}: —")
                continue
            wmax, wmin = werte
            teile.append(f"+{lead}: {wmax}/{wmin}")
            if args.dry_run:
                continue
            # Idempotent: ein erneuter Lauf am selben Tag aktualisiert, statt
            # eine zweite Zeile zu erzeugen (UNIQUE auf target_date+lead+city).
            cur.execute(
                f"UPDATE {TABELLE} SET fetched_utc=%s, wu_max=%s, wu_min=%s, icao=%s "
                f"WHERE target_date=%s AND lead=%s AND city=%s",
                (run, wmax, wmin, icao, ziel, lead, city))
            if cur.rowcount == 0:
                cur.execute(
                    f"INSERT INTO {TABELLE} "
                    f"(fetched_utc, target_date, lead, city, icao, wu_max, wu_min) "
                    f"VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (run, ziel, lead, city, icao, wmax, wmin))
            n_ok += 1
        print(f"  {city:16s} {'  '.join(teile)}")
        time.sleep(0.3)

    if not args.dry_run:
        conn.commit()
    print(f"\n{n_ok} Zeilen geschrieben, {n_fehl} Staedte ohne Wert."
          + ("  (DRY RUN — nichts gespeichert)" if args.dry_run else ""))
    # Laut werden, wenn die Quelle wegbricht: ein stiller Ausfall ueber Wochen
    # macht das Fenster wertlos. Vgl. den Seoul-Fall vom 03.08.
    if n_fehl > len(orte) // 3:
        print("WARNUNG: mehr als ein Drittel der Staedte ohne Prognose.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
