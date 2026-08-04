# -*- coding: utf-8 -*-
"""weather_source_latency_probe.py — Wer sieht eine neue Beobachtung zuerst?

## Die Frage

Fuer ein Intraday-Geschaeft an Wetterbrettern zaehlt nicht die Prognose, sondern
wie schnell man erfaehrt, dass eine Station eine neue Temperatur gemeldet hat —
ein Tagesmaximum ist monoton, steht es erst bei 25, ist der 24er-Bucket tot.
Gemessen wird deshalb **nicht**, wie alt eine Beobachtung ist, sondern **wann
sie in welcher Quelle erstmals sichtbar wird**.

Verglichen werden drei kostenlose Quellen:
  * **WU-Tabelle** (`api.weather.com/.../observations/historical.json`) — die
    Quelle, auf die unsere Nicht-US-Maerkte settlen und die wir heute pollen.
  * **NOAA** (`aviationweather.gov/api/data/metar`) — METAR-Rohdaten. Fuer die
    US-Fahrenheit-Bretter ist NWS/ASOS die *settelnde* Quelle, nicht nur eine
    Naeherung.
  * **IEM** (`mesonet.agron.iastate.edu`) — das Archiv, das die Pipeline schon nutzt.

Ein kostenpflichtiger Dienst (metar.ws, 49 EUR/Monat) lohnt nur, wenn er
schneller ist als die beste freie Quelle. Diese Messung liefert die Basislinie,
gegen die er sich beweisen muesste.

## Methode

Alle `--interval` Sekunden werden alle drei Quellen abgefragt. Fuer jede Station
wird die neueste **valid_time (UTC)** gelesen; taucht eine bisher ungesehene
valid_time auf, wird der Sichtungszeitpunkt festgehalten. Die Differenz der
Sichtungszeiten ist die gesuchte Latenz.

**Zeitzonen-Falle (teuer gelernt):** Verglichen wird ausschliesslich ueber
`valid_time_gmt` bzw. `obsTime` in UTC — niemals ueber lokale Anzeigezeiten.

Aufruf:
    python weather_source_latency_probe.py --hours 12
    python weather_source_latency_probe.py --hours 1 --interval 20   # Kurztest
"""
import argparse
import csv
import datetime as dt
import os
import sys
import time

import requests

from weather_stations import station_info

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

WU_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
WU_URL = "https://api.weather.com/v1/location/{icao}:9:{cc}/observations/historical.json"
NOAA   = "https://aviationweather.gov/api/data/metar"
IEM    = "https://mesonet.agron.iastate.edu/api/1/currents.json"

# Stationen der handelbaren Bretter: US (NWS settelt = METAR) + zwei Nicht-US
# zur Gegenprobe, wo die WU-Tabelle settelt.
STATIONS = {
    "KLGA": ("NYC",     "US", "NY_ASOS"),
    "KORD": ("Chicago", "US", "IL_ASOS"),
    "KSEA": ("Seattle", "US", "WA_ASOS"),
    "KMIA": ("Miami",   "US", "FL_ASOS"),
    "EGLC": ("London",  "GB", None),
    "LFPB": ("Paris",   "FR", None),
}
OUT = "weather_source_latency.csv"
FIELDS = ["seen_utc", "station", "source", "valid_utc", "temp_c", "lag_s"]

S = requests.Session()
S.headers["User-Agent"] = "boersenbot-latency-probe"


def now():
    return dt.datetime.now(dt.UTC)


def f2c(f):
    return None if f is None else round((float(f) - 32) * 5 / 9, 1)


def poll_wu(icao, cc):
    """(valid_utc, temp_c) der neuesten Beobachtung der WU-Tabelle."""
    try:
        d = S.get(WU_URL.format(icao=icao, cc=cc),
                  params={"apiKey": WU_KEY, "units": "m",
                          "startDate": now().strftime("%Y%m%d")}, timeout=20).json()
    except Exception:
        return None
    obs = [o for o in (d.get("observations") or []) if o.get("temp") is not None]
    if not obs:
        return None
    o = max(obs, key=lambda x: x["valid_time_gmt"])
    return dt.datetime.fromtimestamp(o["valid_time_gmt"], dt.UTC), float(o["temp"])


def poll_noaa(icaos):
    """{icao: (valid_utc, temp_c)} aus den METAR-Rohdaten."""
    out = {}
    try:
        j = S.get(NOAA, params={"ids": ",".join(icaos), "format": "json", "hours": 3},
                  timeout=25).json()
    except Exception:
        return out
    for m in j if isinstance(j, list) else []:
        i = m.get("icaoId"); t = m.get("obsTime")
        if not i or not isinstance(t, (int, float)):
            continue
        ts = dt.datetime.fromtimestamp(t, dt.UTC)
        if i not in out or ts > out[i][0]:
            out[i] = (ts, m.get("temp"))
    return out


def poll_iem(network, want):
    try:
        d = S.get(IEM, params={"network": network}, timeout=25).json().get("data", [])
    except Exception:
        return None
    for x in d:
        if "K" + str(x.get("station")) == want or x.get("station") == want.lstrip("K"):
            v = x.get("utc_valid")
            if not v:
                return None
            ts = dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
            return ts, f2c(x.get("tmpf"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12.0,
                    help="0 = endlos (Dauerbetrieb als Dienst)")
    ap.add_argument("--interval", type=int, default=30, help="Sekunden zwischen Runden")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    new_file = not os.path.exists(args.out)
    fh = open(args.out, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    # --- Neustart-Falle (04.08.2026) ---------------------------------------
    # Als Dienst mit Restart=always startet der Prozess irgendwann neu. Ohne
    # Vorsorge waere das erste Polling nach dem Neustart voller "Erstsichtungen"
    # von Beobachtungen, die laengst standen — mit entsprechend riesigem lag_s.
    # Genau der Median dieser Spalte ist die Messgroesse, also zwei Schutze:
    #   1. bereits geschriebene Schluessel aus der CSV zuruecklesen
    #   2. WARMUP: eine Beobachtung, die schon vor Prozessstart gueltig war,
    #      kann von uns nicht "zuerst gesehen" worden sein -> ueberspringen.
    # (2) kostet die erste halbe Stunde nach jedem Start und ist der Preis
    # dafuer, dass keine erfundene Latenz in die Reihe kommt.
    seen = set()          # (station, source, valid_utc) — nur Erstsichtungen zaehlen
    if not new_file:
        try:
            with open(args.out, encoding="utf-8", newline="") as old:
                for r in csv.DictReader(old):
                    seen.add((r["station"], r["source"], r["valid_utc"]))
        except Exception as ex:
            print(f"WARNUNG: CSV nicht lesbar ({ex}) — starte ohne Vorbelegung")
    start = now()
    print(f"{len(seen)} bekannte Sichtungen vorbelegt; Warmup laeuft "
          f"(Beobachtungen aelter als {start:%H:%M}Z werden verworfen)")

    ende = None if args.hours <= 0 else start + dt.timedelta(hours=args.hours)
    print(f"Messung {'endlos' if ende is None else f'bis {ende:%H:%M}Z'}, "
          f"Intervall {args.interval}s, {len(STATIONS)} Stationen -> {args.out}")
    runde = 0
    while ende is None or now() < ende:
        runde += 1
        t0 = now()
        noaa = poll_noaa(list(STATIONS))
        for icao, (name, cc, net) in STATIONS.items():
            for src, val in (("WU", poll_wu(icao, cc)),
                             ("NOAA", noaa.get(icao)),
                             ("IEM", poll_iem(net, icao) if net else None)):
                if not val or val[0] is None:
                    continue
                valid, temp = val
                key = (icao, src, valid.strftime("%Y-%m-%dT%H:%M:%SZ"))
                if key in seen:
                    continue
                seen.add(key)
                if valid < start:        # Warmup, s. Kommentar in main()
                    continue
                seen_utc = now()
                w.writerow({"seen_utc": seen_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "station": icao, "source": src,
                            "valid_utc": valid.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "temp_c": temp,
                            "lag_s": int((seen_utc - valid).total_seconds())})
                print(f"  [{seen_utc:%H:%M:%S}] {icao} {src:5s} valid {valid:%H:%M}Z "
                      f"temp {temp}  Verzug {int((seen_utc-valid).total_seconds())}s")
        fh.flush()
        if runde % 20 == 0:
            print(f"  ... Runde {runde}, {len(seen)} Erstsichtungen")
        time.sleep(max(1, args.interval - (now() - t0).total_seconds()))
    fh.close()
    print(f"\nfertig — {len(seen)} Erstsichtungen in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
