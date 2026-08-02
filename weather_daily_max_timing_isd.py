#!/usr/bin/env python3
"""
weather_daily_max_timing_isd.py — Basisrate "das Tageshoch kommt noch", breit.

FRAGE: Wie oft ist das Tagesmaximum zur Ortsstunde T noch NICHT gefallen? Das ist
Schritt 0 des Waechters (BACKLOG.md) und zugleich die Grundlage der Regel "vor
16:20 Ortszeit keine Lay-Position erhoehen".

STAND BISHER: 13:20 91 % | 14:20 87 % | 15:20 76 % | 16:20 41 % | 17:20 12 %
— gemessen an 125 Stadt-Tagen, fuenf europaeische Staedte, EIN Sommer. Der
Vorbehalt stand im Eintrag; hier wird er zur Messung.

QUELLE: NCEI Integrated Surface Database, global-hourly. Weltweite METAR-
Einzelmeldungen, ohne Token, ohne Registrierung:
  ncei.noaa.gov/access/services/data/v1?dataset=global-hourly&stations=<ISD-ID>
Verifiziert am 01.08.2026 an KMIA und EGLL.

FUENF FALLEN, alle im Code behandelt:
 1. ISD liefert UTC. Die Bretter loesen auf LOKALE Kalendertage auf, und die
    Prueffstunde ist Ortszeit — ohne Umrechnung misst man Unsinn. tz kommt aus
    weather_stations.station_info.
 2. Report-Typen mischen: FM-15 ist METAR, FM-12 SYNOP, FM-16 SPECI. Nur FM-15
    und FM-16 gehoeren zur Settlement-Familie; SYNOP hat andere Messzeiten und
    verzerrt die Stundenverteilung.
 3. TMP steht in ZEHNTEL-Grad mit Qualitaetsflag: "+0300,5" = 30,0 Grad.
    +9999 = fehlend. Flags 2,3,6,7 sind beanstandet und fliegen raus.
 4. Gerundet, nicht roh. Der Waechter arbeitet auf Buckets — "das Hoch kommt
    noch" heisst, das GERUNDETE Maximum steigt noch. Auf Rohwerten faellt die
    Rate zu hoch aus, weil jedes Zehntel zaehlt.
 5. Duenne Tage. Ein Tag mit vier Meldungen sieht aus, als sei das Hoch frueh
    gefallen. Mindestens MIN_OBS Meldungen und Abdeckung ueber die Nachmittags-
    stunden, sonst faellt der Tag raus.

Aufruf:
  python weather_daily_max_timing_isd.py                    # Sommer, alle Staedte
  python weather_daily_max_timing_isd.py --jahre 2023 2024 2025
  python weather_daily_max_timing_isd.py --saison ganzjahr
  python weather_daily_max_timing_isd.py --city London,Paris,Madrid
"""

import argparse
import csv
import io
import sys
from collections import defaultdict
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from weather_stations import station_info

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ISD_DATA = "https://www.ncei.noaa.gov/access/services/data/v1"
ISD_HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

# Prueffstunden wie im Backlog: :20 nach der vollen Stunde, Ortszeit.
STUNDEN = [(13, 20), (14, 20), (15, 20), (16, 20), (17, 20), (18, 20)]
MIN_OBS = 12          # Meldungen je Stadt-Tag
NACHMITTAG = (12, 19)  # Stunden, die belegt sein muessen (Ortszeit)

# Die Staedte des Screens. ICAO -> ISD-ID wird ueber isd-history.csv aufgeloest.
STAEDTE = {
    "London": "EGLC", "Paris": "LFPB", "Madrid": "LEMD", "Munich": "EDDM",
    "Milan": "LIML", "Amsterdam": "EHAM", "Warsaw": "EPWA", "Helsinki": "EFHK",
    "Moscow": "UUWW", "Ankara": "LTAB", "Tel Aviv": "LLBG", "Jeddah": "OEJN",
    "Seoul": "RKSI", "Tokyo": "RJAA", "Beijing": "ZBAA", "Shanghai": "ZSPD",
    "Chengdu": "ZUUU", "Wuhan": "ZHHH", "Taipei": "RCTP", "Kuala Lumpur": "WMKK",
    "Toronto": "CYYZ", "Mexico City": "MMMX", "Panama City": "MPTO",
    "Sao Paulo": "SBSP", "Buenos Aires": "SABE", "Cape Town": "FACT",
    "Wellington": "NZWN",
}

S = requests.Session()
S.headers["User-Agent"] = "boersenbot-research/1.0"


def isd_ids(icaos):
    """ICAO -> ISD-Kennung (USAF-WBAN). Die Stationsliste ist gross, aber
    einmalig; ohne sie akzeptiert der Datenendpunkt keine ICAO-Codes."""
    print("Lade ISD-Stationsliste ...", flush=True)
    r = S.get(ISD_HISTORY, timeout=120)
    r.raise_for_status()
    treffer = {}
    for row in csv.DictReader(io.StringIO(r.text)):
        ic = (row.get("ICAO") or "").strip().upper()
        if ic not in icaos or ic in treffer:
            continue
        usaf, wban = row.get("USAF", "").strip(), row.get("WBAN", "").strip()
        if usaf and wban and usaf != "999999":
            treffer[ic] = f"{usaf}{wban}"
    return treffer


def hole(isd_id, von, bis):
    """Stundenmeldungen einer Station. Gibt [(utc_ts, temp_c)] zurueck."""
    r = S.get(ISD_DATA, params={
        "dataset": "global-hourly", "stations": isd_id,
        "startDate": von, "endDate": bis,
        "dataTypes": "TMP,REPORT_TYPE", "format": "csv",
    }, timeout=300)
    r.raise_for_status()
    out = []
    for row in csv.DictReader(io.StringIO(r.text)):
        typ = (row.get("REPORT_TYPE") or "").strip()
        if typ not in ("FM-15", "FM-16"):   # Falle 2: kein SYNOP
            continue
        tmp = (row.get("TMP") or "").strip()
        if "," not in tmp:
            continue
        wert, flag = tmp.split(",", 1)
        if flag.strip() in ("2", "3", "6", "7"):   # Falle 3: beanstandet
            continue
        try:
            zehntel = int(wert)
        except ValueError:
            continue
        if zehntel == 9999:
            continue
        try:
            ts = datetime.strptime(row["DATE"], "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        out.append((ts, zehntel / 10.0))
    return out


def tage_bilden(obs, tz_name):
    """Nach LOKALEM Kalendertag gruppieren (Falle 1) und duenne Tage verwerfen
    (Falle 5)."""
    tz = ZoneInfo(tz_name)
    je_tag = defaultdict(list)
    for ts, t in obs:
        lok = ts.astimezone(tz)
        je_tag[lok.date()].append((lok, t))
    gut = {}
    for tag, werte in je_tag.items():
        if len(werte) < MIN_OBS:
            continue
        stunden = {w[0].hour for w in werte}
        if not all(h in stunden for h in range(*NACHMITTAG)):
            continue
        gut[tag] = sorted(werte)
    return gut


def basisrate(tage):
    """Je Prueffstunde: Anteil der Tage, an denen das GERUNDETE Tagesmaximum
    danach noch steigt (Falle 4)."""
    zaehler = {s: [0, 0] for s in STUNDEN}
    for tag, werte in tage.items():
        max_ganz = round(max(t for _, t in werte))
        for (hh, mm) in STUNDEN:
            bis_jetzt = [t for lok, t in werte
                         if (lok.hour, lok.minute) <= (hh, mm)]
            if not bis_jetzt:
                continue
            zaehler[(hh, mm)][1] += 1
            if round(max(bis_jetzt)) < max_ganz:
                zaehler[(hh, mm)][0] += 1
    return zaehler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jahre", nargs="+", type=int, default=[2024, 2025])
    ap.add_argument("--saison", choices=["sommer", "ganzjahr"], default="sommer",
                    help="sommer = Juni-August (Nordhalbkugel-Vergleich zur "
                         "125-Tage-Messung)")
    ap.add_argument("--city", default=None, help="Kommagetrennte Teilmenge")
    a = ap.parse_args()

    staedte = dict(STAEDTE)
    if a.city:
        wunsch = {c.strip() for c in a.city.split(",")}
        staedte = {k: v for k, v in staedte.items() if k in wunsch}

    ids = isd_ids(set(staedte.values()))
    fehlend = [c for c, ic in staedte.items() if ic not in ids]
    print(f"ISD-Kennung gefunden fuer {len(ids)} von {len(staedte)} Stationen"
          + (f" — ohne: {', '.join(fehlend)}" if fehlend else ""))

    gesamt = {s: [0, 0] for s in STUNDEN}
    je_stadt = {}
    for stadt, icao in sorted(staedte.items()):
        if icao not in ids:
            continue
        st = station_info(icao) or {}
        tz_name = st.get("tz")
        if not tz_name:
            print(f"  {stadt}: keine Zeitzone, uebersprungen")
            continue
        alle_tage = {}
        for jahr in a.jahre:
            von, bis = ((f"{jahr}-06-01", f"{jahr}-08-31") if a.saison == "sommer"
                        else (f"{jahr}-01-01", f"{jahr}-12-31"))
            try:
                obs = hole(ids[icao], von, bis)
            except Exception as ex:
                print(f"  {stadt} {jahr}: Abruf fehlgeschlagen ({str(ex)[:60]})")
                continue
            alle_tage.update(tage_bilden(obs, tz_name))
        if not alle_tage:
            print(f"  {stadt}: keine verwertbaren Tage")
            continue
        z = basisrate(alle_tage)
        je_stadt[stadt] = (len(alle_tage), z)
        for s in STUNDEN:
            gesamt[s][0] += z[s][0]
            gesamt[s][1] += z[s][1]
        print(f"  {stadt:<14} {len(alle_tage):4d} Tage", flush=True)

    print("\n" + "=" * 66)
    print(f"BASISRATE 'das Tageshoch kommt noch'   Saison: {a.saison}, "
          f"Jahre {a.jahre}")
    print("=" * 66)
    print(f"{'Ortszeit':<10} {'Rate':>7}   {'n':>6}   Vergleich: 125 Stadt-Tage, "
          f"5 Staedte, 1 Sommer")
    alt = {(13, 20): 91, (14, 20): 87, (15, 20): 76, (16, 20): 41, (17, 20): 12}
    for s in STUNDEN:
        noch, n = gesamt[s]
        if not n:
            continue
        rate = 100 * noch / n
        ref = alt.get(s)
        vgl = f"   bisher {ref} %  ({rate - ref:+.0f} pp)" if ref is not None else ""
        print(f"{s[0]:02d}:{s[1]:02d}      {rate:6.1f} %   {n:6d}{vgl}")

    print(f"\nStaedte: {len(je_stadt)} | Stadt-Tage gesamt: "
          f"{sum(v[0] for v in je_stadt.values())}")
    print("\nJE STADT (Rate bei 16:20, die Schwelle der Regel)")
    reihen = []
    for stadt, (n, z) in je_stadt.items():
        noch, ges = z[(16, 20)]
        if ges:
            reihen.append((100 * noch / ges, stadt, ges))
    for rate, stadt, n in sorted(reihen, reverse=True):
        print(f"  {stadt:<14} {rate:5.1f} %   ({n} Tage)")


if __name__ == "__main__":
    main()
