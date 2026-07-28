# -*- coding: utf-8 -*-
"""weather_hko.py — Ist-Werte der Hong Kong Observatory, EINE Quelle.

Warum eigenes Modul: Hong Kong ist die einzige Stadt im Buch, deren Bretter auf
die HKO aufloesen — weder Wunderground noch METAR fuehren diese Station. Die
Werte werden an zwei sehr verschiedenen Stellen gebraucht:

  * weather_ladder_logger (VPS-Timer)  -> Settle-Backfill, taeglich, muss den
    GESTRIGEN Tag kennen. Darf nichts Schweres importieren (kein numpy/scipy).
  * weather_source_compare             -> Kalibrierung, braucht 700 Tage Historie.

Die HKO bietet dafuer ZWEI Endpunkte, beide ohne Auth, und man braucht beide:

  1. Daily Extract, TAGESAKTUELL (bis gestern):
     hko.gov.hk/cis/dailyExtract/dailyExtract_YYYYMM.xml — liefert trotz der
     Endung JSON. stn.data[0].dayData[i] = [Tag, Druck, Max, Mittel, Min, ...].
     Deckt nur einen Monat je Abruf ab. DAS ist die Reihe, an der Polymarket
     tatsaechlich aufloest: die Juli-Bretter waren laengst gesettelt, als die
     Klimareihe noch im Juni stand.
  2. Klimareihe, MONATSVERZUG:
     opendata.php?dataType=CLMMAXT|CLMMINT&station=HKO — ohne year/month die
     komplette Historie ab 1884 (49k+ Tage), aber der laufende Monat fehlt.
     Ideal fuer die Kalibrierung, unbrauchbar fuer den Settle-Backfill.

actual_daily_extremes() legt beide uebereinander (Daily Extract gewinnt), damit
Kalibrierungen nicht mehr an der Monatsgrenze der Klimareihe enden muessen.

BEWUSST LEICHTGEWICHTIG: nur stdlib + requests, analog weather_stations.

Erinnerung zur Bucket-Semantik: HK misst auf EINE Dezimalstelle (32.3), und die
Bucket-Titel meinen [k, k+1) statt half_up — der Umweg ueber
weather_stations.favorit_k() ist deshalb Pflicht, nie selbst runden.
"""

import json
from datetime import date

import requests

DAILY_EXTRACT = "https://www.hko.gov.hk/cis/dailyExtract/dailyExtract_{ym}.xml"
CLIMATE_API = "https://data.weather.gov.hk/weatherAPI/opendata/opendata.php"
CLIMATE_DATATYPE = {"max": "CLMMAXT", "min": "CLMMINT"}

# Spaltenlage in dayData: [Tag, Luftdruck, Max, Mittel, Min, Taupunkt, ...]
_COL = {"max": 2, "min": 4}

_month_cache = {}
_climate_cache = {}

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"


def _num(x):
    """Zahl aus einem dayData-Feld oder None ('***' = fehlt, 'Trace' bei Regen)."""
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def daily_extract_month(year, month):
    """Tagesextrema EINES Monats -> {'YYYY-MM-DD': {'max': float, 'min': float}}.

    Der laufende Monat enthaelt nur die bereits publizierten Tage; die Zeilen
    'Mean/Total' und 'Normal' am Ende sind Monatsstatistik und werden verworfen
    (ihr Tagesfeld ist nicht numerisch — genau daran werden sie erkannt)."""
    key = (int(year), int(month))
    if key in _month_cache:
        return _month_cache[key]
    out = {}
    try:
        r = S.get(DAILY_EXTRACT.format(ym=f"{key[0]:04d}{key[1]:02d}"), timeout=60)
        r.raise_for_status()
        data = json.loads(r.text)
    except Exception as ex:
        print(f"    HKO-Daily-Extract-Fehler {key[0]:04d}-{key[1]:02d}: {ex}")
        return {}          # NICHT cachen: naechster Lauf soll es erneut versuchen
    for stn in data.get("stn", {}).get("data", []):
        m = int(stn.get("month") or key[1])
        for row in stn.get("dayData", []):
            if not row or not str(row[0]).strip().isdigit():
                continue   # 'Mean/Total', 'Normal'
            d = int(str(row[0]).strip())
            vals = {}
            for var, col in _COL.items():
                v = _num(row[col]) if len(row) > col else None
                if v is not None:
                    vals[var] = v
            if vals:
                out[f"{key[0]:04d}-{m:02d}-{d:02d}"] = vals
    _month_cache[key] = out
    return out


def daily_extreme(day, var):
    """Tagesextrem (float, 0,1 C) fuer einen einzelnen Tag oder None.

    None heisst 'noch nicht publiziert' — der Aufrufer soll den Tag dann offen
    lassen und beim naechsten Lauf erneut fragen, nicht auf 0 settlen. Die
    Marktregel sagt ausdruecklich: 'can not resolve until data for this date has
    been published'."""
    if isinstance(day, str):
        day = date.fromisoformat(day)
    return daily_extract_month(day.year, day.month).get(day.isoformat(), {}).get(var)


def climate_series(var):
    """Komplette HKO-Klimareihe ab 1884 -> {'YYYY-MM-DD': float}, mit Monatsverzug.

    Spalten: Jahr,Monat,Tag,Wert,Status — Status 'C' = vollstaendig, '#' =
    unvollstaendig, '***' = fehlt. Nur 'C' wird uebernommen: ein unvollstaendiger
    Tag kann das echte Extremum verfehlt haben und wuerde den Bias verzerren."""
    if var in _climate_cache:
        return _climate_cache[var]
    r = S.get(CLIMATE_API, params={"dataType": CLIMATE_DATATYPE[var], "lang": "en",
                                   "rformat": "csv", "station": "HKO"}, timeout=90)
    r.raise_for_status()
    out = {}
    for line in r.text.splitlines():
        parts = [c.strip().strip('"') for c in line.split(",")]
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        y, m, d, val, status = parts[0], parts[1], parts[2], parts[3], parts[4]
        if status != "C":
            continue
        try:
            out[f"{int(y):04d}-{int(m):02d}-{int(d):02d}"] = float(val)
        except ValueError:
            continue
    _climate_cache[var] = out
    return out


def actual_daily_extremes(var, recent_months=3):
    """Klimareihe + die letzten Monate aus dem Daily Extract -> {'YYYY-MM-DD': float}.

    Damit endet eine Kalibrierung nicht mehr am letzten abgeschlossenen Monat.
    Der Daily Extract hat Vorrang, wo sich beide ueberlappen — er ist die Reihe,
    an der die Bretter aufloesen."""
    out = dict(climate_series(var))
    heute = date.today()
    y, m = heute.year, heute.month
    for _ in range(max(1, recent_months)):
        for tag, vals in daily_extract_month(y, m).items():
            if var in vals:
                out[tag] = vals[var]
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out


if __name__ == "__main__":
    # Kurzer Selbsttest: aktuellster Tag je Quelle + Ueberlappungs-Vergleich.
    for var in ("max", "min"):
        klima = climate_series(var)
        gesamt = actual_daily_extremes(var)
        letzter_klima = max(klima)
        letzter_gesamt = max(gesamt)
        diffs = [(t, klima[t], gesamt[t]) for t in klima if t in gesamt and klima[t] != gesamt[t]]
        print(f"{var}: Klimareihe {len(klima)} Tage bis {letzter_klima} | "
              f"gemerged {len(gesamt)} bis {letzter_gesamt} | "
              f"Widersprueche in der Ueberlappung: {len(diffs)}")
        for t, a, b in diffs[:5]:
            print(f"    {t}: Klima {a} vs Daily-Extract {b}")
