"""Erzeugt vollstaendige, homogene Lead-2-Kalibrierungen (40d + 700d, 28 Staedte).

Warum neu statt flicken: sigma(s) = a + b*Spanne wird ueber den Staedte-POOL
gefittet. Ein Flickenteppich aus Einzelstadt-Laeufen haette je Stadt ein anderes
b (oder gar keins) — die Screens wuerden dann teils mit sigma(s), teils mit dem
festen Sigma rechnen, ohne dass man es der CSV ansieht.

Ablauf je Familie:
  1. ein Lauf ueber ALLE Staedte  -> b aus voller Basis, a je Stadt
  2. fehlende Staedte (Rate-Limit-Ausfaelle) einzeln nach, mit --fix-b-from
     auf den Hauptlauf, damit b identisch bleibt
  (Schritt 3 ENTFALLEN seit 02.08.2026: Shenzhen wurde bis dahin zusaetzlich
   gegen die WU-Reihe kalibriert und ueberschrieb damit die METAR-Zeilen. Der
   Locator ZGSZ:9:CN liefert aber gar nicht Shenzhen, sondern "Lau Fau Shan" in
   Hong Kong — die Sonder-CSVs kalibrierten also gegen eine andere Stadt. ZGSZ
   steht jetzt in NO_WUNDERGROUND, fuer Shenzhen entscheidet METAR wie ueberall
   sonst. Siehe weather_stations.wu_station_passt.)

Laeuft sequenziell mit Pausen — parallele Laeufe treiben Open-Meteo ins
Rate-Limit (genau so entstanden die Luecken am 20.07.).
"""
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
from weather_ladder_logger import STATIONS  # noqa: E402

STAMP = "2026_07_20"
PREREGS = Path("preregs")
TMP = Path("../scratchpad_lead2") if False else Path("preregs/.tmp_lead2")
PAUSE = 4          # s zwischen Laeufen (Rate-Limit schonen)
PER_CITY_TIMEOUT = 900

FAMILIES = [
    # (tage, ziel-csv, wu-csv)
    (40,  PREREGS / f"weather_source_calib40d_lead2_{STAMP}.csv",
          PREREGS / f"weather_source_calib40d_lead2_{STAMP}_shenzhen_wu.csv"),
    (700, PREREGS / f"weather_source_calib_lead2_{STAMP}.csv",
          PREREGS / f"weather_source_calib_lead2_{STAMP}_shenzhen_wu.csv"),
]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(args, timeout=PER_CITY_TIMEOUT):
    try:
        p = subprocess.run([sys.executable, "weather_source_compare.py", *args],
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


def cities_in(path):
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {r["city"] for r in csv.DictReader(f)}


def rows_of(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_rows(ziel, rows):
    """Zeilen anhaengen; vorhandene Stadt/Modell-Kombis werden ersetzt."""
    alt = rows_of(ziel) if ziel.exists() else []
    neu_keys = {(r["city"], r["model"]) for r in rows}
    behalten = [r for r in alt if (r["city"], r["model"]) not in neu_keys]
    alle = behalten + rows
    with ziel.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["city", "model", "n", "bias", "sigma", "a", "b"])
        w.writeheader()
        w.writerows(alle)


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    soll = sorted(STATIONS.keys())
    log(f"Ziel: {len(soll)} Staedte je Familie, Lead 2.")
    bericht = []

    for tage, ziel, wu_ziel in FAMILIES:
        log(f"=== {tage}d: Hauptlauf ueber alle Staedte -> {ziel.name}")
        haupt_tmp = TMP / f"haupt_{tage}.csv"
        haupt_tmp.unlink(missing_ok=True)
        ok, out = run(["--days", str(tage), "--lead", "2", "--calib-csv", str(haupt_tmp)],
                      timeout=3600)
        if not ok or not haupt_tmp.exists():
            log(f"  Hauptlauf FEHLGESCHLAGEN: {out[-400:]}")
            bericht.append(f"{tage}d: Hauptlauf fehlgeschlagen")
            continue
        have = cities_in(haupt_tmp)
        log(f"  Hauptlauf ok: {len(have)}/{len(soll)} Staedte")

        # 2. Luecken einzeln nachziehen, b aus dem Hauptlauf fixieren
        fehlt = [c for c in soll if c not in have]
        for i, city in enumerate(fehlt, 1):
            time.sleep(PAUSE)
            ct = TMP / f"nach_{tage}_{city.replace(' ', '_')}.csv"
            ct.unlink(missing_ok=True)
            log(f"  ({i}/{len(fehlt)}) nachziehen: {city}")
            ok, out = run(["--days", str(tage), "--lead", "2", "--city", city,
                           "--calib-csv", str(ct), "--fix-b-from", str(haupt_tmp)])
            if ok and ct.exists() and cities_in(ct):
                append_rows(haupt_tmp, rows_of(ct))
            else:
                log(f"    -> FEHLGESCHLAGEN ({out.strip()[-200:]})")

        have = cities_in(haupt_tmp)
        fehlt_final = [c for c in soll if c not in have]
        # a/b-Abdeckung pruefen (ensemble_mean traegt sigma(s))
        ohne_ab = sorted({r["city"] for r in rows_of(haupt_tmp)
                          if r["model"] == "ensemble_mean" and not (r.get("b") or "").strip()})

        ziel.parent.mkdir(exist_ok=True)
        if ziel.exists():
            bak = ziel.with_suffix(f".csv.bak-{datetime.utcnow():%Y%m%dT%H%M%SZ}")
            ziel.replace(bak)
            log(f"  alte Datei gesichert: {bak.name}")
        haupt_tmp.replace(ziel)
        log(f"  geschrieben: {ziel.name} ({len(have)} Staedte, "
            f"{len(ohne_ab)} ohne sigma(s))")
        bericht.append(f"{tage}d: {len(have)}/{len(soll)} Staedte"
                       + (f", FEHLT: {', '.join(fehlt_final)}" if fehlt_final else "")
                       + (f", ohne sigma(s): {', '.join(ohne_ab)}" if ohne_ab else ""))

        # Schritt 3 (Shenzhen gegen die WU-Reihe) ist am 02.08.2026 entfallen —
        # er kalibrierte gegen Lau Fau Shan in Hong Kong. Begruendung im
        # Modul-Docstring. wu_ziel bleibt in FAMILIES stehen, damit alte
        # Aufrufe/Pfade nachvollziehbar bleiben, wird aber nicht mehr erzeugt.

    log("=== BERICHT ===")
    for b in bericht:
        log("  " + b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
