# -*- coding: utf-8 -*-
"""weather_tile_resolution_eval.py — Sagt die Kachel-Feinstufe den Bucket-Kipp voraus?

DIE IDEE (Nutzer, 25.07.2026, am Screenshot selbst entdeckt): Die WU-Current-Kachel
laeuft auf ganzen GRAD FAHRENHEIT, die settelnde History-Tabelle dagegen auf ganzen
GRAD CELSIUS (sie wird aus METAR gespeist, und METAR meldet ganze °C). Belegt an
EFHK: 368 Tabellenmeldungen zeigten ausschliesslich die °F-Werte, die ganzen °C
entsprechen (46, 48, 50, ... 79) — 65/67/69/71 kamen nie vor. Die Kachel zeigte am
25.07. aber 67 °F = 19,44 °C.

Ein °F-Schritt sind 0,556 K, ein Bucket ist 1 K breit. Die Kachel teilt den Bucket
also in knapp zwei Unterstufen — sie sagt, WO INNERHALB des laufenden Buckets die
Temperatur steht. Die Tabelle sagt bei "19" nur "18,5 <= T < 19,5"; die Kachel
verengt das auf ~0,55 K. Genau diese Groesse fehlt dem -1-Waechter: der sieht, DASS
das Tageshoch auf dem gelayten Bucket sitzt, aber nicht, ob noch 0,9 K Luft sind
oder 0,1 K.

DAS IST NICHT DER BEGRABENE LATENZ-EDGE ([[weather-tile-vs-table-latency]]): dort
ging es um Geschwindigkeit (Kachel fuehrt die Tabelle ~20 min) — tot, weil der Markt
die Tabelle ebenso fuehrt und den Live-obs in Echtzeit trackt. Hier geht es um
AUFLOESUNG, nicht um Zeit. Die Frage ist nicht "wer ist schneller", sondern "wieviel
mehr weiss ich ueber den Ausgang, wenn ich die Feinstufe dazunehme".

HYPOTHESE
  Zum Entscheidungszeitpunkt am spaeten Nachmittag steht das Tabellen-Tagesmax auf
  Bucket X. Dann sagt der Abstand der Kachel zur Bucket-OBERKANTE
      rest = obere_grenze(X) - tile_max_bisher
  voraus, ob der Tag noch auf X+1 kippt: kleines rest -> kippt eher.
  NULL: rest traegt nichts, X und die Uhrzeit sagen schon alles.

DATENFALLE, die zuerst repariert werden musste: weather_tile_latency_logger.py:168
holt die History-Tabelle fuer den UTC-Tag (`datetime.now(timezone.utc)`), nicht fuer
den lokalen. In den Stunden, in denen sich lokales und UTC-Datum unterscheiden, zeigt
tbl_max_c damit das Maximum des NACHBARTAGES. Amsterdam 26.07.: Polls um lokal 00:35
meldeten noch 77 °F = 25 °C vom UTC-25.; das echte Laufmax des lokalen 26. war
72 °F = 22,2 °C = der Settlement-Bucket 22. Ungefiltert liegt das so gewonnene
"Tagesmax" in 15 von 61 Faellen zu hoch. Deshalb: nur Polls mit
UTC-Datum == lokales Datum.

WAHRHEIT ist settle_k aus bb_WeatherLadders (die geprueften WU-Tabellen-Settlements,
inkl. HKO-Sonderfall) — NICHT das aus der Tile-Tabelle abgeleitete Max, das an
genau der obigen Falle haengt.

EINHEIT ist der STADT-TAG, nicht der Poll. Innerhalb eines Tages sind aufeinander
folgende Polls fast dasselbe Ereignis; sie zu poolen wuerde n kuenstlich aufblasen
und jeden t-Wert wertlos machen. Je Stadt-Tag genau EIN Entscheidungszeitpunkt.

Aufruf:  python weather_tile_resolution_eval.py [--stunde 16] [--alle-stunden]
"""
import argparse
import math
import statistics
import sys
from collections import defaultdict
from datetime import timezone
from zoneinfo import ZoneInfo

import airportsdata
import pymssql

from weather_calib_divergence_eval import DB_CONFIG
from weather_stations import bucket_grenzen, favorit_k

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

AP = airportsdata.load("ICAO")
_TZ = {}


def tz_of(station):
    if station not in _TZ:
        _TZ[station] = ZoneInfo(AP[station]["tz"])
    return _TZ[station]


def lade():
    """Polls je (city, lokaler Tag), bereits um die UTC-Tag-Falle bereinigt."""
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""SELECT city, station, ts_utc, tile_c, tbl_max_c
                   FROM bb_WeatherTileLatency
                   WHERE tile_c IS NOT NULL AND tbl_max_c IS NOT NULL
                   ORDER BY city, ts_utc""")
    tage, verworfen = defaultdict(list), 0
    for city, st, ts, tile_c, tbl_max_c in cur.fetchall():
        lt = ts.replace(tzinfo=timezone.utc).astimezone(tz_of(st))
        if lt.date() != ts.date():
            verworfen += 1          # tbl_max gehoert dort zum Nachbar-UTC-Tag
            continue
        tage[(city, str(lt.date()))].append((lt, float(tile_c), float(tbl_max_c)))
    cur.execute("""SELECT city, target_date, MAX(settle_k) FROM bb_WeatherLadders
                   WHERE var='max' AND settle_k IS NOT NULL
                   GROUP BY city, target_date""")
    settle = {(a, str(b)): int(v) for a, b, v in cur.fetchall()}
    conn.close()
    return tage, settle, verworfen


def faelle(tage, settle, stunde):
    """Ein Datensatz je Stadt-Tag: Stand zur lokalen Stunde + Ausgang.

    tile_max = hoechster Kachelwert BIS zum Entscheidungszeitpunkt. Der Momentanwert
    allein waere die falsche Groesse: das Tagesmax ist ein Laufmax, und die Kachel
    faellt nach dem Peak wieder — sie soll hier die Feinschaetzung DESSELBEN Peaks
    liefern, den die Tabelle grob auf X rundet."""
    out = []
    for key, polls in sorted(tage.items()):
        if key not in settle:
            continue
        bis = [p for p in polls if p[0].hour <= stunde]
        if not bis or bis[-1][0].hour < stunde - 1:
            continue                      # keine Beobachtung nahe der Entscheidungszeit
        city = key[0]
        tbl_max = max(p[2] for p in bis)
        tile_max = max(p[1] for p in bis)
        X = favorit_k(tbl_max, city)
        rest = bucket_grenzen(X, city)[1] - tile_max
        # KONTROLLGROESSE, die OHNE die Kachel auskommt: wie lange steht X schon?
        # Ein Bucket, auf den die Tabelle gerade erst gesprungen ist, kippt eher
        # weiter als einer, der seit Stunden haelt. Liefert diese Groesse dieselbe
        # Trennung, ist die Kachel-Feinstufe kein eigener Edge, sondern nur ein
        # Umweg zu einer Information, die schon in der Tabelle steht.
        seit = None
        for lt, _, tm in bis:
            if favorit_k(tm, city) == X:
                seit = (bis[-1][0] - lt).total_seconds() / 60.0
                break
        out.append({"city": city, "tag": key[1], "X": X, "settle": settle[key],
                    "rest": rest, "seit": seit, "tbl_max": tbl_max, "tile_max": tile_max})
    return out


def t_zwei(a, b):
    """Welch-t fuer zwei Anteile/Gruppen."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = statistics.variance(a), statistics.variance(b)
    s = math.sqrt(va / len(a) + vb / len(b))
    return (statistics.mean(a) - statistics.mean(b)) / s if s > 0 else float("nan")


def auswerten(f, stunde, leise=False):
    kipp = [x for x in f if x["settle"] > x["X"]]
    bleibt = [x for x in f if x["settle"] == x["X"]]
    drunter = [x for x in f if x["settle"] < x["X"]]
    n = len(kipp) + len(bleibt)
    if not leise:
        print(f"\n=== Entscheidungszeit lokal {stunde}:00 — {len(f)} Stadt-Tage ===")
        print(f"  kippt noch (settle > X): {len(kipp)}   bleibt auf X: {len(bleibt)}"
              f"   settle < X: {len(drunter)}")
        if drunter:
            print("  settle < X heisst: das Tabellenmax stand schon ueber dem Settlement —"
                  " Datenkonflikt, fliegt raus:")
            for x in drunter[:6]:
                print(f"     {x['city']:<13} {x['tag']}  X={x['X']}  settle={x['settle']}")
    if n < 8:
        print("  zu wenige Faelle fuer eine Aussage.")
        return None

    paare = kipp + bleibt
    y = [1 if x["settle"] > x["X"] else 0 for x in paare]
    rest = [x["rest"] for x in paare]
    med = statistics.median(rest)
    nah = [yy for yy, r in zip(y, rest) if r <= med]      # Kachel nah an der Oberkante
    fern = [yy for yy, r in zip(y, rest) if r > med]
    t = t_zwei(nah, fern)
    if not leise:
        print(f"\n  Kipp-Quote gesamt: {100*sum(y)/len(y):.0f} %")
        print(f"  Median rest = {med:.2f} K  (Abstand Kachel-Tagesmax zur Bucket-Oberkante)")
        print(f"    rest <= {med:.2f} (nah am Kipp):  {sum(nah)}/{len(nah)} = "
              f"{100*sum(nah)/len(nah):.0f} % kippen")
        print(f"    rest >  {med:.2f} (noch Luft):    {sum(fern)}/{len(fern)} = "
              f"{100*sum(fern)/len(fern):.0f} % kippen")
        print(f"    Welch-t = {t:+.2f}")
        r_k = statistics.mean(x["rest"] for x in kipp) if kipp else float("nan")
        r_b = statistics.mean(x["rest"] for x in bleibt) if bleibt else float("nan")
        print(f"  Gegenprobe: mittleres rest bei Kippern {r_k:+.2f} K, "
              f"bei Bleibern {r_b:+.2f} K")
        kontrolle(paare, y)
    return {"stunde": stunde, "n": n, "quote": 100*sum(y)/len(y),
            "nah": 100*sum(nah)/len(nah) if nah else float("nan"),
            "fern": 100*sum(fern)/len(fern) if fern else float("nan"), "t": t}


def kontrolle(paare, y):
    """Schlaegt die Kachel die tabellen-eigene Information — und wenn ja, zusaetzlich?

    Erst 'seit' allein (nur Tabelle), dann 'rest' INNERHALB der beiden seit-Haelften.
    Ueberlebt der Kachel-Kontrast beide Haelften, steckt in ihm etwas, das die
    Tabelle nicht hergibt."""
    mit = [(x, yy) for x, yy in zip(paare, y) if x["seit"] is not None]
    if len(mit) < 12:
        print("\n  Kontrolle: zu wenige Faelle mit bekanntem Bucket-Sprung.")
        return
    s_med = statistics.median(x["seit"] for x, _ in mit)
    frisch = [yy for x, yy in mit if x["seit"] <= s_med]     # X gerade erst erreicht
    alt = [yy for x, yy in mit if x["seit"] > s_med]
    print(f"\n  KONTROLLE — dieselbe Trennung nur aus der Tabelle "
          f"('X steht seit', Median {s_med:.0f} min):")
    print(f"    frisch: {sum(frisch)}/{len(frisch)} = {100*sum(frisch)/len(frisch):.0f} % kippen")
    print(f"    alt:    {sum(alt)}/{len(alt)} = {100*sum(alt)/len(alt):.0f} % kippen")
    print(f"    Welch-t = {t_zwei(frisch, alt):+.2f}")

    # Schaerfste Alternativerklaerung: die TABELLE hat selbst eine Sub-Bucket-Position.
    # Sie speichert ganze °F, und deren Rueckrechnung nach °C trifft die Bucket-Mitte
    # nicht (72 °F = 22,22 °C sitzt 0,28 K unter der Oberkante von Bucket 22). Traegt
    # schon diese Groesse die Trennung, braucht es die Kachel nicht — der Befund waere
    # ein Artefakt der °F-Quantisierung statt echter Zusatzinformation.
    r2 = [(bucket_grenzen(x["X"], x["city"])[1] - x["tbl_max"], yy) for x, yy in mit]
    m2 = statistics.median(v for v, _ in r2)
    n2 = [yy for v, yy in r2 if v <= m2]
    f2 = [yy for v, yy in r2 if v > m2]
    print(f"\n  ALTERNATIVE — dieselbe Rechnung mit der TABELLE statt der Kachel "
          f"(Median {m2:.2f} K):")
    if n2 and f2:
        print(f"    nah {sum(n2)}/{len(n2)} = {100*sum(n2)/len(n2):.0f} %   "
              f"fern {sum(f2)}/{len(f2)} = {100*sum(f2)/len(f2):.0f} %   "
              f"t = {t_zwei(n2, f2):+.2f}")
    else:
        print("    Tabellen-Restabstand konstant — keine Trennung moeglich.")

    print("  Kachel-Kontrast INNERHALB der seit-Haelften (der eigentliche Test):")
    for name, teil in (("frisch", [(x, yy) for x, yy in mit if x["seit"] <= s_med]),
                       ("alt", [(x, yy) for x, yy in mit if x["seit"] > s_med])):
        if len(teil) < 6:
            print(f"    {name:<7} zu duenn (n={len(teil)})")
            continue
        r_med = statistics.median(x["rest"] for x, _ in teil)
        nah = [yy for x, yy in teil if x["rest"] <= r_med]
        fern = [yy for x, yy in teil if x["rest"] > r_med]
        if not nah or not fern:
            print(f"    {name:<7} rest konstant, keine Trennung moeglich (n={len(teil)})")
            continue
        print(f"    {name:<7} n={len(teil):>2}  nah {sum(nah)}/{len(nah)} = "
              f"{100*sum(nah)/len(nah):>3.0f} %   fern {sum(fern)}/{len(fern)} = "
              f"{100*sum(fern)/len(fern):>3.0f} %   t = {t_zwei(nah, fern):+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stunde", type=int, default=16, help="lokale Entscheidungsstunde")
    ap.add_argument("--alle-stunden", action="store_true",
                    help="13-18 Uhr nebeneinander (Robustheit, NICHT poolen)")
    a = ap.parse_args()

    tage, settle, verworfen = lade()
    print(f"{sum(len(v) for v in tage.values())} verwertbare Polls "
          f"({verworfen} wegen UTC-Tag-Falle verworfen), "
          f"{len(tage)} Stadt-Tage, davon {len([k for k in tage if k in settle])} "
          f"mit Settlement-Wahrheit.")

    if a.alle_stunden:
        print("\nRobustheit ueber die Entscheidungsstunde "
              "(dieselben Tage, daher NICHT unabhaengig — nur Musterpruefung):")
        print(f"  {'Std':>4} {'n':>4} {'Kipp%':>7} {'nah%':>7} {'fern%':>7} {'t':>7}")
        for s in range(13, 19):
            r = auswerten(faelle(tage, settle, s), s, leise=True)
            if r:
                print(f"  {r['stunde']:>4} {r['n']:>4} {r['quote']:>6.0f}% "
                      f"{r['nah']:>6.0f}% {r['fern']:>6.0f}% {r['t']:>+7.2f}")
        return

    auswerten(faelle(tage, settle, a.stunde), a.stunde)


if __name__ == "__main__":
    main()
