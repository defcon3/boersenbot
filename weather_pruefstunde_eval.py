#!/usr/bin/env python3
"""weather_pruefstunde_eval.py — Auswertung zu
`preregs/weather_pruefstunde_2026_08_02.md`.

FRAGE: Welche Pruefstunde gilt je Stadt — abgeleitet nach einer VORAB
festgelegten Regel — und traegt das Waechter-Signal dort noch, wenn die Stunde
stimmt?

DIE REGEL, unverhandelbar und vor den Zahlen festgelegt:
    Pruefstunde(Stadt) = frueheste Ortsstunde T des Rasters mit
    P(Hoch kommt nach T) <= q,  q = 0,12,
    UND alle spaeteren Stunden bleiben ebenfalls <= q.
Die Monotonie-Bedingung ist kein Detail: bei Seebrisen-Staedten (Seoul) gibt es
Tage mit zwei Maxima, und ohne sie liest die Regel ein Zwischental als Gipfel —
derselbe Fehler, der den Helsinki-Verlust am 24.07. ausgeloest hat.

WARUM q = 0,12: Die Erstmessung (24.07.) nutzte 17:20 als trennschaerfste Stunde
(88 % gegen 1 %). In Europa entspricht 17:20 einer Restwahrscheinlichkeit von
exakt 12 %. Uebertragen wird die RESTWAHRSCHEINLICHKEIT, nicht die Uhrzeit. Das
ist der einzige freie Parameter, und er wird NICHT variiert.

RASTER 10:20-20:20, vorab erweitert. Das alte Raster (13:20-18:20) reicht nicht:
Taipei liegt um 16:20 bei 0,6 %, seine Pruefstunde liegt davor und waere dort gar
nicht auffindbar.

ZWEI QUELLEN: Basisraten aus NCEI ISD (zwei Sommer), Intraday fuer G4 aus IEM
ASOS (tagesaktuell, dieselbe Quelle wie das Settlement des Ladder-Loggers). Die
Pruefstunde wird aus ISD ABGELEITET und mit IEM ANGEWENDET, nie vermischt.

WOFUER DAS NICHT IST: Der Waechter bleibt auf Eis. Er schadet dem engen Buch
(24.07.: Halten +10,52 $ gegen Waechter +6,38 $), und Breite ist am 02.08.
abgelehnt. Hier entsteht eine Tabelle von Pruefstunden fuer MANUELLE
Entscheidungen — kein Euro Ertrag, und das ist die vollstaendige Beschreibung.

Aufruf:
  python weather_pruefstunde_eval.py                 # alles
  python weather_pruefstunde_eval.py --nur-basis     # G1-G3, ohne IEM-Abrufe
"""

import argparse
import csv
import io
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import pymssql
import requests

import weather_daily_max_timing_isd as isd
from weather_ladder_logger import DB_CONFIG, IEM, settle_bucket
from weather_stations import station_info

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

Q = 0.12
RASTER = [(h, 20) for h in range(10, 21)]      # 10:20 .. 20:20, vorab festgelegt
ALT = (16, 20)                                  # die bisherige starre Regel
EUROPA = ("London", "Paris", "Madrid")          # Konsistenzprobe G3
S1, S2 = 2024, 2025
LADDER_VON, LADDER_BIS = "2026-07-12", "2026-08-01"
MIN_TAGE_JE_SOMMER = 60

S = requests.Session()


def pruefstunde(zaehler):
    """Die Regel. Gibt (h, mm) oder None — nie einen aufgefuellten Randwert."""
    stunden = sorted(zaehler)
    rate = {s: (zaehler[s][0] / zaehler[s][1] if zaehler[s][1] else None)
            for s in stunden}
    for i, s in enumerate(stunden):
        if rate[s] is None or rate[s] > Q:
            continue
        spaeter = [rate[t] for t in stunden[i + 1:]]
        if all(x is not None and x <= Q for x in spaeter):
            return s
    return None


def hh(s):
    return f"{s[0]:02d}:{s[1]:02d}" if s else "—"


def basisraten_je_stadt(jahre):
    """{stadt: {jahr: {stunde: [zaehler, nenner]}}} ueber ISD."""
    isd.STUNDEN = RASTER      # das Raster dieser Pre-Reg, nicht das alte
    ids = isd.isd_ids(list(isd.STAEDTE.values()))
    out = defaultdict(dict)
    for stadt, icao in isd.STAEDTE.items():
        info = station_info(icao)
        tz = (info or {}).get("tz")
        isd_id = ids.get(icao)
        if not tz or not isd_id:
            print(f"  {stadt:<14} keine Zeitzone/ISD-ID, uebersprungen")
            continue
        for jahr in jahre:
            try:
                obs = isd.hole(isd_id, f"{jahr}-06-01", f"{jahr}-08-31")
            except Exception as ex:
                print(f"  {stadt:<14} {jahr}: Abruf fehlgeschlagen ({str(ex)[:50]})")
                continue
            tage = isd.tage_bilden(obs, tz)
            if tage:
                out[stadt][jahr] = isd.basisrate(tage)
                out[stadt].setdefault("n", {})[jahr] = len(tage)
        print(f"  {stadt:<14} "
              + "  ".join(f"{j}: {out[stadt].get('n', {}).get(j, 0)} Tage"
                          for j in jahre), flush=True)
    return out


def iem_reihe(icao, tz_name, von, bis):
    """Intraday-METAR in LOKALER Zeit, nach Kalendertag gruppiert (fuer G4)."""
    r = S.get(IEM, params={
        "station": icao, "data": "tmpc",
        "year1": von.year, "month1": von.month, "day1": von.day,
        "year2": bis.year, "month2": bis.month, "day2": bis.day,
        "tz": tz_name, "format": "onlycomma", "latlon": "no", "elev": "no",
        "missing": "M", "trace": "T", "direct": "no"}, timeout=90)
    r.raise_for_status()
    je_tag = defaultdict(list)
    for row in csv.DictReader(io.StringIO(r.text)):
        try:
            ts = datetime.strptime(row["valid"].strip(), "%Y-%m-%d %H:%M")
            t = float(row["tmpc"])
        except (ValueError, KeyError):
            continue
        je_tag[ts.date()].append((ts, t))
    return {d: sorted(v) for d, v in je_tag.items()}


def lade_kandidaten():
    """-1-Kandidaten mit Settlement — Lead 1, neuester Snapshot je Stadt-Tag."""
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor(as_dict=True)
    cur.execute(
        "SELECT snapshot_utc, target_date, city, icao, k, settle_k "
        "FROM bb_WeatherLadders WHERE var='max' AND kind='eq' AND offset_fav=-1 "
        "AND settle_k IS NOT NULL AND target_date BETWEEN %s AND %s",
        (LADDER_VON, LADDER_BIS))
    roh = cur.fetchall()
    conn.close()

    neueste = {}
    for r in roh:
        if (r["target_date"] - r["snapshot_utc"].date()).days != 1:
            continue
        key = (r["target_date"], r["city"])
        if key not in neueste or r["snapshot_utc"] > neueste[key]:
            neueste[key] = r["snapshot_utc"]

    out = {}
    for r in roh:
        key = (r["target_date"], r["city"])
        if (r["target_date"] - r["snapshot_utc"].date()).days != 1 \
                or r["snapshot_utc"] != neueste.get(key):
            continue
        out[key] = {"tag": r["target_date"], "city": r["city"], "icao": r["icao"],
                    "k": int(r["k"]), "settle_k": int(r["settle_k"]),
                    "verloren": int(r["settle_k"]) == int(r["k"])}
    return out


def signal(reihe_tag, stunde, k, city):
    """Sitzt das gerundete laufende Maximum zur Pruefstunde auf dem Bucket k?"""
    bis_jetzt = [t for ts, t in reihe_tag
                 if (ts.hour, ts.minute) <= stunde]
    if not bis_jetzt:
        return None
    return settle_bucket(max(bis_jetzt), city) == k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nur-basis", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    print(f"PRUEFSTUNDE JE STADT   q = {Q:.2f}   Raster {hh(RASTER[0])}-{hh(RASTER[-1])}")
    print("=" * 78)
    print(f"Basisraten aus ISD, Sommer {S1} und {S2}:")
    raten = basisraten_je_stadt([S1, S2])

    # ---------------------------------------------------------------- G0
    brauchbar = {c: v for c, v in raten.items()
                 if all(v.get("n", {}).get(j, 0) >= MIN_TAGE_JE_SOMMER
                        for j in (S1, S2))}
    print(f"\nG0  DATENBASIS")
    print(f"  Staedte mit >= {MIN_TAGE_JE_SOMMER} Tagen in BEIDEN Sommern: "
          f"{len(brauchbar)} von {len(raten)}")
    g0 = len(brauchbar) >= 20
    print(f"  Verlangt: >= 20  ->  {'BESTANDEN' if g0 else 'GERISSEN'}")
    if not g0:
        print("  Vor G0 wird nicht ausgewertet — so vorregistriert.")
        return

    # -------------------------------------------- Ableitung je Sommer
    ps = {}
    for c, v in brauchbar.items():
        beide = {j: v[j] for j in (S1, S2) if j in v}
        zus = {s: [sum(beide[j][s][0] for j in beide),
                   sum(beide[j][s][1] for j in beide)] for s in RASTER}
        ps[c] = {"s1": pruefstunde(beide.get(S1, {})),
                 "s2": pruefstunde(beide.get(S2, {})),
                 "beide": pruefstunde(zus)}

    print(f"\n{'Stadt':<16}{'S1':>7}{'S2':>7}{'beide':>8}{'Berlin':>9}   "
          f"Rate um {hh(ALT)}")
    for c in sorted(ps):
        info = station_info(isd.STAEDTE.get(c, ""))
        tz = (info or {}).get("tz")
        p = ps[c]["beide"]
        berlin = "—"
        if p and tz:
            try:
                lok = datetime(2026, 7, 15, p[0], p[1], tzinfo=ZoneInfo(tz))
                berlin = lok.astimezone(ZoneInfo("Europe/Berlin")).strftime("%H:%M")
            except Exception:
                pass
        z = brauchbar[c].get(S2, {}).get(ALT) or [0, 0]
        alt_rate = 100 * z[0] / z[1] if z[1] else float("nan")
        print(f"{c:<16}{hh(ps[c]['s1']):>7}{hh(ps[c]['s2']):>7}"
              f"{hh(p):>8}{berlin:>9}   {alt_rate:5.1f} %")

    # ---------------------------------------------------------------- G1
    mit = {c: p["beide"] for c, p in ps.items() if p["beide"]}
    abw = [c for c, p in mit.items() if abs(p[0] - ALT[0]) >= 1]
    print(f"\nG1  RELEVANZ — weicht die Stunde von {hh(ALT)} ab?")
    print(f"  {len(abw)} von {len(mit)} Staedten weichen um >= 1 Stunde ab "
          f"({100*len(abw)/len(mit):.0f} %)")
    g1 = len(mit) and len(abw) / len(mit) >= 0.50
    print(f"  Verlangt: >= 50 %  ->  {'BELEGT' if g1 else 'NICHT BELEGT'}")

    # ---------------------------------------------------------------- G2
    print(f"\nG2  STABILITAET — haelt die Stunde zwischen den Sommern?")
    paare = [(c, p) for c, p in ps.items() if p["s1"] and p["s2"]]
    stabil = [c for c, p in paare if abs(p["s1"][0] - p["s2"][0]) <= 1]
    print(f"  {len(stabil)} von {len(paare)} Staedten stimmen auf <= 1 Stunde "
          f"({100*len(stabil)/len(paare) if paare else 0:.0f} %)")
    wackelig = [c for c, p in paare if abs(p["s1"][0] - p["s2"][0]) > 1]
    if wackelig:
        print(f"  ohne stabile Pruefstunde: {', '.join(sorted(wackelig))}")
    g2 = paare and len(stabil) / len(paare) >= 0.80
    print(f"  Verlangt: >= 80 %  ->  {'BELEGT' if g2 else 'NICHT BELEGT'}")

    # ---------------------------------------------------------------- G3
    print(f"\nG3  KONSISTENZ — liefert die Regel fuer Europa 17:20-18:20?")
    ok = []
    for c in EUROPA:
        p = mit.get(c)
        treffer = bool(p and 17 <= p[0] <= 18)
        ok.append(treffer)
        print(f"  {c:<10}{hh(p):>7}   {'ok' if treffer else 'ABWEICHUNG'}")
    g3 = all(ok)
    print(f"  Verlangt: alle drei  ->  {'BESTANDEN' if g3 else 'GERISSEN'}")
    if not g3:
        print("  Abbruchregel: q ist falsch uebertragen. Es wird KEIN anderes q")
        print("  gesucht — die Ableitungsregel gilt als gescheitert.")
        return

    if a.nur_basis:
        print("\n--nur-basis: G4 uebersprungen.")
        return

    # ---------------------------------------------------------------- G4
    print(f"\nG4  TRENNSCHAERFE — traegt das Signal mit der richtigen Stunde?")
    kand = lade_kandidaten()
    reihen, verworfen = {}, 0
    for c in sorted({v["city"] for v in kand.values()}):
        icao = next((v["icao"] for v in kand.values() if v["city"] == c), None)
        info = station_info(icao) if icao else None
        tz = (info or {}).get("tz")
        if not (icao and tz and c in mit):
            continue
        try:
            reihen[c] = (iem_reihe(icao, tz,
                                   datetime.strptime(LADDER_VON, "%Y-%m-%d"),
                                   datetime.strptime(LADDER_BIS, "%Y-%m-%d")), tz)
        except Exception as ex:
            print(f"  {c}: IEM-Abruf fehlgeschlagen ({str(ex)[:50]})")

    treffer = {"neu": [0, 0], "alt": [0, 0]}     # [Verlierer bei Signal, Signale]
    ohne = {"neu": [0, 0], "alt": [0, 0]}
    for v in kand.values():
        eintrag = reihen.get(v["city"])
        if not eintrag:
            continue
        reihe = eintrag[0].get(v["tag"])
        if not reihe:
            continue
        # Datenprobe: IEM-Tagesmax muss zum geloggten Settlement passen
        if abs(settle_bucket(max(t for _, t in reihe), v["city"]) - v["settle_k"]) > 1:
            verworfen += 1
            continue
        for lbl, stunde in (("neu", mit[v["city"]]), ("alt", ALT)):
            sig = signal(reihe, stunde, v["k"], v["city"])
            if sig is None:
                continue
            eimer = treffer if sig else ohne
            eimer[lbl][1] += 1
            if v["verloren"]:
                eimer[lbl][0] += 1

    print(f"  Kandidaten {len(kand)}, davon wegen Quellen-Abweichung verworfen: "
          f"{verworfen}")
    for lbl, name in (("alt", f"starr {hh(ALT)}"), ("neu", "stadtspezifisch")):
        t, o = treffer[lbl], ohne[lbl]
        tq = 100 * t[0] / t[1] if t[1] else float("nan")
        oq = 100 * o[0] / o[1] if o[1] else float("nan")
        print(f"  {name:<18} Signal {t[1]:4d}x -> Verlierer {tq:5.1f} %   |   "
              f"kein Signal {o[1]:4d}x -> {oq:5.1f} %")
    tq_neu = treffer["neu"][0] / treffer["neu"][1] if treffer["neu"][1] else 0
    tq_alt = treffer["alt"][0] / treffer["alt"][1] if treffer["alt"][1] else 0
    g4 = tq_neu >= tq_alt and treffer["neu"][1] >= treffer["alt"][1]
    print(f"  Verlangt: Trefferquote nicht schlechter UND nicht weniger Signale")
    print(f"  ->  {'BELEGT' if g4 else 'NICHT BELEGT — Stunde abgeleitet, aber nicht besser'}")

    print("\n" + "=" * 78)
    print("Der Waechter wird in KEINEM Ausgang eingeschaltet — dafuer braeuchte es")
    print("zuerst eine Entscheidung fuer Breite, und die ist gefallen: dagegen.")
    print("Was hier entsteht, ist eine Tabelle fuer MANUELLE Entscheidungen.")


if __name__ == "__main__":
    main()
