#!/usr/bin/env python3
"""
weather_minus1_shadow.py — Schattenbuch der −1-Klasse: was haetten ALLE Kandidaten gebracht?

FRAGE (Betreiber, 26.07.2026): Der Autobuy findet taeglich ~15 −1-Kandidaten und
setzt davon 3. Die uebrigen fallen durch drei Filter — Preis (MAX_NO), Modellspanne
(Spannen-Veto) und das gestufte Guete-Gate. Was waere passiert, wenn jeder Kandidat
mit 5 $ gekauft worden waere? Zielrichtung ist die Skalierung ueber BREITE: mehr
kleine Positionen statt groesserer (Slippage bricht ab ~250 $/Position ein).

Der Vergleich beantwortet zugleich die Review-Frage, ob die Filter Geld SPAREN oder
KOSTEN. Ein Filter, der nur Verlierer aussortiert, ist bares Geld; einer, der
ueberwiegend Gewinner wegwirft, ist eine Bremse — der Unterschied ist rein
empirisch und war bisher nicht gemessen.

DATENBASIS
  bb_WeatherLadders  — fuehrend. Enthaelt JEDEN −1-Kandidaten mit Vortags-Preis
                       und settle_result, auch die, die der Autobuy nie sah.
                       (Das CSV-Log hat Luecken: am 25.07. brach der Lauf nach
                       einem 429 ab und schrieb nur 4 statt ~15 Zeilen.)
  weather_minus1_live_log.csv — optional, liefert die ENTSCHEIDUNG je Kandidat
                       (bought / skip_price / skip_spread / skip_quality). Ohne
                       die Datei laeuft alles, nur die Filter-Aufschluesselung
                       fehlt. Fuehrende Kopie liegt auf dem VPS.

RECHNUNG (identisch zum Bestand, vgl. weather_classb_eval / autopilot)
  Einsatz 5 $ je Lay, Kontrakte n = 5 / NO
  Fee     0,07 * n * min(NO, 1−NO)
  Lay gewinnt (Bucket trifft NICHT ein):  Erloes n * 1,00 − Fee
  Lay verliert:                            0
  Nur Fenster mit VORTAGS-Snapshot (Zieltag-Snapshots waeren intraday und damit
  im Rueckblick zu guenstig) und mit bekanntem settle_result.

Aufruf:
  python weather_minus1_shadow.py
  python weather_minus1_shadow.py --log preregs/weather_minus1_live_log.csv
  python weather_minus1_shadow.py --von 2026-07-20
"""

import argparse
import csv
import sys
import time
from collections import defaultdict

import pymssql

from weather_ladder_logger import DB_CONFIG
from weather_source_compare import MODELS, fetch_model_daily_extreme
from weather_stations import station_info

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

FEE = 0.07
USD = 5.0


def pnl_lay(no, verloren):
    """PnL eines 5-$-Lays. verloren=True heisst: der Bucket ist eingetroffen."""
    if not (0 < no < 1):
        return 0.0, 0.0
    n = USD / no
    fee = FEE * n * min(no, 1.0 - no)
    return ((-USD if verloren else n - USD - fee), fee)


def lade_entscheidungen(pfad):
    """(target_date, city, k) -> decision aus dem Autobuy-Log."""
    out = {}
    try:
        with open(pfad, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    out[(r["target_date"], r["city"], int(r["k"]))] = r["decision"]
                except (ValueError, KeyError):
                    continue
    except OSError:
        print(f"(kein Log unter {pfad} — Filter-Aufschluesselung entfaellt)")
    return out


def lade_kandidaten(von):
    """Alle −1-Fenster mit Vortags-Snapshot und bekanntem Ausgang."""
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor(as_dict=True)
    cur.execute("""
        SELECT city, k, buy_no, target_date, settle_result, snapshot_utc
        FROM bb_WeatherLadders
        WHERE var='max' AND kind='eq' AND offset_fav=-1 AND status='open'
          AND buy_no IS NOT NULL AND buy_no > 0 AND buy_no < 1
          AND settle_result IS NOT NULL
          AND CAST(snapshot_utc AS date) < target_date
          AND target_date >= %s
    """, (von,))
    rows = cur.fetchall()
    conn.close()
    # je (Zieltag, Stadt, k) den SPAETESTEN Vortags-Snapshot
    best = {}
    for r in rows:
        key = (str(r["target_date"]), r["city"], r["k"])
        if key not in best or r["snapshot_utc"] > best[key]["snapshot_utc"]:
            best[key] = r
    return best


def modellspannen(posten, staedte_map):
    """Rohe Modellspanne (max-min der 5 Modelle) je (Stadt, Zieltag) nachrechnen.

    bb_WeatherLadders fuehrt die Spanne NICHT mit, das Spannen-Veto ist deshalb
    aus dem Log allein nicht bewertbar (offener Punkt aus Commit eb2dbeff). Die
    previous_day1-Reihe liefert genau den Forecast-Stand, den der Screen am
    Vortag gesehen haette — ein Fetch je Stadt fuer das ganze Fenster statt einer
    je Kandidat.
    """
    tage = sorted({p["target"] for p in posten})
    start, end = tage[0], tage[-1]
    out = {}
    for city in sorted({p["city"] for p in posten}):
        icao = staedte_map.get(city)
        st = station_info(icao) if icao else None
        if not st:
            continue
        try:
            daily, _tz = fetch_model_daily_extreme(icao, st["lat"], st["lon"],
                                                   start, end, max)
        except Exception as ex:
            print(f"   ({city}: Modellabruf fehlgeschlagen — {ex})")
            continue
        for tag in tage:
            werte = [daily[m][tag] for m in MODELS
                     if m in daily and tag in daily[m]]
            if len(werte) == len(MODELS):
                out[(city, tag)] = max(werte) - min(werte)
        time.sleep(0.5)
    return out


def block(titel, posten):
    if not posten:
        print(f"\n{titel}: keine")
        return None
    ges = sum(p["pnl"] for p in posten)
    eins = USD * len(posten)
    gew = sum(1 for p in posten if p["pnl"] > 0)
    print(f"\n{titel}")
    print(f"   {len(posten):3d} Lays | Einsatz {eins:8.2f} $ | PnL {ges:+8.2f} $ "
          f"| {ges / eins * 100:+6.2f} % | {gew}/{len(posten)} gewonnen "
          f"({gew / len(posten) * 100:.0f} %)")
    ver = [p for p in posten if p["pnl"] < 0]
    if ver:
        print("   Verlierer: " + ", ".join(
            f"{p['city']} {p['k']}° ({p['target']}, NO {p['no']:.2f})" for p in ver[:6]))
    return {"n": len(posten), "pnl": ges, "einsatz": eins}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--von", default="2026-07-20",
                    help="frueheste target_date (default: Autobuy-Start 20.07.)")
    ap.add_argument("--log", default="preregs/weather_minus1_live_log.csv",
                    help="Autobuy-Log fuer die Filter-Aufschluesselung")
    ap.add_argument("--spread", action="store_true",
                    help="Modellspanne je Kandidat nachrechnen und das Spannen-Veto "
                         "rueckwirkend bewerten (langsam: 1 Fetch je Stadt)")
    ap.add_argument("--max-spread", type=float, default=3.0,
                    help="Veto-Schwelle, die geprueft werden soll (default 3.0 = MAX_SPREAD)")
    args = ap.parse_args()

    ent = lade_entscheidungen(args.log)
    best = lade_kandidaten(args.von)
    if not best:
        print("Keine gesettelten −1-Kandidaten im Fenster — noch zu frueh.")
        return 0

    posten = []
    for (tag, city, k), r in sorted(best.items()):
        verloren = bool(r["settle_result"])
        p, fee = pnl_lay(float(r["buy_no"]), verloren)
        posten.append({"target": tag, "city": city, "k": k, "no": float(r["buy_no"]),
                       "verloren": verloren, "pnl": p, "fee": fee,
                       "decision": ent.get((tag, city, k))})

    tage = sorted({p["target"] for p in posten})
    print("=" * 78)
    print(f"SCHATTENBUCH −1-Klasse | {len(posten)} gesettelte Kandidaten | "
          f"Zieltage {tage[0]} bis {tage[-1]} ({len(tage)})")
    print(f"Annahme: JEDER Kandidat mit {USD:.0f} $ gelayt, Vortagspreis, "
          f"Fee {FEE:.2f}*n*min(NO,1-NO)")
    print("=" * 78)

    alle = block("ALLE Kandidaten (das Schattenbuch)", posten)

    # 'sent_unverified' ist ein ECHTER Kauf: die Order ging durch, nur der
    # Fill-Zwischencheck lief in ein 429 (siehe verify_fill). Wer das als
    # Ablehnung zaehlt, rechnet gekaufte Positionen dem Schattenbuch zu.
    GEKAUFT = {"bought", "sent_unverified"}
    echt = [p for p in posten if p["decision"] in GEKAUFT]
    rest = [p for p in posten if p["decision"] and p["decision"] not in GEKAUFT]
    if ent:
        block("davon WIRKLICH gekauft (der Bot)", echt)
        block("vom Bot ABGELEHNT (haette das Schattenbuch gekauft)", rest)
        print("\n   Aufschluesselung der Ablehnungen — sparen die Filter Geld?")
        per = defaultdict(list)
        for p in rest:
            per[p["decision"].split("_")[0] + "_" + p["decision"].split("_")[1]
                if p["decision"].count("_") >= 1 else p["decision"]].append(p)
        for grund, ps in sorted(per.items()):
            g = sum(x["pnl"] for x in ps)
            e = USD * len(ps)
            urteil = "Filter SPART" if g < 0 else "Filter KOSTET"
            print(f"      {grund:<16} {len(ps):3d} Lays  PnL {g:+7.2f} $ "
                  f"({g / e * 100:+6.2f} %)  -> {urteil}")

    print("\n   je Zieltag:")
    per_tag = defaultdict(list)
    for p in posten:
        per_tag[p["target"]].append(p)
    for tag, ps in sorted(per_tag.items()):
        g = sum(x["pnl"] for x in ps)
        print(f"      {tag}  {len(ps):3d} Lays  {g:+7.2f} $")

    if args.spread:
        from weather_ladder_logger import STATIONS as LADDER_STATIONS
        print("\n   Spannen-Veto rueckwirkend (Modellspanne nachgerechnet):")
        sp = modellspannen(posten, LADDER_STATIONS)
        mit = [p for p in posten if (p["city"], p["target"]) in sp]
        if not mit:
            print("      keine Spannen rekonstruierbar")
        else:
            print(f"      {len(mit)}/{len(posten)} Kandidaten mit rekonstruierter Spanne")
            durch = [p for p in mit if sp[(p["city"], p["target"])] <= args.max_spread]
            veto = [p for p in mit if sp[(p["city"], p["target"])] > args.max_spread]
            for lab, grp in ((f"Spanne <= {args.max_spread:.1f} (durchgelassen)", durch),
                             (f"Spanne >  {args.max_spread:.1f} (Veto lehnt ab)", veto)):
                if not grp:
                    continue
                g = sum(x["pnl"] for x in grp)
                e = USD * len(grp)
                gew = sum(1 for x in grp if x["pnl"] > 0)
                print(f"      {lab:<36} {len(grp):3d} Lays  {g:+7.2f} $  "
                      f"({g / e * 100:+6.2f} %)  {gew}/{len(grp)} gewonnen "
                      f"({gew / len(grp) * 100:.0f} %)")
            if veto:
                g = sum(x["pnl"] for x in veto)
                print(f"      -> Das Veto haette {g:+.2f} $ "
                      f"{'vermieden — SPART' if g < 0 else 'entgehen lassen — KOSTET'}")

    if alle:
        print(f"\n{'=' * 78}")
        print(f"Kernzahl fuer die Breiten-Frage: {alle['n']} Lays a {USD:.0f} $ "
              f"= {alle['einsatz']:.0f} $ Einsatz -> {alle['pnl']:+.2f} $ "
              f"({alle['pnl'] / alle['einsatz'] * 100:+.2f} %)")
        print("Aussagekraft: erst ab ~30 gesettelten Lays ueberhaupt lesbar, "
              "belastbar deutlich spaeter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
