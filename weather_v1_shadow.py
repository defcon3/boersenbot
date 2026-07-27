#!/usr/bin/env python3
"""
weather_v1_shadow.py — V1 laeuft als Schattenbuch neben V2 weiter.

AUFTRAG (Betreiber, 27.07.2026): "kannst du v1 als schattenbuch mitlaufen
lassen?" — nachdem V2 (Preisband) ohne Forward-Test scharf geschaltet wurde,
soll die alte Regel weiter mitgemessen werden. Sonst gibt es spaeter keinen
ehrlichen Vergleich, sondern nur die Behauptung, V2 sei besser.

KEIN ZWEITER LIVE-LAUF. Der Autobuy loggt bei jedem Lauf JEDEN Kandidaten mit
Live-Preis, Temperaturabstand und Entscheidung. Daraus laesst sich die
V1-Auswahl exakt nachbilden — ohne zusaetzliche API-Last, ohne Zeitversatz
zwischen zwei Laeufen und ohne die Gefahr, versehentlich doppelt zu kaufen.

Voraussetzung dafuer ist, dass fuer JEDEN Kandidaten feststeht, ob die
Modellspanne haelt. Deshalb prueft der Autobuy seit dem 27.07. das Spannen-Veto
VOR dem Preisband (Commit-Kommentar dort). Fuer Zieltage davor ist die
Rekonstruktion unvollstaendig: Kandidaten ueber 0,90 wurden als
`skip_band_teuer` geloggt, ohne dass die Spanne geprueft war — sie fehlen dem
V1-Buch. Das Skript weist solche Tage als LUECKENHAFT aus, statt sie
stillschweigend mitzurechnen.

DIE BEIDEN REGELN
  V1  handelbar = Spanne ok, Live-NO <= 0,97, nicht im Bestand
      Rangfolge = NO absteigend ("die konservativsten zuerst")
      Auswahl   = erste 3 bedingungslos, dann bis Cap 6 nur mit NO >= 0,85
  V2  handelbar = Spanne ok, 0,70 <= Live-NO < 0,90, nicht im Bestand
      Rangfolge = Temperaturabstand absteigend
      Auswahl   = bis Cap 8, kein Guete-Gate, nie auffuellen

RECHNUNG fuer BEIDE Seiten identisch und hypothetisch (5 $/Lay, Fee
0,07*n*min(NO,1-NO)), damit der Vergleich fair ist. Die echten V2-Fills stehen
zur Kontrolle daneben — sie weichen nur um die Slippage ab (gemessen +0,17 ct).

Aufruf:
  python weather_v1_shadow.py --log preregs/weather_minus1_live_log.csv
  python weather_v1_shadow.py --log ... --von 2026-07-28    # nur die V2-Aera
"""

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict

import pymssql

from weather_ladder_logger import DB_CONFIG

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

FEE = 0.07
USD = 5.0

# V1-Parameter, eingefroren wie im Tag `autobuy-v1`
V1_MAX_NO = 0.97
V1_CAP = 6
V1_QUAL_AFTER = 3
V1_QUAL_MIN = 0.85
# V2-Parameter, wie aktuell deployt
V2_BAND_LO, V2_BAND_HI = 0.70, 0.90
V2_CAP = 8

KAUF = ("bought", "sent_unverified")


def pnl_lay(no, verloren):
    """PnL eines 5-$-Lays. verloren=True heisst: der Bucket ist eingetroffen."""
    n = USD / no
    fee = FEE * n * min(no, 1.0 - no)
    return -USD if verloren else n - USD - fee


def lade_log(pfad, von):
    """Kandidatenzeilen je Zieltag. Nur Laeufe, die einen Live-Preis gesehen haben."""
    je_tag = defaultdict(list)
    with open(pfad, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if von and r["target_date"] < von:
                continue
            je_tag[r["target_date"]].append(r)
    return je_tag


def lade_settlement(tage):
    """(target_date, city, k) -> verloren?  (settle_result=1: Bucket eingetroffen)"""
    if not tage:
        return {}
    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT target_date, city, k, settle_result FROM bb_WeatherLadders "
        "WHERE var='max' AND kind='eq' AND settle_result IS NOT NULL "
        "AND target_date BETWEEN %s AND %s", (min(tage), max(tage)))
    out = {}
    for td, city, k, res in cur.fetchall():
        out[(str(td), city, int(k))] = bool(res)
    conn.close()
    return out


def spanne_geprueft(r):
    """Steht fuer diese Zeile fest, ob die Modellspanne haelt?

    Ja, wenn sie entweder am Spannen-Veto gescheitert ist oder es passiert hat
    (dann traegt sie eine Band-/Kauf-/Abstands-Entscheidung). Nein bei
    skip_band_* aus der Zeit VOR dem 27.07. — damals lief der Band-Check zuerst.
    Unterscheidbar sind die Faelle nicht an der Zeile selbst, sondern nur am
    Zieltag: siehe luecken_pruefung().
    """
    d = r["decision"]
    return (d.startswith("skip_spread") or d.startswith("skip_noforecast")
            or d.startswith("skip_band") or d.startswith("skip_abstand")
            or d.startswith("skip_cap") or d.startswith("skip_quality")
            or d.startswith("skip_cash") or d in KAUF or d in ("fail_send", "dry_run"))


def luecken_pruefung(zeilen):
    """Ist die V1-Rekonstruktion fuer diesen Tag vollstaendig?

    Ein Tag ist lueckenhaft, wenn ein Kandidat ueber BAND_HI als skip_band_teuer
    geloggt ist, OHNE dass an dem Tag ueberhaupt Spannen-Vetos vorkamen — dann
    lief der Band-Check zuerst (V2-Stand vor dem 27.07. abends) und die Spanne
    dieser Kandidaten ist unbekannt.
    """
    hat_spread_pruefung = any(r["decision"].startswith(("skip_spread", "skip_noforecast"))
                              for r in zeilen)
    teure = [r for r in zeilen if r["decision"].startswith("skip_band_teuer")]
    return not (teure and not hat_spread_pruefung)


def no_of(r):
    try:
        return float(r["buy_no_live"])
    except (TypeError, ValueError):
        return None


def abstand_of(r):
    try:
        return float(r["abstand"])
    except (TypeError, ValueError):
        return None


def v1_auswahl(zeilen):
    """Was haette V1 gekauft? Rangfolge NO absteigend, Gate ab Rang 4."""
    kand = []
    for r in zeilen:
        if r["decision"] == "skip_position" or r["decision"].startswith(
                ("skip_spread", "skip_noforecast", "skip_closed", "skip_api", "skip_no_mu")):
            continue
        no = no_of(r)
        if no is None or not (0 < no <= V1_MAX_NO):
            continue
        kand.append((no, r))
    kand.sort(key=lambda x: -x[0])
    picks = []
    for no, r in kand:
        if len(picks) >= V1_CAP:
            break
        if len(picks) < V1_QUAL_AFTER or no >= V1_QUAL_MIN:
            picks.append((no, r))
    return picks


def v2_auswahl(zeilen):
    """Was hat V2 gekauft? Aus dem Log, nicht nachgebildet — das ist die Wahrheit."""
    return [(no_of(r), r) for r in zeilen if r["decision"] in KAUF and no_of(r)]


def bewerte(picks, settle, tag):
    """(pnl, n, verlierer, ungesettelt) fuer eine Auswahl."""
    pnl, verl, offen = 0.0, 0, 0
    for no, r in picks:
        key = (tag, r["city"], int(r["k"]))
        if key not in settle:
            offen += 1
            continue
        verloren = settle[key]
        pnl += pnl_lay(no, verloren)
        verl += 1 if verloren else 0
    return pnl, len(picks) - offen, verl, offen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="preregs/weather_minus1_live_log.csv",
                    help="Autobuy-Log (fuehrend auf dem VPS)")
    ap.add_argument("--von", default="2026-07-28",
                    help="frueheste target_date (Default: Start der V2-Aera)")
    a = ap.parse_args()

    je_tag = lade_log(a.log, a.von)
    if not je_tag:
        print(f"Keine Zeilen ab Zieltag {a.von} in {a.log}.")
        return
    settle = lade_settlement(sorted(je_tag))

    print(f"V1-SCHATTENBUCH gegen V2-LIVE — Zieltage ab {a.von}\n")
    print(f"{'Zieltag':<12} {'V1 Lays':>7} {'V1 PnL':>9} {'V2 Lays':>8} {'V2 PnL':>9}"
          f"  {'Delta':>8}  Anmerkung")

    sum_v1 = sum_v2 = 0.0
    n_v1 = n_v2 = 0
    tage_v1, tage_v2 = [], []
    for tag in sorted(je_tag):
        zeilen = je_tag[tag]
        p1, p2 = v1_auswahl(zeilen), v2_auswahl(zeilen)
        pnl1, k1, verl1, offen1 = bewerte(p1, settle, tag)
        pnl2, k2, verl2, offen2 = bewerte(p2, settle, tag)

        hinweise = []
        if not luecken_pruefung(zeilen):
            hinweise.append("LUECKENHAFT (Spanne nicht fuer alle geprueft)")
        if offen1 or offen2:
            hinweise.append(f"{max(offen1, offen2)} noch ungesettelt")
        if not hinweise:
            sum_v1 += pnl1; sum_v2 += pnl2
            n_v1 += k1; n_v2 += k2
            tage_v1.append(pnl1 / (USD * k1) if k1 else 0.0)
            tage_v2.append(pnl2 / (USD * k2) if k2 else 0.0)

        print(f"{tag:<12} {len(p1):7d} {pnl1:+9.2f} {len(p2):8d} {pnl2:+9.2f}"
              f"  {pnl2 - pnl1:+8.2f}  {'; '.join(hinweise)}")

    if n_v1 or n_v2:
        print(f"\n{'GESAMT':<12} {n_v1:7d} {sum_v1:+9.2f} {n_v2:8d} {sum_v2:+9.2f}"
              f"  {sum_v2 - sum_v1:+8.2f}   (nur vollstaendige, gesettelte Tage)")
        if n_v1:
            print(f"  V1 ROI {sum_v1 / (USD * n_v1) * 100:+6.2f} %", end="")
        if n_v2:
            print(f"   |   V2 ROI {sum_v2 / (USD * n_v2) * 100:+6.2f} %", end="")
        print()
        if len(tage_v1) >= 2:
            d = [b - a_ for a_, b in zip(tage_v1, tage_v2)]
            sd = statistics.stdev(d)
            t = statistics.fmean(d) / (sd / len(d) ** 0.5) if sd else None
            print(f"  Differenz je Tag: Mittel {statistics.fmean(d) * 100:+.2f} pp"
                  + (f", t = {t:.2f} ueber {len(d)} Tage" if t is not None else "")
                  + "  (gepaart — beide Regeln sehen dieselben Maerkte)")
    else:
        print("\n(noch kein vollstaendiger, gesettelter Tag — Vergleich folgt)")

    print("\nHEUTIGE AUSWAHL IM DETAIL (letzter Zieltag)")
    tag = sorted(je_tag)[-1]
    p1 = {r["city"] for _, r in v1_auswahl(je_tag[tag])}
    p2 = {r["city"] for _, r in v2_auswahl(je_tag[tag])}
    for r in sorted(je_tag[tag], key=lambda r: -(no_of(r) or 0)):
        wer = []
        if r["city"] in p1: wer.append("V1")
        if r["city"] in p2: wer.append("V2")
        no, ab = no_of(r), abstand_of(r)
        print(f"  {r['city']:<14} {r['k']:>3}°C  NO {no if no else '  -  '}"
              f"  d {ab if ab is not None else ' - '}  "
              f"{'/'.join(wer) or '  ':<6} {r['decision']}")


if __name__ == "__main__":
    main()
