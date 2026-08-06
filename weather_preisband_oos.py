#!/usr/bin/env python3
"""Haelt der Preisband-Befund out-of-sample? — OOS-Test des V2-Fundaments.

FRAGE DES BETREIBERS (06.08.2026): "kann man aus den bisherigen Verlaeufen
schliessen, dass V2 besser performt als V1?"

WAS HIER GEPRUEFT WIRD: nicht die Bilanz zweier Bots (die haengt an der
Tagesauswahl und an Eingriffen), sondern der EINE Befund, auf dem V2 steht.
Commit 6c097fa vom 27.07.2026:

    NO 0,95-1,00  +2,05 %
    NO 0,85-0,90 +14,01 %
    NO 0,70-0,75 +16,83 %
    NO unter 0,70 -17,64 %
    Band 0,70-0,90: 52 Lays, +13,71 % ROI, t 4,75 ueber 7 Tage

Mit dem dort selbst notierten Vorbehalt: "7 Tage, in-sample, ~10 Varianten
durchprobiert. Der Bandbefund ueberlebt eine grobe Bonferroni-Korrektur knapp,
mehr nicht."

Inzwischen gibt es neun Zieltage (29.07.-06.08.), die bei der Formulierung des
Befunds nicht existierten. Das ist echtes Out-of-Sample.

UNIVERSUM: alle Kandidaten aus dem Autobuy-Log, die bis zum Preisvergleich
durchgelaufen sind — also mit Live-Preis und bestandenem Spannen-Veto. Damit
haengt das Ergebnis NICHT an der Auswahlregel; jedes Band wird an derselben
Grundgesamtheit gemessen, unabhaengig davon, was V1 oder V2 gekauft haetten.

SETTLEMENT: Jupiters Markt-`result`. Keine Wetterquelle — METAR gab am 05.08.
fuer Shenzhen 31,0 Grad aus, waehrend der Markt gegen 32 abrechnete.
"""
import argparse
import csv
import sys
import time
from collections import defaultdict

import pymssql
import requests

sys.path.insert(0, ".")

DB_CONFIG = {"server": "158.181.48.77", "database": "dbdata",
             "user": "326773", "password": "Extaler11!"}
PM_API = "https://prediction-market-api.jup.ag/api"
USD, FEE = 5.0, 0.07
VON, BIS = "2026-07-29", "2026-08-06"
VOR_VETO = ("skip_position", "skip_no_mu", "skip_closed", "skip_api",
            "skip_noforecast", "dry_run")
# Zieltag 06.08. ist bei Jupiter noch offen; Ausgang per METAR, s.
# weather_autobuy_v1_gegenrechnung.ERGEBNIS_0608.
from weather_autobuy_v1_gegenrechnung import ERGEBNIS_0608  # noqa: E402

BAENDER = [("< 0,70", 0.00, 0.70), ("0,70-0,75", 0.70, 0.75),
           ("0,75-0,80", 0.75, 0.80), ("0,80-0,85", 0.80, 0.85),
           ("0,85-0,90", 0.85, 0.90), ("0,90-0,95", 0.90, 0.95),
           ("0,95-1,00", 0.95, 1.01)]


def pnl_lay(no, verloren):
    n = USD / no
    return -USD if verloren else n - USD - FEE * n * min(no, 1 - no)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    with open(args.log, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if VON <= r["target_date"] <= BIS]
    # Doppellauf 06.08.: je (Zieltag, Stadt, k) einmal, Kaufzeile hat Vorrang.
    best = {}
    for r in rows:
        key = (r["target_date"], r["city"], r["k"])
        if key not in best or r["decision"].startswith(("bought", "sent_unver")):
            best[key] = r
    kand = [r for r in best.values()
            if r["buy_no_live"] and not r["decision"].startswith(VOR_VETO)
            and not r["decision"].startswith("skip_spread")]
    print(f"Universum: {len(kand)} Kandidaten, Zieltage {VON} bis {BIS}\n")

    conn = pymssql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT target_date, city, k, market_id FROM bb_WeatherLadders "
                "WHERE var='max' AND kind='eq' AND target_date >= %s AND target_date <= %s "
                "AND market_id IS NOT NULL", (VON, BIS))
    mids = {(str(t), c, int(k)): m for t, c, k, m in cur.fetchall()}

    daten, offen = [], 0
    for i, r in enumerate(kand):
        key = (r["target_date"], r["city"], int(r["k"]))
        verloren = None
        mid = mids.get(key)
        if mid:
            for _, pause in ((1, 3), (2, 12), (3, 0)):
                try:
                    resp = requests.get(f"{PM_API}/v1/markets/{mid}", timeout=25)
                    if resp.status_code == 429:
                        time.sleep(pause)
                        continue
                    m = resp.json()
                    if (m.get("status") or "").lower() == "closed":
                        verloren = (m.get("result") or "").lower() == "yes"
                    break
                except Exception:
                    if pause:
                        time.sleep(pause)
        if verloren is None and r["target_date"] == "2026-08-06":
            verloren = ERGEBNIS_0608.get((r["city"], int(r["k"])))
        if verloren is None:
            offen += 1
            continue
        no = float(r["buy_no_live"])
        daten.append((no, verloren, pnl_lay(no, verloren),
                      r["decision"].startswith(("bought", "sent_unver"))))
        if i % 25 == 0:
            print(f"  ... {i}/{len(kand)}")
    print(f"  {len(daten)} mit Ergebnis, {offen} noch offen.\n")

    print("PREISBAND-STAFFEL OUT-OF-SAMPLE (alle Kandidaten, nicht nur gekaufte)")
    print(f"  {'Band':10s} {'n':>4s} {'Treffer':>9s} {'ROI':>9s} {'Break-even':>11s} {'Delta':>8s}")
    for lab, lo, hi in BAENDER:
        teil = [d for d in daten if lo <= d[0] < hi]
        if not teil:
            continue
        n = len(teil)
        tq = 100 * sum(1 for d in teil if not d[1]) / n
        roi = 100 * sum(d[2] for d in teil) / (n * USD)
        be = 100 * sum(d[0] for d in teil) / n     # Break-even = mittlerer NO-Preis
        print(f"  {lab:10s} {n:4d} {tq:8.1f} % {roi:+8.2f} % {be:10.1f} % {tq-be:+7.1f} pp")

    print("\nZUM VERGLEICH — der Befund vom 27.07. (in-sample, 7 Tage):")
    print("  0,70-0,75  +16,83 %   0,85-0,90  +14,01 %   0,95-1,00  +2,05 %   "
          "< 0,70  -17,64 %")

    # --- Schwellen-Analyse: "ab welchem NO-Preis lohnt es?" ---
    # ACHTUNG, methodisch: diese Tabelle wird auf DENSELBEN Daten gerechnet, aus
    # denen die Frage entstanden ist. Jede Schwelle, die hier gut aussieht, ist
    # in-sample — genau der Fehler vom 27.07. (dort: ~10 Varianten, Band
    # 0,70-0,90 gewaehlt, OOS gefallen). Die Spalten p und t sind deshalb
    # wichtiger als der ROI.
    import statistics as stat
    print("\nSCHWELLEN-ANALYSE  (alles ab NO >= x kaufen)")
    print(f"  {'ab NO':>6s} {'n':>4s} {'Treffer':>8s} {'BE':>7s} {'Delta':>8s} "
          f"{'ROI':>8s} {'t':>7s}")
    for x in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        teil = [d for d in daten if d[0] >= x]
        if len(teil) < 2:
            continue
        n = len(teil)
        tq = 100 * sum(1 for d in teil if not d[1]) / n
        be = 100 * sum(d[0] for d in teil) / n
        roi = 100 * sum(d[2] for d in teil) / (n * USD)
        r = [d[2] / USD for d in teil]
        t = stat.mean(r) / (stat.stdev(r) / n ** 0.5) if stat.stdev(r) else 0.0
        print(f"  {x:6.2f} {n:4d} {tq:7.1f} % {be:6.1f} % {tq-be:+7.1f} pp "
              f"{roi:+7.2f} % {t:+7.2f}")

    # Wie zufaellig ist das beste Band? 0,80-0,85 traegt die ganze Attraktivitaet
    # der Schwelle 0,80 — mit 7 von 7 Treffern. Bei fairer Bepreisung ist genau
    # das keineswegs selten.
    teil = [d for d in daten if 0.80 <= d[0] < 0.85]
    if teil:
        p_mittel = sum(d[0] for d in teil) / len(teil)
        p_alle_treffen = p_mittel ** len(teil)
        print(f"\n  Das Band 0,80-0,85 traegt die Schwelle 0,80: {len(teil)} von "
              f"{len(teil)} getroffen.")
        print(f"  Bei fairer Bepreisung (p = {p_mittel:.3f}) passiert das rein "
              f"zufaellig in {100*p_alle_treffen:.1f} % der Faelle.")
        ohne = [d for d in daten if d[0] >= 0.80 and not (0.80 <= d[0] < 0.85)]
        if ohne:
            print(f"  Schwelle 0,80 OHNE dieses eine Band: {len(ohne)} Lays, "
                  f"{100*sum(d[2] for d in ohne)/(len(ohne)*USD):+.2f} % ROI")

    band = [d for d in daten if 0.70 <= d[0] < 0.90]
    rest = [d for d in daten if d[0] >= 0.90]
    if band and rest:
        import statistics as stat
        rb = [d[2] / USD for d in band]
        rr = [d[2] / USD for d in rest]
        se = (stat.variance(rb) / len(rb) + stat.variance(rr) / len(rr)) ** 0.5
        t = (stat.mean(rb) - stat.mean(rr)) / se
        print(f"\nDAS V2-BAND GEGEN DEN REST:")
        print(f"  Band 0,70-0,90 : {len(band):3d} Lays, "
              f"{100*sum(d[2] for d in band)/(len(band)*USD):+6.2f} % ROI")
        print(f"  NO >= 0,90     : {len(rest):3d} Lays, "
              f"{100*sum(d[2] for d in rest)/(len(rest)*USD):+6.2f} % ROI")
        print(f"  t = {t:+.2f}  (in-sample war es t = +4,75 zugunsten des Bandes)")


if __name__ == "__main__":
    sys.exit(main())
