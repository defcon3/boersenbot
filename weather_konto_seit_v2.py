#!/usr/bin/env python3
"""Was ist auf dem Konto passiert, seit V2 laeuft? Vollstaendige Auflistung.

FRAGE DES BETREIBERS (06.08.2026), woertlich: "am 29.7. wurde umgeschaltet. da
betrug der Kontostand -13. seitdem laeuft v2 und wird nach abrechnung der noch
offenen wetten bei ueber -30 stehen. und nun kannst du gern alles auflisten was
dort passiert ist. ich will am ende eben die rund -18 geld sehen."

Das ist eine Kontrollrechnung mit vorgegebener Zielsumme — sie stimmt oder sie
stimmt nicht. Quelle ist NICHT das Autobuy-Log (das kennt nur die eigenen
Kaeufe), sondern die Jupiter-History der Wallet: alle 144 Positionen, auch die
manuellen.

UMSCHALTPUNKT, gemessen statt angenommen: der V2-Commit 6c097fa ist vom 27.07.
22:05 — also NACH dem Kauflauf 12:45 desselben Tages. Der Lauf fuer Zieltag
28.07. fuhr damit noch V1 (Beleg im Log: skip_price/skip_quality). Der erste
V2-Lauf war 28.07. 12:45 fuer Zieltag 29.07. (Beleg: skip_band_*). Die
REGIMES-Tabelle in weather_minus1_review.py nennt faelschlich den 28.07. als
V2-Start — derselbe Fehlertyp, den sie fuer R3 selbst dokumentiert.

RECHNUNG: realizedPnlUsd ist NETTO nach Gebuehren — geprueft an POLY-3324066
(Entry 0,87, Exit 1,00, 5,52 Kontrakte: brutto 0,72 $, Fee 0,031 $, gemeldet
0,686 $). feesPaidUsd darf also NICHT noch einmal abgezogen werden; genau das
war der Fehler vom 01.08.

OFFENE POSITIONEN werden mit dem erwarteten Ausgang bewertet: gewinnt der Lay
(Bucket verfehlt), Auszahlung 1,00 je Kontrakt minus Einsatz minus Fee; sonst
Totalverlust des Einsatzes. Grundlage ist der gemessene Ist-Wert, nicht der
Marktpreis.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, ".")
from jupiter_wallet_scout import hole_history  # noqa: E402

OWNER = "4XxStoKPzoiEJ6hUGEESfE54dCRo97LcCGk2UFieKjSi"
# Erster V2-Kauflauf: 28.07.2026 12:45 UTC (Zieltag 29.07.). Alles, was danach
# EROEFFNET wurde, ist V2; alles davor V1 oder manuell.
V2_START = datetime(2026, 7, 28, 12, 40, tzinfo=timezone.utc).timestamp()
FEE = 0.07


def usd(mikro):
    return None if mikro in (None, "") else int(mikro) / 1e6


def ts(x):
    return datetime.fromtimestamp(int(x), tz=timezone.utc) if x else None


def autobuy_signaturen(logpfad):
    """Signaturen und (Stadt, Zieltag, k) aller Autobuy-Kaeufe aus dem Log."""
    sigs, tripel = set(), set()
    if not logpfad:
        return sigs, tripel
    with open(logpfad, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r["decision"].startswith(("bought", "sent_unverified")):
                continue
            if r.get("signature"):
                sigs.add(r["signature"])
            tripel.add((r["city"], r["target_date"], str(r["k"])))
    return sigs, tripel


def ist_autobuy(pos, sigs, tripel):
    for ev in pos.get("events") or []:
        if ev.get("signature") in sigs:
            return True
    # Fallback ueber Titel: "Highest temperature in <Stadt> on <Monat> <Tag>?"
    em = pos.get("eventMetadata") or {}
    mm = pos.get("marketMetadata") or {}
    titel = em.get("title") or ""
    if "temperature in " not in titel:
        return False
    stadt = titel.split("temperature in ", 1)[1].split(" on ")[0].strip()
    try:
        rest = titel.split(" on ", 1)[1].rstrip("?").strip()
        d = datetime.strptime(rest + " 2026", "%B %d %Y").strftime("%Y-%m-%d")
    except (IndexError, ValueError):
        return False
    k = (mm.get("title") or "").rstrip("°C").strip()
    return (stadt, d, k) in tripel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/ab/hist.json")
    ap.add_argument("--log", default=None, help="Autobuy-Log fuer die Zuordnung")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if args.refresh:
        h = hole_history(OWNER, 400)
        json.dump(h, open(args.cache, "w"), indent=1)
    else:
        h = json.load(open(args.cache))
    sigs, tripel = autobuy_signaturen(args.log)

    offen, erledigt = [], []
    for p in h:
        pnl = usd(p.get("realizedPnlUsd"))
        rec = {
            "titel": (p.get("eventMetadata") or {}).get("title") or "?",
            "bucket": (p.get("marketMetadata") or {}).get("title") or "?",
            "seite": "No" if not p.get("isYes") else "Yes",
            "status": p.get("status"),
            "auf": ts(p.get("openedAt")),
            "zu": ts(p.get("closedAt")),
            "kontrakte": float(p.get("totalContractsDecimal") or 0),
            "entry": usd(p.get("entryPriceUsd")),
            "pnl": pnl,
            "v2": (p.get("openedAt") or 0) >= V2_START,
            "autobuy": ist_autobuy(p, sigs, tripel),
        }
        (erledigt if p.get("closedAt") else offen).append(rec)

    erledigt.sort(key=lambda r: r["zu"])

    # --- Kumulative realisierte Kurve, wie sie der Chart zeigt ---
    print("REALISIERTE KURVE (Reihenfolge = Abrechnungszeitpunkt, wie im Chart)\n")
    kum = 0.0
    stand_bei_umschaltung = None
    letzter_v1_tag = None
    for r in erledigt:
        kum += r["pnl"] or 0.0
        if stand_bei_umschaltung is None and r["zu"].timestamp() >= V2_START:
            stand_bei_umschaltung = kum - (r["pnl"] or 0.0)
            letzter_v1_tag = r["zu"]
    tage = defaultdict(lambda: [0.0, 0])
    for r in erledigt:
        d = r["zu"].strftime("%Y-%m-%d")
        tage[d][0] += r["pnl"] or 0.0
        tage[d][1] += 1
    kum = 0.0
    print(f"  {'Tag':11s} {'n':>3s} {'Tages-PnL':>10s} {'kumuliert':>11s}")
    for d in sorted(tage):
        kum += tage[d][0]
        marke = "  <- ab hier V2-Positionen" if d == "2026-07-30" else ""
        print(f"  {d:11s} {tage[d][1]:3d} {tage[d][0]:+10.2f} {kum:+11.2f}{marke}")
    print(f"\n  Summe realisiert: {kum:+.2f} $  ({len(erledigt)} Positionen)")

    # --- Trennlinie: eroeffnet unter V1 gegen unter V2 ---
    print("\n" + "=" * 78)
    print("TRENNUNG NACH AUSWAHLREGEL (Zeitpunkt der EROEFFNUNG)")
    print("=" * 78)
    for name, filt in (("V1 / manuell (eroeffnet vor 28.07. 12:40 UTC)", lambda r: not r["v2"]),
                       ("V2 (eroeffnet ab 28.07. 12:40 UTC)", lambda r: r["v2"])):
        teil = [r for r in erledigt if filt(r)]
        s = sum(r["pnl"] or 0 for r in teil)
        ab = [r for r in teil if r["autobuy"]]
        man = [r for r in teil if not r["autobuy"]]
        print(f"\n{name}")
        print(f"  {len(teil):3d} Positionen realisiert, PnL {s:+.2f} $")
        print(f"    davon Autobuy : {len(ab):3d} Pos, {sum(r['pnl'] or 0 for r in ab):+7.2f} $")
        print(f"    davon manuell : {len(man):3d} Pos, {sum(r['pnl'] or 0 for r in man):+7.2f} $")

    # --- Alles seit der Umschaltung, Position fuer Position ---
    print("\n" + "=" * 78)
    print("ALLE POSITIONEN SEIT DER UMSCHALTUNG — eroeffnet ab 28.07. 12:40 UTC")
    print("=" * 78)
    seit = sorted([r for r in erledigt if r["v2"]], key=lambda r: r["auf"])
    print(f"  {'eroeffnet':11s} {'Markt':38s} {'Kf':>5s} {'Ein':>5s} {'PnL':>7s}  Q")
    s = 0.0
    for r in seit:
        s += r["pnl"] or 0
        kurz = r["titel"].replace("Highest temperature in ", "").rstrip("?")
        q = "A" if r["autobuy"] else "M"
        ein = f"{r['entry']:5.2f}" if r["entry"] is not None else "    -"
        pnl = f"{r['pnl']:+7.2f}" if r["pnl"] is not None else "      -"
        print(f"  {r['auf'].strftime('%d.%m %H:%M')} {kurz[:38]:38s} "
              f"{r['kontrakte']:5.2f} {ein} {pnl}  {q}")
    print(f"  {'':11s} {'SUMME realisiert seit Umschaltung':38s} {'':5s} {'':5s} {s:+7.2f}")

    # --- Offene Positionen ---
    print("\n" + "=" * 78)
    print("NOCH OFFEN (nicht im Chart, weil nicht realisiert)")
    print("=" * 78)
    off_summe = 0.0
    for r in sorted(offen, key=lambda r: r["auf"]):
        kurz = r["titel"].replace("Highest temperature in ", "").rstrip("?")
        e = r["entry"] or 0.0
        einsatz = r["kontrakte"] * e
        off_summe += einsatz
        print(f"  {r['auf'].strftime('%d.%m %H:%M')} {kurz[:38]:38s} "
              f"{r['kontrakte']:5.2f} @ {e:.2f} = {einsatz:5.2f} $ "
              f"Einsatz, Status {r['status']}, "
              f"{'Autobuy' if r['autobuy'] else 'manuell'}")
    print(f"\n  {len(offen)} offene Positionen, {off_summe:.2f} $ Einsatz gebunden.")

    # --- Die Bruecke, nach der der Betreiber gefragt hat ---
    print("\n" + "=" * 78)
    print("BRUECKE: vom Stand bei der Umschaltung bis zum Endstand")
    print("=" * 78)
    # Stand am Abend des 29.07. = kumuliert bis einschliesslich diesem Tag.
    kum2, stand_2907 = 0.0, None
    for d in sorted(tage):
        kum2 += tage[d][0]
        if d == "2026-07-29":
            stand_2907 = kum2
    heute = kum2
    print(f"  Stand 29.07.2026 (realisiert, kumuliert)     {stand_2907:+8.2f} $")
    print(f"  Stand heute      (realisiert, kumuliert)     {heute:+8.2f} $")
    print(f"  --> realisierte Veraenderung seit 29.07.     {heute - stand_2907:+8.2f} $")
    print()
    # Ausgang der offenen Positionen, per METAR am 06.08. gegen den gelayten
    # Bucket geprueft (Tokyo 31,0 = getroffen; Wellington 10,0 = getroffen;
    # Chengdu 26,0 gegen Bucket 31 = verfehlt, Lay gewinnt).
    AUSGANG = {"Tokyo on August 6": False, "Wellington on August 6": False,
               "Chengdu on August 6": True}
    print("  Noch nicht im Chart, weil offen (Ausgang per METAR geprueft):")
    erwartet, alt = 0.0, 0.0
    for r in sorted(offen, key=lambda r: r["auf"]):
        kurz = r["titel"].replace("Highest temperature in ", "").rstrip("?")
        e = r["entry"] or 0.0
        einsatz = r["kontrakte"] * e
        if kurz in AUSGANG:
            if AUSGANG[kurz]:
                fee = FEE * r["kontrakte"] * min(e, 1 - e)
                wert = r["kontrakte"] - einsatz - fee
            else:
                wert = -einsatz
            erwartet += wert
            print(f"    {kurz[:44]:44s} {wert:+7.2f} $  "
                  f"({'Lay gewinnt' if AUSGANG[kurz] else 'Bucket getroffen'})")
        else:
            # Altbestand von VOR der Umschaltung — gehoert nicht in diese Rechnung.
            alt += -einsatz
            print(f"    {kurz[:44]:44s} {-einsatz:+7.2f} $  "
                  f"(vom {r['auf'].strftime('%d.%m.')}, NICHT V2-Aera — separat)")
    print(f"  --> Summe offene V2-Positionen               {erwartet:+8.2f} $")
    print()
    print(f"  ENDSTAND nach Abrechnung (ohne Altbestand)   "
          f"{heute + erwartet:+8.2f} $")
    print(f"  GESAMTVERAENDERUNG seit dem 29.07.           "
          f"{heute + erwartet - stand_2907:+8.2f} $")

    # --- Woraus die Veraenderung besteht ---
    print("\n" + "-" * 78)
    print("  WORAUS SIE BESTEHT")
    v2_ab = sum(r["pnl"] or 0 for r in erledigt if r["v2"] and r["autobuy"])
    v2_man = sum(r["pnl"] or 0 for r in erledigt if r["v2"] and not r["autobuy"])
    nach = [r for r in erledigt if not r["v2"]
            and r["zu"].strftime("%Y-%m-%d") > "2026-07-29"]
    alt_nach = sum(r["pnl"] or 0 for r in nach)
    print(f"    Autobuy V2, realisiert ({sum(1 for r in erledigt if r['v2'] and r['autobuy']):2d} Pos)      {v2_ab:+8.2f} $")
    print(f"    Autobuy V2, noch offen  ( 3 Pos)      {erwartet:+8.2f} $")
    print(f"    manuelle Trades seit Umschaltung ({sum(1 for r in erledigt if r['v2'] and not r['autobuy']):2d})  {v2_man:+8.2f} $")
    print(f"    Altpositionen, nach dem 29.07. bezahlt ({len(nach):2d}) {alt_nach:+7.2f} $")
    print(f"    {'':38s} {'-'*9}")
    print(f"    Summe                                 {v2_ab + erwartet + v2_man + alt_nach:+8.2f} $")

    # --- Der Vergleich, um den es geht ---
    print("\n" + "-" * 78)
    print("  AUTOBUY: V1-AERA GEGEN V2-AERA (nur der Bot, echte Abrechnung)")
    v1_ab = [r for r in erledigt if not r["v2"] and r["autobuy"]]
    v2_alle = v2_ab + erwartet
    print(f"    V1-Aera (Zieltage 21.-28.07.): {len(v1_ab):2d} Pos, "
          f"{sum(r['pnl'] or 0 for r in v1_ab):+7.2f} $")
    print(f"    V2-Aera (ab Zieltag 29.07.)  : "
          f"{sum(1 for r in erledigt if r['v2'] and r['autobuy']) + 3:2d} Pos, {v2_alle:+7.2f} $")


if __name__ == "__main__":
    sys.exit(main())
