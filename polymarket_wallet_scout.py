#!/usr/bin/env python3
"""
polymarket_wallet_scout.py — Handelsprofil einer fremden Polymarket-Adresse.

Gegenstueck zu jupiter_wallet_scout.py, aber fuer native Polymarket-Wallets
(Polygon, 0x...). Beide Welten sind NICHT verknuepfbar: Jupiter-Nutzer haben
Solana-Pubkeys und werden ueber einen Keeper ausgefuehrt, sie tauchen im
Polymarket-Tape nicht einzeln auf. Fuer eine 0x-Adresse also dieses Skript,
fuer eine Solana-Pubkey das andere.

Alle Endpunkte sind oeffentlich und ohne Auth (siehe Notiz polymarket-public-data-api).

Aufruf:
    python polymarket_wallet_scout.py 0x6b50... [--limit 500]

Ausgewertet wird:
  - Equity-Kurve (user-pnl): Hoch, aktueller Stand, groesster Tagesverlust.
    Der letzte Punkt ist der realisierte + unrealisierte Gesamtstand.
  - Aktivitaet (/activity, max. 500 Eintraege = nur die juengsten Tage):
    Kauf/Verkauf-Verhaeltnis, Einsatzgroessen, Preisbaender, Maerkte.
    WICHTIG: Preisbaender werden nach EINSATZ gewichtet ausgewiesen, nicht nach
    Stueckzahl - sonst dominieren Kleinstorders von 15 Cent das Bild.
  - Offene Positionen (/positions): Groesse, Buchgewinn, Totalverluste.

Grenze: /activity liefert maximal 500 Eintraege. Bei einer aktiven Wallet sind
das nur wenige Tage; die Equity-Kurve deckt dagegen die volle Historie ab.
"""

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import requests

DATA = "https://data-api.polymarket.com"
LB = "https://lb-api.polymarket.com"
PNL = "https://user-pnl-api.polymarket.com"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def hole(url, params, default=None):
    try:
        r = requests.get(url, params=params, timeout=30)
        return r.json() if r.ok else default
    except requests.RequestException:
        return default


def f_datum(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%d.%m")


def kopf(addr):
    prof = hole(f"{LB}/profit", {"window": "all", "limit": 1, "address": addr}, [])
    vol = hole(f"{LB}/volume", {"window": "all", "limit": 1, "address": addr}, [])
    val = hole(f"{DATA}/value", {"user": addr}, [])
    p = prof[0].get("amount") if prof else 0
    v = vol[0].get("amount") if vol else 0
    w = val[0].get("value") if val else 0
    name = (prof[0].get("pseudonym") if prof else "") or "(kein Pseudonym)"
    print("=" * 78)
    print(f"{name}   {addr}")
    print(f"Gewinn gesamt {p:+,.2f} $ | Volumen {v:,.2f} $ | Depotwert {w:,.2f} $")
    if v:
        print(f"Rendite auf das Volumen: {p / v * 100:+.2f} %")
    print("=" * 78)


def equity(addr):
    kurve = hole(f"{PNL}/user-pnl", {"user_address": addr, "interval": "all",
                                     "fidelity": "1d"}, [])
    if not isinstance(kurve, list) or not kurve:
        print("\nKeine Equity-Kurve abrufbar.")
        return
    pts = [(x.get("t"), x.get("p")) for x in kurve if isinstance(x, dict)]
    hoch = max(pts, key=lambda x: x[1])
    jetzt = pts[-1]
    print(f"\nEQUITY ({len(pts)} Tage, {f_datum(pts[0][0])} bis {f_datum(jetzt[0])})")
    print(f"   Hoechststand {hoch[1]:+,.2f} $ am {f_datum(hoch[0])}")
    print(f"   aktuell      {jetzt[1]:+,.2f} $")
    print(f"   Abstand zum Hoch: {jetzt[1] - hoch[1]:+,.2f} $ "
          f"({(jetzt[1] - hoch[1]) / hoch[1] * 100:+.1f} %)" if hoch[1] else "")
    taegl = [(pts[i][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts))]
    schlecht = sorted(taegl, key=lambda x: x[1])[:5]
    gut = sorted(taegl, key=lambda x: -x[1])[:3]
    print("   schlechteste Tage: " + ", ".join(f"{f_datum(t)} {d:+,.0f} $" for t, d in schlecht))
    print("   beste Tage       : " + ", ".join(f"{f_datum(t)} {d:+,.0f} $" for t, d in gut))
    pos_tage = sum(1 for _, d in taegl if d > 0)
    print(f"   {pos_tage} von {len(taegl)} Tagen positiv ({pos_tage / len(taegl) * 100:.0f} %)")


def aktivitaet(addr, limit):
    acts = hole(f"{DATA}/activity", {"user": addr, "limit": limit}, [])
    if not acts:
        print("\nKeine Aktivitaet abrufbar.")
        return
    ts = sorted(a.get("timestamp", 0) for a in acts)
    print(f"\nAKTIVITAET ({len(acts)} Eintraege, {f_datum(ts[0])} bis {f_datum(ts[-1])})")
    print("   Typen: " + ", ".join(f"{k}: {v}" for k, v in
                                   Counter(a.get("type") for a in acts).most_common()))

    trades = [a for a in acts if a.get("type") == "TRADE"]
    if not trades:
        return
    seiten = Counter(a.get("side") for a in trades)
    print("   Seiten: " + ", ".join(f"{k}: {v}" for k, v in seiten.most_common()))
    kauf = [a for a in trades if a.get("side") == "BUY"]
    verk = [a for a in trades if a.get("side") == "SELL"]
    print(f"   Kaufvolumen {sum(a.get('usdcSize', 0) for a in kauf):,.2f} $ | "
          f"Verkaufsvolumen {sum(a.get('usdcSize', 0) for a in verk):,.2f} $")

    gr = sorted(a.get("usdcSize", 0) for a in kauf)
    if gr:
        print(f"   Kaufgroessen: Median {gr[len(gr) // 2]:,.2f} $ | "
              f"groesste {gr[-1]:,.2f} $ | kleinste {gr[0]:,.2f} $")

    print("\n   Einstiegspreise der KAEUFE (nach Einsatz gewichtet):")
    baender = [(0, 0.05, "unter 0,05"), (0.05, 0.20, "0,05 - 0,20"),
               (0.20, 0.50, "0,20 - 0,50"), (0.50, 0.80, "0,50 - 0,80"),
               (0.80, 0.95, "0,80 - 0,95"), (0.95, 1.01, "ab 0,95")]
    ges = sum(a.get("usdcSize", 0) for a in kauf) or 1
    for lo, hi, lab in baender:
        grp = [a for a in kauf if lo <= (a.get("price") or 0) < hi]
        if grp:
            e = sum(a.get("usdcSize", 0) for a in grp)
            print(f"      {lab:<12} {len(grp):4d} Trades   {e:9,.2f} $   "
                  f"{e / ges * 100:5.1f} % des Kaufvolumens")

    kat = defaultdict(float)
    for a in trades:
        t = (a.get("title") or "?").lower()
        if "temperature" in t:
            schl = "Wetter"
        elif any(w in t for w in ("bitcoin", "ethereum", "solana", " up or down")):
            schl = "Krypto"
        else:
            schl = "Sonstiges"
        kat[schl] += a.get("usdcSize", 0)
    gesamt = sum(kat.values()) or 1
    print("\n   Themen (nach Volumen): " + ", ".join(
        f"{k} {v / gesamt * 100:.0f} %" for k, v in sorted(kat.items(), key=lambda x: -x[1])))

    staedte = defaultdict(float)
    for a in trades:
        t = a.get("title") or ""
        if "temperature in " in t:
            staedte[t.split("temperature in ")[1].split(" be ")[0].split(" on ")[0]] += \
                a.get("usdcSize", 0)
    if staedte:
        print("   Top-Staedte: " + ", ".join(
            f"{s} {v:,.0f} $" for s, v in sorted(staedte.items(), key=lambda x: -x[1])[:8]))


def positionen(addr):
    pos = hole(f"{DATA}/positions", {"user": addr, "limit": 200}, [])
    if not pos:
        print("\nKeine offenen Positionen.")
        return
    wert = sum(p.get("currentValue") or 0 for p in pos)
    einsatz = sum(p.get("initialValue") or 0 for p in pos)
    print(f"\nOFFENE POSITIONEN: {len(pos)} | Einsatz {einsatz:,.2f} $ | "
          f"aktueller Wert {wert:,.2f} $")
    tot = [p for p in pos if (p.get("percentPnl") or 0) <= -99]
    if tot:
        print(f"   davon {len(tot)} Totalverluste "
              f"({sum(p.get('initialValue') or 0 for p in tot):,.2f} $ Einsatz):")
        for p in sorted(tot, key=lambda x: -(x.get("initialValue") or 0))[:6]:
            print(f"      {(p.get('title') or '?')[:56]:<56} "
                  f"{p.get('initialValue') or 0:8,.2f} $ @ {p.get('avgPrice') or 0:.3f}")
    lebt = [p for p in pos if (p.get("currentValue") or 0) > 1]
    if lebt:
        print(f"   groesste lebende Positionen:")
        for p in sorted(lebt, key=lambda x: -(x.get("currentValue") or 0))[:6]:
            print(f"      {(p.get('title') or '?')[:56]:<56} "
                  f"{p.get('currentValue') or 0:8,.2f} $ @ {p.get('avgPrice') or 0:.3f} "
                  f"-> {p.get('curPrice') or 0:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("address", help="Polymarket-Adresse (0x...)")
    ap.add_argument("--limit", type=int, default=500, help="max. Aktivitaets-Eintraege")
    args = ap.parse_args()
    kopf(args.address)
    equity(args.address)
    aktivitaet(args.address, args.limit)
    positionen(args.address)
    return 0


if __name__ == "__main__":
    sys.exit(main())
