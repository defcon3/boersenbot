#!/usr/bin/env python3
"""
jupiter_wallet_scout.py — Handelsprofil einer fremden Jupiter-Wallet auslesen.

Alle Daten sind oeffentlich on-chain bzw. ueber die Jupiter-Prediction-API
abrufbar (kein Auth). Zweck ist Scouting: WAS handelt eine Wallet, in welchen
Groessen, zu welchen Preisen und zu welchem Zeitpunkt relativ zum Marktschluss.

Aufruf:
    python jupiter_wallet_scout.py <ownerPubkey> [--max 200]

Ausgewertet wird /v2/history (geschlossene + offene Positionen, aggregiert je
Position). Die entscheidende Kennzahl ist 'Kauf vs. Marktschluss': Positionen,
die erst NACH closeTime eroeffnet werden, sind Nachzuegler-Kaeufe auf ein
faktisch feststehendes Ergebnis - ein voellig anderes Geschaeft als eine
Prognosewette im Voraus.
"""

import argparse
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import requests

PM_API = "https://prediction-market-api.jup.ag/api"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def usd(x):
    try:
        return int(x) / 1e6
    except (TypeError, ValueError):
        return 0.0


def hole_history(owner, maximum=200, block=25):
    """Paginiert /v2/history durch. Gibt die Rohliste zurueck."""
    out, start = [], 0
    while start < maximum:
        r = requests.get(f"{PM_API}/v2/history",
                         params={"ownerPubkey": owner, "start": start,
                                 "end": min(start + block, maximum)}, timeout=25)
        if r.status_code == 429:
            time.sleep(5)
            continue
        r.raise_for_status()
        j = r.json()
        out += j.get("data", [])
        pag = j.get("pagination") or {}
        if not pag.get("hasNext"):
            break
        start += block
        time.sleep(0.5)
    return out


def auswerten(rows):
    if not rows:
        print("Keine Positionen gefunden.")
        return

    pos = []
    for r in rows:
        mm, em = r.get("marketMetadata") or {}, r.get("eventMetadata") or {}
        kontr = float(r.get("totalContractsDecimal") or 0)
        entry = usd(r.get("entryPriceUsd"))
        pos.append({
            "stadt_frage": em.get("title", "?"),
            "bucket": mm.get("title", "?"),
            "kategorie": em.get("category", "?"),
            "is_yes": bool(r.get("isYes")),
            "status": r.get("status"),
            "ergebnis": mm.get("result"),
            "entry": entry,
            "exit": usd(r.get("exitPriceUsd")),
            "kontrakte": kontr,
            "einsatz": entry * kontr,
            "pnl": usd(r.get("realizedPnlUsd")),
            "fees": usd(r.get("feesPaidUsd")),
            "opened": r.get("openedAt") or 0,
            "closed": r.get("closedAt") or 0,
            "close_time": (mm.get("closeTime") or em.get("closeTime") or 0),
        })

    n = len(pos)
    einsatz = sum(p["einsatz"] for p in pos)
    pnl = sum(p["pnl"] for p in pos)
    fees = sum(p["fees"] for p in pos)
    tage = sorted({datetime.fromtimestamp(p["opened"], timezone.utc).date()
                   for p in pos if p["opened"]})

    print("=" * 76)
    print(f"{n} Positionen | Einsatz {einsatz:,.2f} $ | realisiert {pnl:+,.2f} $ "
          f"| Gebuehren {fees:,.2f} $")
    if einsatz:
        print(f"Rendite auf den Einsatz: {pnl / einsatz * 100:+.2f} %")
    if tage:
        print(f"Aktiv an {len(tage)} Tagen: {tage[0]} bis {tage[-1]}")
    print("=" * 76)

    kat = Counter(p["kategorie"] for p in pos)
    print("\nKategorien:", ", ".join(f"{k}: {v}" for k, v in kat.most_common()))
    seite = Counter("YES" if p["is_yes"] else "NO" for p in pos)
    print("Seite     :", ", ".join(f"{k}: {v}" for k, v in seite.most_common()))

    print("\nEinstiegspreise:")
    grenzen = [(0.0, 0.5, "unter 0,50"), (0.5, 0.9, "0,50 - 0,90"),
               (0.9, 0.97, "0,90 - 0,97"), (0.97, 0.995, "0,97 - 0,995"),
               (0.995, 1.01, "ab 0,995")]
    for lo, hi, lab in grenzen:
        grp = [p for p in pos if lo <= p["entry"] < hi]
        if grp:
            e = sum(g["einsatz"] for g in grp)
            g_pnl = sum(g["pnl"] for g in grp)
            print(f"   {lab:<12} {len(grp):3d} Pos.  Einsatz {e:9,.2f} $  "
                  f"PnL {g_pnl:+8,.2f} $  ({g_pnl / e * 100:+.2f} %)" if e else "")

    print("\nPositionsgroessen (Einsatz je Position):")
    gr = sorted(p["einsatz"] for p in pos)
    print(f"   kleinste {gr[0]:,.2f} $ | Median {gr[len(gr) // 2]:,.2f} $ | "
          f"groesste {gr[-1]:,.2f} $")

    # Kernfrage: wann wird gekauft, relativ zum Marktschluss?
    print("\nKaufzeitpunkt relativ zum Marktschluss (closeTime):")
    davor, danach = [], []
    for p in pos:
        if not (p["opened"] and p["close_time"]):
            continue
        (danach if p["opened"] > p["close_time"] else davor).append(p)
    for lab, grp in (("VOR Schluss ", davor), ("NACH Schluss", danach)):
        if not grp:
            continue
        e = sum(g["einsatz"] for g in grp)
        g_pnl = sum(g["pnl"] for g in grp)
        preise = sorted(g["entry"] for g in grp)
        print(f"   {lab}: {len(grp):3d} Pos. | Einsatz {e:9,.2f} $ | "
              f"PnL {g_pnl:+8,.2f} $ | Median-Einstieg {preise[len(preise) // 2]:.3f}")
    if danach:
        std = sorted((p["opened"] - p["close_time"]) / 3600 for p in danach)
        print(f"   Nachzuegler kaufen im Median {std[len(std) // 2]:.1f} h "
              f"nach Schluss (Spanne {std[0]:.1f} - {std[-1]:.1f} h)")

    gew = [p for p in pos if p["pnl"] > 0]
    ver = [p for p in pos if p["pnl"] < 0]
    print(f"\nTreffer: {len(gew)} Gewinner / {len(ver)} Verlierer "
          f"({len(gew) / n * 100:.0f} %)")
    if ver:
        print("   Verlierer:")
        for p in sorted(ver, key=lambda x: x["pnl"])[:8]:
            print(f"      {p['stadt_frage'][:44]:<44} {p['bucket']:>6} "
                  f"{'YES' if p['is_yes'] else 'NO':>3} @ {p['entry']:.3f}  "
                  f"{p['pnl']:+8,.2f} $")

    staedte = defaultdict(lambda: [0, 0.0])
    for p in pos:
        stadt = p["stadt_frage"].replace("Highest temperature in ", "").split(" on ")[0]
        staedte[stadt][0] += 1
        staedte[stadt][1] += p["pnl"]
    print("\nMaerkte:")
    for s, (c, g) in sorted(staedte.items(), key=lambda x: -x[1][0])[:12]:
        print(f"   {s[:38]:<38} {c:3d} Pos.  {g:+8,.2f} $")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("owner", help="ownerPubkey der zu scoutenden Wallet")
    ap.add_argument("--max", type=int, default=200, help="max. Positionen")
    args = ap.parse_args()
    print(f"Wallet {args.owner}")
    auswerten(hole_history(args.owner, args.max))
    return 0


if __name__ == "__main__":
    sys.exit(main())
