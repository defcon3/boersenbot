#!/usr/bin/env python3
"""
green_up.py — Green-up / Hedge-Tool für Jupiter Prediction Markets.

Idee (vom Nutzer 2026-06-30): Du hältst Leg 1 (eine Position in einem binären
Up/Down- bzw. Yes/No-Markt). Dieses Tool platziert eine RUHENDE Limit-Gegen-Order
auf der ANDEREN Seite DESSELBEN Markts, zu einem Preis, der — falls je gefüllt —
einen garantierten Gewinn sichert, EGAL wie die Range auflöst. Wird der Limit-Preis
nie erreicht, ruht die Order folgenlos (kein Risiko, keine Aktion).

Mathematik (binär, A = gehaltene Seite, B = Gegenseite, je $1 Auszahlung bei Sieg):
  Halte N Kontrakte A @ Schnitt a.  Kaufe M = N Kontrakte B @ Limit b.
    A gewinnt: Auszahlung N,  Kosten N*a + N*b  ->  Gewinn N*(1 - a - b)
    B gewinnt: Auszahlung N,  Kosten N*a + N*b  ->  Gewinn N*(1 - a - b)
  Bei M = N also IDENTISCHER Gewinn auf beiden Seiten = N*(1 - a - b),
  positiv genau dann, wenn a + b < 1.
  Für gesicherten Gewinn >= p je Kontrakt: Limit  b = 1 - a - p  (auf Cent abgerundet).

WICHTIG: Der Lock entsteht nur, WENN der Markt sich zugunsten von Leg 1 bewegt
(B wird billig genug, dass der Limit füllt). Bewegt er sich dagegen, füllt die
Order nie und du trägst weiter das volle Risiko von Leg 1. Kein Arbitrage-Gelddrucker.

Wiederverwendung: CreateOrder/Sign/Execute aus jupiter_buy.py (verifizierter Pfad),
Positions-Abruf + Key-Lader aus jupiter_sell.py.

CLI:
  python green_up.py --market POLY-2734938-0 --profit 0.05            # Dry-Run, volle Position hedgen
  python green_up.py --market POLY-2734938-0 --profit 0.05 --send     # echt platzieren
  python green_up.py --market ... --contracts 5 --entry 0.30 ...      # Overrides
"""

import argparse
import json
import math
import sys

import requests
from solders.keypair import Keypair  # noqa: F401 (über jupiter_buy genutzt)

sys.path.insert(0, ".")
from jupiter_sell import load_keypair, get_position, API  # noqa: F401
from jupiter_buy import sign_owner_slot, execute_order, build_order, JUPUSD  # noqa: F401

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Green-up: garantierte-Gewinn-Gegenwette per Limit.")
    ap.add_argument("--market", required=True, help="marketId von Leg 1, z. B. POLY-2734938-0")
    ap.add_argument("--profit", type=float, default=0.03,
                    help="gesicherter Gewinn je Kontrakt (USD), default 0.03")
    ap.add_argument("--contracts", type=float, default=None,
                    help="Hedge-Stückzahl (default = volle Position = identischer Gewinn beidseits)")
    ap.add_argument("--entry", type=float, default=None,
                    help="Schnitt-Einstieg Leg 1 (default = avgPriceUsd der Position)")
    ap.add_argument("--limit", type=float, default=None,
                    help="Limit-Preis manuell überschreiben (sonst = 1 - entry - profit, auf Cent gefloort)")
    ap.add_argument("--send", action="store_true", help="ECHT platzieren (sonst Dry-Run)")
    args = ap.parse_args()

    kp = load_keypair()
    owner = str(kp.pubkey())

    pos = get_position(owner, args.market)
    if not pos:
        print(f"❌ Keine offene Position im Markt {args.market} (Wallet {owner[:10]}…).")
        print("   Tipp: erst Leg 1 setzen (jupiter_buy.py), dann hedgen.")
        sys.exit(1)

    held_is_yes = bool(pos.get("isYes"))
    avg = (args.entry if args.entry is not None else int(pos.get("avgPriceUsd") or 0) / 1e6)
    n_held = float(pos.get("contractsDecimal") or 0)
    n_hedge = args.contracts if args.contracts is not None else n_held
    title = pos.get("marketMetadata", {}).get("title", "?")

    held_lbl = "YES/Up" if held_is_yes else "NO/Down"
    hedge_is_yes = not held_is_yes
    hedge_lbl = "YES/Up" if hedge_is_yes else "NO/Down"

    # Limit der Gegenseite: b = 1 - a - p, auf Cent ABGERUNDET (sichert >= p)
    if args.limit is not None:
        b = args.limit
    else:
        b = math.floor((1.0 - avg - args.profit) * 100) / 100.0

    print("=" * 66)
    print(f"GREEN-UP  |  {owner[:10]}…  |  {args.market}")
    print(f"Markt     : {title}")
    print(f"Leg 1 hält: {n_held:g} Kontrakte {held_lbl} @ Schnitt {avg:.3f}")
    print("=" * 66)

    if b < 0.01:
        print(f"❌ Lock unmöglich: Limit b = 1 - {avg:.3f} - {args.profit:.3f} = {b:.3f} < 0.01.")
        print(f"   Leg 1 liegt noch nicht weit genug im Gewinn, um {args.profit:.2f}/Kontrakt zu sichern.")
        print(f"   Nötig wäre Einstieg+Profit < 0.99 — entweder günstigeren Einstieg oder kleineres --profit.")
        sys.exit(2)
    if b > 0.99:
        b = 0.99

    locked = n_hedge * (1.0 - avg - b)  # bei M=N identisch auf beiden Seiten
    deposit = n_hedge * b
    cost_leg1 = n_held * avg

    print(f"HEDGE     : kaufe {n_hedge:g} Kontrakte {hedge_lbl}  Limit @ {b:.2f}")
    print(f"            (ruht, füllt nur wenn {hedge_lbl} auf <= {b:.2f} fällt)")
    print(f"            Einsatz Hedge max ~{deposit:.2f}$  (Leg-1-Kosten waren ~{cost_leg1:.2f}$)")
    print("-" * 66)
    if abs(n_hedge - n_held) < 1e-9:
        print(f"SZENARIO (M=N, identisch beidseits):")
        print(f"  Up/Down egal  ->  Auszahlung {n_held:g}$,  Gesamtkosten {cost_leg1 + deposit:.2f}$")
        print(f"  ==> GESICHERTER GEWINN  +{locked:.2f}$   ({100*locked/(cost_leg1+deposit):+.1f}% auf Einsatz)")
    else:
        win_a = n_held - cost_leg1 - deposit            # A (gehalten) gewinnt
        win_b = n_hedge - cost_leg1 - deposit           # B (hedge) gewinnt
        print(f"SZENARIO (M={n_hedge:g} != N={n_held:g}, asymmetrisch):")
        print(f"  Leg 1 ({held_lbl}) gewinnt -> {win_a:+.2f}$")
        print(f"  Hedge ({hedge_lbl}) gewinnt -> {win_b:+.2f}$")
        if min(win_a, win_b) <= 0:
            print(f"  ⚠️  Nicht in jedem Fall positiv! (min {min(win_a, win_b):+.2f}$)")
    print("=" * 66)
    if deposit < 5.0:
        need = math.ceil(5.0 / b)
        print(f"❌ Hedge-Einsatz {deposit:.2f}$ < 5$-Mindestorder von Jupiter — nicht platzierbar.")
        print(f"   Für diese Strategie müsste Leg 1 >= {need} Kontrakte halten "
              f"(bei Limit {b:.2f}), oder ein höherer Limit-Preis.")
        sys.exit(3)

    # bauen (skipSigning, gefahrlos), dann signieren; nur mit --send executen.
    # Verifizierter Pfad nutzt depositAmount; bei Fill zum Limit b => n_hedge Kontrakte.
    tx_b64, meta, info = build_order(owner, args.market, hedge_is_yes, deposit, b,
                                     skip_signing=not args.send)
    order = info.get("order", {})
    print(f"gebaut: orderPubkey={str(order.get('orderPubkey', '?'))[:12]}…  "
          f"blockhash={str(meta.get('blockhash', '?'))[:10]}…")
    signed = sign_owner_slot(tx_b64, kp)
    if not args.send:
        print(f"🟡 DRY-RUN: Owner-Slot signiert, NICHT gesendet ({len(signed)} b64-Bytes). "
              f"Mit --send echt platzieren.")
        return
    resp = execute_order(signed, owner)
    print(f"🔴 /execute -> {json.dumps(resp)[:400]}")
    ok = str(resp.get("status", "")).lower() in ("success", "ok", "confirmed", "pending")
    print("✅ Hedge-Limit platziert (ruht im Orderbook)." if ok else "❌ Platzieren fehlgeschlagen.")


if __name__ == "__main__":
    main()
