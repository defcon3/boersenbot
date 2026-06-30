#!/usr/bin/env python3
"""
jupiter_buy.py — CreateOrder-Pipeline (Kauf) für Jupiter Prediction Markets.

Gegenstück zu jupiter_sell.py (Verkauf/Claim). Verifizierter Flow (vgl. Projekt-
notiz 2026-06-24, hier 2026-06-30 für Wimbledon-Picks erneut genutzt):
  1. POST {API}/orders  Body CreateOrderRequest -> base64-Tx (2 Signer-Slots:
       Slot 0 Owner LEER, Slot 1 Jupiter-Relayer 5uFXJogU… wird beim execute gefüllt;
       requiredSigners=[owner]).
  2. NUR den Owner-Slot signieren (NICHT die strikte sign_close_tx aus jupiter_sell,
       die 2/2 erzwingt — hier ist die Tx absichtlich 1/2, Relayer kommt serverseitig).
  3. POST {API}/execute {signedTransaction, ownerPubkey} -> {status:"Success", ...}
       (Jupiter relayed on-chain, KEIN eigenes RPC nötig).

Limit-Order: maxBuyPriceUsd in micro-USD (10000..999999 = $0.01..$0.99). Marketable
gesetzt (knapp über aktuellem Ask) -> matched sofort gegen das Orderbook, ruht nicht.
"""

import argparse
import base64
import json
import sys
import time

import requests
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.message import to_bytes_versioned

sys.path.insert(0, ".")
from jupiter_sell import load_keypair, API  # gleiche Basis + Key-Lader wiederverwenden

JUPUSD = "JuprjznTrTSp2UFa3ZBUFgwdAmtZCq4MQCwysN55USD"
ZERO64 = bytes(64)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_order(owner, market_id, is_yes, deposit_usd, max_price_usd, skip_signing=False):
    """POST /orders -> (tx_b64, meta, order_info). Baut nur."""
    body = {
        "ownerPubkey": owner,
        "isBuy": True,
        "marketId": market_id,
        "isYes": is_yes,
        "orderType": "limit",
        "maxBuyPriceUsd": int(round(max_price_usd * 1e6)),
        "depositAmount": int(round(deposit_usd * 1e6)),
        "depositMint": JUPUSD,
    }
    if skip_signing:
        body["skipSigning"] = True
    r = requests.post(f"{API}/orders", headers={"Content-Type": "application/json"},
                      json=body, timeout=20)
    if r.status_code >= 400:
        raise RuntimeError(f"build /orders HTTP {r.status_code}: {r.text[:300]}")
    j = r.json()
    return j["transaction"], j.get("txMeta", {}), j


def sign_owner_slot(tx_b64, keypair):
    """Fuellt NUR den Owner-Slot (Slot 0). Relayer-Slot bleibt leer -> /execute fuellt ihn."""
    tx = VersionedTransaction.from_bytes(base64.b64decode(tx_b64))
    msg = tx.message
    keys = list(msg.account_keys)
    sigs = list(tx.signatures)
    my_index = keys.index(keypair.pubkey())
    sigs[my_index] = keypair.sign_message(to_bytes_versioned(msg))
    signed = VersionedTransaction.populate(msg, sigs)
    filled = sum(1 for s in signed.signatures if bytes(s) != ZERO64)
    if filled < 1:
        raise RuntimeError("Owner-Slot nicht signiert")
    return base64.b64encode(bytes(signed)).decode()


def execute_order(signed_b64, owner):
    """POST /execute -> Antwort-Dict (status, signature, ...)."""
    r = requests.post(f"{API}/execute", headers={"Content-Type": "application/json"},
                      json={"signedTransaction": signed_b64, "ownerPubkey": owner}, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"/execute HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def place(owner, market_id, is_yes, deposit_usd, max_price_usd, keypair, send=False):
    tx_b64, meta, info = build_order(owner, market_id, is_yes, deposit_usd, max_price_usd,
                                     skip_signing=not send)
    order = info.get("order", {})
    print(f"  gebaut: orderPubkey={order.get('orderPubkey','?')[:12]}… "
          f"positionPubkey={order.get('positionPubkey','?')[:12]}… "
          f"blockhash={meta.get('blockhash','?')[:10]}…")
    if not send:
        # bis zum Signieren gehen, aber NICHT executen (gefahrlos)
        signed = sign_owner_slot(tx_b64, keypair)
        print(f"  DRY-RUN: Owner-Slot signiert, NICHT an /execute gesendet. (Tx {len(signed)} b64-Bytes)")
        return {"ok": True, "dry": True, "order": order}
    signed = sign_owner_slot(tx_b64, keypair)
    resp = execute_order(signed, owner)
    print(f"  /execute -> {json.dumps(resp)[:400]}")
    return {"ok": str(resp.get("status", "")).lower() in ("success", "ok", "confirmed"),
            "dry": False, "resp": resp, "order": order}


def main():
    ap = argparse.ArgumentParser(description="Jupiter Prediction Market — Limit-Kauf.")
    ap.add_argument("--market", required=True, help="marketId, z. B. POLY-2702239-1")
    ap.add_argument("--no", action="store_true", help="NO statt YES kaufen (default YES)")
    ap.add_argument("--usd", type=float, default=5.0, help="Einsatz in USD (Mindest 5)")
    ap.add_argument("--limit", type=float, required=True, help="max. Kaufpreis je Kontrakt, z. B. 0.84")
    ap.add_argument("--send", action="store_true", help="ECHT senden (sonst Dry-Run)")
    args = ap.parse_args()

    kp = load_keypair()
    owner = str(kp.pubkey())
    print("=" * 60)
    print(f"{'🔴 ECHTER KAUF' if args.send else '🟡 DRY-RUN'} | {owner[:10]}… | "
          f"{args.market} {'NO' if args.no else 'YES'} {args.usd}$ @ limit {args.limit}")
    print("=" * 60)
    res = place(owner, args.market, not args.no, args.usd, args.limit, kp, send=args.send)
    print("Ergebnis:", json.dumps({k: v for k, v in res.items() if k not in ("order",)})[:300])


if __name__ == "__main__":
    main()
