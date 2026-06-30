#!/usr/bin/env python3
"""
jupiter_sell.py — Verifizierte Verkauf-Pipeline für Jupiter Prediction Markets.

Verifizierter Flow (2026-06-20):
  1. GET    /prediction/v1/positions?ownerPubkey=...        -> Position finden
  2. DELETE /prediction/v1/positions/{positionPubkey}       -> base64-Tx
       Body {"ownerPubkey": ...}; Jupiter VORSIGNIERT Slot 1 (Relayer).
  3. Unsere Signatur in unseren Slot setzen, Jupiters erhalten -> 2/2 signiert
  4. An Solana RPC senden + bestätigen

WICHTIG: KEIN API-Key senden! Die API ist öffentlich; ein ungültiger Key -> 401.
"""

import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path

import requests
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.message import to_bytes_versioned

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

API = "https://api.jup.ag/prediction/v1"
# Gut synchronisierte öffentliche RPCs (der Solana-Labs-Default hinkt oft hinterher
# -> "BlockhashNotFound"). Wird der Reihe nach probiert.
RPCS = [
    "https://solana-rpc.publicnode.com",
    "https://api.mainnet-beta.solana.com",
]
RPC = RPCS[0]
KEYPAIR_FILE = Path("hot_wallet/keypair.json")
ZERO64 = bytes(64)

log = logging.getLogger("jupiter_sell")


def load_keypair() -> Keypair:
    return Keypair.from_bytes(bytes(json.loads(KEYPAIR_FILE.read_text())))


def get_position(owner: str, market_id: str) -> dict | None:
    """Findet die offene Position im Markt. Retry bei 429 (öffentliche API)."""
    r = None
    for attempt in range(4):
        r = requests.get(f"{API}/positions", params={"ownerPubkey": owner}, timeout=10)
        if r.status_code != 429:
            break
        ra = r.headers.get("Retry-After", "")
        wait = float(ra) if ra.replace(".", "", 1).isdigit() else 5 * 2 ** attempt
        log.warning(f"get_position 429 (#{attempt + 1}) — warte {wait:.0f}s")
        time.sleep(wait)
    r.raise_for_status()
    for p in r.json().get("data", []):
        if p.get("marketId") == market_id and str(p.get("contracts", "0")) not in ("0", ""):
            return p
    return None


def fetch_position_by_pubkey(owner: str, pubkey: str) -> dict | None:
    """Holt die Position per pubkey (mit 429-Retry) — unabhängig von contracts/claimed.
    Anders als get_position(): liefert die Position auch nach dem Claim (claimed=True),
    damit der Idempotenz-Check im Claim-Loop den Erfolg erkennen kann."""
    r = None
    for attempt in range(4):
        r = requests.get(f"{API}/positions", params={"ownerPubkey": owner}, timeout=10)
        if r.status_code != 429:
            break
        ra = r.headers.get("Retry-After", "")
        wait = float(ra) if ra.replace(".", "", 1).isdigit() else 5 * 2 ** attempt
        log.warning(f"fetch_position 429 (#{attempt + 1}) — warte {wait:.0f}s")
        time.sleep(wait)
    r.raise_for_status()
    for p in r.json().get("data", []):
        if p.get("pubkey") == pubkey:
            return p
    return None


def build_close_tx(position_pubkey: str, owner: str) -> tuple[str, dict]:
    """DELETE-Request -> (base64-Tx, txMeta). Baut nur, sendet NICHT."""
    r = requests.delete(
        f"{API}/positions/{position_pubkey}",
        headers={"Content-Type": "application/json"},
        json={"ownerPubkey": owner},
        timeout=15,
    )
    r.raise_for_status()
    j = r.json()
    return j["transaction"], j.get("txMeta", {})


def build_claim_tx(position_pubkey: str, owner: str) -> tuple[str, dict]:
    """POST /positions/{pubkey}/claim -> (base64-Tx, txMeta). Baut nur, sendet NICHT.

    Verifiziert 2026-06-23 an echter claimbarer Position (Portugal POLY-1897228):
    Anders als der Verkauf (DELETE, Jupiter vorsigniert Relayer-Slot 1) erfordert
    die Claim-Tx NUR die Owner-Signatur (num_required_signatures=1, Owner = Slot 0).
    Der Owner zahlt das Gas selbst -> Hot-Wallet braucht etwas SOL.
    """
    r = requests.post(
        f"{API}/positions/{position_pubkey}/claim",
        headers={"Content-Type": "application/json"},
        json={"ownerPubkey": owner},
        timeout=15,
    )
    r.raise_for_status()
    j = r.json()
    return j["transaction"], j.get("txMeta", {})


def sign_close_tx(tx_b64: str, keypair: Keypair) -> VersionedTransaction:
    """
    Setzt UNSERE Signatur in unseren Slot und erhält Jupiters Vorsignatur.
    Wirft, wenn die Tx danach nicht voll signiert ist.
    """
    tx = VersionedTransaction.from_bytes(base64.b64decode(tx_b64))
    msg = tx.message
    keys = list(msg.account_keys)
    sigs = list(tx.signatures)

    my_index = keys.index(keypair.pubkey())
    sigs[my_index] = keypair.sign_message(to_bytes_versioned(msg))
    signed = VersionedTransaction.populate(msg, sigs)

    n = msg.header.num_required_signatures
    filled = sum(1 for s in signed.signatures if bytes(s) != ZERO64)
    if filled != n:
        raise RuntimeError(f"Tx nicht voll signiert: {filled}/{n}")
    return signed


def send_tx(signed: VersionedTransaction, rpc: str = RPC) -> str:
    """
    Sendet die signierte Tx ans Solana-Netz. Gibt die Signatur zurück.
    skipPreflight=True: umgeht die Simulation des (evtl. hinterherhinkenden)
    RPC-Knotens; der echte Block-Leader validiert den frischen Blockhash.
    """
    raw = base64.b64encode(bytes(signed)).decode()
    r = requests.post(
        rpc,
        json={
            "jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
            "params": [raw, {"encoding": "base64", "skipPreflight": True, "maxRetries": 5}],
        },
        timeout=30,
    )
    j = r.json()
    if "result" in j:
        return j["result"]
    raise RuntimeError(f"Send fehlgeschlagen: {j.get('error')}")


def confirm_tx(sig: str, rpc: str = RPC, timeout: int = 60) -> tuple[bool, str]:
    """Pollt den Bestätigungsstatus. (ok, status_or_error)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.post(
                rpc,
                json={
                    "jsonrpc": "2.0", "id": 1, "method": "getSignatureStatuses",
                    "params": [[sig], {"searchTransactionHistory": True}],
                },
                timeout=15,
            )
            st = (r.json().get("result", {}).get("value") or [None])[0]
            if st:
                if st.get("err"):
                    return False, str(st["err"])
                if st.get("confirmationStatus") in ("confirmed", "finalized"):
                    return True, st["confirmationStatus"]
        except Exception:
            pass
        time.sleep(3)
    return False, "timeout"


def sell_position(owner: str, market_id: str, keypair: Keypair,
                  send: bool = False, attempts: int = 4) -> dict:
    """
    Verkauft die Position im Markt. send=False -> Dry-Run (baut+signiert, sendet nicht).
    Mit Retry: pro Versuch frischer Blockhash + RPC-Rotation. Vor jedem Versuch
    wird die Position neu geprüft -> kein Doppelverkauf.
    """
    pos = get_position(owner, market_id)
    if not pos:
        return {"ok": False, "reason": "keine offene Position"}

    sell_price = int(pos.get("sellPriceUsd", 0)) / 1e6
    cost = int(pos.get("totalCostUsd", 0)) / 1e6
    value = int(pos.get("valueUsd", 0)) / 1e6
    pnl_pct = pos.get("pnlUsdAfterFeesPercent")
    log.info(f"Position {pos['pubkey'][:10]}…  Kontrakte={pos.get('contractsDecimal')}  "
             f"Kosten={cost:.4f}  Wert={value:.4f}  sellPrice={sell_price:.4f}  PnL={pnl_pct}%")

    # Dry-Run: nur bauen + signieren
    if not send:
        tx_b64, meta = build_close_tx(pos["pubkey"], owner)
        sign_close_tx(tx_b64, keypair)
        log.info(f"Close-Tx gebaut + 2/2 signiert (blockhash {meta.get('blockhash','?')[:10]}…). DRY-RUN: nicht gesendet.")
        return {"ok": True, "dry": True, "position": pos}

    last_err, last_sig = None, None
    for attempt in range(1, attempts + 1):
        # Idempotenz: Position noch offen? Sonst hat ein früherer Send gegriffen.
        cur = get_position(owner, market_id)
        if cur is None:
            log.info("Position nicht mehr offen -> bereits verkauft (Erfolg).")
            return {"ok": True, "dry": False, "signature": last_sig,
                    "status": "closed", "position": pos}

        rpc = RPCS[(attempt - 1) % len(RPCS)]
        try:
            tx_b64, meta = build_close_tx(cur["pubkey"], owner)  # frischer Blockhash
            signed = sign_close_tx(tx_b64, keypair)
            sig = send_tx(signed, rpc=rpc)
            last_sig = sig
            log.info(f"Versuch {attempt}/{attempts} via {rpc.split('//')[1].split('/')[0]}: "
                     f"gesendet {sig}")
            log.info(f"  Explorer: https://solscan.io/tx/{sig}")
            ok, status = confirm_tx(sig, rpc=rpc)
            if ok:
                log.info(f"✅ Bestätigt ({status}).")
                return {"ok": True, "dry": False, "signature": sig,
                        "status": status, "position": pos}
            last_err = status
            log.warning(f"Versuch {attempt} nicht bestätigt: {status}")
        except Exception as e:
            last_err = str(e)
            log.warning(f"Versuch {attempt} Fehler: {e}")
        time.sleep(2)

    return {"ok": False, "dry": False, "reason": last_err,
            "signature": last_sig, "position": pos}


def claim_position(owner: str, position_pubkey: str, keypair: Keypair,
                   send: bool = False, attempts: int = 4) -> dict:
    """
    Löst die Auszahlung einer GEWONNENEN, claimbaren Position ein.
    send=False -> Dry-Run (baut+signiert, sendet nicht).

    Idempotent: vor jedem Versuch wird die Position neu geprüft; ist sie bereits
    `claimed`, gilt der Claim als erfolgreich (kein Doppel-Claim). Pro Versuch
    frischer Blockhash + RPC-Rotation, analog sell_position().
    """
    pos = fetch_position_by_pubkey(owner, position_pubkey)
    if not pos:
        return {"ok": False, "reason": "Position nicht gefunden"}
    if pos.get("claimed"):
        return {"ok": True, "already": True, "status": "claimed", "position": pos}
    if not pos.get("claimable"):
        return {"ok": False, "reason": "Position nicht claimbar", "position": pos}

    payout = int(pos.get("payoutUsd", 0)) / 1e6
    log.info(f"Claim {position_pubkey[:10]}…  payout={payout:.4f} USDC")

    # Dry-Run: nur bauen + signieren
    if not send:
        tx_b64, meta = build_claim_tx(position_pubkey, owner)
        sign_close_tx(tx_b64, keypair)  # generisch: füllt unseren Slot, prüft Vollständigkeit
        log.info(f"Claim-Tx gebaut + signiert (blockhash {meta.get('blockhash','?')[:10]}…). DRY-RUN: nicht gesendet.")
        return {"ok": True, "dry": True, "position": pos}

    last_err, last_sig = None, None
    for attempt in range(1, attempts + 1):
        cur = fetch_position_by_pubkey(owner, position_pubkey)
        if cur is None:
            log.info("Position verschwunden -> als geclaimt gewertet (Erfolg).")
            return {"ok": True, "dry": False, "signature": last_sig,
                    "status": "gone", "position": pos}
        if cur.get("claimed"):
            log.info("Position bereits geclaimt -> Erfolg.")
            return {"ok": True, "dry": False, "signature": last_sig,
                    "status": "claimed", "position": pos}

        rpc = RPCS[(attempt - 1) % len(RPCS)]
        try:
            tx_b64, meta = build_claim_tx(position_pubkey, owner)  # frischer Blockhash
            signed = sign_close_tx(tx_b64, keypair)
            sig = send_tx(signed, rpc=rpc)
            last_sig = sig
            log.info(f"Claim-Versuch {attempt}/{attempts} via {rpc.split('//')[1].split('/')[0]}: "
                     f"gesendet {sig}")
            log.info(f"  Explorer: https://solscan.io/tx/{sig}")
            ok, status = confirm_tx(sig, rpc=rpc)
            if ok:
                log.info(f"✅ Claim bestätigt ({status}).")
                return {"ok": True, "dry": False, "signature": sig,
                        "status": status, "position": pos}
            last_err = status
            log.warning(f"Claim-Versuch {attempt} nicht bestätigt: {status}")
        except Exception as e:
            last_err = str(e)
            log.warning(f"Claim-Versuch {attempt} Fehler: {e}")
        time.sleep(2)

    return {"ok": False, "dry": False, "reason": last_err,
            "signature": last_sig, "position": pos}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        handlers=[logging.FileHandler("logs/jupiter_sell.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="POLY-1897148", help="Markt-ID (default Draw Germany-Ivory)")
    ap.add_argument("--claim", metavar="POSITION_PUBKEY",
                    help="CLAIM-Modus: Auszahlung dieser (claimbaren) Position einlösen statt verkaufen")
    ap.add_argument("--send", action="store_true", help="ECHT senden (sonst Dry-Run)")
    args = ap.parse_args()

    kp = load_keypair()
    owner = str(kp.pubkey())
    log.info("=" * 60)
    if args.claim:
        log.info(f"{'🔴 ECHTER CLAIM' if args.send else '🟡 DRY-RUN'}  |  Owner {owner[:10]}…  Position {args.claim[:10]}…")
        log.info("=" * 60)
        res = claim_position(owner, args.claim, kp, send=args.send)
    else:
        log.info(f"{'🔴 ECHTER VERKAUF' if args.send else '🟡 DRY-RUN'}  |  Owner {owner[:10]}…  Markt {args.market}")
        log.info("=" * 60)
        res = sell_position(owner, args.market, kp, send=args.send)
    log.info(f"Ergebnis: {json.dumps({k: v for k, v in res.items() if k != 'position'})}")


if __name__ == "__main__":
    main()
