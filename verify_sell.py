#!/usr/bin/env python3
"""
verify_sell.py — Testet den echten Verkauf einer Jupiter-Position.

Ablauf:
  1. Findet deine offene NO-Draw-Position (POLY-1897148)
  2. Verkauft sie SOFORT (unabhängig vom Preis)
  3. Loggt Transaktion + Signatur

Sicherheit: Vor Ausführung prüfen, dass die Position wirklich die richtige ist!
"""

import argparse
import base64
import json
import logging
import sys
from pathlib import Path

import requests
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

HOT = Path("hot_wallet")
API_KEY_FILE = HOT / "jupiter_api_key.txt"
KEYPAIR_FILE = HOT / "keypair.json"

TRADE_API = "https://api.jup.ag/prediction/v1"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "verify_sell.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("verify_sell")


def load_secrets():
    api_key = API_KEY_FILE.read_text().strip()
    keypair = Keypair.from_bytes(bytes(json.loads(KEYPAIR_FILE.read_text())))
    return api_key, keypair


def find_position(owner_pubkey: str, market_id: str, api_key: str) -> str | None:
    """Findet deine offene Position im Markt, gibt positionPubkey zurück."""
    try:
        r = requests.get(
            f"{TRADE_API}/positions",
            params={"owner": owner_pubkey},
            headers={"x-api-key": api_key},
            timeout=10,
        )
        r.raise_for_status()
        positions = r.json()
        if isinstance(positions, dict):
            positions = positions.get("positions", [])

        log.info(f"Positionen gefunden: {len(positions)}")
        for pos in positions:
            if pos.get("marketId") == market_id:
                pos_pubkey = pos.get("positionPubkey") or pos.get("pubkey")
                log.info(f"✓ Position im Markt {market_id} gefunden: {pos_pubkey}")
                return pos_pubkey

        log.warning(f"Keine Position im Markt {market_id} gefunden.")
        return None
    except Exception as e:
        log.error(f"Fehler beim Abrufen der Positionen: {e}")
        return None


def place_sell_order(
    position_pubkey: str, market_id: str, api_key: str, keypair: Keypair
) -> str | None:
    """
    Verkauft die Position über die Jupiter-Trade-API.

    ⚠️ ECHTE TRANSAKTION — keine Rückgängigmachung möglich!
    """
    owner = str(keypair.pubkey())
    try:
        log.warning(f"VERKAUF wird ausgelöst für Position {position_pubkey}...")

        # 1) Order anlegen → unsignierte Transaktion
        r = requests.post(
            f"{TRADE_API}/orders",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "ownerPubkey": owner,
                "marketId": market_id,
                "isYes": False,   # NO-Seite
                "isBuy": False,   # verkaufen
                "depositMint": USDC_MINT,
            },
            timeout=15,
        )
        r.raise_for_status()
        order = r.json()
        tx_b64 = order.get("transaction")
        if not tx_b64:
            log.error(f"Keine Transaktion in Order-Antwort: {order}")
            return None

        log.info("Transaktion erhalten, signiere...")

        # 2) signieren
        raw = base64.b64decode(tx_b64)
        tx = VersionedTransaction.from_bytes(raw)
        signed = VersionedTransaction(tx.message, [keypair])

        log.info("Sende Transaktion an Solana RPC...")

        # 3) senden (Solana RPC)
        send = requests.post(
            "https://api.mainnet-beta.solana.com",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendTransaction",
                "params": [
                    base64.b64encode(bytes(signed)).decode(),
                    {"encoding": "base64", "skipPreflight": False},
                ],
            },
            timeout=30,
        )
        result = send.json()
        sig = result.get("result")
        if sig:
            log.info(f"✅ VERKAUF GESENDET!")
            log.info(f"   Signatur: {sig}")
            log.info(f"   Explorer: https://solscan.io/tx/{sig}")
            return sig

        error = result.get("error", {})
        log.error(f"Senden fehlgeschlagen: {error}")
        return None
    except Exception as e:
        log.error(f"Verkauf-Fehler: {e}", exc_info=True)
        return None


def main():
    ap = argparse.ArgumentParser(description="Verkauf einer Jupiter-Position (Test)")
    ap.add_argument(
        "--market",
        default="POLY-1897148",
        help="Draw-Markt-ID (Germany-Ivory), default POLY-1897148",
    )
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="MUSS gesetzt sein, um echten Verkauf auszuführen!",
    )
    args = ap.parse_args()

    if not args.confirm:
        log.error("SICHERHEIT: --confirm Flag MUSS gesetzt sein zum Verkauf!")
        log.error("  python verify_sell.py --confirm")
        sys.exit(1)

    api_key, keypair = load_secrets()
    owner = str(keypair.pubkey())

    log.info("=" * 60)
    log.info(f"VERKAUFS-TEST (ECHTE TRANSAKTION)")
    log.info(f"Owner: {owner}")
    log.info(f"Markt: {args.market}")
    log.info("=" * 60)

    # Position finden
    pos_pubkey = find_position(owner, args.market, api_key)
    if not pos_pubkey:
        log.error("Position nicht gefunden. Abbruch.")
        sys.exit(1)

    # Verkauf
    sig = place_sell_order(pos_pubkey, args.market, api_key, keypair)
    if sig:
        log.info(f"\n✅ Verkauf erfolgreich!")
        log.info(f"Warte ~20s auf Blockchain-Finalisierung...")
        import time
        time.sleep(20)
        log.info("Prüfe Status...")
    else:
        log.error("\n❌ Verkauf fehlgeschlagen!")
        sys.exit(1)


if __name__ == "__main__":
    main()
