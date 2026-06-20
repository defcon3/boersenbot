#!/usr/bin/env python3
"""
sell_autonomous.py — Vollautomatischer Verkauf einer Jupiter-Position.

Läuft kontinuierlich auf dem VPS:
  1. Pollt alle N Sekunden den Draw-Markt-Preis
  2. Bei Trigger (sellNo ≥ Zielpreis): verkauft SOFORT
  3. Signiert + sendet Transaktion autonom (Private Key auf VPS)
  4. Beendet sich nach Verkauf

⚠️ RISIKO: Private Key auf VPS. Nur die TRADING-WALLET (`4XxS…`),
   nicht die Phantom-Hauptwallet. Bei Server-Hack ist das Trade-Kapital weg.
   Aber: Vollautomatisch, keine Handy-Klicks nötig.

Aufruf:
  python sell_autonomous.py --entry 0.813 --event POLY-351748 --market POLY-1897148
"""

import argparse
import base64
import json
import logging
import sys
import time
from datetime import datetime, timezone
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

READ_API = "https://prediction-market-api.jup.ag/api/v1"
TRADE_API = "https://api.jup.ag/prediction/v1"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "sell_autonomous.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("sell_autonomous")


# ---------------------------------------------------------------------------
def load_secrets():
    api_key = API_KEY_FILE.read_text().strip()
    keypair = Keypair.from_bytes(bytes(json.loads(KEYPAIR_FILE.read_text())))
    return api_key, keypair


def fetch_market(event_id: str, market_id: str, api_key: str) -> dict | None:
    """Holt aktuellen Draw-Markt-Preis."""
    try:
        r = requests.get(
            f"{READ_API}/events/{event_id}",
            params={"includeAllMarkets": "true"},
            headers={"User-Agent": "Mozilla/5.0", "x-api-key": api_key},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        market = next(
            (m for m in data.get("markets", []) if m.get("marketId") == market_id),
            None,
        )
        if not market:
            return None
        p = market.get("pricing", {})
        return {
            "status": market.get("status"),
            "sell_no": p.get("sellNoPriceUsd", 0) / 1e6,
            "event_active": data.get("isActive", True),
        }
    except Exception as e:
        log.warning(f"Abruf fehlgeschlagen: {e}")
        return None


def place_sell_order_autonomous(
    market_id: str, api_key: str, keypair: Keypair
) -> str | None:
    """
    Verkauft NO im Draw-Markt über Jupiter-Trade-API.
    Signiert + sendet autonom (Private Key auf diesem Server).
    """
    owner = str(keypair.pubkey())
    try:
        log.warning(f"🔴 AUTONOMER VERKAUF wird ausgelöst...")

        # 1) Order anlegen → unsignierte Transaktion
        r = requests.post(
            f"{TRADE_API}/orders",
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json={
                "ownerPubkey": owner,
                "marketId": market_id,
                "isYes": False,  # NO-Seite
                "isBuy": False,  # verkaufen
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

        log.info("✓ Order-Transaktion erhalten. Signiere mit Keypair...")

        # 2) Mit Hot-Wallet signieren
        raw = base64.b64decode(tx_b64)
        tx = VersionedTransaction.from_bytes(raw)
        signed = VersionedTransaction(tx.message, [keypair])

        log.info("✓ Signiert. Sende an Solana Mainnet...")

        # 3) An Solana RPC senden
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
            log.info(f"✅ VERKAUF ERFOLGREICH GESENDET!")
            log.info(f"   Signatur: {sig}")
            log.info(f"   Explorer: https://solscan.io/tx/{sig}")
            return sig

        error = result.get("error", {})
        log.error(f"❌ Senden fehlgeschlagen: {error}")
        return None
    except Exception as e:
        log.error(f"❌ Verkauf-Fehler: {e}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
def run(args):
    api_key, keypair = load_secrets()
    owner = str(keypair.pubkey())
    target = args.entry * (1 + 0.10)  # +10% hard-coded

    log.info("=" * 70)
    log.info(f"🔴 AUTONOMER VERKAUF BOT")
    log.info(f"Event {args.event}  |  Draw-Markt {args.market}")
    log.info(f"Owner (Hot-Wallet): {owner}")
    log.info(f"Einstieg: {args.entry:.6f}  |  Verkauf ab: {target:.6f} (+10%)")
    log.info(f"Poll-Intervall: {args.interval}s")
    log.info(f"⚠️  Private Key wird AUTONOM zum Signieren genutzt!")
    log.info("=" * 70)

    fails = 0
    polls = 0
    while True:
        polls += 1
        m = fetch_market(args.event, args.market, api_key)
        if not m:
            fails += 1
            if fails >= 10:
                log.error(f"Zu viele Abruf-Fehler. Beende sicherheitshalber.")
                return
            log.warning(f"Abruf fehlgeschlagen (Fehler #{fails}), versuche erneut...")
            time.sleep(args.interval)
            continue

        fails = 0
        sell_no = m["sell_no"]
        pnl_pct = (sell_no / args.entry - 1) * 100

        if polls % 12 == 0:  # Alle 12 Polls loggen (= alle 60s bei 5s Intervall)
            log.info(
                f"[Poll #{polls}] sellNo={sell_no:.6f}  PnL={pnl_pct:+.2f}%  "
                f"Ziel={target:.6f}  Status={m['status']}"
            )

        # Markt geschlossen?
        if m["status"] not in ("open", "active") or not m["event_active"]:
            log.info(f"⚠️  Markt nicht mehr offen (status={m['status']}). Beende.")
            return

        # TRIGGER?
        if sell_no >= target:
            log.warning(f"🎯 TRIGGER! sellNo {sell_no:.6f} ≥ {target:.6f}")
            log.warning(f"   PnL: {pnl_pct:+.2f}%")

            sig = place_sell_order_autonomous(args.market, api_key, keypair)
            if sig:
                log.info(f"✅ Verkauf abgeschlossen!")
                log.info(f"   Warte 30s auf Blockchain-Finalisierung...")
                time.sleep(30)
                log.info(f"Bot beendet sich.")
                return
            else:
                log.error(f"❌ Verkauf fehlgeschlagen. Versuche beim nächsten Poll erneut...")
                # Weiter probieren

        time.sleep(args.interval)


def main():
    ap = argparse.ArgumentParser(
        description="Autonomer Verkaufs-Bot (Private Key signiert autonom)"
    )
    ap.add_argument("--entry", type=float, required=True, help="Einstiegspreis, z.B. 0.813")
    ap.add_argument("--event", default="POLY-351748", help="Event-ID")
    ap.add_argument("--market", default="POLY-1897148", help="Draw-Markt-ID")
    ap.add_argument("--interval", type=int, default=5, help="Poll-Intervall (Sekunden)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
