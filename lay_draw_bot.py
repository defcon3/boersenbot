#!/usr/bin/env python3
"""
Lay-the-Draw Bot für Jupiter Prediction Markets (Variante A).

Strategie (vom Nutzer festgelegt):
  - Du hast NO im Draw-Markt gekauft (= gegen das Unentschieden gewettet).
  - KEIN Stop-Loss: fällt kein Führungstor, läuft die Position bis Spielende
    und verfällt ggf. (−100 %) — bewusst akzeptiert.
  - Take-Profit: sobald der Verkaufskurs (sellNo) ≥ Einstieg × (1 + profit)
    liegt, wird ZUM AKTUELLEN MARKTPREIS verkauft. Der +10 %-Wert ist nur der
    Auslöser; steht der Kurs höher (spätes Tor), wird der höhere Kurs erzielt.

SICHERHEIT:
  - Dry-Run ist Standard. Echtes Verkaufen NUR mit --live.
  - State-File verhindert Doppelverkauf.
  - Der Signing-/Sende-Teil (place_sell_order) ist klar isoliert und VOR dem
    ersten Live-Einsatz separat zu verifizieren (verify_sell.py).

Aufruf-Beispiele:
  Dry-Run (sicher, testet Polling + Trigger):
    python lay_draw_bot.py --entry 0.80
  Trigger-Mechanik sofort sehen (künstlich niedriger Entry):
    python lay_draw_bot.py --entry 0.70
  Scharf (erst nach Verifikation!):
    python lay_draw_bot.py --entry 0.80 --live
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---- UTF-8 Konsole (Windows) ----
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---- Pfade / Konstanten ----
HOT = Path("hot_wallet")
API_KEY_FILE = HOT / "jupiter_api_key.txt"
KEYPAIR_FILE = HOT / "keypair.json"

READ_API = "https://prediction-market-api.jup.ag/api/v1"   # verifiziert (Sniffer)
# Trade-/Order-API laut Jupiter-Doku — VOR Live-Einsatz verifizieren:
TRADE_API = "https://api.jup.ag/prediction/v1"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

# Defaults: Germany vs. Côte d'Ivoire, Draw-Markt
DEFAULT_EVENT = "POLY-351748"
DEFAULT_MARKET = "POLY-1897148"

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "lay_draw_bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("lay_draw_bot")


# ---------------------------------------------------------------------------
# Laden von Secrets
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    if not API_KEY_FILE.exists():
        log.error(f"API-Key fehlt: {API_KEY_FILE}")
        sys.exit(1)
    return API_KEY_FILE.read_text(encoding="utf-8").strip()


def load_keypair():
    """Erst beim Live-Verkauf nötig. Gibt solders-Keypair zurück."""
    from solders.keypair import Keypair  # lazy import
    if not KEYPAIR_FILE.exists():
        log.error(f"Keypair fehlt: {KEYPAIR_FILE}")
        sys.exit(1)
    return Keypair.from_bytes(bytes(json.loads(KEYPAIR_FILE.read_text())))


# ---------------------------------------------------------------------------
# Marktdaten lesen
# ---------------------------------------------------------------------------
def fetch_market(event_id: str, market_id: str, api_key: str) -> dict | None:
    """
    Holt den Event und gibt den gewünschten Markt-Datensatz zurück
    (inkl. pricing + Event-Status). None bei Fehler.
    """
    try:
        r = requests.get(
            f"{READ_API}/events/{event_id}",
            params={"includeAllMarkets": "true"},
            headers={"User-Agent": "Mozilla/5.0", "x-api-key": api_key},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"Abruf fehlgeschlagen: {e}")
        return None

    market = next((m for m in data.get("markets", []) if m.get("marketId") == market_id), None)
    if not market:
        log.warning(f"Markt {market_id} nicht im Event gefunden")
        return None

    p = market.get("pricing", {})
    return {
        "title": market.get("title", "?"),
        "status": market.get("status", "?"),
        "event_active": data.get("isActive", True),
        "buy_no": p.get("buyNoPriceUsd", 0) / 1e6,
        "sell_no": p.get("sellNoPriceUsd", 0) / 1e6,
        "buy_yes": p.get("buyYesPriceUsd", 0) / 1e6,
        "sell_yes": p.get("sellYesPriceUsd", 0) / 1e6,
        "live_score": data.get("liveScore", {}),
    }


# ---------------------------------------------------------------------------
# Verkauf (LIVE) — isoliert, VOR Einsatz separat verifizieren!
# ---------------------------------------------------------------------------
def place_sell_order(market_id: str, api_key: str, keypair) -> str | None:
    """
    Verkauft die NO-Position über die Jupiter-Trade-API.

    ⚠️ NICHT GEGEN DIE ECHTE API VERIFIZIERT. Ablauf laut Doku:
      1. POST /orders  (isYes=false, isBuy=false)  -> unsignierte Base64-Tx
      2. Tx mit Hot-Wallet signieren (solders VersionedTransaction)
      3. an Solana senden (sendTransaction)
    Vor Live-Einsatz mit verify_sell.py / kleinem Testbetrag prüfen.
    """
    from solders.transaction import VersionedTransaction
    from solders.message import to_bytes_versioned
    import base64

    owner = str(keypair.pubkey())
    try:
        # 1) Order anlegen -> unsignierte Transaktion
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

        # 2) signieren
        raw = base64.b64decode(tx_b64)
        tx = VersionedTransaction.from_bytes(raw)
        signed = VersionedTransaction(tx.message, [keypair])

        # 3) senden (Solana RPC)
        send = requests.post(
            "https://api.mainnet-beta.solana.com",
            json={
                "jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
                "params": [base64.b64encode(bytes(signed)).decode(),
                           {"encoding": "base64", "skipPreflight": False}],
            },
            timeout=20,
        )
        result = send.json()
        sig = result.get("result")
        if sig:
            log.info(f"✅ Verkauf gesendet. Signatur: {sig}")
            return sig
        log.error(f"Senden fehlgeschlagen: {result}")
        return None
    except Exception as e:
        log.error(f"Verkauf-Fehler: {e}")
        return None


# ---------------------------------------------------------------------------
# State (Idempotenz)
# ---------------------------------------------------------------------------
def state_path(market_id: str) -> Path:
    return HOT / f"bot_state_{market_id}.json"


def load_state(market_id: str) -> dict:
    p = state_path(market_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"sold": False, "signature": None}


def save_state(market_id: str, state: dict):
    state_path(market_id).write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Hauptschleife
# ---------------------------------------------------------------------------
def run(args):
    api_key = load_api_key()
    target = args.entry * (1 + args.profit)
    mode = "🔴 LIVE" if args.live else "🟡 DRY-RUN (kein echter Verkauf)"

    log.info("=" * 64)
    log.info(f"Lay-the-Draw Bot  |  {mode}")
    log.info(f"Event {args.event}  Markt {args.market}")
    log.info(f"Einstieg (buyNo): {args.entry:.3f}  |  Take-Profit ab sellNo ≥ {target:.3f} (+{args.profit*100:.0f}%)")
    log.info(f"Poll-Intervall: {args.interval}s  |  KEIN Stop-Loss")
    log.info("=" * 64)

    state = load_state(args.market)
    if state.get("sold"):
        log.info(f"Laut State bereits verkauft (sig={state.get('signature')}). Beende.")
        return

    keypair = load_keypair() if args.live else None
    if args.live:
        log.warning("LIVE-Modus: echter Verkauf bei Trigger!")

    fails = 0
    while True:
        m = fetch_market(args.event, args.market, api_key)
        if not m:
            fails += 1
            if fails >= 10:
                log.error("Zu viele Abruf-Fehler hintereinander. Beende sicherheitshalber.")
                return
            time.sleep(args.interval)
            continue
        fails = 0

        sell_no = m["sell_no"]
        pnl_pct = (sell_no / args.entry - 1) * 100
        ls = m["live_score"]
        score = f"{ls.get('homeTeam','?')} {ls.get('score','?')} {ls.get('awayTeam','?')} ({ls.get('elapsed','?')}')" if ls else "—"
        log.info(f"sellNo={sell_no:.3f}  PnL={pnl_pct:+.1f}%  Ziel≥{target:.3f}  Status={m['status']}  Score: {score}")

        # Marktschluss?
        if m["status"] not in ("open", "active") or not m["event_active"]:
            log.info(f"Markt nicht mehr offen (status={m['status']}). Beende ohne Verkauf.")
            return

        # Trigger?
        if sell_no >= target:
            log.info(f"🎯 TRIGGER erreicht: sellNo {sell_no:.3f} ≥ {target:.3f} (PnL {pnl_pct:+.1f}%)")
            if args.live:
                sig = place_sell_order(args.market, api_key, keypair)
                if sig:
                    state.update({"sold": True, "signature": sig,
                                  "sold_at": datetime.now(timezone.utc).isoformat(),
                                  "sell_price": sell_no})
                    save_state(args.market, state)
                    log.info("Verkauf abgeschlossen. Bot beendet sich.")
                    return
                else:
                    log.error("Verkauf fehlgeschlagen — versuche beim nächsten Poll erneut.")
            else:
                log.info("🟡 DRY-RUN: WÜRDE JETZT VERKAUFEN (es wird nichts gesendet).")
                # Im Dry-Run weiterlaufen, um das Verhalten über die Zeit zu sehen.

        time.sleep(args.interval)


def main():
    ap = argparse.ArgumentParser(description="Lay-the-Draw Bot (Jupiter Prediction Markets)")
    ap.add_argument("--entry", type=float, required=True, help="Einstiegspreis (buyNo), z.B. 0.80")
    ap.add_argument("--profit", type=float, default=0.10, help="Take-Profit-Schwelle, default 0.10 (=10%%)")
    ap.add_argument("--interval", type=int, default=5, help="Poll-Intervall in Sekunden, default 5")
    ap.add_argument("--event", default=DEFAULT_EVENT, help=f"Event-ID, default {DEFAULT_EVENT}")
    ap.add_argument("--market", default=DEFAULT_MARKET, help=f"Draw-Markt-ID, default {DEFAULT_MARKET}")
    ap.add_argument("--live", action="store_true", help="ECHTER Verkauf (sonst Dry-Run)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
