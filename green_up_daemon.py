#!/usr/bin/env python3
"""
green_up_daemon.py — persistenter Daemon für green_up.py: verwaltet EXPLIZIT
benannte Positionen (kein Auto-Hedge auf alles), platziert die Hedge-Limit-Order
sobald sie lockbar wird, überwacht sie bis zum Fill, und hält den geteilten
Centron-State (bb_GreenUpHedges) aktuell, den autopilot.py als Skip-Liste liest
(siehe green_up_state.py — Autopilot darf eine Position NICHT verkaufen, solange
für sie eine Hedge-Order offen ist oder bereits gefüllt wurde, sonst entsteht
eine ungewollte nackte Position auf der Hedge-Seite).

Design-Entscheidungen (2026-07-03, siehe Memory green-up-daemon-todo):
- Ein Wallet, Skip-Liste statt zweitem Wallet (kleinste Änderung, kein Workflow-
  Bruch — Nutzer kauft Leg 1 weiter normal ins Haupt-Wallet).
- NUR explizit per --add benannte Märkte (kein Blanket-Auto-Hedge — sonst würde
  z. B. der Langläufer POLY-2591183 unerwünscht gehedgt).
- Fee/Slippage-Check (2026-07-03, /history verifiziert am Birrell/Gibson-
  Roundtrip): BUY-Seite ist auf Jupiter fee-frei (feeUsd=null), nur der spätere
  VERKAUF trägt Fee. Green-up hält BEIDE Seiten bis zum gebührenfreien Claim —
  die Lock-Mathematik in green_up.compute_hedge() ist daher bereits netto exakt,
  keine Korrektur nötig.

CLI:
  python green_up_daemon.py --add POLY-2734938-0 --profit 0.03   # zur Verwaltung hinzufügen
  python green_up_daemon.py --remove POLY-2734938-0              # rausnehmen (storniert offene Order)
  python green_up_daemon.py --list                                # Status aller verwalteten Märkte
  python green_up_daemon.py --loop --interval 30                  # der eigentliche Daemon
  python green_up_daemon.py --loop --dry                          # Daemon im Dry-Run (nichts sendet)
"""

import argparse
import json
import logging
import smtplib
import sys
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests

import green_up_state
from green_up import compute_hedge
from jupiter_buy import build_order, execute_order, sign_owner_slot
from jupiter_sell import API, load_keypair

# Mail wie autopilot.py (GMX, hardcoded — Projektkonvention).
MAIL_HOST, MAIL_PORT = "mail.gmx.net", 587
MAIL_USER = MAIL_TO = "veit.luther@gmx.de"
MAIL_PASS = "Extaler00!"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "green_up_daemon.log", encoding="utf-8"),
             logging.StreamHandler()],
)
log = logging.getLogger("green_up_daemon")

_last_wait_log: dict[str, float] = {}   # Throttle fuer "wartet noch"-Zeilen


def notify(subject: str, text: str):
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = MAIL_USER
        msg["To"] = MAIL_TO
        msg.attach(MIMEText(text, "plain", "utf-8"))
        with smtplib.SMTP(MAIL_HOST, MAIL_PORT, timeout=30) as s:
            s.starttls()
            s.login(MAIL_USER, MAIL_PASS)
            s.sendmail(MAIL_USER, [MAIL_TO], msg.as_string())
        log.info(f"Mail gesendet: {subject}")
    except Exception as e:
        log.warning(f"Mail-Versand fehlgeschlagen: {e}")


# ---------------------------------------------------------------- Positions-Helfer

def get_position_side(owner, market_id, is_yes):
    """Anders als jupiter_sell.get_position() (side-agnostisch, gibt die ERSTE
    Position im Markt zurück) MUSS der Daemon nach dem Hedge-Fill BEIDE Seiten
    desselben Markts unterscheiden können -> eigener Fetch mit isYes-Filter."""
    r = requests.get(f"{API}/positions", params={"ownerPubkey": owner}, timeout=15)
    r.raise_for_status()
    for p in r.json().get("data", []):
        if (p.get("marketId") == market_id and bool(p.get("isYes")) == is_yes
                and str(p.get("contracts", "0")) not in ("0", "")):
            return p
    return None


def cancel_order(owner, kp, order_pubkey, send=True):
    """Storniert eine ruhende Limit-Order. Flow verifiziert 2026-06-24 (siehe Memory
    jupiter-prediction-bot): POST /orders/cancel liefert eine Memo+System-Tx (KEIN
    echtes On-Chain-CloseOrder) PLUS ein 'execution'-Objekt; die signierte Tx geht
    an /execute zusammen mit dem INNEREN execution.context (nicht 'execution' selbst!).
    Ein Jupiter-Keeper schließt das Order-Konto asynchron. Diese Reimplementierung
    ist frisch geschrieben (altes Skript nicht mehr im Repo) — vor Verlass auf den
    Auto-Cancel-Pfad (naked-risk-Fall unten) einmal live an einer echten kleinen
    Order gegenprüfen."""
    if not order_pubkey:
        return {"ok": False, "reason": "kein orderPubkey bekannt"}
    r = requests.post(f"{API}/orders/cancel",
                      json={"ownerPubkey": owner, "orderPubkey": order_pubkey}, timeout=20)
    if r.status_code >= 400:
        return {"ok": False, "reason": f"HTTP {r.status_code}: {r.text[:300]}"}
    j = r.json()
    tx_b64 = j.get("transaction")
    context = (j.get("execution") or {}).get("context")
    if not tx_b64 or not context:
        return {"ok": False, "reason": f"unerwartete /orders/cancel-Antwort: {json.dumps(j)[:300]}"}
    if not send:
        return {"ok": True, "dry": True}
    signed = sign_owner_slot(tx_b64, kp)
    resp = requests.post(f"{API}/execute",
                         json={"signedTransaction": signed, "context": context}, timeout=20)
    if resp.status_code >= 400:
        return {"ok": False, "reason": f"/execute HTTP {resp.status_code}: {resp.text[:300]}"}
    rj = resp.json()
    ok = str(rj.get("status", "")).lower() in ("cancel_requested", "success", "ok")
    return {"ok": ok, "resp": rj}


def log_throttled(market_id, msg, every=300):
    now = time.time()
    if now - _last_wait_log.get(market_id, 0) >= every:
        log.info(msg)
        _last_wait_log[market_id] = now


# ---------------------------------------------------------------- Kernlogik je Markt

def handle_watching(owner, kp, row, dry):
    mid = row["market_id"]
    pos = get_position_side(owner, mid, is_yes=True) or get_position_side(owner, mid, is_yes=False)
    if not pos:
        log.warning(f"{mid}: 'watching', aber keine Leg-1-Position mehr gefunden — cancelled.")
        green_up_state.update_hedge(mid, status="cancelled", note="Leg-1-Position verschwunden")
        return

    avg = int(pos.get("avgPriceUsd") or 0) / 1e6
    n_held = float(pos.get("contractsDecimal") or 0)
    held_is_yes = bool(pos.get("isYes"))
    hedge_is_yes = not held_is_yes
    title = pos.get("marketMetadata", {}).get("title", "?")

    r = compute_hedge(avg, n_held, row["profit_target"])
    if not r["ok"]:
        log_throttled(mid, f"{mid} ({title}): noch nicht lockbar ({r['reason']}, "
                          f"avg={avg:.3f}, Ziel-Profit={row['profit_target']:.2f}) — warte.")
        return

    tx_b64, meta, info = build_order(owner, mid, hedge_is_yes, r["deposit"], r["b"],
                                     skip_signing=dry)
    order = info.get("order", {})
    signed = sign_owner_slot(tx_b64, kp)
    if dry:
        log.info(f"[dry] {mid} ({title}): würde Hedge platzieren — {r['n_hedge']:g} Kontrakte "
                 f"@ Limit {r['b']:.2f}, Lock {r['locked']:+.2f}$")
        return
    resp = execute_order(signed, owner)
    ok = str(resp.get("status", "")).lower() in ("success", "ok", "confirmed", "pending")
    if not ok:
        log.error(f"{mid}: Hedge-Order fehlgeschlagen: {json.dumps(resp)[:300]} — Retry nächster Poll.")
        return
    green_up_state.update_hedge(mid, status="placed", hedge_is_yes=int(hedge_is_yes),
                                hedge_order_pk=order.get("orderPubkey"),
                                note=f"Limit {r['b']:.2f}, Lock {r['locked']:+.2f}$")
    log.warning(f"🔒 {mid} ({title}): Hedge platziert — {r['n_hedge']:g} Kontrakte @ Limit {r['b']:.2f}, "
               f"gesicherter Gewinn {r['locked']:+.2f}$ sobald gefüllt.")
    notify(f"🔒 Green-up: Hedge platziert ({title})",
          f"{title} ({mid})\nHedge-Limit @ {r['b']:.2f}, {r['n_hedge']:g} Kontrakte.\n"
          f"Gesicherter Gewinn bei Fill: {r['locked']:+.2f}$\n"
          f"Ruht im Orderbook, füllt nur wenn der Preis erreicht wird.")


def handle_placed(owner, kp, row, dry):
    mid = row["market_id"]
    hedge_is_yes = bool(row["hedge_is_yes"])
    hedge_pos = get_position_side(owner, mid, hedge_is_yes)
    if hedge_pos and float(hedge_pos.get("contractsDecimal") or 0) > 0:
        title = hedge_pos.get("marketMetadata", {}).get("title", "?")
        green_up_state.update_hedge(mid, status="locked",
                                    note=f"Hedge gefüllt: {hedge_pos.get('contractsDecimal')} Kontrakte")
        log.warning(f"✅ {mid} ({title}): GELOCKT — Hedge gefüllt, beide Seiten gehalten bis Claim.")
        notify(f"✅ Green-up GELOCKT ({title})",
              f"{title} ({mid})\nHedge gefüllt — beide Seiten der Position werden jetzt "
              f"gebührenfrei bis zur Auflösung gehalten (Auto-Claim übernimmt der Autopilot).")
        return

    leg1_pos = get_position_side(owner, mid, not hedge_is_yes)
    if not leg1_pos:
        log.error(f"🚨 {mid}: Leg-1 ist WEG, Hedge-Order steht noch offen -> NACKTES Risiko! "
                 f"Storniere Hedge-Order automatisch.")
        res = cancel_order(owner, kp, row.get("hedge_order_pk"), send=not dry)
        note = "Leg-1 verschwand, Hedge storniert" if res.get("ok") else \
              f"Leg-1 verschwand, Hedge-Storno FEHLGESCHLAGEN ({res.get('reason')}) — MANUELL PRÜFEN"
        green_up_state.update_hedge(mid, status="cancelled", note=note)
        notify(f"🚨 Green-up: Leg-1 verschwunden ({mid})",
              f"{mid}: Leg-1-Position ist weg, während die Hedge-Order noch offen stand.\n"
              f"Storno-Versuch: {'erfolgreich' if res.get('ok') else 'FEHLGESCHLAGEN — bitte manuell in Jupiter prüfen!'}\n"
              f"Reason: {res.get('reason', '-')}")
        return

    log_throttled(mid, f"{mid}: Hedge ruht noch (Leg 1 unverändert), warte auf Fill.")


def handle_locked(owner, row):
    mid = row["market_id"]
    hedge_is_yes = bool(row["hedge_is_yes"])
    leg1_pos = get_position_side(owner, mid, not hedge_is_yes)
    hedge_pos = get_position_side(owner, mid, hedge_is_yes)
    if not leg1_pos and not hedge_pos:
        green_up_state.update_hedge(mid, status="done", note="beide Seiten aufgelöst/geclaimt")
        log.warning(f"🎉 {mid}: Green-up abgeschlossen (Markt aufgelöst, Autopilot hat geclaimt).")


def poll_once(owner, kp, dry):
    try:
        rows = [r for r in green_up_state.list_hedges()
               if r["status"] in green_up_state.ACTIVE_STATUSES]
    except Exception as e:
        log.error(f"DB-Abruf fehlgeschlagen ({e}) — überspringe diesen Poll.")
        return
    if not rows:
        return
    for row in rows:
        try:
            if row["status"] == "watching":
                handle_watching(owner, kp, row, dry)
            elif row["status"] == "placed":
                handle_placed(owner, kp, row, dry)
            elif row["status"] == "locked":
                handle_locked(owner, row)
        except Exception as e:
            log.error(f"{row['market_id']}: Fehler im Poll ({e}) — weiter mit nächstem Markt.")


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="Green-up-Daemon: verwaltet explizit benannte Hedges.")
    ap.add_argument("--add", metavar="MARKET", help="Markt zur Verwaltung hinzufügen (mit --profit)")
    ap.add_argument("--profit", type=float, default=0.03, help="gesicherter Gewinn/Kontrakt (mit --add)")
    ap.add_argument("--remove", metavar="MARKET", help="Markt aus Verwaltung nehmen (storniert offene Hedge-Order)")
    ap.add_argument("--list", action="store_true", help="Status aller verwalteten Märkte zeigen")
    ap.add_argument("--loop", action="store_true", help="Daemon-Loop starten")
    ap.add_argument("--interval", type=int, default=30, help="Poll-Intervall Sekunden (default 30)")
    ap.add_argument("--dry", action="store_true", help="nichts senden (Orders/Storno nur simulieren)")
    args = ap.parse_args()

    green_up_state.ensure_table()

    if args.add:
        green_up_state.add_hedge(args.add, args.profit)
        print(f"✅ {args.add} zur Verwaltung hinzugefügt (Ziel-Profit {args.profit:.2f}$/Kontrakt, Status 'watching').")
        print("   Autopilot überspringt diesen Markt jetzt beim Verkauf. Daemon-Loop muss laufen, um die Hedge zu platzieren.")
        return

    if args.remove:
        row = green_up_state.get_hedge(args.remove)
        if row and row.get("hedge_order_pk") and row["status"] == "placed":
            kp = load_keypair()
            owner = str(kp.pubkey())
            res = cancel_order(owner, kp, row["hedge_order_pk"], send=not args.dry)
            print(f"Storno offene Hedge-Order: {'OK' if res.get('ok') else 'FEHLGESCHLAGEN — ' + str(res.get('reason'))}")
        n = green_up_state.remove_hedge(args.remove)
        print(f"✅ {args.remove} entfernt (Zeilen aktualisiert: {n}). Autopilot regelt diesen Markt wieder normal.")
        return

    if args.list:
        rows = green_up_state.list_hedges()
        if not rows:
            print("Keine verwalteten Märkte.")
            return
        for r in rows:
            print(f"{r['market_id']:>20}  {r['status']:<10}  Ziel {r['profit_target']:.2f}$  "
                 f"aktualisiert {r['updated_utc']}  {r.get('note') or ''}")
        return

    if args.loop:
        kp = load_keypair()
        owner = str(kp.pubkey())
        log.info("=" * 68)
        log.info(f"GREEN-UP-DAEMON  |  {'DRY-RUN' if args.dry else 'LIVE'}  |  Wallet {owner}")
        log.info(f"Poll-Intervall {args.interval}s  |  Nur explizit per --add benannte Märkte")
        log.info("=" * 68)
        while True:
            t0 = time.time()
            poll_once(owner, kp, args.dry)
            time.sleep(max(1.0, args.interval - (time.time() - t0)))
        return

    ap.print_help()


if __name__ == "__main__":
    main()
