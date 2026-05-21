#!/usr/bin/env python3
"""
Signal -> Alpaca Order Translator.

Liest today_signal.json (Hybrid SPY) + hyg_today_signal.json (HYG),
berechnet Soll-Allokation gegen Ist-Positionen und feuert Market-Orders
(notional, USD-basiert -> fractional shares moeglich).

Strategie: 50/50 Kapital-Split zwischen Hybrid und HYG.

Sicherheits-Guards:
  * Default Dry-Run (env BOERSENBOT_LIVE=1 setzt scharf)
  * Min-Rebalance-Threshold: 1% des Equity
  * Markt-Clock-Check (skip bei geschlossener Boerse)
  * Fehler-Mail (gleicher SMTP-Pfad wie Email-Report)

Cron-Slot empfohlen: 14:35 UTC (5 Min nach US-Open).
"""
import json
import os
import smtplib
import sys
import traceback
from datetime import datetime, timezone
from email.mime.text import MIMEText

import requests

ALPACA_KEY = "PK7C52Q5VZXZ5DDOIDCEEY7CKD"
ALPACA_SECRET = "BqgRvkRyUeanetTS8AEzvnTp5GMaPHcuWqFkCDjTgyLa"
ALPACA_BASE = "https://paper-api.alpaca.markets"

MAIL_HOST, MAIL_PORT = "mail.gmx.net", 587
MAIL_USER = "veit.luther@gmx.de"
MAIL_PASS = "Extaler00!"
MAIL_TO = "veit.luther@gmx.de"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_SIGNAL = os.path.join(SCRIPT_DIR, "today_signal.json")
HYG_SIGNAL = os.path.join(SCRIPT_DIR, "hyg_today_signal.json")
LOG_FILE = os.path.join(SCRIPT_DIR, "logs", "signal_to_orders.log")

HYBRID_SYMBOL = "SPY"
HYG_SYMBOL = "HYG"
HYBRID_BUCKET = 0.5          # 50% Kapital fuer Hybrid
HYG_BUCKET = 0.5             # 50% Kapital fuer HYG
MIN_REBALANCE_PCT = 0.01     # < 1% Equity Differenz -> skip
SIGNAL_MAX_AGE_HOURS = 36    # Signal-File darf nicht aelter als 36h sein

LIVE = os.environ.get("BOERSENBOT_LIVE") == "1"
FORCE_NO_CLOCK = os.environ.get("BOERSENBOT_FORCE") == "1"  # debug: skip clock-check

ALPACA_HEADERS = {
    "APCA-API-KEY-ID": ALPACA_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET,
}


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def get_account():
    r = requests.get(f"{ALPACA_BASE}/v2/account", headers=ALPACA_HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def get_clock():
    r = requests.get(f"{ALPACA_BASE}/v2/clock", headers=ALPACA_HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def get_position(symbol):
    r = requests.get(f"{ALPACA_BASE}/v2/positions/{symbol}", headers=ALPACA_HEADERS, timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def place_order(symbol, notional, side):
    """Market-Order, notional (USD), DAY-TIF."""
    body = {
        "symbol": symbol,
        "notional": f"{notional:.2f}",
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    r = requests.post(f"{ALPACA_BASE}/v2/orders", headers=ALPACA_HEADERS, json=body, timeout=20)
    if not r.ok:
        log(f"  ORDER FAILED: {symbol} {side} ${notional:.2f} -> {r.status_code} {r.text}")
        r.raise_for_status()
    o = r.json()
    log(f"  ORDER OK: id={o.get('id')} {symbol} {side} ${notional:.2f}")
    return o


def load_signal(path, max_age_h):
    with open(path, "r", encoding="utf-8") as f:
        sig = json.load(f)
    ts = sig.get("timestamp")
    if ts:
        try:
            sig_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            age_h = (datetime.now(timezone.utc) - sig_dt).total_seconds() / 3600
            if age_h > max_age_h:
                raise ValueError(f"Signal {path} ist {age_h:.1f}h alt (>{max_age_h}h)")
            log(f"  {os.path.basename(path)}: age {age_h:.1f}h, action={sig.get('action')}")
        except Exception as e:
            raise ValueError(f"Konnte Signal-Alter nicht pruefen: {e}")
    return sig


def hybrid_target_pct(sig):
    """Hybrid-Allokation 0.0-1.0 innerhalb des Buckets."""
    action = (sig.get("action") or "").upper()
    if action == "FLAT":
        return 0.0
    return float(sig.get("position_size") or 0)


def hyg_target_pct(sig):
    """HYG-Allokation 0.0-1.0 innerhalb des Buckets."""
    return float(sig.get("exposure_pct") or 0) / 100.0


def rebalance(symbol, target_usd, equity, dry_run):
    pos = get_position(symbol)
    current_usd = float(pos.get("market_value", 0)) if pos else 0.0
    diff = target_usd - current_usd
    pct_of_equity = abs(diff) / equity if equity else 0

    log(f"  {symbol}: ist=${current_usd:,.2f} soll=${target_usd:,.2f} "
        f"diff=${diff:+,.2f} ({pct_of_equity*100:.2f}% Equity)")

    if pct_of_equity < MIN_REBALANCE_PCT:
        log(f"  -> skip (unter Rebalance-Threshold {MIN_REBALANCE_PCT*100:.1f}%)")
        return {"action": "skip", "diff_usd": diff}

    side = "buy" if diff > 0 else "sell"
    notional = abs(diff)

    if side == "sell" and current_usd <= 0:
        log(f"  -> skip (SELL ohne Position)")
        return {"action": "skip_no_pos", "diff_usd": diff}

    # SELL darf nicht groesser sein als die Position
    if side == "sell" and notional > current_usd:
        notional = current_usd
        log(f"  -> SELL-Notional auf Position-Wert begrenzt: ${notional:,.2f}")

    if dry_run:
        log(f"  -> DRY-RUN: would {side.upper()} ${notional:,.2f} {symbol}")
        return {"action": "dry", "side": side, "notional": notional}

    order = place_order(symbol, notional, side)
    return {"action": "live", "side": side, "notional": notional, "order_id": order.get("id")}


def send_summary_mail(summary_lines, subject_prefix, is_error=False):
    body = "<html><body><pre style='font-family:Consolas,Menlo,monospace;font-size:13px'>"
    body += "\n".join(summary_lines)
    body += "</pre></body></html>"

    msg = MIMEText(body, "html", "utf-8")
    msg["Subject"] = (
        f"[Boersenbot] {subject_prefix} {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    msg["From"] = MAIL_USER
    msg["To"] = MAIL_TO

    try:
        with smtplib.SMTP(MAIL_HOST, MAIL_PORT, timeout=30) as s:
            s.ehlo()
            s.starttls()
            s.login(MAIL_USER, MAIL_PASS)
            s.send_message(msg)
        log("Summary-Mail versendet")
    except Exception as e:
        log(f"WARN: Summary-Mail fehlgeschlagen: {e}")


def main():
    dry_run = not LIVE
    mode = "LIVE" if not dry_run else "DRY-RUN"
    log(f"=== signal_to_orders start ({mode}) ===")
    summary = [f"Mode: {mode}", ""]

    try:
        clock = get_clock()
        is_open = bool(clock.get("is_open"))
        log(f"Markt offen: {is_open} (next_open={clock.get('next_open')}, next_close={clock.get('next_close')})")
        summary.append(f"Markt offen: {is_open}")

        if not is_open and not FORCE_NO_CLOCK and not dry_run:
            log("Markt geschlossen -> Abbruch (kein Live-Order)")
            summary.append("ABBRUCH: Markt geschlossen")
            send_summary_mail(summary, "Orders SKIPPED (Markt zu)")
            return

        account = get_account()
        equity = float(account.get("equity") or 0)
        log(f"Equity: ${equity:,.2f}, Cash: ${float(account.get('cash') or 0):,.2f}")
        summary.append(f"Equity: ${equity:,.2f}")
        summary.append("")

        hyb_sig = load_signal(HYBRID_SIGNAL, SIGNAL_MAX_AGE_HOURS)
        hyg_sig = load_signal(HYG_SIGNAL, SIGNAL_MAX_AGE_HOURS)

        hyb_pct = hybrid_target_pct(hyb_sig)
        hyg_pct = hyg_target_pct(hyg_sig)

        hyb_target_usd = equity * HYBRID_BUCKET * hyb_pct
        hyg_target_usd = equity * HYG_BUCKET * hyg_pct

        log(f"Hybrid: action={hyb_sig.get('action')} size={hyb_pct:.2f} "
            f"-> target ${hyb_target_usd:,.2f} ({HYBRID_SYMBOL})")
        log(f"HYG:    exposure={hyg_pct*100:.0f}% "
            f"-> target ${hyg_target_usd:,.2f} ({HYG_SYMBOL})")
        summary.append(f"Hybrid {HYBRID_SYMBOL}: action={hyb_sig.get('action')} size={hyb_pct:.2f} -> ${hyb_target_usd:,.2f}")
        summary.append(f"HYG    {HYG_SYMBOL}: exposure={hyg_pct*100:.0f}% -> ${hyg_target_usd:,.2f}")
        summary.append("")

        results = {}
        for sym, target in [(HYBRID_SYMBOL, hyb_target_usd), (HYG_SYMBOL, hyg_target_usd)]:
            r = rebalance(sym, target, equity, dry_run)
            results[sym] = r
            summary.append(f"{sym}: {r}")

        send_summary_mail(summary, f"Orders {mode}")
        log("=== signal_to_orders done ===")

    except Exception as e:
        log(f"ERROR: {e}\n{traceback.format_exc()}")
        summary.append("")
        summary.append("ERROR:")
        summary.append(traceback.format_exc())
        send_summary_mail(summary, "Orders FEHLER", is_error=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
