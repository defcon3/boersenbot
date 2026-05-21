#!/usr/bin/env python3
"""
Random-5 Crossover-Bot.

Liest random5_universe.json (5 fixe S&P-500-Aktien).
Berechnet pro Aktie MA20/MA50 aus Alpaca Bars (IEX-Feed).
Soll-Position:
  * MA20 > MA50 -> +qty (LONG, $500 Marktwert)
  * MA20 < MA50 -> -qty (SHORT, $500 Marktwert)
Differenz zu Ist-Position als Market-Order. Vorzeichen-Flip: 2 Orders (close + open).

Sicherheits-Guards:
  * Default DRY-RUN, scharf via BOERSENBOT_LIVE=1
  * Markt-Clock-Check
  * Summary-Mail pro Lauf

Pure Python: requests + stdlib.
"""
import json
import os
import smtplib
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

import requests

ALPACA_KEY = "PK7C52Q5VZXZ5DDOIDCEEY7CKD"
ALPACA_SECRET = "BqgRvkRyUeanetTS8AEzvnTp5GMaPHcuWqFkCDjTgyLa"
ALPACA_TRADE = "https://paper-api.alpaca.markets"
ALPACA_DATA = "https://data.alpaca.markets"

MAIL_HOST, MAIL_PORT = "mail.gmx.net", 587
MAIL_USER = "veit.luther@gmx.de"
MAIL_PASS = "Extaler00!"
MAIL_TO = "veit.luther@gmx.de"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_FILE = os.path.join(SCRIPT_DIR, "random5_universe.json")
LOG_FILE = os.path.join(SCRIPT_DIR, "logs", "random5_crossover.log")

POSITION_SIZE_USD = 500    # Marktwert pro Position (long oder short)
MA_FAST = 20
MA_SLOW = 50
FILL_POLL_SECONDS = 3
FILL_POLL_MAX = 10

LIVE = os.environ.get("BOERSENBOT_LIVE") == "1"
FORCE_NO_CLOCK = os.environ.get("BOERSENBOT_FORCE") == "1"

HDR = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def get_clock():
    r = requests.get(f"{ALPACA_TRADE}/v2/clock", headers=HDR, timeout=15)
    r.raise_for_status()
    return r.json()


def get_account():
    r = requests.get(f"{ALPACA_TRADE}/v2/account", headers=HDR, timeout=15)
    r.raise_for_status()
    return r.json()


def get_position_qty(symbol):
    r = requests.get(f"{ALPACA_TRADE}/v2/positions/{symbol}", headers=HDR, timeout=15)
    if r.status_code == 404:
        return 0
    r.raise_for_status()
    p = r.json()
    return float(p.get("qty") or 0)  # bei short ist qty negativ


def get_bars(symbol, days=120):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    params = {
        "timeframe": "1Day",
        "start": start.isoformat(),
        "limit": 1000,
        "feed": "iex",
        "adjustment": "split",
    }
    r = requests.get(f"{ALPACA_DATA}/v2/stocks/{symbol}/bars", headers=HDR, params=params, timeout=20)
    r.raise_for_status()
    bars = r.json().get("bars", [])
    return [(b["t"], float(b["c"])) for b in bars]


def sma(values, n):
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def compute_signal(symbol):
    """Liefert (soll_sign, last_close, ma_fast, ma_slow). soll_sign: +1, -1, 0 (data missing)."""
    bars = get_bars(symbol, days=120)
    closes = [c for _, c in bars]
    if len(closes) < MA_SLOW + 1:
        log(f"  {symbol}: zu wenige Bars ({len(closes)})")
        return 0, None, None, None
    last = closes[-1]
    mf = sma(closes, MA_FAST)
    ms = sma(closes, MA_SLOW)
    sign = 1 if mf > ms else -1
    return sign, last, mf, ms


def place_order(symbol, qty, side, dry_run):
    body = {
        "symbol": symbol,
        "qty": str(int(qty)),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    if dry_run:
        log(f"    DRY: would {side.upper()} {qty} {symbol}")
        return {"id": "dry", "status": "dry", "side": side, "qty": qty}

    r = requests.post(f"{ALPACA_TRADE}/v2/orders", headers=HDR, json=body, timeout=20)
    if not r.ok:
        log(f"    ORDER FAILED: {symbol} {side} {qty} -> {r.status_code} {r.text}")
        raise RuntimeError(f"order failed: {r.text}")
    o = r.json()
    log(f"    ORDER OK: id={o.get('id')[:8]}... {symbol} {side} {qty}")
    return o


def wait_filled(order_id):
    if order_id == "dry":
        return True
    for _ in range(FILL_POLL_MAX):
        r = requests.get(f"{ALPACA_TRADE}/v2/orders/{order_id}", headers=HDR, timeout=15)
        if r.ok:
            status = r.json().get("status")
            if status in ("filled", "partially_filled"):
                return True
            if status in ("canceled", "rejected", "expired"):
                log(f"    Order {order_id[:8]}... endete in Status {status}")
                return False
        time.sleep(FILL_POLL_SECONDS)
    log(f"    Order {order_id[:8]}... nicht innerhalb {FILL_POLL_SECONDS*FILL_POLL_MAX}s gefuellt")
    return False


def reconcile(symbol, soll_sign, last_close, dry_run):
    """Bringt Ist-Position auf Soll-Sign mit qty so dass |qty*last_close| ~ POSITION_SIZE_USD."""
    if not last_close or last_close <= 0:
        return {"action": "skip_no_price"}
    target_qty_abs = int(POSITION_SIZE_USD // last_close)
    if target_qty_abs < 1:
        return {"action": "skip_too_expensive", "price": last_close}
    target_qty = soll_sign * target_qty_abs
    ist_qty = get_position_qty(symbol)
    log(f"  {symbol}: ist={ist_qty:+g} soll={target_qty:+g} (price=${last_close:.2f})")

    if int(ist_qty) == target_qty:
        return {"action": "skip_already_aligned", "qty": target_qty}

    same_sign = (ist_qty > 0 and target_qty > 0) or (ist_qty < 0 and target_qty < 0)

    if ist_qty == 0:
        # neu eroeffnen
        side = "buy" if target_qty > 0 else "sell"  # sell = short open
        o = place_order(symbol, abs(target_qty), side, dry_run)
        return {"action": "open", "side": side, "qty": abs(target_qty), "order": o.get("id")}

    if same_sign:
        # adjusten in gleiche Richtung
        diff = target_qty - ist_qty
        side = "buy" if diff > 0 else "sell"
        o = place_order(symbol, abs(diff), side, dry_run)
        return {"action": "adjust", "side": side, "qty": abs(diff), "order": o.get("id")}

    # Vorzeichen-Flip: erst close (qty=|ist|), dann open (qty=|soll|)
    close_side = "sell" if ist_qty > 0 else "buy"
    open_side = "buy" if target_qty > 0 else "sell"
    log(f"  {symbol}: FLIP {ist_qty:+g} -> {target_qty:+g}")
    o1 = place_order(symbol, abs(int(ist_qty)), close_side, dry_run)
    if not dry_run:
        if not wait_filled(o1.get("id")):
            return {"action": "flip_close_failed", "close": o1.get("id")}
    o2 = place_order(symbol, abs(target_qty), open_side, dry_run)
    return {
        "action": "flip",
        "close": {"side": close_side, "qty": abs(int(ist_qty)), "order": o1.get("id")},
        "open": {"side": open_side, "qty": abs(target_qty), "order": o2.get("id")},
    }


def send_summary(lines, subject_suffix):
    body = "<html><body><pre style='font-family:Consolas,Menlo,monospace;font-size:13px'>"
    body += "\n".join(lines)
    body += "</pre></body></html>"
    msg = MIMEText(body, "html", "utf-8")
    msg["Subject"] = f"[Boersenbot] Random5 {subject_suffix} {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
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
    log(f"=== random5_crossover start ({mode}) ===")
    summary = [f"Mode: {mode}", ""]

    try:
        with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
            universe = json.load(f)
        symbols = universe["symbols"]
        log(f"Universum: {symbols}")
        summary.append(f"Universum: {symbols}")

        clock = get_clock()
        is_open = bool(clock.get("is_open"))
        log(f"Markt offen: {is_open}")
        summary.append(f"Markt offen: {is_open}")
        if not is_open and not FORCE_NO_CLOCK and not dry_run:
            log("Markt geschlossen -> Abbruch")
            summary.append("ABBRUCH: Markt geschlossen")
            send_summary(summary, "SKIPPED (Markt zu)")
            return

        account = get_account()
        equity = float(account.get("equity") or 0)
        summary.append(f"Equity: ${equity:,.2f}, Buying-Power: ${float(account.get('buying_power') or 0):,.2f}")
        summary.append("")

        for sym in symbols:
            sign, last, mf, ms = compute_signal(sym)
            if sign == 0:
                summary.append(f"{sym}: SKIP (data)")
                continue
            direction = "LONG" if sign > 0 else "SHORT"
            spread = (mf - ms) / ms * 100 if ms else 0
            log(f"  {sym}: MA20={mf:.2f} MA50={ms:.2f} ({spread:+.2f}%) -> {direction}")
            result = reconcile(sym, sign, last, dry_run)
            summary.append(
                f"{sym}: MA20={mf:.2f} MA50={ms:.2f} ({spread:+.2f}%) "
                f"price=${last:.2f} -> {direction} :: {result.get('action')}"
            )

        send_summary(summary, mode)
        log("=== random5_crossover done ===")
    except Exception as e:
        log(f"ERROR: {e}\n{traceback.format_exc()}")
        summary.append("")
        summary.append("ERROR:")
        summary.append(traceback.format_exc())
        send_summary(summary, "FEHLER")
        sys.exit(1)


if __name__ == "__main__":
    main()
