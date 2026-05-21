#!/usr/bin/env python3
"""
Daily Email-Report fuer Boersenbot-Paper-Account.

Laeuft taeglich 06:00 UTC (nach US-Boersenschluss) per Cron auf dem NAS.

Inhalt:
  1. Transaktionen seit letztem Report (BUY/SELL, gefuellt)
  2. Aktive Positionen (qty, avg entry, current, unrealized P&L)
  3. Anzahl geschlossener Positionen + aggregierter Saldo (Equity-Delta)

Pure Python: requests + stdlib. Keine pandas/numpy/yfinance.
"""

import json
import os
import smtplib
import sys
import traceback
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
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
STATE_FILE = os.path.join(SCRIPT_DIR, ".last_report_state.json")
LOG_FILE = os.path.join(SCRIPT_DIR, "logs", "email_report.log")

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


def fetch_account():
    r = requests.get(f"{ALPACA_BASE}/v2/account", headers=ALPACA_HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_positions():
    r = requests.get(f"{ALPACA_BASE}/v2/positions", headers=ALPACA_HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_closed_orders(since_iso):
    """Gefuellte Orders seit since_iso (ISO-8601 UTC)."""
    params = {
        "status": "closed",
        "after": since_iso,
        "limit": 500,
        "direction": "asc",
        "nested": "false",
    }
    r = requests.get(f"{ALPACA_BASE}/v2/orders", headers=ALPACA_HEADERS, params=params, timeout=15)
    r.raise_for_status()
    orders = r.json()
    # Nur tatsaechlich gefuellte Orders (nicht canceled/rejected)
    return [o for o in orders if o.get("filled_at")]


def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"WARN: state file korrupt ({e}), starte ohne")
        return None


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def fmt_money(x, currency="USD"):
    try:
        return f"{float(x):,.2f} {currency}"
    except Exception:
        return f"{x} {currency}"


def fmt_pct(x):
    try:
        return f"{float(x) * 100:+.2f}%"
    except Exception:
        return str(x)


def aggregate_orders(orders):
    """Aggregiere BUY/SELL je Symbol."""
    by_side = {"buy": {"count": 0, "qty": 0.0, "value": 0.0},
               "sell": {"count": 0, "qty": 0.0, "value": 0.0}}
    sells = 0
    for o in orders:
        side = o.get("side", "").lower()
        if side not in by_side:
            continue
        qty = float(o.get("filled_qty") or 0)
        price = float(o.get("filled_avg_price") or 0)
        by_side[side]["count"] += 1
        by_side[side]["qty"] += qty
        by_side[side]["value"] += qty * price
        if side == "sell":
            sells += 1
    return by_side, sells


def build_html_report(account, positions, orders, last_state):
    now = datetime.now(timezone.utc)
    currency = account.get("currency", "USD")

    equity = float(account.get("equity") or 0)
    cash = float(account.get("cash") or 0)
    portfolio_value = float(account.get("portfolio_value") or 0)

    last_equity = None
    last_ts = None
    if last_state:
        last_equity = last_state.get("equity")
        last_ts = last_state.get("timestamp")

    if last_equity is not None:
        delta_equity = equity - float(last_equity)
        delta_pct = (delta_equity / float(last_equity)) if float(last_equity) else 0
        period_str = f"seit {last_ts}"
    else:
        delta_equity = 0
        delta_pct = 0
        period_str = "Erster Report (kein Vergleichszeitraum)"

    agg, closed_count = aggregate_orders(orders)

    css = """
    <style>
      body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
             color: #222; max-width: 760px; margin: 0 auto; padding: 16px; }
      h1 { font-size: 20px; border-bottom: 2px solid #333; padding-bottom: 6px; }
      h2 { font-size: 16px; margin-top: 24px; color: #444; }
      table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 13px; }
      th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }
      th { background: #f4f4f4; text-align: center; }
      td:first-child, th:first-child { text-align: left; }
      .pos { color: #0a7a1f; font-weight: 700; }
      .neg { color: #c41313; font-weight: 700; }
      .zero { color: #555; font-weight: 600; }
      .muted { color: #888; font-size: 12px; }
      .summary td { font-size: 14px; }
      .summary td:first-child { width: 55%; }
      .hero { margin: 14px 0 20px 0; padding: 18px 22px; border-radius: 10px;
              font-size: 22px; font-weight: 700; text-align: center;
              border: 2px solid; letter-spacing: 0.2px; }
      .hero.pos { background: #e6f7ea; border-color: #0a7a1f; color: #0a7a1f; }
      .hero.neg { background: #fde9e9; border-color: #c41313; color: #c41313; }
      .hero.zero { background: #f1f1f1; border-color: #888; color: #555; }
      .hero .label { display: block; font-size: 12px; font-weight: 600;
                     letter-spacing: 1px; text-transform: uppercase;
                     opacity: 0.75; margin-bottom: 4px; }
      .summary td.delta { font-size: 15px; font-weight: 700; }
      .summary td.delta.pos { background: #e6f7ea; }
      .summary td.delta.neg { background: #fde9e9; }
    </style>
    """

    def cls(v):
        try:
            return "pos" if float(v) > 0 else ("neg" if float(v) < 0 else "zero")
        except Exception:
            return "zero"

    def arrow(v):
        try:
            f = float(v)
            return "&#9650;" if f > 0 else ("&#9660;" if f < 0 else "&#9646;")
        except Exception:
            return ""

    delta_cls = cls(delta_equity)

    html = [f"<html><head>{css}</head><body>"]
    html.append(f"<h1>Boersenbot Daily Report &mdash; {now.strftime('%Y-%m-%d %H:%M UTC')}</h1>")
    html.append(f"<p class='muted'>Zeitraum: {period_str}</p>")

    html.append(
        f"<div class='hero {delta_cls}'>"
        f"<span class='label'>Saldo {period_str}</span>"
        f"{arrow(delta_equity)} {fmt_money(delta_equity, currency)} &nbsp;({fmt_pct(delta_pct)})"
        f"</div>"
    )

    html.append("<h2>Konto</h2>")
    html.append("<table class='summary'>")
    html.append(f"<tr><td>Equity</td><td>{fmt_money(equity, currency)}</td></tr>")
    html.append(f"<tr><td>Cash</td><td>{fmt_money(cash, currency)}</td></tr>")
    html.append(f"<tr><td>Portfolio Value</td><td>{fmt_money(portfolio_value, currency)}</td></tr>")
    html.append(
        f"<tr><td>Equity-Delta {period_str}</td>"
        f"<td class='delta {delta_cls}'>{arrow(delta_equity)} {fmt_money(delta_equity, currency)} ({fmt_pct(delta_pct)})</td></tr>"
    )
    html.append(f"<tr><td>Account-Status</td><td>{account.get('status')}</td></tr>")
    html.append("</table>")

    html.append("<h2>Transaktionen seit letztem Report</h2>")
    if not orders:
        html.append("<p class='muted'>Keine gefuellten Orders.</p>")
    else:
        html.append("<table>")
        html.append("<tr><th>Zeit (UTC)</th><th>Side</th><th>Symbol</th><th>Qty</th><th>Fill-Price</th><th>Value</th></tr>")
        for o in orders:
            qty = float(o.get("filled_qty") or 0)
            price = float(o.get("filled_avg_price") or 0)
            value = qty * price
            html.append(
                f"<tr><td>{(o.get('filled_at') or '')[:19]}</td>"
                f"<td>{o.get('side', '').upper()}</td>"
                f"<td>{o.get('symbol')}</td>"
                f"<td>{qty:g}</td>"
                f"<td>{fmt_money(price, currency)}</td>"
                f"<td>{fmt_money(value, currency)}</td></tr>"
            )
        html.append("</table>")
        html.append(
            f"<p class='muted'>Aggregat: "
            f"BUY {agg['buy']['count']} Orders, {agg['buy']['qty']:g} Shares, "
            f"{fmt_money(agg['buy']['value'], currency)} &middot; "
            f"SELL {agg['sell']['count']} Orders, {agg['sell']['qty']:g} Shares, "
            f"{fmt_money(agg['sell']['value'], currency)}</p>"
        )

    html.append("<h2>Aktive Positionen</h2>")
    if not positions:
        html.append("<p class='muted'>Keine offenen Positionen.</p>")
    else:
        html.append("<table>")
        html.append("<tr><th>Symbol</th><th>Qty</th><th>Entry</th><th>Current</th><th>Market Value</th><th>Unrealized P&amp;L</th><th>%</th></tr>")
        for p in positions:
            qty = float(p.get("qty") or 0)
            entry = float(p.get("avg_entry_price") or 0)
            cur = float(p.get("current_price") or 0)
            mv = float(p.get("market_value") or 0)
            upl = float(p.get("unrealized_pl") or 0)
            uplpc = float(p.get("unrealized_plpc") or 0)
            html.append(
                f"<tr><td>{p.get('symbol')}</td>"
                f"<td>{qty:g}</td>"
                f"<td>{fmt_money(entry, currency)}</td>"
                f"<td>{fmt_money(cur, currency)}</td>"
                f"<td>{fmt_money(mv, currency)}</td>"
                f"<td class='{cls(upl)}'>{fmt_money(upl, currency)}</td>"
                f"<td class='{cls(upl)}'>{fmt_pct(uplpc)}</td></tr>"
            )
        html.append("</table>")

    html.append("<h2>Geschlossene Positionen</h2>")
    html.append(
        f"<p>Anzahl SELL-Orders im Zeitraum: <b>{closed_count}</b></p>"
        f"<div class='hero {delta_cls}'>"
        f"<span class='label'>Aggregierter Saldo (Equity-Delta)</span>"
        f"{arrow(delta_equity)} {fmt_money(delta_equity, currency)} &nbsp;({fmt_pct(delta_pct)})"
        f"</div>"
    )

    html.append(f"<p class='muted'>Account {account.get('account_number', '')} &middot; Paper-Trading &middot; Quelle: Alpaca API</p>")
    html.append("</body></html>")
    return "".join(html)


def send_email(subject, html_body):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = MAIL_USER
    msg["To"] = MAIL_TO
    msg.attach(MIMEText("HTML-Mail. Bitte HTML-faehigen Client nutzen.", "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(MAIL_HOST, MAIL_PORT, timeout=30) as s:
        s.ehlo()
        s.starttls()
        s.login(MAIL_USER, MAIL_PASS)
        s.send_message(msg)


def main():
    log("=== Email-Report run start ===")
    try:
        state = load_state()
        if state and state.get("timestamp"):
            since = state["timestamp"]
        else:
            # Erster Lauf: 7 Tage zurueck (genug fuer Frischbestand)
            since = (datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)).isoformat()
            since = since.replace("+00:00", "Z")

        log(f"since={since}")

        account = fetch_account()
        positions = fetch_positions()
        orders = fetch_closed_orders(since)
        log(f"account.equity={account.get('equity')} positions={len(positions)} orders={len(orders)}")

        html = build_html_report(account, positions, orders, state)

        now = datetime.now(timezone.utc)
        subject = (
            f"[Boersenbot] Daily Report {now.strftime('%Y-%m-%d')} "
            f"({len(orders)} Trades, {len(positions)} Positionen)"
        )
        send_email(subject, html)
        log("Mail versendet")

        new_state = {
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "equity": float(account.get("equity") or 0),
            "cash": float(account.get("cash") or 0),
            "portfolio_value": float(account.get("portfolio_value") or 0),
        }
        save_state(new_state)
        log(f"state gespeichert: {new_state}")

    except Exception as e:
        log(f"ERROR: {e}\n{traceback.format_exc()}")
        try:
            err_html = (
                f"<html><body><h2>Email-Report FAILED</h2>"
                f"<pre>{traceback.format_exc()}</pre></body></html>"
            )
            send_email(f"[Boersenbot] REPORT FEHLER {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", err_html)
        except Exception as e2:
            log(f"konnte auch Fehler-Mail nicht versenden: {e2}")
        sys.exit(1)

    log("=== Email-Report run done ===")


if __name__ == "__main__":
    main()
