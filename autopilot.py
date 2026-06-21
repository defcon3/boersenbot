#!/usr/bin/env python3
"""
autopilot.py — Autonomer Take-Profit-Exit für Jupiter Prediction Markets.

Überwacht KONTINUIERLICH alle offenen Positionen der Hot-Wallet und verkauft
jede automatisch, sobald ihr Verkaufspreis +PROFIT über dem eigenen Einstieg
(avgPrice der Position) liegt. Nutzt die verifizierte Pipeline aus jupiter_sell.py.

- Kein manuelles --entry: der Einstieg kommt direkt aus der Position (avgPriceUsd).
- Variante A: verkauft beim ersten Erreichen von +PROFIT zum Marktpreis.
- KEIN Stop-Loss (bewusst): fällt kein Tor, läuft die Position bis Spielende.

Für den VPS gedacht (systemd, läuft 24/7, auch bei ausgeschaltetem PC).

Aufruf:
  python autopilot.py            # ECHT (verkauft autonom)
  python autopilot.py --dry      # Dry-Run (loggt, verkauft nicht)
  python autopilot.py --profit 0.10 --interval 20 --idle-interval 90

Rate-Limit (429) der öffentlichen Jupiter-API: adaptives Polling — bei OFFENER
Position schnell (--interval), sonst langsam (--idle-interval); bei 429
exponentielles Backoff (respektiert Retry-After).
"""

import argparse
import logging
import smtplib
import sys
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests

from jupiter_sell import load_keypair, sell_position, API

# Mail-Benachrichtigung (GMX, hardcoded wie im Projekt üblich)
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
    handlers=[
        logging.FileHandler(log_dir / "autopilot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("autopilot")


def notify(subject: str, html: str, text: str):
    """Schickt eine Benachrichtigungs-Mail. Fehler crashen den Bot NICHT."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = MAIL_USER
        msg["To"] = MAIL_TO
        msg.attach(MIMEText(text, "plain", "utf-8"))
        msg.attach(MIMEText(html, "html", "utf-8"))
        with smtplib.SMTP(MAIL_HOST, MAIL_PORT, timeout=30) as s:
            s.starttls()
            s.login(MAIL_USER, MAIL_PASS)
            s.sendmail(MAIL_USER, [MAIL_TO], msg.as_string())
        log.info(f"Mail gesendet: {subject}")
    except Exception as e:
        log.warning(f"Mail-Versand fehlgeschlagen: {e}")


def notify_sale(title, side, avg, sellp, pnl, contracts, sig):
    erloes = contracts * sellp
    win = pnl > 0
    color = "#2e7d32" if pnl >= 0 else "#c62828"
    link = f"https://solscan.io/tx/{sig}" if sig else "#"
    praise_html = (
        '<div style="margin-top:16px;padding:14px;background:#e8f5e9;border-radius:8px;'
        'text-align:center;font-size:18px;font-weight:800;color:#2e7d32;">'
        '🎉 Du bist der Geilste überhaupt.</div>'
    ) if win else ""
    html = f"""\
<div style="font-family:Segoe UI,Arial,sans-serif;max-width:480px;margin:auto;border-radius:12px;overflow:hidden;border:1px solid #eee;">
  <div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;text-align:center;color:#fff;">
    <div style="font-size:20px;font-weight:800;">✅ Position verkauft</div>
    <div style="font-size:14px;opacity:.9;">{title} &middot; {side}</div>
  </div>
  <div style="padding:20px;background:#fff;color:#333;font-size:15px;line-height:1.8;">
    Einstieg: <b>{avg:.3f}</b> USDC<br>
    Verkauf: <b>{sellp:.3f}</b> USDC<br>
    Kontrakte: <b>{contracts:.2f}</b> &rarr; Erlös ~<b>{erloes:.2f}</b> USDC<br>
    <span style="font-size:22px;font-weight:800;color:{color};">{pnl:+.1f}%</span>
    {praise_html}
    <div style="margin-top:14px;"><a href="{link}" style="color:#667eea;font-size:12px;">🔗 Transaktion (Solscan)</a></div>
  </div>
</div>"""
    text = (f"Position verkauft: {title} [{side}]\n"
            f"Einstieg {avg:.3f} -> Verkauf {sellp:.3f} USDC | {pnl:+.1f}%\n"
            f"Kontrakte {contracts:.2f}, Erloes ~{erloes:.2f} USDC\n"
            f"Tx: {link}")
    if win:
        text += "\n\nDu bist der Geilste ueberhaupt."
    notify(f"✅ Jupiter Bot: {title} verkauft ({pnl:+.1f}%)", html, text)


def notify_fail(title, reason):
    html = (f'<div style="font-family:Arial;color:#c62828;">'
            f'<b>⚠️ Verkauf fehlgeschlagen:</b> {title}<br>Grund: {reason}<br>'
            f'Der Bot versucht es weiter. Ggf. manuell prüfen.</div>')
    notify(f"⚠️ Jupiter Bot: Verkauf fehlgeschlagen ({title})",
           html, f"Verkauf fehlgeschlagen: {title}\nGrund: {reason}\nBot versucht weiter.")


class RateLimited(Exception):
    """429 von der Jupiter-API. retry_after = empfohlene Wartezeit (s) oder None."""
    def __init__(self, retry_after: float | None = None):
        super().__init__("rate limit exceeded")
        self.retry_after = retry_after


def get_open_positions(owner: str) -> list[dict]:
    """Alle offenen (nicht geschlossenen/geclaimten) Positionen der Wallet."""
    r = requests.get(f"{API}/positions", params={"ownerPubkey": owner}, timeout=12)
    if r.status_code == 429:
        ra = r.headers.get("Retry-After", "")
        raise RateLimited(float(ra) if ra.replace(".", "", 1).isdigit() else None)
    r.raise_for_status()
    out = []
    for p in r.json().get("data", []):
        try:
            contracts = float(p.get("contractsDecimal", 0) or 0)
        except (TypeError, ValueError):
            contracts = 0
        if contracts > 0 and not p.get("claimed"):
            out.append(p)
    return out


def run(args):
    kp = load_keypair()
    owner = str(kp.pubkey())

    log.info("=" * 68)
    log.info(f"AUTOPILOT  |  {'DRY-RUN' if args.dry else 'LIVE (verkauft autonom)'}")
    log.info(f"Wallet {owner}")
    log.info(f"Take-Profit: +{args.profit*100:.0f}%  |  Poll: {args.interval}s aktiv / "
             f"{args.idle_interval}s idle  |  kein Stop-Loss")
    log.info("=" * 68)

    MAX_BACKOFF = 300  # s
    fails = 0
    polls = 0
    sold_markets: set[str] = set()
    notified_fail: set[str] = set()
    while True:
        polls += 1
        try:
            positions = get_open_positions(owner)
            fails = 0
        except RateLimited as e:
            fails += 1
            # Retry-After respektieren, sonst exponentiell ab idle_interval
            wait = e.retry_after or min(args.idle_interval * 2 ** min(fails, 4), MAX_BACKOFF)
            log.warning(f"Rate-Limit (429) #{fails} — warte {wait:.0f}s")
            time.sleep(wait)
            continue
        except Exception as e:
            fails += 1
            wait = min(args.interval * 2 ** min(fails, 5), MAX_BACKOFF)
            log.warning(f"Positions-Abruf fehlgeschlagen ({fails}): {e} — warte {wait:.0f}s")
            time.sleep(wait)
            continue

        # Adaptiv: keine offene Position -> langsam pollen (schont Rate-Limit);
        # sobald eine Position läuft -> schnell pollen (Exit nicht verpassen).
        if not positions:
            if polls % 10 == 1:  # Heartbeat
                log.info(f"Keine offene Position — warte ({args.idle_interval}s).")
            time.sleep(args.idle_interval)
            continue

        for p in positions:
            mid = p.get("marketId")
            if mid in sold_markets:
                continue
            avg = int(p.get("avgPriceUsd", 0)) / 1e6
            sellp = int(p.get("sellPriceUsd", 0)) / 1e6
            if avg <= 0:
                continue
            try:
                contracts = float(p.get("contractsDecimal", 0) or 0)
            except (TypeError, ValueError):
                contracts = 0.0
            target = avg * (1 + args.profit)
            pnl = (sellp / avg - 1) * 100
            title = p.get("marketMetadata", {}).get("title", "?")
            side = "NO" if not p.get("isYes") else "YES"
            log.info(f"{title} [{side}] {mid}: Einstieg={avg:.3f} sell={sellp:.3f} "
                     f"PnL={pnl:+.1f}% Ziel≥{target:.3f}")

            if sellp >= target:
                log.warning(f"🎯 TRIGGER {title}: PnL {pnl:+.1f}% ≥ +{args.profit*100:.0f}%")
                if args.dry:
                    log.info("DRY-RUN: würde jetzt verkaufen (nichts gesendet).")
                else:
                    res = sell_position(owner, mid, kp, send=True)
                    if res.get("ok"):
                        log.info(f"✅ Verkauft: {title}  sig={res.get('signature')}  status={res.get('status')}")
                        sold_markets.add(mid)
                        notify_sale(title, side, avg, sellp, pnl, contracts, res.get("signature"))
                    else:
                        log.error(f"❌ Verkauf fehlgeschlagen für {title}: {res.get('reason')} — Retry nächster Poll.")
                        if mid not in notified_fail:
                            notify_fail(title, res.get("reason"))
                            notified_fail.add(mid)

        time.sleep(args.interval)


def main():
    ap = argparse.ArgumentParser(description="Autonomer Take-Profit-Exit (Jupiter Prediction)")
    ap.add_argument("--profit", type=float, default=0.10, help="Take-Profit-Schwelle (default 0.10 = 10%%)")
    ap.add_argument("--interval", type=int, default=20,
                    help="Poll-Intervall bei OFFENER Position, Sekunden (default 20)")
    ap.add_argument("--idle-interval", type=int, default=90,
                    help="Poll-Intervall OHNE offene Position, Sekunden (default 90)")
    ap.add_argument("--dry", action="store_true", help="Dry-Run: loggt, verkauft NICHT")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
