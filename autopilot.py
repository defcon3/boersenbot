#!/usr/bin/env python3
"""
autopilot.py — Autonomer Take-Profit-Exit für Jupiter Prediction Markets.

Überwacht KONTINUIERLICH alle offenen Positionen der Hot-Wallet und verkauft
jede automatisch, sobald ihr Verkaufspreis +PROFIT über dem eigenen Einstieg
(avgPrice der Position) liegt. Nutzt die verifizierte Pipeline aus jupiter_sell.py.

- Kein manuelles --entry: der Einstieg kommt direkt aus der Position (avgPriceUsd).
- Variante A: verkauft beim ersten Erreichen von +PROFIT zum Marktpreis.
- KEIN Stop-Loss (bewusst): fällt kein Tor, läuft die Position bis Spielende.

WICHTIG (14.07.2026): Für WETTER-Lays ist der Take-Profit ABGESCHALTET
(--no-tp-category, Default "weather"). Gemessen an 138 Lays
(preregs/weather_tp_vs_hold_2026_07_14.md) kostet er dort 6,6pp gegenüber Halten.
Der Grund ist strukturell: Ein steigender NO-Preis IST das Signal, dass der Lay
gewinnt — der TP kann per Konstruktion nur auf GEWINNERN auslösen. 62 von 62
Auslösungen wären Gewinner geworden (im Schnitt um 14,7pp gekappt), NULL
Rettungen; die Verlierer lief er ungeschützt voll. Bei In-Play-Scalps (Tennis)
bleibt er sinnvoll — dort IST die Edge eine Preisbewegung, kein Settlement.
Der Auto-Claim ist ausdrücklich NICHT betroffen.

Für den VPS gedacht (systemd, läuft 24/7, auch bei ausgeschaltetem PC).

Aufruf:
  python autopilot.py            # ECHT (verkauft autonom; Wetter ausgenommen)
  python autopilot.py --dry      # Dry-Run (loggt, verkauft nicht)
  python autopilot.py --profit 0.10 --interval 20 --idle-interval 90
  python autopilot.py --no-tp-category ''   # alter Zustand: TP fuer ALLES

Rate-Limit (429) der öffentlichen Jupiter-API: adaptives Polling — bei OFFENER
Position schnell (--interval), sonst langsam (--idle-interval); bei 429
exponentielles Backoff (respektiert Retry-After).
"""

import argparse
import logging
import re
import smtplib
import sys
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests

from datetime import date

from jupiter_sell import load_keypair, sell_position, claim_position, API
import green_up_state
# Wetter-Anreicherung der Settlement-Mails (Ist-Wert laut WU = Settlement-Quelle);
# STATIONS/TITLE_RE/wu_extreme aus dem Ladder-Logger = identische Definitionen.
from weather_ladder_logger import STATIONS, TITLE_RE, title_target_date, wu_extreme

# Mail-Benachrichtigung (GMX, hardcoded wie im Projekt üblich)
MAIL_HOST, MAIL_PORT = "mail.gmx.net", 587
MAIL_USER = MAIL_TO = "veit.luther@gmx.de"
MAIL_PASS = "Extaler00!"

# Cash-Stand der Hot-Wallet für die Mail-Fußzeile (Pubkey ist öffentlich).
# Auszahlungen kommen in JupUSD (JuprjznT…), NICHT Standard-USDC — beide Mints checken.
# Mehrere RPCs (wie jupiter_sell.RPCS): publicnode hängt gelegentlich komplett.
HOT_WALLET = "4XxStoKPzoiEJ6hUGEESfE54dCRo97LcCGk2UFieKjSi"
RPC_URLS = ["https://solana-rpc.publicnode.com", "https://api.mainnet-beta.solana.com"]

# v2-History (fuer die Verlust-Erkennung — verlorene Maerkte werden nie claimable)
PM_API = "https://prediction-market-api.jup.ag/api"
LOST_WATERMARK_FILE = Path("autopilot_lost_watermark.txt")
LOST_SCAN_INTERVAL = 600  # s
JUPUSD_MINT = "JuprjznTrTSp2UFa3ZBUFgwdAmtZCq4MQCwysN55USD"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

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


def wallet_cash():
    """Freies Guthaben der Hot-Wallet: (jupusd, usdc, sol) — None, wenn kein RPC antwortet."""
    # commitment=confirmed: confirm_tx() wartet nur bis 'confirmed' — mit dem
    # Default 'finalized' sähe der Nachher-Stand direkt nach Claim/Verkauf noch alt aus.
    def _tok(rpc, mint):
        r = requests.post(rpc, json={
            "jsonrpc": "2.0", "id": 1, "method": "getTokenAccountsByOwner",
            "params": [HOT_WALLET, {"mint": mint},
                       {"encoding": "jsonParsed", "commitment": "confirmed"}],
        }, timeout=15)
        r.raise_for_status()
        total = 0.0
        for acc in r.json().get("result", {}).get("value", []):
            ui = acc["account"]["data"]["parsed"]["info"]["tokenAmount"].get("uiAmount")
            total += ui or 0.0
        return total

    for rpc in RPC_URLS:
        try:
            jup = _tok(rpc, JUPUSD_MINT)
            usdc = _tok(rpc, USDC_MINT)
            r = requests.post(rpc, json={"jsonrpc": "2.0", "id": 1, "method": "getBalance",
                                         "params": [HOT_WALLET, {"commitment": "confirmed"}]},
                              timeout=15)
            r.raise_for_status()
            sol = (r.json().get("result", {}).get("value") or 0) / 1e9
            return jup, usdc, sol
        except Exception as e:
            log.warning(f"Cash-Stand via {rpc.split('//')[1].split('/')[0]} nicht abrufbar: {e}")
    return None


# Wetter-Buckets heißen nur "34°C"/"20°C or below" — die Stadt steht im Event-Titel.
WEATHER_TITLE_RE = re.compile(r"^Highest temperature in (.+?) on .+\?$")


def market_label(ev_title: str, title: str) -> str:
    """Kompakte Markt-Bezeichnung für Betreff/Header: bei Wetter-Buckets
    'Stadt Bucket' (z. B. 'Shanghai 34°C'), sonst 'Event — Markt'."""
    ev_title, title = ev_title or "", title or ""
    m = WEATHER_TITLE_RE.match(ev_title)
    if m and title and title != ev_title:
        return f"{m.group(1)} {title}"
    if title and ev_title and title != ev_title:
        return f"{ev_title} — {title}"
    return ev_title or title or "?"


def _event_line(ev_title: str, label: str) -> str:
    """Voller Event-Titel als Zusatzzeile, wenn das Label ihn nicht schon enthält
    (bei Wetter-Buckets steht dort das Datum)."""
    return ev_title if ev_title and ev_title not in label else ""


def _total(cash) -> float:
    """Kontostand = JupUSD + USDC (SOL ist nur Gas)."""
    jup, usdc, _ = cash
    return jup + usdc


def _cash_after(cash_before):
    """Kontostand nach der Aktion. Der RPC kann der frisch bestätigten Tx ein
    paar Slots hinterherhinken -> bei unverändertem Stand kurz nachfassen."""
    cash = wallet_cash()
    if cash_before and cash:
        for _ in range(3):
            if abs(_total(cash) - _total(cash_before)) > 0.005:
                break
            time.sleep(5)
            cash = wallet_cash() or cash
    return cash


def _cash_footer(cash_before, cash):
    """(html, text) der Kontostand-Fußzeile; mit cash_before als Vorher/Nachher."""
    if cash:
        jup, usdc, sol = cash
        detail = f"{jup:.2f} JupUSD + {usdc:.2f} USDC | Gas: {sol:.3f} SOL"
        if cash_before:
            delta = _total(cash) - _total(cash_before)
            dcol = "#2e7d32" if delta >= 0 else "#c62828"
            html = (f'💰 Kontostand: <b>{_total(cash_before):.2f} $</b> &rarr; '
                    f'<b>{_total(cash):.2f} $</b> '
                    f'(<span style="color:{dcol};font-weight:700;">{delta:+.2f} $</span>)'
                    f'<br><span style="font-size:11px;">{detail}</span>')
            text = (f"Kontostand vorher : {_total(cash_before):.2f} $\n"
                    f"Kontostand nachher: {_total(cash):.2f} $ ({delta:+.2f} $)\n"
                    f"({detail})")
            return html, text
        return f"💰 Cash: {detail}", f"Cash: {detail}"
    if cash_before:
        t = f"Kontostand vorher: {_total(cash_before):.2f} $ — aktueller Stand nicht abrufbar"
        return f"💰 {t}", t
    return "", ""


def notify(subject: str, html: str, text: str, cash_before=None):
    """Schickt eine Benachrichtigungs-Mail mit Kontostand-Fußzeile. cash_before
    (wallet_cash()-Tupel von VOR der Aktion) -> Fußzeile zeigt Vorher/Nachher
    + Delta statt nur den aktuellen Stand. Fehler crashen den Bot NICHT."""
    fhtml, ftext = _cash_footer(cash_before, _cash_after(cash_before))
    if fhtml:
        html += (f'<div style="max-width:480px;margin:8px auto 0;padding:10px 14px;'
                 f'background:#f5f5f5;border-radius:8px;font-family:Segoe UI,Arial,sans-serif;'
                 f'font-size:13px;color:#555;text-align:center;">{fhtml}</div>')
        text += f"\n\n{ftext}"
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


def notify_sale(ev_title, title, side, avg, sellp, pnl, contracts, sig, cash_before=None):
    label = market_label(ev_title, title)
    ev_line = _event_line(ev_title, label)
    erloes = contracts * sellp
    win = pnl > 0
    color = "#2e7d32" if pnl >= 0 else "#c62828"
    link = f"https://solscan.io/tx/{sig}" if sig else "#"
    event_html = (f'<div style="font-size:12px;color:#888;margin-bottom:6px;">{ev_line}</div>'
                  if ev_line else "")
    praise_html = (
        '<div style="margin-top:16px;padding:14px;background:#e8f5e9;border-radius:8px;'
        'text-align:center;font-size:18px;font-weight:800;color:#2e7d32;">'
        '🎉 Du bist der Geilste überhaupt.</div>'
    ) if win else ""
    html = f"""\
<div style="font-family:Segoe UI,Arial,sans-serif;max-width:480px;margin:auto;border-radius:12px;overflow:hidden;border:1px solid #eee;">
  <div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;text-align:center;color:#fff;">
    <div style="font-size:20px;font-weight:800;">✅ Position verkauft</div>
    <div style="font-size:14px;opacity:.9;">{label} &middot; {side}</div>
  </div>
  <div style="padding:20px;background:#fff;color:#333;font-size:15px;line-height:1.8;">
    {event_html}Einstieg: <b>{avg:.3f}</b> USDC<br>
    Verkauf: <b>{sellp:.3f}</b> USDC<br>
    Kontrakte: <b>{contracts:.2f}</b> &rarr; Erlös ~<b>{erloes:.2f}</b> USDC<br>
    <span style="font-size:22px;font-weight:800;color:{color};">{pnl:+.1f}%</span>
    {praise_html}
    <div style="margin-top:14px;"><a href="{link}" style="color:#667eea;font-size:12px;">🔗 Transaktion (Solscan)</a></div>
  </div>
</div>"""
    text = (f"Position verkauft: {label} [{side}]\n"
            + (f"Event: {ev_line}\n" if ev_line else "")
            + f"Einstieg {avg:.3f} -> Verkauf {sellp:.3f} USDC | {pnl:+.1f}%\n"
            f"Kontrakte {contracts:.2f}, Erloes ~{erloes:.2f} USDC\n"
            f"Tx: {link}")
    if win:
        text += "\n\nDu bist der Geilste ueberhaupt."
    notify(f"✅ Jupiter Bot: {label} verkauft ({pnl:+.1f}%)", html, text, cash_before)


def notify_fail(ev_title, title, reason):
    label = market_label(ev_title, title)
    html = (f'<div style="font-family:Arial;color:#c62828;">'
            f'<b>⚠️ Verkauf fehlgeschlagen:</b> {label}<br>Grund: {reason}<br>'
            f'Der Bot versucht es weiter. Ggf. manuell prüfen.</div>')
    notify(f"⚠️ Jupiter Bot: Verkauf fehlgeschlagen ({label})",
           html, f"Verkauf fehlgeschlagen: {label}\nGrund: {reason}\nBot versucht weiter.")


def notify_claimable(ev_title, title, side, payout, auto):
    """Markt aufgelöst & GEWONNEN -> Auszahlung abholbar."""
    label = market_label(ev_title, title)
    ev_line = _event_line(ev_title, label)
    event_html = f'<span style="font-size:12px;color:#888;">{ev_line}</span><br>' if ev_line else ""
    aktion = ("Der Bot versucht jetzt automatisch zu claimen."
              if auto else
              "Auto-Claim ist noch nicht aktiviert — bitte manuell in Jupiter claimen "
              "(oder warten, bis der Endpoint verifiziert ist).")
    html = (f'<div style="font-family:Segoe UI,Arial;max-width:480px;margin:auto;">'
            f'<div style="background:#2e7d32;color:#fff;padding:18px;text-align:center;'
            f'border-radius:10px 10px 0 0;font-size:18px;font-weight:800;">🏆 Gewonnen — abholbar!</div>'
            f'<div style="padding:18px;background:#fff;color:#333;font-size:15px;line-height:1.7;">'
            f'{label} [{side}]<br>{event_html}Auszahlung: <b>{payout:.2f}</b> USDC<br><br>{aktion}</div></div>')
    notify(f"🏆 Jupiter Bot: {label} gewonnen ({payout:.2f} USDC abholbar)",
           html, (f"GEWONNEN: {label} [{side}] — {payout:.2f} USDC abholbar.\n"
                  + (f"Event: {ev_line}\n" if ev_line else "") + aktion))


def notify_claim_gaveup(ev_title, title, side, payout, tries, reason):
    """Claim nach --claim-max-fails Zyklen aufgegeben -> Handarbeit nötig.

    Bewusst dringlicher als notify_claimable: hier holt der Bot das Geld NICHT
    mehr, es bleibt bis zum manuellen Claim liegen."""
    label = market_label(ev_title, title)
    ev_line = _event_line(ev_title, label)
    event_html = f'<span style="font-size:12px;color:#888;">{ev_line}</span><br>' if ev_line else ""
    html = (f'<div style="font-family:Segoe UI,Arial;max-width:480px;margin:auto;">'
            f'<div style="background:#b71c1c;color:#fff;padding:18px;text-align:center;'
            f'border-radius:10px 10px 0 0;font-size:18px;font-weight:800;">⛔ Claim aufgegeben</div>'
            f'<div style="padding:18px;background:#fff;color:#333;font-size:15px;line-height:1.7;">'
            f'{label} [{side}]<br>{event_html}'
            f'Auszahlung: <b>{payout:.2f}</b> USDC — <b>liegt noch bei Jupiter.</b><br><br>'
            f'Der Bot hat den Claim nach <b>{tries}</b> Zyklen eingestellt und versucht ihn '
            f'NICHT mehr. Letzter Fehler:<br><code style="font-size:12px;">{reason}</code><br><br>'
            f'Bitte manuell in Jupiter claimen. Die übrigen Positionen überwacht der Bot normal weiter.'
            f'</div></div>')
    notify(f"⛔ Jupiter Bot: Claim {label} aufgegeben ({payout:.2f} USDC offen)",
           html, (f"CLAIM AUFGEGEBEN: {label} [{side}] — {payout:.2f} USDC liegen noch bei Jupiter.\n"
                  + (f"Event: {ev_line}\n" if ev_line else "")
                  + f"Nach {tries} Zyklen eingestellt. Letzter Fehler: {reason}\n"
                  + "Bitte manuell claimen."))


def notify_claimed(ev_title, title, side, payout, sig, cash_before=None, avg=None):
    """Auszahlung erfolgreich eingelöst (on-chain bestätigt).
    avg = Einstandspreis der Position (avgPriceUsd in USDC); daraus ergibt sich
    der Gewinn in Prozent: jeder Kontrakt zahlt 1 USDC aus, gekostet hat er avg,
    also Rendite = (1 - avg)/avg."""
    pct = ((1.0 - avg) / avg * 100.0) if (avg and avg > 0) else None
    label = market_label(ev_title, title)
    ev_line = _event_line(ev_title, label)
    event_html = (f'<div style="font-size:12px;color:#888;margin-bottom:6px;">{ev_line}</div>'
                  if ev_line else "")
    link = f"https://solscan.io/tx/{sig}" if sig else "#"
    wline = weather_actual_line(ev_title, title)
    weather_html = (f'<div style="margin-top:10px;padding:10px;background:#e3f2fd;'
                    f'border-radius:8px;font-size:13px;">🌡️ {wline}</div>' if wline else "")
    html = f"""\
<div style="font-family:Segoe UI,Arial,sans-serif;max-width:480px;margin:auto;border-radius:12px;overflow:hidden;border:1px solid #eee;">
  <div style="background:linear-gradient(135deg,#2e7d32,#43a047);padding:20px;text-align:center;color:#fff;">
    <div style="font-size:20px;font-weight:800;">🏆 Gewinn eingelöst</div>
    <div style="font-size:14px;opacity:.9;">{label} &middot; {side}</div>
  </div>
  <div style="padding:20px;background:#fff;color:#333;font-size:15px;line-height:1.8;">
    {event_html}Auszahlung: <span style="font-size:22px;font-weight:800;color:#2e7d32;">+{payout:.2f} USDC</span>
    {(f'<div style="margin-top:6px;font-size:14px;color:#2e7d32;font-weight:700;">Gewinn: +{pct:.1f}% (Einstand {avg:.3f} USDC/Kontrakt)</div>') if pct is not None else ''}
    {weather_html}<br>
    <div style="margin-top:16px;padding:14px;background:#e8f5e9;border-radius:8px;text-align:center;font-size:18px;font-weight:800;color:#2e7d32;">
      🎉 Du bist der Geilste überhaupt.</div>
    <div style="margin-top:14px;"><a href="{link}" style="color:#667eea;font-size:12px;">🔗 Transaktion (Solscan)</a></div>
  </div>
</div>"""
    text = (f"Gewinn eingeloest: {label} [{side}]\n"
            + (f"Event: {ev_line}\n" if ev_line else "")
            + f"Auszahlung +{payout:.2f} USDC (on-chain bestaetigt)\n"
            + (f"Gewinn +{pct:.1f}% (Einstand {avg:.3f} USDC/Kontrakt)\n" if pct is not None else "")
            + (f"{wline}\n" if wline else "")
            + f"Tx: {link}\n\nDu bist der Geilste ueberhaupt.")
    notify(f"🏆 Jupiter Bot: {label} eingelöst (+{payout:.2f} USDC)", html, text, cash_before)


def weather_actual_line(ev_title, own_title):
    """Fuer Wetter-Maerkte: Ist-Wert laut Wunderground (= Settlement-Quelle) plus
    Abstand zum eigenen Bucket ('wie viel Luft war noch') — sonst None.
    Nutzer-Wunsch 19.07.: Settlement-Mails so ausfuehrlich wie moeglich."""
    try:
        m = TITLE_RE.match(ev_title or "")
        if not m:
            return None
        var = "max" if m.group(1) == "Highest" else "min"
        city = m.group(2)
        icao = STATIONS.get(city)
        target = title_target_date(m.group(3), m.group(4), date.today())
        if not (icao and target):
            return None
        actual = wu_extreme(icao, var, target)
        if actual is None:
            return None
        line = f"Ist laut Wunderground ({city}, {target:%d.%m.}): {round(actual):.0f} °C"
        km = re.search(r"(-?\d+)", own_title or "")
        if km:
            diff = abs(round(actual) - int(km.group(1)))
            line += (" — dein Bucket exakt getroffen" if diff == 0
                     else f" — {diff}° neben deinem Bucket")
        return line
    except Exception as e:
        log.debug(f"weather_actual_line: {e}")
        return None


def notify_lost(ev_title, title, side, stake, closed_ts):
    """Markt aufgeloest & VERLOREN -> Negativ-Mail (Nutzer-Wunsch 19.07.:
    auch Verluste melden, nicht nur Claims)."""
    label = market_label(ev_title, title)
    ev_line = _event_line(ev_title, label)
    wline = weather_actual_line(ev_title, title)
    event_html = (f'<div style="font-size:12px;color:#888;margin-bottom:6px;">{ev_line}</div>'
                  if ev_line else "")
    weather_html = (f'<div style="margin-top:10px;padding:10px;background:#fff3e0;'
                    f'border-radius:8px;font-size:13px;">🌡️ {wline}</div>' if wline else "")
    when = time.strftime("%d.%m. %H:%M UTC", time.gmtime(closed_ts)) if closed_ts else "?"
    html = f"""\
<div style="font-family:Segoe UI,Arial,sans-serif;max-width:480px;margin:auto;border-radius:12px;overflow:hidden;border:1px solid #eee;">
  <div style="background:linear-gradient(135deg,#b71c1c,#e53935);padding:20px;text-align:center;color:#fff;">
    <div style="font-size:20px;font-weight:800;">💥 Position verloren</div>
    <div style="font-size:14px;opacity:.9;">{label} &middot; {side}</div>
  </div>
  <div style="padding:20px;background:#fff;color:#333;font-size:15px;line-height:1.8;">
    {event_html}Einsatz weg: <span style="font-size:22px;font-weight:800;color:#c62828;">−{stake:.2f} USDC</span><br>
    <span style="font-size:12px;color:#888;">aufgelöst {when}</span>
    {weather_html}
  </div>
</div>"""
    text = (f"VERLOREN: {label} [{side}]\n"
            + (f"Event: {ev_line}\n" if ev_line else "")
            + f"Einsatz weg: -{stake:.2f} USDC, aufgeloest {when}\n"
            + (f"{wline}\n" if wline else ""))
    notify(f"💥 Jupiter Bot: {label} verloren (−{stake:.2f} USDC)", html, text)


def check_lost_positions(owner):
    """History-Scan: neue status='lost'-Eintraege -> einmalige Negativ-Mail.
    Watermark (Unix-s, Datei) ueberlebt Neustarts; Erst-Start mailt nichts nach."""
    try:
        wm = float(LOST_WATERMARK_FILE.read_text().strip())
    except Exception:
        LOST_WATERMARK_FILE.write_text(str(time.time()))
        log.info("Lost-Scan: Watermark initialisiert (alte Verluste werden nicht nachgemailt).")
        return
    try:
        # end=25: an aktiven Tagen (19.07.: 12 neue Eintraege) rutschen aeltere
        # Settlements schnell nach hinten — 25 deckt bei 10-min-Scans locker ab.
        r = requests.get(f"{PM_API}/v2/history",
                         params={"ownerPubkey": owner, "start": 0, "end": 25}, timeout=12)
        if r.status_code == 429:
            return
        r.raise_for_status()
        rows = r.json().get("data", [])
    except Exception as e:
        log.warning(f"Lost-Scan: History nicht abrufbar ({e})")
        return
    new_wm = wm
    for row in rows:
        closed = row.get("closedAt") or 0
        if row.get("status") != "lost" or closed <= wm:
            continue
        mm = row.get("marketMetadata") or {}
        em = row.get("eventMetadata") or {}
        title = mm.get("title", "?")
        ev_title = em.get("title", "") or title
        side = "YES" if row.get("isYes") else "NO"
        try:
            stake = abs(int(row.get("realizedPnlUsd") or 0)) / 1e6
        except (TypeError, ValueError):
            stake = 0.0
        log.warning(f"💥 VERLOREN {ev_title} [{side}]: -{stake:.2f} USDC — Negativ-Mail.")
        notify_lost(ev_title, title, side, stake, closed)
        new_wm = max(new_wm, closed)
    if new_wm > wm:
        LOST_WATERMARK_FILE.write_text(str(new_wm))


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


def tp_blocked_category(p: dict, blocked: set[str]) -> str | None:
    """Kategorie/Tag der Position, wenn sie vom Take-Profit AUSGENOMMEN ist — sonst None.

    Prüft category UND tags: Jupiter füllt nicht immer beides zuverlässig, und ein
    stiller Feldwechsel würde den Filter sonst lautlos deaktivieren."""
    if not blocked:
        return None
    em = p.get("eventMetadata", {}) or {}
    cats = {str(em.get("category") or "").lower()}
    cats |= {str(t).lower() for t in (em.get("tags") or [])}
    hit = cats & blocked
    return sorted(hit)[0] if hit else None


def is_imminent(p: dict, now: float, near_seconds: float) -> bool:
    """True, wenn der Markt OFFEN ist und bald schließt (laufendes Spiel) → schnell
    pollen. Eine langlaufende Position (z. B. Politik-Markt mit closeTime in Monaten)
    ist NICHT imminent → langsam pollen, schont das Rate-Limit-Budget.
    closeTime unbekannt (0) → sicherheitshalber als imminent behandeln."""
    mm = p.get("marketMetadata", {}) or {}
    if mm.get("status") != "open":
        return False
    ct = int(mm.get("closeTime", 0) or 0)
    if ct <= 0:
        return True
    return now < ct and (ct - now) <= near_seconds


def run(args):
    kp = load_keypair()
    owner = str(kp.pubkey())

    no_tp_cats = {c.strip().lower() for c in (args.no_tp_category or "").split(",") if c.strip()}
    no_tp_logged: set[str] = set()

    log.info("=" * 68)
    log.info(f"AUTOPILOT  |  {'DRY-RUN' if args.dry else 'LIVE (verkauft autonom)'}")
    log.info(f"Wallet {owner}")
    log.info(f"Take-Profit: +{args.profit*100:.0f}%  |  Poll: {args.interval}s nah / "
             f"{args.far_interval}s fern / {args.idle_interval}s idle  "
             f"(nah = <{args.near_hours:g}h vor Schluss)  |  kein Stop-Loss")
    if no_tp_cats:
        log.info(f"Take-Profit AUS für Kategorie(n): {', '.join(sorted(no_tp_cats))} "
                 f"— läuft dort bis Settlement, Claim bleibt aktiv "
                 f"(Befund 14.07.: TP kostet bei Wetter-Lays 6,6pp)")
    else:
        log.warning("Take-Profit für ALLE Kategorien aktiv — auch für Wetter-Lays, "
                    "wo er nachweislich 6,6pp kostet (--no-tp-category leer gesetzt).")
    log.info("=" * 68)

    MAX_BACKOFF = 300  # s
    fails = 0
    polls = 0
    sold_markets: set[str] = set()
    claimed_markets: set[str] = set()
    notified_fail: set[str] = set()
    notified_claimable: set[str] = set()
    notified_claim_fail: set[str] = set()
    claim_fails: dict[str, int] = {}   # marketId -> fehlgeschlagene Claim-Zyklen
    claim_gaveup: set[str] = set()     # Claim aufgegeben (Cap erreicht) -> nicht mehr versuchen
    closed_logged: set[str] = set()
    green_up_logged: set[str] = set()
    green_up_markets: set[str] = set()
    last_green_up_refresh = 0.0
    last_lost_scan = 0.0
    while True:
        polls += 1
        try:
            positions = get_open_positions(owner)
            fails = max(0, fails - 1)  # sticky: Drosselung wirkt nach, kein harter Reset auf 0
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

        # Verlust-Erkennung (Nutzer-Wunsch 19.07.): alle 10 min die History nach
        # neuen 'lost'-Settlements scannen — verlorene Maerkte werden nie claimable
        # und wuerden sonst KEINE Mail ausloesen. Läuft bewusst auch bei leerem
        # Positionsbuch (der letzte Verlust räumt das Buch ja gerade leer).
        if time.time() - last_lost_scan >= LOST_SCAN_INTERVAL:
            last_lost_scan = time.time()
            check_lost_positions(owner)

        # Adaptiv: keine offene Position -> langsam pollen (schont Rate-Limit);
        # sobald eine Position läuft -> schnell pollen (Exit nicht verpassen).
        if not positions:
            if polls % 10 == 1:  # Heartbeat
                log.info(f"Keine offene Position — warte ({args.idle_interval}s).")
            time.sleep(args.idle_interval)
            continue

        now = time.time()
        if now - last_green_up_refresh >= 60:
            green_up_markets = green_up_state.get_active_markets()  # fail-open bei DB-Fehler
            last_green_up_refresh = now

        for p in positions:
            mid = p.get("marketId")
            if mid in sold_markets or mid in claimed_markets:
                continue
            mm = p.get("marketMetadata", {}) or {}
            title = mm.get("title", "?")
            ev_title = (p.get("eventMetadata", {}) or {}).get("title", "") or title
            side = "NO" if not p.get("isYes") else "YES"

            # (1) Markt aufgelöst & GEWONNEN -> Auszahlung autonom einlösen (Claim).
            #     Claim-Endpoint verifiziert 2026-06-23; signierender Call in
            #     jupiter_sell.claim_position(). Fehlversuche sind GEDECKELT
            #     (--claim-max-fails): ein dauerhaft kaputter Claim (z. B. der
            #     Jupiter-Formatwechsel vom 20.07.) hat sonst 7258× retried und
            #     dabei den ganzen Positions-Loop lahmgelegt, während systemd
            #     weiter 'active' meldete. Nach dem Cap: Mail + überspringen,
            #     der Rest der Positionen läuft normal weiter.
            if p.get("claimable") and mid in claim_gaveup:
                continue
            if p.get("claimable"):
                payout = int(p.get("payoutUsd", 0)) / 1e6
                avg_in = int(p.get("avgPriceUsd") or 0) / 1e6  # Einstand für Gewinn-%
                if args.dry:
                    if mid not in notified_claimable:
                        log.warning(f"🏆 CLAIMBAR {ev_title} [{side}] {mid}: payout {payout:.2f} USDC "
                                    f"— DRY-RUN: würde jetzt claimen.")
                        notify_claimable(ev_title, title, side, payout, auto=True)
                        notified_claimable.add(mid)
                    continue
                log.warning(f"🏆 CLAIMBAR {ev_title} [{side}] {mid}: payout {payout:.2f} USDC — claime autonom…")
                cash_before = wallet_cash()  # Vorher-Stand für die Mail-Fußzeile
                res = claim_position(owner, p["pubkey"], kp, send=True)
                if res.get("ok"):
                    claimed_markets.add(mid)
                    claim_fails.pop(mid, None)
                    log.info(f"✅ Eingelöst: {ev_title}  sig={res.get('signature')}  status={res.get('status')}")
                    if not res.get("already"):
                        notify_claimed(ev_title, title, side, payout, res.get("signature"), cash_before, avg=avg_in)
                else:
                    n = claim_fails.get(mid, 0) + 1
                    claim_fails[mid] = n
                    if args.claim_max_fails > 0 and n >= args.claim_max_fails:
                        claim_gaveup.add(mid)
                        log.error(f"⛔ Claim für {ev_title} nach {n} Zyklen AUFGEGEBEN "
                                  f"({res.get('reason')}) — bitte manuell claimen. "
                                  f"Loop läuft für die übrigen Positionen normal weiter.")
                        notify_claim_gaveup(ev_title, title, side, payout, n, res.get("reason"))
                    else:
                        log.error(f"❌ Claim fehlgeschlagen für {ev_title} ({n}/{args.claim_max_fails}): "
                                  f"{res.get('reason')} — Retry nächster Poll.")
                        if mid not in notified_claim_fail:
                            notify_claimable(ev_title, title, side, payout, auto=True)
                            notified_claim_fail.add(mid)
                continue

            # (1.5) Markt steht unter Green-up-Verwaltung (bb_GreenUpHedges) ->
            #       NICHT verkaufen, sonst reisst das eine offene/gefüllte Hedge-
            #       Gegenwette auseinander (nackte Position). Claim oben bleibt
            #       ausdrücklich erlaubt — Einlösen gefährdet den Lock nicht.
            if mid in green_up_markets:
                if mid not in green_up_logged:
                    log.info(f"{ev_title} [{side}] {mid}: unter Green-up-Verwaltung — Verkauf pausiert.")
                    green_up_logged.add(mid)
                continue

            # (1.6) Take-Profit für diese Kategorie abgeschaltet -> NICHT verkaufen,
            #       bis zum Settlement laufen lassen. Claim oben bleibt ausdrücklich
            #       erlaubt — der holt die Gewinne ab.
            #
            #       Belegt am 14.07. (preregs/weather_tp_vs_hold_2026_07_14.md, 138 Lays):
            #       Bei Wetter-Lays kostet der +10-%-TP 6,6pp gegenüber Halten. Der Grund
            #       ist strukturell — ein steigender NO-Preis IST das Signal, dass der Lay
            #       gewinnt, also kann der TP per Konstruktion nur auf GEWINNERN auslösen:
            #       62 von 62 Auslösungen wären Gewinner geworden (im Schnitt um 14,7pp
            #       gekappt), NULL Rettungen. Die 47 Verlierer im Sample lief er
            #       ungeschützt voll. Unsere Edge realisiert sich erst beim Settlement.
            #       (Der Beijing-33-Lay, den der TP rettete, war 1 von ~64 — Glück.)
            blocked_cat = tp_blocked_category(p, no_tp_cats)
            if blocked_cat:
                if mid not in no_tp_logged:
                    log.info(f"{ev_title} [{side}] {mid}: Kategorie '{blocked_cat}' — Take-Profit "
                             f"AUS, läuft bis Settlement (Claim bleibt aktiv).")
                    no_tp_logged.add(mid)
                continue

            # (2) Markt geschlossen / nicht mehr handelbar -> NICHT verkaufen,
            #     auf Auflösung/Claim warten (verhindert Endlos-Fehlversuche).
            closeTime = int(mm.get("closeTime", 0) or 0)
            tradable = mm.get("status") == "open" and (closeTime == 0 or now < closeTime)
            if not tradable:
                if mid not in closed_logged:
                    log.info(f"{ev_title} [{side}] {mid}: Markt geschlossen "
                             f"(status={mm.get('status')}) — kein Verkauf, warte auf Auflösung/Claim.")
                    closed_logged.add(mid)
                continue

            # (3) Markt offen -> normale Take-Profit-Verkaufslogik.
            avg = int(p.get("avgPriceUsd") or 0) / 1e6
            sellp = int(p.get("sellPriceUsd") or 0) / 1e6
            if avg <= 0:
                continue
            try:
                contracts = float(p.get("contractsDecimal", 0) or 0)
            except (TypeError, ValueError):
                contracts = 0.0
            target = avg * (1 + args.profit)
            pnl = (sellp / avg - 1) * 100
            log.info(f"{title} [{side}] {mid}: Einstieg={avg:.3f} sell={sellp:.3f} "
                     f"PnL={pnl:+.1f}% Ziel≥{target:.3f}")

            if sellp >= target:
                # Brutto-Bedingung erreicht (billiger Vorfilter, da netto <= brutto).
                # Vor dem Verkauf NETTO prüfen: Orderbuch-Walk (Slippage bei Größe)
                # + Fee-Schätzung. Nur verkaufen, wenn netto noch >= Ziel.
                if args.net_check:
                    est = estimate_net_exit(mid, bool(p.get("isYes")), contracts, avg, args.fee_rate)
                    if est is None:
                        log.warning(f"⏸️ {title}: Brutto-Trigger (+{pnl:.1f}%), aber Orderbuch nicht "
                                    f"abrufbar — kein Verkauf, Retry nächster Poll.")
                        continue
                    net_pct = est["net_pnl"] * 100
                    if est["net_pnl"] < args.profit:
                        log.warning(f"⏸️ {title}: brutto +{pnl:.1f}% ABER netto {net_pct:+.1f}% "
                                    f"(Fill~{est['est_fill']:.3f}, Fee~{est['est_fee']:.2f}, "
                                    f"fill {100*est['fill_ratio']:.0f}%) < Ziel +{args.profit*100:.0f}% "
                                    f"— Slippage/Fee zu hoch, kein Verkauf.")
                        continue
                    log.warning(f"🎯 TRIGGER {title}: brutto +{pnl:.1f}% / NETTO {net_pct:+.1f}% "
                                f"(Fill~{est['est_fill']:.3f}) ≥ +{args.profit*100:.0f}%")
                else:
                    log.warning(f"🎯 TRIGGER {title}: PnL {pnl:+.1f}% ≥ +{args.profit*100:.0f}% (Netto-Check AUS)")
                if args.dry:
                    log.info("DRY-RUN: würde jetzt verkaufen (nichts gesendet).")
                else:
                    cash_before = wallet_cash()  # Vorher-Stand für die Mail-Fußzeile
                    res = sell_position(owner, mid, kp, send=True)
                    if res.get("ok"):
                        log.info(f"✅ Verkauft: {title}  sig={res.get('signature')}  status={res.get('status')}")
                        sold_markets.add(mid)
                        notify_sale(ev_title, title, side, avg, sellp, pnl, contracts,
                                    res.get("signature"), cash_before)
                    else:
                        log.error(f"❌ Verkauf fehlgeschlagen für {title}: {res.get('reason')} — Retry nächster Poll.")
                        if mid not in notified_fail:
                            notify_fail(ev_title, title, res.get("reason"))
                            notified_fail.add(mid)

        # Kadenz an Zeit-bis-Marktschluss koppeln: schnell pollen NUR, wenn ein Markt
        # bald schließt (laufendes Spiel). Eine langlaufende Position (Politik-Markt,
        # closeTime in Monaten) pollt langsam und sprengt so nicht das Rate-Limit-Budget.
        now = time.time()
        near_seconds = args.near_hours * 3600
        if any(is_imminent(p, now, near_seconds) for p in positions):
            time.sleep(args.interval)
        else:
            time.sleep(args.far_interval)


def estimate_net_exit(market_id, held_is_yes, contracts, avg, fee_rate):
    """Schätzt den NETTO-Erlös eines Marktverkaufs der vollen Position.

    Zieht das Orderbuch und simuliert den Verkauf von `contracts` durch die
    Bid-Leiter der gehaltenen Seite (höchster Bid zuerst) -> realistischer
    Durchschnitts-Fill inkl. Slippage. Fee aus Audit-Modell (~fee_rate * Stück *
    min(p,1-p)). Orderbuch-Semantik verifiziert 2026-06-30: ob[side] = Bids dieser
    Seite, Preis in Cent, bester Bid = höchster Preis (= sellPriceUsd).
    Rückgabe dict oder None bei Abruf-Fehler.
    """
    side = "yes" if held_is_yes else "no"
    try:
        r = requests.get(f"{API}/orderbook/{market_id}", timeout=10)
        r.raise_for_status()
        ob = r.json()
    except Exception:
        return None
    bids = sorted((b for b in ob.get(side, []) if b and len(b) >= 2),
                  key=lambda x: x[0], reverse=True)
    remaining, proceeds = contracts, 0.0
    for price_c, size in bids:
        take = min(remaining, float(size or 0))
        proceeds += take * (price_c / 100.0)
        remaining -= take
        if remaining <= 1e-9:
            break
    filled = contracts - max(0.0, remaining)
    if filled <= 0 or avg <= 0:
        return {"net_pnl": -1.0, "est_fill": 0.0, "est_fee": 0.0, "fill_ratio": 0.0}
    est_fill = proceeds / filled
    est_fee = fee_rate * filled * min(est_fill, 1.0 - est_fill)
    net_pnl = (proceeds - est_fee) / (contracts * avg) - 1.0
    return {"net_pnl": net_pnl, "est_fill": est_fill, "est_fee": est_fee,
            "fill_ratio": filled / contracts}


def main():
    ap = argparse.ArgumentParser(description="Autonomer Take-Profit-Exit (Jupiter Prediction)")
    ap.add_argument("--profit", type=float, default=0.10,
                    help="Take-Profit-Schwelle, jetzt NETTO nach Fee+Slippage (default 0.10 = 10%%)")
    ap.add_argument("--no-tp-category", default="weather",
                    help="Kommagetrennte Kategorien/Tags, fuer die KEIN Take-Profit gefahren "
                         "wird — die Position laeuft bis zum Settlement, der Auto-Claim bleibt "
                         "aktiv. Default 'weather': dort ist der TP belegt EV-negativ "
                         "(-6,6pp, preregs/weather_tp_vs_hold_2026_07_14.md). "
                         "--no-tp-category '' schaltet den Filter ab (alter Zustand).")
    ap.add_argument("--fee-rate", type=float, default=0.07,
                    help="Fee-Schätzrate für Netto-Check: Fee ~ rate*Stück*min(p,1-p) (default 0.07)")
    ap.add_argument("--no-net-check", dest="net_check", action="store_false",
                    help="Netto-Check (Orderbuch-Slippage+Fee) AUS -> alter Brutto-Trigger")
    ap.set_defaults(net_check=True)
    ap.add_argument("--interval", type=int, default=20,
                    help="Poll-Intervall bei OFFENER Position, Sekunden (default 20)")
    ap.add_argument("--idle-interval", type=int, default=90,
                    help="Poll-Intervall OHNE offene Position, Sekunden (default 90)")
    ap.add_argument("--far-interval", type=int, default=180,
                    help="Poll-Intervall bei offener, aber NICHT bald schließender Position (default 180)")
    ap.add_argument("--near-hours", type=float, default=3.0,
                    help="Markt gilt als 'nah' (schnell pollen), wenn closeTime < near-hours entfernt (default 3)")
    ap.add_argument("--claim-max-fails", type=int, default=5,
                    help="Nach so vielen fehlgeschlagenen Claim-Zyklen je Markt aufgeben "
                         "(Mail + überspringen) statt endlos zu retryen. 0 = unbegrenzt (alt).")
    ap.add_argument("--dry", action="store_true", help="Dry-Run: loggt, verkauft NICHT")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
