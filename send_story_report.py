#!/usr/bin/env python3
"""
Einmaliger Story-Report: Jupiter Prediction Bot — von der Arbitrage-Idee
bis zum erfolgreichen vollautonomen Verkauf. HTML-Mail an den Betreiber.
"""

import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

MAIL_HOST, MAIL_PORT = "mail.gmx.net", 587
MAIL_USER = "veit.luther@gmx.de"
MAIL_PASS = "Extaler00!"
MAIL_TO = "veit.luther@gmx.de"

TX = "2Pdrju99Lyr5Tjp8tAJkYshqKRY6M6XSUyL82VrYaZFYeaE4XjfuKDDSbrKJYodMxaoXPaH7mfoK1mSFJj6LsEh8"
DATUM = datetime.now().strftime("%d.%m.%Y %H:%M")

HTML = f"""\
<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#0f0c29;font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#0f0c29;padding:24px 0;">
<tr><td align="center">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;">

  <!-- Header -->
  <tr><td style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:14px 14px 0 0;padding:32px 28px;text-align:center;">
    <div style="font-size:30px;font-weight:800;color:#ffffff;">⚡ Jupiter Prediction Bot</div>
    <div style="font-size:15px;color:#e8e6ff;margin-top:6px;">Projektbericht — von der Arbitrage-Idee zum Autopiloten</div>
    <div style="font-size:12px;color:#cfc9ff;margin-top:10px;">{DATUM}</div>
  </td></tr>

  <!-- Ergebnis-Banner -->
  <tr><td style="background:#1b5e20;padding:20px 28px;text-align:center;">
    <div style="font-size:13px;color:#a5d6a7;letter-spacing:1px;text-transform:uppercase;">Meilenstein erreicht</div>
    <div style="font-size:22px;font-weight:800;color:#ffffff;margin-top:4px;">✅ Vollautonomer Verkauf funktioniert — on-chain bestätigt</div>
  </td></tr>

  <!-- Story -->
  <tr><td style="background:#ffffff;padding:28px;">

    <div style="font-size:13px;font-weight:700;color:#764ba2;text-transform:uppercase;letter-spacing:1px;">① Die Ausgangsidee</div>
    <p style="font-size:15px;color:#333;line-height:1.6;margin:8px 0 22px;">
      Gesucht war eine <b>Arbitrage</b> zwischen Prediction Markets (Polymarket → später Jupiter)
      und Buchmacher-Quoten. Idee: dieselbe Wette auf zwei Plattformen zu unterschiedlichen
      Preisen = risikoloser Gewinn.
    </p>

    <div style="font-size:13px;font-weight:700;color:#764ba2;text-transform:uppercase;letter-spacing:1px;">② Der ehrliche Befund</div>
    <p style="font-size:15px;color:#333;line-height:1.6;margin:8px 0 10px;">
      Kein „free lunch". Jupiter und der schärfste Buchmacher (Pinnacle) waren sich
      auf <b>&lt; 1 Prozentpunkt</b> einig. Und intern frisst der Spread den Overround exakt auf:
    </p>
    <div style="background:#f3f0ff;border-radius:8px;padding:14px;text-align:center;font-size:16px;color:#333;margin-bottom:22px;">
      <b style="color:#c62828;">0,99</b> (Verkauf) &nbsp;&lt;&nbsp; <b>1,00</b> (fair) &nbsp;&lt;&nbsp; <b style="color:#c62828;">1,02</b> (Kauf)
      <div style="font-size:12px;color:#888;margin-top:6px;">Lehrbuch-effizienter Markt — keine risikolose Arbitrage</div>
    </div>

    <div style="font-size:13px;font-weight:700;color:#764ba2;text-transform:uppercase;letter-spacing:1px;">③ Der Pivot — „Lay the Draw"</div>
    <p style="font-size:15px;color:#333;line-height:1.6;margin:8px 0 22px;">
      Statt Arbitrage eine gerichtete Wette: <b>gegen das Unentschieden</b> (NO im Draw-Markt).
      Take-Profit bei <b>+10 %</b>, Verkauf zum Marktpreis (höher, wenn ein Tor den Kurs treibt).
      Kein Stop-Loss — fällt kein Führungstor, ist die Wette futsch. Bewusst akzeptiert.
    </p>

    <div style="font-size:13px;font-weight:700;color:#764ba2;text-transform:uppercase;letter-spacing:1px;">④ Die Maschine</div>
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin:8px 0 22px;">
      <tr><td style="font-size:14px;color:#333;line-height:1.9;">
        🔐 Dediziertes <b>Hot-Wallet</b> (nur Trade-Kapital, getrennt von der Hauptwallet)<br>
        🔎 API <b>reverse-engineered</b> — öffentlich, kein Key nötig (der Key verursachte sogar einen 401!)<br>
        ✍️ Signier-Kette geknackt: Jupiter <b>vorsigniert</b> einen Slot, der Bot signiert den anderen → 2/2<br>
        🤖 Bot pollt den Kurs, verkauft autonom & bestätigt die Transaktion
      </td></tr>
    </table>

    <div style="font-size:13px;font-weight:700;color:#764ba2;text-transform:uppercase;letter-spacing:1px;">⑤ Der Test — erfolgreich</div>
    <p style="font-size:15px;color:#333;line-height:1.6;margin:8px 0 10px;">
      Eine kleine Testposition (5,82 NO-Draw-Kontrakte) wurde <b>vollautomatisch verkauft</b> —
      Transaktion auf Solana bestätigt:
    </p>
    <div style="background:#f3f0ff;border-radius:8px;padding:12px;margin-bottom:22px;text-align:center;">
      <a href="https://solscan.io/tx/{TX}" style="color:#667eea;font-size:12px;word-break:break-all;text-decoration:none;">🔗 {TX[:32]}… (Solscan)</a>
    </div>

  </td></tr>

  <!-- P&L Karte -->
  <tr><td style="background:#ffffff;padding:0 28px 28px;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-radius:12px;overflow:hidden;border:2px solid #ffcdd2;">
      <tr><td style="background:#fff5f5;padding:22px;text-align:center;">
        <div style="font-size:13px;color:#999;text-transform:uppercase;letter-spacing:1px;">Test-Trade Gewinn / Verlust</div>
        <div style="font-size:40px;font-weight:800;color:#c62828;margin:6px 0;">−0,085 USDC</div>
        <div style="font-size:16px;color:#c62828;font-weight:700;">−1,8 %</div>
        <div style="font-size:13px;color:#777;margin-top:10px;line-height:1.6;">
          Kauf 4,71 USDC → Verkauf ~4,63 USDC.<br>
          Der „Verlust" ist reiner <b>Spread + Gebühren</b> — die geplante Testgebühr.
          Das Ziel war nicht Gewinn, sondern: <b>funktioniert der Autopilot?</b> → Ja.
        </div>
      </td></tr>
    </table>
  </td></tr>

  <!-- Ausblick -->
  <tr><td style="background:#ffffff;padding:0 28px 28px;border-radius:0 0 14px 14px;">
    <div style="background:#e8f5e9;border-left:5px solid #2e7d32;border-radius:8px;padding:16px;">
      <div style="font-size:14px;font-weight:700;color:#2e7d32;">Status: einsatzbereit</div>
      <div style="font-size:13px;color:#444;line-height:1.6;margin-top:6px;">
        Die autonome Verkauf-Pipeline ist verifiziert. Für die echte Wette kann der Bot
        eine NO-Draw-Position überwachen und bei +10 % von selbst verkaufen — auch wenn
        dein Rechner aus ist.<br><br>
        <b>Risiko-Hinweis:</b> Der Private Key des Hot-Wallets liegt für den Autopiloten
        auf dem Server. Im Ernstfall ist nur das Trade-Kapital (~21 USDC) exponiert,
        niemals die Hauptwallet.
      </div>
    </div>
    <div style="text-align:center;font-size:11px;color:#aaa;margin-top:18px;">
      Automatisch erstellt vom Börsenbot · Jupiter Prediction Research
    </div>
  </td></tr>

</table>
</td></tr>
</table>
</body></html>
"""

TEXT = f"""Jupiter Prediction Bot - Projektbericht

MEILENSTEIN: Vollautonomer Verkauf funktioniert (on-chain bestaetigt).

1) Idee: Arbitrage Prediction Markets vs Buchmacher.
2) Befund: Kein free lunch - Markt effizient (0,99 < 1,00 < 1,02).
3) Pivot: Lay the Draw (gegen Unentschieden, +10% Take-Profit).
4) Maschine: dediziertes Hot-Wallet, API reverse-engineered (kein Key noetig),
   Signier-Kette geknackt (Jupiter vorsigniert, Bot signiert -> 2/2), autonomer Verkauf.
5) Test ERFOLGREICH: 5,82 NO-Draw-Kontrakte automatisch verkauft.
   Tx: https://solscan.io/tx/{TX}

GEWINN/VERLUST Test-Trade: -0,085 USDC (-1,8%).
Reiner Spread + Gebuehren (geplante Testgebuehr). Ziel war der Funktionstest - bestanden.

Status: einsatzbereit. Risiko: Hot-Wallet-Key auf Server, nur Trade-Kapital exponiert.
"""


def main():
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "⚡ Jupiter Bot — Autopilot funktioniert (Test: −1,8 %)"
    msg["From"] = MAIL_USER
    msg["To"] = MAIL_TO
    msg.attach(MIMEText(TEXT, "plain", "utf-8"))
    msg.attach(MIMEText(HTML, "html", "utf-8"))

    print(f"Verbinde zu {MAIL_HOST}:{MAIL_PORT} ...")
    with smtplib.SMTP(MAIL_HOST, MAIL_PORT, timeout=30) as s:
        s.starttls()
        s.login(MAIL_USER, MAIL_PASS)
        s.sendmail(MAIL_USER, [MAIL_TO], msg.as_string())
    print(f"✅ Report gesendet an {MAIL_TO}")


if __name__ == "__main__":
    main()
