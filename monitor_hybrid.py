#!/usr/bin/env python3
"""
HYBRID SYSTEM MONITOR (NAS-nativ)

Laeuft per Cron AUF der NAS (23:00 UTC, nach Hybrid-Calc 22:00 / HYG-Calc 22:05).
Liest hybrid.log + hyg.log direkt lokal (kein SSH/paramiko noetig) und prueft:
  - taegliche Ausfuehrung (heutiger Datums-String im Log vorhanden?)
  - Fehlerzeilen (Traceback / Error / Exception / Fail)
Bei Problemen: Alert per GMX-SMTP (gleicher Pfad wie email_report / signal_to_orders).
"""
import os
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText

BASE_DIR = "/var/services/homes/benutzername/boersenbot"
LOGS = {
    "hybrid": os.path.join(BASE_DIR, "hybrid.log"),
    "hyg": os.path.join(BASE_DIR, "hyg.log"),
}

# SMTP wie email_report / signal_to_orders (GMX)
MAIL_HOST, MAIL_PORT = "mail.gmx.net", 587
MAIL_USER = "veit.luther@gmx.de"
MAIL_PASS = "Extaler00!"
MAIL_TO = "veit.luther@gmx.de"


def tail(path, n=120):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except FileNotFoundError:
        return None


def check_errors(lines):
    kws = ("traceback", "error", "exception", "fail")
    return [l.strip() for l in lines if any(k in l.lower() for k in kws)]


def send_alert(subject, body):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = f"[HYBRID-MONITOR] {subject}"
    msg["From"] = MAIL_USER
    msg["To"] = MAIL_TO
    try:
        with smtplib.SMTP(MAIL_HOST, MAIL_PORT, timeout=30) as s:
            s.ehlo()
            s.starttls()
            s.login(MAIL_USER, MAIL_PASS)
            s.send_message(msg)
        print(f"  Alert versendet: {subject}")
    except Exception as e:
        print(f"  WARN: Alert-Mail fehlgeschlagen: {e}")


def main():
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    print("=" * 60)
    print(f"HYBRID MONITOR  {now.isoformat()}")
    print("=" * 60)

    problems = []
    for name, path in LOGS.items():
        lines = tail(path)
        if lines is None:
            problems.append(f"{name}.log fehlt ({path})")
            print(f"[{name}] LOG FEHLT")
            continue
        content = "".join(lines)
        ran = today in content
        errs = check_errors(lines)
        print(f"[{name}] heute ausgefuehrt: {ran} | Fehlerzeilen: {len(errs)}")
        if not ran:
            problems.append(f"{name}: kein Log-Eintrag fuer heute ({today})")
        if errs:
            problems.append(f"{name}: {len(errs)} Fehlerzeile(n), z.B. '{errs[0][:120]}'")

    if problems:
        body = ("Hybrid-Monitor hat Probleme erkannt:\n\n"
                + "\n".join("- " + p for p in problems)
                + f"\n\nStand: {now.isoformat()}")
        send_alert("Probleme erkannt", body)
        print(f"PROBLEME: {len(problems)}")
    else:
        print("Alles OK - keine Alerts.")


if __name__ == "__main__":
    main()
