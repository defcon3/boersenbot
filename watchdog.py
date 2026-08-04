# -*- coding: utf-8 -*-
"""watchdog.py — merkt, wenn etwas still stehenbleibt.

## Anlass (04.08.2026)

Zwei Faelle am selben Tag, beide unbemerkt ueber Tage:

* **`veitluther.de` war 12 Tage tot.** `boersenbot_dashboard` wurde am 23.07.
  sauber gestoppt und deaktiviert; nginx lief ins Leere, niemand hat es gesehen.
* **Die Latenz-Sonde lief nicht.** Sie war am 01.08. still ausgelaufen, waehrend
  ein gleichnamiger anderer Dienst `active` meldete.

Daraus die zwei Regeln, nach denen dieser Waechter gebaut ist:

1. **`systemctl is-active` ist kein Lebenszeichen.** Ein Prozess kann laufen und
   trotzdem nichts mehr tun. Deshalb wird zusaetzlich die **Frische** der Dateien
   geprueft, die ein lebender Loop zwangslaeufig anfasst.
2. **Nur Zustandswechsel melden.** Eine Mail bei OK->FAIL und eine bei
   FAIL->OK — nie eine dritte. Ein Waechter, der stuendlich dieselbe Stoerung
   meldet, wird nach zwei Tagen weggefiltert und ist dann so nutzlos wie keiner.

## Schwellen

Gemessen im Normalbetrieb am 04.08. (Alter der Dateien in Minuten): die drei
Dauerschleifen schreiben im Minutentakt, `weather_source_latency.csv` nur bei
Erstsichtungen (~12 min). Die Schwellen liegen bewusst um ein Vielfaches
darueber — ein Waechter, der bei normalem Rauschen anschlaegt, erzieht zum
Wegsehen.

Aufruf:
    python watchdog.py               # ein Durchlauf, Mail nur bei Wechsel
    python watchdog.py --verbose     # Status aller Pruefungen ausgeben
    python watchdog.py --force-mail  # Mail erzwingen (Zustellung testen)
"""
import argparse
import json
import os
import socket
import smtplib
import subprocess
import sys
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(BASE, ".watchdog_state.json")

# Mail wie im uebrigen Projekt (autopilot.py, monitor_hybrid.py)
MAIL_HOST, MAIL_PORT = "mail.gmx.net", 587
MAIL_USER = MAIL_TO = "veit.luther@gmx.de"
MAIL_PASS = "Extaler00!"

# --- Was geprueft wird ------------------------------------------------------

# Oeffentliche Routen. Der Dashboard-Ausfall waere hier sofort aufgefallen.
URLS = [
    ("web /",                   "https://veitluther.de/"),
    ("web /fazit",              "https://veitluther.de/fazit"),
    ("web /done",               "https://veitluther.de/done"),
    ("web /dividend-watch",     "https://veitluther.de/dividend-watch"),
    ("web /overnight-intraday", "https://veitluther.de/overnight-intraday"),
    ("web /analysis",           "https://veitluther.de/analysis"),
    ("web /optionen",           "https://veitluther.de/optionen"),
]

# Dienste, die dauerhaft laufen SOLLEN (Stand 04.08.2026).
# Timer-gesteuerte Units (ladder, minus1, eps_logger) stehen bewusst NICHT drin —
# die sind zwischen zwei Laeufen korrekt inaktiv.
UNITS = [
    "boersenbot_dashboard", "boersenbot_analysis", "boersenbot_optionen",
    "boersenbot_streaming", "boersenbot_autopilot", "boersenbot_btcflip",
    "boersenbot_green_up", "boersenbot_source_latency",
    "boersenbot_weather_latency", "boersenbot_weather_tile_latency",
]

# (Name, Pfad relativ zu BASE, maximales Alter in Minuten)
# Das ist die Pruefung, die den Sonden-Tod erwischt haette.
FRESH = [
    ("loop source-latency", "logs/weather_source_latency.log",  60),
    ("daten source-latency", "weather_source_latency.csv",     180),
    ("loop weather-latency", "logs/weather_latency.log",        60),
    ("loop tile-latency",   "logs/weather_tile_latency.log",    60),
]

DISK_MAX_PCT = 90


def check_urls():
    out = {}
    for name, url in URLS:
        try:
            r = requests.get(url, timeout=25, allow_redirects=True)
            out[name] = (r.status_code == 200, f"HTTP {r.status_code}")
        except Exception as ex:
            out[name] = (False, f"{type(ex).__name__}: {str(ex)[:80]}")
    return out


def check_units():
    out = {}
    for u in UNITS:
        try:
            r = subprocess.run(["systemctl", "is-active", u],
                               capture_output=True, text=True, timeout=20)
            s = r.stdout.strip()
            out[f"dienst {u}"] = (s == "active", s or "unbekannt")
        except Exception as ex:
            out[f"dienst {u}"] = (False, f"{type(ex).__name__}")
    return out


def check_freshness():
    """Die Lehre aus beiden Ausfaellen: laeuft der Prozess noch, oder tut er nur so?"""
    out = {}
    now = time.time()
    for name, rel, max_min in FRESH:
        p = os.path.join(BASE, rel)
        if not os.path.exists(p):
            out[name] = (False, "Datei fehlt")
            continue
        age = (now - os.path.getmtime(p)) / 60
        out[name] = (age <= max_min, f"{age:.0f} min alt (max {max_min})")
    return out


def check_disk():
    try:
        st = os.statvfs("/")
        used = 100.0 * (1 - st.f_bavail / st.f_blocks)
        return {"platte /": (used < DISK_MAX_PCT, f"{used:.0f}% belegt")}
    except Exception as ex:
        return {"platte /": (False, str(ex)[:60])}


def check_db():
    try:
        import pymssql
        c = pymssql.connect(server="158.181.48.77", database="dbdata",
                            user="326773", password="Extaler11!", timeout=20,
                            login_timeout=20)
        cur = c.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        c.close()
        return {"datenbank": (True, "erreichbar")}
    except Exception as ex:
        return {"datenbank": (False, f"{type(ex).__name__}: {str(ex)[:70]}")}


def send_mail(subject, body):
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = MAIL_USER
        msg["To"] = MAIL_TO
        msg.attach(MIMEText(body, "plain", "utf-8"))
        with smtplib.SMTP(MAIL_HOST, MAIL_PORT, timeout=30) as s:
            s.starttls()
            s.login(MAIL_USER, MAIL_PASS)
            s.sendmail(MAIL_USER, [MAIL_TO], msg.as_string())
        return True
    except Exception as ex:
        print(f"Mail-Versand fehlgeschlagen: {ex}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--force-mail", action="store_true",
                    help="Mail auch ohne Zustandswechsel (Zustellung testen)")
    a = ap.parse_args()

    res = {}
    for f in (check_urls, check_units, check_freshness, check_disk, check_db):
        res.update(f())

    try:
        with open(STATE, encoding="utf-8") as fh:
            alt = json.load(fh)
    except Exception:
        alt = {}

    neu_kaputt = [k for k, (ok, _) in res.items() if not ok and alt.get(k, True)]
    neu_heil = [k for k, (ok, _) in res.items() if ok and alt.get(k) is False]

    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if a.verbose or neu_kaputt or neu_heil:
        print(f"[{stamp}] {sum(1 for ok, _ in res.values() if ok)}/{len(res)} ok")
        for k, (ok, info) in sorted(res.items()):
            if a.verbose or not ok:
                print(f"  {'OK  ' if ok else 'FAIL'} {k:34s} {info}")

    if neu_kaputt or neu_heil or a.force_mail:
        host = socket.gethostname()
        if neu_kaputt:
            betreff = f"[boersenbot] STOERUNG: {', '.join(neu_kaputt[:3])}"
            if len(neu_kaputt) > 3:
                betreff += f" (+{len(neu_kaputt) - 3})"
        elif neu_heil:
            betreff = f"[boersenbot] wieder ok: {', '.join(neu_heil[:3])}"
        else:
            betreff = "[boersenbot] Waechter-Testmail"

        zeilen = [f"Waechter {host}, {stamp}", ""]
        if neu_kaputt:
            zeilen += ["NEU AUSGEFALLEN:"] + \
                      [f"  {k}: {res[k][1]}" for k in neu_kaputt] + [""]
        if neu_heil:
            zeilen += ["WIEDER IN ORDNUNG:"] + \
                      [f"  {k}: {res[k][1]}" for k in neu_heil] + [""]
        zeilen += ["Vollstaendiger Stand:"]
        zeilen += [f"  {'OK  ' if ok else 'FAIL'} {k:34s} {info}"
                   for k, (ok, info) in sorted(res.items())]
        send_mail(betreff, "\n".join(zeilen))

    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump({k: ok for k, (ok, _) in res.items()}, fh, indent=1)

    return 0 if all(ok for ok, _ in res.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
