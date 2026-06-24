# Deploy-Artefakte (VPS 144.91.98.234, veitluther.de)

systemd-Unit-Files (Stand: aktiv aus `/etc/systemd/system/`) + Setup-Skript für
den Börsenbot-Webstack. Alle Services laufen als User `veit`, WorkingDir
`/home/veit/boersenbot`, gunicorn aus `venv/`.

## Services

| Unit | Port | App | Route (nginx) |
|------|------|-----|---------------|
| `boersenbot_dashboard.service` | 5000 | `app:app` | `/` (+ `/fazit`, `/done`, `/dividend-watch`, `/overnight-intraday`) |
| `boersenbot_analysis.service`  | 5001 | `analysis_app:app` | `/analysis` (Prefix wird gestrippt; `--timeout 300` wg. ML-Predict) |
| `boersenbot_optionen.service`  | 5051 | `optionen_vergleich:app` | `/optionen` |
| `boersenbot_streaming.service` | –    | `alpaca_streaming_simple.py` | kein HTTP (Alpaca-Stream) |

## Hintergrund-Jobs (kein HTTP)

| Unit | Typ | Was |
|------|-----|-----|
| `boersenbot_autopilot.service` | simple/Loop | Jupiter-Positionsüberwachung + Auto-Claim |
| `boersenbot_football_odds.service` | simple/Loop | minütliche Polymarket-Quoten → `bb_FootballOdds_1min` |
| `boersenbot_football_backfill.service` | simple/Loop | Endstände → `bb_FootballMatches` |
| `boersenbot_tennis_paper.timer` | **Timer** (30 min) | Pre-Match-Snapshot ATP/WTA → `bb_TennisPaperBets` |
| `boersenbot_tennis_settle.timer` | **Timer** (alle 6h) | Sieger nachtragen (Jupiter-Events fallen nach ~1–2 Tagen raus) |

**Tennis-Timer aktivieren** (oneshot-Service + Timer, beide hochladen):
```bash
scp -i ~/.ssh/boersenbot_key deploy/boersenbot_tennis_paper.{service,timer} \
    deploy/boersenbot_tennis_settle.{service,timer} veit@144.91.98.234:/tmp/
ssh -i ~/.ssh/boersenbot_key veit@144.91.98.234 '
  sudo mv /tmp/boersenbot_tennis_*.{service,timer} /etc/systemd/system/ &&
  sudo systemctl daemon-reload &&
  sudo systemctl enable --now boersenbot_tennis_paper.timer boersenbot_tennis_settle.timer &&
  systemctl list-timers "boersenbot_tennis_*"'
```
Logs: `logs/tennis_paper.log`, `logs/tennis_settle.log`. Einmaliger Direkt-Test:
`sudo systemctl start boersenbot_tennis_paper.service` (läuft `--once`, beendet sich).

## nginx-Routing (Kurzform)

```nginx
location /analysis { rewrite ^/analysis(.*) /$1 break; proxy_pass http://127.0.0.1:5001; }
location /optionen { proxy_pass http://127.0.0.1:5051; }
location /         { proxy_pass http://127.0.0.1:5000; }
```

Statische Seiten gibt es nicht — alles läuft über Flask. Neue Seite =
Template unter `templates/` + Route in `app.py` + `systemctl restart`.
**Jinja cached Templates in Prod**: nach Template-Änderungen den jeweiligen
Service neu starten (dashboard *und* analysis *und* optionen, je nach Seite).

## Deploy / Restart

```bash
# Datei hoch (Zielpfad exakt — templates/ nicht vergessen)
scp -i ~/.ssh/boersenbot_key app.py veit@144.91.98.234:/home/veit/boersenbot/
# Service neu starten
ssh -i ~/.ssh/boersenbot_key veit@144.91.98.234 "sudo systemctl restart boersenbot_dashboard"
```

## ⚠️ Secrets

`setup_dashboard.sh` enthält Centron-DB-Zugangsdaten im Klartext. Dieselben
Creds liegen bereits in mehreren getrackten `.py` (`app.py`, `analysis_app.py`
u.a.) und in der Git-History. **Empfehlung:** DB-Passwort rotieren und künftig
über ENV/`.env` (gitignored) statt Hardcoding ziehen — eine Sanitierung nur
dieser einen Datei bringt nichts, solange die Creds anderswo im Repo stehen.
