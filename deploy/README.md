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
