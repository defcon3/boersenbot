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
| `boersenbot_green_up.service` | simple/Loop | Hedge-Daemon für explizit benannte Positionen (`bb_GreenUpHedges`); Autopilot überspringt deren Verkauf |
| `boersenbot_football_odds.service` | simple/Loop | minütliche Polymarket-Quoten → `bb_FootballOdds_1min` |
| `boersenbot_football_backfill.service` | simple/Loop | Endstände → `bb_FootballMatches` |
| `boersenbot_tennis_paper.timer` | **Timer** (30 min) | Pre-Match-Snapshot ATP/WTA → `bb_TennisPaperBets` |
| `boersenbot_tennis_settle.timer` | **Timer** (alle 6h) | Sieger nachtragen (Jupiter-Events fallen nach ~1–2 Tagen raus) |
| `boersenbot_weather_ladder.timer` | **Timer** (täglich 12:30 UTC) | Jupiter-Preisleitern → `bb_WeatherLadders` + METAR-Settle-Backfill |
| `boersenbot_eps_logger.timer` | **Timer** (täglich 07:00 UTC) | EPS-Member-Tagesmaxima (28 Städte × 122 Member) → `preregs/weather_eps_log.csv`; **führende Reihe liegt auf dem VPS** — vor der Auswertung per scp holen + committen (Pre-Reg `preregs/weather_eps_sigma_prereg_2026_07_18.md`) |

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

## Backup VPS → NAS → Infomaniak (seit 03.08.2026)

Contabo verkauft Backups nur im Premium-Vertrag — stattdessen sichert das
Synology-NAS den VPS selbst. **Das NAS zieht**, weil es im LAN steht und der VPS
nicht hineinreicht.

```
VPS /home/veit  --rsync/ssh (pull)-->  NAS ~/contabo_backup/home/
                                        └─ CloudSync-Session 10 --> Infomaniak kDrive
                                           (WebDAV, upload-only, /DiskStation_backup/home)
```

| Wann | Wo | Was |
|------|-----|-----|
| 03:40 täglich | VPS-Crontab (`veit`) | `deploy/collect_system_config.sh` → sammelt systemd-Units, nginx-Config, Crontab, dpkg-Liste, pip-freeze nach `/home/veit/_backup_meta/` (alles, was **außerhalb** von `/home/veit` liegt und der Pull sonst nicht sähe) |
| 04:00 täglich | NAS `/etc/crontab` (`benutzername`) | `deploy/nas_contabo_backup.sh` → `rsync`-Pull von ganz `/home/veit` |

**Semantik: additiv, kein `--delete`.** Was auf dem VPS gelöscht wird, bleibt auf
NAS und in der Cloud liegen. Umfang: komplettes `/home/veit` inkl. `venv/` und
`data/` (~9,5 GB, 61 k Dateien); ausgenommen nur der gunicorn-Socket.

**Zugang:** eigener Key `~/.ssh/vps_backup` auf dem NAS. Auf dem VPS ist er in
`authorized_keys` gebunden an
`restrict,command="/usr/bin/rrsync -ro /home/veit"` → **nur Lesen, keine Shell**.
Ein Login-Versuch bringt `SSH_ORIGINAL_COMMAND does not run rsync`. Deshalb kann
der NAS-Job den Collector auch nicht selbst anstoßen — daher der eigene
VPS-Cron um 03:40.

```bash
# Status prüfen (vom PC aus, NAS-SSH: Passwort, SFTP ist aus → base64-Pipe)
ssh benutzername@192.168.178.32 \
  'cat ~/contabo_backup/LAST_SUCCESS.txt; tail -20 ~/boersenbot/contabo_backup.log'
# Lauf von Hand
ssh benutzername@192.168.178.32 '~/boersenbot/nas_contabo_backup.sh'
```

**Synology-Fallen:** `/etc/crontab` braucht **Tabs** als Trenner (Leerzeichen →
Zeile wird stillschweigend ignoriert); Neustart per
`sudo /usr/syno/bin/synosystemctl restart crond` (`synosystemctl` liegt nicht im
PATH). `flock` verhindert Überlappung — der Erstlauf dauerte ~1 h bei ~3 MB/s
(CPU des DS215j ist die Bremse), die täglichen Deltas sind klein.

**Wiederherstellung** (neuer VPS): `~/contabo_backup/home/` vom NAS auf den neuen
Server schieben (`rsync -av`, diesmal als Push vom NAS aus — der Key darf nur
lesen), dann aus `_backup_meta/` die Units nach `/etc/systemd/system/`, die
nginx-Configs nach `/etc/nginx/` und `cron/crontab_veit.txt` per `crontab -`
zurückspielen. `venv/` liegt zwar im Backup, ist aber pfadgebunden — bei
abweichendem Zielpfad neu bauen aus `system/pip_freeze_*.txt`.
Let's-Encrypt-Zertifikate sind **nicht** im Backup (root-only) → Certbot neu.

**Bewusst in Kauf genommen:** Das Backup enthält `/home/veit/.ssh`, `config.py`
und `.fred_key` — Secrets landen damit auch bei Infomaniak. Passt zur
Hardcoding-Entscheidung unten, ist aber der Preis für „alles 1:1".

## ⚠️ Secrets

`setup_dashboard.sh` enthält Centron-DB-Zugangsdaten im Klartext. Dieselben
Creds liegen bereits in mehreren getrackten `.py` (`app.py`, `analysis_app.py`
u.a.) und in der Git-History. **Empfehlung:** DB-Passwort rotieren und künftig
über ENV/`.env` (gitignored) statt Hardcoding ziehen — eine Sanitierung nur
dieser einen Datei bringt nichts, solange die Creds anderswo im Repo stehen.
