# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Börsenbot — Projekt-Übersicht

**Ziel:** Automatisierter Bot zur Datenabfrage von Yahoo Finance, Analyse und Signalgenerierung.

**Deployment:** Contabo VPS (144.91.98.234)

**Status:** Initial setup

---

## Zugriff & Zugänge

### Contabo VPS (Production/Development)

```bash
ssh veit@144.91.98.234
# Passwort: abc
```

**Speicherorte für Daten/Code auf VPS:**
- Code: `~/boersenbot/` (zu erstellen)
- Daten: `~/data/` (zu erstellen)
- Logs: `~/logs/boersenbot.log`

---

## Technologie Stack

- **Sprache:** Python 3.8+
- **Datenquelle:** Yahoo Finance (yfinance library)
- **Datenbank:** SQLite (lokal) oder PostgreSQL (optional später)
- **Automation:** Cron-Jobs auf VPS
- **Monitoring:** Logs zu Datei

---

## Datenbeschaffung von Yahoo Finance

**Aktuell geplant:**
- Stock-Kurse (Open, High, Low, Close, Volume)
- Historische Daten (OHLCV)
- Echtzeit-Quotes
- Technische Indikatoren (später)

**yfinance-Installation:**
```bash
pip install yfinance pandas
```

**Beispiel (lokal testen):**
```python
import yfinance as yf
data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')
print(data.head())
```

---

## Projekt-Struktur (geplant)

```
boersenbot/
├── data_fetch.py          # Yahoo Finance Datenabbau
├── db.py                  # Datenbank-Verwaltung
├── signals.py             # Signal-Generierung (später)
├── config.py              # Konfiguration
├── requirements.txt
├── logs/
└── data/                  # CSV/DB-Dateien
```

---

## Häufige Commands

### Setup auf VPS (Initial)

```bash
ssh veit@144.91.98.234
mkdir -p ~/boersenbot ~/data ~/logs
cd ~/boersenbot
git clone <REPO_URL>  # falls Remote-Repo existiert
pip install -r requirements.txt
```

### Daten abrufen (manuell)

```bash
python data_fetch.py --symbol AAPL --start 2024-01-01
```

### Logs prüfen

```bash
tail -f ~/logs/boersenbot.log
```

### Cron-Job für tägliche Abfrage (später)

```bash
0 9 * * * cd ~/boersenbot && python data_fetch.py --daily >> ~/logs/boersenbot.log 2>&1
```

---

## Wichtige Hinweise

1. **Credentials sicher speichern:** Keine API-Keys oder Passwörter in Code-Commits
2. **Fehlerbehandlung:** Netzwerk-Fehler bei Yahoo-Abfragen robusthaft handhaben
3. **Rate Limiting:** Yahoo Finance hat Limits — nicht zu häufig abfragen
4. **Logging:** Alle Aktionen loggen (erfolg, Fehler, Timestamps)

---

## Session-Historie & Änderungen

Siehe `~/BOERSENBOT_HISTORY.md` für detailliertes Changelog.

---

## Nächste Schritte (Roadmap)

- [ ] Git-Repo initialisieren (lokal oder auf VPS)
- [ ] Basis-Struktur erstellen (data_fetch.py, config.py)
- [ ] Yahoo Finance Datenabbau testen
- [ ] Datenbank-Schema designen
- [ ] Cron-Job einrichten für automatische Abfragen
- [ ] Signal-Generierung entwickeln (später)
