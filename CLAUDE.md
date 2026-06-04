# CLAUDE.md

Leitfaden für Claude Code (claude.ai/code) bei der Arbeit in diesem Repository.

## Börsenbot — Projekt-Übersicht

**Was das ist:** Ein quantitatives **Trading-Edge-Research-Projekt**. Ziel ist
es, zuverlässige, automatisierbare Handels-Edges (primär auf SPY / S&P 500) zu
finden — und parallel ein konservatives Risk-Management-System live zu betreiben.
Begleitend gibt es ein Flask-Webfrontend (veitluther.de), das die Ergebnisse und
Live-Signale darstellt.

**Kein** simpler Yahoo-Daten-Scraper mehr (das war der Ursprung im Mai 2026).
Der Schwerpunkt liegt heute auf sauberer Backtest-Methodik und der Veröffentlichung
ehrlicher Befunde — inklusive der vielen falsifizierten Hypothesen.

**Deployment:** Contabo VPS (144.91.98.234, veitluther.de)
**Status:** Aktiv in Entwicklung; Webstack + ein Risk-Management-Bot laufen live.

---

## Forschungs-Methodik (zentral!)

Strategien werden **vor** dem Lauf als Hypothese mit Gates **G1–G5** vorregistriert
(siehe `preregs/` und die `*_PLAN.md`-Dokumente). Typisches Gate-Schema:

- **G1** In-Sample (z. B. 2014–2021): Edge vorhanden, t > Schwelle
- **G2** Out-of-Sample (z. B. 2022–2026): gleiche Parameter, Edge hält, t > 1.5
- **G3** Netto nach Kosten (Slippage + Gebühren, z. B. 5 bps + 0.1 %), t > 1.0
- **G4** Mindestzahl Signale/Monat
- **G5** Nicht durch COVID-Periode getrieben

Bei Parameter-Grids gilt **Bonferroni-Korrektur** (viele Tests → höhere
t-Schwelle). **OOS schlägt IS** — nur Out-of-Sample-Performance zählt. Befunde
werden committet, auch (gerade!) die FAILs. Survivorship-Bias wird vermieden, u. a.
über Point-in-Time-S&P-500-Mitgliedschaft (`sp500_hist_components.csv`, `g3_pit_ohlc.pkl`).

**Bisheriges Gesamtbild:** Einfache TA-Signale (MA-Crossover, Mean-Reversion,
Streaks) sind OOS falsifiziert — der Markt ist zu effizient. Der einzige bisher
deployte „Gewinner" ist ein **Hybrid Risk-Management-System** (Trend-Filter
MA50/200 + VIX-Vol-Sizing), OOS-Sharpe 0.90 vs SPY 0.74, ohne Overfitting.

---

## Architektur

### Datenquellen
- **Yahoo Finance** (`yfinance`) — OHLCV, primäre Kursquelle
- **Alpaca** (Paper Trading) — Streaming + 1D-Bars (`alpaca_streaming_simple.py`)
- **FRED** (`fred_helper.py`, API-Key in `.fred_key`, Cache `.fred_cache/`) — Makrodaten
- **GDELT** (`gdelt_market_tone_etl.py`) — News-Sentiment
- **Kaggle 1-Min-Block** — historische Minutendaten (mit Live-yfinance vereint)

### Datenbank (Centron SQL Server, geteilt mit Fußballbot)
- Server `158.181.48.77`, DB `dbdata`, Tabellen mit Präfix **`bb_`**
- `bb_Stocks`, `bb_StockPrices`, `bb_StockPrices_1min`, `bb_TechnicalIndicators`, `bb_Signals`
- View `bb_StockPrices_1min_Combined` (Kaggle ∪ Live, harmonisiert ET→UTC)
- Zugriff via `pymssql` (kein ODBC nötig)
- Separate lokale SQLite: `dividend_tracker.db` (reine Beobachtung, keine Empfehlung)

### Web-Stack (Flask, hinter nginx auf veitluther.de) — Details in `deploy/README.md`
| systemd-Unit | Port | App | Route |
|---|---|---|---|
| `boersenbot_dashboard` | 5000 | `app:app` | `/`, `/fazit`, `/done`, `/dividend-watch`, `/overnight-intraday` |
| `boersenbot_analysis` | 5001 | `analysis_app:app` | `/analysis` (ML-Predict, `--timeout 300`) |
| `boersenbot_optionen` | 5051 | `optionen_vergleich:app` | `/optionen` |
| `boersenbot_streaming` | – | `alpaca_streaming_simple.py` | kein HTTP |

Templates unter `templates/`. **Jinja cached in Prod** → nach Template-Änderung
betroffenen Service `systemctl restart`. Neue Seite = Template + Route in `app.py`
+ Restart.

---

## Repo-Struktur (Auswahl, ~75 Skripte)

```
boersenbot/
├── app.py / analysis_app.py / optionen_vergleich.py   # Flask-Frontends
├── templates/                                          # Jinja-Templates
├── deploy/                                             # systemd-Units + Deploy-README
├── preregs/                                            # Pre-Registrierungen (G1–G5)
├── *_PLAN.md / FINAL_STATUS.md                         # Strategie-Pläne & Befunde
│
├── Edge-Suche:      mean_reversion_*.py, crossover_*.py, streak_*.py,
│                    gap_bounce_*.py, sector_momentum_test.py, overnight_intraday_*.py
├── Externe Signale: fred_macro_tests*.py, vix_termstructure_filter.py,
│                    credit_stress_*.py, yen_stress_test.py, gdelt_market_tone_etl.py
├── Hybrid live:     hybrid_*.py, hyg_*.py, nas_*.py, monitor_hybrid.py
├── Dividenden:      dividend_tracker.py, *dividend_capture*.py
├── ML/Direction:    ml_spy_classifier.py, direction_predict.py, som_regime.py
├── Optionen:        warrant_search.py, warrant_fetch.py
├── Ausführung:      signal_to_orders.py, random5_crossover_bot.py
└── Utils:           timeutil.py, fred_helper.py, daily_refresh.py
```

Große `.pkl`/`.csv`-Caches (z. B. `sp500_ohlc_2015_2026.pkl`, `g3_pit_ohlc.pkl`)
sind **reproduzierbar** über die jeweiligen `*.py` und teils gitignored.

---

## Zugriff & Zugänge

### Contabo VPS (Production)
```bash
ssh -i ~/.ssh/boersenbot_key veit@144.91.98.234   # SSH-Key-Auth
# WorkingDir: /home/veit/boersenbot, venv/ aktiv, gunicorn aus venv
```

### Deploy / Restart (Beispiel)
```bash
scp -i ~/.ssh/boersenbot_key app.py veit@144.91.98.234:/home/veit/boersenbot/
ssh -i ~/.ssh/boersenbot_key veit@144.91.98.234 "sudo systemctl restart boersenbot_dashboard"
```

---

## Häufige Aufgaben

```bash
# Backtest/Strategie lokal laufen lassen (Beispiele)
python overnight_intraday_g3_pit.py        # Overnight vs Intraday, PIT-S&P-500
python sector_momentum_test.py
python hybrid_simple.py                     # deploytes Risk-Mgmt-System

# Tagesaktuelle Signale erzeugen
python hyg_signal_calculator.py             # -> hyg_today_signal.json
python daily_refresh.py

# Frontend lokal testen
python app.py                               # Flask-Dashboard
```

---

## Wichtige Hinweise

1. **Methodik vor Ergebnis:** Neue Strategie immer erst als Pre-Reg (G1–G5)
   formulieren, dann testen, dann committen — auch wenn sie scheitert.
2. **OOS ist heilig:** Niemals Parameter auf OOS-Daten tunen. Keine Survivorship-Bias.
3. **Reproduzierbarkeit:** Caches (`.pkl`/`.csv`) müssen sich aus den `*.py`
   regenerieren lassen; viele sind bewusst gitignored.
4. **Templates in Prod cachen:** Nach Template-Änderung den passenden Service neu starten.
5. **⚠️ Secrets:** DB-Credentials (Centron) liegen aktuell **hartcodiert** in
   mehreren getrackten `.py` (`app.py`, `analysis_app.py` u. a.) und in der
   Git-History. Bekanntes Problem — langfristig auf ENV/`.env` umstellen und
   Passwort rotieren. Keine neuen Secrets im Klartext committen.
6. **Robustheit:** Yahoo/Alpaca-Netzfehler abfangen; Rate-Limits respektieren.

---

## Doku & Historie

- `FINAL_STATUS.md` — Stand der Edge-Such-Kampagne (Option 1/2/3)
- `EDGE_SEARCH_MASTER_PLAN.md`, `EMAIL_REPORT_PLAN.md`, `EXTERNAL_SIGNALS_ROADMAP.md`
- `BOERSENBOT_HISTORY.md` — frühe Session-Historie (Ursprung als Daten-Scraper)
- `deploy/README.md` — VPS-Webstack, nginx-Routing, Deploy-Befehle
- `SESSION_*.md` — Detail-Logs (gitignored)
```
