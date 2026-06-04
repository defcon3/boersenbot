# Börsenbot — Session-Historie

Detaillierte Dokumentation aller Arbeitsschritte, Änderungen und Beschlüsse.

---

## Session 2026-05-15 — Projekt-Initialisierung & Deployment

**Zeitraum:** 2026-05-15, 08:00–08:25 UTC

**Teilnehmer:** Nutzer (veit), Claude Haiku 4.5

### Was wurde gemacht

1. **Projekt-Setup** (Teil 1)
   - Contabo VPS eingerichtet (144.91.98.234)
   - SSH-Key-Auth konfiguriert (boersenbot_private_key)
   - Git-Repo auf VPS initialisiert (`/home/veit/boersenbot`)

2. **Tech Stack aufgebaut**
   - Python 3.12 venv
   - Dependencies: yfinance, pandas, pymssql, requests
   - Centron SQL Server DB (dbdata) als Datenquelle (shared mit Fußballbot)

3. **Datenbankschema erstellt**
   - `bb_Stocks` — Aktienmetadaten (6 Aktien)
   - `bb_StockPrices` — OHLCV-Daten täglich
   - `bb_TechnicalIndicators` — Technische Indikatoren (später)
   - `bb_Signals` — Handelssignale (später)
   - **Präfix `bb_`** für Unterscheidung vom Fußballbot

4. **Yahoo Finance Datenabbau implementiert**
   - `data_fetch.py`: OHLCV-Import (1600 Zeilen × 6 Aktien = 9600 Preise)
   - `init_stocks.py`: Stocks-Tabelle initialisieren
   - `config.py`: Konfiguration (YAHOO_SYMBOLS, DB-Credentials)

5. **Probleme gelöst**
   - ✓ pyodbc → pymssql (keine ODBC-Treiber nötig)
   - ✓ yfinance MultiIndex-Columns handling
   - ✓ Foreign Key Constraint (Stocks-Tabelle voraus)
   - ✓ Tabellennamen mit bb_-Präfix für Clarity

6. **Cron-Job eingerichtet**
   - Täglich 09:00 UTC automatische Abfrage
   - Output: `logs/boersenbot.log`

### Commits (Git History)

1. `cb8612d` Initial Börsenbot Setup: Yahoo Finance + Centron SQL Server
2. `3b2efc0` Switch to pymssql (keine ODBC-Dependencies nötig)
3. `62d5f9b` First successful test: 9600 stock prices imported
4. `1a2de86` Rename tables with bb_-prefix for clarity

### Beschlüsse & Festlegungen

- **Datenquelle:** Yahoo Finance (yfinance library)
- **Datenbank:** Centron SQL Server, DB `dbdata`, Tabellen mit `bb_`-Präfix
- **Aktien:** AAPL, MSFT, GOOGL, AMZN, TSLA, SAP (konfigurierbar)
- **Historische Daten:** 2020-01-01 bis heute (1600+ Zeilen pro Aktie)
- **Automation:** Cron-Job täglich 09:00 UTC
- **Intervall:** Täglich (1d) — Basis für technische Indikatoren

### Verfügbare Daten (Stand 2026-05-15)

```
SQL Server: dbdata
├── bb_Stocks: 6 Einträge (AAPL, MSFT, GOOGL, AMZN, TSLA, SAP)
├── bb_StockPrices: 9600 Zeilen (Spalten: Symbol, Date, OHLCV, Volume)
├── bb_TechnicalIndicators: 0 Zeilen (später)
└── bb_Signals: 0 Zeilen (später)
```

### Offene Punkte / Roadmap

- [ ] Technische Indikatoren (RSI, MACD) berechnen
- [ ] Handelssignale generieren (Buy/Sell/Hold)
- [ ] Optionsdaten abrufen (yfinance)
- [ ] Analystenratings integrieren (ticker.recommendations)
- [ ] Grafana-Dashboard für Visualisierung
- [ ] Backtest-Framework für Strategien
- [ ] Auto-Trading (später, sicherheitsrelevant)

### Notizen für nächste Sessions

- **Cron-Job aktiv:** `0 9 * * * cd ~/boersenbot && source venv/bin/activate && python3 data_fetch.py >> logs/boersenbot.log 2>&1`
- **Memory-System etabliert:** Contabo VPS-Zugang + SQL Server Credentials persistent
- **yfinance Optionen:** 184 Info-Felder + History + Options + Ratings verfügbar (Dokumentation in Memory)
- **Logs monitorbar:** `tail -f ~/boersenbot/logs/boersenbot.log`

---

---

## Session 2026-05-15 (Fortsetzung) — Alpaca Integration & Backtest-Framework

**Zeitraum:** 2026-05-15, 08:25–09:35 UTC

**Teilnehmer:** Nutzer (veit), Claude Haiku 4.5

### Was wurde gemacht

1. **Alpaca API Fix**
   - ✓ Problem: REST.__init__() parameter naming (`api_key` → `key_id`)
   - ✓ Überprüfung der korrekten Signatur: `key_id`, `secret_key`, `base_url`
   - ✓ Erfolgreich verbunden mit Paper Trading Account
   - Credentials: API_KEY (PK7C52Q5VZXZ5DDOIDCEEY7CKD), Paper-Trading URL
   - **Status:** Paper Trading aktiv mit $200.000 Buying Power

2. **1-Minute Daten-Abruf versucht**
   - Alpaca-Datenbeschränkung: "subscription does not permit querying recent SIP data"
   - Paper Trading hat begrenzte Zugänge zu Live-Daten
   - Entscheidung: Mit historischen täglichen Daten für Backtest arbeiten
   - Alternative: Alpaca 1D bars funktionieren ✓

3. **Backtesting-Framework implementiert**
   - ✓ TA-Lib installiert für RSI/MACD/SMA Indikatoren
   - ✓ Mehrere Strategien verglichen:
     - RSI(30/70): AAPL +93%, TSLA +294%
     - MACD: AAPL +196%, TSLA +1120% (unrealistisch)
     - SMA(20/50): GOOGL +251%, TSLA +377%
   - ✓ Mit Stop-Loss & Transaktionskosten realistischer:
     - RSI mit 2% Stop-Loss: TSLA +16% (beste)
     - Meiste Strategien zeigen Verluste mit realistischen Kosten

4. **Trading Agent entwickelt**
   - ✓ `trading_agent.py`: Automatische RSI-basierte Signale
   - Thresholds: Buy <35, Sell >65
   - Aktuelle Signale (2026-05-15 09:31):
     - AAPL: SELL (RSI 74.8)
     - MSFT: HOLD (RSI 50.3)
     - GOOGL: SELL (RSI 74.2)
   - ✓ Cron-Job eingerichtet: Every 15 min während Market Hours (14:30-21:00 UTC)

5. **Portfolio Monitoring**
   - ✓ `check_portfolio.py`: Tägliche Account-Überwachung
   - Logs: `logs/portfolio.log` + `logs/trading_agent.log`

### Commits (neue Session)

- (Wird nach Code Review committed)

### Beschlüsse & Festlegungen

- **Strategie:** RSI-basiert (Oversold <35, Overbought >65)
- **Daten:** Alpaca 1D historisch für Backtesting + tägliche DB-Updates
- **Execution:** Paper Trading automatisch 15min-Intervalle
- **Risiko:** 2% Stop-Loss pro Trade
- **Ziel:** Tägliche +100-999$ Gewinne im Live Trading

### Verfügbare Daten (Stand 2026-05-15 aktuell)

```
SQL Server: dbdata
├── bb_Stocks: 3 Einträge (AAPL, MSFT, GOOGL)
├── bb_StockPrices: 9600 Zeilen (6 Aktien × 1600 Tage)
├── bb_StockPrices_1min: 0 Zeilen (Alpaca-Limit)
├── bb_TechnicalIndicators: 0 Zeilen
└── bb_Signals: 0 Zeilen

Alpaca Account:
├── Portfolio Value: $200,000 (Paper)
├── Buying Power: $200,000
├── Strategy: RSI-based signals
└── Cron: Every 15 min (US Market Hours)
```

### Probleme gelöst

1. ✓ Alpaca parameter naming: `api_key` → `key_id`
2. ✓ Alpaca data subscription: 1D bars statt 1Min
3. ✓ Backtesting realism: Stop-Loss + Transaktionskosten hinzugefügt
4. ✓ Bash script escaping: Anführungszeichen korrekt gehandhabt

### Offene Punkte / Roadmap

- [ ] Live Trading starten (Paper Trading testing)
- [ ] Performance Dashboard (Daily P&L tracking)
- [ ] Strategie-Optimierung basierend auf Live-Ergebnissen
- [ ] Multi-Symbol Portfolio Rebalancing
- [ ] Risk Management Enhancement (Position Sizing)
- [ ] Alternative Datenquellen (Kaggle 1Min Data)
- [ ] ML-basierte Signal-Verbesserung (später)

### Notizen für nächste Sessions

- **Trading Agent läuft:** Cron job active, signals generated every 15 min
- **Paper Trading:** Keine realen Gewinne/Verluste, nur Backtesting momentan
- **Nächster Schritt:** Manueller Backtest auf echten Trades, dann Live execution testen
- **Logs:** `tail -f ~/boersenbot/logs/trading_agent.log` oder `portfolio.log`

---

## Nächste Session — TODO

1. Live Paper Trading starten und Monitor 1 Woche
2. Performance evaluieren (Win Rate, Average Profit)
3. Strategie optimieren (RSI thresholds anpassen)
4. Risk Management verbessern (Position Sizing, Drawdown limits)
5. Optional: Alpaca Kaggle 1-Min Daten integrieren

---

> **Lücke 2026-05-16 … 2026-06-03 (Edge-Such-Kampagne).** In diesem Zeitraum
> verschob sich das Projekt vom RSI-Trading-Agent zu disziplinierter
> Edge-Forschung mit Pre-Reg-Gates (G1–G5), Bonferroni, OOS-Tests. ~60
> Hypothesen getestet, die meisten OOS falsifiziert; deployed wurden Hybrid-SPY
> + HYG-Stress-Buy (Combined). Nicht hier protokolliert — siehe
> `FINAL_STATUS.md`, `EDGE_SEARCH_MASTER_PLAN.md`, `EXTERNAL_SIGNALS_ROADMAP.md`,
> die `preregs/` und das Git-Log.

---

## Session 2026-06-04 — Grok-Review-Runde: Overnight robust geschlossen + Leverage-Sackgasse

**Teilnehmer:** Nutzer (veit), Claude Opus 4.8

**Ausgangspunkt:** Externes Review (Grok) zur Fazit-Seite und zur
Overnight/Intraday-Analyse. Zwei konkrete Vorschläge wurden umgesetzt
(Rolling/Bootstrap-Robustheit, ES-Futures-Handelbarkeit) plus die
Leverage/Combined-Idee als Zusatz.

### Was wurde gemacht

1. **CLAUDE.md an realen Projektstand angepasst**
   - Von „Initial setup / Yahoo-Scraper" → Quant-Edge-Research + Flask-Webstack
   - Methodik (G1–G5, Bonferroni, OOS, PIT), Centron-SQL-Server, Web-Services,
     reale Repo-Struktur, SSH-Key statt Klartext-Passwort dokumentiert
   - Secrets-Hinweis als **bewusst akzeptiert** markiert (keine .env-Umstellung)

2. **Overnight/Intraday — Robustheit (Grok-Vorschlag 1)**
   - `overnight_intraday_rolling_bootstrap.py`: ersetzt den schwachen
     1-Median-Split durch Rolling-Fenster (3J/5J), Stationary-Bootstrap-KIs
     und sup-Wald/QLR-Strukturbruchtest mit Block-Bootstrap-Kritikwerten
   - Befund: deskriptiv robust (on>id in 89,5 % der 5J-Fenster, Sharpe-Diff
     +0,65, p=0,015), aber mittleres d_t grenzwertig (p=0,059); Strukturbruch
     2008 **nicht** signifikant (p≈0,20)
   - **Patton-Politis-White** automatische Blocklänge als Robustheits-Check
     (selbst implementiert, kein `arch`): p 0,059→0,050, Schluss STABIL

3. **ES-Futures — Handelbarkeit (Grok-Vorschlag 2)**
   - Pre-Reg `preregs/overnight_es_futures_2026_06_04.md` (bewusst als
     gefährlichste Hypothese markiert: Gate-Hacking-Risiko, Daten-K.o. G0)
   - `overnight_es_futures_g0check.py`: **G0 FAIL** — yfinance `ES=F` ist nicht
     session-sauber (Globex-Vollsession; ES=F-„Overnight" −35,9 % vs Intraday
     +954,9 %; Overnight-Korr zu SPY nur +0,24), freie Intraday-Historie zu kurz
   - Pre-Reg diszipliniert verworfen statt mit Approximation zu rechnen

4. **Overnight-Strang YELLOW final geschlossen + deployed**
   - `templates/overnight_intraday.html`: neue Sektion „6 · Robustheit &
     finaler Abschluss" + YELLOW-Schließungsbox
   - Auf VPS deployed (Backup → scp → `systemctl restart boersenbot_dashboard`),
     live verifiziert (veitluther.de/overnight-intraday, HTTP 200)

5. **Leverage/Combined (Grok-Vorschlag 3) — Sackgasse, sauber belegt**
   - `combined_leverage_test.py`: konstante Leverage 1,25–2× auf Combined 50/50
     mit Financing. Selbst ×2 schlägt SPY nicht (+14,1 % vs +20,8 %), Sharpe
     sinkt (1,31→0,98), COVID-Tail bläht DD auf −44 %
   - `combined_voltarget_test.py`: Vol-Targeting rettet Sharpe **nicht** (bei
     gleicher Vol 0,74 vs konstant 0,96 — nachlaufende Vol delevert zu spät);
     einzige Stärke Tail-Kontrolle, die unleveraged ohnehin besser ist
   - Fazit: unleveraged Combined-Sleeve (Sharpe 1,31, MaxDD −9,5 %) ist der
     Sweet Spot; Leverage-Strang geschlossen

6. **Infrastruktur**
   - `fred_helper.py`: hartkodierter Windows-Pfad → repo-relativ (portabel,
     auch Linux-VPS), Env-Override `FRED_KEY_FILE`/`FRED_CACHE_DIR`
   - Combined-Tests + `fred_helper.py` + `.fred_key` (chmod 600) auf VPS
     deployed; Verifikationslauf erfolgreich

7. **Grok-Kommunikation** (User hat kein Bezahl-Abo)
   - Zwei Copy-Paste-Review-Dokumente auf dem Desktop aufbereitet
     (`review-fuer-grok.md`, `review-fuer-grok-2.md`) mit Methoden, exakten
     Zahlen und gezielten Rückfragen

### Commits (diese Session)

- `7a5f3e47` CLAUDE.md an realen Projektstand angepasst
- `60fdee1c` CLAUDE.md: Secrets-Hinweis als bewusst akzeptiert
- `97b3b371` Rolling-Stabilität + Bootstrap-KI + Strukturbruchtest
- `0c6b0487` Pre-Reg-Anchor ES-Futures (vorsichtig)
- `c21ce7ce` PW-Robustheit + ES G0-FAIL + YELLOW-Schließung
- `1eace7fc` Combined-Leverage-Test (Grok-Vermutung nicht bestätigt)
- `5049debe` fred_helper: Pfade repo-relativ
- `05c51509` Combined-Vol-Targeting-Test

### Beschlüsse & Festlegungen

- **Overnight-Effekt:** reales Stylized Fact, aber kein handelbarer Edge —
  YELLOW, Strang abgeschlossen (auch Futures-Weg an Datenrealität gescheitert)
- **Leverage:** erzeugt keinen Edge, opfert nur den Downside-Schutz der
  Combined — als Sackgasse abgehakt
- **Secrets:** bewusst hartkodiert/auf VPS akzeptiert, keine .env-Umstellung
- **Pre-Reg-Disziplin:** G0-Vorab-K.o. hat funktioniert (kein Rechnen mit
  kaputter Approximation)

### Offene Punkte

- Grok-Rückmeldung zu `review-fuer-grok-2.md` (60d-Vol/Vol-of-Vol als
  allerletzter Rest-Check des Vol-Targetings) — optional
- True-OOS 2026+ bleibt der entscheidende Test für die deployten Strategien
