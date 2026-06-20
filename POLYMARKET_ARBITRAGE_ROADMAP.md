# Polymarket ↔ Buchmacher Arbitrage — Roadmap

**Status:** Pre-Research  
**Priorität:** Phase 1 (vor SPY-Signalquelle)  
**Methodik:** Strikt pre-reg, G1–G5 Gates, ehrliche Befunde auch bei FAILs

---

## Phase 1: Mess-Infrastruktur & Datenerfassung (Wochen 1–3)

**Ziel:** Zero-Capital-Exploration. Verstehe die echte Verteilung von Diskrepanzen, bevor ein Euro aufs Spiel gesetzt wird.

### Task 1.1: Read-only Logger — Polymarket
- **Input:** Polymarket CLOB/Gamma-API (https://gamma-api.polymarket.com/)
- **Output:** JSON-Stream (5min-Polling)
  - Event ID, Title, Yes/No Prices, Depth, Last Update
  - Speichern in `polymarket_raw.log` (SQLite oder CSV)
- **Tools:** `polymarket_logger.py` (asyncio, requests)
- **KPI:** Uptime > 95%, Latency < 5s

### Task 1.2: Read-only Logger — Buchmacher
- **Quellen Priorität:**
  1. **Betfair Exchange API** (Wettbörse, Kommission 2–5%, effizienteste Preise)
  2. **Pinnacle API** (Sportwettanbieter, toleriert Gewinner)
  3. **Odds-Aggregator** (z. B. OddsAPI, falls Betfair/Pinnacle Limits)
- **Kategorie:** Sports (Tennis, Fußball, Boxer) + Politics (US Elections, Rezession)
- **Output:** Parallel-Logs für beide Quellen
- **Tools:** `bookmaker_logger.py`
- **KPI:** Mind. 2 Quellen, Latenz < 2s

### Task 1.3: Match & Spread-Kalkulator
- **Input:** Polymarket-Log + Buchmacher-Log
- **Logik:**
  - Event-Matching (Title Levenshtein + Manual Mapping)
  - Spread-Berechnung: `|(Poly Yes) - (Bookie Yes)| / avg(Poly, Bookie)`
  - Friction: Polymarket (0,2% Taker Fee) + Bookie (5% Wettsteuer in DE)
  - **Net Arb %** = Spread – Friction
- **Output:** CSV `arbitrage_spreads.csv` (Timestamp, Event, Poly-Odds, Bookie-Odds, Spread%, Net%)
- **Tool:** `spread_calculator.py`

### Task 1.4: Datensammlung (2 Wochen mindestens)
- **Zeitfenster:** Mo–Fr 09:00–22:00 CEST (Marktstunden)
- **Ziel:** ~500–1000 Event-Snapshots sammeln
- **Metrik:** Verteilung der Net-Arbs, Häufigkeit > 2%, > 5%, > 10%

---

## Phase 2: Risiko-Validierung (Woche 3–4)

**Gating:** Nur wenn Phase 1 zeigt, dass echte Arbs > 3% netto häufig genug vorkommen.

### Task 2.1: Resolution-Match-Rate
- **Input:** Abgeschlossene Events aus Phase 1 (mind. 100 Events)
- **Prozess:**
  - Manuell/scrape beide Seiten: Wie hat Polymarket resolved? Wie der Buchmacher?
  - Match-Rate = % identischer Outcomes
  - Dokumentiere Diskrepanzen (AGB-Unterschiede, Void-Handling, etc.)
- **Gate G1:** Match-Rate > 95% (sonst Resolution-Risiko zu hoch)
- **Output:** `resolution_analysis.md` + Diskrepanz-Katalog

### Task 2.2: Legging-Risiko Simulation
- **Input:** Historische Preis-Serien (Poly + Bookie, 1min-Auflösung)
- **Prozess:**
  - Simuliere: „Fill auf Polymarket bei t=0, Fill auf Bookie bei t=5s–30s"
  - Wie oft dreht sich der „Arb" ins Negative?
  - Worst-Case: Quote bewegt sich gegen dich
- **Gate G2:** Leg-Flip-Rate < 20% der Opportunities
- **Output:** `legging_risk_analysis.csv`

### Task 2.3: Kapitalbindungs-Annualisierung
- **Input:** Typische Event-Duration (Politik: 3–6 Wochen, Sports: Stunden)
- **Formel:** 
  ```
  APR = (Net Arb %) × (365 / Duration in Days)
  ```
- **Gate G3:** APR > 10% after Friction (bei 5% Arb über 6 Wochen = 4.3% APR → FAIL)
- **Output:** Realistischer APR-Forecast

---

## Phase 3: Limit-Test & Scaling (Woche 5+)

**Gating:** Nur wenn G1–G3 bestanden.

### Task 3.1: Konto-Eröffnung (Minimal-Capital)
- Betfair Exchange (Wettbörse)
  - Min. Deposit: €50
  - Strategie: Limit-Test mit 1–5 € Einsätze
- Pinnacle (Sportwettanbieter)
  - Min. Deposit: €20
  - Reputation: Toleriert Gewinner besser
- Polymarket
  - USDC on Polygon (MetaMask)
  - Min. ~$100 für Testing

### Task 3.2: Live Limit-Experiment (2 Wochen)
- Platziere 20–30 echte Arb-Wetten (kleine Größen)
- Messe:
  - Acceptance Rate (werden alle Orders filled?)
  - Gubbing Signs (Account-Limits, Reduzierung)
  - Fill-Latenz (Seconds zwischen Poly/Bookie)
- Gate G4: Scaling-Potential erkannt oder Account limitiert
- **Output:** `live_test_log.md`

### Task 3.3: Deutsche Steuer-Nachrechnung
- Buchmacher (DE-lizenziert): 5% Wettsteuer
- Polymarket (Krypto): Einkommen + Gewinnebesteuerung
- Konsult. Steuerberater falls nötig
- Gate G5: Netto-APR nach Steuern noch > 5%?

---

## Phase 4: Entscheidung & Pivot (Woche 6)

| Gate | Ausgang | Aktion |
|---|---|---|
| **G1** (Match-Rate < 95%) | FAIL | Resolution-Risiko zu hoch → **Abort** |
| **G2** (Legging-Flip > 20%) | FAIL | Latenz-Sync nicht möglich → **Abort** |
| **G3** (APR < 10% netto) | FAIL | Zu viel Kapitalbindung → **Abort** |
| **G4** (Gubbing/Limits) | FAIL | Nicht skalierbar → **Abort** |
| **G5** (Steuer frisst Marge) | FAIL | Wirtschaftlich unmöglich (DE) → **Abort** |
| **Alle bestanden** | PASS | Live-Bot mit echtem Geld (Phase 5) |
| **Irgendwo FAIL** | — | → **Pivot zu Phase 2B: Polymarket als Signalquelle** |

---

## Phase 2B: Polymarket als Makro-Signal (Fallback)

Falls Arbitrage in Phase 1–4 scheitert: Nutze die gesammelten Daten als **externe Signalquelle für SPY-Trading**:

- **Implizite Wahrscheinlichkeiten:**
  - Fed Rate Hold vs. Futures-Diskrepanz
  - Rezessions-Wahrscheinlichkeit (Polymarket) vs. VIX/Credit Spreads
  - Election Outcome vs. Sector Rotation
- **Pre-reg:** G1–G5 für 2–3 Makro-Predictions
- **Integration:** In bestehende Pipeline (FRED + GDELT + VIX + **Polymarket**)

---

## Deliverables & Timeline

| Phase | Deliverable | ETA |
|---|---|---|
| 1 | `polymarket_logger.py`, `bookmaker_logger.py`, `spread_calculator.py` | +2w |
| 1 | `arbitrage_spreads.csv` (2 Wochen Daten) | +4w |
| 2 | `resolution_analysis.md`, `legging_risk_analysis.csv` | +5w |
| 2 | `POLYMARKET_ARBITRAGE_GATE_REPORT.md` (G1–G5 Results) | +6w |
| 3+ | Entscheidung: Live-Bot oder Pivot zur Signalquelle | +7w |

---

## Ressourcen & API-Zugang

### Polymarket
- **Gamma API (read-only):** https://gamma-api.polymarket.com/docs/
  - Kein Auth nötig, Rate Limit: großzügig
  - Endpoints: `/markets`, `/orderbook/{id}`, `/trades`

### Betfair (Preferred)
- **API:** https://docs.betfair.com/display/1smk3HBS/Betfair+Core+API+Docs
- **Auth:** API-Key + Certificate
- **Kommission:** 2–5% (Netto)

### Pinnacle
- **API:** https://www.pinnacle.com/en/api/sports/
- **Vorteil:** Keine Limits für Gewinner

### Odds-API (Fallback)
- **https://theOddsAPI.com** — Aggregator für mehrere Books

---

## Notizen zur Methodik (nach Fable)

**Warum diese Reihenfolge:**
1. **Read-only first** — Null Kapital, maximale Info
2. **Risiken in Größenordnung:** Resolution > Legging > Kapitalbindung > Limits > Steuern
3. **Honest Gates:** Jede FAIL stoppt echtes Geld (nicht Wunschdenken)
4. **Fallback Plan:** Wenn Arb scheitert, nicht verloren — Signalquelle ist Plan B

**Wenn Phase 1 zeigt, dass Spreads < 2% netto sind:**
→ Sofort zu Phase 2B pivoter (Signalquelle), keine Live-Tests mehr

**Commit-Kultur:**
- Jeden Befund committen, inklusive FILLs
- Session-Logs in `polymarket_sessions/`
- Final Report: `FINAL_POLYMARKET_STATUS.md` (komplett ehrlich)

---

## Nächste Schritte

1. ✅ Roadmap approve
2. → Task 1.1: Polymarket-Logger-Scaffold (skeleton, testing)
3. → Task 1.2: Buchmacher-Auswahl finalisieren (Betfair vs. Pinnacle API-Check)
4. → Datensammlung-Loop starten (24/7 Polling)
