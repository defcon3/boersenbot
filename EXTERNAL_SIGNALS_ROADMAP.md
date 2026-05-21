# Externe Signale Roadmap — MA mit Makro/Event-Daten

**Ziel:** Validiere externe Datenquellen (nicht TA) mit ehrlicher Pre-Reg + OOS-Falsifikation.

**Prinzipien (wie in den letzten 3 Tests):**
- Pre-Reg Gates G1..G5 VORHER festnageln
- OOS-Split 2022-01-01, COVID-Ausschluss 2020-02-15..04-30
- Bonferroni-Korrektur über mehrere Tests
- Alle Ergebnisse reportet, auch Fehler

---

## Reihenfolge (Machbarkeit + Signalstärke-Potenzial)

### 1. EARNINGS-ÜBERRASCHUNGEN (H1)
**Hypothese:** "Tage mit vielen positiven Earnings-Überraschungen (>5%) haben positive Excess-Returns."

**Datenquelle:** 
- Earnings-Kalender + Surprises (Quandl, IEX, oder einfach yfinance earnings-dates)
- Definition: "Überraschung" = (actual - estimate) / estimate > 5%

**Pre-Reg Gates (G1..G5):**
- G1: IS 2014-2021, Excess vs SPY Mittel > 0, t > +2.0
- G2: OOS 2022-2026, Excess > 0, t > +1.5
- G3: Netto @ 5 bps round-trip OOS: Mittel > 0, t > +1.0
- G4: Median offene Positionen pro Tag >= 5
- G5: Nicht durch COVID dominiert (Test ohne COVID ebenfalls > 0)
- **Bonferroni:** |t| > 2.95 für OOS-Signifikanz (gegen 11 andere Tests)

**Status:** PENDING

---

### 2. FED-RATE-ÄNDERUNGS-WAHRSCHEINLICHKEIT (H2)
**Hypothese:** "Tage, an denen die Wahrscheinlichkeit auf Fed-Zinserhöhung steigt, haben negative Returns (Rezessions-Angst)."

**Datenquelle:**
- CME FedWatch Tool (täglich, kostenlos, scrape-bar oder API)
- Variable: Prob(nächste Hike) − Prob(nächste Hike am Vortag)
- Wenn Delta > 0 (Hike-Wahrscheinlichkeit steigt): Long SPY oder Short?

**Pre-Reg Gates:**
- G1: IS, wenn Delta > 0.05: Negative Returns vs SPY, t < −2.0
- G2: OOS, Negative Excess, t < −1.5
- G3: @ 5bps netto, t < −1.0
- G4: Median events pro Monat >= 3 (nicht zu wenig Datenpunkte)
- G5: Nicht nur in Rezessionen gültig

**Challenge:** CME-Daten nur sporadisch täglich verfügbar (eher wöchentlich Update). Weniger Datenpunkte.

**Status:** PENDING

---

### 3. RENTENKURVEN-STEIGUNG (2Y-10Y SPREAD) (H3)
**Hypothese:** "Tage, an denen der 2Y-10Y-Spread sinkt (Inversionstrend), haben negative Excess-Returns (Rezessions-Signal)."

**Datenquelle:**
- FRED API (Federal Reserve Economic Data, kostenlos)
- Serie: DGS2 (2-Year), DGS10 (10-Year)
- Variable: Spread(t) − Spread(t-1) < 0 (Kurve wird flacher)

**Pre-Reg Gates:**
- G1: IS, wenn Spread-Delta < −10bps: Negative Excess, t < −2.0
- G2: OOS, Negative Excess, t < −1.5
- G3: @ 5bps netto, t < −1.0
- G4: Mindestens 2 Änderungen pro Monat
- G5: Vorher/nachher einer tatsächlichen Rezession (2020, 2022+) signalsicher?

**Status:** PENDING

---

### 4. MAKRO-ÜBERRASCHUNGEN (CESIUSD INDEX) (H4)
**Hypothese:** "Hohe Makro-Überraschungen (ISurprise > +50) → positive Returns; tiefe ISurprise (< −50) → negative."

**Datenquelle:**
- Citibank Economic Surprise Index (frei verfügbar via Trading View / FRED)
- Täglich (wenn verfügbar)
- Interpretation: > 0 = wirtschaftliche Daten überraschen positiv

**Pre-Reg Gates:**
- G1: IS, ISurprise > +50 vs < −50, Excess-Differenz > 0, t > +2.0
- G2: OOS, gleich, t > +1.5
- G3: @ 5bps, t > +1.0
- G4: Mindestens 5 Tage pro Monat mit |ISurprise| > 50
- G5: Nicht nur während Fed-Zinserhöhungs-Phase

**Status:** PENDING

---

### 5. VIX-TERM-STRUKTUR (H5) [Optional, falls 1-4 alle falsifizieren]
**Hypothese:** "Flache/inverse VIX-Term-Struktur (VIX-3M < VIX-1M) signalisiert Volatility-Überschwang → kontrarian Long?"

**Datenquelle:**
- CBOE VIX Futures Kurven (yfinance oder CBOE-Website)
- Täglich verfügbar

**Status:** OPTIONAL (nur wenn 1-4 interessant wirken)

---

## Durchführungs-Plan

1. **Skripte bauen** (parallel möglich):
   - `earnings_surprise_test.py`
   - `fed_rate_test.py`
   - `yield_curve_test.py`
   - `macro_surprise_test.py`

2. **Tests laufen lassen** (der Reihe nach, weil Abhängigkeiten):
   - H1 (Earnings) — wenn PASS → dranbleiben, wenn FAIL → nächste
   - H2 (Fed-Rates) — unabhängig, auch parallel möglich
   - H3 (Yield Curve)
   - H4 (Macro Surprise)

3. **Resultate dokumentieren:**
   - Für jeden Test: Ergebnis-Tabelle (wie bei Sektoren)
   - Bonferroni-Korrektur über alle 4 Tests (|t| > 3.35 statt 2.95)
   - Zusammenfassung: Welche Hypothesen halten OOS?

4. **Nächste Session entscheiden:**
   - Wenn eine Hypothese Bonferroni hält: Kombinieren mit TA oder Solo traden?
   - Wenn alle falsifizieren: Hypothesen-Raum ist erschöpft, andere Strategie?

---

## Metadaten

- **Beginn:** 2026-05-21
- **Universum:** S&P 500 (wie zuvor, 98 Ticker mit OHLC)
- **Pre-Reg-Discipline:** Vor Lauf schriftlich fixieren, dann erst rechnen
- **Code-Basis:** Klone aus `cc_meanrev_excess.py` (Excess-Methodik)

---

## Status pro Test

| Test | Hypothese | Status | Commit | Notes |
|------|-----------|--------|--------|-------|
| H1 | Earnings-Surprise | PENDING | — | Datenquelle validieren |
| H2 | Fed-Rate-Delta | PENDING | — | CME-Daten-Verfügbarkeit prüfen |
| H3 | Yield-Curve | PENDING | — | FRED-API Setup |
| H4 | Macro-Surprise | PENDING | — | ISurprise-Quelle testen |
| H5 | VIX-Term | OPTIONAL | — | Nur wenn 1-4 interessant |

