# Pre-Registered Hypothesis #2: Selective Dividend Capture DE (Top-10 aus Train)

**Status:** Pre-registered (vor Test-Lauf)
**Datum:** 2026-05-25
**Autor:** Veit Luther (mit Claude Opus 4.7 als Sparring)
**Commit-Anchor:** dieses File wird VOR dem Test-Lauf committed und gepusht.
**Vorlauf:** Pre-Reg #1 ([`dividend_capture_de_2026_05_25.md`](dividend_capture_de_2026_05_25.md), Commit `871bac1b`) hat die universelle Strategie auf allen 90 Aktien getestet und falsifiziert (mean Test-Excess negativ in allen 5 Hold-Windows). Im Post-Mortem zeigte sich aber: einige Aktien hatten *deskriptiv* positive Dividenden-Captures. Sanity-Check (Random-5d-Fenster vs Div-5d-Fenster) ergab, dass die Outperformance der Top 10 *spezifisch auf Div-Tage konzentriert* ist (+2.71 % vs ~0 %). Diese neue Pre-Reg testet, ob das ein echter selektiver Edge ist oder ein Cherry-Picking-Artefakt.

---

## 1. Hypothese

**H2:** Aktien, die in der **Train-Periode** den höchsten Dividend-Capture-Excess gezeigt haben, haben auch in der **Test-Periode** einen positiven Dividend-Capture-Excess gegenüber dem DAX-Buy-and-Hold. Konkret: die Top-10-Train-Aktien (gerankt nach mean(Excess) bei Hold N=5 in 2010-2018) liefern in 2019-2025 ein positives Mean-Excess > 0.

**Operationalisierung:**
- Hold-Window fixiert auf **N = 5** Trading-Tage (gleicher Wert wie im Sanity-Check, kein Tuning)
- Steuer-Faktor unverändert: 0.73625 (KapESt 25 % + Soli 5.5 %)
- Selection-Datensatz: Train 2010-01-01 bis 2018-12-31 (COVID-Filter irrelevant, weil pre-COVID)
- Test-Datensatz: Train 2019-01-01 bis 2025-12-31, COVID-Exclusion 2020-02-15 bis 2020-04-30
- Selection-Universum: alle Aktien mit ≥ 5 Dividenden-Events in Train (Robustheits-Schwelle)

**Erwartete Richtung:** Mean(Test-Excess der Top-10-Train) > 0. Einseitiger t-Test.

## 2. Mechanismus / Vermutung

Der Sanity-Check hat gezeigt, dass die deskriptiv-Top-10 ihre Outperformance konzentriert auf Div-Tage haben, nicht durch allgemeinen Drift. Mögliche Ursachen:

- **Steuer-/Investor-Heterogenität pro Aktie:** Einzelne Aktien haben spezifische Eigentümer-Strukturen (z. B. großer Streubesitz, viele steuerbefreite institutionelle Anleger), die den Ex-Day-Drop systematisch dämpfen.
- **Liquiditäts-Mikrostruktur:** Bei manchen Aktien gibt es nach dem Ex-Tag verstärkte Kauf-Aktivität (z. B. Index-Tracker, die nach Drop nachkaufen).
- **Yield-bezogene Selektion:** Hohe Yields ziehen besondere Investor-Klassen an, die den Drop reduzieren.

Diese Mechanismen wären aktien-spezifisch — daher der selektive Ansatz statt universell.

## 3. Voraussetzungen / Annahmen

- Daten-Sample identisch zu Pre-Reg #1 (84 von 90 Tickern, `dax_mdax_divs_2010_2025.pkl`)
- yfinance-Dividenden-Daten ab 2010 verfügbar
- Train hat genug Dividenden-Events pro Top-10-Aktie für robuste Rangbildung (Threshold K = 5 Events)
- Die *Identität* der Outperformer ist über Train→Test halbwegs stabil (das ist genau die These des Tests)

## 4. Daten & Setup

| Element | Wert |
|---|---|
| Hold-Window | N = 5 (FIX, kein Multi-Test) |
| Train-Periode | 2010-01-01 bis 2018-12-31 |
| Test-Periode | 2019-01-01 bis 2025-12-31, ohne COVID (2020-02-15 bis 2020-04-30) |
| Selektions-Kriterium | mean(Train-Excess bei N=5), absteigend, Top 10 |
| Mindest-Events Train | ≥ 5 pro Ticker (sonst Aktie ausgeschlossen) |
| Steuer-Faktor | 0.73625 |
| Benchmark | `^GDAXI` (DAX Performance-Index) |

**Erwarteter Sample-Size Test:** etwa 60-70 Events (10 Aktien × ~6-7 Test-Jahre × 1 Div/Jahr).

## 5. Statistische Tests

| Test | Methode |
|---|---|
| Mittelwert-Test | One-sample t-Test gegen 0, einseitig (Excess > 0), kein Bonferroni weil nur EINE Hypothese |
| Median-Test | Wilcoxon signed-rank (einseitig, alternative='greater') |
| Aktien-spezifischer Sub-Check | Pro Top-10-Aktie: wie viele Test-Events sind positiv? (deskriptiv, keine Gate-Bedingung) |
| Power-Analyse | n ≈ 60-70 Events, σ ≈ 5 %/Trade → MDE bei 80 % Power ≈ ±1.6 %. Erwarteter Effekt: +2 bis +3 % bei Edge, also Power für mittleren Effekt ausreichend |

## 6. Pre-Registered Gates

| Gate | Bedingung |
|---|---|
| **G1 (Mean-Positivität)** | Mean(Test-Excess der Top-10) > 0 |
| **G2 (Signifikanz)** | t-Test > +2.0, einseitig (kein Bonferroni, weil 1 Hypothese) |
| **G3 (Median-Konsistenz)** | Median(Test-Excess) > 0 UND Wilcoxon-p < 0.05 |
| **G4 (Practical Significance)** | Mean(Test-Excess) > +0.5 %/Trade (höher als in Pre-Reg #1 wegen Selektions-Charakter) |
| **G5 (Aktien-Robustheit)** | Mindestens 6 von 10 Aktien haben *individuelles* Test-Mean > 0 — sonst getrieben von 1-2 Ausreißern |

Alle 5 Gates müssen passen → Edge-Kandidat.

## 7. Falsifikations-Bedingungen

- G1 oder G2 failed → RED, Selektion auf Train war zufällig
- G3 failed → RED, Outlier-getrieben
- G4 failed → "Inconclusive" oder "Praktisch irrelevant" je nach Größe
- G5 failed → "Outlier-Konzentration", nicht broad pattern

## 8. Was die Pre-Reg NICHT erlaubt

- **Re-Selection in Test.** Die Top 10 werden EINMAL aus Train identifiziert und dann in Test gehandelt, kein Update.
- **Anderes Hold-Window.** N = 5 ist fixiert. Wenn N = 5 RED ergibt, gilt das. Tests mit N ∈ {1, 3, 10, 20} wären eine *neue* Pre-Reg.
- **Yield-Subset.** Aufteilen nach Dividend-Yield-Brackets wäre eine neue Pre-Reg.
- **Sektor-Subset.** Aufteilen nach Sektoren wäre eine neue Pre-Reg.

## 9. Erwartetes Ergebnis (Prior)

Mein Prior nach dem Sanity-Check: **20-25 % Edge-Wahrscheinlichkeit**. Argumente:

**Pro Edge:**
- Sanity-Check zeigt Drift-Hypothese als falsch (Random-5d ist neutral, nicht positiv)
- Outperformance ist Div-spezifisch konzentriert (+2.71 % vs ~0 %)
- Mechanismus wäre aktien-spezifisch plausibel

**Contra Edge:**
- Top 10 wurden post-hoc aus Test gewählt — wenn Train-Selektion eine andere Top 10 ergibt, ist das Rauschen
- 84 Aktien × Top-10-Auswahl ist implizites Multi-Test (Bonferroni-Strafe via Train-Test-Split abgefangen)
- Selektive Strategien sind in Liquid-Markets selten stabil

## 10. Code-Plan

- `selective_dividend_capture_test.py` — neues Skript
- Re-Use von `dax_mdax_divs_2010_2025.pkl` und `gdaxi_2010_2025.pkl`
- Output:
  - Top-10-Train-Tabelle (Tickers + Train-Mean + n_events)
  - Test-Stats für genau diese 10
  - Gate-Pass/Fail
  - Pro-Aktien-Test-Aufgliederung (für G5)

## 11. Pre-Reg-Bruch-Klausel

Wenn beim Coding herauskommt, dass die Pre-Reg unsauber definiert ist (z. B. Top-10 ist nicht eindeutig wegen Ties, oder Train hat zu wenig Events), wird das **dokumentiert** und die Pre-Reg ggf. neu committed — nicht stillschweigend angepasst.

---

## Ergebnisse

*Dieser Abschnitt wird NACH dem Test-Lauf ergänzt — als separater Commit unterhalb dieser Linie.*
