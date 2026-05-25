# Pre-Registered Hypothesis: Dividend Capture auf deutschen Aktien

**Status:** Pre-registered (vor Test-Lauf)
**Datum:** 2026-05-25
**Autor:** Veit Luther (mit Claude Opus 4.7 als Sparring)
**Commit-Anchor:** dieses File wird VOR dem Test-Lauf committed und gepusht. Spätere Änderungen an Gates/Parametern gelten als Pre-Reg-Bruch und falsifizieren die Hypothese automatisch.

---

## 1. Hypothese

**H1 (Haupthypothese):** Eine Strategie, die deutsche Aktien (DAX-40 + MDAX-50) am Handelstag T−1 vor dem Ex-Dividenden-Tag kauft und an Tag T+N wieder verkauft, erzielt einen positiven netto-Excess-Return gegenüber einem DAX-Index-Buy-and-Hold über das gleiche Fenster.

**Operationalisierung des Effekts:**

```
Excess_i,N = (Verkaufskurs_T+N − Kaufkurs_T−1 + 0.73625 · Brutto_Div_i) / Kaufkurs_T−1
           − (DAX_T+N − DAX_T−1) / DAX_T−1
```

- `Brutto_Div_i`: ausgeschüttete Dividende des Events i (vor Steuer)
- `0.73625`: Netto-Faktor nach Kapitalertragsteuer 25 % + Solidaritätszuschlag 5.5 % (= 26.375 % Abzug); Kirchensteuer wird nicht berücksichtigt (Privatfall ohne Kirchenmitgliedschaft als Baseline)
- Sparerpauschbetrag (€1000/Jahr) wird **nicht** berücksichtigt — würde die Strategie künstlich besser machen, weil er nur einmalig hilft

**Erwartete Richtung:** Excess > 0. Gerichtete Hypothese, einseitiger t-Test.

## 2. Mechanismus / Vermutung

Empirisch ist der Ex-Day-Drop in vielen Märkten kleiner als die Brutto-Dividende (Drop-Ratio < 1, akademisch dokumentiert). Wenn diese Lücke groß genug ist, könnte sie auch nach Steuer und Transaktionskosten noch positiv sein. Mögliche Ursachen für eine sub-1.0-Drop-Ratio:

- Steuerheterogene Investor-Klassen (steuerbefreite Pensions-/Stiftungs-Investoren treiben den Preis post-Ex)
- Mikrostruktur-Effekte am Stichtag (große institutionelle Käufe nach Ex)
- Verhaltens-Effekte (Retail-Investoren überschätzen die Recovery)

## 3. Voraussetzungen / Annahmen

- yfinance liefert verlässliche Brutto-Dividenden-Daten für DAX-40 + MDAX-50 ab 2010
- Adjusted-Close für DAX-Index als sauberer Benchmark verfügbar
- Spread + Slippage werden **nicht** zusätzlich modelliert (eine Erweiterung wäre möglich, aber der Test prüft erstmal Brutto-Strategie nach Steuer)
- Steuerbehandlung wie oben (26.375 % auf Brutto-Dividende, keine Sparerpauschbetrag-Anrechnung)
- Wenn ein Ticker zwischen Train und Test delisted wurde, wird er aus beiden ausgeschlossen (Survivor-Bias-Reduktion durch Symmetrie)

## 4. Daten

| Element | Wert |
|---|---|
| Universum | DAX-40 + MDAX-50 (Stand 2026, fixe Liste, kein Index-Rebalancing) |
| Datenquelle | yfinance (`yf.Ticker(...).history()` + `.dividends`) |
| Zeitraum gesamt | 2010-01-01 bis 2025-12-31 |
| Train-Periode | 2010-01-01 bis 2018-12-31 |
| Test-Periode | 2019-01-01 bis 2025-12-31 |
| COVID-Exclusion | Ex-Dates zwischen 2020-02-15 und 2020-04-30 ausschließen (in Train und Test) |
| Erwartete n (Dividenden-Events) | ~90 Aktien × ~10 Jahre × 1 Div/Jahr ≈ 900 Events Train, ~700 Test |
| Benchmark | `^GDAXI` (DAX Performance Index) |

**Hinweis zu Survivor-Bias:** Der DAX-40 und MDAX-50 nach Stand 2026 enthält nicht die delisteten oder ausgekippten Firmen der Vergangenheit. Das ist ein bekannter optimistischer Bias zugunsten der Strategie. Falls die Strategie *trotzdem* failed, ist der Befund robust nach unten. Falls sie passed, müsste in einer Folge-Iteration eine Point-in-Time-Komposition geprüft werden.

## 5. Statistische Tests

| Test | Methode |
|---|---|
| Mittelwert-Test | One-sample t-Test (Welch) gegen 0, einseitig (Excess > 0) |
| Median-Test | Wilcoxon signed-rank (zweiseitig), als Robustheits-Check |
| Power-Analyse | Vor dem Lauf: bei erwartetem σ ≈ 5 %/Trade und n ≈ 700 Test-Events liefert das Setup eine MDE von ca. ±0.4 %/Trade bei 80 % Power. Erwarteter Effekt bei Edge-Vermutung: 0.3 – 0.8 %/Trade. → Power-Lage ist *ausreichend* für mittlere Effekte. |
| Multiple Testing | Bonferroni über K = 5 Hold-Windows (N ∈ {1, 3, 5, 10, 20}). Naive-Schwelle t = 1.96, Bonferroni t ≈ 2.58 (K=5 zweiseitig-äquivalent für einseitiges α=0.01) |

## 6. Pre-Registered Gates

Die Strategie gilt nur dann als handelbarer Edge, wenn **alle vier** Gates passieren:

| Gate | Bedingung |
|---|---|
| **G1 (Train-Direction)** | mean(Train-Excess) > 0 und Train-t > +2.0 (einseitig) in mindestens einem Hold-Window N |
| **G2 (Test-Bonferroni)** | Im *gleichen* N (das G1 bestanden hat): Test-t > +2.58 (einseitig, Bonferroni-korrigiert für K=5) |
| **G3 (Median-Konsistenz)** | Test-Median(Excess) > 0 im *gleichen* N. Wilcoxon-p < 0.05 |
| **G4 (Practical Significance)** | Mean-Excess(Test) > 0.3 %/Trade (über typischen Spread+Slippage hinaus). Wenn der Effekt zwar signifikant, aber < 0.3 %/Trade ist, ist er ökonomisch irrelevant — vgl. Fall C der Power-Analyse auf done-Seite. |

## 7. Falsifikations-Bedingungen (was die Hypothese killt)

- Wenn G1 in *keinem* Hold-Window passed → RED, strukturell falsifiziert
- Wenn G1 passed aber G2 in dem gleichen N failed → RED (Train-Test-Bruch / Overfit)
- Wenn G1+G2 passed aber G3 failed → RED (Outlier-getrieben, nicht robust)
- Wenn G1+G2+G3 passed aber G4 failed → "Praktisch irrelevant" (analog zu CC-MR auf done-Seite)
- Wenn alle vier passen → Edge-Kandidat, geht in Forward-Test (separate Pre-Reg)

**Was die Hypothese NICHT rettet:**
- Subsetting nach Sektor/Größe nach dem Lauf — wäre Post-hoc-Slicing
- Tunen der Hold-Windows N nach dem Lauf — wäre Pre-Reg-Bruch
- Steuer-Optimierung (z. B. Sparerpauschbetrag annehmen) ohne neue Pre-Reg

## 8. Erwartetes Ergebnis (Prior)

Mein Prior vor dem Test: ~10 % Edge-Wahrscheinlichkeit. Begründung:
- Steuerlich strukturell ungünstig (26.375 % Abzug auf Brutto-Div vs. 100 %-Drop)
- Akademische Literatur findet zwar Drop-Ratios < 1, aber selten so deutlich, dass nach Steuer noch Edge übrigbleibt
- Wäre das Edge offensichtlich, hätten institutionelle Akteure es längst kassiert

Trotzdem testenswert wegen deutscher Besonderheiten (Jahresdividende statt Quartalsdividende, Investor-Heterogenität).

## 9. Code-Plan

- `dividend_capture_test.py` — neues Skript im Repo-Root
- Datenfetch (yfinance), 1× lokal cachen als `dax_mdax_divs_2010_2025.pkl`
- Backtest: pro Dividenden-Event Excess-Return je Hold-Window berechnen
- Output: Tabelle Train/Test je N mit n, mean, t, p, median; sowie Pass/Fail je Gate
- Log nach `dividend_capture.log`

## 10. Pre-Reg-Bruch-Klausel

Falls während des Codings oder beim Testen herauskommt, dass irgendeine Annahme oben nicht funktioniert (z. B. yfinance liefert lückenhafte Dividenden-Daten), wird die Pre-Reg **dokumentiert geändert oder verworfen** — nicht stillschweigend angepasst. Eine geänderte Pre-Reg ist eine neue Pre-Reg, mit neuem Datum und neuem Commit.

---

## Ergebnisse (Lauf 2026-05-25)

**Verdict: RED — strukturell falsifiziert. Sogar *negativer* Edge.**

### Daten-Yield

- 84 von 90 Tickern erfolgreich geladen (6 mit 0 Tagen oder 0 Dividenden-Events, ausgeschlossen — keine Survivor-Bias-Korrektur, weil symmetrisch in Train+Test)
- Benchmark `^GDAXI`: 4059 Datenpunkte
- 5475 Event-Zeilen über alle 5 Hold-Windows; davon Train 3010, Test 2465
- Pro N: ca. 602 Train-Events, 493 Test-Events — Power für mittlere Effekte ausreichend (vgl. Pre-Reg §5)

### Statistik pro Hold-Window N

| N | n_train | mean_train (%) | t_train | p₁_train | n_test | mean_test (%) | t_test | p₁_test | median_test (%) | wilcox_p |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 602 | −0.299 | −4.64 | 1.000 | 493 | −0.306 | −3.44 | 1.000 | −0.269 | 1.000 |
| 3 | 602 | −0.328 | −3.50 | 1.000 | 493 | −0.471 | −3.83 | 1.000 | −0.420 | 1.000 |
| 5 | 602 | −0.308 | −2.48 | 0.993 | 493 | −0.423 | −2.57 | 0.995 | −0.348 | 0.999 |
| 10 | 602 | −0.299 | −1.85 | 0.968 | 493 | −0.824 | −3.64 | 1.000 | −0.773 | 1.000 |
| 20 | 602 | +0.040 | +0.17 | 0.431 | 493 | −1.016 | −2.99 | 0.999 | −1.354 | 1.000 |

(p₁ = einseitiger p-Wert *für die Hypothese-Richtung* Excess > 0. p₁ ≈ 1.0 heißt: Excess ist hochsignifikant in der *entgegengesetzten* Richtung.)

### Gate-Check

| N | G1 (Train>0, t>2.0) | G2 (Test t>2.58) | G3 (Median>0, wilcox<.05) | G4 (Test>0.3%) | Alle pass |
|---|---|---|---|---|---|
| 1 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 3 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 5 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 10 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 20 | ❌ | ❌ | ❌ | ❌ | ❌ |

**Kein einziges Hold-Window besteht auch nur G1.** Der schwächste Train-Verlust ist N=20 mit mean = +0.040 % (nicht signifikant, t = +0.17) — im Test kollabiert genau dieses N auf mean = −1.02 % (t = −2.99, signifikant negativ).

### Lehre

1. **Drop-Ratio ≥ 1 in Deutschland (im untersuchten Sample).** Der Ex-Day-Drop ist im Mittel mindestens so groß wie die Brutto-Dividende — der akademisch dokumentierte "Drop-Ratio < 1"-Effekt taucht hier nicht handelbar auf.

2. **Steuer-Mechanik schlägt voll durch.** Die Strategie verliert ca. 0.3 %/Trade bei kurzem Hold (N=1) — das deckt sich mit der Größenordnung des Brutto-Steuerabzugs auf typische Dividenden-Yields (typisch 2–4 % p.a.; bei ~26 % Abzug = 0.5–1 % je Event Brutto-Steuer-Verlust, abzüglich Drop-Ratio-Vorteil).

3. **Längeres Halten verstärkt den Verlust.** N=20 zeigt im Test mean = −1.02 % — vermutlich Mean Reversion gegen den DAX-Index in der Test-Periode (deutsche Single-Names underperformen den Index strukturell, ungeachtet der Dividende).

4. **Pre-Reg-Disziplin bewahrt.** Die Hypothese wäre verlockend zum "Tuning" gewesen (Subsetting nach Sektor, nach Größe, nach Yield-Bracket). Stattdessen: ein Lauf, klares RED, keine Reparatur. Wenn der User die Hypothese in einer differenzierten Form weiterverfolgen will (z. B. "nur Aktien mit Yield > 5 %"), wäre das eine **neue Pre-Reg**, kein Patch dieser hier.

### Einordnung in die done-Liste

Diese Hypothese wird Eintrag **#11** auf der `done`-Seite (sobald in die Hauptseite aufgenommen):

| Spalte | Wert |
|---|---|
| Hypothese | Dividend Capture DE (DAX-40 + MDAX-50) |
| n (Test) | 493 pro Hold-Window |
| Observed Effect | −0.31 % bis −1.02 % (negativ!) |
| Power Level | **High** (n groß, Effekt klar messbar in *Gegenrichtung*) |
| Verdict Type | **Sauber falsifiziert** (sogar Direction-Flip wäre falsche Sprache — der Effekt ist eindeutig negativ in Train *und* Test, gleiche Richtung, hochsignifikant) |
| Kommentar | Drop-Ratio ≥ 1, Steuer schlägt voll durch, längeres Hold = mehr Verlust |

### Code-Referenz

- `dividend_capture_test.py`
- `dividend_capture.log`, `dividend_capture_events.csv`, `dividend_capture_stats.csv`, `dividend_capture_gates.csv`
- Daten-Cache: `dax_mdax_divs_2010_2025.pkl`, `gdaxi_2010_2025.pkl`
- Pre-Reg-Commit-Anchor: `871bac1b`

