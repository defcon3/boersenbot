# Pre-Registered Hypothesis: Gold-Sleeve als dritter Combined-Baustein

**Status:** Pre-registered (vor Test-Lauf)
**Datum:** 2026-06-04
**Autor:** Veit Luther (mit Claude Opus 4.8 als Sparring; Idee aus Grok-Brainstorm 3b)
**Commit-Anchor:** dieses File wird VOR dem Test-Lauf committed und gepusht. Spätere Änderungen an Gates/Parametern gelten als Pre-Reg-Bruch und falsifizieren die Hypothese automatisch.

---

## 0. Warum diese Idee (und warum diszipliniert)

Aus dem Grok-Brainstorm die **am wenigsten overfitting-gefährdete** Idee: Gold als dritter Sleeve neben Hybrid-SPY und HYG. Es ist eine **Allokationsfrage, kein Signal-Mining** — wenige Forscher-Freiheitsgrade. Gold hat eine echte ökonomische Story als Diversifikator (Flight-to-Quality, Debasement/Inflation-Hedge) mit niedriger/teils negativer Korrelation zu *beidem*, Equity und Credit.

**Zentrale Ehrlichkeits-Falle (vorab benannt):** Das OOS-Fenster 2019–2025 ist **gold-freundlich** (Rally 2019/20, Surge 2024/25). Ein naiver Backtest schmeichelt Gold. Deshalb ist das härteste Gate (G5) ein **held-out gold-BÄRENMARKT-Fenster 2013–2018** — der Diversifikationsnutzen muss aus der niedrigen *Korrelation* kommen, nicht aus Golds jüngster *Beta*.

## 1. Hypothese

**H1 (Haupt):** Ein statischer Gold-Sleeve (GLD) im Anteil g, gegenfinanziert aus der bestehenden Combined-50/50 (Hybrid-SPY + HYG), verbessert die **risikoadjustierte** Performance — Portfolio P_g = (1−g)·Combined + g·GLD erzielt OOS einen höheren Sharpe als die Combined ohne Gold, ohne den Drawdown zu verschlechtern.

**Operationalisierung:**
```
P_g(t) = (1 − g) · r_combined(t) + g · r_gld_net(t)
```
- `r_combined`: Tagesreturn der Combined 50/50 (net), exakt wie in `combined_strategy_test.py`
- `r_gld_net`: GLD-Tagesreturn minus Rebalancing-Slippage (5 bps auf |Δ Gewicht|, monatliches Rebalancing wie die Combined)
- **Primär-Gewicht g = 20 %** (modest sleeve). Sensitivität: g ∈ {10 %, 30 %}.

**Erwartete Richtung:** Sharpe(P_20%) > Sharpe(Combined). Gerichtet.

## 2. Mechanismus / Vermutung

Gold ist über lange Strecken niedrig bis negativ mit Aktien UND Credit korreliert, besonders in Flight-to-Quality-Phasen. Ein kleiner statischer Sleeve sollte die Portfolio-Vol senken und/oder den Drawdown in genau den Stress-Phasen abfedern, in denen Hybrid (Equity) und HYG (Credit) gemeinsam leiden. Der Effekt ist **strukturell** (Korrelationsstruktur), nicht prognostisch — daher geringes Overfitting-Risiko.

**Gegenargument (ernst):** Golds eigener Return ist regimeabhängig und phasenweise stark negativ (2013–2018). Wenn der „Nutzen" im OOS-Fenster nur Golds Bull-Beta ist, verschwindet er — oder kehrt sich um — im Bärenfenster. Genau das prüft G5.

## 3. Voraussetzungen / Annahmen

- GLD (Inception 2004-11) liefert sauberen Gold-Proxy via yfinance; HYG ab 2007, Hybrid ab 2007 → Combined und P_g sind 2013–2018 UND 2019–2025 vollständig berechenbar.
- Statischer Sleeve, **kein** Gold-Timing-Signal (bewusst, um Freiheitsgrade niedrig zu halten). Ein Timing-Overlay wäre eine separate Pre-Reg.
- Monatliches Rebalancing auf Zielgewichte, 5 bps Slippage auf Gewichtsänderungen.
- COVID-Fenster 2020-02-15…2020-04-30 separat ausgewiesen (Haupt-Gate o. COVID, Stress-Check m. COVID).

## 4. Daten

| Element | Wert |
|---|---|
| Sleeves | Hybrid-SPY (MA50/200 + VIX-Norm), HYG-Stress (STLFSI4-Q75), GLD |
| Datenquelle | yfinance (SPY, ^VIX, HYG, GLD), FRED (STLFSI4) |
| Combined-Basis | 50/50 Hybrid+HYG, net, identisch zu `combined_strategy_test.py` |
| Haupt-OOS | 2019-01-01 … 2025-12-31 (o. COVID primär, m. COVID als Stress-Check) |
| Kontroll-Fenster (G5) | **2013-01-01 … 2018-12-31** (Gold-Bärenmarkt, held-out) |
| Primär-Gewicht | g = 20 %; Sensitivität g ∈ {10 %, 30 %} |
| Benchmark | Combined ohne Gold (g = 0) + SPY B&H als Referenz |

## 5. Statistische Tests

| Test | Methode |
|---|---|
| Sharpe-Differenz | P_20% vs Combined, mit Stationary-Bootstrap-KI (Block 21d, gepaart) — wiederverwendet aus `overnight_intraday_rolling_bootstrap.py` |
| Drawdown | MaxDD-Vergleich P_g vs Combined |
| Diversifikation | Korrelation(GLD, Combined) über OOS; zusätzlich COVID-Fenster-Korrelation |
| Robustheit | G1-Richtung über g ∈ {10 %, 30 %} |
| Kontroll-Fenster | identische Metriken auf 2013–2018 (G5) |

## 6. Pre-Registered Gates

Gold-Sleeve gilt nur als wertvoll, wenn **alle fünf** passen:

| Gate | Bedingung |
|---|---|
| **G1 (Risk-adjusted Lift, primär)** | g=20 %, OOS 2019–2025 o. COVID: Sharpe(P_20%) > Sharpe(Combined) UND Sortino(P_20%) ≥ Sortino(Combined) |
| **G2 (Kein DD-Schaden)** | MaxDD(P_20%) ≥ MaxDD(Combined) (gleich oder weniger negativ) im selben Fenster |
| **G3 (Gewichts-Robustheit)** | G1-Richtung (Sharpe-Lift) hält bei g=10 % UND g=30 % — kein einzelnes Glücks-Gewicht |
| **G4 (Echte Diversifikation)** | Korr(GLD, Combined) < 0,40 über OOS; UND P_20% verschlechtert den COVID-Stress-Drawdown nicht |
| **G5 (Gold-Bären-Kontrolle, das härteste)** | Im held-out 2013–2018: Sharpe(P_20%) ≥ Sharpe(Combined) − 0,15. D. h. Gold darf auch in seinem Bärenmarkt die Combined nicht material schädigen — sonst war der OOS-„Nutzen" nur Bull-Beta, keine Diversifikation. |

## 7. Falsifikations-Bedingungen

- G1 FAIL → Gold bringt keinen risikoadjustierten Mehrwert → RED.
- G1 pass aber G2 FAIL → Gold erkauft Return mit mehr Drawdown → kein Diversifikator → RED.
- G3 FAIL → Effekt hängt am Gewicht → fragil, RED.
- G4 FAIL → keine echte Diversifikation (Korr zu hoch / Stress-Schutz fehlt) → RED.
- **G5 FAIL → der entscheidende Befund:** Gold half nur, weil 2019–2025 ein Gold-Bullenmarkt war. Im Bärenfenster killt es die Combined → es war **Period-Mining**, kein struktureller Diversifikator → RED.
- Alle fünf pass → Gold-Sleeve-Kandidat, geht in Forward-Test (separate Pre-Reg) bevor das deployte Combined-Setup geändert wird.

**Was die Hypothese NICHT rettet:** das beste g nach dem Lauf wählen; das Kontroll-Fenster nachträglich verschieben; ein Gold-Timing-Signal einführen, um G5 zu bestehen (= neue Pre-Reg).

## 8. Erwartetes Ergebnis (Prior)

**Prior: ~35 % Edge-Wahrscheinlichkeit** — höher als die toten Equity-Tests, weil Gold-Diversifikation eine robuste, dokumentierte Struktureigenschaft ist UND es eine Allokations- statt Signal-Frage ist. Aber: der naive OOS-Backtest wird wegen des Gold-Bullen 2019–2025 zu optimistisch sein. **Die ehrliche Entscheidung fällt an G5.** Wahrscheinlichstes Muster: G1–G4 pass, G5 wackelt — dann ist die ehrliche Antwort „hilft im richtigen Regime, kein Free Lunch".

## 9. Code-Plan

- `gold_sleeve_test.py` — neues Skript im Repo-Root
- Combined-Basis aus `combined_strategy_test.py` spiegeln (wie `combined_leverage_test.py`)
- GLD laden, P_g für g ∈ {10, 20, 30 %} bauen, Metriken + Bootstrap-KI
- Drei Fenster: OOS o. COVID, OOS m. COVID, Kontroll 2013–2018
- Output: Gate-Tabelle + JSON-Export für evtl. Web-Einbettung
- Cache: `gold_sleeve_ohlc.pkl`

## 10. Pre-Reg-Bruch-Klausel

Wenn beim Coding eine Annahme bricht (z. B. GLD-Datenlücke), wird die Pre-Reg **dokumentiert geändert oder verworfen** — nicht stillschweigend angepasst. Geänderte Pre-Reg = neue Pre-Reg, neues Datum/Commit.

---

## Ergebnisse (Lauf 2026-06-04)

**Verdict: RED — G5 FAIL. Genau das vorab benannte Muster (G1–G4 pass, G5 wackelt).**
Der OOS-„Nutzen" war größtenteils Golds Bull-Beta 2019–2025, nicht reine Diversifikation. Ohne das held-out Gold-Bären-Fenster hätte das wie ein Free Lunch ausgesehen — der Pre-Reg-Mechanismus hat genau seinen Zweck erfüllt.

### Performance über die drei Fenster

**OOS 2019–2025 ohne COVID (Haupt):**

| Variante | AnnRet | Vol | Sharpe | Sortino | MaxDD |
|---|---|---|---|---|---|
| Combined (g=0) | +9,17 % | 7,23 % | 1,27 | 1,69 | −9,50 % |
| +Gold 10 % | +10,14 % | 6,84 % | 1,48 | 2,01 | −8,50 % |
| **+Gold 20 %** | +11,09 % | 6,81 % | **1,63** | 2,26 | −8,24 % |
| +Gold 30 % | +12,03 % | 7,16 % | 1,68 | 2,36 | −8,72 % |

**KONTROLLE 2013–2018 (Gold-Bärenmarkt, held-out):**

| Variante | AnnRet | Vol | Sharpe | Sortino | MaxDD |
|---|---|---|---|---|---|
| Combined (g=0) | +5,99 % | 6,55 % | **0,91** | 1,18 | −11,37 % |
| +Gold 10 % | +4,98 % | 6,02 % | 0,83 | 1,07 | −10,55 % |
| **+Gold 20 %** | +3,97 % | 5,92 % | **0,67** | 0,87 | −10,47 % |
| +Gold 30 % | +2,93 % | 6,28 % | 0,47 | 0,61 | −10,72 % |

### Gate-Check

| Gate | Ergebnis | Status |
|---|---|---|
| G1 Risk-adj Lift (g=20, OOS) | Sharpe 1,63 vs 1,27; Sortino 2,26 vs 1,69 | ✅ PASS |
| G2 Kein DD-Schaden | MaxDD −8,2 % vs −9,5 % | ✅ PASS |
| G3 Gewichts-Robustheit | Lift bei 10 % und 30 % | ✅ PASS |
| G4 Echte Diversifikation | Korr +0,11 < 0,40; COVID-DD ok | ✅ PASS |
| **G5 Gold-Bär-Kontrolle** | **Sharpe 0,67 vs 0,91 = −0,24 (Tol −0,15)** | ❌ **FAIL** |

Bootstrap-KI des primären Sharpe-Lifts (g=20 %, OOS o. COVID): **+0,36, 95%-KI [−0,03, +0,66], p=0,072** — der Lift ist nicht einmal im gold-freundlichen Fenster sauber signifikant (KI streift die 0).

### Lehre

1. **G5 ist der ganze Punkt.** OHNE das Bärenfenster: G1–G4 alle grün, Sharpe 1,27→1,63, kleinerer Drawdown — ein scheinbarer Slam-Dunk. MIT Bärenfenster: der Lift kollabiert (Sharpe −0,24), weil er substanziell Golds Bull-Run 2019–2025 war. Period-Mining sauber entlarvt.
2. **Diversifikation ist TEILWEISE real, aber nicht gratis.** Korrelation echt niedrig (+0,11), und selbst im Bärenmarkt senkt Gold Vol (6,55→5,92 %) und Drawdown (−11,4→−10,5 %). Gold *reduziert Risiko* — aber wenn Gold selbst fällt, bezahlt man mit Rendite, und der Sharpe leidet. Kein Free Lunch.
3. **Versuchung korrekt widerstanden:** g=10 % läge mit −0,08 *innerhalb* der G5-Toleranz. Das nachträglich zum „Pass" zu erklären, wäre exakt der in §6/§7 verbotene „bestes-g-nach-dem-Lauf"-Bruch. Ein kleinerer, regime-robusterer Sleeve wäre eine **neue Pre-Reg** (mit g vorab fixiert), kein Patch dieser hier.

### Einordnung in die done-Liste

| Spalte | Wert |
|---|---|
| Hypothese | Gold-Sleeve (g=20 %) auf Combined 50/50 |
| Observed Effect | OOS-Sharpe +0,36 (p=0,072), aber Bärenmarkt −0,24 |
| Power Level | Mittel (4.817 Combined-Tage, 3 vorregistrierte Fenster) |
| Verdict Type | **Sauber falsifiziert via held-out Kontrolle** (Bull-Beta statt Diversifikation) |
| Kommentar | Diversifikation teilweise real (Korr +0,11, Vol/DD runter), aber regimeabhängig; OOS-Lift war Gold-Bull-Beta |

### Code-Referenz

- `gold_sleeve_test.py`, `gold_sleeve_results.json`
- Pre-Reg-Commit-Anchor: `f3692f51`
