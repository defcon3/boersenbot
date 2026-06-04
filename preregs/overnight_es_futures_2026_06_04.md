# Pre-Registered Hypothesis: Overnight-Edge via ES-Index-Futures (statt SPY-ETF)

**Status:** ⚠️ PRE-REGISTERED, **NOCH NICHT AUSGEFÜHRT** — bewusst mit Vorsicht zu behandeln (siehe §0).
**Datum:** 2026-06-04
**Autor:** Veit Luther (mit Claude Opus 4.8 als Sparring)
**Vorgänger:** `overnight_intraday_test.py` (G2 FAIL, YELLOW), `overnight_intraday_g3_pit.py` (G3-PIT FAIL), `overnight_intraday_rolling_bootstrap.py` (G1 deskriptiv robust, aber nicht handelbar)
**Commit-Anchor:** dieses File wird VOR dem Test-Lauf committed und gepusht. Spätere Änderungen an Gates/Parametern gelten als Pre-Reg-Bruch und falsifizieren die Hypothese automatisch.

---

## 0. ⚠️ VORSICHTS-KENNZEICHNUNG (zuerst lesen)

Diese Pre-Reg ist **gefährlicher** als die bisherigen, aus drei Gründen — sie ist deshalb ausdrücklich als „mit Vorsicht zu genießen" markiert:

1. **Gefahr des Gate-Hackings.** Der einzige Grund, warum diese Hypothese überhaupt existiert, ist dass G2 (Handelbarkeit) auf dem SPY-ETF **gescheitert** ist. Ein anderes Instrument zu wählen, um ein gescheitertes Gate doch noch zu bestehen, ist **per Default verdächtig**. Diese Pre-Reg ist nur legitim, weil ES-Futures eine **ökonomisch fundamental andere Kostenstruktur** haben (nicht weil wir „so lange suchen, bis etwas passt"). Würde der Test passen, ist das **kein** bestätigter Edge, sondern bestenfalls ein Forward-Test-Kandidat. Der Prior bleibt niedrig (§8).

2. **Datenrisiko ist real und könnte die Pre-Reg vorab killen.** Gratis-yfinance liefert `ES=F` nur als fortlaufenden Tages-Kontrakt. Ob dessen Tages-`Open`/`Close` die **Cash-Equity-Overnight-Session** (US-Close 16:00 ET → nächster Cash-Open 09:30 ET) sauber abgrenzt, ist **unklar**. ES handelt nahezu 23 h/Tag; „overnight" im Futures-Sinn ≠ „overnight" im Cash-Sinn. Ohne saubere Intraday-Futures-Bars (Globex-Session-Grenzen) ist der Test **nicht valide**. Siehe §3/§9 — wenn die Daten das nicht hergeben, wird die Pre-Reg **dokumentiert verworfen**, nicht gebogen.

3. **Echtes Geld, echtes Risiko, falls es je live geht.** Futures sind gehebelt (ES ≈ 6000 Punkte × 50 \$ ≈ 300 000 \$ Nominal pro Kontrakt bei ~15 000 \$ Margin). Overnight-Gap-Risiko, Roll-Kosten (vierteljährlich), Financing und Margin-Calls sind reale Friktionen, die ein Aktien-Backtest **nicht** abbildet. Ein „Pass" hier rechtfertigt **kein** Live-Trading ohne separaten Forward-Test mit Paper-Futures.

**Kurzform:** Diese Pre-Reg testet eine *plausible* Instrument-These, aber sie ist die mit der größten Selbsttäuschungs-Gefahr im ganzen Projekt. Lieber RED akzeptieren als das Gate weichklopfen.

---

## 1. Hypothese

**H1 (Haupthypothese):** Der deskriptiv robuste Overnight-Übergewinn von SPY (Close→Open, vgl. Rolling/Bootstrap-Lauf: Sharpe(on)−Sharpe(id) = +0.65, KI [+0.10, +1.19], p=0.015) bleibt **nach realistischen Futures-Friktionen** als positiver, risk-adjusted Edge erhalten, wenn er über **ES-Index-Futures** statt SPY-ETF gehandelt wird — weil die Round-Trip-Kosten in Basispunkten dramatisch niedriger sind.

**Operationalisierung:** Eine „Nur-Overnight"-Futures-Strategie hält ES über die Overnight-Session (Kauf nahe US-Cash-Close, Verkauf nahe nächstem Cash-Open), flat während der Cash-Session.

**Erwartete Richtung:** Netto-Sharpe(Overnight-Futures) > Netto-Sharpe(Buy&Hold-Futures). Gerichtet.

## 2. Mechanismus / Vermutung

- **Kostenargument (der ganze Punkt):** SPY-ETF-Overnight handelt 2 Legs/Tag; bei 5 bps/Leg = 10 bps/Tag ≈ 25 % p.a. Reibung — das tötet G2. ES-Futures: Kommission ~1–2 \$/Round-Turn + 1 Tick Spread (0.25 Pt = 12.50 \$) auf ~300 000 \$ Nominal ≈ **0.4–0.5 bps Round-Trip**. Größenordnung **~20× billiger** pro Umschlag.
- **Wenn** der Overnight-Effekt im Futures-Markt dieselbe Richtung hat **und** die Kosten so viel kleiner sind, **könnte** der nach SPY-Kosten erstickte Edge netto überleben.
- **Gegenargument (ernst nehmen):** Genau deshalb, weil Futures so billig sind, sind dort die effizientesten Akteure aktiv. Wenn ein handelbarer Overnight-Edge existierte, wäre er hier am ehesten wegarbitriert. Das drückt den Prior.

## 3. Voraussetzungen / Annahmen (Daten-Validität ZUERST prüfen)

- **Daten-Gate (Vorab-K.o.):** Es muss eine Quelle gefunden werden, die ES-Futures **session-sauber** in Overnight (Globex, nach Cash-Close) vs. Cash-Session (RTH 09:30–16:00 ET) trennt. Kandidaten in Reihenfolge:
  1. yfinance `ES=F` Tages-OHLC — **nur akzeptabel, wenn** verifiziert ist, dass `Open`≈RTH-Open und `Close`≈RTH-Close (dann ist on = RTH-Open/Vortags-RTH-Close − die Übernacht-Lücke inkl. Globex). Muss empirisch gegen bekannte RTH-Settlements geprüft werden.
  2. Falls (1) nicht sauber: Intraday-Bars (z. B. via Anbieter mit RTH/ETH-Flag). Falls nicht gratis verfügbar → **Pre-Reg verworfen** (§10).
- Roll-Logik: fortlaufender Front-Month, Roll ~8 Tage vor Verfall, Roll-Kosten = 1 Tick je Roll explizit abgezogen.
- Margin/Financing wird im Backtest als Renditereihe (unverzinst) modelliert; der reale Margin-Hebel wird NICHT zur Renditesteigerung genutzt (Vergleich auf 1× Nominal, damit Sharpe ehrlich bleibt).

## 4. Daten

| Element | Wert |
|---|---|
| Instrument | ES (E-mini S&P 500), fortlaufender Front-Month |
| Benchmark | ES Buy&Hold (24h gehalten), gleiches Nominal |
| Datenquelle | yfinance `ES=F` (Stufe 1) — Validität gegen RTH ZUERST verifizieren |
| Zeitraum gesamt | 2010-01-01 bis 2026-06-30 (ES=F-Historie bei yfinance i. d. R. ab ~2000, aber Qualität früher prüfen) |
| Train | 2010-01-01 bis 2018-12-31 |
| Test (OOS) | 2019-01-01 bis 2026-06-30 |
| COVID-Handling | Crash-Fenster 2020-02-15…2020-04-30 separat ausweisen, nicht aus G-Bewertung kippen lassen |
| Kosten | Round-Trip 0.5 bps (Basis), zusätzlich Sensitivität bei 1.0 und 2.0 bps |
| Roll-Kosten | 1 Tick je Quartals-Roll, explizit |

## 5. Statistische Tests

| Test | Methode |
|---|---|
| Sharpe-Differenz | Overnight-Net vs B&H, mit **Stationary-Bootstrap-KI** (gleiche Maschinerie wie `overnight_intraday_rolling_bootstrap.py`, erw. Block 21 d) |
| Subperioden | Train/Test getrennt; zusätzlich 3J/5J-Rolling-Anteil on>id auf Futures |
| Strukturbruch | sup-Wald/QLR auf d_t (Futures), Block-Bootstrap-Kritikwerte |
| Kosten-Sensitivität | Gate-Ergebnis bei 0.5 / 1.0 / 2.0 bps Round-Trip ausweisen |

## 6. Pre-Registered Gates

Edge-Kandidat nur, wenn **alle** passen:

| Gate | Bedingung |
|---|---|
| **G0 (Daten-Validität)** | `ES=F` Open/Close bilden RTH-Grenzen plausibel ab (Abweichung gegen bekannte Settlements klein). FAIL → Pre-Reg verworfen, kein Weiterrechnen. |
| **G1 (Deskriptiv, Futures)** | Overnight kumuliert > Intraday/Cash-Session kumuliert in Train UND Test |
| **G2 (Handelbar, der Kernpunkt)** | Overnight-Net-Sharpe @ 0.5 bps Round-Trip > B&H-Sharpe in der **Test**-Periode, und Bootstrap-KI der Sharpe-Differenz schließt 0 aus |
| **G3 (Kosten-Robustheit)** | G2 hält auch noch bei **1.0 bps** Round-Trip (nicht nur am Best-Case-Kostenpunkt) |
| **G4 (Roll-ehrlich)** | Ergebnis bleibt positiv NACH expliziten Roll-Kosten |

## 7. Falsifikations-Bedingungen

- G0 FAIL → Pre-Reg **verworfen** (Daten geben es nicht her), als ehrlicher Friedhof-Eintrag „nicht testbar mit Gratis-Daten".
- G1 FAIL → Overnight-Effekt überträgt sich nicht auf den Futures-Markt → RED.
- G2 FAIL → selbst bei ~20× niedrigeren Kosten nicht handelbar → **das ist das wahrscheinliche Ergebnis**, und es wäre ein starker, sauberer Befund (Effekt ist real, aber selbst im billigsten Instrument tot).
- G3 FAIL (G2 nur bei 0.5 bps) → zu knapp, praktisch fragil → RED.
- Alle pass → **kein** Live-Go, sondern separate Forward-Test-Pre-Reg mit Paper-Futures.

**Was die Hypothese NICHT rettet:** Kostenpunkt nach dem Lauf senken; Session-Definition nachträglich verschieben; nur die „guten" Jahre behalten; Hebel zur Sharpe-Schönung einsetzen.

## 8. Erwartetes Ergebnis (Prior)

**Prior: ~15 % Edge-Wahrscheinlichkeit** (etwas höher als bei den toten Aktien-Tests, weil das Kostenargument ökonomisch echt ist — aber niedrig, weil Futures-Märkte hocheffizient sind und genau solche Mikrostruktur-Effekte dort zuerst verschwinden). Wahrscheinlichstes Resultat: **G2 FAIL** → der Overnight-Effekt ist ein Stylized Fact, das selbst im billigsten verfügbaren Instrument keinen risk-adjusted Edge nach Kosten liefert. Das wäre die finale, saubere Beerdigung der Overnight-Handelbarkeit.

## 9. Code-Plan

- `overnight_es_futures_test.py` — neues Skript im Repo-Root
- Schritt 0: `ES=F` laden, Validitäts-Check Open/Close vs RTH (G0) — **erst danach** weiterrechnen
- Bootstrap/Strukturbruch-Funktionen aus `overnight_intraday_rolling_bootstrap.py` wiederverwenden (importieren, nicht kopieren)
- Output: Gate-Tabelle Train/Test, Kosten-Sensitivität 0.5/1.0/2.0 bps, JSON-Export für Web
- Cache: `es_futures_ohlc.pkl`

## 10. Pre-Reg-Bruch-Klausel

Wenn beim Coding herauskommt, dass G0 nicht erfüllbar ist (Gratis-Daten trennen die Sessions nicht sauber), wird diese Pre-Reg **dokumentiert verworfen** — nicht auf eine schlechtere Daten-Approximation heruntergeschraubt, nur um „etwas" zu testen. Eine geänderte Daten- oder Session-Definition ist eine **neue** Pre-Reg mit neuem Datum/Commit.

---

## Ergebnisse

*(noch nicht ausgeführt — wird nach dem Lauf hier ergänzt)*
