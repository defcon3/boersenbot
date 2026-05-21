# EDGE SEARCH MASTER PLAN — 2026-05-21

## SITUATION

S&P-500 ist hocheffizient. Bisherige Tests (TA, Makro, Mean-Reversion) alle OOS-falsifiziert:

| Strategie | IS Edge | OOS t | Resultat |
|-----------|---------|-------|----------|
| MA-Crossover | +0.25% | -1.45 | FAIL |
| Streak-Mean-Rev | +0.25% | -1.97 | FAIL |
| Mean-Reversion 2% | -0.15% | -0.32 | FAIL |

**Schluss:** Einfache Signale funktionieren nicht. Edges sind klein oder nicht-existenzen.

---

## 3 OPTIONEN (Nacheinander)

### OPTION 1: PARAMETER-BRUTE-FORCE (Mean-Reversion Varianten)

**Idee:** Mean-Reversion-Kern ist solide Theorie (Überreaktionen revert). 
Aber 2%-Threshold + 1-Tag-Hold war zu simpel.
Teste alle Kombinationen systematisch.

**Parameter-Raum:**
- Threshold: 1%, 1.5%, 2%, 2.5%, 3%
- Hold-Period: 1, 2, 3 Tage
- Stop-Loss: -1%, -1.5%, -2%, None
- Long-Only vs Long+Short

**Pre-Reg (2026-05-21):**
- G1: IS 2014-2021, Best-Param-Set mit t > +2.0, Excess > 0%
- G2: OOS 2022-2026, gleiche Params, t > +1.5, Excess > 0%
- G3: Netto @ 5bps+0.1%, t > +1.0
- G4: Min 20 Signals/Monat OOS
- G5: Not COVID
- **Bonferroni-Korrektur:** 5×5×4×2 = 200 Param-Sets → |t| > 4.2 für Signifikanz (!)

**Erwartung:** 
- Mit 200 Tests: ~10 werden zufällig "signifikant" sein
- Wenn keine über Bonferroni: Dann ist Mean-Reversion wirklich tot
- Wenn eine über Bonferroni: Legitimate edge gefunden

**Zeit:** ~1-2 Stunden Coding + Lauf

---

### OPTION 2: MACHINE LEARNING (Random Forest / XGBoost auf OHLCV)

**Idee:** Einfache TA funktioniert nicht. Aber Marktmikrostruktur-Features könnten Signale enthalten.

**Features (pro Ticker/Tag):**
- Momentum: 5-day, 20-day returns
- Volatility: 20-day std, ATR
- Volume: 20-day avg, heute vs avg
- OHLC ratios: Close/Open, High-Low
- Price Action: Gaps, Reversals

**Model:**
- Target: "Next-Day SPY Excess > 0.5%" (binary classification)
- Train: IS 2014-2021 (mit randomized K-Fold)
- Test: OOS 2022-2026
- Metric: Sharpe Ratio, Max-Drawdown, Win-Rate

**Pre-Reg (2026-05-21):**
- G1: IS In-Sample Accuracy > 60% (besser als 50% Zufall)
- G2: OOS Out-of-Sample Excess > 0%, t > +1.5
- G3: Netto @ 5bps, t > +1.0
- G4: Min Trades/Monat >= 20
- G5: Not COVID
- **No Bonferroni** (Model ist Black-Box, aber pre-reg ist strict)

**Erwartung:**
- ML kann Overfitting haben → OOS wahrscheinlich schwächer als IS
- Wenn OOS t > 1.5: Indiz dass Features tatsächlich Signal enthalten
- Wenn OOS t < 0.5: Features sind rein Noise + Overfitting

**Zeit:** ~3-4 Stunden (Feature-Eng + Train + Backtest)

---

### OPTION 3: PRAGMATISCHER HYBRID (Risk-Management + Trend + Vol-Regime)

**Idee:** Nicht auf Edge suchen, sondern auf Risiko optimieren.
Regeln:
1. **Trend-Filter:** Nur Long wenn SPY 50-MA über 200-MA
2. **Vol-Filter:** Position-Size inverse zu VIX (High VIX → small)
3. **Rebalance:** Monthly equal-weight across 10 Sectors
4. **Stop:** Hard stop at -2% portfolio monthly
5. **Automatic Reporting:** Daily NAV, Monthly Metrics

**Pre-Reg (2026-05-21):**
- G1: IS 2014-2021, Sharpe > 1.0 (besser als Buy-Hold ~0.8)
- G2: OOS 2022-2026, Sharpe > 0.8
- G3: Max-DD < 20% (vs SPY ~30%)
- G4: Sortino > 1.5
- G5: Not COVID-biased

**Erwartung:**
- Hybrid hat niedrigere Returns (~8% vs SPY ~11%)
- Aber bessere Risk-adjusted (Sharpe, DD)
- **Ziel:** Konsistent automatisierbar, nicht perfekt, sondern solide

**Zeit:** ~2-3 Stunden Code + Daily Monitoring

---

## REIHENFOLGE & KRITERIEN

### Exekution:

1. **OPTION 1 (Brute-Force)** — schnellster ROI, klare Ja/Nein Antwort
   - Wenn OK (|t| > 4.2 OOS): STOP hier, wir haben Edge!
   - Wenn FAIL: Mean-Reversion ist tot, gehen zu Option 2

2. **OPTION 2 (ML)** — wenn Brute-Force failed
   - Wenn OK (OOS t > 1.5): Tune Model, deployen als Live-Signal
   - Wenn FAIL: Pure Feature-Noise, akzeptieren Option 3

3. **OPTION 3 (Hybrid)** — wenn Option 1 & 2 beide failed
   - **Default-Position:** Immer laufen (Risk-Management ist sicher)
   - Nicht auf Edge-Suche, sondern auf Konsistenz
   - Deployable auf NAS-Cron

---

## SUCCESS CRITERIA

| Option | Win-Condition | OOS-Metric | Deploy? |
|--------|---------------|-----------|---------|
| 1 (Brute) | \|t\| > 4.2 OOS | Excess > 0.5% | Ja, sofort |
| 2 (ML) | OOS t > 1.5, Sharpe > 0.5 | Excess > 0.2% | Ja, mit Caution |
| 3 (Hybrid) | Sharpe > 0.8, DD < 20% | Return-Consistency | Ja, Default |

---

## DEPLOYMENT (wenn erfolgreich)

### Stack:
- **Local:** Backtest-Skripte (Python)
- **NAS Cron:** Daily Signal-Generator (14:30 UTC = 16:30 CEST), schreib zu DB
- **VPS Dashboard:** Live Position-Tracking, P&L, Alerts
- **Execution:** Manual (für Lernen) oder Broker-API later (wenn profitable)

### Monitoring:
- Daily: NAV, Trade-Count, Slippage-Actual
- Weekly: Sharpe, Win-Rate, Max-DD
- Monthly: Review if Still OOS-Valid (Parameter-Drift-Check)

---

## TIMELINE

- **Option 1:** Today (2026-05-21) evening, ~2h
- **Option 2:** 2026-05-22 morning, ~4h
- **Option 3:** 2026-05-22 afternoon, ~3h

**Total:** 1-2 days für Full Clarity.

---

## MEMORY & GIT

- Commit nach jedem Option-Lauf (auch FAILs)
- Memory aktualisieren: Befunde + gewählter Path forward
- CLAUDE.md updaten: Neue Success-Kriterien für Autonomous Deploy

