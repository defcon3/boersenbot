# Ex-Tag-Rebound (Delayed Dip) — DE — FAIL (2026-06-23)

**Test:** `exday_rebound_test.py` · **Daten:** DAX40+MDAX50, 2010–2025
(Cache `dax_mdax_divs_2010_2025.pkl`, Benchmark `^GDAXI`), ohne COVID
(15.02.–30.04.2020). 1.095 Dividenden-Events. OOS-Split: Train ≤2018, Test ≥2019.

## Hypothese (visuell von veitluther.de-Charts)
„2-3 Tage **nach** dem Ex-Tag ist der Kurs genauso tief oder tiefer, dann
2-5 Tage später ist er wieder **deutlich erhöht**." Implizierte Strategie:
Kauf in die Delle (Ex+2 / Ex+3), Verkauf 2-5 Trading-Tage später. **Kein**
Dividend-Capture (Kauf liegt nach Ex-Tag) → reine Kurs-Mean-Reversion-These.

## Gates
- G1 IS: t_train > 2.0 & mean > 0
- G2 OOS: t_test > 2.51 (Bonferroni K=8, einseitig) & mean > 0
- Excess **immer** vs ^GDAXI (DE-Dividenden clustern in der HV-Saison Frühjahr
  → roher Anstieg wäre sonst nur Bullenmarkt-Drift).

## Ergebnis — RED
**Teil 1 (Delle) bestätigt sich deskriptiv:** Excess-Median Ex+2 = −0,08 %,
Ex+3 ≈ 0. Kurz nach Ex-Tag relativ flach/leicht negativ.

**Teil 2 (Rebound) ist eine Benchmark-Illusion:** Der *rohe* Kurs-Median
steigt sichtbar (Ex+3 +0,20 %, Ex+8 +0,30 %, Ex+12 +0,20 %) — genau die
Chart-Linie, die der Nutzer sieht. Der **Excess** vs Index wird über die Zeit
aber **immer negativer** (Ex+8 −0,25 %, Ex+12 −0,45 %): Die Dividendenaktie
steigt *weniger* als der Markt. Der „Rebound" ist Markt-Beta, kein
dividendenspezifischer Effekt.

**Strategie-Grid (Kauf Ex+kb, Verkauf Ex+ks, kb∈{2,3}, hold 2–5):**
- IS schon flach (bestes t_train = 0,95) → kein Edge in-sample.
- OOS bestes t_test = 0,42 (mean +0,05 %); Mehrzahl negativ; kb=3/ks=8 sogar
  signifikant **negativ** (t=−2,59). Trefferquote durchweg < 50 %.

→ Kein Window besteht G1/G2. **Hypothese widerlegt.** Konsistent mit dem
früheren Cum-Tag-Dividend-Capture-FAIL (`dividend_capture_de_2026_05_25`):
Der Markt preist den Ex-Tag-Drop effizient.

## Lesson
Klassischer Chart-Trugschluss: absolute Kurslinie erholt sich → aber das ist
Beta, nicht Edge. Ohne Benchmark-Abzug handelt man nur „Aktie steigt im
Bullenmarkt". Reproduktion: `python exday_rebound_test.py`
(CSV `exday_rebound_paths.csv` reproduzierbar, daher gitignored/uncommittet).
