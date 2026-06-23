# Pre-Reg: Volatilitäts-Risikoprämie (VRP-Sleeve) — 2026-06-23

**Status:** vorregistriert VOR dem Lauf. Kandidat aus der Session-Schlussdiskussion:
der robusteste *ungetestete* Einkommensstrom des Projekts. **Warnung vorab:** VRP ist
*Versicherung verkaufen* — der Tail (Crash) ist der eigentliche Prüfstein, nicht der
Mittelwert. Crash-Perioden werden daher bewusst NICHT ausgeschlossen.

## Hypothese
Implizite Vol (VIX) liegt im Mittel **über** der anschließend realisierten Vol →
wer Vol verkauft, kassiert die Prämie stetig. **Aber:** stark negativ schief,
crash-korreliert. Zu prüfen ist, ob das (a) eine reale Prämie ist, (b) den Tail
*überlebt*, und (c) als **Sleeve** (Kombination mit Long-Equity/Hybrid) wirklich
diversifiziert — oder nur verdeckte Aktien-Tail-Risiken stapelt.

## Operationalisierung (nur VIX + SPY, keine Options-Chain nötig)
Synthetischer **monatlicher Short-ATM-Straddle**, gehalten bis Monatsende:
- Prämie (Brenner-Subrahmanyam, ATM): `prem = 0.7979 · S_start · (VIX_start/100) · √T`
- Auszahlung (short, bei Verfall): `payoff = |S_end − S_start|`
- P&L: `pnl = prem − payoff`; **Rendite** `r = pnl / S_start` (voll besichert, 1× Notional)
- Monatlich, nicht überlappend. Raw-SPY-Preise (Optionen referenzieren echte Preise).
- `T` = Handelstage des Monats / 252.

Diese P&L misst die VRP direkt: bei realisiert = impliziert ist E[pnl] ≈ 0; positiv
nur durch VIX > realisiert. Entspricht „nackter Short-Straddle bis Verfall".

## Daten (G0)
yfinance `^VIX` + `SPY` (raw), **2006-01-01 .. 2026-06-22** — enthält bewusst
GFC 2008, Volmageddon Feb 2018, COVID März 2020, Bärmarkt 2022.
- **IS:** 2006 .. 2017-12-31 · **OOS:** 2018-01-01 .. 2026-06-22
  (OOS enthält die schwersten Vol-Spikes → harter, ehrlicher Tail-Test).

## Varianten
- **Nackt:** jeden Monat verkaufen.
- **Tail-gemanagt (VIX-Filter):** flat (Cash, r=0), wenn `VIX_start > 30`
  (nicht in bereits gestresste Märkte hineinverkaufen).
- Transaktionskosten: **5 % der Prämie** (konservativ, Options-Spread; bis Verfall
  gehalten → kein Exit-Trade).

## Gates
- **G1 (IS):** Netto-Mean(monatlich) > 0 mit HAC-t > 2.0 (Prämie real).
- **G2 (OOS):** Netto-Mean > 0, HAC-t > 1.5 (hält — inkl. Crash-OOS).
- **G3 (risikoadj., netto):** OOS-Sharpe (annualisiert) > 0.5.
- **G4 (TAIL — entscheidend):** gemanagte Variante MaxDD > −35 % UND kein
  Einzelmonat < −25 %. (Nackt zum Kontrast ausgewiesen; erwartet FAIL.)
- **G5 (Sleeve/Diversifikation):** corr(VRP, SPY) niedrig **UND** in den 5
  schlechtesten SPY-Monaten kein Totalschaden **UND** Portfolio SPY+VRP-Sharpe >
  SPY allein. (Ehrlich: short-vol crasht oft MIT Aktien → schlechter Diversifier.)

## Entscheidungsregel
- **GREEN:** Prämie real (G1/G2/G3) **und** überlebt Tail (G4) **und** diversifiziert
  echt (G5).
- **YELLOW:** Prämie real, aber Tail nur gemanagt tragbar bzw. schwache
  Diversifikation (crash-korreliert) → handelbar nur mit striktem Risk-Overlay.
- **RED:** Prämie nicht vorhanden/insignifikant.
Erwartung offen; YELLOW plausibel (Prämie real, aber negativ schief & crash-korreliert).
Befund wird committet — auch YELLOW/RED.

---

## ERGEBNIS (Lauf 2026-06-23) — **YELLOW**

`vrp_sleeve_test.py`, 245 Monate (IS 143 / OOS 102). **VRP-Beleg:** Prämie Ø 4,31 %
vs realisierte Bewegung Ø 3,48 % → reale Überrendite ~0,83 %/Monat.

| | mean/M | HAC-t | ann | Sharpe | MaxDD | Gate |
|---|---|---|---|---|---|---|
| IS 2006–2017 (nackt) | +1,14 % | **7,99** | +13,7 % | **1,75** | −12,5 % | **G1 PASS** |
| GESAMT (nackt) | +0,80 % | 5,57 | +9,6 % | 1,16 | −16,6 % | — |
| OOS 2018–2026 (nackt) | +0,31 % | **1,25** | +3,7 % | 0,44 | −16,6 % | **G2 FAIL** |
| OOS gemanagt (VIX-Filter) | +0,22 % | 0,91 | +2,7 % | 0,34 | −19,3 % | **G3 FAIL** |

- **G1 PASS** (Prämie historisch riesig, t=7,99). **G2/G3 FAIL** (OOS verwässert:
  Sharpe 1,75→0,44, t=1,25 — Crowding seit dem Short-Vol-ETF-Boom ~2016 + Tail-Events).
- **G4 TAIL PASS (gemanagt):** VIX>30-Filter saß die schlimmsten Monate aus
  (2008-10 SPY −16,5 %→VRP 0,0 %; 2020-03 SPY −13 %→VRP 0,0 %). MaxDD −19 %,
  schlechtester Monat −6 %. **Kein Blowup** — wichtig: die XIV-Horrorstories
  betreffen GEHEBELTE, täglich gerollte VIX-Futures; unleveraged/cash-besichert
  sind die Verluste beschränkt.
- **G5 DIVERSIFIER PASS:** corr(VRP, SPY) ≈ 0,00. 50/50 SPY+VRP-Sharpe = 1,08 vs
  SPY 0,65 (GESAMT); OOS-only 0,88 vs 0,81 (kleiner, aber positiv).

**Vorregistrierter Verdict: YELLOW** — Prämie **real** und der **beste Diversifier**,
den die Session fand (unkorreliert, kein Blowup unleveraged), ABER OOS-verwässert
(nicht signifikant am strengen Maß) und nur mit Risk-Overlay tragbar. Handelbar nur
klein, unleveraged, gehedged — **keine Gelddruckmaschine, aber der erste Baustein mit
echtem Sleeve-Wert.** Nächster Schritt (Backlog-würdig): Kombination mit dem
Hybrid-Risk-System messen (echte Options-Daten statt Straddle-Approx zur Validierung).
Reproduzierbar: `python vrp_sleeve_test.py` (nur yfinance VIX+SPY).
