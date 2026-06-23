# Pre-Reg: Market Intraday Momentum (SPY) — 2026-06-23

**Status:** vorregistriert VOR dem Lauf. Motiviert durch den Volumen-U-Form-Befund
(`intraday_volume_profile.py`, Commit `78b88f18`): Volumen konzentriert sich an
Open (~2.5×) und Close (~1.5×) — den beiden Fenstern mit informiertem/institutionellem
Handel. Hypothese aus Gao, Han, Li & Zhou (2018, JFE, "Market Intraday Momentum").

## Hypothese
Die Rendite der **ersten halben Stunde** (09:30→10:00 ET, `ret1`) prognostiziert
**positiv** die Rendite der **letzten halben Stunde** (15:30→16:00 ET, `ret_last`).
Sekundär: die 12. Halbstunde (15:00→15:30, `ret12`) prognostiziert ebenfalls.

**Mechanismus (warum es kein reiner Data-Mining wäre):** Positionierung/Information
vom Open wird von institutionellen Akteuren am Close vollendet (late-informed trading,
Rebalancing in die Schlussauktion) — exakt die Volumen-Spitzen aus dem Profil.

## Daten (G0 — erfüllt)
SPY 30-Min-Bars, **Alpaca SIP-Feed** (Vollmarkt, gratis für Historie >15 Min),
**2016-01-01 .. 2026-06-22**, `adjustment=raw` (reale Handelspreise — wir handeln
echte Preise). Zeitzone UTC→America/New_York mit DST. Reguläre Session = 13
Halbstunden 09:30..15:30 ET. Halbe Handelstage (kein 15:30-Bar) werden verworfen.

## Definitionen
- `ret1 = bar(09:30).close / bar(09:30).open − 1` (intraday, OHNE Overnight-Gap)
- `ret_last = bar(15:30).close / bar(15:30).open − 1`
- `ret12 = bar(15:00).close / bar(15:00).open − 1`
- **Strategie:** beobachte `ret1` um 10:00; gehe um 15:30 `sign(ret1)` long/short SPY,
  schließe um 16:00. Tagesrendite = `sign(ret1) · ret_last − Kosten`. Sonst flat.

## Splits
- **IS:** 2016-01-01 .. 2021-12-31
- **OOS:** 2022-01-01 .. 2026-06-22
- COVID-Fenster 2020-02-15..2020-04-30 für G5-Kontrolle separat ausweisbar.

## Gates
- **G1 (IS):** Newey-West(HAC, 5 Lags) t(β) > 2.0 in `ret_last = α + β·ret1`
  UND β > 0 UND Strategie-Mean(IS) > 0.
- **G2 (OOS):** gleiche Regel (sign aus IS), Strategie-Mean(OOS) > 0, t(OOS) > 1.5.
- **G3 (Netto):** OOS nach Round-Trip-Kosten. Basis-Annahme **2 bps** Round-Trip
  (SPY: ~0.2 bp Spread + Impact/Slippage), zusätzlich 1 bp ausgewiesen.
  Gate: net-Mean(OOS) > 0 UND net-t(OOS) > 1.0. Annualisierte Netto-Sharpe berichtet.
- **G4 (Signale):** ≥ 200 Trading-Tage/Jahr mit Signal (durch tägliches Setup trivial,
  aber explizit geprüft inkl. fehlender/halber Tage).
- **G5 (Robustheit):** Edge überlebt Ausschluss des COVID-Fensters; kein einzelnes
  Jahr trägt das Ergebnis allein (Per-Jahr-Tabelle); Ergebnis stabil für `ret12`-Signal.

## Entscheidungsregel
**GREEN** nur wenn G1–G5 alle bestehen (insb. G2 + G3 netto OOS). Bonferroni nicht
nötig (Einzel-Hypothese, kein Parameter-Grid). Andernfalls Befund ehrlich als
YELLOW/RED committen — auch wenn er scheitert.

---

## ERGEBNIS (Lauf 2026-06-23) — **RED**

`intraday_momentum_test.py` auf SPY SIP 30-Min, 2631 Handelstage (IS 1511 / OOS 1120).

| | β (ret1→ret_last) | HAC-t | Strategie-Mean | t | Sharpe (ann) |
|---|---|---|---|---|---|
| IS 2016–2021 | **−0.0795** | −0.55 | −0.63 bps/Tag | −0.65 | −0.29 |
| OOS 2022–2026 | −0.0269 | −0.65 | −0.70 bps/Tag | −0.84 | −0.42 |
| OOS netto 2bp | — | — | −2.70 bps/Tag | −3.25 | −1.63 |

- **G1 FAIL** (t=−0.55, β<0 — Vorzeichen sogar falsch), **G2 FAIL**, **G3 FAIL**, G4 PASS.
- Per-Jahr Münzwurf (Hit 46–54 %); nur 2023 positiv (t=1.54), 2022 stark negativ.
- Ohne COVID: β=−0.008 (t=−0.26) → kein Effekt, nicht COVID-getrieben.
- Alt-Signal `ret12`: +0.54 bps OOS, aber t=0.73 (insignifikant).

**Befund:** Kein Intraday-Momentum in SPY 2016–2026. Der publizierte Effekt
(Gao/Han/Li/Zhou 2018, Sample 1993–2013) ist post-publication **wegarbitriert**
(Anomalie-Decay). Konsistent mit der Projekt-Gesamtlage: einfache/publizierte
Edges überleben OOS nicht. Vorregistrierung verhinderte Post-hoc-Subset-Fishing.
Daten (`spy_30min_sip_2016_2026.pkl`) reproduzierbar via `fetch_spy_30min.py` (VPS).
