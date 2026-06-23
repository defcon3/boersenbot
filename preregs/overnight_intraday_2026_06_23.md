# Pre-Reg: Overnight- vs. Intraday-Rendite (Overnight-Edge) — 2026-06-23

**Status:** vorregistriert VOR dem Lauf. Umsetzung des Backlog-Eintrags
(`BACKLOG.md`). Knüpft an den früher YELLOW/G0-geschlossenen Overnight-Strang an
(`overnight_intraday_*.py`, Commits `97b3b371`/`0c6b0487`/`c21ce7ce` — damals
ES-Futures-G0-FAIL). Vorbefund 2026-06-23 (`cross_session_test.py`): milde
Overnight-Reversion der Spätbewegung (PM→Gap −0,156), R²~2,4 %.

## Hypothese
**Deskriptiv:** Die Aktien-Risikoprämie fällt überwiegend **über Nacht**
(Close→Open) an; der **Intraday**-Teil (Open→Close) ist ~flach/negativ
(Lou/Polk/Skouras 2019, „A Tug of War"). → E[ON] > 0 und E[ON] > E[ID].

**Handelbar:** Eine Overnight-only-Strategie (MOC-Kauf, MOO-Verkauf, nur über
Nacht investiert) schlägt risikoadjustiert Buy&Hold — **trotz** täglicher
Round-Trip-Kosten. (Das ist die eigentliche Härte: ~252 Round-Trips/Jahr.)

## Definitionen
- `ON[t]  = Open[t] / Close[t-1] − 1`   (Overnight, dividendenbereinigt → Total Return)
- `ID[t]  = Close[t] / Open[t] − 1`     (Intraday)
- `Full[t]= (1+ON)(1+ID) − 1`           (Buy&Hold-Tagesrendite)
- Strategien (täglich): Overnight-only = `ON[t]`; Intraday-only = `ID[t]`;
  Buy&Hold = `Full[t]`. Netto: Overnight-only − Round-Trip-Kosten/Tag.

## Daten (G0)
yfinance **Tages-OHLC, auto_adjust=True** (Dividenden korrekt dem Overnight
zugerechnet, da Ex-Drop am Open passiert), **2010-01-01 .. 2026-06-22**.
Primär **SPY**; Robustheit **QQQ, IWM**. Cross-Check der Open/Close-Qualität
gegen `spy_30min_sip_2016_2026.pkl` (Intraday-Korrelation yfinance vs SIP).

## Splits & Kosten
- **IS:** 2010 .. 2021-12-31 · **OOS:** 2022-01-01 .. 2026-06-22.
- COVID 2020-02-15..2020-04-30 für G5-Kontrolle separat.
- **Round-Trip-Kosten:** Basis **2 bps/Tag** (SPY: ~0,1–0,2 bp Spread + Impact,
  MOC/MOO konservativ); zusätzlich 1 bp ausgewiesen. Buy&Hold-Kosten ~0 (einmalig).

## Gates
- **G1 (IS):** ON-Mean > 0 mit HAC-t > 2.0 UND ON-Jahresrendite > ID-Jahresrendite.
- **G2 (OOS):** ON-Mean > 0, HAC-t > 1.5, ON > ID hält.
- **G3 (HANDELBAR, netto):** Overnight-only nach 2 bp/Tag in OOS:
  Netto-Jahresrendite > 0 **UND** Netto-Sharpe > Buy&Hold-Sharpe (OOS).
- **G4:** ≥ 200 Handelstage/Jahr (trivial, explizit geprüft).
- **G5:** Effekt überlebt COVID-Ausschluss; ON > ID in der Mehrheit der Einzeljahre.

## Entscheidungsregel
- **GREEN:** deskriptiv (G1,G2,G5) **und** handelbar (G3) bestehen → Forward-Test.
- **YELLOW:** deskriptiv besteht, aber G3 scheitert (Edge real, aber durch
  tägliche Kosten nicht erntbar).
- **RED:** schon deskriptiv (G1/G2) gescheitert.
Erwartung ehrlich offen; YELLOW am wahrscheinlichsten (Kosten fressen tägliche
Round-Trips). Befund wird committet — auch FAIL/YELLOW.

---

## ERGEBNIS (Lauf 2026-06-23) — **RED** (per vorregistrierter Regel, SPY primär)

`overnight_edge_test.py`. Datenqualität bestätigt: Intraday-Korr yfinance~SIP = **0,998**.

**SPY (primär):** kumuliert 2010–2026 Overnight ×3,94 vs Intraday ×2,24.
| | ON ann | ON t | ID ann | Gate |
|---|---|---|---|---|
| IS 2010–2021 | +9,9 % | 3,30 | +5,4 % | **G1 PASS** (ON>ID, t>2) |
| OOS 2022–2026 | +6,4 % | 1,32 | +6,8 % | **G2 FAIL** (t<1,5, ID holte auf) |

- **G3 FAIL (handelbar):** Overnight-only netto 2 bp = +1,3 % ann / Sharpe 0,12 vs
  **Buy&Hold +13,0 % / 0,74**. Selbst brutto (6,4 %) < B&H. Tägliche Round-Trips
  (~5 %/J Kosten) + verzichteter Intraday-Teil → nie erntbar.
- **G5:** ON>ID in 11/17 Jahren; ohne COVID ON +10,1 % (t=4,72) vs ID +5,5 %.

**Vorregistrierter Verdict: RED** — SPY-OOS-Gate (G2) gescheitert, Overnight-Prämie
hielt 2022–2026 für SPY nicht (2022 ON −13 %, 2023/2025 ID>ON).

**Robustheit / Einordnung (nachrichtlich, ändert den Verdict NICHT):** Das
**deskriptive Phänomen ist real** und für andere Indizes OOS intakt:
- QQQ: ON ×7,39 vs ID ×2,48; OOS ON +9,9 % (t=1,66) > ID +7,5 %.
- IWM: ON ×7,58 vs ID ×**0,77** (Intraday verlor über 16 J Geld!); OOS ON +9,9 % > ID +0,7 %.

→ Praktisches Gesamtbild: Overnight-Prämie historisch/Cross-Sectional real, aber
(1) für SPY zuletzt verblasst und (2) **G3 überall FAIL** — Overnight-only schlägt
Buy&Hold nie netto. **Keine Gelddruckmaschine.** Konsistent mit Projekt-Gesamtlage.
Daten reproduzierbar via `overnight_edge_test.py` (yfinance, SIP-Cross-Check optional).
