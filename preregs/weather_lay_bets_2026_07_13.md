# Trade-Reg: 3 Echtgeld-NO-Lays, Zieltag 2026-07-13

**Gesetzt:** 2026-07-12 ~18:15 UTC, je 5 $, Jupiter (jupiter_buy.py, Limit-Taker).
**Anlass:** Nutzer-Anweisung „beste 3 Lay-Wetten, je 5 $". Auswahl aus
`weather_outlier_screen[_low].py --date 2026-07-13` (Stand 17:43 UTC) +
40d-Nachprüfung (`weather_source_compare.py --days 40`) + Regen-Check (Open-Meteo).

## Gesetzte Wetten

| # | Markt | NO-Fill | Rendite | P 700d | P 40d | Markt-ID / Tx |
|---|---|---|---|---|---|---|
| 1 | London Lowest 18 °C | 0,989 | 1,1 % | 0,4 % | ~0 % | POLY-2878074 / 5Jto6Dyb… |
| 2 | Taipei High 36 °C | 0,921 | 8,6 % | 5,9 % | n/a¹ | POLY-2877818 / hB8Ut4Xo… |
| 3 | Beijing High 31 °C | 0,978 | 2,3 % | 0,7 % | 4,1 %² | POLY-2877834 / 4WTaZu8U… |

¹ Taipei: 40d-Kalibrierung nicht verfügbar (Station wird von
`weather_source_compare.py` stillschweigend geskippt — TODO fixen). Dafür
6–18 mm Regen prognostiziert (alle 5 Modelle), der das Tageshoch drückt;
Bucket liegt **über** dem Forecast (ENS korr. 32,6), Regen wirkt also FÜR den
Lay (umgekehrter Shanghai-Fall vom 11.07.).
² Beijing verletzt formal die Doppel-Kalibrierungs-Regel (40d: mu 33,96 σ 1,51
→ P 4,1 % > BE 2,3 %; 700d: ENS 35,0±1,5 → P 0,7 %). Auf explizite
Nutzer-Anweisung trotzdem gesetzt; hier transparent festgehalten.

## Angewandte Vetos (nicht gesetzt)

- **Shenzhen High 31 °C NO @0,958:** Regen-Veto — 2,0–2,6 mm in 4/5 Modellen,
  Bucket liegt **unter** dem Forecast (ENS 33,4) → Regen zieht das Hoch Richtung
  Bucket. Exakt der Shanghai-Fall. Zudem 40d-P 6,0 % > BE 4,2 %.
- **Munich High 32 °C NO @0,920:** 40d-Bias −1,70 °C (Modelle rechnen München
  aktuell deutlich zu kalt) → P(≥32°) ≈ 14 % bei 8,7 % Rendite, klar −EV.
- **Beijing-Alternativen / Rest der Leitern:** nach 700d-P vs Rendite −EV
  (u. a. Cape Town 20°, Taipei 37+, Karachi 35°, London-High 29°, Paris-Min 24°).

Dazu läuft weiter: London Lowest-13° NO @0,97 / 6 $ (gesetzt 11.07., gleicher
Zieltag 13.07.; Forecast-Minimum ~15,3 °C).

## Settlement (13./14.07.)

Ist = METAR-Extrem des lokalen Kalendertags 13.07. (EGLC min, RCSS max,
ZBAA max). Ergebnis + Claim-Status hier nachtragen.
