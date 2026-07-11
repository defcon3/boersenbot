# Pre-Reg: YES-Flanken auf Tagestief-Buckets (Paper-Forward, Tag 2)

**Registriert:** 2026-07-11 ~13:05 UTC, VOR Beginn der Settlement-Nacht
(Seoul-Minimum fällt ~18–21 UTC am 11.07., Paris ~02–05 UTC am 12.07.).
**Modus:** NUR PAPIER (Fortsetzung der Serie aus
`weather_low_yes_flanks_2026_07_11.md`, Nutzer-Entscheid 10.07.).
Einsatz je Pick fiktiv 5 $ zum notierten YES-Ask (Taker).

## Hypothese

Unverändert Tag 1: Markt-implizite Streuung der „Lowest temperature"-Buckets
enger als kalibrierte Forecast-Streuung → Flanken-Fenster mit Modell-P über
Break-even-P nach **beiden** Kalibrierungen (700d Ganzjahr UND 40d Sommer)
sind +EV für YES-Käufe. Gegenhypothese: Normal-Annahme/Sigma übertreibt die
Flanken (warme Flanke bei Minima besonders verdächtig).

## Registrierte Picks für Zieltag 2026-07-12 (Preise 11.07. 12:55 UTC)

| # | Stadt (Station) | Fenster | YES-Ask | BE-P | P 700d | P 40d | EV700 | EV40 | Flanke |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Seoul (RKSI) | 26 °C | 0,07 | 7,0 % | 8,9 % | 20,3 % | +1,34 $ | +9,47 $ | warm |

Nur 2 Städte hatten zum Registrierungszeitpunkt Low-Märkte für den 12.07.
(Paris, Seoul; Hong Kong ohne aufgelöste Station geskippt) → N heute klein.

**Nicht qualifiziert** (Doppel-Kalibrierungs-Regel): Seoul 23 °C @ 0,15
(P700 23,4 % +, aber P40 4,1 % − → Kalibrierungen uneins, kein Pick);
Paris 23 °C @ 0,16 (P700 11,8 / P40 14,2 — beide unter BE);
Paris 24 °C @ 0,08 (P700 2,3 / P40 2,6).

**Außerhalb der Serie notiert (kein Flanken-Fenster, kein Pick):**
Paris „21 °C or below" @ 0,42 — Modell-P 57,0 % (700d) bzw. 50,0 % (40d),
beide über BE 42 %. Das ist aber ein Median-Shift-Statement (Markt-Favorit
22 °C vs Modell-Median ~21,3–21,5), nicht die Flanken-These; zudem
Modell-Split GFS korr. 20,0 vs ECMWF korr. 22,8 → Normal-Annahme wackelig.
Wird hier nur als Beobachtung festgehalten (fiktive Verfolgung: träfe ≤21 °C
ein, wäre das +6,90 $ Paper, sonst −5 $ — zählt NICHT in den Serien-PnL).

Kalibrier-Basis: mu = roh-ENS(5 Modelle, Open-Meteo-Forecast 11.07. mittags)
− Bias. 700d-Min (`weather_source_calib_min_2026_07_10.csv`): Paris ENS-Bias
+0,709 σ 1,126 → mu 21,3; Seoul +0,961 σ 1,443 → mu 23,8. 40d-Sommer (Lauf
11.07., `weather_source_compare.py --var min --days 40 --city Paris,Seoul`):
Paris Bias +0,51 σ 1,05 → mu 21,5; Seoul Bias −0,13 σ 0,81 → mu 24,9.
Auffällig: Seoul-mu divergiert 1,1 °C zwischen 700d und 40d — die
Ganzjahres-Min-Kalibrierung zieht Seoul im Sommer zu stark nach unten.
P = Normal-Fensterwahrscheinlichkeit [k−0,5, k+0,5).

Korr. Modell-Forecasts (700d) zum Registrierungszeitpunkt:
Paris GFS 20,0 / ICON 20,7 / UKMO 21,2 / JMA 21,7 / ECMWF 22,8;
Seoul GFS 22,5 / ICON 24,4 / UKMO 23,3 / JMA 23,9 / ECMWF 25,1.

## Auswertung (12./13.07.)

Analog Tag 1: Ist = auf ganze °C gerundetes Tagesminimum des lokalen
Kalendertags 12.07. je Station (IEM/aviationweather-METAR). Fenster getroffen
→ Papier-Payout 5/Ask, sonst −5 $. Zusätzlich Markt-Settlement notieren.
Urteilsregel unverändert (kumulierter Paper-PnL + Trefferquote vs gepoolte
BE-P über ~2 Wochen).
