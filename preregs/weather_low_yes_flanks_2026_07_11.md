# Pre-Reg: YES-Flanken auf Tagestief-Buckets (Paper-Forward, Tag 1)

**Registriert:** 2026-07-10 ~08:00 UTC, VOR Beginn der Settlement-Nacht.
**Modus:** NUR PAPIER (Nutzer-Entscheid 10.07. — kein Echtgeld, bis die These
Forward-Daten hat). Einsatz je Pick fiktiv 5 $ zum notierten YES-Ask (Taker).

## Hypothese

Die Jupiter/Polymarket-„Lowest temperature"-Buckets sind im Verteilungszentrum
überkonfident gepreist: Die markt-implizite Streuung ist enger als die
kalibrierte Forecast-Streuung (700d-Min-Kalibrierung, Lead 24h,
`preregs/weather_source_calib_min_2026_07_10.csv`). Dann sind Flanken-Fenster,
deren Modell-P nach **beiden** Kalibrierungen (700d Ganzjahr UND 40d Sommer)
über der Break-even-P des YES-Preises liegt, +EV für YES-Käufe.

Gegenhypothese: Unsere Normal-Annahme/Sigma übertreibt die Flanken (Minima-
Fehler sind linksschief; warme Flanke besonders verdächtig) — dann verlieren
die Picks im Mittel und der Markt hat recht.

## Registrierte Picks für Zieltag 2026-07-11 (Preise 10.07. 07:36 UTC)

| # | Stadt (Station) | Fenster | YES-Ask | BE-P | P 700d | P 40d | EV700 | EV40 | Flanke |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Paris (LFPB) | 20 °C | 0,10 | 10,0 % | 21,7 % | 18,0 % | +5,87 $ | +4,02 $ | kalt |
| 2 | London (EGLC) | 17 °C | 0,04 | 4,0 % | 10,0 % | 5,1 % | +7,52 $ | +1,40 $ | kalt |
| 3 | Shanghai (ZSPD) | 28 °C | 0,09 | 9,0 % | 16,0 % | 24,6 % | +3,89 $ | +8,67 $ | warm |
| 4 | Seoul (RKSI) | 25 °C | 0,09 | 9,0 % | 10,6 % | 24,9 % | +0,87 $ | +8,85 $ | warm |

Kalibrier-Basis: mu = roh-ENS(5 Modelle, Open-Meteo-Forecast 10.07. früh) − Bias.
700d: Paris 21,1±1,13 / London 18,5±0,79 / Shanghai 26,5±1,19 / Seoul 23,0±1,44.
40d-Sommer (ENS-Zeilen, reproduzierbar via
`python weather_source_compare.py --var min --days 40 --city Shanghai,Seoul,London,Paris`):
Paris bias +0,52 σ 1,05 / London −0,08 σ 0,60 / Seoul −0,10 σ 0,84 / Shanghai
(bias −0,10 σ 0,78 — aus Lauf 10.07.). P = Normal-Fensterwahrscheinlichkeit
[k−0,5, k+0,5).

## Auswertung (11./12.07.)

Ist = auf ganze °C gerundetes Tagesminimum des lokalen Kalendertags 11.07. je
Station (IEM/aviationweather-METAR; bisher deckungsgleich mit
Wunderground-Settlement). Fenster getroffen → Papier-Payout 5/Ask, sonst −5 $.
Zusätzlich notieren: tatsächliches Markt-Settlement der 4 Buckets.

**Urteilsregel:** Tag 1 entscheidet nichts (N=4). Bei Weiterverfolgung täglich
gleiche Registrierung ~2 Wochen (N≈30–60 Fenster); These lebt, wenn kumulierter
Paper-PnL > 0 UND Trefferquote > gepoolte Break-even-P. Stirbt sie, ist das der
Beleg „Markt-Sigma korrekt, Kalibrier-Sigma zu breit" — dann Flanken-YES endgültig
verwerfen und ggf. empirische Fehler-Quantile statt Normal-Annahme prüfen.

## ERGEBNIS Tag 1 (final, nachgetragen 12.07. — IEM-METAR, lokaler Tag komplett)

| # | Stadt | Pick | Ist-Min | Treffer | Paper-PnL |
|---|---|---|---|---|---|
| 1 | Paris (LFPB) | 20 °C | **21 °C** (05:00) | ✗ | −5 $ |
| 2 | London (EGLC) | 17 °C | **18 °C** (04:20) | ✗ | −5 $ |
| 3 | Shanghai (ZSPD) | 28 °C | **27 °C** (04:30) | ✗ | −5 $ |
| 4 | Seoul (RKSI) | 25 °C | **24 °C** (05:30) | ✗ | −5 $ |

**0/4, −20 $ Paper.** Auffällig: alle vier Ist-Minima lagen exakt 1 °C neben dem
Pick — jeweils in Richtung Markt-Favorit. Ein Punkt für die Gegenhypothese
(„Markt-Sigma korrekt, Kalibrier-Sigma zu breit"); Serie läuft weiter, Urteil
erst am Serienende. Serienstand nach Tag 1: 0/4, −20 $.
