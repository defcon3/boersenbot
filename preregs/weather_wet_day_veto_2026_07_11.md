# Befund 11.07.2026 (Abend-Screen): Regentag-Veto + 48h-Lead-Check

**Kontext:** Abendliches Lay-Screening (~20:00–20:15 UTC, `weather_outlier_screen*.py`)
für die Zieltage 12.07. (Lead 24 h) und 13.07. (Lead 48 h), Highs + Lows.
Zwei methodische Erkenntnisse, beide als Werkzeug ins Repo gehoben.

## 1) Regentag-Veto: Shanghai „28 °C" (12.07.) NO @ 0,94 VERWORFEN

Der stärkste Filter-Pass des Abends (Rendite 6,4 %, P700 1,2 %, P40 2,5 %,
BE netto 5,6 % → Doppel-Kalibrierungs-Regel formal BESTANDEN, kein
Einzelmodell-Veto). Aber: alle 5 Modelle sahen für den 12.07. Regen
(ENS ~9,9 mm, 6–16 mm je Modell), Böen bis 74 km/h (JMA), Bewölkung 82–100 %
— Taifun-Ausläufer-Setup.

**Neues Werkzeug `weather_wet_conditional.py`** (Fehlerverteilung konditioniert
auf die vom Modell selbst vorhergesagte Tagesregensumme, previous_day1,
477 Tage ZSPD):

| Split (fc-Regen ENS) | n | Bias (Ist−ENS) | Sigma | P(korr. err ≤ −1,5°) |
|---|---|---|---|---|
| trocken (<1 mm) | 344 | +1,46 | 1,11 | 5,5 % |
| 1–5 mm | 58 | +0,92 | 1,22 | 17,2 % |
| nass (≥5 mm) | 75 | +0,76 | 1,20 | 21,3 % |
| sehr nass (≥9 mm) | 41 | +0,76 | 1,40 | 26,8 % |
| Sommer + ≥9 mm | 28 | +0,70 | 1,35 | 32,1 % |

An Regen-Forecast-Tagen halbiert sich der Zu-kühl-Bias (die pauschale
Korrektur hebt mu ~0,6° zu hoch), Sigma steigt, die kalte Flanke wird 4–6×
fetter. Fürs 28er-Fenster (mu-Standard 31,2): empirische Fensterquote an
Nass-Tagen 2,4–3,6 %, Normal-Rechnung mit Nass-Parametern ~5,1 % ≈ BE →
EV ~0 bis +3 % statt +4,7 %. Der Markt preiste das Fenster 6–8 % — plausibel
korrekt (08.07. blieb Shanghai nach Regenfront real bei max 26 °C; Ist-Hochs
davor 29/32/31). **Entscheidung: kein Trade.** Mildernder Umstand (macht es
grenzwertig, nicht gut): Regen fiel im Stundenprofil aller Modelle nachts
00–06 lokal, Nachmittag trocken — der historische Nass-Split mischt auch
Ganztags-Regen.

**Regel-Kandidat (ab jetzt Teil des Screen-Workflows, siehe Docstring
weather_outlier_screen.py):** Lay-Kandidat + nennenswerter Forecast-Regen am
Zieltag → Drilldown laufen lassen, P gegen den passenden Nass-Split halten,
nicht gegen die Allwetter-Kalibrierung.

## 2) 48h-Lead-Check: London „Lowest 13 °C" (13.07.) NO @ 0,97 GESETZT

Kandidat aus dem Low-Screen 13.07. (P700 0,1 %, dist 2,9°, kein Modell >7,1 %,
Rendite 3,1 % brutto / ~2,9 % netto, BE 2,9 %). Das Tief fällt erst ~31 h nach
Kauf → die 24h-Kalibrierung ist zu optimistisch. Dafür `weather_source_compare.py`
um `--lead N` erweitert (previous_dayN statt previous_day1):

| Rechnung | ENS-Bias | Sigma | mu | P(13er) |
|---|---|---|---|---|
| 700d Lead 24h (CSV) | −0,08 | 0,79 | 15,9 | 0,10 % |
| 40d Lead 24h | −0,07 | 0,59 | ~15,9 | <0,1 % |
| 700d Lead 48h | −0,26 | 0,90 | 16,1 | 0,19 % |
| 40d Lead 48h | −0,22 | 0,72 | 16,0 | 0,02 % |

Empirisch (48h, 700d, n=598): Ist ≥2,5° unter ENS in nur 0,2 % der Tage;
im 40d-Fenster 0/40. Lead-Aufschlag ist messbar (σ 0,79→0,90), trägt den
Trade aber locker. Zusatz-Struktur: Markt-Favorit = Modell-Favorit (16°) →
kein Zentrumsstreit, echte −3-Flanke (bei Minima die dünne kalte Seite);
METAR-Minima fielen 24→22→21→18. **Gesetzt: POLY-2878057 NO, 5,97 Kontrakte
@ 0,97 (6 $), Sig `3d6Zioqu…`.** Gewinn bei No-Touch ≈ +0,17 $ netto
(Claim gebührenfrei). Ausführungs-Detail: Limit-Preise brauchen 2
Dezimalstellen (0,975 → HTTP 400 „tick size", 0,98 ok, Fill 0,97).

## 3) Rest des Screens (der Vollständigkeit halber)

- **Tel Aviv 30° (12.07.) NO @0,977:** P_ens 2,7 % > BE 2,3 % → schon brutto ≈ −EV, raus.
- **Shanghai 27° (12.07.) NO @0,990:** Rendite 1,0 % — unter der ~3-%-Untergrenze (wie mittags).
- **Kuala Lumpur 35° (13.07.) NO @0,959:** P700 4,4 % > BE 3,8 % → EV-negativ, gestrichen
  (der Screen-Filter prüft kein EV — Ranking nach Rendite genügt nicht, immer P vs BE rechnen).
- **Watchlist für 12.07. mittags** (13.07.-Highs, dann Lead 24 h + frische 12z-Läufe):
  Shenzhen 36° (P 0,7 %, Rend. 3,2 %), Munich 33° (1,7 %, 3,1 %),
  Beijing 29°/≤28° (0,8 %/0,1 %, Rendite nur 2,7 %/2,4 %).
- **Lows 12.07.:** keine Kandidaten (Paris ECMWF-Veto 24–35 %, Seoul settelte bereits).
- Open-Meteo-Forecast-API hatte ~20:00 UTC einen transienten 503-Komplettausfall (Minuten später ok).
