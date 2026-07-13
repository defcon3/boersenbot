# Trade-Reg: 1 Echtgeld-NO-Lay, Zieltag 2026-07-15 (48h-Lead-Test)

**Gesetzt:** 2026-07-13 ~13:20 UTC, 6 $ Budget, Jupiter (jupiter_buy.py, Limit-Taker).
**Anlass:** Nutzer-These: „Leitern 2 Tage voraus sind zerklüftet, viele Buckets
mit zweistelliger Rendite — prüfen." Erster Echtgeld-Trade mit **Lead-48h-
Kalibrierung** (`weather_source_compare.py --lead 2`, 40d + 700d) statt der
Lead-24h-Kalibrierung des Screens.

## Methodik-Befund 48h (Kern der Prüfung)

- **ENS verliert 24h→48h fast nichts:** MAE 1,02→1,02 °C (40d, n=160,
  Madrid/Jeddah/Helsinki/Cape Town gepoolt); Madrid-ENS-Sigma sogar 0,55→0,53.
  Der Markt preist bei 48h aber deutlich mehr Unsicherheit → strukturelle
  Edge-Quelle real.
- **Einzelmodelle degradieren teils** (ICON 1,00→1,19; JMA auf Lead 2 für
  manche Städte kaputt, Bias −2,5) → Modell-Vetos häufiger.
- **3 der 4 formalen Screen-Passes des Tages waren nach Lead-2-Kalibrierung
  −EV** (Jeddah 34°, Helsinki 24°, Cape Town 15° — Renditen 1,4–2,0 % decken
  die echte 48h-Rest-P nicht). Lehre: Bei Zieltag >24h nie mit Screen-P
  (Lead-1) handeln, immer Lead-2 nachrechnen.

## Gesetzte Wette

| Markt | NO-Fill | Kontrakte | Kosten | Rendite | BE | Tx |
|---|---|---|---|---|---|---|
| Madrid High 35 °C, 15.07. (POLY-2902148) | 0,8047 | 7,18 | 5,78 $ | 24,3 % | 19,5 % | 25emFa2L… (finalized) |

Gewinn bei Verfehlen **+1,40 $**; Verlust −5,78 $. Limit 0,81 (Ask 0,79–0,80,
dünnes Buch füllte bis 0,81; erster Sende-Versuch 429 rate-limit, Retry ok).

**P(35er-Fenster) nach allen vier Kalibrier-Sichten** (Roh-ENS 36,5 → korr. ~37,2):
700d-Lead1 5,5 % | **700d-Lead2 5,0 %** (Bias −0,85, σ 1,16) | 40d-Lead1 0,1 %
| **40d-Lead2 0,3 %** (Bias −0,66, σ 0,53). Alle klar unter BE. Schärfstes
Einzelmodell ECMWF 9–15 % < BE. **Regen 0,0 mm in allen 5 Modellen.**

**Lage:** Madrid-Dip heute (Ist 38→34→~32), Rückerwärmung laut ALLEN Modellen
schon am 14.07. (36,2–37,3), 15.07. = Plateau-Tag 2 (35,7–37,3) — kein
Sprungtag. Modelle treffen den heutigen Ist-Tag exakt (~32).

**Transparent benannte Restrisiken:**
1. Übergangslage (Rückerwärmung) — Normal-Kalibrierung erfasst Timing-Fehler
   solcher Lagen strukturell schlecht (Plateau-Struktur mildert).
2. **Zentrumsstreit light:** Markt-Fav 36° (YES 0,43) vs Kalibrier-Zentrum 37,2°;
   das gelayte 35er ist Markt-Fav−1. Hat der Markt recht, ist P(35er) eher
   15–20 % → EV ≈ 0. Der Trade wettet „Kalibrierung schlägt Markt-Zentrum" —
   die YES-Paper-Serie hat diese Wette bisher verloren (0/5), allerdings auf
   der YES-Seite mit −EV-Preisen; hier trägt die Prämie das Risiko.

## Geprüft und NICHT gesetzt (15.07.)

- **Jeddah 36° @0,77 (30 %):** GFS-L2 29 % > BE 23 %; JMA-Chaos (roh 31 vs 39).
- **Helsinki 26° @0,78 (28 %):** ICON-L2 26 % > BE 22 % (ICON/ECMWF ~26).
- **London 27° @0,87 (15 %):** UKMO 25 % (sieht 27,8); dist 1,7 < 2.
- **Tokyo 30° @0,89 (12 %):** ICON 28 % am Fenster; ECMWF-Ausreißer 35,3.
- **Paris 34° @0,90 (11 %):** GFS 19 % (35,2 vs JMA 29,7 = 5,5° Spanne).
- **Madrid 34° @0,974:** einziger überlebender formaler Pass, Rendite 2,7 % zu dünn.
- **Lows:** London-Min 14/15/16 Rendite ≤2,1 % (Spread frisst alles), Paris-Min
  ECMWF/UKMO-Vetos.

## Settlement (15./16.07.)

Ist = LEMD-Hoch lokaler Kalendertag 15.07. (22:00 UTC 14.07. – 22:00 UTC
15.07.); Wahrheit = Wunderground, METAR Fallback. Verlustfenster nur Ist-Hoch
34,5–35,49. Resolution frühestens 16.07., Autopilot claimt. Ergebnis hier
nachtragen — auch als Datenpunkt für die Frage „Kalibrierung vs Markt-Zentrum".
