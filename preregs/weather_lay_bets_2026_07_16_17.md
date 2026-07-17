# Trade-Reg: 6 Echtgeld-NO-Lays, Zieltage 2026-07-16 (BA-Brett) + 2026-07-17

> **Prozess-Hinweis (ehrlich):** Diese Reg wurde am **17.07. nachgetragen** —
> Auswahlregeln, Kandidaten und Zwischenstände waren ab dem 15.07. nur im
> Session-Memory dokumentiert, nicht als committete Datei. Alle Angaben unten
> zu Setzung/Zwischenständen stammen aus diesen zeitgestempelten Notizen; die
> Settlement-Zahlen aus der Jupiter-History bzw. WU/METAR. Lücke benannt,
> damit sie nicht wie eine saubere Vorregistrierung aussieht.

**Gesetzt:** 15.07.2026 ~11:20 UTC, je ~5 $, Jupiter (Limit-Taker), Wallet
`4XxStoKPzo…`. **Auswahlregel:** `weather_outlier_screen.py` + frische Preise;
nur EV > 0 (EV = BE − P_pess, d. h. NO-Ask < 1 − P_pess) UND Modellspanne ≤ 3°
(harter Veto). Sicher aussehende −EV-Tails (Ankara 28°, Helsinki, Wellington …)
bewusst NICHT gesetzt — kein wissentlicher Minus-Trade, um „auf 3 zu kommen".
Kein Take-Profit (Wetter-TP seit b08788ed aus), Hold-to-Settlement + Auto-Claim.

## Zieltag 16.07. — Buenos-Aires-Brett (einziges +EV-Angebot des Tages)

| Markt | Bucket | NO-Fill | P_pess (Setzung) | EV |
|---|---|---|---|---|
| POLY-2929402 | BA 23° | 0,740 | 26 % (GFS-Ausreißer, dist 1,6°) | +11,9 pp |
| POLY-2929403 | BA 24° | 0,948 | 2,7 % | +2,2 pp |
| POLY-2929404 | BA 25° | 0,980 | 0,2 % | +1,3 pp |

Forecast ~21,4°, Markt-Fav 22° → drei Warm-Tails auf einem Brett; die
Konzentration war der ehrliche Preis dafür, keine −EV-Streuung zu kaufen.

**Verlauf 16.07. (zeitgestempelte Notizen):** Präfrontale Warmlage (CAVOK,
N-Wind, Druck 1005→1001 fallend). Frische Läufe ~1° wärmer (korr. 22,2±1,3 /
22,4±1,1), Markt machte 23° zum Fav. M2M mittags ≈ −2,1 $, ~17:30 UTC ≈ −3,0 $.
**Gehalten** (eigene P ≈ Markt-P, Verkauf = ~9 % Vig zahlen). Nutzer-Idee
„22er-Lay dazu" geprüft und VERWORFEN: reiner Brett-Hedge, friert −2,9…−3,5 $
ein, killt das einzige Plus-Szenario, −EV @0,82, in-play gegen schnellere
Trader (Lay-all-Lektion `weather_layall_tp.py`). 18:00Z-METAR 23 °C → Lesart
„23er verloren, außer Durchschuss ≥23,5".

**SETTLEMENT (17.07. ~03:40 UTC, Auto-Claim autonom):**

**Ist-Hoch SAEZ 16.07. = 24 °C** (WU-Settlement-Reihe Max 24, Peak 16:00 lokal;
METAR/IEM 24,0 → konsistent). Der Durchschuss trat ein — **die Rollen drehten
sich:** der Risiko-Pick 23er GEWANN, der „sichere" 24er VERLOR.

| Bucket | Ausgang | PnL (Jupiter-History) |
|---|---|---|
| 23° | **gewonnen**, geclaimt 03:38Z | **+1,59 $** |
| 24° | **VERLOREN** (position_lost 03:48Z) | **−4,77 $** |
| 25° | gewonnen, geclaimt 03:37Z | +0,09 $ |
| **Brett** | | **−3,09 $** |

(Szenario-Rechnung vom Vorabend: Max 23 → −4,63 / Max 24 → −3,14 $ — real
−3,09, minimal besser wegen Fill-Details.)

**Kalibrier-Datenpunkte aus dem Brett:**
1. **Ist +2,6° über dem Setz-Forecast** (21,4 → 24) und ~+1,6–1,8° über den
   korrigierten Sichten vom Morgen des Zieltags (22,2/22,4). ALLE Modelle zu
   kalt; am nächsten dran ECMWF-Stundenprofil (22,5–22,6). Gegenstück zu
   Beijing/Madrid (dort Ist 1,7°/1,2° UNTER korr. ENS): **µ-Fehler in
   Übergangslagen ist groß und lagebedingt in beide Richtungen** — kein
   einseitiger Warm-Bias. Das dist≥2°-Gate + Spannen-Veto bleiben die Abwehr.
2. **Ein P_pess-2,7-%-Tail ist eingetreten** (24er). Erster Verlust eines
   „sicheren" Tails seit Serienstart 08.07.; der als riskant registrierte
   23er (P_pess 26 %) wurde vom selben Durchschuss gerettet. Einzelereignis,
   noch keine Statistik — gehört in die Klassen-Forward-Auswertung
   (bb_WeatherLadders, ~Ende Juli).
3. Die 18Z-Lesart „23er verloren" war voreilig — der Peak kam 19Z (16 h
   lokal) mit 24 °C. Nowcast-Urteile erst nach dem klimatologischen
   Peak-Fenster fällen (BA: 15–17 lokal, nicht 15:00).

## Zieltag 17.07. — Shenzhen 33/34 + Ankara 27 (settlet 18.07.)

| Markt | Bucket | NO-Fill | P_pess (Setzung) | Anmerkung |
|---|---|---|---|---|
| POLY-2930787 | Shenzhen 33° | 0,940 | 0,5 % | dist 2,7° |
| POLY-2930788 | Shenzhen 34° | 0,944 | 0,0 % | dist 3,7°, Spanne 1,3° |
| POLY-2930607 | Ankara 27° | 0,946 | 3,4 % | Kalt-Tail unter Fav 29 |

**Bekanntes Risiko (16.07. entdeckt, vorab notiert):** Shenzhen settlet auf
einer WU-Reihe, die an 11/15 Tagen ±1–2° vom METAR abweicht — die Kalibrierung
misst dort das falsche Ziel; reale Verlust-P eher ~8 % (33er) + ~3 % (34er).
**Beim Settlement 18.07. wu_settle_k vs METAR-Max notieren.**

**Zwischenstand 17.07. ~04:45 UTC (Shenzhen 12:45 lokal):** Shenzhen kühl —
METAR-Max bisher 29,0 (12 h lokal), WU-Reihe zeigt sogar nur 27 (und hinkt
nach; live erneut ±2°-Divergenz zur METAR-Reihe). 33/34 praktisch außer
Reichweite. Ankara 07:20 lokal erst 18 °C (Tages-Max dort ~15–16 h lokal,
Mitternachtswert 21 °C zählt bereits als Tageswert) — offen, P klein.

**Ausgang 17.07.:** _wird nach Settlement am 18.07. nachgetragen._

## Buchstand der Echtgeld-Serie (seit 08.07., Stand 17.07. früh)

08.07. +0,95 | 09.07. +0,48 (BA-20 schwebt: +0,15 ausstehend, Polymarket-Ops-
Ausfall, unresolved seit 09.07.) | 10.07. +0,16 (+ Nutzer-Solo London +1,64)
| 13.07. +0,70 | TP-Exits Beijing/Madrid +1,15 | **16.07. −3,09** | 17.07.
offen (max +0,86). Serie gesamt damit ≈ **+0,35 $** vor den 17.07.-Lays —
die Vig-Marge ist dünn; ein einziger Tail-Treffer frisst ~9 Gewinner. Genau
dafür läuft der Klassen-Forward (±2/±3-Empirie sagt +2…+6 %).
