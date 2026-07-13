# Trade-Reg: 1 Echtgeld-NO-Lay, Zieltag 2026-07-14

**Gesetzt:** 2026-07-13 ~12:05 UTC, 6 $ Budget, Jupiter (jupiter_buy.py, Limit-Taker).
**Anlass:** Nutzer-Wunsch nach höherer Rendite („Märkte 1–2 Tage voraus sind
zerklüftet, viele Buckets >10 %") → Screening `weather_outlier_screen[_low].py
--date 2026-07-14` (11:36/11:43 UTC) + 40d-Nachprüfung + Regen-Check.

## Gesetzte Wette

| Markt | NO-Fill | Kontrakte | Kosten | Rendite | P 700d | P 40d | BE | Tx |
|---|---|---|---|---|---|---|---|---|
| Beijing High 33 °C (POLY-2889148) | 0,790 | 7,18 | 5,67 $ | 26,6 % | 4,3 % | 13,0 % | 21 % | 4ar95txq… (finalized) |

Gewinn bei Verfehlen: **+1,51 $**; Verlust: −5,67 $. Limit 0,81, Fill am Ask 0,79.

**Begründung:**
- Trockener, stabiler Hitzetag, alle 5 Modelle einig (roh ZBAA-Grid 33,9–37,5;
  700d-korr. ENS 35,8±1,4, 40d-korr. ENS 34,8±1,52).
- **Doppel-Kalibrierungs-Regel BESTANDEN** (anders als der Beijing-31er vom
  13.07.): P(33er) 4,3 % (700d) bzw. 13,0 % (40d), beide klar unter BE 21 %.
- **Stresstest Einzelmodelle:** selbst das pessimistischste (GFS 40d: 17 %)
  bleibt unter BE — einziges >10 %-Bucket des Tages mit dieser Eigenschaft.
- **Kein Wet-Veto:** 0,0 mm Regen in allen 5 Modellen (Bucket liegt unter dem
  Forecast — mit Regen wäre das die Shanghai-Falle gewesen).
- 40d-Kalibrierung tagesfrisch validiert: ihr mu traf das Beijing-Ist am 13.07.
  exakt (33,96 → 34,0); ihr mu für 14.07. = 34,8 → 33er ~1,3 σ entfernt.

**Dokumentierte Filter-Überstimmung:** Der formale Screen führte den 33er unter
„knapp verfehlt" (GFS-P 12,2 % > 10-%-Schwelle MAX_PMODEL). Bewusst überstimmt,
weil (a) GFS das mit Abstand schwächste ZBAA-Modell ist (700d σ 2,30; 40d Bias
+1,51/σ 2,32), (b) selbst die GFS-Sicht den Lay +EV lässt (12–17 % < BE 21 %),
(c) alle anderen Modelle 700d ≤ 11,2 % geben. Die 10-%-Schwelle ist eine
Einigkeits-Heuristik, kein EV-Urteil — hier als regelbasierte Ausnahme mit
EV-Marge nach JEDEM Einzelmodell festgehalten, nicht als Freistil.

## Geprüft und NICHT gesetzt

- **Beijing 32 °C NO @0,945** (einziger formaler Screen-Pass, Rendite 5,3 %):
  40d-P 5,0 % = BE 5,0 % → EV 0 nach konservativer Sicht, Doppel-Regel
  verletzt. Beleg, dass Renditen um 5 % bei NO ≥0,94 kaum tragen.
- **Wuhan 39° @0,809 (24 %):** UKMO korr. 39,2 mitten im Fenster (P 18 % ≈ BE);
  Modellspanne 34–39° (JMA-Ausreißer).
- **Tel Aviv 31° @0,82 (22 %):** ICON korr. 32,1 → P 21 % > BE 18 %.
- **Milan 33° @0,85 (18 %):** P_ens 17,4 % > BE 15 %; GFS 38,9 = 8° Spanne.
- **Jeddah 36° @0,89 (12 %):** GFS-P 22 % > BE 11 %.
- **Kuala Lumpur 34° @0,89 (12 %):** UKMO-P 18 % > BE 11 %.
- **Lows (nur 5 Städte gelistet):** kein Kandidat — Paris 23° hat ECMWF exakt
  im Fenster (P 35 %), Seoul 26° ECMWF 29 %; Renditen ohnehin ≤6 %.

Kernbefund fürs Muster: Die „Zerklüftung" bei 24–40h-Lead ist überwiegend
korrekt gepreiste Modell-Uneinigkeit — bei jedem verworfenen >10 %-Bucket macht
mindestens ein Einzelmodell den Lay −EV. Lay-Value entsteht nur, wo die Modelle
einig sind UND der Markt trotzdem Restangst preist.

## Settlement (14./15.07.)

Ist = Extrem des lokalen Kalendertags 14.07. an ZBAA (16:00 UTC 13.07. –
16:00 UTC 14.07.); Wahrheit = Wunderground (`wu_settle_k`), METAR als Fallback.
Resolution frühestens 15.07. (erster WU-Datenpunkt des Folgetags), Autopilot
claimt. Ergebnis + Claim-Status hier nachtragen.
