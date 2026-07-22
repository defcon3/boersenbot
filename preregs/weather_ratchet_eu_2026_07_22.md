# Nowcast-Ratchet — EU-Städte-Test (2026-07-22) — ERGEBNIS: FAIL

Fortsetzung von `weather_chengdu_ratchet_2026_07_22.md`. Gleicher Test
(tote Masse auf Buckets < laufendem Max + Ratchet-Event-Study + Antizipation)
auf vier europäische Städte. Daten: `bb_WeatherLatency`, Juli 2026.
Skript: `weather_ratchet_test.py`.

| Stadt | Station | Tage | tote Masse mean/max | Ratchet-Schritte >0,02 | Fav-Vorlauf (Vorm.) | Overshoot |
|---|---|---|---|---|---|---|
| München | EDDM | 16 | 0,003 / 0,052 | 0/149 | +6,8 | 8/16 (Ø +0,1) |
| Paris | LFPB | 10 | 0,002 / 0,014 | 0/76 | +7,3 | 0/9 |
| Madrid | LEMD | 12 | 0,002 / 0,399 | 0/108 | +7,8 | 0/12 |
| London | EGLC | 10 | 0,002 / 0,015 | 0/59 | +6,7 | 2/10 |

**Alle vier FAIL** — identisch zu Chengdu/Seoul. Markt antizipatorisch (Favorit
+6,7 bis +7,8 Buckets über dem Vormittags-Max), unmögliche Buckets sofort auf ~0.
München am lebhaftesten (8/16 Overshoot-Tage, konvektiv) — trotzdem keine tote
Masse zum Abgreifen.

**Einzige echte Lag-Instanz:** Madrid 03.07. 14:02 — obs_max=35, Bucket 34 handelt
noch 0,397 (bereits unmöglich), fällt binnen ~2–4 min auf ~0. 1 Ereignis über
~500 Ratchet-Schritte in 6 Städten; Fenster zu kurz und zu selten für 5-$-Handel
mit 2×Fee.

## Gesamt-Fazit Prio-1-Nowcast-Ratchet (6 Städte)
Der mechanische Running-Max-Ratchet-Edge (Quelle A) existiert in diesen dünnen
Jupiter-Wetter-Märkten **nicht**: der Markt ankert am Forecast-Peak und trackt
live das Stationshoch, unmögliche Buckets werden praktisch verzögerungsfrei
genullt. Konsistent mit dem Projekt-Gesamtbild „Markt scharf". Kapitel für diese
Städte geschlossen; kein Echtgeld. Caveat: kleines N (Logger seit 01.07.);
re-testbar mit mehr Historie. Advektions-Prognose (Quelle B) ist separat und hier
nicht getestet. Siehe [[weather-intraday-nowcast-scalp]], [[weather-seoul-seabreeze]].
