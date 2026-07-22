# Chengdu Nowcast-Ratchet — Markt-Test (2026-07-22) — ERGEBNIS: FAIL

**Kontext:** Intraday-Nowcast-Scalp (Prio 1). Kern-These: Während der Tag
heizt, ratcht das laufende Stationshoch die Buckets hoch — Buckets *unter* dem
Max werden garantiert unmöglich, obere werden wahrscheinlicher. Wenn der dünne
Markt nachhinkt, ist der Reprice-Lag ein Scalp-Edge.

**Daten:** `bb_WeatherLatency` (Centron), Chengdu (Settlement ZUUU), 11 Markttage
Juli 2026, 2-min-Snapshots inkl. `all_prices` (Bucket→Preis). Skript:
`weather_chengdu_ratchet.py`.

## Test 1 — Tote Masse auf unmöglichen Buckets (Buckets < laufendes Max)
Garantierte Verlierer, jeder Restpreis = Lay-Gratisgeld.

| Metrik | Wert |
|---|---|
| mittlere tote Masse (alle Snapshots) | 0,002–0,005 |
| max. tote Masse (irgendein Tag) | ≤ 0,010 |

Der Markt hält auf unmöglichen Buckets faktisch **null**.

## Test 2 — Event-Study, 63 Ratchet-Schritte (rmax steigt um 1)
| Zeitpunkt | tote Masse |
|---|---|
| beim Schritt (t=0) | 0,0024 |
| +10 min | 0,0024 |
| +20 min | 0,0023 |

**0 von 63** Schritten mit toter Masse > 0,02. **Kein Reprice-Lag** — es gibt
schlicht nichts nachzupreisen.

## Test 3 — Warum: Markt ist antizipatorisch
Vormittags (6–11 h lokal) liegt der Markt-Favorit im Schnitt **+6,6 Buckets
über** dem bisher gemessenen Max. Der Markt ankert am Forecast-Peak; untere
Buckets sind tot, *bevor* die Station sie erreicht.

## Test 4 — Overshoot / Aufhol-Fälle (der letzte Faden)
Vormittags-Favorit vs. realisierter Endbucket: Overshoot nur an **2/11 Tagen**
(2.07. +3, 7.07. +1), Mittel −0,4 Buckets. Der Vormittags-Favorit ist ein gut
kalibrierter, minimal zu hoher Forecast-Anker. Selbst am +3-Tag (2.07.) kein
Reprice-Lag an den Ratchet-Schritten (siehe Test 2).

## Fazit
**FAIL.** Chengdu-Markt ist antizipatorisch und effizient: unmögliche Buckets
sofort genullt, Peak vorab gepreist, Overshoots selten und klein. Nowcast-
Ratchet hat keinen Edge. Konsistent mit Seoul
([[weather-seoul-seabreeze]]) und dem Projekt-Gesamtbild „Markt scharf".
Caveat: kleines N (Logger seit 01.07.); re-testbar bei mehr Historie.
