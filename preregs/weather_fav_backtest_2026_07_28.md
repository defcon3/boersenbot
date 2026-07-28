# Befund 28.07.2026 — „Immer den eigenen Favoriten backen"

**Status:** nachträgliche Messung, keine Pre-Registrierung. Die Frage kam am
28.07. auf („was wäre passiert, wenn wir immer den Favoriten mit 5 $ gebackt
hätten"); die Daten dafür lagen durch den Ladder-Logger bereits vor.

**Skript:** `weather_fav_backtest.py` (reproduziert alle Zahlen unten mit
`python weather_fav_backtest.py`)

## Aufbau

| | |
|---|---|
| Datenbasis | `bb_WeatherLadders`, Zieltage 12.–27.07.2026, 38 Städte |
| Snapshot | Lead 1 = Vortag ~12:30 UTC (der Kaufzeitpunkt des −1-Autobuy) |
| Regel | je Stadt-Tag den Bucket mit `offset_fav == 0` YES zum Ask kaufen, 5 $ |
| Settlement | `wu_settle_k` (Wunderground = die Polymarket-Settlement-Quelle) |
| Fee | `rate * Stück * min(p, 1−p)`, ausgewiesen für 0 / 0,04 / 0,07 |

Lead 0 ist bewusst ausgeschlossen: dort läuft der Zieltag in vielen Zeitzonen
bereits, die Preise stehen teils auf 0,001/1,0.

**Fee-Nachmessung:** Das 0,07-Modell aus `autopilot.py` liegt gegen fünf echte
Positionen konsistent zu hoch (ist/modell = 0,51 … 0,69, effektive Rate ≈ 0,04).
Der Report weist deshalb alle drei Varianten aus.

## Ergebnis

| Strategie | N | Treffer | Ø-Preis | ROI (Fee 0,04) |
|---|---|---|---|---|
| **eigener Favorit YES** | 274 | 33,2 % | 0,338 | **−11,5 %** (−157 $ auf 1.370 $) |
| Markt-Favorit YES | 293 | 45,7 % | 0,444 | −4,8 % |

Fee-Sensitivität eigener Favorit: brutto −7,6 % · 0,04 → −11,5 % · 0,07 → −14,4 %.
t = −1,30 (Fee 0,04), also nicht signifikant negativ — aber in keiner Variante
positiv.

**Break-even:** Ø gezahlter Preis 0,338 gegen 33,2 % tatsächliche Trefferquote.
Der Markt bepreist unseren Favoriten im Mittel fair; brutto bleibt eine kleine
negative Differenz, die Gebühr macht daraus den vollen Verlust.

**Robustheit:** Lead 2 ergibt −16,1 % (N=214) bei gleichem Muster. Die Tageskurve
läuft stetig abwärts (10 von 16 Zieltagen negativ), kein Einzelereignis.
Aufgeteilt: max −11,0 % (N=244), min −42,0 % (N=30, kleines N).

## Paariger Vergleich eigener vs. Markt-Favorit (N=274)

| | Trefferquote |
|---|---|
| eigener Ensemble-Favorit | 33,2 % |
| Markt-Favorit | 47,4 % |

- gleicher Bucket in nur 118 Fällen (43,1 %)
- nur eigener trifft: 34 · nur Markt trifft: 73 → McNemar χ² = 13,50, **p < 0,01**
- **nur bei Uneinigkeit (N=156): eigener 21,8 %, Markt 46,8 %**

Als Punktprognose ist das Fünf-Modell-Ensemble dem Markt unterlegen, und der
Abstand ist zu groß für eine Rauschfrage.

## Abgrenzungen

1. **Das ist kein Test der Lay-Doktrin.** Lays setzen darauf, welcher Grad
   sicher *nicht* fällt, nicht darauf, welcher Grad genau fällt. Der Favoriten-
   Test misst ausschließlich Letzteres.
2. **Zeile C im Report ist nicht das Live-Buch.** Sie kauft jeden −1-Lay im
   Preisband 0,70–0,90 ohne die Gates des Autobuy (Doppel-Kalibrierung,
   Spannen-Veto, Mindestabstand, P_pess) und kommt auf −3,1 %; das Live-Buch
   machte im selben Zeitraum +8,33 % (25 Lays, Review 27.07.). Die Differenz ist
   der Beitrag der Gates, nicht ein Ergebnis dieses Tests.
3. **`mu_ens` stammt aus der 700d-Kalibrierung**, die der Ladder-Logger schreibt.
   Nach dem Befund vom 28.07. liefert das 40d-Sommerfenster die bessere
   Favorit-Prognose. Der Test misst also die schlechtere der beiden Sichten; die
   25-pp-Lücke bei Uneinigkeit erklärt das nicht.

## Konsequenz

Keine Änderung am laufenden Buch. Der Befund schließt eine Strategieklasse aus:
alles, was unsere Punktprognose als Punktprognose verwendet (Favorit backen,
Abweichung des Markt-Favoriten als Fehlbepreisung lesen).
