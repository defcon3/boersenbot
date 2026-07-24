# Befund + Pre-Reg-Kandidat: Wächter für die −1-Lay-Klasse — 2026-07-24

**Status:** Explorativer Befund, IN-SAMPLE. Auswertung `weather_lay_guardrail_eval.py`.
**Kein Deploy-Vorschlag** — Vorlage für den Autobuy-Review am 27.07.

## Anlass

Der Nutzer kann nur am Wochenende eingreifen; unter der Woche läuft der Autobuy
(`weather_minus1_autobuy.py`) unbeaufsichtigt und hält jede Position bis zum
Settlement. Am 24.07. rettete ein manueller Eingriff eine kippende Position
(−3,11 $ statt −14,24 $). Frage: Lässt sich dieser Eingriff mechanisieren?

Der Bot layt den Bucket **eine Klasse unter dem Favoriten**. Verloren wird genau
dann, wenn das Tageshoch **exakt auf diesem Bucket stehenbleibt**. Da das laufende
Maximum monoton steigt, läuft es auf die Gefahr zu und muss darüber hinaus.

**Signal:** gerundetes laufendes Tagesmaximum == gelayter Bucket, zur Ortsstunde T.

## Daten

`bb_WeatherLadders` (var='max', kind='eq', offset_fav=−1, **Lead 1**, buy_no
0,50–0,97, abgerechnet), Zieltage 11.–23.07.2026 = **13 Tage**.
Intraday-METAR über die WU-Historie; Ausstiegspreise aus Polymarkets
`prices-history` (NO-Token, 10-Min-Auflösung, s. `POLYMARKET_DATA_API.md`).
Verkaufsgebühr 3,6 % von min(p, 1−p) abgezogen.

## Ergebnis 1 — das Signal trennt scharf

Lead 1, alle Kandidaten (144 auswertbar, 21,5 % Verlierer):

| Ortszeit | Signal | davon Verl. | Treffer | ohne Signal | davon Verl. |
|---|---|---|---|---|---|
| 13:20 | 33 | 9 | 27 % | 111 | 19 % |
| 15:20 | 43 | 22 | 51 % | 101 | 8 % |
| 16:20 | 42 | 28 | **67 %** | 102 | 2 % |
| 17:20 | 33 | 29 | **88 %** | 111 | 1 % |
| 18:20 | 31 | 29 | 94 % | 113 | 1 % |

## Ergebnis 2 — der Markt preist das Signal nicht ein

Ausstiegspreis gegen tatsächliche Gewinnquote auf Signal-Tagen (Prüfstunde 16:20,
Menge buy_no 0,50–0,97 über alle Leads, n = 92 Signale / 13 Tage):

| Ortszeit | Ø Ausstiegspreis | echte Gewinnquote | Differenz |
|---|---|---|---|
| 13:20 | 0,734 | 0,727 | **+0,006** |
| 15:20 | 0,654 | 0,436 | +0,218 |
| 16:20 | 0,545 | 0,293 | **+0,252** |
| 17:20 | 0,384 | 0,146 | +0,238 |
| 18:20 | 0,269 | 0,066 | +0,204 |

Früh am Tag ist der Markt exakt kalibriert; ab 15:20 überzahlt er das NO
konstant um 20–25 ct. **Nach Tages-Clustering: 13 von 13 Tagen positiv,
Mittel +0,294, Streuung 0,194, t = 5,46**; ohne die zwei besten Tage +0,234.

Ausgeschlossene Artefakte: Preise sind frisch (Medianalter 10 min, keiner > 30 min);
der effektive Spread liegt bei **1 ct** (s. `weather_market_making_2026_07_24.md`)
und kann eine 25-ct-Lücke um zwei Größenordnungen nicht erklären.

## Ergebnis 3 — der Nutzen hängt an der Breite der Auswahl

P&L über 13 Tage, 5 $ je Position, Lead 1:

| Menge | Positionen | Verliererquote | Halten | Wächter 17:20 | Differenz |
|---|---|---|---|---|---|
| alle Kandidaten | 144 | 21,5 % | +30,60 $ | **+79,75 $** | **+49,16 $** |
| Live-Auswahl des Bots | 38 | **7,5 %** | +10,52 $ | +6,38 $ | **−4,14 $** |

**Der Wächter schadet dem heutigen Bot und nützt einem breiteren.** Die
Live-Auswahl nimmt die konservativsten Buckets und verliert nur in 7,5 % der
Fälle — bei 92,5 % Trefferquote kappt jeder Wächter mehr Gewinner, als er
Verlierer rettet (über 13 Tage gab es dort ganze **vier** Verlierer). Auf der
breiten Menge kippt das Verhältnis.

## Deutung

Der Wächter ist keine Verbesserung des bestehenden engen Buchs, sondern die
**Voraussetzung für Breite**: Er erlaubt, weniger konservative Buckets
mitzunehmen, ohne dass der Verlustschwanz mitwächst. Das ist genau das, was der
beobachteten Groß-Wallet fehlt ([[weather-whale-wallet-shallow-diadem]]:
446 Käufe : 7 Verkäufe, −103.560 $ in vier Tagen).

## Vorbehalte

1. **13 Tage, alles Hochsommer, alles in-sample.** Keine OOS-Bestätigung.
2. Die Prüfstunde wurde nach Sichtung aller Stunden gewählt. Der Effekt ist
   allerdings über 15:20–18:20 durchgehend vorhanden, also kein Knife-Edge.
3. Ausstiegspreise stammen aus der Last-Trade-Reihe, nicht aus dem ausführbaren
   Geld. Bei 1 ct Spread ist der Unterschied klein, aber nicht null.
4. Der Verkauf selbst ist über Jupiter nicht getestet — Ausführungszeit lag am
   24.07. bei 4–45 s.

## Vorschlag für den 27.07.

Nicht „lief gut, also mehr Einsatz", sondern **Einsatz-Erhöhung gekoppelt an die
Breite und den Wächter**: Erst vorregistrieren (Prüfstunde, Schwelle, Menge),
dann forward testen, und die Auswahl erst dann weiten, wenn der Wächter live
bestätigt ist. Die Erhöhung des Einsatzes im engen Buch braucht ihn nicht — dort
ist er nachweislich schädlich.
