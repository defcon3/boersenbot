# Befund: Market Making auf den Wetter-Märkten — 2026-07-24

**Status:** Nachträglich dokumentierter Befund (KEINE echte Vorregistrierung — die
These entstand und starb am selben Tag). Auswertung: `weather_mm_spread_test.py`.
Ergebnis: **weitgehend FAIL, ein schmales Restband überlebt.**

## Anlass

Beim manuellen Kauf einer Helsinki-Position am 24.07. zeigte Jupiter einen
**Spread von 59 ct** an. Naheliegende Idee: statt nur zu layen selbst Liquidität
stellen und diesen Spread einnehmen.

## Hypothese

**H1:** Auf den täglichen Temperatur-Märkten ist beidseitiges Quoten netto
profitabel, d. h.

    halber effektiver Spread  >  Gebühr je Kontrakt

**Gate G-A (Kosten):** In mindestens einer Preislage muss der halbe Spread die
Gebühr übersteigen.
**Gate G-B (Fluss):** In derselben Preislage muss der Fluss zweiseitig sein
(Taker-Kaufanteil 40–60 %). Ein Market Maker, der 3 von 4 Fills auf derselben
Seite bekommt, stellt keinen Markt, sondern sammelt Bestand an.

## Daten

Polymarkets öffentlicher Trade-Tape (kein Auth, kostenlos, historisch —
s. `POLYMARKET_DATA_API.md`). Jupiters Märkte *sind* Polymarket-Märkte, die IDs
sind durchgereicht (`POLY-3025209` → `3025209`).

Helsinki, alle 11 Buckets, 18.–24.07.2026: **13.611 Trades, 75.189 $ Notional.**

Effektiver Spread aus den Handelsseiten rekonstruiert (Normalisierung auf
Yes-Äquivalent, dann je 10-Min-Fenster `mean(Taker-Kauf) − mean(Taker-Verkauf)`).
Gebühr empirisch aus eigenen Fills: **3,6–4,5 % von min(p, 1−p)**, konservativ
mit 3,6 % gerechnet.

## Ergebnis

Fluss stabil und unauffällig: Trade alle 21–51 s, Ø Größe 4–8 $, rund 10.700 $
Notional pro Tag. Der 24.07. war trotz des Chaos ein völlig normaler Volumentag.

**Effektiver Spread über 1.132 Fenster: Median 1,0 ct** (25 % 0,1 | 75 % 2,2 |
90 % 5,0). Robust über Fensterlängen von 60 s bis 600 s — also keine Preisdrift-Artefakt.

| Preislage | n | Spread | halber | Gebühr | netto | Kauf-Anteil | |
|---|---|---|---|---|---|---|---|
| Mitte 0,40–0,60 | 163 | 1,5 ct | 0,8 | 1,80 | **−1,05** | 59 % | zweiseitig |
| 0,25–0,40 | 253 | 1,2 ct | 0,6 | 1,17 | **−0,56** | 49 % | zweiseitig |
| **0,10–0,25** | 264 | 1,8 ct | 0,9 | 0,63 | **+0,27** | 46 % | **überlebt** |
| Rand 0,03–0,10 | 213 | 1,5 ct | 0,7 | 0,23 | +0,52 | **27 %** | einseitig |
| Tail < 0,03 | 239 | 0,3 ct | 0,2 | 0,05 | +0,10 | **25 %** | einseitig |

**Zwei Wege sind zu:**

1. **Preismitte — G-A gerissen.** Die Gebühr skaliert mit min(p, 1−p) und ist
   dort mit 1,2–1,8 ct größer als der halbe Spread (0,6–0,8 ct). Kosten fressen
   die Einnahme um Faktor 2–3.
2. **Rand und Tail — G-B gerissen.** Dort trägt der Spread die Gebühr zwar
   (netto +0,52 bzw. +0,10 ct), aber der Fluss ist mit 73–75 % Verkäufen massiv
   einseitig. Man kann nicht round-trippen, sondern kauft nur den Bestand auf,
   den andere gerade als tot abstoßen — und die stoßen ihn ab, *weil* der Bucket
   unmöglich geworden ist. Das ist Adverse Selection in Reinform, keine Marge.

**Restband 0,10–0,25:** besteht beide Gates, netto **+0,27 ct** je Kontrakt bei
zweiseitigem Fluss. Ehrlich gesagt: dünn. Auf einen Kontrakt von ~0,175 sind das
rund 1,5 % je Round-Trip, und der Round-Trip braucht zwei Fills. Hochgerechnet
auf Helsinki ergäbe eine 20-%-Beteiligung am Fluss dieses Bandes grob 5 $/Tag.

## Was die Daten nicht beantworten

Die entscheidende offene Frage ist **nicht** die Marge, sondern die
Warteschlangen-Position: Ein Spread von 1,0 ct heißt, dass dort bereits jemand
professionell quotet. Um Fills zu bekommen, müsste man ihn an der Spitze des
Buchs schlagen — und bekäme dann vermutlich genau dann den Fill, wenn der
Incumbent zurückweicht, also wenn es gefährlich ist. Das ist aus dem Tape nicht
messbar; dafür bräuchte man Buch-Snapshots über die Zeit.

**Entscheidung:** nicht weiterverfolgen. Der Erwartungswert des Restbands
rechtfertigt weder den Dauerbetrieb noch das gebundene Kapital, und der
Wettbewerb an der Buchspitze ist ungeklärt.

## Nebenbefunde

- **Umkehrung einer Fehlannahme:** Das Kreuzen des Spreads kostet nur ~0,5 ct und
  ist damit *nicht* die teure Zeile bei der Ausführung. Die **Gebühr** ist es,
  2–4× größer. Sie ist am Rand fast null und in der Preismitte am teuersten —
  die bestehende Lay-Doktrin auf 0,95er-Buckets (~0,2 ct/Kontrakt) ist also
  bereits gebührenoptimal, mittelpreisige Buckets sind teuer.
- **Wer auf der Gegenseite Kasse macht (Helsinki 21 °C, 24.07.):** Bis 14:46 UTC
  handelte der Bucket bei 0,08–0,15. Die WU-Beobachtung mit 21 °C trägt
  `valid_time_gmt` = **14:50:00 UTC**. Um **14:50:10 UTC** — zehn Sekunden später
  — nahm jemand mit einem Schlag **429 Kontrakte zu 0,829** (das ~50–100-fache
  der normalen Handelsgröße von 4–8 $), zwei Sekunden darauf stand der Markt bei
  0,95. Das ist ein Neunfaches in Sekunden, aber es ist ein reines
  Geschwindigkeitsrennen gegen automatisierte METAR-Leitungen, **kein
  Analyse-Edge**. Nicht spielbar: eigene Fills über den Jupiter-Pfad brauchten am
  selben Tag 4, 4 und 45 s von der Order bis zur Ausführung — das gesamte
  Repricing war nach 2 s vorbei. Bestätigt damit unabhängig die frühere
  Falsifikation des Latenz-Fensters, diesmal auf sauberen Daten (echter Tape
  statt `all_prices`).
- **Zeitzonen-Falle:** Die WU-Website rendert die Stundentabelle in der Zeitzone
  des *Betrachters*. Die 14:50-UTC-Messung erscheint deutschen Nutzern als 16:50,
  in Helsinki als 17:50. Vergleiche zwischen Messzeit und Marktbewegung immer
  über `valid_time_gmt` führen — sonst erfindet man Latenzfenster, die es nicht gibt.
- **Methodenwarnung:** `bb_WeatherLatency.all_prices` ist für Mikrostruktur
  unbrauchbar. Die Bucket-Preise eines Zeitpunkts summieren sich auf Median 1,08
  und bis 1,54 statt auf 1,00 — es sind veraltete Einzelnotierungen aus
  verschiedenen Momenten. Eine erste Autokorrelations-Auswertung darauf zeigte
  scheinbare Mean Reversion (r = −0,11, t = −54), die reines Artefakt ist.
  Für Mikrostruktur ausschließlich den Polymarket-Tape verwenden.
