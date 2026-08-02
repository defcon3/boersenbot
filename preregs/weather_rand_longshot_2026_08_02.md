# Pre-Registrierung: Trägt der Rand? Longshot-Bias auf der Bucket-Leiter — 2026-08-02

**Status:** Vorregistrierung eines **Forward-Tests**. Das retrospektive Fenster
(12.07.–01.08.2026) ist für diese Frage **verbraucht** — die −2-Zahlen wurden am
02.08. vollständig gesehen und stehen unten offen. Auf diesen Daten wird kein
Gate mehr gerechnet. Der Test beginnt mit Zieltag **03.08.2026**.

Auswertung folgt in `weather_rand_longshot_eval.py`, geschrieben und committet
vor dem ersten Settlement des Forward-Fensters.

---

## Anlass

`weather_minus1_ursache_2026_08_02.md` hat die Ursachenfrage beantwortet — die
−1-Klasse ist fair bepreist — und dabei einen Nebenbefund erzeugt, der nicht
zur Frage gehörte: die **−2-Klasse** liefert auf demselben Fenster **+2,00 %**
ROI nach Gebühr, bei 5,7 % Trefferquote gegen einen positionsweisen Break-even
von 7,3 %.

Das Gate G5 dort gilt als **gerissen**, und das bleibt so. Diese Pre-Reg holt den
Befund nicht zurück, sondern stellt ihn zum ersten Mal auf einen tragfähigen
Test.

## Was beim letzten Mal an G5 falsch war — beide Mängel

**Erstens, der offensichtliche:** Die Schwelle „P_ist(−2) < 3,7 %" wurde aus dem
**Median**preis 0,960 abgeleitet. Der tatsächlich gehandelte **Mittel**preis liegt
bei 0,922, dort steht der Break-even bei 7,3 %. Die Preisverteilung der
−2-Klasse ist linksschief — ein Median beschreibt sie nicht. **Lehre: Schwellen
nie aus einer Lagekennzahl ziehen, wenn die zugrunde liegende Verteilung schief
ist; den Break-even positionsweise aus dem echten Preis rechnen.**

**Zweitens, der schwerere:** G5 hatte **keine t-Bedingung**. G1 und G2 verlangten
t > 2,0, G5 verlangte nur „ROI > 0 und beide Hälften gleich". Bei diesem
Auszahlungsprofil — 94 % kleine Gewinne von +0,39 $ gegen 6 % Verluste von
−5,03 $ — beträgt die Streuung je Position rund 1,26 $ bei einem Erwartungswert
von 0,08 $. Über 316 Positionen ergibt das **t ≈ 1,1**. Der Befund wäre also auch
mit korrekter Schwelle nicht belastbar gewesen; er hätte nur bequemer ausgesehen.

Der zweite Mangel ist der eigentliche, und er bestimmt den Aufbau unten: **bei
dieser Ökonomie ist die Teststärke das Problem, nicht die Schwelle.**

## Was hier NICHT behauptet wird

Nicht: „die −2-Klasse verdient." Genau das ist die offene Frage.

Nicht: „der Longshot-Bias ist neu." Er ist auf Wettmärkten seit Jahrzehnten
beschrieben. Neu wäre allenfalls, dass er auf diesen Wetterbrettern in einer
Größe existiert, die nach Gebühr etwas übrig lässt.

Nicht: „daraus wird ein Buch." Selbst ein vollständig bestandener Test führt
nicht zu einem handelbaren Buch — siehe die Kapitalrechnung unter G5. Er führt
zu einer Auswahlfrage, und die wäre erneut zu registrieren.

Sondern: **Trifft der Rand der Leiter seltener, als er bepreist wird — und
reicht die Differenz nach Gebühr?**

---

## Was schon gesehen wurde — vollständige Offenlegung

Alles aus dem Lauf vom 02.08. (`weather_minus1_ursache_eval.py`, Lead 1, 325
Stadt-Tage, 21 Zieltage 12.07.–01.08., 30 Städte):

| Offset | P_ist | P_modell | P_markt | P_ist − P_markt |
|---|---|---|---|---|
| −3 | 1,8 % | 2,6 % | 2,5 % | **−0,7 pp** |
| −2 | 5,8 % | 9,3 % | 8,6 % | **−2,8 pp** |
| −1 | 23,7 % | 22,5 % | 25,1 % | −1,4 pp |
| 0 | 32,6 % | 30,5 % | 32,3 % | +0,3 pp |
| +1 | 24,6 % | 22,0 % | 22,5 % | +2,1 pp |
| +2 | 7,1 % | 8,9 % | 9,4 % | **−2,3 pp** |
| +3 | 2,8 % | 2,5 % | 2,6 % | **+0,2 pp** |

Die −2-Klasse im Detail: 316 Lays, Ø NO 0,922, Treffer 5,7 %, Break-even 7,3 %,
**ROI +2,00 %** (Gebühr 0,07) bzw. +2,29 % (0,04); Hälften +1,30 % / +2,80 %;
ohne Shenzhen +2,16 %. Die 74 Kandidaten im Preisband 0,70–0,90 verlieren
(17,6 % Treffer, −0,93 %); für die restlichen 242 bleiben rechnerisch rund
+2,9 % bei etwa 1 % Trefferquote gegen 4 % bepreiste.

Mengen je Zieltag (Lead 1, gesettelt, über 21 Tage): −3: 14,5 · −2: 15,1 ·
+2: 14,1 · +3: 12,2 — zusammen rund **56 Randpositionen je Zieltag**.

**Diese Zahlen tragen kein Gate.** Sie sind der Anlass und die Grundlage der
Teststärke-Rechnung, nichts sonst.

---

## Die Teststärke-Rechnung, die den Aufbau bestimmt

Der Test ist ein Vergleich der realisierten Trefferquote gegen die bepreiste. Bei
einer bepreisten Quote von 7,3 % und einer realisierten von 5,7 % — also der
beobachteten Differenz — braucht man für **z = 2** rund

    n = ( 2 · sqrt(0,073 · 0,927) / 0,016 )² ≈ 1.050 Positionen

**Für die −2-Klasse allein** sind das bei 15,1 Positionen je Zieltag rund **70
Zieltage**, also Auswertung nicht vor Mitte Oktober — und das nur, wenn Städte
eines Tages unabhängig wären, was sie nicht sind.

**Über alle vier Randklassen** (|offset| ≥ 2, rund 56 Positionen je Zieltag) sind
dieselben 1.050 Positionen nach **19 Zieltagen** erreicht. Weil das t über
Tagesmittel gerechnet wird und Städte desselben Zieltags an derselben Großlage
hängen, wird auf **30 Zieltage** aufgeschlagen.

**Daraus folgt die Testarchitektur, und zwar aus der Statistik, nicht aus
Bequemlichkeit: Haupttest ist der Mechanismus über den ganzen Rand (H1). Die
−2-Klasse allein (H3) ist der nachgeordnete Spezialfall mit schwächerer Schwelle
und der ausdrücklichen Erwartung, nach 30 Zieltagen noch nicht entscheidbar zu
sein.**

---

## Universum, Daten, Fenster

- **Fenster:** Zieltage **ab 03.08.2026**, `var='max'`, `kind='eq'`, Lead 1,
  neuester Snapshot je Stadt-Tag. Der Lead-1-Snapshot für den 03.08. ist beim
  Schreiben bereits geschrieben — **das Settlement ist es nicht**, und nur darauf
  kommt es an. Kein Kandidat wird nachträglich ausgeschlossen.
- **Randklassen:** `offset_fav` ∈ {−3, −2, +2, +3}. Die Grenze bei |offset| ≥ 2
  ist **vorab** gesetzt und wird nicht verschoben: |offset| ≤ 1 ist die Zone, in
  der die Leiter nachweislich fair ist (Befund vom 02.08.), |offset| ≥ 4 ist mit
  NO ≈ 0,998 ökonomisch leer.
- **Wahrheit:** `settle_k` (METAR bzw. HKO), wie im Lauf vom 02.08. — dort stimmte
  `settle_result` in 2.925 von 2.925 Fällen damit überein.
- **Preis:** `buy_no` des Lead-1-Snapshots. **Break-even positionsweise** aus
  genau diesem Preis, nie aus einem Median oder Mittelwert.
- **Rechnung:** 5 $ je Lay, Gebühr `0,07 · n · min(NO, 1−NO)`; 0,04 als
  Sensitivität ausgewiesen, entscheidend ist 0,07.
- **Signifikanz:** t über **Tages**-Mittel. Bei 56 Positionen je Tag ist das der
  Unterschied zwischen einem belastbaren und einem dreifach zu großen t.
- **Hälften:** Das Forward-Fenster wird zur Robustheitsprobe hälftig geteilt,
  Schnitt nach der Hälfte der Zieltage — nicht nach Kalenderdatum.

---

## Hypothesen

**H1 (Mechanismus, Haupttest):** Über alle Randklassen zusammen liegt die
realisierte Trefferquote **unter** der bepreisten: P_ist < P_markt. Das ist die
Longshot-Aussage — der Markt bezahlt den unwahrscheinlichen Ausgang zu teuer,
also verdient dessen Lay.

**H2 (Geld):** Der Mechanismus überlebt die Gebühr: ROI der Rand-Lays > 0 bei
Gebühr 0,07.

**H3 (−2-Klasse, nachgeordnet):** Der Spezialfall, der den Anlass gab. Schwächere
Schwelle wegen kleinerer Stichprobe; **erwartet wird, dass er nach 30 Zieltagen
nicht entscheidbar ist.**

**H4 (Preisrichtung, diagnostisch):** Innerhalb der Randklassen steigt der Ertrag
mit dem NO-Preis — die teuren tragen, die billigen nicht. Das ist die
**Umkehrung** der −1-Doktrin, wo der Ertrag im Band 0,70–0,90 liegt. Genau
**ein** Schnitt (Median-NO der Randmenge), keine Schwellensuche.

**H5 (Modell-Gegenprobe, keine Handelsthese):** Unser Ensemble schätzt die Ränder
zu breit (P_modell > P_ist). Falls belegt, betrifft das den P_pess-Filter des
Autobuy — der rechnet dann mit zu viel Randmasse und verwirft Kandidaten ohne
Grund. Konsequenz wäre eine eigene Pre-Reg, keine Änderung aus dieser hier.

**H0:** Der Rand ist so fair bepreist wie die Mitte. Die 316 Lays mit t ≈ 1,1
waren ein Sommerfenster, und die drei negativen Vorzeichen in der Tabelle oben
sind drei Münzwürfe.

---

## Gates

| Gate | Bedingung |
|---|---|
| **G0** Laufzeit und Basis | ≥ **30 Zieltage** ab 03.08., ≥ 1.400 Randpositionen mit Settlement, ≥ 20 Städte. Vorher wird nicht ausgewertet |
| **G1** Mechanismus | Über alle Randklassen: P_markt − P_ist ≥ **1,5 pp** bei **t > 2,0** über Tagesmittel |
| **G2** Geld | ROI der Rand-Lays **> 0** bei Gebühr 0,07, **t > 2,0** über Tagesmittel |
| **G3** Robustheit | Beide Hälften gleiches Vorzeichen; ohne den besten Zieltag **und** ohne die stärkste Stadt bleibt der ROI > 0; kein Zieltag trägt > 35 % des Effekts |
| **G4** −2-Klasse | Nachgeordnet: ROI > 0 bei **t > 1,5**. Reißt nur G4, gilt der Mechanismus als belegt und die −2-Klasse als weiterhin offen |
| **G5** Ausführung und Kapital | Bei NO ≥ 0,93 wird die Buchtiefe über die Polymarket-Public-Data-API gegen die unterstellten 5 $ geprüft. Die Kapitalbindung wird beziffert, **ohne** dass ein negatives Ergebnis den Befund umstößt — G5 entscheidet über Handelbarkeit, nicht über Wahrheit |

**Bonferroni:** Fünf Hypothesen, davon drei mit Gate-Charakter. Die t-Schwellen
stehen deshalb auf 2,0 statt 1,5. Es wird **eine** Randmenge geprüft (|offset| ≥ 2),
**ein** Preisschnitt (Median) und **keine** Suche nach Städten, Monaten oder
Klassen, in denen es besser aussieht.

**Sequenzregeln — vorab, gegen Peeking:**

1. **Sicherheits-Zwischenschau nach 15 Zieltagen** (~18.08.): Sie darf **nur nach
   unten** entscheiden. Liegt der ROI dann unter −5 %, wird abgebrochen. Ein gutes
   Zwischenergebnis führt zu **nichts** — insbesondere nicht zu einem früheren
   Live-Gang.
2. **Verlängerung statt Entscheidung:** Liegt das t nach 30 Zieltagen zwischen
   1,0 und 2,0, wird **einmalig** auf 60 Zieltage verlängert und danach endgültig
   entschieden. Diese Verlängerung ist hier festgelegt, damit sie später nicht
   als Rettung erfunden werden muss.
3. Fällt das t unter 1,0, ist der Test beendet und die These verworfen.

---

## Designfallen, die diesen Test definieren

**1. Das Auszahlungsprofil täuscht über die Varianz.** 94 % Gewinner fühlen sich
nach einer sicheren Sache an; ein einzelner Verlusttag kostet das Ergebnis von
dreizehn guten. Deshalb ist die Zwischenschau asymmetrisch und deshalb steht die
Teststärke-Rechnung vor den Gates statt in einer Fußnote.

**2. Die Randklassen sind nicht unabhängig voneinander.** Trifft an einem Tag der
+2-Bucket, kann −2 nicht getroffen haben. Das t über Tagesmittel fängt das auf;
positionsweise gerechnet wäre es grob überhöht.

**3. Der Preis entscheidet, welche Klasse man misst.** Ein −2-Bucket zu NO 0,80
ist kein billiger −2-Bucket, sondern ein Fall, in dem der Markt die Verteilung
für breit hält — im Fenster vom Juli trafen genau diese zu 17,6 % statt 5,7 %.
Die Randmenge wird deshalb **ohne** Preisfilter gebildet; H4 prüft die
Preisrichtung getrennt und diagnostisch.

**4. Buchtiefe bei NO 0,96 ist nicht Slippage bei 250 $.** Die
Slippage-Messung vom 25.07. betraf die Größe der Order. Hier ist die Frage eine
andere: ob auf der Gegenseite eines 0,96-Buckets überhaupt genug liegt, um 5 $ zu
füllen — und ob der `markPrice` das hergibt, den der Screen nicht zeigt
([[weather-screen-price-vs-book]]). G5 misst das an echten Büchern, nicht am
Snapshot.

**5. Ein Papier-Schattenbuch überschätzt sich selbst.** Es füllt immer, zum
Snapshot-Preis, ohne Wartezeit. Der gemessene ROI ist damit eine Obergrenze. Bei
einem erwarteten Effekt von 2 % ist das kein Detail — schon 0,005 Differenz im
Fill-Preis bei NO 0,96 frisst mehr als die Hälfte.

**6. Shenzhen und Hong Kong bleiben Sonderfälle.** Shenzhen läuft seit dem 02.08.
METAR-kalibriert, Hong Kong über HKO mit floor-Buckets. Beide bleiben drin, aber
der Lauf ohne sie wird als Kontrolle ausgewiesen — vorab festgelegt, wie beim
letzten Mal.

---

## Vorab-Erwartung (damit sie nicht zurechtgebogen wird)

**Ich erwarte, dass die Richtung hält und die Signifikanz knapp wird.** Der
Longshot-Bias ist ein robustes Marktphänomen, und drei der vier Randklassen
zeigten im Juli in dieselbe Richtung. Aber der Effekt ist mit 1,5–2,8 pp klein,
und die einzige bisherige Messung stand bei t ≈ 1,1. Mein Tipp: nach 30 Zieltagen
ein t zwischen 1,2 und 2,0 — also **die vorab festgelegte Verlängerung auf 60
Zieltage**, und die Entscheidung fällt im Oktober.

**H3 (−2 allein) erwarte ich als nicht entscheidbar**, und zwar aus der
Teststärke-Rechnung heraus, nicht als Ausrede: 30 Zieltage liefern rund 450
Positionen, nötig wären etwa 1.050.

**H5 halte ich für die wahrscheinlichste Bestätigung** — P_modell lag im Juli an
beiden Rändern über P_ist (9,3 gegen 5,8 und 8,9 gegen 7,1). Das wäre der
praktisch nützlichste Nebenbefund, weil er den P_pess-Filter betrifft, der heute
Kandidaten mit zu viel unterstellter Randmasse verwirft.

**Zur Größenordnung, damit niemand zu viel erwartet:** 56 Positionen je Zieltag zu
5 $ sind 280 $ Umsatz täglich; bei 2 % ROI wären das **5,60 $ am Tag** — gegen
eine Kapitalbindung von rund 560 $ bei zwei überlappenden Tagen, während zuletzt
etwa 93 $ frei waren. **Der Test kann also gelingen, ohne dass sich daraus ein
Buch bauen lässt.** Was dann bliebe, wäre eine Auswahlfrage — und die wäre neu zu
registrieren, mit demselben Ergebnis-Risiko wie beim Preisband.

## Abbruchregel

Reißt **G1**, ist der Longshot-Mechanismus auf diesen Brettern nicht belegt. Dann
wird **nicht** nach einer Randklasse gesucht, die es doch trägt, **nicht** die
Grenze von |offset| ≥ 2 auf ≥ 3 verschoben und **nicht** ein Preisband gesucht, in
dem es funktioniert. Der Befund vom 02.08. gilt dann endgültig als das, was er
ist: ein Nebenprodukt mit t ≈ 1,1.

Besteht G1, reißt aber **G2** (Mechanismus da, Gebühr frisst ihn), lautet der
Befund „der Rand ist leicht fehlbepreist, aber nicht handelbar". Auch das ist ein
Ergebnis und wird so committet.

Bestehen G1 und G2, geht **nichts live**. Es folgt ein zweites Fenster als
Schattenbuch mit echten Fill-Preisen aus der Polymarket-API statt aus dem
Snapshot — Designfalle 5 ist bei einem 2-%-Effekt nicht mit einem Papierlauf
abzuräumen.

Reißt **G5** (nicht finanzierbar oder nicht füllbar), bleibt der Befund als
Wissen bestehen und die Umsetzung unterbleibt. Ein wahres Ergebnis wird nicht
deshalb verworfen, weil das Konto zu klein ist — und ein zu kleines Konto wird
nicht deshalb überzogen, weil das Ergebnis wahr ist.

Unberührt von dieser Pre-Reg laufen weiter: **Fenster D des Preisband-Tests**
(Spannen-Veto, ab 03.08., ≥ 20 Zieltage) und der **Ensemble-µ-Forward-Test**
(Auswertung frühestens Oktober). Beide werden nicht angefasst.
