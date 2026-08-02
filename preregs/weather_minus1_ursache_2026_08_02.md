# Pre-Registrierung: Warum verliert die −1-Klasse — und trägt die −2-Klasse? — 2026-08-02

**Status:** Vorregistrierung. Geschrieben, **bevor** irgendeine Trefferquote,
irgendein PnL oder die Verteilung des realisierten Offsets gerechnet wurde. Die
Sondierung sah ausschließlich Mengen, Feldvollständigkeit und Preise — alles
davon steht unten unter „Was schon gesehen wurde".

Auswertung folgt in `weather_minus1_ursache_eval.py`.

---

## Anlass

Der Test vom selben Tag (`weather_band_breite_2026_08_02.md`) hat G1 gerissen und
dabei einen Befund freigelegt, der größer ist als die Frage, die er beantworten
sollte: **die −1-Klasse verliert als Ganzes**, in allen vier Teilfenstern,
zwischen −6,31 % und −15,81 %, über 525 entduplizierte Kandidaten. Der Autobuy
handelt also eine strukturell verlierende Grundgesamtheit und rettet sich über
einen Preisfilter, dessen Ertrag zwischen −4,99 % und +18,07 % schwankt.

Dort wurde ausdrücklich offengelassen, **warum**. Das ist die Frage hier.

Zweiter Anlass, vom Nutzer am 01.08.: sein handgesetzter Chengdu-**26**er gewann
(+1,27 $), während der **27**er des Bots verlor (−4,83 $). Ein tieferer Bucket ist
sicherer und zahlt weniger — welche Seite dieses Tauschs überwiegt, ist nie
gemessen worden. Das ist eine Einzelbeobachtung und trägt kein Argument; sie ist
der Anlass, nicht der Beleg.

## Was hier NICHT behauptet wird

Nicht: „`mu_ens` ist kaputt." Der Verdacht ist notiert
([[weather-minus1-klasse-verliert]]), aber die Preisleiter unten spricht schon
vorab dagegen. Diese Pre-Reg prüft ihn, sie setzt ihn nicht voraus.

Nicht: „die −2-Klasse ist die Rettung." Ein tieferer Bucket kostet mehr Prämie,
als er an Sicherheit bringt — das ist der Normalfall auf einer fair bepreisten
Leiter. Gefragt wird, ob dieser Normalfall hier zutrifft.

Nicht: „hier entsteht eine neue Handelsregel." Das ist eine **Diagnose**, kein
Strategietest. Aus keinem Ausgang folgt eine Änderung am laufenden Bot. Was
folgen darf, ist eine eigene Vorregistrierung mit Forward-Fenster.

Sondern: **Liegt der Verlust der −1-Klasse an unserem Anker, an unserer
Sicherheit, oder schlicht daran, dass eine faire Leiter nach Gebühr keinen Lay
trägt?** Diese drei Ursachen führen zu drei völlig verschiedenen Konsequenzen,
und ohne die Unterscheidung ist jede Reparatur geraten.

---

## Was schon gesehen wurde — vollständige Offenlegung

Beim Schreiben bekannt, aus `sondierung_mengen.py` (nur Mengen und Preise):

1. **Bestand:** 16.614 `kind='eq'`-Zeilen, Zieltage 10.07.–04.08.2026, 39 Städte.
   Für `var='max'`: 517 Stadt-Tage, davon **342 mit `mu_ens` und `settle_k`**
   (22 Zieltage, 30 Städte) — das ist die Stichprobe. `market_fav_k` steht bei
   allen 517.
2. **Mengen je Offset** (Lead 1, neuester Snapshot, mit Settlement): −4: 250,
   −3: 305, −2: 317, −1: 323, 0: 317, +1: 311, +2: 296, +3: 256, +4: 167.
   Für −1 wie für −2: Median 15 Kandidaten je Zieltag, Maximum 22, 21 Zieltage.
3. **Die Preisleiter — und das ist die Zahl, die die Erwartung unten formt:**

   | Offset | −2 | −1 | 0 | +1 | +2 |
   |---|---|---|---|---|---|
   | Median `buy_no` | 0,960 | **0,770** | 0,670 | **0,780** | 0,960 |

   Der Markt bepreist die Leiter um **unseren** Favoriten herum praktisch
   symmetrisch. Eine grobe systematische Verschiebung unseres µ wäre hier bereits
   sichtbar — sie ist es nicht. Das ist ein Preis-Argument, kein Ist-Argument
   (der Markt kann irren), aber es genügt, um die Bias-These vorab zu schwächen.
4. **Handelbare −2-Kandidaten gibt es:** 81 der 353 liegen im Preisband
   0,70–0,90, obwohl der Median bei 0,960 steht. Das ist eine Selektion, keine
   Stichprobe — siehe Designfalle 3.
5. **Nicht bekannt** sind: jede realisierte Trefferquote, die Verteilung von
   `settle_k − k0`, jeder ROI, jede Aufteilung nach Stadt oder Tag.

Aus dem Bestand ebenfalls bekannt und hier ausdrücklich mitgeführt: der
Favoriten-Backtest vom 28.07. (`weather_fav_backtest_2026_07_28.md`) — unser
Ensemble-Favorit trifft **33,2 %**, der Markt-Favorit **47,4 %**, bei Uneinigkeit
21,8 % gegen 46,8 % (McNemar p < 0,01).

---

## Die Rechnung, die vor der Messung feststeht

Ein Lay auf den Bucket mit Offset *o* verliert genau dann, wenn das Settlement in
diesem Bucket landet. Bei NO-Preis *p*, Einsatz *S* = 5 $ und Gebühr
`0,07 · n · min(p, 1−p)` mit *n* = *S*/*p* ist der Erwartungswert null bei einer
Trefferquote *q*, die sich aus *p* allein ergibt. Für die Medianpreise oben:

| Klasse | NO-Preis | Break-even *q* ohne Gebühr | **mit Gebühr 0,07** | mit Gebühr 0,04 |
|---|---|---|---|---|
| −1 | 0,770 | 23,0 % | **21,4 %** | 21,9 % |
| −2 | 0,960 | 4,0 % | **3,7 %** | 3,8 % |

**Diese beiden Zahlen — 21,4 % und 3,7 % — sind die Messlatte, und sie stehen vor
der Messung fest.** Trifft der −1-Bucket häufiger als 21,4 %, verliert die Klasse
zwangsläufig; jede weitere Erklärung ist dann Ausschmückung.

Dazu die Referenz aus dem eigenen Modell: bei σ = 1,0 K, einem 1-K-Bucket und µ
in Bucketmitte liefert die Normalverteilung P(Favorit) ≈ 38 %, **P(−1) ≈ 24 %**,
**P(−2) ≈ 6 %**. Beide liegen **über** ihrer Break-even-Schwelle. Das heißt im
Klartext: **bei fairer Bepreisung und σ ≈ 1,0 K ist auf der eq-Leiter keine
Lay-Klasse profitabel.** Ein Ertrag kann dann nur aus Lagen kommen, in denen
unser σ tatsächlich kleiner ist als das vom Markt unterstellte — also aus
Selektion, nicht aus der Klasse.

---

## Universum, Daten, Fenster

- **Einheit:** Stadt-Tag (`target_date`, `city`), `var='max'`, `kind='eq'`.
- **Snapshot:** **Lead 1**, neuester Snapshot je Stadt-Tag — der Kaufzeitpunkt des
  Autobuy (14:45), identisch zum Favoriten-Backtest vom 28.07. Lead 0 und 2 gehen
  in kein Gate ein; Lead 2 dient als Robustheitsprobe.
- **Anker:** k0 = `favorit_k(mu_ens, city)`, also `k − offset_fav`. Stadtabhängig
  wegen der floor-Buckets in Hong Kong.
- **Wahrheit:** `settle_k` (METAR bzw. HKO). Nach dem Befund vom 02.08. folgt der
  Markt in allen fünf divergenten Fällen METAR, nie WU
  ([[weather-settlement-wu-vs-metar]]). `wu_settle_k` wird **nicht** verwendet;
  als Datenprobe wird ausgewiesen, in wie vielen Fällen `settle_result` der aus
  `settle_k` abgeleiteten Trefferaussage widerspricht.
- **Die zentrale Größe:** *d* = `settle_k` − k0, der **realisierte Offset**. Aus
  seiner Verteilung folgen alle Trefferquoten: P(*d* = −1) ist die Verliererquote
  der −1-Klasse, P(*d* = −2) die der −2-Klasse.
- **Drei Wahrscheinlichkeiten je Offset**, die gegeneinander gestellt werden:
  - **P_ist** — beobachtete Häufigkeit von *d*
  - **P_modell** — aus `mu_ens`, `sigma_ens` und `bucket_grenzen(k, city)`
  - **P_markt** — Mitte aus `buy_yes` und `1 − buy_no`
- **Signifikanz:** t über **Tages**-Mittel, nie über Stadt-Tage. Dreißig Städte
  eines Zieltags hängen an derselben Großwetterlage.
- **Robustheit:** Jede Hauptaussage wird auf zwei Hälften geprüft —
  **11.–21.07.** und **22.07.–01.08.** Das Vorzeichen muss in beiden gleich sein.

---

## Hypothesen

**H1 (Anker):** Unsere Bucket-Achse ist verschoben — E[*d*] < 0, wir layen also
im Mittel den Bucket, in dem das Settlement tatsächlich landet.
*Konsequenz falls belegt:* µ korrigieren. Kein Preisfilter repariert einen
falschen Anker.

**H2 (Breite):** Die Achse stimmt, aber unsere Sicherheit ist eingebildet — die
realisierte Streuung von *d* ist größer als `sigma_ens` unterstellt, P_ist(−1)
liegt signifikant über P_modell(−1).
*Konsequenz falls belegt:* σ anheben; damit fallen Kandidaten aus dem
P_pess-Filter, das Buch wird enger statt breiter.

**H3 (fairer Preis — der banale Kandidat):** Weder noch. P_ist liegt nahe an
P_markt, und die Klasse verliert, weil eine fair bepreiste Leiter nach Gebühr
keinen Lay trägt. *Konsequenz falls belegt:* Der Ertrag des Buchs stammt
ausschließlich aus Selektion (Preisband, Spannen-Veto, P_pess) — und die
Grundgesamtheit ist kein Reservoir, aus dem sich mehr schöpfen ließe.

**H4 (Symmetrie — die Kontrolle, die H1 von H2/H3 trennt):** Verliert die
**+1**-Klasse ebenso wie die −1-Klasse? Sie ist mit NO 0,780 fast identisch
bepreist, also direkt vergleichbar.
- nur −1 verliert, +1 gewinnt → spricht für **H1**
- beide verlieren etwa gleich → spricht für **H2** oder **H3**

**H5 (−2-Klasse):** P_ist(−2) liegt **unter** 3,7 %, die Klasse trägt also nach
Gebühr. Gegenerwartung ist die Modellreferenz von ≈ 6 %.

**H6 (Anker-Vergleich, nachgeordnet):** Relativ zum **Markt**-Favoriten gemessen
(*d*ₘ = `settle_k` − `market_fav_k`) trifft der −1-Bucket seltener als relativ zu
unserem Anker. *Diese Hypothese führt zu keiner Codeänderung*, sondern höchstens
zu einer eigenen Forward-Pre-Reg — sie prüft nur, ob der Reparaturweg überhaupt
in diese Richtung zeigt.

**H0:** *d* ist symmetrisch um 0 verteilt, P_ist ≈ P_modell ≈ P_markt, und beide
Klassen verlieren um den Betrag der Gebühr. Dann ist nichts kaputt, und der
Befund vom Vormittag beschreibt kein Leck, sondern die Bauart des Marktes.

---

## Gates

| Gate | Bedingung |
|---|---|
| **G0** Datenbasis | ≥ 250 Stadt-Tage mit `mu_ens` und `settle_k` bei Lead 1, ≥ 18 Zieltage, ≥ 20 Städte. Widerspruchsquote `settle_result` gegen `settle_k` < 3 % — darüber wird abgebrochen und erst die Datenlage geklärt |
| **G1** Anker (H1) | Belegt bei \|E[*d*]\| ≥ 0,25 Bucket **und** t > 2,0 über Tagesmittel **und** gleiches Vorzeichen in beiden Hälften |
| **G2** Breite (H2) | Belegt, wenn P_ist(−1) − P_modell(−1) ≥ 4 pp bei t > 2,0, in beiden Hälften gleichgerichtet |
| **G3** Fairness (H3) | Belegt, wenn \|P_ist(−1) − P_markt(−1)\| < 3 pp **und** G1 wie G2 gerissen sind |
| **G4** Symmetrie (H4) | Rein diagnostisch, keine Schwelle: ausgewiesen werden P_ist(−1) und P_ist(+1) samt Differenz und t. Die Entscheidungstabelle unten liest sie |
| **G5** −2-Klasse (H5) | Trägt, wenn P_ist(−2) < 3,7 % **und** der ROI der −2-Klasse nach Gebühr 0,07 positiv ist **und** beide Hälften dasselbe Vorzeichen zeigen |

**Bonferroni:** Sechs Hypothesen, aber **eine** Messung und **eine** Konsequenz.
Die t-Schwellen stehen deshalb auf 2,0 statt der sonst üblichen 1,5. Es wird
genau ein Schnitt je Frage gerechnet; es werden keine Preisbänder, keine
Abstandsschwellen und keine Teilmengen von Städten gesucht.

**Entscheidungstabelle — vorab, damit hinterher nicht die bequemste Erklärung
gewinnt:**

| G1 | G2 | G4 (+1) | Diagnose | Was daraus folgen darf |
|---|---|---|---|---|
| belegt | – | +1 gewinnt | **Anker verschoben** | eigene Pre-Reg zur µ-Korrektur |
| gerissen | belegt | +1 verliert auch | **σ zu klein** | eigene Pre-Reg zu σ / P_pess |
| gerissen | gerissen | +1 verliert auch | **fair bepreist (H3)** | keine Reparatur; die Grundgesamtheit ist kein Reservoir |
| belegt | belegt | beliebig | beides | µ zuerst, σ danach — getrennt vorregistriert |

---

## Designfallen, die diesen Test definieren

**1. Shenzhen hat bis heute gegen die falsche Station kalibriert.** Der Locator
`ZGSZ:9:CN` lieferte „Lau Fau Shan" in Hong Kong; die Kalibrierung war bis zum
Fix vom 02.08. gegen eine 25 km entfernte Station gefittet (σ 1,491 → 1,084 nach
der Korrektur). Alle historischen Shenzhen-Zeilen tragen damit ein verzerrtes
`mu_ens` und ein falsches k0 — **genau die Größe, die hier gemessen wird**.
Vorab festgelegt: Hauptlauf mit allen Städten, Kontrolllauf **ohne Shenzhen**.
Weichen die Hauptaussagen ab, gilt der Lauf ohne Shenzhen.

**2. µ liegt nicht in der Bucketmitte.** Die Referenzzahlen (24 % / 6 %) gelten
für µ in der Mitte. Liegt µ am Bucketrand, verschiebt sich Masse in den
Nachbarbucket. P_modell wird deshalb **je Stadt-Tag einzeln** über die echten
Bucketgrenzen integriert und erst dann gemittelt — nie aus einer
Durchschnittsposition gerechnet.

**3. Die 81 handelbaren −2-Kandidaten sind eine Selektion.** Ein −2-Bucket, der
statt 0,96 nur 0,80 kostet, ist keine billige Gelegenheit, sondern ein Fall, in
dem der Markt die Verteilung für breit hält. Wer die −2-Klasse „im Preisband
0,70–0,90" testet, misst unruhige Lagen, nicht die −2-Klasse. **H5 wird deshalb
auf der vollen −2-Menge zum jeweiligen Marktpreis gerechnet**, und die
Band-Teilmenge nur diagnostisch daneben ausgewiesen.

**4. Der Lead bestimmt den Anker.** `mu_ens` und damit k0 ändern sich zwischen
den Snapshots. Wer über Leads mittelt, mischt Prognosen verschiedener Frische und
verwischt genau die Verschiebung, die gesucht wird. Lead 1 ist fix; Lead 2 läuft
nur als Robustheitsprobe.

**5. Städte ohne Kalibrierung fehlen systematisch.** Nur 387 von 517 Stadt-Tagen
haben ein `mu_ens` — Städte ohne Station oder ohne Kalibrierung bekommen bewusst
keins. Die Stichprobe ist damit die Menge der Städte, auf denen der Bot handelt,
und **nicht** repräsentativ für die Bretter insgesamt. Das ist für die Frage
richtig so, aber es begrenzt die Aussage.

**6. Zwei Sommerwochen bleiben zwei Sommerwochen.** 22 Zieltage tragen kein
starkes t. Der Test taugt zur Falsifikation einer Ursache, nicht zum Beweis
einer anderen — dieselbe Asymmetrie wie beim konditionalen Ausstieg und beim
Preisband.

---

## Vorab-Erwartung (damit sie nicht zurechtgebogen wird)

**Ich erwarte H3, nicht H1.** Die Preisleiter ist um unseren Favoriten symmetrisch
(0,770 gegen 0,780), und die Modellreferenz sagt, dass P(−1) ≈ 24 % gegen eine
Break-even-Schwelle von 21,4 % ohnehin verliert. Beides zusammen erklärt den
Befund vom Vormittag vollständig, ohne dass irgendetwas defekt sein müsste. Meine
Erwartung ist P_ist(−1) zwischen 22 % und 27 % und ein E[*d*] nahe null.

**H1 halte ich trotzdem nicht für erledigt**, und zwar wegen der 25-pp-Lücke bei
Uneinigkeit aus dem Favoriten-Backtest: unser Favorit trifft dort 21,8 %, der
Markt-Favorit 46,8 %. Ein solcher Abstand ist mit „Achse stimmt, nur verrauscht"
schwer vereinbar. Möglich ist, dass die Achse im Mittel stimmt und **je Stadt**
verschoben ist — das würde bei E[*d*] ≈ 0 unentdeckt bleiben. Deshalb wird die
Streuung von E[*d*] **über Städte** mit ausgewiesen, ausdrücklich diagnostisch
und ohne Gate.

**H5 erwarte ich als gerissen.** P(−2) ≈ 6 % gegen 3,7 % Break-even ist kein
knapper Fall. Der Chengdu-Tag des Nutzers wäre dann das, was er statistisch ist:
ein gewonnener Einzelfall in einer Klasse, die im Mittel zu teuer bezahlt wird.
Sollte es anders kommen, ist das der interessanteste Befund dieser Messung.

**Was der Ausgang H3 bedeuten würde — vorab benannt, weil es unbequem ist:** Dann
hat das Buch keinen strukturellen Ertrag auf der eq-Leiter, sondern nur einen
Selektionsertrag. Die realisierten +2,51 % über 79 Positionen wären dann kein
kleiner Edge auf einer breiten Basis, sondern der ganze Edge — und jede
Vergrößerung der Grundgesamtheit würde ihn verdünnen statt vermehren. Das ist
die Lesart, die der Skalierungsfrage widerspricht, und sie steht hier **vor** der
Messung.

## Abbruchregel

Reißt **G0**, wird nicht gerechnet, sondern erst die Datenlage geklärt.

Ist die Diagnose **H3** (fair bepreist), wird **nicht** nach einer Stadt, einem
Monat oder einem Preisband gesucht, in dem die −1-Klasse doch trägt. Der Befund
lautet dann: die Klasse ist kein Reservoir, der Ertrag kommt aus der Auswahl —
und die nächste Frage gehört an die Auswahl, nicht an die Klasse.

Ist die Diagnose **H1** oder **H2**, folgt daraus **keine** sofortige Änderung am
Autobuy. Die Korrektur wird eigens vorregistriert und **forward** geprüft. Die
V2-Erfahrung — ohne Forward-Test live, mit ausdrücklich deklarierter Abweichung
von der Projektmethodik — wird nicht wiederholt.

Trägt **H5**, geht die −2-Klasse ebenfalls nicht live, sondern zuerst als
Schattenbuch. Ein einzelner gewonnener Chengdu-Tag ändert daran nichts.

Für die laufenden Fäden gilt unverändert: **H4 des Preisband-Tests (Spannen-Veto)
bleibt Forward ab 03.08.**, und der Ensemble-µ-Forward-Test wird vor Oktober
nicht angefasst.

---

# ERGEBNIS — die Leiter ist fair bepreist (02.08.2026)

Gerechnet mit `weather_minus1_ursache_eval.py` unmittelbar nach der
Vorregistrierung (Commit `0ec4682`), ohne Parameteränderung. Lead 1, 325
Stadt-Tage, 21 Zieltage (12.07.–01.08.), 30 Städte. **G0 bestanden**, und zwar
deutlich: `settle_result` widerspricht `settle_k` in **0 von 2.925** Fällen.

## Die Verteilung des realisierten Offsets

| Offset | P_ist | P_modell | P_markt | Break-even (Fee 0,07) |
|---|---|---|---|---|
| −3 | 1,8 % | 2,6 % | 2,5 % | 1,9 % |
| **−2** | **5,8 %** | 9,3 % | 8,6 % | **7,3 %** |
| **−1** | **23,7 %** | 22,5 % | 25,1 % | **22,6 %** |
| 0 | 32,6 % | 30,5 % | 32,3 % | 29,3 % |
| **+1** | **24,6 %** | 22,0 % | 22,5 % | **20,2 %** |
| +2 | 7,1 % | 8,9 % | 9,4 % | 8,1 % |
| +3 | 2,8 % | 2,5 % | 2,6 % | 2,0 % |

## G1, G2, G3 — die Diagnose lautet H3

- **G1 Anker: NICHT belegt.** E[*d*] = **+0,035** Bucket, t = +0,57. Die
  Vorzeichen der beiden Hälften widersprechen sich ohnehin (−0,135 / +0,190).
  Unsere Bucket-Achse ist **im Mittel nicht verschoben**.
- **G2 Breite: NICHT belegt.** P_ist(−1) 23,7 % gegen P_modell(−1) 22,5 % —
  **+1,3 pp** bei t = +0,73, verlangt waren 4 pp und t > 2,0. `sigma_ens` ist
  nicht nennenswert zu klein; über die ganze Leiter liegt das Modell sogar eher
  zu breit (−2: 9,3 % modelliert gegen 5,8 % real).
- **G3 Fairness: BELEGT.** P_ist(−1) 23,7 % gegen P_markt(−1) 25,1 % —
  **−1,4 pp**, innerhalb der geforderten 3 pp, bei gerissenem G1 und G2.

**Damit ist die Frage beantwortet, und die Antwort ist die unbequeme:** Die
−1-Klasse verliert nicht, weil etwas kaputt ist. Sie verliert, weil sie zu
23,9 % trifft, während der Break-even bei 22,6 % liegt. **1,3 Prozentpunkte** —
das ist der ganze Abstand zwischen „strukturell verlierende Grundgesamtheit" und
„fair". Der ROI von **−3,85 %** über 322 Lays ist genau das, was daraus folgt.

## G4 Symmetrie — es ist nicht der Anker

| Klasse | n | Ø NO | Treffer | Break-even | ROI |
|---|---|---|---|---|---|
| −1 | 322 | 0,758 | 23,9 % | 22,6 % | **−3,85 %** |
| +1 | 310 | 0,783 | 25,8 % | 20,2 % | **−8,35 %** |

P_ist(−1) − P_ist(+1) = −0,3 pp, t = −0,08. **Beide Nachbarklassen verlieren, die
obere sogar stärker.** Eine Verschiebung des Ankers hätte genau das Gegenteil
erzeugt. Die Entscheidungstabelle liest das als Bestätigung von H3.

## Der Fund, für den es kein Gate gab

Die Achse stimmt **im Mittel** — und ist **je Stadt** massiv verschoben:

**sd = 0,699 Bucket über 28 Städte. Beijing −1,09, Taipei +1,45.**

Ein ganzer Bucket Verschiebung ist die Größenordnung, in der die Lay-Auswahl
komplett danebengreift. Im Mittelwert über alle Städte hebt sich das auf, und
genau deshalb zeigt G1 nichts. Das war in der Vorab-Erwartung als Möglichkeit
benannt („die Achse könnte im Mittel stimmen und je Stadt verschoben sein — das
würde bei E[*d*] ≈ 0 unentdeckt bleiben") und deshalb ohne Gate ausgewiesen. Es
ist der stärkste Einzelbefund dieser Messung und **hat keinen Beleg-Status**:
28 Städte mit im Mittel 11 Tagen tragen kein t. Er gehört in eine eigene
Vorregistrierung, nicht in eine Codeänderung.

## G5 — die −2-Klasse, und ein Fehler in meinem eigenen Gate

| Menge | n | Ø NO | Treffer | Break-even | ROI (0,07) | ROI (0,04) |
|---|---|---|---|---|---|---|
| volle −2-Menge | 316 | 0,922 | **5,7 %** | **7,3 %** | **+2,00 %** | +2,29 % |
| davon Preisband 0,70–0,90 | 74 | — | 17,6 % | — | −0,93 % | — |

Beide Hälften positiv (+1,30 % / +2,80 %), ohne Shenzhen +2,16 %.

**G5 gilt trotzdem als GERISSEN**, weil die vorregistrierte Bedingung
„P_ist(−2) < 3,7 %" nicht erfüllt ist. Diese Schwelle war jedoch **falsch
parametrisiert**: die 3,7 % wurden aus dem *Median*preis 0,960 abgeleitet, der
tatsächlich gehandelte Mittelpreis liegt bei **0,922**, und dort steht der
Break-even bei **7,3 %**. Zwei der drei Bedingungen (ROI > 0, beide Hälften
gleich) sind erfüllt, die dritte misst dieselbe Sache mit der falschen
Preisannahme.

**Das wird hier nicht zurechtgebogen.** Ein Gate, das man nach Sicht der Daten
umdeutet, ist kein Gate. Der Befund gilt als **unbelegt** und ist der Anlass für
eine eigene Vorregistrierung — mit Break-even positionsweise aus dem echten
Preis, nicht aus einem Median, und mit Forward-Fenster. Die Lehre für die nächste
Pre-Reg: **Schwellen nie aus einer Lagekennzahl ableiten, wenn die zugrunde
liegende Verteilung schief ist.**

**Post hoc, ausdrücklich kein Gate:** die 74 Kandidaten im Preisband verlieren
(17,6 % Treffer), die teuren tragen. Rechnerisch bleiben für die restlichen 242
etwa +2,9 % ROI bei rund 1 % Trefferquote gegen 4 % bepreiste. Das ist das Muster
eines **Longshot-Bias** — der Markt bezahlt den unwahrscheinlichen Ausgang zu
teuer, also verdient dessen Lay. Es ist zugleich die **Umkehrung der
−1-Doktrin**: dort liegt der Ertrag im Band 0,70–0,90, hier bei NO ≈ 0,96.
Designfalle 3 hat sich damit bestätigt — wer die −2-Klasse im Preisband testet,
misst das Gegenteil ihrer Ökonomie.

**Ökonomisch ist der Befund klein und teuer:** +2,00 % auf 5 $ sind 0,10 $ je
Position, bei ~15 Kandidaten je Zieltag und zwei überlappenden Tagen also rund
150 $ gebundenes Kapital gegen zuletzt ~93 $ frei. Dazu kommt die Frage, ob bei
NO 0,96 überhaupt zu diesem Preis gefüllt wird ([[weather-screen-price-vs-book]]).

## H6 — der Markt-Favorit ist der bessere Anker

| | Anker trifft | Bucket unter dem Anker trifft |
|---|---|---|
| eigener Ensemble-Favorit | 32,6 % | **23,7 %** |
| Markt-Favorit | **46,8 %** | **19,7 %** |

Der Favoriten-Befund vom 28.07. (33,2 % gegen 47,4 %) reproduziert sich auf einer
anderen Stichprobe fast punktgenau. Neu ist die zweite Spalte: **unter dem
Markt-Anker trifft der −1-Bucket nur 19,7 %** — das liegt **unter** dem
Break-even von 22,6 %, während unsere eigenen 23,7 % darüber liegen. Kein Gate,
keine Codeänderung; aber von allem, was hier gemessen wurde, zeigt das am
klarsten in eine Richtung.

## Kontrollläufe

- **Ohne Shenzhen** (319 Stadt-Tage): jede Gate-Entscheidung identisch, alle
  Kennzahlen innerhalb weniger Zehntel. Die Sorge aus Designfalle 1 war
  berechtigt, der Effekt ist folgenlos.
- **Lead 2 nicht auswertbar:** 249 Stadt-Tage gegen die vorregistrierte
  Mindestgröße von 250. Die Abbruchregel greift wörtlich; die Schwelle wird
  **nicht** um einen Stadt-Tag gesenkt, um die Probe doch noch zu bekommen. Für
  die nächste Pre-Reg: Mindestgrößen für Nebenproben getrennt vom Hauptlauf
  festlegen.

## Was das heißt

**Es gibt kein Leck zu stopfen.** Die −1-Klasse ist keine defekte
Grundgesamtheit, sondern eine fair bepreiste. Der Ertrag des Buchs stammt damit
**ausschließlich aus Selektion** — Preisband, Spannen-Veto, P_pess — und nicht
aus der Klasse. Das war als unbequeme Lesart vorab benannt und ist jetzt
gemessen.

Daraus folgt unmittelbar die Bestätigung der Skalierungs-Absage vom Vormittag:
**mehr Kandidaten verdünnen den Edge, sie vermehren ihn nicht.** Wer die
Grundgesamtheit vergrößert, kauft sich in eine Menge mit ROI −3,85 % ein.

**Aus dieser Messung folgt keine Änderung am laufenden Autobuy**, wie
vorregistriert.

## Was nicht geprüft und bewusst offen ist

- **Die Verschiebung je Stadt ist der nächste Faden**, aber sie ist hier nur
  gesehen, nicht belegt. Ob Beijing und Taipei ein Kalibrierungsproblem, ein
  Stationsproblem oder Rauschen aus elf Tagen sind, entscheidet diese Messung
  nicht.
- **Warum das Modell die Ränder zu breit schätzt** (−2: 9,3 % modelliert gegen
  5,8 % real, +2: 8,9 % gegen 7,1 %), ist nicht untersucht. Es ist die
  Spiegelseite des Longshot-Befunds und könnte dieselbe Ursache haben.
- **Der Zeitraum bleibt ein Sommerfenster.** 21 Zieltage, zwei Hälften à elf —
  die Robustheitsprobe ist schwach, und keine Aussage hier ist gegen eine andere
  Jahreszeit geprüft.
- **H6 misst Anker gegen Anker, nicht Strategie gegen Strategie.** Ob ein am
  Markt-Favoriten ausgerichtetes Buch nach Gebühr und Slippage verdient, ist
  damit nicht gezeigt — nur, dass die Trefferquote auf der richtigen Seite des
  Break-even läge.
