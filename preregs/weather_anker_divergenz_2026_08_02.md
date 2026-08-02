# Pre-Registrierung: Sagt die Kalibrierungs-Divergenz den Lay-Ausgang voraus? — 2026-08-02

**Status:** Vorregistrierung. Der retrospektive Teil ist ausdrücklich **Bezifferung
und kein Beleg** — zwei der Extremstädte sind bereits bekannt (siehe unten), das
Juli-Fenster ist für diese Frage angefasst. Der **Beleg muss forward kommen**, ab
Zieltag 03.08.2026.

Auswertung folgt in `weather_anker_divergenz_eval.py`.

---

## Anlass und was die Nachprüfung ergeben hat

Die Ursachen-Messung vom 02.08. fand einen Befund, für den bewusst kein Gate
existierte: die Bucket-Achse stimmt im Mittel (E[*d*] = +0,035) und ist **je Stadt
massiv verschoben — sd = 0,699 Bucket über 28 Städte, Beijing −1,09, Taipei
+1,45**. Die naheliegende Aufgabe wäre gewesen, diese Verschiebung zu vermessen.
Die Nachprüfung zeigt, dass genau das schon geschehen ist — und verschiebt die
Frage.

**Erstens: die Ursache ist seit dem 28.07. gemessen.**
`weather_calib_divergence_eval.py` (Commit `c4e5e70`) hat den Zusammenhang
zwischen der Kalibrierungs-Divergenz **D = bias₇₀₀d − bias₄₀d** und der Richtung
des realisierten Fehlers mit **r = +0,752, t = +5,71** über 27 Städte belegt. Die
beiden heute bekannten Extremwerte bestätigen ihn auf **Bucket**-Ebene:

| Stadt | b₇₀₀d | b₄₀d | **D** | gemessenes **d̄** |
|---|---|---|---|---|
| Beijing | −0,873 | +0,130 | **−1,003** | **−1,09** |
| Taipei | −0,542 | −1,743 | **+1,201** | **+1,45** |

**d̄ ≈ D.** Die Ankerverschiebung ist damit keine unbekannte Größe, sondern aus
einer Zahl vorhersagbar, die **vor dem Kauf** feststeht.

**Zweitens: sieben von 31 Städten liegen über der Doktrin-Schwelle.**

| über 0,7 K | D | | unter 0,7 K (Auswahl) | D |
|---|---|---|---|---|
| Jeddah | −1,213 | | Mexico City | −0,521 |
| Taipei | +1,201 | | Kuala Lumpur | +0,482 |
| Seoul | +1,179 | | Toronto | −0,406 |
| München | +1,163 | | Paris | +0,352 |
| Beijing | −1,003 | | London | −0,042 |
| NYC | −1,003 | | Madrid | −0,007 |
| Tel Aviv | −0,902 | | | |

**Drittens — und das ist kein Forschungsbefund, sondern ein Zustand: der Autobuy
wendet die Doktrin nicht an.** `MAX_DIVERGENZ = 0.7` steht in
`weather_outlier_screen.py:195` und wird von beiden Screens geprüft
([[weather-low-min-calibration-blocker]], „seit 14.07. im Code erzwungen").
`weather_minus1_autobuy.py` importiert aus derselben Datei **nur** `MAX_SPREAD`;
seine dokumentierte Regelkette (Schritte 1–8) kennt kein Divergenz-Gate.
Gemessen am Bestand: **20 von 147 Band-Kandidaten (13,6 %)** stammen aus Städten,
die ein Screen abgelehnt hätte — Beijing 7, München 4, Seoul 4, Jeddah 4,
Taipei 1.

## Was hier NICHT behauptet wird

**Nicht: „auf die 40d-Kalibrierung umstellen."** Das ist am 28.07. gemessen und
verworfen worden (`weather_calib_lay_pnl_eval.py`, Commit `593d99c`): im Lay-Buch
kam davon nichts an (GATE t = −0,16, BAND t = −1,13). `load_calib()` bleibt auf
der 700d-Basis. Diese Pre-Reg fasst die Kalibrierung **nicht** an.

**Nicht: „die Verschiebung ist ein Fehler, der weg muss."** Für ein Lay-Buch ist
das Vorzeichen entscheidend, nicht der Betrag — siehe die einseitige These unten.
Ein Anker, der zu **kalt** liegt, schiebt das Lay-Ziel **weiter weg** vom wahren
Favoriten und ist harmlos bis nützlich.

**Nicht: „hier wird ein Filter gebaut."** Aus dieser Pre-Reg folgt keine
Codeänderung.

**Und ausdrücklich nicht: „Städte werden gesperrt."** Der Betreiber hat am
02.08. entschieden, dass **keine Stadt aus dem Kaufuniversum fallen soll** — auch
Beijing, Jeddah, München, Seoul und Taipei nicht. Die fehlende
`MAX_DIVERGENZ`-Prüfung im Autobuy bleibt damit bewusst bestehen und ist keine
offene Frage mehr.

Das ist keine Einschränkung des Tests, sondern eine Schärfung: **die Diagnose
lautet „der Anker ist verschoben", und die Reparatur eines verschobenen Ankers
ist seine Korrektur, nicht das Streichen der Stadt.** Der Ausschluss wäre nur der
billige Weg über ein bereits vorhandenes Gate gewesen. Der Verwertungsweg dieser
Pre-Reg ist deshalb **H5** unten.

Sondern: **Trägt D Information über den Ausgang eines −1-Lays — und in welche
Richtung?**

---

## Die einseitige These, und warum sie unbequem ist

Gelayt wird der Bucket **unter** unserem Favoriten. Daraus folgt eine Asymmetrie,
die ein symmetrisches |D| ≤ 0,7-Gate nicht abbildet:

- **D < 0** (unser µ ist **wärmer** als die Sommersicht): unser Favorit sitzt zu
  hoch, der gelayte −1-Bucket rutscht in die Nähe des **wahren** Favoriten. Ein
  Feld mit hoher Eintrittswahrscheinlichkeit ist das **schlechteste** Lay-Ziel.
  ⇒ **Diese Städte müssten verlieren.**
- **D > 0** (unser µ ist **kälter**): der gelayte Bucket rutscht **weiter weg**
  vom wahren Favoriten. ⇒ **Diese Städte müssten gewinnen oder neutral sein.**

Das ist exakt der Mechanismus, den die 28.07.-Auswertung als Lehre formuliert hat
(„eine Sicht, die den Favoriten öfter trifft, schiebt das Lay-Ziel auf das Feld,
das vorher als Favorit galt") — hier nur mit dem Vorzeichen versehen, das dort
nicht gebraucht wurde.

**Und hier ist die unbequeme Konsequenz, die vorab benannt gehört:** Sollte sich
die These bestätigen, lautet der ökonomisch attraktivste Schluss „behalte den
Kalibrierungsfehler dort, wo er nützt" — also den Anker **nur** bei D < 0
korrigieren und ihn bei D > 0 stehen lassen. **Das ist keine Strategie, sondern
eine Wette darauf, dass ein Modellfehler stabil bleibt.** Er verschwindet, sobald
die 40d-CSVs nachgeführt werden — deren Stand ist bei den meisten Städten der
17.07., ein Teil jeder gemessenen Divergenz ist schlicht Alterung.

**Deshalb wird H5 in beiden Richtungen geprüft und die Korrektur nur dann
vorgeschlagen, wenn sie beidseitig trägt.** Eine einseitige Korrektur, die den
Fehler dort behält, wo er zufällig zahlt, wird aus dieser Pre-Reg **nicht**
abgeleitet. Das steht hier, damit es später nicht anders gelesen wird.

---

## Was schon gesehen wurde — vollständige Offenlegung

1. Der 28.07.-Befund vollständig (r = +0,752, t = +5,71; 40d im Lay-Buch
   wertlos, GATE t = −0,16 bei nur 3 von 78 abweichenden Lay-Zielen innerhalb
   des Gates — der Test hatte dort praktisch keine Trennschärfe).
2. **d̄ für genau zwei Städte:** Beijing −1,09 und Taipei +1,45, beide aus der
   Ergebnistabelle vom 02.08. Beide liegen über 0,7 K, und **beide bestätigen die
   Vorzeichenthese**. Damit ist der retrospektive Ausgang teilweise
   vorweggenommen — der Grund, warum er unten kein Gate trägt.
3. Die vollständige D-Tabelle (oben) und die Mengen: 323 −1-Kandidaten Lead 1
   mit Settlement, davon **77 (23,8 %)** in Städten über 0,7 K; im Preisband 147
   Kandidaten, davon **20 (13,6 %)**.
4. sd der Stadt-Mittelwerte = 0,699 Bucket, E[*d*] gesamt = +0,035.
5. **Nicht bekannt:** d̄ für die übrigen 26 Städte, jede Trefferquote je Stadt,
   jeder ROI je Stadt oder Gruppe, jede Aufteilung nach dem Vorzeichen von D.

---

## Universum, Daten, Fenster

- **Kandidaten:** `bb_WeatherLadders`, `var='max'`, `kind='eq'`, `offset_fav=−1`,
  Lead 1, neuester Snapshot je Stadt-Tag, Settlement über `settle_k`.
- **D je Stadt:** `bias₇₀₀d − bias₄₀d` für `model='ensemble_mean'`, geladen mit
  derselben Vorrangregel wie `load_calib()` (spätere Datei gewinnt, `_lead`- und
  `_min_`-Dateien ausgeschlossen). **D ist eine Konstante je Stadt, kein Tageswert**
  — das ist eine Schwäche, siehe Designfalle 2.
- **Rechnung:** 5 $ je Lay, Gebühr `0,07 · n · min(NO, 1−NO)`, Break-even
  positionsweise aus dem echten Preis.
- **Signifikanz:** t über Tagesmittel. Zusätzlich die stetige Auswertung über
  Städte, weil sie die Stichprobe deutlich besser ausnutzt als ein
  Gruppenvergleich.
- **Fenster R** (retrospektiv, Bezifferung): Zieltage 12.07.–01.08.2026.
  **Fenster F** (Forward, Beleg): ab 03.08.2026.

---

## Hypothesen

**H1 (stetig, Haupttest):** D sagt den Lay-Ausgang vorzeichenrichtig voraus — je
negativer D, desto häufiger trifft der gelayte −1-Bucket. Gemessen als Korrelation
zwischen D und der Trefferquote je Stadt, über Kandidatenzahl gewichtet.

**H2 (Gruppen, ökonomisch):** Lays in Städten mit **D < −0,7** liefern einen
schlechteren ROI als die übrigen. Die Schwelle 0,7 wird **nicht** gesucht,
sondern aus der bestehenden Doktrin übernommen.

**H3 (Gegenprobe, die H1 vor einer Scheinbestätigung schützt):** Städte mit
**D > +0,7** dürfen **nicht** ebenfalls schlechter abschneiden. Wäre der Effekt
in beiden Richtungen negativ, dann misst |D| nur „unsichere Stadt" und nicht die
gerichtete Ankerverschiebung — und die einseitige These wäre falsch.

**H4 (Vollständigkeit, diagnostisch):** Nach Herausrechnen von D bleibt keine
nennenswerte Stadt-Verschiebung übrig. Prüft, ob D die Sache erklärt oder nur
einen Teil. Kein Gate.

**H5 (Verwertung — Ankerkorrektur statt Sperre):** Für Städte mit |D| > 0,7
liefert das −1-Lay relativ zum **korrigierten** Anker einen besseren ROI als
relativ zum heutigen. Der korrigierte Anker ist
k0′ = `favorit_k(mu_ens + D, city)` — denn aus µ = ens_raw − bias folgt
**µ₄₀d = µ₇₀₀d + D**, und genau dorthin zeigt die gemessene Verschiebung
(d̄ ≈ D). Gelayt wird dann k0′ − 1; der Preis dafür steht im Ladder-Log, weil die
ganze Leiter geloggt wird.

**Abgrenzung zum 28.07.:** Das ist *nicht* die dort verworfene Umstellung auf
40d. Jene Messung lief **innerhalb** des Gates D < 0,7 K, wo die beiden Sichten
sich in nur **3 von 78** Leitern überhaupt unterscheiden — sie kann über die
sieben divergenten Städte nichts sagen, weil sie sie ausgeschlossen hat. H5
prüft ausschließlich diese Städte, und die 700d-Basis bleibt für alle anderen
unangetastet.

**Beidseitig geprüft, aus Prinzip:** getrennt für D < −0,7 und D > +0,7. Eine
Korrektur, die nur dort greift, wo der Fehler zufällig zahlt, wird nicht
vorgeschlagen.

**H0:** D trägt keine Information über den Lay-Ausgang. Die 0,699 sind
Stichprobenrauschen aus im Mittel elf Tagen je Stadt, und die beiden bekannten
Extremwerte sind zwei Münzwürfe, die zufällig zur These passen.

---

## Gates

| Gate | Bedingung |
|---|---|
| **G0** Basis (Fenster F) | ≥ **30 Zieltage** ab 03.08., ≥ 400 −1-Kandidaten mit Settlement, ≥ 20 Städte, davon ≥ 4 mit \|D\| > 0,7 |
| **G1** Stetig (H1) | Gewichtete Korrelation zwischen D und Trefferquote je Stadt: **r ≤ −0,40** bei **t > 2,0**. Vorzeichen muss negativ sein — ein positives r widerlegt die These, es dreht sie nicht um |
| **G2** Ökonomisch (H2) | ROI der Gruppe D < −0,7 liegt **≥ 6 pp** unter dem der Gruppe \|D\| ≤ 0,7, t > 2,0 über Tagesmittel |
| **G3** Gegenprobe (H3) | Die Gruppe D > +0,7 liegt **nicht** ebenfalls ≥ 6 pp darunter. Reißt G3, gelten G1 und G2 als **nicht** belegt, unabhängig von ihren eigenen Zahlen |
| **G4** Robustheit | Beide Hälften von Fenster F gleiches Vorzeichen; der Effekt überlebt das Streichen der stärksten Einzelstadt |
| **G5** Verwertung (H5) | Auf den Städten mit \|D\| > 0,7: korrigierter Anker **≥ 4 pp** ROI über dem heutigen, **t > 2,0** über Tagesmittel — **und** in **keiner** der beiden Richtungen (D < −0,7 / D > +0,7) signifikant schlechter als heute |

**Bonferroni:** Fünf Hypothesen, t-Schwellen auf 2,0. **Genau eine** Schwelle
(0,7, aus der bestehenden Doktrin), **genau ein** Gruppenschnitt, **genau eine**
Korrekturformel (µ + D, nicht gefittet), **keine** Suche nach einer besseren
Grenze, keine Auswahl von Städten.

**Fenster R trägt kein Gate.** Es wird mit denselben Kennzahlen ausgewiesen, klar
als Bezifferung gekennzeichnet, und dient einem einzigen Zweck: der Abschätzung,
ob Fenster F überhaupt eine realistische Chance auf Trennschärfe hat.

---

## Designfallen, die diesen Test definieren

**1. Der retrospektive Ausgang ist teilweise bekannt.** Zwei Extremstädte, beide
über 0,7 K, beide vorzeichenkonform. Wer daraus einen Beleg macht, belegt seine
eigene Vorauswahl. Deshalb ist Fenster F der Test.

**2. D ist eine Konstante je Stadt — der Test ist ein Städtevergleich, kein
Tagesvergleich.** Effektiv gibt es 31 unabhängige Beobachtungen, nicht 323. Jede
Kennzahl, die so tut, als seien es 323, ist um etwa √10 zu optimistisch. Deshalb
läuft H1 über Stadt-Mittelwerte und nicht über Positionen.

**3. Ein Teil der Divergenz ist Alterung, kein Klima.** Der Stand der meisten
40d-CSVs ist der 17.07.; das Sommerfenster ist inzwischen weitergewandert. Wenn
D sich mit einer Neuberechnung ändert, ändert sich der Filter — und ein Filter,
der von der Aktualität einer CSV abhängt, ist fragil. Vorab festgelegt: **die
40d-CSVs werden während des Forward-Fensters nicht neu gerechnet.** Sonst misst
der Test zwei Dinge gleichzeitig.

**4. Städte mit großem D sind nicht zufällig ausgewählt.** Es sind überwiegend
Städte mit ausgeprägtem Jahresgang oder schwieriger Lage (Beijing, Seoul, Taipei,
Jeddah). Ein Gruppenunterschied kann deshalb „schwierige Stadt" bedeuten statt
„verschobener Anker". Genau das prüft die Gegenprobe H3: „schwierig" wirkt in
beide Richtungen, „verschoben" nur in eine.

**5. Die Gruppe D < −0,7 ist klein.** Vier Städte (Jeddah, Beijing, NYC, Tel
Aviv), im Juli-Fenster 11 von 147 Band-Kandidaten. Auf der breiten −1-Menge sind
es mehr, aber auch dort trägt die Gruppe den Test allein nicht — das ist der
Grund, warum H1 stetig formuliert ist und H2 nur nachgeordnet.

**6. Die Zeitachse der Kalibrierung ist nicht die des Tests.** `mu_ens` im
Ladder-Log stammt aus der 700d-Basis zum Zeitpunkt des Snapshots. Wird eine
700d-CSV nachgezogen, ändert sich D rückwirkend, die geloggten `mu_ens` aber
nicht. Für Fenster R wird deshalb der **heutige** Stand verwendet und dieser
Vorbehalt mitgeführt.

---

## Vorab-Erwartung (damit sie nicht zurechtgebogen wird)

**Ich erwarte, dass H1 in Fenster R deutlich bestätigt aussieht und in Fenster F
schwächer.** Der Grund ist Designfalle 1: der retrospektive Ausgang ist an den
beiden Extremen bekannt, und die 28.07.-Korrelation von r = 0,752 stand auf
derselben Datenbasis. Ein r um −0,6 in Fenster R würde mich nicht überraschen und
nichts belegen.

**In Fenster F erwarte ich Trennschärfe-Probleme.** Bei effektiv rund 30
unabhängigen Städten und 30 Zieltagen ist ein t > 2,0 möglich, aber nicht
wahrscheinlich, wenn der wahre Effekt bei einer halben Bucket-Verschiebung liegt.
Mein Tipp: r zwischen −0,3 und −0,5, t um 1,5 — also **G1 knapp gerissen**, mit
der Notwendigkeit einer Verlängerung.

**H3 halte ich für die eigentliche Entscheidung.** Wenn auch die D > +0,7-Gruppe
schlechter läuft, ist die ganze gerichtete Erklärung hinfällig, und übrig bleibt
das banale „unsichere Städte sind unsichere Städte" — was für ein symmetrisches
Gate spräche und gegen jede Feinsteuerung.

**H5 halte ich für offen und für den interessantesten Teil.** Die Korrektur
verschiebt das Lay-Ziel bei sieben Städten um einen Bucket — bei D < 0 weg vom
wahren Favoriten (müsste helfen), bei D > 0 auf ihn zu (müsste schaden). Wenn
beides eintritt, hebt es sich im Mittel auf, G5 reißt, und der ehrliche Befund
lautet: **die Verschiebung ist real, aber ihre Korrektur ist ein Nullsummenspiel,
solange der Markt jeden Bucket fair bepreist.** Genau das legt der Befund vom
02.08. Mittag nahe, und es wäre die konsistenteste Auflösung.

**Zur Größenordnung:** Es fällt **nichts** weg — die 13,6 % divergenten
Band-Kandidaten bleiben im Buch, ihr Lay-Ziel würde sich höchstens um einen
Bucket verschieben. Betroffen wären rund 20 von 147 Kandidaten. **Das ist kein
neuer Edge, sondern Hygiene** — und genau die Art von Auswahlverbesserung, die
der Befund vom 02.08. nahelegt, weil der Ertrag des Buchs ausschließlich aus der
Auswahl kommt.

## Abbruchregel

Reißt **G3** (Gegenprobe), ist die gerichtete These falsifiziert. Dann wird
**nicht** auf ein symmetrisches |D|-Kriterium ausgewichen und **nicht** nach einer
anderen Schwelle gesucht — das wäre eine neue These und bräuchte eine neue
Vorregistrierung.

Reißt **G1**, trägt D keine Information. Dann bleibt der Autobuy ohne
Divergenz-Prüfung, und die Abweichung von der Screen-Doktrin ist damit
**begründet** statt nur festgestellt.

Bestehen G1–G4, reißt aber **G5**, lautet der Befund: die Verschiebung ist real
und vorhersagbar, ihre Korrektur bringt aber nichts. Dann bleibt der Autobuy
unverändert, und das Ergebnis ist Wissen über den eigenen Anker, kein Werkzeug.
**Es wird dann nicht ersatzweise doch ein Ausschluss vorgeschlagen** — der ist
durch die Betreiber-Entscheidung vom 02.08. erledigt, unabhängig davon, was die
Zahlen sagen.

Bestehen G1–G5, folgt **kein** automatischer Einbau, sondern ein Vorschlag: den
Anker für die divergenten Städte um D zu korrigieren, zuerst als Schattenbuch
über ein zweites Fenster. Mit der Warnung aus dem Abschnitt „unbequeme
Konsequenz": die Wirkung hängt an der Aktualität der 40d-CSVs, und eine
Korrektur, die von einer alternden Datei lebt, ist auf Sicht selbst ein Risiko.
Wer sie einbaut, übernimmt die Pflicht, die 40d-Basis nachzuführen — und muss
damit rechnen, dass die Korrektur beim ersten Nachführen kleiner ausfällt.

**Keine Stadt fällt weg, in keinem Ausgang.** Das ist gesetzt und wird von diesem
Test nicht berührt.

Unberührt laufen weiter: Fenster D des Preisband-Tests (Spannen-Veto, ab 03.08.),
der Rand-Longshot-Forward-Test (Zwischenschau ~18.08.) und der
Ensemble-µ-Forward-Test (frühestens Oktober).
