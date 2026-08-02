# Pre-Registrierung: Trägt das Preisband, und darf die Auswahl daraus breiter werden? — 2026-08-02

**Status:** Vorregistrierung. Geschrieben, **bevor** für das Holdout-Fenster A
(10.–19.07.) irgendein PnL, ROI oder Trennschärfemaß gerechnet wurde. Die
Sondierung sah dort nur Mengen: Zahl der Kandidaten, Zahl der Verlierer, Zahl
der Zieltage. Was aus anderen Fenstern bereits bekannt ist, steht unten unter
„Was schon gesehen wurde" — vollständig, damit hinterher niemand rekonstruieren
muss, welche Zahl den Test geformt hat.

Auswertung folgt in `weather_band_breite_eval.py`.

---

## Anlass

Der System-Review am 02.08.2026 fand zwei Befunde, die sich widersprechen.

**Erstens: das Preisband hält out-of-sample.** Auf den Zieltagen 27.–31.07.,
also auf Tagen, die beim Bau des Bandes am 27.07. nicht existierten:

| Menge | n | ROI | t (Tagesmittel) |
|---|---|---|---|
| NO 0,70–0,90 | 38 | **+12,53 %** | 3,80 |
| NO 0,95–1,01 | 19 | −8,44 % | −0,80 |
| NO unter 0,70 | 26 | −37,76 % | −3,37 |
| alle Kandidaten | 90 | −6,73 % | −1,50 |

In-sample waren es +13,71 %. Der Befund reproduziert sich fast auf den
Punkt.

**Zweitens: die Live-Auswahl aus genau diesem Band steht im Minus.** Regime R5
(Zieltage ab 28.07.), 12 Positionen: **−12,78 %** wie gelaufen, −6,87 % beim
regelkonformen Halten. Über alle gesettelten Band-Kandidaten im Autobuy-Log:

| Menge | n | ROI |
|---|---|---|
| Preisband gesamt | 67 | +10,55 % |
| davon **gekauft** | 23 | **+3,76 %** |
| davon **liegengelassen** | 44 | **+14,10 %** |

Der Bot lässt den besseren Teil seines eigenen Bandes liegen. Zwei Kriterien
entscheiden darüber, welche Kandidaten er nimmt, und beide stehen unter Verdacht:

- **Der Temperaturabstand als Rangfolge.** Im OOS-Lauf konditional geprüft:
  innerhalb NO 0,70–0,85 laufen die Kandidaten nahe an der Bucket-Kante mit
  +13,11 %, die weit entfernten mit +11,42 %. Die Größe, nach der sortiert
  wird, trennt nicht — und wurde bei ihrer Einführung am 27.07. auch nur
  deshalb als Rang statt als Veto eingebaut, weil sie schon damals den ROI
  nicht hob.
- **Das Spannen-Veto.** Es verwarf im Zeitraum 21.–31.07. sechzehn
  Band-Kandidaten. Am 01.08. hat es zwei Verlierer verhindert; an den vier
  Tagen davor hat es Gewinner gekostet.

## Was hier NICHT behauptet wird

Nicht: „die 12 Live-Positionen zeigen, dass V2 kaputt ist." Vier Verlierer
tragen kein Urteil, und die Richtung — Auswahl schlechter als Grundgesamtheit —
ist bei einem informationslosen Rang das erwartete Verhalten, kein Beweis für
eines.

Nicht: „das Spannen-Veto gehört weg." Das Veto hat einen anderen Zweck als
Renditeoptimierung: es schützt vor einem korrumpierten µ (Beijing-Lehre vom
14.07., Memory `weather-low-min-calibration-blocker`). Ein Renditevergleich
kann diesen Zweck weder bestätigen noch widerlegen. Gefragt wird ausschließlich,
ob es **innerhalb des Bandes zusätzlich** trennt.

Nicht: „Breite ist gut, weil mehr Positionen mehr Gewinn bedeuten." Bei
identischem Erwartungswert je Position ist Breite nur eine Varianzfrage — und
sie kostet Kapital, das gebunden ist.

Sondern: **Steht das Band auf Daten, die es nicht gebaut haben? Und wenn ja:
schränken die beiden Auswahlkriterien den Bot ein, ohne etwas dafür zu
liefern?** Nur dann ist Weiten begründet.

---

## Was schon gesehen wurde — vollständige Offenlegung

Die Pre-Reg ist nicht blind. Bekannt sind beim Schreiben:

1. Fenster B (20.–26.07.) ist das Fenster, auf dem das Band gebaut wurde. Alle
   Zahlen daraus sind in-sample und gehen in **kein** Gate ein.
2. Fenster C (27.07.–01.08.) wurde heute vollständig ausgewertet — die Tabellen
   oben. Es geht in **kein** Gate ein und dient nur als Vorzeichenkontrolle.
3. Aus Fenster A ist die **Mengenstruktur** bekannt: 94 Kandidaten, **18
   Verlierer**, 9 Zieltage. Das ist eine Verliererquote von **19,1 %** gegen
   11,3 % in Fenster B. Die Break-even-Verliererquote liegt je nach Preis
   zwischen 20 % (bei NO 0,80) und 25 % (bei NO 0,75). **Damit ist vorab
   sichtbar, dass Fenster A dem Band nahe an die Nulllinie geht.** Diese Zahl
   hat die Gate-Schwellen unten mitgeformt und wird deshalb hier genannt, nicht
   nachträglich als Erklärung nachgereicht.
4. Nicht bekannt sind: die Preisverteilung innerhalb Fenster A, jeder ROI, jede
   Aufteilung nach Abstand, jede Tagesstruktur.

---

## Universum, Daten, Fenster

- **Kandidaten:** `bb_WeatherLadders`, `var='max'`, `kind='eq'`,
  `offset_fav = -1`, Settlement bekannt. **Entdupliziert auf (Zieltag, Stadt,
  k)** — die Leiter wird je Zieltag mehrfach geschrieben (Lead 0/1/2, dazu vier
  Testläufe am 26.07.); Zeilen sind keine Positionen. Rohzeilen 1058 → 542
  Positionen → 221 im Band → 184 mit Settlement.
- **Preis:** `buy_no`, gemittelt über die Zeilen desselben Schlüssels.
- **Abstand:** `mu_ens` minus Oberkante des gelayten Buckets, über
  `weather_stations.bucket_grenzen(k, city)` — nie selbst gerundet, wegen der
  floor-Buckets in Hong Kong (Memory `weather-hongkong-hko-buckets`).
- **Rechnung:** 5 $ je Lay, Gebühr `0,07 · n · min(NO, 1−NO)`, identisch zu
  `weather_minus1_shadow` und `weather_minus1_ppess_filter`, damit die Zahlen
  vergleichbar bleiben.
- **Signifikanz:** t über **Tages**-Mittel, nie über Einzelpositionen. Städte
  desselben Zieltags hängen an derselben Wetterlage.

| Fenster | Zeitraum | n | Zieltage | Rolle |
|---|---|---|---|---|
| **A** | 10.–19.07.2026 | 94 | 9 | **Holdout — trägt G1 und G2** |
| B | 20.–26.07.2026 | 53 | 7 | in-sample, kein Gate |
| C | 27.07.–01.08.2026 | 37 | 5 | heute ausgewertet, kein Gate |
| **D** | ab 03.08.2026 | offen | offen | **Forward — trägt G4** |

Fenster A liegt **vor** dem Bandbau. Das ist kein nachträglich gezogener Split,
sondern ein Zeitraum, in dem der Ladder-Logger bereits schrieb, während es das
Band noch nicht gab und der Autobuy noch nicht lief.

---

## Hypothesen

**H1 (Fundament):** Das Preisband 0,70–0,90 schlägt auf Fenster A die
Gesamtmenge der −1-Kandidaten desselben Fensters. Ohne H1 ist alles Weitere
gegenstandslos — dann wäre das Band ein Artefakt zweier Sommerwochen.

**H2 (Rang):** Der Temperaturabstand trennt **innerhalb** des Bandes nicht.
Formuliert als Beweislast beim Kriterium: Der Rang gilt als belegt, wenn die
obere Abstandshälfte die untere um **≥ 8 Prozentpunkte ROI** schlägt bei
**t > 2,0**. Andernfalls gilt er als **nicht belegt** und darf keine Auswahl
mehr steuern.

**H3 (Geldtest Breite):** Ein Buch, das alle Band-Kandidaten eines Zieltags bis
zu einem Cap kauft, liefert auf Fenster A einen ROI, der **nicht niedriger** ist
als das Buch mit heutiger enger Auswahlregel — nach Gebühr und nach der
gemessenen Slippage-Kurve.

**H4 (Spannen-Veto, nur Forward):** Innerhalb des Bandes trennt das Spannen-Veto
nicht. Retrospektiv **nicht testbar** — die Modellspanne steht nur im
Autobuy-Log, das erst ab dem 20.07. existiert, und dieses Fenster ist
verbraucht. Deshalb ausschließlich Forward (Fenster D).

**H0:** Das Band ist ein Artefakt der beiden Wochen, auf denen es gebaut wurde;
der Abstand trennt sehr wohl; die enge Auswahl ist besser als ihr Ruf. Dann
bleibt der Bot exakt wie er ist, und die 12 Live-Positionen waren schlicht ein
schlechter Lauf.

---

## Gates

| Gate | Bedingung |
|---|---|
| **G1** Fundament, Holdout | Auf Fenster A (n = 94, 9 Zieltage): Band-ROI **> 0** und mindestens **8 pp** über dem ROI aller −1-Kandidaten desselben Fensters, t > 1,5 über Tagesmittel |
| **G2** Robustheit des Fundaments | G1 überlebt das Streichen des besten Zieltags **und** der stärksten Einzelstadt; kein Zieltag trägt > 35 % des Effekts |
| **G3** Rang | H2 wie oben: Rang gilt nur als belegt bei ≥ 8 pp **und** t > 2,0. Genau **ein** Schnitt wird geprüft (Median des Abstands), keine Schwellensuche |
| **G4** Breite, Forward | Auf Fenster D, ab 03.08., mindestens 20 Zieltage: das breite Schattenbuch liegt nicht unter dem engen, t > 1,0. Erst danach darf der Bot geweitet werden |
| **G5** Kapital und Ausführung | Die breite Menge ist finanzierbar: Median 8 Kandidaten je Zieltag × 2 überlappende Tage × Einsatz ≤ freies Guthaben, und die Slippage-Kurve bleibt im Bereich bis 100 $ je Bucket (Memory `weather-scaling-plan`) |

**Bonferroni:** Vier Hypothesen, aber nur H1/H2/H3 werden retrospektiv geprüft.
Die t-Schwellen oben sind bereits angehoben (1,5 statt 1,0 für ein einzelnes
Fenster mit 9 Tagen; 2,0 für den Rang, der gegen die bestehende Regel antritt).
Es werden **keine** weiteren Preisbänder, Abstandsschwellen oder Caps probiert.

**Warum G1 zwei Bedingungen hat:** Ein Band-ROI knapp über null bei einem
Gesamt-ROI von deutlich unter null wäre formal ein „Vorteil", aber ökonomisch
wertlos — wir würden Kapital binden, um nicht zu verlieren. Der Abstand von 8 pp
ist die Größenordnung, die die bisherigen Messungen behaupten (+12,5 gegen
−6,7 = 19 pp); die Hälfte davon zu verlangen, ist keine hohe Hürde und trotzdem
eine echte.

---

## Designfallen, die diesen Test definieren

**1. Der Preis ist nicht der Fill.** `buy_no` ist der Vortagspreis aus dem
Ladder-Snapshot, nicht der Preis, zu dem gekauft wurde. Im Live-Review beträgt
die Differenz im Mittel 0,0001–0,0040. Das ist klein, aber es verschiebt
Kandidaten **über die Bandgrenze**: ein Snapshot-Preis von 0,899 kann live 0,901
sein. Die Bandzugehörigkeit wird deshalb über den Snapshot-Preis entschieden und
das Ergebnis mit einer Grenzverschiebung von ±0,01 auf Stabilität geprüft.

**2. Fenster A hat mehr Kandidaten je Tag als heute** (Median 14 gegen 6 in
Fenster C). Das ist keine Zufälligkeit: Mitte Juli lief eine Hitzewelle über
Europa und Asien, breite Verteilungen erzeugen mehr Kandidaten im mittleren
Preisband. **Ein Band, das nur in unruhigen Lagen trägt, ist kein Band.**
Deshalb G2 mit Städte- und Tages-Streichung.

**3. Das Überlappungsproblem.** Ein Zieltag bindet Kapital über zwei
Kalendertage (Kauf 14:45 am Vortag, Settlement am Zieltag abends). Ein
Schattenbuch, das jeden Tag frisch 8 Positionen eröffnet, unterstellt doppeltes
Kapital. G5 rechnet deshalb mit zwei überlappenden Tagen, nicht mit einem.

**4. Die Settlement-Quelle ist ungeklärt.** Am 02.08. wurde gemessen, dass
Markt und WU-Referenz an 5 von 319 Stadt-Tagen auseinanderliegen und der Markt
in **allen fünf** Fällen METAR folgte, nie WU (Shenzhen 4×, Seoul 1×). Betroffen
ist auch Buenos Aires am 01.08. Für diesen Test gilt: **gewertet wird
`settle_result`, also die Markt-Auflösung** — das ist das Geld. Wo sie fehlt,
fällt der Kandidat heraus; es wird **nicht** auf WU ausgewichen.

**5. Ein informationsloses Kriterium ist nicht dasselbe wie ein schädliches.**
Wenn der Rang nicht trennt, kostet er nichts — er wählt dann zufällig aus einer
Menge mit gleichem Erwartungswert. Der Schaden entsteht erst durch die
**Verkleinerung** der Menge. Deshalb ist H3 der eigentliche Geldtest und H2 nur
seine Voraussetzung.

---

## Vorab-Erwartung (damit sie nicht zurechtgebogen wird)

**G1 halte ich für gefährdet, und zwar wegen der 19,1 %.** Die Verliererquote in
Fenster A liegt in der Nähe der Break-even-Schwelle. Ob das Band dort positiv
ist, hängt an der Preisverteilung — bei durchschnittlich NO 0,80 wären 19,1 %
knapp gewinnend, bei 0,75 knapp verlierend. Ich rechne mit einem Ergebnis in der
Größenordnung ±5 %, also **unter** der geforderten Schwelle. Falls es so kommt,
ist die ehrliche Lesart: das Band trägt in ruhigen Lagen, nicht in der Hitzewelle
— und dann darf der Bot **nicht** geweitet werden.

**H2 erwarte ich als bestätigt** (Rang trennt nicht), weil das schon zweimal
unabhängig gemessen wurde: bei der Einführung am 27.07. und heute im OOS-Lauf.
Das ist die einzige Aussage hier, die ich für ziemlich sicher halte — und
zugleich die, die allein am wenigsten wert ist.

**H3 ist offen.** Breite hilft nur, wenn der Erwartungswert je Position beim
Weiten nicht sinkt. Das ist keineswegs ausgemacht: die heutige Auswahl ist zwar
schlechter als das Band, aber sie ist auch die teurere Hälfte, und teurer heißt
sicherer.

**Zur Größenordnung, damit niemand zu viel erwartet:** Selbst wenn alles hält,
verdoppelt sich die Zahl der Positionen von ~2,5 auf ~5 je Tag. Bei 5 $ Einsatz
und dem gemessenen Band-ROI wären das etwa 1,50 $ mehr pro Tag. Das ist kein
neuer Edge, sondern die Ausnutzung eines bereits gemessenen — und es ist genau
der Weg, den die Slippage-Messung vorgibt: Breite, nicht Größe.

## Abbruchregel

Reißt **G1**, ist das Band als Fundament nicht belegt. Dann wird **nicht**
geweitet, **nicht** auf ein anderes Preisband ausgewichen und **nicht** nach
einem Fenster gesucht, in dem es doch trägt. Der Bot bleibt exakt wie er ist,
und die Frage nach der Breite ist bis auf Weiteres beantwortet: nein.

Reißt nur **G2** (Fundament nur dank einzelner Tage/Städte), lautet der Befund
„das Band ist lagenabhängig". Dann wird es nicht geweitet, sondern als
konditionale These neu vorregistriert — mit dem Lagenkriterium **vorab**
definiert, nicht aus den Daten gelesen.

Besteht **G1/G2**, aber der Rang gilt nach **G3** als belegt, bleibt die
Rangfolge unverändert. Es wird dann nicht behauptet, sie sei „trotzdem
verzichtbar".

Bestehen G1–G3, geht die breite Auswahl **nicht** sofort live, sondern zuerst
als **Schattenbuch** (Fenster D, G4, mindestens 20 Zieltage). Der Bot handelt
in dieser Zeit unverändert weiter. Die V2-Erfahrung — ohne Forward-Test live,
mit ausdrücklich deklarierter Abweichung von der Projektmethodik — wird nicht
wiederholt.

Für **H4** gibt es keinen retrospektiven Lauf. Wer das Spannen-Veto vor
Abschluss von Fenster D anfasst, tut es ohne Beleg.

---

# ERGEBNIS — G1 gerissen, es wird nicht geweitet (02.08.2026)

Gerechnet mit `weather_band_breite_eval.py` am 02.08.2026, unmittelbar nach der
Vorregistrierung (Commit `c921421`), ohne jede Parameteränderung.

Fenster A: 421 Rohzeilen → 225 entduplizierte Positionen → 94 im Band, 9 Zieltage.

## G1 — gerissen an der absoluten Bedingung

| Menge | n | PnL | ROI | Verliererquote |
|---|---|---|---|---|
| Preisband 0,70–0,90 | 94 | −8,16 $ | **−1,74 %** | 19,1 % |
| alle −1-Kandidaten | 225 | −111,75 $ | **−9,93 %** | 24,9 % |

Vorteil **+8,20 pp**, Tagesmittel +11,52 pp, **t = +1,80** über 9 Tage.

Die *relative* Bedingung ist erfüllt (≥ 8 pp) und die t-Schwelle auch (> 1,5).
Gerissen ist **ROI > 0**. Genau dieser Fall war beim Formulieren des Gates
benannt: *„Ein Band-ROI knapp über null bei einem Gesamt-ROI von deutlich unter
null wäre formal ein ‚Vorteil', aber ökonomisch wertlos."* Hier liegt er sogar
darunter. Das Band ist auf Daten, die es nicht gebaut haben, **Schadensbegrenzung
statt Ertrag**.

Die Vorab-Erwartung — „G1 dürfte reißen, wegen der 19,1 %" — trifft, und zwar aus
exakt dem genannten Grund: 19,1 % Verliererquote liegen im Bereich der
Break-even-Schwelle, und der mittlere Fill-Preis reichte nicht, um sie zu tragen.

## G2, G3, H3, G5

- **G2 gerissen**, folgt aus G1: ohne den besten Zieltag −2,11 %, ohne die
  stärkste Stadt (München) −3,63 %. Die Kennzahl „Anteil des besten Tages" ist
  bei negativem Gesamteffekt nicht sinnvoll definiert (sie kommt als −20,0 %
  heraus) — ein Mangel des Eval-Codes, der hier folgenlos bleibt, weil G1 die
  Abbruchregel schon ausgelöst hat.
- **G3: Rang NICHT belegt.** Median-Schnitt bei +0,43 K: nah an der Kante
  −7,55 % (12/47 Verlierer), weit von der Kante +4,08 % (6/47). Unterschied
  +11,63 pp, aber **t = +0,71**. Damit ist die 8-pp-Bedingung erfüllt, die
  t-Bedingung (> 2,0) nicht.
  **Der Punktschätzer zeigt hier in die vom Bot unterstellte Richtung** — in
  Fenster C zeigte er in die Gegenrichtung. Zwei Fenster, zwei Vorzeichen, kein
  t: das ist Rauschen. Der Rang bleibt unbelegt, aber er ist damit auch nicht
  widerlegt; er wird nicht angefasst.
- **H3: +3,29 pp, t = +0,39.** Breit (62 Lays, −1,59 %) gegen eng (25 Lays,
  −4,88 %) — kein Unterschied, der etwas trägt.
- **G5:** Median 10 Kandidaten je Zieltag, Maximum 18. Bei 5 $ und zwei
  überlappenden Tagen wären 100 $ gebunden, in der Spitze 180 $ — gegen zuletzt
  rund 93 $ freies Guthaben. Die breite Menge wäre **ohnehin nicht finanzierbar**
  gewesen.

## Der Befund, der wichtiger ist als das Gate

Diagnostisch nachgerechnet (vier Teilfenster, **post hoc, kein Gate**):

| Teilfenster | Band | Verlierer | alle −1-Kandidaten | Verlierer |
|---|---|---|---|---|
| 10.–13.07. | +10,29 % | 10,0 % | −15,81 % | 26,3 % |
| **14.–19.07.** | **−4,99 %** | **21,6 %** | −7,94 % | 24,4 % |
| 20.–26.07. | +8,77 % | 11,3 % | −6,31 % | 21,8 % |
| 27.07.–01.08. | +18,07 % | 5,4 % | −11,63 % | 24,8 % |

**Die −1-Klasse verliert als Ganzes, durchgehend** — in allen vier Teilfenstern,
zwischen −6,31 % und −15,81 %, über 525 entduplizierte Kandidaten. Der Bot
handelt eine strukturell verlierende Grundgesamtheit und rettet sich über einen
Filter, dessen Ertrag zwischen −4,99 % und +18,07 % schwankt. Das erklärt die
dünne Marge der Live-Serie (+2,51 % am 01.08., +0,66 % am 02.08.) besser als
jede Einzelposition.

Der *relative* Vorteil des Bandes ist dagegen in allen vier Teilfenstern positiv
(+26,10 / +2,95 / +15,09 / +29,71 pp). Das Band **sortiert** zuverlässig; ob es
**verdient**, hängt am Umfeld.

**Eine Vermutung ist dabei widerlegt worden.** Beim ersten Blick auf das
G1-Ergebnis lag nahe, Fenster A sei durch die Screen-Verbesserungen vom 14.07.
(Doppel-Kalibrierung im Code erzwungen) und 17.07. (Debias-vor-Mittelung,
WU-Kalibrierung) verzerrt. Das Gegenteil ist der Fall: *vor* diesen Änderungen
war das Band mit +10,29 % gut, das einzige negative Teilfenster liegt *danach*.
Die Erklärung ist damit erledigt und wird nicht weiterverfolgt.

## Was das heißt — die Abbruchregel greift

**Es wird nicht geweitet.** Nicht auf ein anderes Preisband ausgewichen, nicht
nach einem Fenster gesucht, in dem es doch trägt, keine Schwellensuche am
Abstand. Der Autobuy bleibt exakt wie er ist: Preisband 0,70–0,90, Rang nach
Temperaturabstand, Spannen-Veto, Cap 8, 5 $.

**V2 ist damit nicht falsifiziert, aber sein Fundament ist schwächer als
angenommen.** Die Live-Auswahl schlägt die Grundgesamtheit deutlich (+3,76 %
gegen −9,93 %); was nicht belegt ist, ist die Behauptung, das Preisband allein
liefere einen verlässlich positiven Ertrag.

**H4 (Spannen-Veto) bleibt offen und ausschließlich Forward.** Fenster D beginnt
unverändert am 03.08.; G4 verlangt mindestens 20 Zieltage.

## Was nicht geprüft und bewusst offen ist

- **Warum die −1-Klasse als Ganzes verliert, ist nicht gemessen.** Der Verdacht
  liegt bei der Bucket-Wahl über `mu_ens` (Fehlerquelle 2 der Studie vom 14.07.,
  0,6 K in µ), nicht beim Preis. Das ist eine eigene These und braucht eine
  eigene Vorregistrierung — hier wurde sie weder getestet noch gestreift.
- **Neun Zieltage tragen kein t-Argument.** G1 wäre auch bei ROI > 0 mit t = 1,80
  ein schwacher Beleg gewesen. Der Test taugt als Falsifikation, nicht als
  Bestätigung — dieselbe Asymmetrie wie beim konditionalen Ausstieg.
- **Das enge Buch in H3 ist ohne Spannen-Veto nachgebildet**, weil die
  Modellspanne erst ab dem 20.07. existiert. Es ist damit eher zu gut als zu
  schlecht dargestellt.
- **Die Bandgrenzen wurden nicht auf ±0,01 verschoben**, obwohl die Pre-Reg das
  als Stabilitätsprobe vorsah (`--grenzen`). Bei gerissenem G1 hätte die Probe
  nur zeigen können, ob das Minus etwas größer oder kleiner wird; sie wurde
  bewusst unterlassen, um nicht doch noch nach einer Grenze zu suchen, bei der
  es trägt.
- **Fenster A endet am 19.07.** Ob die −1-Klasse vor dem 10.07. anders aussah,
  ist nicht erhoben — der Ladder-Logger reicht nicht weiter zurück.
