# Pre-Registrierung: Konditionaler Ausstieg bei beobachtetem Bucket — 2026-08-01

**Status:** Vorregistrierung. Geschrieben, **bevor** ein Ausstiegs-PnL, eine
Erholungsrate oder irgendeine Ertragskennzahl gerechnet wurde. Die Sondierung
sah nur Mengen an (Zahl der Kandidaten, Zahl der Verlierer) sowie **einen**
vollständig ausgewerteten Einzelfall, der unten als Anlass offengelegt ist.

Auswertung folgt in `weather_conditional_exit_eval.py`.

---

## Anlass

Am 01.08.2026 wurde gemessen, dass Halten die von Hand getätigten Ausstiege
schlägt: gehalten **+3,82 %** (n = 70) gegen vorzeitig verkauft **−8,00 %**
(n = 9); zwei der drei Verlustausstiege waren in Wahrheit gewinnende
Positionen. Daraus wurde die Doktrin „keine Verlustausstiege von Hand"
abgeleitet (Memory `weather-no-manual-exits`).

Am selben Tag hielt der Betreiber dieser Doktrin den laufenden Chengdu-Lay
entgegen. Der Fall im Detail (Ist-Max 27 °C, Hoch durch):

| Position | Einstieg | Ausgang |
|---|---|---|
| NO 26 °C | 0,78 | gewinnt, +1,26 $ |
| NO 27 °C | 0,70 | verliert, −4,76 $ |

Preisverlauf des NO 27 gegen die Beobachtung (483 Trades im Polymarket-Tape):

```
12:00 lokal   Ist 25    NO 0,500     Markt preist 50 %, Beobachtung sagt nichts
13:00 lokal   Ist 26    NO 0,303
14:00 lokal   Ist 27    NO 0,320     Bucket ist beobachtet
15:00 lokal   Ist 26    NO 0,384     erholt sich nochmal
16:00 lokal   Ist 27    NO 0,040     erledigt
```

Ein Ausstieg um 14:00, in dem Moment, in dem WU den 27er meldet, hätte rund
2,10 $ zurückgebracht — Verlust **−2,66 $ statt −4,76 $**.

**Der Einwand hält also, und die Messung deckt ihn nicht ab.** Sie hat
„Ausstieg" als *eine* Kategorie behandelt. Er zerfällt in zwei:

- **uninformiert** — der Preis läuft weg, die These ist noch offen. In Chengdu
  wäre das der Ausstieg um 12:00 lokal zu 0,50 gewesen, bei einem Ist-Wert von
  25 und intakter Prognose. Das ist die gemessene Klasse, sie kostet −8,00 %.
- **informiert** — die settelnde Quelle hat den gelayten Bucket beobachtet, die
  These ist damit tot oder fast tot. **Nie gemessen.**

Diese Pre-Reg misst die zweite Klasse.

## Was hier NICHT behauptet wird

Nicht: „Chengdu zeigt, dass Ausstiege helfen." Das ist ein Fall, nachträglich
angesehen, mit bekanntem Ausgang — genau die Beweisform, die in diesem Repo
regelmäßig an G2 stirbt.

Nicht: eine Aufweichung der Doktrin. Die Regel unten ist **mechanisch und
beobachtungsgetrieben**; sie gibt keinerlei Ermessen zurück. Reagiert wird
ausschließlich auf eine Messung der Settlement-Quelle, nie auf einen Preis, nie
auf einen Buchverlust.

Sondern: **Ist ein Bucket, der einmal beobachtet wurde, so tot, wie der Preis in
diesem Moment behauptet — oder toter?** Nur wenn er toter ist, lohnt der
Ausstieg.

---

## Die Regel, exakt

Vorab fixiert, damit sie hinterher nicht gebogen wird.

> **Verkaufe die volle NO-Position, sobald die settelnde Quelle für den Zieltag
> einen Wert meldet, dessen Rundung dem gelayten Bucket k entspricht.**

- Auslöser ist `round(temp) == k` in einer WU-History-**Tabellenzeile** (nicht
  der Kachel — Memory `weather-settlement-wu-vs-metar`).
- Maßgeblich ist `valid_time_gmt`, nie eine lokale Anzeigezeit (Memory
  `weather-tile-vs-table-latency`).
- Ausgeführt wird zum **nächsten** im Tape belegten Preis nach dem
  Sichtzeitpunkt (Definition siehe Falle 1), nicht zum Preis des
  Beobachtungszeitpunkts.
- Kein Teilverkauf, kein Nachkauf, kein Wiedereinstieg.

**Variante A (primär):** ohne Zeitfilter, feuert bei der ersten Beobachtung.
**Variante B (sekundär):** feuert nur ab 16:20 Ortszeit, analog zur bestehenden
Tageshoch-Regel (Memory `weather-daily-max-timing`).

Zwei Varianten, **zwei Tests** — Bonferroni, Gate-Schwelle t > 2,5. Es werden
keine weiteren Zeitschwellen, Preisbänder oder Teilverkaufsquoten probiert.

---

## Hypothese

**H1 (primär):** Die bedingte Erholungsrate — P(Endmax > k | k wurde beobachtet)
— liegt **unter** der vom Markt zum Ausstiegszeitpunkt eingepreisten Rate. In
Chengdu preiste der Markt 32 % ein; erholt hat sich nichts.

**H2 (der Geldtest):** Über alle Kandidaten liefert das Buch mit konditionalem
Ausstieg einen höheren ROI je Einsatz als das reine Halte-Buch, **netto nach
der zweiten Gebühr**.

**H3 (Verträglichkeit):** Die Regel feuert selten genug, dass sie das Buch nicht
umbaut — sie greift bei unter 25 % der Positionen.

**H0:** Der Markt preist die Erholungschance korrekt oder zu niedrig. Dann kostet
jeder Ausstieg im Mittel Geld, die Doktrin „halten" bleibt unverändert
bestehen, und der Chengdu-Fall war Rauschen.

### Universum, Daten, Split

- **Kandidaten:** `bb_WeatherLadders`, `var='max'`, `kind='eq'`,
  `offset_fav=-1`, Preisband `0,70 <= buy_no < 0,90`, Settlement bekannt.
  Gemessen am 01.08.: **n = 330**, davon **61 verloren** (`wu_settle_k = k`).
  Das ist das kontrafaktische Universum der −1-Klasse, nicht nur das eigene
  Buch — 79 eigene Positionen mit ~8 Verlierern tragen keinen Test.
- **Preise:** Polymarket-Tape (`data-api.polymarket.com/trades`), minutengenau,
  frei, bereits in `weather_mm_spread_test.py` erschlossen.
- **Beobachtungen:** WU-History-Tabelle je ICAO und Zieltag,
  `weather_foreknowledge_eval.wu_observations`.
- **Zeitraum:** 2026-07-10 bis 2026-08-02 — **24 Tage, 39 Städte, reiner
  Sommer.**

**Zum Split, offen benannt:** 24 Tage tragen keinen ehrlichen IS/OOS-Schnitt.
Ein Zeitsplit gäbe zwei Fenster von je zwölf Tagen aus derselben Wetterlage;
ein Städtesplit vergleicht Klimaregime statt Zeitpunkte. Deshalb gilt hier:

> **Der retrospektive Lauf ist G1 — er kann die These nur töten, nicht
> bestätigen.** Entscheidend ist der Forward-Test (G2), gegen den bereits
> laufenden Ensemble-Forward-Test gestaffelt, Auswertung frühestens Oktober.

---

## Gates

| Gate | Bedingung |
|---|---|
| **G1** Retrospektiv, Existenz | Über n ≥ 250 auswertbare Kandidaten: Erholungsrate liegt mindestens 5 Prozentpunkte unter der eingepreisten, t > 2,5 (Bonferroni, zwei Varianten) |
| **G2** Forward, Kern | Auf ab dem 02.08. vorregistrierten Kandidaten, ohne jede Parameteränderung: dasselbe Vorzeichen, t > 1,5 |
| **G3** Netto nach Gebühr | ROI-Vorteil überlebt die **zweite** Gebühr von 3,6 % auf min(p, 1−p) je Ausstieg, plus die Geld-Seite statt des Mittelkurses |
| **G4** Ausführbarkeit | Bei ≥ 80 % der Auslösungen existiert innerhalb von 30 min nach dem Sichtzeitpunkt ein Tape-Trade. Kandidaten ohne Handel zählen als **nicht ausgestiegen**, nie als Ausstieg zum letzten bekannten Preis |
| **G5** Robustheit | Gilt nach Streichen der größten Einzelstadt; kein einzelner Zieltag trägt > 30 % des Effekts; gilt für Variante A und B mit gleichem Vorzeichen |

**Warum G3 die zweite Gebühr betont:** Halten bis Settlement endet
gebührenfrei — das ist der Grund, warum die Lay-Doktrin überhaupt
gebührenoptimal ist (Memory `weather-market-making-idea`). Jeder Ausstieg fügt
eine Gebühr hinzu, die das Halten nicht hat. Die Regel startet also mit
strukturellem Rückstand, und der muss im Ergebnis stehen, nicht in einer
Fußnote.

**Warum G4 so hart formuliert ist:** In den unteren Preisregionen ist der
Handel dünn. Ein Backtest, der zum „letzten bekannten Preis" ausführt,
unterstellt Liquidität, die es nicht gab, und erfindet damit genau die Rettung,
die er messen soll.

---

## Designfallen, die den Test definieren

**1. Latenz — die zentrale Falle.** Am 01.08. gemessen (168 Erstsichtungen):
WU hängt im Median **616 s** hinter der Beobachtung, an den europäischen
Brettern **EGLC 2108 s** und **LFPB 1980 s**. Ein Backtest, der auf
`valid_time_gmt` auslöst, unterstellt uns Wissen, das wir zu diesem Zeitpunkt
nicht hatten — und misst einen Vorsprung, der live nicht existiert.

> **Sichtzeitpunkt := `valid_time_gmt` + stationsspezifischer WU-Latenzaufschlag.**
> Verwendet wird das **obere** Ende der Sondenschätzung, nicht der Median. Die
> Sonde umfasst eine Nacht und sechs Stationen; wo keine Stationsschätzung
> vorliegt, gilt pauschal **1800 s**.

Das ist bewusst pessimistisch. Ein Effekt, der nur bei optimistischer Latenz
überlebt, ist kein Effekt.

**2. Der Test ist NICHT trivial gewinnend.** Ein beobachteter Tagesmax-Bucket
ist nicht erledigt: das Maximum kann weiter steigen, dann gewinnt das NO doch.
In Chengdu stand das Thermometer um 14:00 auf 27, um 15:00 wieder auf 26 — bei
28 °C hätte die Position gewonnen und der Ausstieg zu 0,32 wäre ein Fehler
gewesen. Genau diese Fälle sind der Preis der Regel, und ein einziger frisst
rund zwei gerettete.

**3. Selektion beim Zählen.** Die 61 bekannten Verlierer sind **nicht** die
Menge der Auslösungen. Die Regel feuert bei jedem Kandidaten, dessen Bucket
irgendwann berührt wurde — auch bei später überschrittenen, also bei Gewinnern.
Diese Zahl ist vorab **nicht** erhoben worden; sie ist Teil des Ergebnisses.

**4. Geld- gegen Briefseite.** Verkaufen heißt die Geld-Seite nehmen; der
Tape-Preis ist ein gehandelter Preis, nicht der, den wir bekommen hätten
(Memory `weather-screen-price-vs-book`). Wo Buchdaten fehlen, wird der
Tape-Preis um den gemessenen effektiven Spread von 1,0 ct zu unseren Ungunsten
verschoben.

---

## Vorab-Erwartung (damit sie nicht zurechtgebogen wird)

**Ich erwarte, dass H1 knapp scheitert oder bestenfalls knapp besteht.** Der
Grund steht im eigenen Repo: In der Nachbeobachtungsphase ist der Markt
nachweislich gut. Bei Uneinigkeit trifft der Markt-Favorit 46,8 % gegen unsere
21,8 % (Memory `weather-markt-schlaegt-eigenen-favoriten`), und die
Foreknowledge-Messung hat gezeigt, dass der Markt sehr präzise weiß, was **nicht
mehr** kommt (t = +3,02/+3,25 ab 15 h, Memory
`weather-market-foreknowledge-question`). Ausgerechnet gegen diese Stärke setzt
die Regel.

**G4 halte ich für am gefährdetsten**, nicht G1. Ein NO, dessen Bucket gerade
beobachtet wurde, handelt bei 0,03–0,30; dort ist das Buch dünn. Wenn die Regel
scheitert, dann vermutlich daran, dass zum Ausstiegszeitpunkt niemand kauft.

**Variante B wird weniger retten als A**, weil der Preis um 16:20 meist schon
kollabiert ist — dafür wird sie seltener falsch liegen. Ob der Saldo positiv
ist, ist offen; ich habe dazu keine Erwartung und will auch keine bilden.

**Zur Größenordnung:** Selbst voller Erfolg bewegt wenig. 61 Verlierer auf 330
Kandidaten, davon vielleicht die Hälfte rettbar, je ~2 $ — gegen ein Buch, das
insgesamt +10,16 $ steht. Die Regel ist eine Leckabdichtung, kein Edge, und
darf im Erfolgsfall nicht als solcher verkauft werden.

## Kopplung an die offene NOAA-Frage

Diese Pre-Reg macht die Screen-Frage **geldwert und damit entscheidbar**. Die
Regel lebt von der Reaktionszeit: NOAA sieht die Beobachtung im Median 390 s
früher als WU, an EGLC/LFPB rund 1800 s früher. Besteht die Regel, ist der
Wechsel der Live-Screens von WU auf NOAA kein Komfortthema mehr, sondern
bezifferbar — die Differenz lässt sich direkt als zweiter Lauf mit
NOAA-Latenzaufschlag messen.

**Settlement bleibt in jedem Fall WU.** Es geht ausschließlich darum, worauf wir
schauen, nie darum, woran abgerechnet wird.

## Abbruchregel

Reißt **G1**, ist die These falsifiziert: der informierte Ausstieg ist dann
genauso wertlos wie der uninformierte, die Doktrin „halten" gilt unverändert
und in voller Härte. Es wird **nicht** auf andere Auslöser ausgewichen —
weder auf Nachbarstationen, noch auf Modell-Updates, noch auf Preisschwellen —
bis einer hält.

Reißt nur **G4**, lautet der Befund „die Regel ist richtig, aber nicht
handelbar". Dann wird sie **nicht** trotzdem eingebaut und auch nicht mit einer
Limit-Order-Variante gerettet; sie wird als nicht ausführbar abgelegt.

Besteht sie, geht sie **nicht** sofort live, sondern zuerst als Schattenlauf in
`weather_minus1_autobuy.py` — protokollieren, was sie getan hätte, ohne zu
handeln. Die V2-Erfahrung (ohne Forward-Test live, deklarierte Abweichung von
der Projektmethodik) wird hier nicht wiederholt.

---

# ERGEBNIS — G1 gerissen, These falsifiziert (01.08.2026)

Gerechnet mit `weather_conditional_exit_eval.py` am 01.08.2026, wenige Stunden
nach der Vorregistrierung. Keine Parameteränderung gegenüber dem Text oben.

## Zwei Präzisierungen, die vor dem Urteil offengelegt gehören

**1. Der Kandidatenbegriff war falsch gezählt.** Oben steht n = 330 / 61
Verlierer. Das ist die Zahl der **Zeilen** in `bb_WeatherLadders` — die Leiter
wird je Zieltag mehrfach geschrieben (Lead 0/1/2, dazu vier Testläufe am
26.07.). Eine Position ist aber ein (Stadt, Zieltag, k)-Tripel. Entdupliziert
bleiben **223 Kandidaten / 47 Verlierer**, davon 221 auswertbar (Hong Kong hat
keine WU-Reihe). Damit ist die Mengenbedingung von G1 (n ≥ 250) **nicht
erfüllbar**, bevor überhaupt ein Effekt gemessen ist. Das wird hier ausgewiesen
und nicht stillschweigend abgesenkt.

**2. Der Auslöser war unterspezifiziert.** Wörtlich steht oben „`round(temp) ==
k` in einer Tabellenzeile". Das trifft auch die Zeilen auf dem *Weg nach oben*:
bei k = 25 und einem Tagesgang 24 → 25 → 26 → 27 feuert die Regel um 10 Uhr
morgens, obwohl die Position danach sicher gewinnt. Gemessen feuert diese
Lesart bei **90 %** aller Kandidaten, und **155 von 199** Auslösungen sind
spätere Gewinner. Gemeint ist erkennbar das **laufende Tagesmaximum** — H1
fragt nach `P(Endmax > k | k beobachtet)`, was nur Sinn hat, wenn k das
bisherige Hoch ist; der Anlassfall Chengdu ist ein monotoner Anstieg, dort
fallen beide Lesarten zusammen; und der Wächter vom 24.07., auf den die Pre-Reg
verweist, ist bereits so definiert. Gewertet wird deshalb „laufendes Max
erreicht k"; die wörtliche Fassung läuft als Kontrolle mit und ändert am
Ergebnis nichts (letzte Zeile der Tabelle).

**Variante B ist entsprechend eine Prüfung zur Ortsstunde**, keine verzögerte
Reaktion auf ein Ereignis von morgens: sitzt das bis 16:20 **sichtbare**
Tagesmaximum noch auf k? Steht es höher, ist die Position sicher und es gibt
nichts zu verkaufen.

## Die Zahlen

221 auswertbare Kandidaten, 11.–31.07.2026, 29 Städte.

| Lauf | feuert | eingepreist | realisiert | Differenz | t | ROI ggü. Halten |
|---|---|---|---|---|---|---|
| **A / WU** | 81,4 % | 0,745 | 0,758 | **−1,3 pp** | −0,50 | **−2,64 pp** |
| **B / WU** | 26,7 % | 0,396 | 0,421 | **−2,5 pp** | −0,51 | **−1,36 pp** |
| A / NOAA | 81,4 % | 0,714 | 0,754 | −4,0 pp | −1,48 | −5,51 pp |
| B / NOAA | 25,3 % | 0,315 | 0,321 | −0,6 pp | −0,12 | −0,68 pp |
| A / ohne Latenz | 81,4 % | 0,717 | 0,757 | −4,0 pp | −1,45 | −5,56 pp |
| B / ohne Latenz | 24,9 % | 0,293 | 0,291 | +0,2 pp | +0,04 | −0,37 pp |
| A / WU / wörtlich | 90,0 % | 0,755 | 0,768 | −1,3 pp | −0,50 | −2,68 pp |

**H1 ist in jeder Fassung gerissen.** Die realisierte Erholungsrate liegt
nirgends unter der eingepreisten; in sechs von sieben Läufen liegt sie
**darüber**. Verlangt waren mindestens 5 Prozentpunkte darunter bei t > 2,5.

**H2 ebenfalls:** der ROI-Vorteil ist in allen sieben Läufen negativ, netto nach
der zweiten Gebühr und dem Spread.

**H3 gerissen:** die Regel feuert bei 25–81 % statt unter 25 % — sie ist keine
punktuelle Leckabdichtung, sondern baut das Buch um.

**G4 als einziges bestanden** (97–100 % handelbar). Die Vorab-Erwartung, G4 sei
am gefährdetsten, war falsch: das Buch ist auch in den billigen Bändern dicht
genug. Gescheitert ist der Kern, nicht die Ausführung.

## Die Rechnung hinter dem Urteil

|  | Verlierer gerettet | Gewinner gekappt | Saldo |
|---|---|---|---|
| Variante A | 43 → **+24,72** Einsätze | 135 → **−30,55** | **−5,82** |
| Variante B | 33 → **+7,42** Einsätze | 24 → **−10,42** | **−3,00** |

Das ist exakt die vorab benannte Mechanik: *„ein einziger [gekappter Gewinner]
frisst rund zwei gerettete."* Bei Variante B rettet die Regel sogar mehr
Positionen als sie kappt (33 gegen 24) und verliert trotzdem — weil ein
gekappter Gewinner rund 0,60 pro Kontrakt kostet, eine gerettete Position aber
nur 0,40 einbringt.

## Antwort auf die NOAA-Frage

**Über diesen Kanal ist der Screenwechsel wertlos.** Schneller sehen macht die
Regel nicht besser, sondern bei Variante A messbar **schlechter** (−5,51 gegen
−2,64 pp). Der Grund ist unangenehm einleuchtend: wer sofort verkauft, verkauft
im Moment der frischen schlechten Nachricht, also zum tiefsten Preis. Die
WU-Verzögerung von rund 30 Minuten sammelt derweil Information ein — in dieser
Zeit steigt das Maximum oft schon weiter, der Preis erholt sich, und der
Ausstieg fällt günstiger aus. Die Latenz ist hier kein Nachteil, sondern eine
zufällige Wartefrist.

Und der Diagnoselauf **ohne jeden Latenzaufschlag** — perfekte Information, live
unerreichbar — findet den Effekt genauso wenig (+0,2 pp, t = +0,04 bei
Variante B). Es liegt also nicht an der Quelle. Es gibt nichts zu holen.

Das entscheidet die NOAA-Frage **nicht** generell: gemessen ist nur dieser eine
Verwertungskanal. Wer für einen anderen Zweck auf NOAA umstellen will, braucht
ein eigenes Argument.

## Was das für die Doktrin heißt

Die Abbruchregel greift wörtlich: G1 gerissen ⇒ **der informierte Ausstieg ist
genauso wertlos wie der uninformierte.** Die Doktrin „keine Verlustausstiege von
Hand" gilt unverändert und in voller Härte. Der Einwand des Betreibers am
Chengdu-Fall war berechtigt formuliert und ist jetzt beantwortet: der Fall war
Rauschen, kein übersehener Kanal. Es wird **nicht** auf andere Auslöser
ausgewichen — weder Nachbarstationen noch Modell-Updates noch Preisschwellen.

Kein Forward-Test (G2), kein Schattenlauf, kein Einbau in
`weather_minus1_autobuy.py`.

## Was nicht geprüft und bewusst offen ist

- **Die Mengenbedingung G1a ist verletzt** (221 statt 250). Untermacht wäre ein
  ernster Einwand, wenn der Punktschätzer positiv und nur knapp wäre. Er ist in
  sechs von sieben Läufen **negativ**; mehr Daten drehen kein Vorzeichen. Der
  Test ist damit als Falsifikation belastbar, als Bestätigung wäre er es nicht
  gewesen.
- **Ein Sommer, 21 Tage, 29 Städte.** Ob die Marktkalibrierung in der
  Übergangsjahreszeit ebenso gut ist, ist nicht gemessen.
- **Die Latenzsonde trägt eine Nacht und sechs Stationen**, davon nur zwei
  (EGLC, LFPB) in diesem Universum. Alle übrigen 27 Städte laufen auf dem
  Pauschalwert 1800 s. Der Diagnoselauf ohne Latenz begrenzt den Schaden dieser
  Unsicherheit: selbst am optimistischen Ende gibt es keinen Effekt.
- **Ausstiegspreise stammen aus dem Tape**, nicht aus dem Buch; angesetzt ist
  ein pauschaler Abschlag von 1,0 ct. Bei einer Regel, die verliert, verschiebt
  eine genauere Geld-Seite das Ergebnis nur weiter ins Minus.
- **Teilverkäufe, Limit-Order-Varianten und Wiedereinstiege** wurden nicht
  getestet — so vorregistriert und so eingehalten.
