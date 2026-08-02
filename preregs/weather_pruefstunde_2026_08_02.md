# Pre-Registrierung: Die Prüfstunde je Stadt — 2026-08-02

**Status:** Vorregistrierung. Geschrieben, **bevor** die Stundenkurve irgendeiner
Stadt außer den drei europäischen angesehen wurde. Der Zweck dieser Pre-Reg ist
ausdrücklich, **die Ableitungsregel vor den Zahlen festzulegen** — die
Vorgängermessung räumt selbst ein: *„Die Prüfstunde wurde nach Sichtung aller
Stunden gewählt"* (`weather_lay_guardrail_2026_07_24.md`, Vorbehalt 2). Genau das
soll sich nicht wiederholen.

Auswertung folgt in `weather_pruefstunde_eval.py`.

---

## Anlass

Am 02.08. wurde die Basisrate „das Tageshoch kommt noch" breit gemessen
(`weather_daily_max_timing_isd.py`, Commit `4ca01eb`): 4.375 Stadt-Tage, 26
Städte, zwei Sommer über NCEI ISD. Ergebnis: **die 16:20-Regel ist eine
europäische Regel.** Global kommt das Hoch um 16:20 Ortszeit nur noch in
**13,1 %** der Fälle statt der bisher angenommenen 41 %. Die Spreizung ist
gewaltig: **Madrid 71,3 % gegen Taipei, Tokyo und Tel Aviv je 0,6 %.**

Daran hängen zwei Dinge, die beide auf der schmalen Basis stehen:

1. **Die manuelle Handlungsregel** „vor 16:20 Ortszeit keine Lay-Position
   erhöhen" ([[weather-daily-max-timing]]). Sie entstand aus einem konkreten
   Verlust — am 24.07. wurde in Helsinki um 15:20 ein Rückfall 20 → 19 Grad als
   Gipfel gelesen und die Position von 4,69 $ auf 14,05 $ erhöht; das Hoch kam
   danach noch. Für Europa ist die Regel korrekt und wurde heute fast punktgenau
   reproduziert. **Für Taipei ist sie sinnlos** — dort ist um 16:20 praktisch
   jeder Tag entschieden, die Regel verbietet also eine Handlung, die längst
   sicher wäre.
2. **Der Wächter für −1-Lays**, dessen Trennschärfe am 24.07. mit „16:20: 67 %
   Treffer gegen 2 % Basisrate" beziffert wurde — auf denselben fünf
   europäischen Städten.

## Was hier NICHT behauptet wird

**Nicht: „der Wächter kommt in den Bot."** Er ist am 24.07. gemessen worden und
**schadet dem engen Buch** zu jeder Prüfstunde (Live-Auswahl: Halten +10,52 $,
Wächter 17:20 +6,38 $) — bei 7,5 % Verliererquote kappt er mehr Gewinner, als er
Verlierer rettet. Er ist die Voraussetzung für **Breite**, und Breite ist am
02.08. abgelehnt worden ([[weather-minus1-klasse-verliert]]: die Grundgesamtheit
verliert, mehr Kandidaten verdünnen den Ertrag). Der Wächter bleibt damit
**Vorrat**, kein Bauvorhaben.

**Nicht: „die 16:20-Regel war falsch."** Sie war richtig für die Städte, an denen
sie gemessen wurde. Falsch ist ihre Übertragung auf alles andere.

**Nicht: „hier wird ein Geldtest gerechnet."** Der Ausstiegspreis-Teil der
24.07.-Messung wird **nicht** wiederholt. Ohne Breite gibt es nichts zu verdienen,
und ein Geldtest ohne Verwertungsweg ist eine Einladung zum Weitersuchen.

Sondern: **Welche Prüfstunde gilt je Stadt, abgeleitet nach einer vorab
festgelegten Regel — und trägt das Signal dort überhaupt noch, wenn die Stunde
stimmt?**

---

## Der eigentliche Nutzen, ehrlich benannt

Der Wächter liegt auf Eis. Was **sofort** wirkt, ist die manuelle Regel: Der
Betreiber handelt in Hongkong, Chengdu, Seoul, Taipei und Tokio — also genau
dort, wo die europäische Stunde am weitesten danebenliegt. Der Fehler wirkt in
beide Richtungen:

- **In Europa zu früh handeln** kostet Geld (der Helsinki-Fall).
- **In Asien zu spät handeln** kostet Gelegenheit: wer bis 16:20 Ortszeit wartet,
  obwohl der Tag um 14:00 entschieden ist, verkauft in einen Markt, der die
  Information längst hat ([[weather-market-foreknowledge-question]]: der Markt
  weiß nicht, was kommt, sondern was **nicht mehr** kommt).

**Das ist der Ertrag dieser Pre-Reg — eine korrigierte Handlungsregel, kein
neuer Edge.** Wer mehr erwartet, wird enttäuscht.

---

## Die Ableitungsregel — das Herzstück, vorab und unverhandelbar

**Prüfstunde(Stadt) = früheste Ortsstunde T des Rasters, für die gilt
P(Hoch kommt nach T | Stadt) ≤ q, mit q = 12 %.**

**Warum q = 12 % und nichts anderes:** Die Erstmessung nutzte 17:20 als
trennschärfste Stunde (88 % Treffer gegen 1 %). In Europa entspricht 17:20 einer
Restwahrscheinlichkeit von exakt **12 %**. Übertragen wird also **die
Restwahrscheinlichkeit, nicht die Uhrzeit** — die Größe, die den Wächter trug,
bleibt konstant, während die Stunde je Stadt wandert. Das ist eine Übertragung
aus vorhandener Arbeit, keine neue Wahl, und es ist der einzige Freiheitsgrad in
dieser Pre-Reg.

**Es wird genau ein q geprüft.** Kein Vergleich von 8 % gegen 12 % gegen 20 %,
keine Optimierung auf Trennschärfe. Fällt die Regel durch, fällt sie durch.

**Das Stundenraster wird vorab erweitert** auf **10:20 bis 20:20** in
Ein-Stunden-Schritten. Das bestehende Raster (13:20–18:20) reicht nicht: Taipei
liegt um 16:20 bei 0,6 %, seine Prüfstunde muss also deutlich früher liegen und
wäre im alten Raster gar nicht auffindbar. Die Erweiterung geschieht **hier und
vor der Messung**, nicht später, wenn eine Stadt keine Stunde findet.

**Randfälle, vorab entschieden:**
- Erfüllt **keine** Stunde des Rasters q, gilt die Stadt als **ohne Prüfstunde**
  und fällt aus der Auswertung — sie wird nicht mit 20:20 aufgefüllt.
- Erfüllt schon **10:20** die Bedingung, wird 10:20 vergeben und die Stadt
  gesondert ausgewiesen: dort ist das Signal vermutlich wertlos, weil der Markt
  zu dieser Stunde ohnehin nichts mehr zu entscheiden hat.

---

## Universum, Daten, Fenster

- **Basisraten:** NCEI ISD, `global-hourly`, dieselben Fallen wie im
  bestehenden Skript (nur FM-15/FM-16, Zehntelgrad mit Qualitätsflag, **gerundetes**
  Maximum, ≥ 12 Meldungen je Stadt-Tag, Nachmittagsabdeckung).
- **Saison:** Sommer, wie in der Basismessung. Die Prüfstunde wandert mit dem
  Sonnenstand — die Regel gilt ausdrücklich **nur für den Sommer**.
- **Split für die Stabilitätsprobe:** die beiden Sommer der Basismessung.
  **Sommer 1 leitet ab, Sommer 2 prüft.** Nicht umgekehrt, nicht gemittelt.
- **Signal (für die Trennschärfe):** gerundetes laufendes Tagesmaximum sitzt zur
  Prüfstunde **exakt auf dem gelayten Bucket** — identisch zur 24.07.-Definition,
  damit die Zahlen vergleichbar bleiben.
- **Zwei Quellen, vorab benannt:** Die Basisraten kommen aus **NCEI ISD** (zwei
  Sommer, historisch), die Intraday-Reihe für G4 aus **IEM ASOS** — dieselbe
  Quelle, aus der der Ladder-Logger settelt, und im Gegensatz zu ISD tagesaktuell
  genug für Juli 2026. Beide sind METAR, aber verschieden aufbereitet; die
  Prüfstunde wird deshalb **aus ISD abgeleitet und mit IEM angewendet**, nie
  vermischt. Weicht die IEM-Reihe an einem Stadt-Tag um mehr als 1 Grad vom
  geloggten `settle_k` ab, fällt der Tag aus G4 — das ist eine Datenprobe, keine
  Auswahl.
- **Kandidaten:** `bb_WeatherLadders`, `var='max'`, `kind='eq'`,
  `offset_fav=−1`, Lead 1, Settlement über `settle_k`. Zieltage 12.07.–01.08.
- **Ausgang:** verloren = `settle_k` gleich dem gelayten Bucket. **Keine Preise**,
  kein PnL — siehe „Was hier nicht behauptet wird".

---

## Hypothesen

**H1 (Relevanz):** Die abgeleitete Prüfstunde weicht bei **mindestens der Hälfte**
der Städte um **≥ 1 Stunde** von 16:20 ab. Ohne H1 ist die ganze Übung
gegenstandslos — dann wäre 16:20 zufällig doch global brauchbar.

**H2 (Stabilität):** Die aus Sommer 1 abgeleitete Prüfstunde stimmt in Sommer 2
auf **≤ 1 Stunde** überein, bei **≥ 80 %** der Städte. Eine Stunde, die zwischen
zwei Sommern springt, ist keine Regel, sondern eine Anekdote.

**H3 (Konsistenzprobe):** Für London, Paris und Madrid liefert die Regel eine
Prüfstunde im Bereich **17:20–18:20**. Diese drei reproduzieren die alte Messung
fast punktgenau; käme die Regel dort auf etwas anderes, wäre sie mit der Arbeit
unvereinbar, aus der ihr q stammt. **Das ist die schärfste Kontrolle dieser
Pre-Reg**, weil ihr Ergebnis vorhersagbar ist und ein Fehlschlag die Regel sofort
erledigt.

**H4 (Trennschärfe):** Mit stadtspezifischer Prüfstunde ist die Trennschärfe des
Wächter-Signals **nicht schlechter** als mit starrer 16:20-Prüfung. Formuliert
als Beweislast bei der neuen Regel: Sie muss sich behaupten, nicht die alte.

**H0:** Die Stundenkurven sind zwischen den Sommern instabil, oder die
Trennschärfe hängt gar nicht an der Stunde. Dann bleibt es bei einer einzigen
globalen Regel, und die ehrliche Fassung lautet: „vor 16:20 nichts erhöhen" gilt
weiter für Europa und ist anderswo unbelegt.

---

## Gates

| Gate | Bedingung |
|---|---|
| **G0** Datenbasis | ≥ 20 Städte mit ≥ 60 verwertbaren Stadt-Tagen je Sommer; das Stundenraster 10:20–20:20 vollständig belegt |
| **G1** Relevanz (H1) | ≥ 50 % der Städte weichen um ≥ 1 Stunde von 16:20 ab |
| **G2** Stabilität (H2) | ≥ 80 % der Städte: \|Prüfstunde(S1) − Prüfstunde(S2)\| ≤ 1 Stunde |
| **G3** Konsistenz (H3) | London, Paris, Madrid landen **alle drei** im Bereich 17:20–18:20 |
| **G4** Trennschärfe (H4) | Signal-Trefferquote mit stadtspezifischer Stunde ≥ Trefferquote mit starrer 16:20-Stunde, bei mindestens gleicher Zahl an Signal-Fällen. Reißt G4, gilt die Prüfstunde als **abgeleitet, aber nicht als besser** |

**Bonferroni:** Vier Hypothesen, aber **ein** freier Parameter (q = 12 %) und
**ein** Raster. Es werden keine weiteren Schwellen, keine alternativen
Definitionen des Signals und keine Teilmengen von Städten geprüft. Deshalb steht
hier keine t-Schwelle: G1–G3 sind Abzählungen, keine Signifikanztests. **G4 ist
bewusst als Nicht-Verschlechterung formuliert** — bei erwartbar rund 100
Signal-Fällen trüge ein t nichts, und ein Gate, das ohnehin nichts belegen kann,
soll auch nicht so tun.

---

## Designfallen, die diesen Test definieren

**1. Die Prüfstunde kann vor dem Kaufzeitpunkt liegen — für die Bewertung ist das
egal, für die Handlung nicht.** Gekauft wird am Vortag 14:45 Berlin, geprüft wird
am **Zieltag** zur Ortsstunde. Für Taipei bedeutet eine Prüfstunde von 13:20
Ortszeit 07:20 Berlin — mitten in der Nacht des Betreibers. Eine Regel, die
niemand befolgen kann, ist für die manuelle Anwendung wertlos; die Prüfstunde
wird deshalb **zusätzlich in Berliner Zeit** ausgewiesen.

**2. Gerundet, nicht roh.** Der Wächter arbeitet auf Buckets. „Das Hoch kommt
noch" heißt: das **gerundete** Maximum steigt noch. Auf Rohwerten fällt die Rate
zu hoch aus, weil jedes Zehntel zählt. Steht schon im Basisskript und wird
übernommen — mit dem Zusatz, dass für Hong Kong die **floor**-Buckets gelten
([[weather-hongkong-hko-buckets]]), sonst wird dort das falsche Maximum gerundet.

**3. Zwei Sommer sind zwei Sommer.** Die Stundenkurve hängt am Sonnenstand und
an der Lage; ein Jahr mit ungewöhnlicher Zirkulation verschiebt sie. G2 prüft
Stabilität zwischen zwei Sommern — das ist das Minimum, nicht der Beweis.

**4. Die Basisrate ist nicht die Trennschärfe.** Dass das Hoch um T noch in 12 %
der Fälle kommt, sagt nichts darüber, ob der gelayte Bucket dann noch gerissen
wird. Die beiden hängen zusammen, sind aber nicht dasselbe — G4 misst deshalb
getrennt.

**5. Küstenstädte mit Seebrise kippen früher, aber nicht monoton.** Der
Seoul-Fall ([[weather-seoul-seabreeze]]) zeigt Tage mit zwei Maxima. Die Regel
sucht die **früheste** Stunde unter q; bei nicht-monotonen Kurven kann das eine
Stunde treffen, nach der die Rate wieder steigt. Vorab entschieden: **die Regel
verlangt zusätzlich, dass alle späteren Stunden ebenfalls ≤ q bleiben.** Sonst
wird ein Zwischental als Gipfel gelesen — derselbe Fehler, der den Helsinki-Fall
ausgelöst hat, nur eine Ebene höher.

**6. Der Wächter bleibt auf Eis, egal wie gut die Zahlen aussehen.** Ohne Breite
schadet er dem engen Buch. Wenn G1–G4 alle bestehen, ist das Ergebnis eine
Tabelle von Prüfstunden und ein bestätigtes Signal — kein Grund, ihn
einzuschalten.

---

## Vorab-Erwartung (damit sie nicht zurechtgebogen wird)

**G1 und G3 halte ich für sicher.** Bei einer Spreizung von 71,3 % gegen 0,6 % um
dieselbe Uhrzeit kann 16:20 unmöglich für die Hälfte der Städte richtig sein, und
für die drei europäischen muss die Regel 17:20–18:20 liefern, sonst ist ihr q
falsch übertragen.

**G2 ist die eigentliche Frage.** Meine Erwartung: die Prüfstunde ist bei
kontinentalen Städten stabil und bei Küsten- und Monsunstädten wackelig. Wenn 80 %
nicht erreicht werden, liegt es vermutlich an einer Handvoll Städte mit
nicht-monotonen Kurven — und dann ist die ehrliche Antwort **nicht**, die
Toleranz auf zwei Stunden zu heben, sondern diese Städte als „ohne stabile
Prüfstunde" auszuweisen.

**G4 erwarte ich als knapp.** Die alte Messung mischte zwei entgegengesetzte
Fehler: in Madrid war 16:20 **zu früh** (71,3 % Restwahrscheinlichkeit, das Signal
ist dort größtenteils Fehlalarm), in Taipei **zu spät** (0,6 %, der Tag ist
entschieden, das Signal kommt zu spät zum Handeln). Beide Fehler zusammen ergaben
67 % Trefferquote. Eine korrigierte Stunde sollte das verbessern — aber die
Stichprobe ist klein, und die Städte mit den größten Korrekturen sind nicht
zwingend die mit den meisten Kandidaten.

**Was das Ergebnis wert ist, wenn alles hält:** Eine Tabelle mit einer Prüfstunde
je Stadt, in Ortszeit und in Berliner Zeit. Nutzbar sofort für manuelle
Entscheidungen, nutzbar später für den Wächter, falls Breite je wiederkommt.
**Kein Euro Ertrag** — und das ist keine Untertreibung, sondern die vollständige
Beschreibung.

## Abbruchregel

Reißt **G3** (Konsistenzprobe Europa), ist q falsch übertragen. Dann wird **kein
anderes q gesucht**, sondern die Ableitungsregel gilt als gescheitert und die
16:20-Regel bleibt, was sie ist: eine europäische Faustregel ohne Entsprechung
anderswo.

Reißt **G2** (Stabilität), gibt es keine stadtspezifische Prüfstunde, sondern nur
für die stabile Teilmenge. Die instabilen Städte werden benannt und bekommen
**keine** Regel — nicht die europäische, nicht die eigene. Für sie gilt dann
ausdrücklich: Uhrzeit ist kein Kriterium.

Reißt **G4**, bleibt die Prüfstunde als Basisraten-Tabelle bestehen und das
Wächter-Signal gilt als **nicht verbessert**. Es wird dann nicht nach einer
anderen Signaldefinition gesucht.

**In keinem Ausgang wird der Wächter eingeschaltet.** Dafür bräuchte es zuerst
eine Entscheidung für Breite, und die ist gefallen — dagegen.

---

# ERGEBNIS — G3 gerissen, und zwar an einem Fehler in G3 (02.08.2026)

Gerechnet mit `weather_pruefstunde_eval.py` unmittelbar nach der
Vorregistrierung (Commit `ca74a88`). ISD, Sommer 2024 und 2025, **26 von 27
Städten** mit ≥ 60 Tagen je Sommer (Jeddah liefert keine verwertbaren Tage).

## Die abgeleiteten Prüfstunden

| Ortszeit | Städte | Berliner Zeit |
|---|---|---|
| **14:20** | Tokyo, Taipei, Shanghai, Tel Aviv, Wellington, Panama City | 07:20–21:20 |
| **15:20** | Seoul, São Paulo, Cape Town | 08:20–20:20 |
| **16:20** | Beijing, Kuala Lumpur, Wuhan, Mexico City | 00:20–10:20 |
| **17:20** | Amsterdam, Ankara, Chengdu, Helsinki, London, Milan, Moscow, München, Toronto, Warschau, Buenos Aires | 11:20–23:20 |
| **18:20** | Paris | 18:20 |
| **19:20** | Madrid | 19:20 |

## G1 und G2 — beide belegt, G2 sehr deutlich

- **G1 Relevanz: BELEGT.** 22 von 26 Städten (85 %) weichen um ≥ 1 Stunde von
  16:20 ab. Die starre Regel passt für vier Städte.
- **G2 Stabilität: BELEGT, 26 von 26 (100 %).** Zwischen zwei Sommern stimmt
  jede einzelne Prüfstunde auf ≤ 1 Stunde. Die erwartete Schwäche bei Küsten-
  und Monsunstädten tritt **nicht** ein — Seoul, Taipei und Kuala Lumpur liefern
  in beiden Sommern identische Stunden.

## G3 — gerissen, aber die Diagnose der Pre-Reg ist falsch

| Stadt | Prüfstunde | verlangt |
|---|---|---|
| London | 17:20 | ok |
| Paris | 18:20 | ok |
| **Madrid** | **19:20** | **Abweichung** |

Die Abbruchregel greift wörtlich: **es wird kein anderes q gesucht.**

Ihre Begründung — „dann ist q falsch übertragen" — ist jedoch **nicht** die
Ursache. Der Fehler steckt in G3 selbst: **die 12 % der Erstmessung waren ein
Durchschnitt über fünf Städte** (Helsinki, München, Paris, Madrid, London),
nicht Madrids eigene Restwahrscheinlichkeit. Ein Aggregat wurde als
Einzelstadt-Größe verwendet und dann als Erwartung an jede Einzelstadt gestellt.

**Und das war aus bekannten Zahlen vermeidbar.** Madrids 71,3 % um 16:20 standen
in der Offenlegung dieser Pre-Reg. Wer sie liest, sieht sofort, dass die Rate
nicht binnen einer Stunde von 71 % auf 12 % fallen kann — Madrids Prüfstunde
*muss* später als 18:20 liegen. Die Kontrolle, die die Regel prüfen sollte, war
mit den eigenen offengelegten Daten unvereinbar.

Gemessen bestätigt sich das: Madrid 71,4 % um 16:20, Paris 45,5 %, London 17,9 %
— drei Städte, drei völlig verschiedene Kurven. Genau das ist der Befund, der
diese Pre-Reg ausgelöst hat, und G3 hat ihn für die drei Städte verboten, an
denen er zuerst sichtbar wurde.

## Was das für den Status bedeutet

**Die Ableitungsregel gilt nach eigener Vorregistrierung als gescheitert.** Die
Tabelle oben ist damit **gemessen, aber nicht belegt**, und sie wird nicht als
Handlungsgrundlage verwendet — auch nicht „vorläufig".

Das ist die zweite falsch parametrisierte Gate-Schwelle desselben Tages: bei G5
der Ursachen-Pre-Reg kam sie aus einem Median statt aus dem gehandelten
Mittelpreis, hier aus einem Fünf-Städte-Durchschnitt statt aus der Einzelstadt.
**Das Muster ist dasselbe — eine Aggregatzahl wird zur Schwelle für Einzelfälle
gemacht.** Die Lehre für jede weitere Pre-Reg: **jedes Gate gegen die bereits
offengelegten Einzelzahlen gegenrechnen, bevor es festgeschrieben wird.**

**G4 wurde nicht gerechnet**, wie vorregistriert — das Skript bricht nach G3 ab,
bevor die IEM-Reihen geholt werden. Die Trennschärfe des Wächter-Signals bleibt
damit auf der europäischen Messung vom 24.07. stehen.

## Was nicht geprüft und bewusst offen ist

- **Ob die Regel mit einem korrekt formulierten Konsistenz-Gate bestünde**, ist
  offen und wird hier nicht nachgeschoben. G1 und G2 sprechen dafür, aber das
  entscheidet eine eigene Vorregistrierung, nicht eine Umdeutung dieser.
- **Jeddah fehlt** (keine verwertbaren ISD-Tage) und ist damit eine der sieben
  divergenten Städte ohne Prüfstunde.
- **Die Berliner Zeiten sind unbrauchbar für manuelles Handeln**, wo sie in die
  Nacht fallen: Mexico City 00:20, Buenos Aires 22:20, Toronto 23:20, Tokyo
  07:20. Designfalle 1 ist damit bestätigt, aber nicht gelöst.
- **Der Sommer bleibt der Sommer.** Zwei Jahre, Juni bis August; für jede andere
  Jahreszeit ist keine dieser Stunden belegt.
