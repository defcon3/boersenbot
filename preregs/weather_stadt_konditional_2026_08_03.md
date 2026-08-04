# Pre-Reg: Stadt-konditionale Modellgewichtung — nur dort, wo das Gefälle es hergibt

**Angelegt:** 2026-08-03, vor jeder Auswertung.
**Status:** registriert, **nicht gefahren**. Auswertung durch
`weather_stadt_konditional_eval.py` (noch zu schreiben).
**Löst ein:** den offenen Betreiber-Entscheid aus
`weather_calib_prio235_prereg_2026_07_18.md` (Abschnitt „Muster über alle drei").

## Anlass

Am 18.07. ist die Modellgewichtung (inverse Kovarianz, monatlicher Refit,
walk-forward) **knapp** gescheitert: gepoolt **+5,63 %** σ-Reduktion (Gate ≥ 5 % ✓),
aber **Stadt-t 1,91** (Gate > 2 ✗). Das Fazit dort, wörtlich:

> Der Nutzen konzentriert sich konsistent in Städten mit großem Modell-Güte-Gefälle
> bzw. starker Saisondynamik (Seoul, Tel Aviv, München, Jeddah, Wuhan, Beijing);
> in ausgeglichenen Städten kosten alle drei Verfahren leicht. Eine
> stadt-konditionale Anwendung wäre JETZT post hoc — falls gewünscht, als NEUE
> Pre-Reg mit a-priori-Kriterium und Forward-Gate registrieren.

**Was seither dazugekommen ist:** Die Bucket-Abstandsmessung vom 03.08.
(`weather_bucket_abstand_2026_08_03.md`) kennt diese Liste nicht und zeigt
trotzdem auf dieselben Städte — Tel Aviv −1,11 · München +1,31 · Wuhan +1,08 ·
Seoul +1,00 (min sogar +1,58) · Beijing −1,09. Zwei getrennte Datenquellen
(700d-Modellresiduen damals, Live-Ladder-Log heute), dieselbe Menge.

Der globale Bias ist dabei **null** (+0,04 Bucket über 341 Stadttage). Es gibt
keinen Gesamtfehler zu heben — Verbesserung ist nur stadtweise möglich.

## Hypothese

Ein **mechanisches, vorab festgelegtes Kriterium** trennt die Städte, in denen
Gewichtung hilft, von denen, in denen sie schadet. Auf die so ausgewählte Gruppe
angewandt, besteht die Gewichtung das Städte-t-Gate, an dem die pauschale Fassung
gescheitert ist.

## Das Kriterium — rangbasiert, damit es keine Schwelle zu drehen gibt

    G_Stadt = sigma_ens(gleichgewichtet) / min_m sigma_m

G > 1 heißt: das gleichgewichtete Ensemble ist **schlechter als das beste
Einzelmodell** — dort hat Umgewichtung Luft. (Für Seoul wurde am 18.07. genau das
gemessen: ENS 2,13 gegen ECMWF allein 1,27.)

**Auswahl = oberstes Quartil nach G**, geschätzt **walk-forward** aus den Daten
*vor* dem jeweiligen Bewertungsmonat, monatlicher Refit wie bei H5.

Rangbasiert statt Schwellenwert, weil eine absolute Schwelle ohne Blick auf die
G-Verteilung geraten wäre — und mit Blick darauf wäre sie geschnüffelt. Das
Quartil wählt immer ~7 von 29 Städten und generalisiert auf neue Städte (relevant
wegen der acht stationslosen Bretter ab 02.09.).

Die Sechser-Liste vom 18.07. ist **nicht** das Kriterium, sondern die
**Validierung**: trifft das mechanische Quartil sie ungefähr, stützt das beides.

## Datenbasis

- `preregs/weather_konvektiv_sigma_residuen_2026_08_03.csv.gz` — **29 Städte**,
  5 Modelle, 2024-09-01 … 2026-08-01, 646–697 Tage je Stadt, 97.360 Zeilen.
  Es fehlen nur **Moskau** (settelt über NOAA, nicht über METAR) und **Hong Kong**
  (hat kein METAR, läuft über die HKO-Reihe) — beide bewusst.

  ⚠️ **Shenzhen läuft hier gegen METAR, nicht gegen WU.** Der Prio-0-Entscheid
  vom 17.07. („Shenzhen gegen die WU-Reihe kalibrieren") ist durch den
  Lau-Fau-Shan-Fund vom 02.08. **überholt**: `--actuals wu` bricht für ZGSZ
  inzwischen mit der eigenen Schutzprüfung ab („WU liefert für ZGSZ die fremde
  Station 'Lau Fau Shan'"), weil unter diesem Locator gar nicht Shenzhen
  zurückkommt. Der Markt folgte in allen abweichenden Fällen dem METAR der
  echten Station — METAR ist damit die richtige Quelle. **Das gilt über diese
  Pre-Reg hinaus und sollte beim nächsten Shenzhen-Kalibrierlauf beachtet
  werden.**
- `bb_WeatherLadders` (Lead 1, ab 10.07.2026) für die Buchebene.

## Gates

### G1 — Wirkung in der Auswahlgruppe (walk-forward, OOS)
Mittlere σ-Reduktion ≥ **10 %** **und** Städte-t > **2,0**, gerechnet **nur über
die ausgewählten Städte**.

> **Gegenrechnung** (Pflichtübung 02.08.): Die am 18.07. offengelegten
> Einzelwerte der mutmaßlichen Gruppe sind Seoul +43,1 · München +19,5 ·
> Tel Aviv +18,4 · Wuhan +12,7 · Jeddah +10,3 → Mittel 20,8 %, sd 13,1, t ≈ 3,5
> bei n = 5. Das Gate ist erreichbar, aber nicht geschenkt: **ohne Seoul** sinkt
> das Mittel auf 15,2 % und t auf ≈ 3,0 — es trägt also auch ohne den Ausreißer.
> Warum 10 % und nicht 5: die pauschale Fassung brachte 5,63 %; wenn
> Konditionierung den Aufwand rechtfertigen soll, muss sie das in der Gruppe
> ungefähr verdoppeln.

### G2 — Fehlklassifikation
Höchstens **20 %** der ausgewählten Städte dürfen OOS schlechter werden.

> **Gegenrechnung:** pauschal waren 17/27 besser, also **37 % schlechter**
> (bekannt: Toronto −10,3 %, Mexico City −7,8 %). Das Kriterium muss diese Quote
> in der Auswahl auf unter 20 % drücken — genau das ist seine Aufgabe. Ein Gate
> „kein Schaden außerhalb der Gruppe" wäre dagegen **wertlos**: außerhalb wird gar
> nicht gewichtet, die Änderung ist per Konstruktion null.

### G3 — Trennschärfe des Kriteriums
Spearman ρ zwischen G (walk-forward geschätzt) und realisierter σ-Änderung über
**alle** Städte > **0,4**. Sonst ist G keine Regel, sondern eine Umschreibung
derselben sechs Namen.

> **Gegenrechnung:** Hierzu existiert **keine** Vorabzahl. Das ist das eigentlich
> neue Stück Evidenz dieser Pre-Reg — und das Gate, das am ehesten reißt.

### G4 — Buchebene (klein, deshalb Gruppen- und kein Einzelstadt-Gate)
Auf dem Ladder-Log muss das gewichtete µ die MAE **im Mittel der Auswahlgruppe**
um ≥ **0,2 Bucket** senken, ohne die übrigen Städte zu verschlechtern.

> **Gegenrechnung:** Tel Aviv liegt bei MAE 1,11 mit Mittel −1,11; eine
> Ankerverschiebung um ein Bucket brächte die MAE auf ≈ 0,4. Seoul 1,25/+1,00,
> München 1,31/+1,31 entsprechend. 0,2 ist damit konservativ und scheitert nur,
> wenn die Gewichtung den Anker gar nicht bewegt. **Kein Einzelstadt-Gate**, weil
> bei n = 11–18 der Standardfehler der MAE bei ≈ 0,25 Bucket liegt — eine
> Einzelstadt kann das nicht belegen.

### G5 — Robustheit
Leave-one-city-out über die Auswahlgruppe. Ohne Seoul muss G1 weiter halten
(≥ 8 %, t > 1,5). Das Ergebnis darf nicht an einer Stadt hängen.

## Was ausdrücklich NICHT geprüft wird

**Keine Handelsregeln je Stadt.** Die Abstandsmessung vom 03.08. zeigt, warum: Von
30 Städten überlebt genau **eine** (Tel Aviv) die Bonferroni-Korrektur. Wer aus
11–18 Live-Tagen je Stadt Regeln ableitet, baut 29 Zufallsmuster. Geprüft wird
ausschließlich die **Kalibrierung** — ein Verfahren, geschätzt auf ~660 Tagen,
konditional angewandt.

**Keine Stadt wird gesperrt.** Betreiber-Entscheidung vom 02.08.: Verbesserungen
über die Ankerkorrektur, nie über Sperren.

## Umsetzungssperre

Ein PASS erlaubt **keine** Live-Änderung. Es folgt ein Forward-Fenster wie bei den
vier laufenden Linien. Vorher ohnehin nichts anfassen: Bis **02.09.2026** laufen
drei vorregistrierte Forward-Tests (Rand-Longshot, Anker-Divergenz, Markt-Anker) —
eine Änderung an µ oder am Universum mitten im Fenster macht deren zweite Hälfte
zu einer anderen Stichprobe und die Auswertung wertlos.

## Bekannte Schwächen (vorab benannt)

- **Zeitliche Überlappung:** Der Residuen-Dump endet am 01.08., das Ladder-Log
  beginnt am 10.07. G4 läuft also teilweise auf Tagen, die auch in den σ-Fit
  eingehen. Deshalb ist der Walk-forward-Zwang keine Formalie: die Gewichte eines
  Tages dürfen ausschließlich aus früheren Daten stammen.
- **Die Auswahlgruppe ist klein** (~7 Städte). Städte-t mit n = 7 ist empfindlich
  gegen einen einzelnen Ausreißer — dagegen steht G5.
- **G kennt nur die σ-Seite.** Die Abstandsmessung zeigt einen **Anker**-Fehler
  (enge, verschobene Verteilung). Gewichtung verändert µ *und* σ; ob sie den Anker
  genug bewegt, ist offen — G4 ist deshalb das Gate, das inhaltlich zählt, auch
  wenn G3 statistisch am ehesten reißt.
- Die 40d/700d-Doppelkalibrierung bleibt unangetastet; Saison-Harmonik ist am
  18.07. gegen das gleitende 40d-Fenster gescheitert (+0,6 %, t 0,7) und wird
  hier **nicht** wieder aufgemacht.

---

# ERGEBNIS (gefahren 2026-08-04, `weather_stadt_konditional_eval.py`)

**G1 GRÜN · G2 GRÜN · G3 GRÜN · G4 GRÜN · G5 ROT — damit kein PASS.**

Datenlage: 97.360 Zeilen, 29 Städte, 2024-09-01 … 2026-08-01. Anlauf 180
Serien-Tage, 18 Bewertungsmonate (2025-03 … 2026-08), Quartil = 7 von 29 je Monat.

## Reproduktionsprobe (ohne Gate) — die Mechanik ist dieselbe

Pauschale Gewichtung über alle Städte: **+5,59 %** MAE-Gewinn gegen die
**+5,63 %** vom 18.07. Auf 0,04 pp, und das über einen anderen Dump, ein anderes
Fenster und mit Shenzhen jetzt drin statt ausgeschlossen. Die Gegenrechnungen
dieser Pre-Reg tragen also.

**Nebenbefund mit Folgen:** der Städte-t der pauschalen Fassung liegt jetzt bei
**+2,26** — sie hätte das Gate vom 18.07. (t > 2) bestanden. Das ist kein
Freibrief (zwei Städte mehr, anderes Fenster), verschiebt aber die Beweislast:
Konditionierung muss sich gegen eine Referenz behaupten, die nicht mehr
eindeutig gescheitert ist.

## G1 — BELEGT, aber schwächer als die Gegenrechnung erwartet

Mittel über 11 Auswahlstädte **+11,00 %**, Städte-t **+2,41** (verlangt ≥ 10 %, t > 2).

| Stadt | Tage | konditional | pauschal |
|---|---|---|---|
| Seoul | 357 | +43,0 % | +43,0 % |
| Buenos Aires | 31 | +20,5 % | +4,4 % |
| Munich | 360 | +20,0 % | +20,0 % |
| Tel Aviv | 305 | +17,9 % | +19,8 % |
| Wuhan | 357 | +12,5 % | +12,5 % |
| Amsterdam | 122 | +11,7 % | +3,6 % |
| Jeddah | 358 | +8,1 % | +8,1 % |
| Ankara | 310 | +5,4 % | +6,5 % |
| Beijing | 284 | +3,1 % | +6,1 % |
| NYC | 47 | −8,4 % | +11,4 % |
| Warsaw | 98 | −12,8 % | +2,0 % |

## G2 — BELEGT

2 von 11 Auswahlstädten werden schlechter = **18 %** (verlangt ≤ 20 %). Die
pauschale Fassung hatte 37 %. Das Kriterium tut also genau das, wofür es da ist —
aber es schrammt am Gate vorbei, und beide Fehlklassifikationen (NYC, Warsaw)
sind Städte mit unter 100 Bewertungstagen.

## G3 — BELEGT, und das ist das eigentliche Ergebnis

Spearman ρ zwischen G und realisiertem Gewinn über alle 29 Städte: **+0,791**
(p < 10⁻⁴, verlangt > 0,4). Das war das Gate ohne jede Vorabzahl und laut
Pre-Reg das, „das am ehesten reißt". Es ist mit Abstand das stärkste Ergebnis:
G ist keine Umschreibung der sechs Namen vom 18.07., sondern eine Regel.

**Validierung bestanden:** alle **6 von 6** Städten der 18.07.-Liste werden vom
mechanischen Quartil gewählt, vier davon (Seoul, München, Wuhan, Jeddah) in
14 von 18 Monaten.

## G4 — BELEGT, trotz der vorab benannten dünnen Datenlage

| Gruppe | Stadttage | Städte | MAE gleich | gewichtet | Δ |
|---|---|---|---|---|---|
| Auswahl | 85 | 7 | 1,07 | **0,79** | **+0,28** |
| übrige | 236 | 21 | 0,86 | 0,85 | +0,01 |

Verlangt waren ≥ 0,2 Bucket in der Auswahlgruppe ohne Schaden außerhalb. Die
Gewichtung **bewegt den Anker** — das war die inhaltlich offene Frage („G kennt
nur die σ-Seite", Schwächen-Abschnitt). Bei n = 85 liegt der Standardfehler
allerdings in der Größenordnung des Gates selbst; belastbar ist das nicht.

## G5 — ROT, an 0,20 Prozentpunkten

Ohne Seoul: **+7,80 %**, t **+2,17**. Verlangt waren ≥ 8 % **und** t > 1,5. Die
t-Bedingung hält deutlich, die Effektgröße reißt knapp. Leave-one-city-out über
alle elf: Seoul ist der schlechteste Fall, keine andere Stadt drückt tiefer.

**Warum die Gegenrechnung 15,2 % erwartet hatte und 7,80 % herauskommt:** sie
rechnete auf der handverlesenen Fünfer-Liste. Das mechanische Quartil wählt über
18 Monate **11** verschiedene Städte, darunter vier Randfälle mit 31–122
Bewertungstagen, die um ±20 pp schwanken (Warsaw −12,8 · NYC −8,4 · Amsterdam
+11,7 · Buenos Aires +20,5). In einem städte-gleichgewichteten Mittel — so
vorregistriert, so am 18.07. gerechnet — schlägt dieses Rauschen voll durch. Die
Fragilität sitzt damit nicht bei Seoul, sondern im Schwanz der selten gewählten
Städte; G5 kann das per Konstruktion nicht auseinanderhalten.

## Was daraus folgt

Kein PASS. An 0,2 pp wird nicht gedreht, und eine nachträgliche Änderung der
Auswahlregel wäre genau das, was die Pre-Reg mit „rangbasiert, damit es keine
Schwelle zu drehen gibt" ausschließen wollte.

**Post-hoc-Beobachtung, ausdrücklich kein Ergebnis:** beschränkt auf die sieben
Städte, die in ≥ 10 von 18 Monaten gewählt werden, steht das Mittel bei 15,7 %,
ohne Seoul bei 11,2 % mit t ≈ 4,0. Ein Mindest-Stabilitätskriterium („G muss
eine Stadt wiederholt oben einsortieren, bevor gewichtet wird") wäre der
naheliegende nächste Schritt — als **neue** Pre-Reg mit eigenem Forward-Gate,
nicht als Nachbesserung dieser hier. Dagegen steht der Nebenbefund oben: wenn
die pauschale Fassung inzwischen selbst t > 2 erreicht, ist womöglich
„gewichten, überall" die einfachere Antwort und die Konditionierung Zierat.

Beides bleibt liegen, bis die drei Forward-Fenster am 02.09.2026 ausgewertet
sind. Die Umsetzungssperre oben gilt unverändert.
