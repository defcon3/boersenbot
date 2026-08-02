# Pre-Registrierung: Marktauswahl nach Stadt × Metrik — 2026-07-31

**Status:** Vorregistrierung. Geschrieben, **bevor** eine Rangliste, ein
OOS-Wert oder eine Ertragskennzahl gerechnet wurde. Die Sondierung sah nur
Mengen an: Historientiefe der Endpunkte, Zahl der Zellen, Handelsdichte.

Auswertung folgt in `weather_cell_selection_eval.py`.

---

## Anlass

Beim Prüfen der Wallet „ColdMath" (0x594edB91…, s. Memory) fiel auf, dass sie
fast ausschließlich **Tagestief-Bretter** handelt — London 571 „lowest" gegen
6 „highest". Eine explorative Messung an 8 Städten (Juni–Juli 2026, n = 53 je
Zelle) zeigte daraufhin:

| | min MAE | max MAE | Veto-Quote min | Veto-Quote max |
|---|---|---|---|---|
| London | **0,44 K** | 0,82 K | **4 %** | 57 % |
| Madrid | 1,39 K | **0,74 K** | 64 % | **2 %** |
| München | 1,87 K | 1,86 K | 70 % | 60 % |

Gepoolt sind Minima und Maxima **gleich gut** (1,00 vs 1,04 K). Der Unterschied
sitzt in der **Zelle**, nicht in der Metrik.

**Diese Tabelle ist Anlass, nicht Evidenz.** Sie entstand aus 16 angeschauten
Zellen, aus denen die beste zitiert wird — bei 16 Zellen ist ein Ausreißer nach
oben zu erwarten. Dazu 53 Tage, reiner Sommer. Genau solche Rückblick-Tabellen
sterben in diesem Repo regelmäßig an G2.

## Was hier NICHT behauptet wird

Nicht: „London-Minimum ist der beste Markt." Dieser Einzelwert ist selektiert
und wird durch Regression zur Mitte mit hoher Wahrscheinlichkeit schrumpfen.

Sondern: **Ist Prognosegenauigkeit eine stabile Eigenschaft einer Zelle — oder
Rauschen?** Nur wenn sie stabil ist, darf man nach ihr auswählen. Das ist eine
Frage über die *Rangliste als Ganzes*, nicht über eine Zelle.

---

## Hypothese

**H1 (primär):** Die in-sample gemessene Genauigkeitsrangfolge der Zellen
(Stadt × Metrik) sagt die out-of-sample Rangfolge vorher. Formal: Spearman-
Rangkorrelation ρ zwischen IS-MAE und OOS-MAE über alle Zellen ist deutlich
positiv.

**H2 (sekundär, Bonferroni t > 2,5):** Zellen des besten IS-Terzils sind OOS in
der **Lay-Zone** (Modell-P ≤ 10 %) besser kalibriert als die des schlechtesten.

**H3 (der Geldtest):** Ein auf das beste IS-Terzil beschränktes Lay-Buch erzielt
OOS einen höheren ROI je Einsatz als das Buch über alle Zellen.

**H0:** Genauigkeit ist überwiegend Rauschen; die Rangliste hält nicht. Dann
bleibt die Marktauswahl wie sie ist, und der Befund lautet, dass Zellenfilter
nichts bringen.

### Zellen, Daten, Split

- **Zelle** = (Stadt, Metrik), Metrik ∈ {Tagesminimum, Tagesmaximum}. Städte
  aus `weather_source_compare.STATIONS` → **~62 Zellen**.
- **Prognose:** Mittel der fünf Punktmodelle, `previous_day1` (Lead 1), lokale
  Tagesgrenzen — identisch zum Livebetrieb.
- **Bias:** 40-Tage-Fenster, gleitend, **nur Tage vor dem Zieltag**. Beide
  Seiten des Vergleichs identisch behandelt.
- **Ist-Werte:** settelnde Quelle (WU-Tabelle bzw. HKO/NOAA je Stadt).
- **Zeitraum:** 2024-08-01 bis 2026-07-23 (Endpunkte geprüft, Werte vorhanden).
- **Split — bewusst zwei volle Jahreszyklen:**
  **IS = 2024-08 bis 2025-07**, **OOS = 2025-08 bis 2026-07**.
  Kein 70/30-Schnitt: Ein zeitlicher Bruch mitten im Jahr würde IS und OOS
  verschiedene Jahreszeiten zuweisen, und der Bias dreht saisonal das Vorzeichen.
  So enthält jedes Fenster jede Jahreszeit genau einmal.

---

## Gates

| Gate | Bedingung |
|---|---|
| **G1** IS-Struktur | Die Zell-MAEs streuen deutlich stärker als bei reinem Rauschen: Spannweite bester/schlechtester Zelle > Faktor 2 **und** Levene/Permutationstest p < 0,01 |
| **G2** OOS-Stabilität (Kern) | Spearman ρ(IS-MAE, OOS-MAE) über alle Zellen **> 0,5** bei p < 0,01 |
| **G3** Kalibrierung, **bin-weise** | In den Bins 2–5 %, 5–10 %, 10–20 % **je einzeln**: bestes Terzil realisiert innerhalb [0,75×; 1,25×] der vorhergesagten Rate. **Nicht über die Lay-Zone mitteln** |
| **G4** Geld & Breite | OOS-ROI des Top-Terzil-Buchs > ROI des vollen Buchs — **und** das Top-Terzil liefert noch ≥ 40 % der Signale des vollen Buchs |
| **G5** Robustheit | Gilt nach Streichen der besten Zelle; kein einzelner Monat trägt > 30 % des Effekts; ρ hält auch bei Split nach Metrik |

**Warum G3 bin-weise:** Das G3 der Sigma-Pre-Reg vom 14.07. mittelte über die
Lay-Zone, wurde von Buckets mit P ≈ 0 dominiert und bestand deshalb sogar für
das Modell, das den Beijing-Verlust erzeugte — dort selbst als Design-Fehler
protokolliert. Die Konsequenz wird hier zur Vorbedingung.

**Warum die Breiten-Bedingung in G4:** Eine Zellenbeschränkung senkt die Zahl
der Signale. Die Skalierungsdoktrin lautet aber ausdrücklich *Breite statt
Größe*, weil die Ausführung ab 250 $ je Position wegbricht. Ein Filter, der den
ROI hebt und dabei das Buch halbiert, ist deshalb **kein** Fortschritt — er
verlagert das Problem nur. Die 40-%-Schwelle ist vorab gesetzt, nicht im
Nachhinein an das Ergebnis angepasst.

**Bonferroni:** H1/G2 ist *ein* Test auf der Gesamtrangliste, nicht 62 Tests auf
Einzelzellen. Einzelzell-Werte werden berichtet, aber **kein** Gate darf über
eine einzelne Zelle erfüllt werden — insbesondere nicht über London-Minimum,
den Anlassfall.

## Vorab-Erwartung (damit sie nicht zurechtgebogen wird)

**G1 besteht vermutlich.** Stationseigenschaften sind physisch stabil:
Küstenlage, Beckenlage, Höhe, Stadtwärmeinsel. Dass München auf beiden Seiten
schlecht ist (Alpennähe, Föhn, Konvektion), ist kein Zufall.

**G2 ist die eigentliche Frage, und ich halte ρ für moderat** — geschätzt 0,3
bis 0,6. Das reicht möglicherweise **nicht** für die Schwelle 0,5.

**Der Anlasswert wird schrumpfen.** London-Minimum mit MAE 0,44 K stammt aus
16 Zellen mit je 53 Tagen; ich erwarte OOS eher 0,6–0,8 K.

**G4 halte ich für am gefährdetsten** — nicht wegen des ROI, sondern wegen der
Breite. Ein Drittel der Zellen liefert grob ein Drittel der Signale, und das
könnte die 40 % reißen, selbst wenn der ROI steigt.

## Härtetest (deskriptiv, KEIN Gate)

Auf die dokumentierten Verlust-Trades angewandt: **München 23 °C** (−9,44 $,
konvektive Schauer) und **Beijing 32/33 °**. Liegen diese Zellen im
schlechtesten Terzil? Wenn ja, hätte der Filter sie gesperrt — das wäre das
praktische Argument. Liegen sie im besten, ist H3 wertlos, egal was G2 sagt.

## Abbruchregel

Reißt **G2**, ist die These falsifiziert: Genauigkeit ist dann keine
handelbare Zelleneigenschaft, und es wird **nicht** auf andere Kennzahlen
(RMSE, Trefferquote, gewichtete Mischungen) ausgewichen, bis eine hält. Reißt
nur **G4** an der Breiten-Bedingung, lautet der Befund „Filter hebt den ROI,
kostet aber zu viel Buch" — und der Filter wird als **Gewichtung** statt als
Sperre weiterverfolgt, in einer eigenen Pre-Reg.

---

## Nachtrag 2026-08-01 — Notiz, **KEIN Gate, keine Hypothese**

Nachträglich angehängt, **nach** dem Schreiben der Vorregistrierung oben und
ohne jede Auswertung. Die Gates, Hypothesen und die Vorab-Erwartung bleiben
unverändert; nichts hier darf zur Erfüllung eines Gates herangezogen werden.
Festgehalten wird nur ein Kandidat, der beim Prüfen einer Adresse auffiel und
methodisch hierher gehört.

**MET Norway über Open-Meteo (`open-meteo.com/en/docs/metno-api`)** ist ein
**Ein-Stadt-Modell** und damit per Konstruktion ein Zellen-Effekt — genau die
Größe, die diese Pre-Reg untersucht. Am 01.08. geprüft:

- **Prognose-Historie vorhanden:** `models=metno_seamless` mit
  `temperature_2m_previous_day1` liefert (stündliche Variable; die tägliche
  `temperature_2m_max_previous_day1` existiert **nicht**). Gegenprobe mit
  `icon_seamless` identisch. Das Modell ist also kalibrierbar — die Bedingung,
  an der neue Quellen hier sonst scheitern.
- **Abdeckung nur Skandinavien.** Verifiziert: Helsinki liefert Daten, London
  und Mexiko-Stadt antworten mit `latitude: nan, longitude: nan`. Von den
  Städten in `STATIONS` ist **genau eine** abgedeckt.
- **Produktklasse:** MET Nordic, 1 km, aus dem 2,5-km-MetCoOp-Ensemble mit
  ECMWF-Initialisierung, nachbearbeitet gegen Messungen und Radar, stündliche
  Aktualisierung. Unsere fünf Punktmodelle laufen auf 9–25 km.

**Wenn das je getestet wird, dann in einer eigenen Pre-Reg** mit eigenen Gates
(Vergleich metno gegen das 5-Modell-Mittel für die Zelle Helsinki × Maximum,
auf demselben 40d/700d-Verfahren, walk-forward). `weather_source_compare.py`
hat `--models` und die Bias/σ-Maschinerie bereits; der Aufwand ist ein
Parameter, nicht ein Projekt.

**Zwei Warnungen dazu, damit sie nicht verlorengehen:**

1. Die Doku bewirbt „updates every hour" und Radar-Nachbearbeitung. Das ist
   **kein** Anlass, den Intraday-Nowcast wieder aufzumachen — Ratchet und
   Advektion sind beide gemessen gescheitert. Zulässig ist ausschließlich ein
   besseres **Tages-µ** für die eine Zelle.
2. Eine Stadt von 28 sind 3,5 % des Buches. Der Prio-3-Test hat bereits
   gezeigt, dass drei zusätzliche **globale** Modelle die gepoolte Kennzahl
   nicht bewegen (+1,5 % gegen Gate 3 %); ein Ein-Stadt-Modell kann sie
   erst recht nicht bewegen. Der Nutzen wäre zellenlokal oder gar keiner.

---

# ERGEBNIS — die Rangliste hält glänzend, das Geld folgt ihr nicht (02.08.2026)

Gerechnet mit `weather_cell_selection_eval.py`, **60 Zellen** mit ≥ 120 Tagen in
beiden Fenstern, zwei volle Jahreszyklen wie vorregistriert.

## G1, G2, G5 — grün, und G2 weit über Erwartung

- **G1 IS-Struktur: GRÜN.** Beste Zelle Panama City min (MAE 0,535 K),
  schlechteste Buenos Aires min (1,805 K) — **Faktor 3,38**, Permutationstest
  **p = 0,0005**.
- **G2 OOS-Stabilität: GRÜN, ρ = +0,893 (p = 9·10⁻²²).** Gefordert war ρ > 0,5,
  meine Vorab-Erwartung lautete 0,3–0,6 und „reicht möglicherweise nicht".
  **Deutlich unterschätzt.** Die Genauigkeitsrangfolge eines Jahres sagt die des
  nächsten fast punktgenau voraus: Panama City min und London min bleiben Rang 0
  und 1, Buenos Aires min bleibt Rang 59.
- **G5 Robustheit: GRÜN.** Ohne die beste Zelle ρ = +0,887; getrennt nach Metrik
  ρ = +0,906 (min) und +0,862 (max).

**Prognosegenauigkeit ist damit eine hochstabile Eigenschaft einer Zelle — keine
Rauschgröße.** Das ist der klarste Einzelbefund dieser Messreihe.

## G3 — rot, und in die überraschende Richtung

| Terzil | Bin 2–5 % | Bin 5–10 % |
|---|---|---|
| **bestes** | 3,33 % vorhergesagt / **2,48 %** real (0,75×) | 7,29 % / **5,08 %** (0,70×) |
| mittleres | 3,33 % / 2,44 % (0,73×) | 7,29 % / 4,96 % (0,68×) |
| **schlechtestes** | 3,34 % / 2,66 % (0,80×) ✓ | 7,33 % / **6,67 %** (0,91×) ✓ |

Nicht die ungenauen Zellen sind schlecht kalibriert, sondern die **genauen**: Dort
tritt das Ereignis **seltener** ein als das Modell sagt — es ist also **zu
pessimistisch**. Ausgerechnet das schlechteste Terzil liegt im geforderten Band.

**Das ist kein Ausreißer, sondern ein Muster.** Am selben Tag ergab die
Ursachen-Messung dasselbe von der anderen Seite: das Modell schätzt die Ränder zu
breit (P_modell(−2) = 9,3 % gegen 5,8 % real).
**Zwei unabhängige Messungen sagen: unser σ ist zu groß.** Praktische Folge — der
P_pess-Filter verwirft Kandidaten, die sicherer sind als das Modell glaubt.

## G4 — rot, und wirtschaftlich hohl

| Buch | n | ROI |
|---|---|---|
| voll | 165.469 | +9,63 % |
| bestes Terzil | 43.461 | **+9,78 %** |

**Der Vorteil beträgt +0,15 pp, der Signalanteil 26,3 % statt der geforderten
40 %.** Der Filter kostet also drei Viertel des Buchs und liefert dafür nichts,
was sich von null unterscheiden ließe — zumal G4 laut Code-Hinweis mit einem
**festen Lay-Preis von 0,90** rechnet, weil historische Marktpreise über zwei
Jahre fehlen. Er misst damit Kalibrierungsqualität bei fairer Bepreisung, nicht
Marktertrag; die Abweichung war vorab deklariert.

**Härtetest** (deskriptiv): München max liegt im schlechtesten Terzil — der
Filter hätte den −9,44-$-Verlust gesperrt. Beijing max liegt im **mittleren**,
wäre also durchgerutscht.

## Was daraus folgt — und wo ich von der Abbruchregel abweiche

Die Abbruchregel sieht für „nur G4 reißt an der Breite" vor, den Filter **als
Gewichtung statt als Sperre** in einer eigenen Pre-Reg weiterzuverfolgen. Formal
greift sie: ROI(Terzil) > ROI(voll) ist mit 9,78 > 9,63 erfüllt.

**Ihre Voraussetzung ist es wirtschaftlich nicht.** Sie unterstellt, der Filter
*hebe* den ROI — bei +0,15 pp auf einem Testaufbau mit fixiertem Preis ist das
Rauschen. Eine Gewichtungs-Pre-Reg auf dieser Grundlage würde eine Größe
optimieren, die keinen messbaren Ertrag trägt. **Empfehlung: keine Folge-Pre-Reg.**
Die Entscheidung gehört dem Betreiber; sie wird hier nicht stillschweigend
getroffen, sondern offen benannt.

## Die Antwort auf die Regionalspezialisten-Frage

Die drei untersuchten Wallets sind je Region spezialisiert — die Vermutung war,
dass Konzentration den Ertrag hebt. **Über Prognosegenauigkeit lässt sich das
nicht erklären.** Man kann die Zellen nach Genauigkeit sortieren, die Rangfolge
hält über Jahre (ρ = 0,89), und trotzdem bringt die Konzentration auf das beste
Drittel praktisch nichts (+0,15 pp) bei drei Vierteln weniger Buch.

Wenn Spezialisierung bei jenen Wallets funktioniert, dann **aus anderen Gründen**
— Zeitzone, Aufmerksamkeit, Liquidität, Handelsfrequenz (die Asien-Wallet dreht
83 % ihres Volumens). Nicht, weil ihre Städte besser prognostizierbar wären.

## Was nicht geprüft und bewusst offen ist

- **G4 hat nie echte Marktpreise gesehen.** Ob ein Zellenfilter *mit* realen
  Preisen mehr bringt, ist damit nicht ausgeschlossen — nur nicht belegt. Für die
  drei Wochen, für die Preise vorliegen, wäre die Stichprobe je Zelle zu dünn.
- **Der Bin 10–20 % war in allen drei Terzilen leer.** Die Kalibrierungsaussage
  steht damit nur auf den beiden unteren Bins.
- **Warum σ zu groß ist**, bleibt offen. Zwei Messungen zeigen es jetzt
  unabhängig; die Ursache ist keine davon nachgegangen.
- **Zellen ≠ Regionen.** Diese Messung sortiert nach Genauigkeit, nicht nach
  Erdteil. Ob eine *regionale* Konzentration etwas anderes leistet als eine
  genauigkeitsbasierte, ist nicht geprüft.
