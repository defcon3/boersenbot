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
