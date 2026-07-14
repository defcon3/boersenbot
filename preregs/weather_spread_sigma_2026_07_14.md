# Pre-Reg: Spannen-konditioniertes Sigma für Wetter-Buckets

**Registriert:** 2026-07-14, **vor** dem Lauf. Auswertung folgt in diesem Dokument.
**Anlass:** Der Beijing-33-Verlust (`weather_lay_postmortem_2026_07_14_beijing.md`)
und der daraus gemessene Befund, dass der Ensemble-Fehler stark von der
Modellspanne abhängt (700 d, n = 5.587: Tail-Risiko P(|err| > 2,5 °) steigt von
4,2 % auf 13,3 % oberhalb 3 ° Spanne — Faktor **3,15**).

---

## 1. Das Problem

Beide Screens rechnen mit einem **festen Sigma je Stadt** (ZBAA-ENS: 1,447) und
einer Normal-Annahme. Real ist Sigma an ruhigen Tagen ~1,06 und an zerklüfteten
~1,77. Die Folge ist ein systematischer Fehler **genau dort, wo gehandelt wird**:
Ein Bucket ist teuer gepreist (= hohe Lay-Rendite), *weil* die Modelle streiten —
und ausgerechnet dann ist unser Sigma zu eng.

Die Notlösung vom 14.07. ist ein **harter Veto** (Spanne > 3 ° → kein Kandidat).
Der ist empirisch gedeckt, aber grob: Er sperrt **37 % aller Tage** pauschal,
statt sie korrekt zu bepreisen.

## 2. Hypothese

> **H:** Modelliert man Sigma als Funktion der Tagesspanne statt als Konstante,
> werden die Bucket-Wahrscheinlichkeiten **ehrlich** — auch an zerklüfteten Tagen.
> Dann kann der harte Veto durch korrekte Preise ersetzt werden.

**Gegenhypothese H0 (ernst zu nehmen):** Die Zerklüftung ist nicht nur breiter,
sondern *anders* (schiefe/multimodale Fehler, z. B. bei Frontdurchgängen). Dann
rettet auch ein größeres Sigma die Normal-Annahme nicht, und der harte Veto bleibt
die richtige Antwort.

## 3. Was hier NICHT geprüft werden kann (Datentragfähigkeit)

**Die eigentliche Geldfrage — „sind zerklüftete Buckets vom Markt falsch
bepreist?" — ist mit diesen Daten NICHT beantwortbar.** Dafür bräuchte es
historische Marktpreise für genau diese Buckets; `bb_WeatherLadders` sammelt die
erst seit dem 11.07.2026 (N viel zu klein). Das wird hier **nicht** behauptet und
**nicht** getestet.

Was geprüft wird, ist die **Vorfrage**: Ist unser Wahrscheinlichkeitsmodell auf
zerklüfteten Tagen ehrlich? Ohne ein „ja" darauf ist die Geldfrage gar nicht
stellbar — man würde nur eine falsche Zahl gegen den Markt halten. Die Geldfrage
selbst wird als **Forward-Test** registriert (siehe G5).

## 4. Modell (vor dem Lauf festgelegt, keine Formsuche im Nachhinein)

Sei `s` = Spanne der 5 rohen Modell-Tageshochs (max − min) am Zieltag.

- **Primär:** `sigma_city(s) = a_city + b · s`
  — Achsenabschnitt **je Stadt** (jede Stadt hat ihre eigene Grundunsicherheit),
  Steigung **b gepoolt** (die Spannen-Sensitivität wird als gemeinsam angenommen;
  spart Parameter, ~500 Tage/Stadt tragen keine 28 eigenen Steigungen).
  Fit per **Maximum-Likelihood** (Gauß, sigma_i = a_city + b·s_i), untere
  Schranke sigma ≥ 0,3.
- **Bias:** bleibt konstant je Stadt (wie bisher). Ob der Bias selbst mit `s`
  korreliert, wird **berichtet**, aber nicht angepasst (kein Tuning).
- **Sekundär (ebenfalls vorab festgelegt):** dasselbe Sigma, aber statt der
  Normal-Annahme die **empirischen Quantile der standardisierten Residuen**
  `z = err / sigma_city(s)`. Deckt den bekannten Formfehler ab
  (`weather_error_quantiles.py`, Commit 6c541527: Zentrum schärfer als Normal,
  Flanken schief). Wird berichtet, entscheidet aber nur mit, wenn G3 es verlangt.

**Vergleichsmodell (Status quo):** `sigma_city = const`, Normal — exakt das, was
die Screens heute rechnen.

## 5. Daten & Split (vor dem Lauf festgelegt)

- Quelle: Open-Meteo Previous-Runs (`temperature_2m_previous_day1`, echter
  24-h-Lead, archivierter Originallauf) + IEM-METAR als Ist (`report_type` 3+4),
  lokaler Kalendertag. Identisch zu `weather_source_compare.py`.
- Städte: alle 28 aus `STATIONS`; Stadt fliegt raus bei < 300 nutzbaren Tagen.
- Zeitraum: 700 Tage bis gestern.
- **Split: zeitlich, die ältesten 70 % = IS, die jüngsten 30 % = OOS.**
  Kein Zufalls-Split — Wetter ist stark autokorreliert, benachbarte Tage würden
  leaken.
- **Vorab deklarierter Zusatzschnitt (damit er nicht post-hoc ist):** dieselben
  Metriken zusätzlich auf dem **Sommer-Teil (Juni–August) des OOS** — das ist das
  Regime, in dem tatsächlich gehandelt wird.

## 6. Gates

- **G1 (IS):** Die Steigung `b` ist positiv und deutlich von 0 verschieden
  (t > 4), und die empirische Sigma-Kurve ist über die Spannen-Bins monoton.
- **G2 (OOS):** Das Sigma-Modell schlägt das feste Sigma out-of-sample in der
  **Log-Loss der Bucket-Wahrscheinlichkeiten** (alle ganzzahligen Buckets je
  Stadt-Tag, Bucket = [k−0,5; k+0,5)). Verbesserung muss klar sein, nicht Rauschen.
- **G3 (die entscheidende Kalibrierung, OOS):** In der **Lay-Zone** — alle
  Buckets, denen das Modell **P ≤ 10 %** gibt — darf die realisierte Trefferquote
  **1,25 × der vorhergesagten** nicht überschreiten, **und zwar in BEIDEN
  Spannen-Regimen** (< 3 ° und ≥ 3 °) getrennt.
  *Das ist das Gate, das über Handelbarkeit entscheidet.* Das Status-quo-Modell
  wird hier voraussichtlich krachend scheitern (es sagt ~4 % und liefert ~13 %);
  wenn das neue Modell hier ebenfalls scheitert, bleibt der harte Veto — dann ist
  H0 wahr und die zerklüfteten Tage sind schlicht nicht bepreisbar.
- **G4 (Praxis, deskriptiv):** Auf die **aktuellen** Leitern (15./16.07.)
  angewandt: Welche heute vetoierten Buckets würden mit ehrlichem Sigma eine
  EV-Marge ≥ 5 pp behalten? Ergebnis wird berichtet, egal wie es ausfällt — es ist
  **kein** Gate (eine leere Liste falsifiziert nichts).
- **G5 (Ehrlichkeit / Forward):** Die Geldfrage („Markt bepreist Zerklüftung
  falsch") wird **hier nicht beantwortet**. Sie wird als Forward-Test auf
  `bb_WeatherLadders` registriert: ab sofort für jeden Lay-Kandidaten Spanne,
  Modell-P (neu) und Marktpreis mitloggen; **Auswertung frühestens ab N = 40
  zerklüfteten Buckets mit bekanntem Settlement.**

## 7. Was bei welchem Ausgang passiert

| Ausgang | Konsequenz |
|---|---|
| G1–G3 bestanden | Screens rechnen künftig mit `sigma(s)`; der harte Spannen-Veto wird durch die EV-Marge ersetzt (Spanne fließt dann über Sigma ein, nicht als Sperre). 37 % der Tage werden wieder handelbar — mit ehrlichen Zahlen. |
| G3 gescheitert (Normal) | Sekundärmodell (empirische z-Quantile) prüfen. Besteht es G3, wird **es** eingebaut. |
| G3 auch dort gescheitert | **H0 gilt.** Harter Veto bleibt, Begründung wird von „plausibel" auf „gemessen" gehoben. Zerklüftete Tage sind für uns nicht handelbar — Punkt. Das wäre ein sauberer FAIL und wird als solcher committet. |

**Vorab-Erwartung (damit sie nicht nachträglich zurechtgebogen wird):** G1 und G2
werden bestehen (der Effekt ist groß und monoton). **G3 ist offen** — ich halte
es für gut möglich, dass die Fehler an zerklüfteten Tagen nicht nur breiter,
sondern auch schief sind (Frontdurchgänge, Regen-Timing), und dass die
Normal-Annahme dort auch mit korrektem Sigma nicht trägt. In dem Fall gewinnt H0.
