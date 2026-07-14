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

---

# ERGEBNIS (gerechnet 2026-07-14, `weather_spread_sigma_fit.py` + `_diag.py`)

**Datenbasis:** 15.371 Stadt-Tage, 28 Städte, 700 d. Split wie registriert:
IS bis 2025-12-05 (n = 10.046), OOS ab 2025-12-05 (n = 5.325).

## Kurzfassung: H ist gespalten — der Messteil hält, der operative Kern ist FALSIFIZIERT

Sigma hängt tatsächlich stark von der Spanne ab, und `sigma(s)` macht das Modell
messbar ehrlicher. **Aber der Zweck der Übung — den harten Veto durch korrekte
Preise zu ersetzen — scheitert.** `sigma(s)` verbreitert die Glocke; der Schaden
der Zerklüftung sitzt jedoch im **Mittelwert**, nicht in der Breite. Ein
symmetrisch verbreitertes Sigma um ein µ, das ein Ausreißer verzogen hat,
schützt nicht.

## G1 (IS): BESTANDEN

Empirisches Sigma über die Spannen-Bins, monoton:

| Spanne | n | Sigma |
|---|---|---|
| 0–1,5 ° | 1.748 | 0,934 ° |
| 1,5–3 ° | 4.304 | 1,109 ° |
| 3–5 ° | 2.888 | 1,459 ° |
| > 5 ° | 1.106 | 1,726 ° |

MLE: **`sigma_city(s) = a_city + 0,147 · s`**, Block-Bootstrap-SE 0,007 →
**t = 21,4** (verlangt: t > 4). a_city: min 0,51 / median 0,71 / max 1,71.

## G2 (OOS): BESTANDEN

Log-Loss der Bucket-Wahrscheinlichkeiten: Status quo **1,6826** → sigma(s)+Normal
**1,6547** → sigma(s)+empirische z **1,6441**. Das neue Modell ist besser.

## G3 wie vorregistriert: BESTANDEN — aber das Gate war UNTAUGLICH

Alle drei Modelle bestehen, **einschließlich des Status quo** — den wir aus dem
Beijing-Verlust als fehlkalibriert kennen. Ursache: Das Gate mittelt über die
gesamte Lay-Zone (P ≤ 10 %) und wird von den vielen Buckets mit P ≈ 0 dominiert
(mittlere vorhergesagte P nur ~1,3 %). **Das ist ein Design-Fehler dieser Pre-Reg,
kein Erfolg.** Er wird hier als solcher protokolliert, nicht stillschweigend durch
ein besseres Gate ersetzt.

**Bin-weise Reliability (EXPLORATIV, nachträglich — kein Gate-Pass):** Erst der
vorab deklarierte **Sommer-Schnitt** legt den echten Defekt frei:

| Modell | Bin | Regime | vorherg. | realisiert | Faktor |
|---|---|---|---|---|---|
| **Status quo** | 2–5 % | **zerklüftet** | 3,38 % | **5,15 %** | **1,52×** |
| Status quo | 2–5 % | ruhig | 3,31 % | 1,41 % | 0,43× |
| Status quo | 5–10 % | ruhig | 7,36 % | 4,41 % | 0,60× |
| sigma(s)+emp. z | 5–10 % | zerklüftet | 7,26 % | 7,38 % | 1,02× |
| sigma(s)+emp. z | 5–10 % | ruhig | 7,20 % | 6,95 % | 0,97× |
| sigma(s)+emp. z | 10–20 % | zerklüftet | 14,95 % | 15,00 % | 1,00× |

Der Status quo ist im Sommer **zweifach falsch**: auf zerklüfteten Tagen
**überkonfident** (1,52× — genau Zone und Regime des Beijing-Trades) und auf
ruhigen Tagen **viel zu breit** (0,43–0,60×). Das feste Sigma ist eben ein
Mittelwert. `sigma(s)` + empirische z-Quantile repariert beide Seiten.

## G4 (deskriptiv): das Killer-Ergebnis

Auf die vom Spannen-Veto blockierten Buckets des 16.07. angewandt:

| Bucket | Spanne | sigma heute | sigma(s) | P heute | P mit sigma(s) | BE | EV neu | |
|---|---|---|---|---|---|---|---|---|
| Tokyo 33° | 6,2 ° | 1,05 | 1,59 | 5,7 % | 10,7 % | 22,0 % | +11,3 pp | käme durch |
| Tokyo 32° | 6,2 ° | 1,05 | 1,59 | 0,6 % | 4,0 % | 13,0 % | +9,0 pp | käme durch |
| Milan 32° | 8,0 ° | 0,73 | 1,91 | 6,2 % | 14,6 % | 22,0 % | +7,4 pp | käme durch |
| **Beijing 32°** | 5,2 ° | 1,43 | 1,69 | 11,1 % | 12,1 % | 18,0 % | **+5,9 pp** | **käme durch** |
| Jeddah 36° | 8,2 ° | 1,31 | 1,73 | 3,2 % | 6,2 % | 26,0 % | +19,8 pp | käme durch |

**Alle fünf.** Inklusive Beijing 32° — dem Bucket mit exakt der Signatur des
Verlierers (JMA roh 38,4 gegen 33,2–34,5 der anderen vier). Hätte man den Veto
durch `sigma(s)` ersetzt, hätte man genau die Trades wieder geöffnet, die Geld
gekostet haben.

## Gegencheck (explorativ): Hätte sigma(s) den Beijing-33-Verlust verhindert?

**Nein.** Beijings Spanne war 3,6 °, sein Median liegt bei 2,8 ° — sigma steigt
dadurch nur von 1,32 auf 1,45. P(33er) geht von 4,5 % auf 5,6 %; bei BE 21 %
bleibt der Lay in **allen vier** Varianten (volles/robustes ENS × festes/neues
Sigma) klar +EV. Was den Trade wirklich kippte, war das **µ**: Ausreißer raus
(−0,6 °) und Sommer- statt Ganzjahres-Bias (−0,9 °) → P 20,3 %.

**Rangfolge der Fehlerquellen beim Verlust-Trade:**
1. Bias-Vorzeichenwechsel 700d → 40d (0,9 ° in µ)
2. JMA-Ausreißer im Ensemble-Mittel (0,6 ° in µ)
3. Sigma (praktisch irrelevant: 1,32 → 1,45)

## Verdikt

**H0 gewinnt — aber aus einem anderen Grund als vermutet.** Die Vorab-Erwartung
lautete: „die Fehler an zerklüfteten Tagen sind womöglich schief, dann trägt die
Normal-Annahme auch mit korrektem Sigma nicht." Falsch geraten: Die Fehler sind
mit `sigma(s)` sogar sehr gut kalibriert (1,00–1,02× in den relevanten Bins). Der
wahre Grund ist ein anderer: **Die Spanne ist nicht (nur) ein Breiten-Signal,
sondern vor allem ein Warnsignal für ein korrumpiertes µ.** Kein Sigma der Welt
repariert einen verzogenen Mittelwert.

## Konsequenzen

1. **Der harte Spannen-Veto BLEIBT.** Er wird *nicht* durch `sigma(s)` ersetzt.
   Das ist eine bewusste **Abweichung von der in §7 vorregistrierten Aktion** —
   die war unter der Annahme formuliert, ein bestandenes G3 mache zerklüftete
   Tage handelbar. G4 zeigt, dass das nicht stimmt. Einer schlecht entworfenen
   Regel blind in einen Verlust zu folgen wäre absurd; die Abweichung wird hier
   offen protokolliert statt kaschiert.
2. **`sigma(s)` ist trotzdem wertvoll — aber für die andere Richtung.** Auf
   *ruhigen* Sommertagen ist das feste Sigma **1,7–2,3× zu breit**. Der Screen
   rechnet dort also zu hohe P und **lässt sichere Lays liegen**. Das ist die
   eigentliche Ausbeute dieser Studie: nicht mehr Risiko zulassen, sondern die
   ruhige Zone schärfer bepreisen. **Eingebaut** — siehe unten.
3. **G5 (Forward) bleibt wie registriert:** Ob der Markt Zerklüftung falsch
   bepreist, ist weiterhin unbeantwortet und braucht `bb_WeatherLadders`-Daten
   (N ≥ 40 zerklüftete Buckets mit Settlement).

---

# EINBAU (14.07., nach der Auswertung)

`sigma(s)` steckt jetzt in beiden Screens (`ens_sigma()` in
`weather_outlier_screen.py`). Drei Dinge, die beim Bauen erst auffielen:

**1. Die Steigung ist saisonabhängig — also darf sie keine Code-Konstante sein.**
Gemessen: Sommer **0,107**, Winter **0,177**, Ganzjahr **0,140**. Hätte man dem
40d-Sommer-Fenster die Ganzjahres-Steigung aufgezwungen, käme Sigma auf ruhigen
Sommertagen ~5 % **zu eng** heraus — also in die gefährliche Richtung.
Konsequenz: `weather_source_compare.py` fittet `b` **je Kalibrierfenster neu**
(gemeinsam über alle Städte, `a` je Stadt) und schreibt beide in die CSV. Der
Screen liest sie von dort. Die tatsächlich gefitteten Werte:

| Fenster | b |
|---|---|
| Tageshoch, 700 d (Ganzjahr) | 0,140 |
| Tageshoch, 40 d (Sommer) | **0,099** |
| Tagestief, 700 d | 0,082 |
| Tagestief, 40 d | 0,074 |

Das Tagestief hat also eine **halb so große** Spannen-Sensitivität wie das
Tageshoch. Eine hartkodierte 0,147 wäre für den Low-Screen fast doppelt falsch
gewesen.

**2. Beide Sichten mussten umgestellt werden, nicht nur eine.** Der Screen nimmt
das pessimistischste P über 700d *und* 40d. Hätte nur eine Sicht `sigma(s)`
bekommen, hätte die andere mit ihrem breiten Fest-Sigma weiter dominiert und die
Änderung wäre wirkungslos geblieben.

**3. Einzelmodelle behalten ihr festes Sigma.** `sigma(s)` gilt nur für die
ENS-Sicht. Die Einzelmodell-Schranke (`MAX_PMODEL`) bleibt unangetastet — sie ist
der Wächter gegen genau die dissentierende Stimme, die den Beijing-Trade gekippt
hat.

## Wirkung (Leitern 15./16.07.)

Auf ruhigen Tagen steigt die EV-Marge deutlich — σ wird bei 1,5 ° Spanne um
11–21 % enger, und weil P in den Flanken stark auf σ reagiert, schlägt das durch:

| Bucket | Spanne | EV vorher | EV mit sigma(s) |
|---|---|---|---|
| Tel Aviv 31° (16.07.) | 1,8 ° | +16,9 pp | **+21,3 pp** |
| Madrid 34° (16.07.) | 1,4 ° | +5,4 pp | **+12,3 pp** |
| Madrid 35° (15.07.) | 1,5 ° | +5,3 pp | **+7,3 pp** |
| Wuhan 38° (15.07.) | 1,4 ° | +3,6 pp | **+5,8 pp** |

**Trotzdem null Kandidaten** an beiden Tagen: Alle scheitern jetzt an `MIN_DIST`
(1,4–1,9 ° < 2,0 °) und am Einzelmodell-Veto — durchgehend **ICON** (32–41 %).
Diese Schranken wurden bewusst **nicht** angefasst; sie zu lockern, um Kandidaten
zu erzeugen, wäre genau der Fehler, um den es in diesem Kapitel geht. Der Nutzen
von `sigma(s)` ist kumulativ und zeigt sich an Tagen, an denen die Modelle einig
sind *und* der Markt trotzdem Restangst preist.

## Regressionsschutz

`weather_screen_selftest.py` friert die Forecasts ein, die beim Beijing-33-Trade
live waren, und verlangt, dass der Screen sie ablehnt. Läuft nach dem Einbau grün
(EV jetzt +0,1 pp statt +0,7 pp — noch klarer abgelehnt). Der Test ist bewusst
kein Unit-Test der Einzelfunktionen, sondern des **Urteils**: „Hätten wir den
Trade heute noch gemacht?"
