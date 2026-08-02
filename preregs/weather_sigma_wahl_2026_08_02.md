# Pre-Registrierung: Welches σ ist das richtige? — 2026-08-02

**Status:** Vorregistrierung. Geschrieben, **bevor** eine der drei σ-Varianten
gegen das Kalibrierungskriterium gerechnet wurde. Auswertung folgt in
`weather_sigma_wahl_eval.py`.

---

## Anlass

Zwei unabhängige Messungen desselben Tages sagen dasselbe: **unser σ ist zu
groß.**

1. **Zellenauswahl** (60 Zellen, zwei Jahreszyklen): im besten Terzil sagt das
   Modell 3,33 % vorher und es treten 2,48 % ein (Faktor **0,75**), im Bin
   5–10 % sind es 7,29 % gegen 5,08 % (**0,70**).
2. **Ursachen-Messung** (325 Stadt-Tage, Juli 2026), und dort besonders
   aussagekräftig, weil sie die **ganze Leiter** zeigt:

   | Offset | P_modell | P_ist | |
   |---|---|---|---|
   | 0 | 30,5 % | **32,6 %** | Modell zu niedrig |
   | ±1 | 22,5 / 22,0 % | 23,7 / 24,6 % | Modell zu niedrig |
   | ±2 | 9,3 / 8,9 % | **5,8 / 7,1 %** | Modell zu hoch |

   **Zu wenig Masse im Zentrum, zu viel an den Rändern — das ist die Signatur
   einer zu breiten Verteilung**, nicht einer falschen Verteilungsform. Wäre die
   Form schuld, müsste das Zentrum stimmen.

**Die Wurzel ist in beiden Fällen dieselbe: ein Ganzjahres-σ.** Der Ladder-Logger
schreibt `sigma_ens` aus der 700d-Basis; das Zellen-Eval rechnet σ als
Standardabweichung über das **komplette** IS-Jahr (`sigma_of()`), obwohl der Bias
dort gleitend über 40 Tage korrigiert wird. Beide mitteln über Jahreszeiten mit
verschieden großer Streuung.

Gemessen über alle 31 Städte mit beiden Kalibrierungen: **σ(700d) = 1,282 K
gegen σ(40d Sommer) = 1,072 K**, in 25 von 31 Städten ist 700d größer, mittlere
Differenz **+0,210 K**. Extremfälle Tel Aviv (Faktor 2,22), Seoul (1,88), Taipei
(1,77). Die Korrelation zwischen |Bias-Differenz| und σ-Differenz beträgt
**r = +0,479**.

## Was hier NICHT behauptet wird

**Nicht: „der Bot verliert dadurch Geld."** Am 02.08. am Code verifiziert:
`weather_minus1_autobuy.py` verwendet σ **überhaupt nicht** — er filtert über
Preisband, Spannen-Veto, Temperaturabstand, Cap und Guthaben. Ein σ-Fehler
kostet keine Bot-Position. **Er verzerrt die Screens** (manuelle Kandidatensuche)
und jede Auswertung, die `sigma_ens` benutzt. Der Nutzen dieser Pre-Reg ist
Mess- und Entscheidungsqualität, **kein direkter Ertrag**.

**Nicht: „40d ist die Antwort."** Siehe Gegenrechnung 2 — die Überschlagsrechnung
sagt, dass 40d überschießen dürfte. Und für den **Bias** ist 40d im Lay-Buch
bereits falsifiziert ([[weather-40d-schlaegt-700d]], GATE t = −0,16); das betraf
die Bucket-Wahl, nicht die Streuung, aber es ist eine Warnung.

**Nicht: „hier wird σ umgestellt."** Aus dieser Messung folgt ein Vorschlag, keine
Änderung. Die Screens sind live.

Sondern: **Welche der drei σ-Definitionen ist am besten kalibriert?**

---

## Die drei Kandidaten — vorab fixiert, genau drei

| | Definition | Status heute |
|---|---|---|
| **A** | **Ganzjahres-σ**: Standardabweichung aller bias-korrigierten Residuen des vorangehenden Jahres | **Referenz** — so rechnen Logger und Zellen-Eval heute |
| **B** | **Gleitendes 40-Tage-σ**: Standardabweichung der Residuen der letzten 40 Tage **vor** dem Zieltag | die saisonale Korrektur |
| **C** | **σ(s) = a + b·Spanne**: aus der Modellspanne des Tages, Fit auf dem IS-Fenster | seit 17.07. in den Screens, aber **nicht** im Logger |

**Es werden keine Mischungen, keine Gewichtungen und keine vierte Variante
geprüft.** Der `SIGMA_FLOOR` von 0,3 K bleibt bei allen dreien identisch.

---

## Die Gegenrechnung der Gates

*(Pflichtübung seit dem Prüfstunden-Fehler: jedes Gate vor dem Festschreiben
gegen die offengelegten Zahlen gegenrechnen.)*

**Gegenrechnung 1 — taugt das Kriterium?** Gemessen wird bin-weise wie in der
Zellen-Pre-Reg (Bins 2–5 %, 5–10 %, 10–20 %, Band [0,75×; 1,25×]). Aus dem
Zellenlauf ist bekannt, dass der Bin **10–20 % in allen drei Terzilen leer**
war. Ein Gate, das ihn verlangt, wäre unerfüllbar. **Konsequenz, vorab gezogen:
G1 stützt sich auf die Bins 2–5 % und 5–10 %**; der dritte wird berichtet, wenn
er sich füllt, trägt aber kein Gate.

**Gegenrechnung 2 — wird B das Ziel treffen?** Vermutlich nicht. Für einen
Bucket mit modellierter Wahrscheinlichkeit 7,3 % bei σ = 1,282 liegt der
z-Abstand bei rund 1,86. Rechnet man denselben physischen Abstand gegen
σ = 1,072, ergibt sich **etwa 3,1 %** — also Faktor ≈ 0,43 statt der gemessenen
0,70. **Die volle 40d-Reduktion überschießt.** Das ist der Grund, warum C
überhaupt im Rennen ist: eine tagesweise Anpassung könnte treffen, wo eine
pauschale Verkleinerung vorbeischießt. **Und es ist der Grund, warum G1 als
Abstand zum Idealwert 1,0 formuliert ist und nicht als „kleiner ist besser".**

**Gegenrechnung 3 — Look-ahead.** Die vorhandenen 40d-CSVs sind auf ein
**festes** Sommerfenster (Stand 17.07.) gefittet und enthalten für Juli-Zieltage
Daten aus der Zukunft. Sie dürfen deshalb **nicht** verwendet werden. B und C
werden ausschließlich **walk-forward** aus dem Cache gerechnet, mit Tagen vor dem
Zieltag. Der Cache trägt je Stadt-Tag `(µ_roh, Modellspanne, Ist)` über 542 Tage
je Zelle — genug für beides.

---

## Hypothese und Gates

**H1:** Mindestens eine der Varianten B oder C ist über die geprüften Bins besser
kalibriert als A, gemessen als **mittlerer absoluter Abstand des Faktors zu 1,0**.

**H0:** Alle drei liegen ähnlich daneben. Dann ist die Streuung nicht das
Problem, sondern etwas, das keine dieser Definitionen erfasst — und der Befund
lautet, dass σ ein offener Posten bleibt.

| Gate | Bedingung |
|---|---|
| **G0** Basis | ≥ 50 Zellen mit ≥ 120 Tagen im OOS-Fenster; je Bin ≥ 2.000 Beobachtungen je Variante |
| **G1** Kalibrierung | Der Gewinner liegt in **beiden** tragenden Bins (2–5 %, 5–10 %) im Band [0,75; 1,25] **und** hat einen kleineren mittleren Abstand zu 1,0 als A |
| **G2** Deutlichkeit | Der Abstandsvorteil gegenüber A beträgt **≥ 0,10** (in Faktor-Einheiten). Ein Gewinn von 0,02 wäre Rauschen |
| **G3** Breite | Der Gewinner ist in **≥ 70 % der Zellen einzeln** besser als A — nicht nur im Aggregat. Sonst trägt ihn eine Handvoll Städte |
| **G4** Kein Schaden am Zentrum | Die realisierte Trefferquote des **Favoriten-Buckets** weicht bei der Gewinner-Variante um ≤ 3 pp von der vorhergesagten ab. Ein σ, das die Ränder repariert und dafür das Zentrum verzieht, ist keine Verbesserung |

**Bonferroni:** Drei Varianten gegen eine Referenz, das sind zwei Vergleiche. Die
Schwelle in G2 ist deshalb absolut gesetzt (≥ 0,10) statt über ein t — bei
Hunderttausenden Bucket-Beobachtungen wäre jedes t astronomisch und würde
Winzigkeiten signifikant machen. **Die Größe des Effekts entscheidet, nicht seine
Signifikanz.**

---

## Designfallen

**1. Look-ahead** — siehe Gegenrechnung 3. Die einzige, die das Ergebnis
komplett wertlos machen würde.

**2. σ und Bias hängen zusammen.** Wird σ aus 40 Tagen geschätzt, stammt der
Bias aus denselben 40 Tagen — beide Fehler korrelieren. Vorab festgelegt: **der
Bias bleibt in allen drei Varianten identisch** (gleitend 40 Tage, wie im
Zellen-Eval), variiert wird **ausschließlich** σ. Sonst misst der Test zwei
Änderungen gleichzeitig.

**3. Ein kleineres σ ist nicht automatisch besser.** Es verschiebt Masse ins
Zentrum. Wenn das Zentrum vorher schon zu niedrig war (und das war es: 30,5 %
modelliert gegen 32,6 % real), hilft das — bis es kippt. **G4 ist der Wächter
dagegen.**

**4. Die Ist-Werte sind ganzzahlig.** Der Cache führt das gerundete Tagesextrem,
weil die Bretter so auflösen. Die Bucket-Wahrscheinlichkeit wird deshalb über
`[k−0,5; k+0,5]` integriert — für Hong Kong gälten floor-Buckets, aber die Zelle
ist im Cache mit derselben Konvention geführt wie alle anderen. **Das ist eine
bekannte Ungenauigkeit für genau eine Stadt** und wird nicht nachträglich
repariert.

**5. Zwei Jahreszyklen sind zwei Jahreszyklen.** Ein σ, das für 2024/25 und
2025/26 passt, muss für 2026/27 nicht passen.

---

## Vorab-Erwartung

**Ich erwarte C als Gewinner, B als Überschießer, A als Verlierer** — aus
Gegenrechnung 2. Eine pauschale Verkleinerung um 16 % trifft ein Ziel nicht, das
bei Faktor 0,70 liegt; eine tagesweise Anpassung an die Modellspanne kann es.

**G2 halte ich für gefährdet.** Der Unterschied zwischen den Varianten könnte
kleiner ausfallen als 0,10, weil alle drei denselben Ganzjahres-Bias verwenden
und σ nur einen Teil des Fehlers ausmacht.

**Was das Ergebnis wert ist:** Eine bessere Zahl in den Screens und in künftigen
Auswertungen. **Kein Euro** — der Bot verwendet σ nicht. Wer hier einen Ertrag
erwartet, hat den Abschnitt „Was hier nicht behauptet wird" nicht gelesen.

## Abbruchregel

Reißt **G1** für beide Varianten, bleibt σ wie es ist, und der Befund lautet:
die Streuung ist nicht über eine andere Fensterwahl zu reparieren. Es wird
**nicht** auf Mischungen, Skalierungsfaktoren oder eine vierte Definition
ausgewichen.

Reißt nur **G2** (Vorteil zu klein), lautet der Befund „richtige Richtung, zu
wenig Wirkung" — und σ bleibt ebenfalls unverändert, weil eine Umstellung der
Screens Aufwand und Risiko gegen einen Gewinn im Rauschbereich tauschen würde.

Besteht G1–G4, folgt ein **Vorschlag**: `sigma_ens` im Ladder-Logger und die
Screens auf die Gewinner-Variante umzustellen. Nicht als stille Änderung —
die Screens sind live, und ein anderes σ verändert sofort, welche Kandidaten sie
zeigen.

**Am Autobuy ändert sich in keinem Ausgang etwas.** Er verwendet σ nicht.
