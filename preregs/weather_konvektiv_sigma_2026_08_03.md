# Pre-Reg: Konditionales σ — trägt die Konvektionslage Information, die die Modellspanne nicht schon hat?

**Angelegt:** 2026-08-03, vor dem ersten Blick auf die Zielgröße.
**Status:** registriert, noch nicht gefahren.
**Auswertung durch:** `weather_konvektiv_sigma_eval.py` (noch zu schreiben).

## Anlass — und warum die naheliegende Fassung der Frage schon tot ist

Die Warm-Bias-Pre-Reg von heute (`weather_konvektiv_warmbias_2026_08_03.md`, ROT)
hatte einen Nebenbefund: in der Primärgruppe traf der Markt-Favorit den Bucket zu
43,3 % exakt, unser `round(mu_ens)` nur zu 16,7 % (n = 30). Lesart wäre gewesen:
*an Konvektionstagen sind wir blind.*

**Der auslösende Tag selbst widerlegt diese Lesart.** Panama City, 03.08.:

| Größe | Wert |
|---|---|
| Lead-1-Modelle (GFS/ICON/UKMO/JMA/ECMWF) | 32,1 / 30,4 / 33,2 / 30,0 / 29,8 |
| Lead-1-Ensemble-Mittel | **31,10** |
| real (MPMG) | **31,0** |
| Markt-Favorit | 33+ |
| Modellspanne | **3,40 K** → `MAX_SPREAD = 3.0` **gerissen** |

Unser µ war auf 0,1 K genau, der Markt lag daneben — und Screen *wie* Autobuy
hätten den Tag ohnehin mit `skip_spread_3.4` übersprungen
(`weather_minus1_autobuy.py:399`). Die Unsicherheit dieses Tages war also bereits
sichtbar, und zwar über den Kanal, den das System schon benutzt.

**Damit bleibt genau eine nicht-redundante Frage übrig:**

> σ ist seit dem 14.07. spannen-konditioniert — `σ(s) = max(a_city + b·s, SIGMA_FLOOR)`.
> Trägt die vortags erkennbare Konvektionslage Information über die Streuung, die
> **innerhalb gleicher Modellspanne** noch nicht enthalten ist?

Alles andere ist bereits eingepreist. Ein Konvektionseffekt, der nur über die
Spanne wirkt, ist kein Fund, sondern eine Umbenennung.

## Warum die Frage überhaupt Geld wert wäre

Nicht als weiteres Veto — davon haben wir genug. Sondern in der **Gegenrichtung**:
[[weather-sigma-zu-gross]] hat am 02.08. in zwei unabhängigen Tests gezeigt, dass
σ zu groß ist (genaueste Zellen 0,70–0,75×, Ränder 9,3 % modelliert gegen 5,8 %
real). `P_pess` verwirft deshalb zu viel. Wenn die Konvektionstage diejenigen
sind, die σ aufblähen, dann darf σ an den **klaren** Tagen kleiner sein — und
genau dort werden zusätzliche Zellen handelbar. Das zahlt auf die Breiten-
Skalierung ein ([[weather-scaling-plan]]: Slippage begrenzt die Tiefe, also muss
die Breite wachsen).

## Zielgröße

Je Stadttag der **standardisierte Fehler**

    z = (actual − mu_ens) / σ_ens

mit `mu_ens` und `σ_ens` **exakt wie im Live-Screen** (bias-korrigiertes
Ensemble-Mittel nach `OUTLIER_DEG`-Bereinigung; σ aus `ens_sigma()`).
Ist σ richtig, gilt sd(z) = 1. Bekannt ist sd(z) < 1 global.

**Hypothese H:** sd(z | konvektiv) / sd(z | klar) > 1 — **innerhalb gleicher
Spannen-Klassen** gemessen.

## Datenbasis (Machbarkeit vorab geprüft)

- **Residuen:** `weather_source_compare.py --days 700 --var max --lead 1
  --dump-residuals` liefert `(city, model, date, forecast, actual)` je Tag. Die
  Kalibrier-CSVs enthalten **nur** `bias/sigma/a/b` je Stadt-Modell — die
  Tagesreihe muss also neu gezogen werden, sie existiert nirgends als Cache.
- **Bedingung, handelbar:** Open-Meteo previous-runs, `previous_day1` —
  `regen_mm` = Σ Niederschlag 06–18 h lokal, `wolken_tag` = ⌀ Bewölkung 09–18 h
  lokal. Beides steht am Vortag fest.
  **Archivtiefe geprüft:** Werte ab ~Juni 2024 (2024-06-01 liefert, 2024-01-01
  nicht). Das 700d-Fenster ab 2024-09-02 liegt vollständig darin — aber knapp;
  ein längeres Fenster ist mit dieser Quelle nicht zu haben.
- **Umfang:** ~28 Städte × 700 Tage ≈ **19.000 Stadttage**, gegen 39 in der
  Warm-Bias-Pre-Reg.
- Konvektiv := `regen_mm ≥ 1` ∧ `wolken_tag ≥ 60` (identisch zur Warm-Bias-Pre-Reg,
  bewusst nicht neu justiert).

## Gates — bewusst NICHT auf t-Werten

Bei n ≈ 19.000 ist jede noch so kleine Differenz signifikant. Ein t-Gate wäre
hier reine Dekoration. Die Gates messen deshalb **Effektgröße, OOS-Bestand und
Nutzen gegen eine faire Referenz**.

- **G1 Materialität (IS: 2024-09 … 2025-12):**
  r = sd(z|konv) / sd(z|klar) ≥ **1,15**, gebildet als gewichtetes Mittel
  **innerhalb der Spannen-Terzile**. Darunter lohnt keine zweite Stellschraube.
- **G2 Bestand (OOS: 2026-01 … 2026-08):** r ≥ **1,10**, ohne Nachjustierung von
  Schwellen oder Terzilgrenzen.
- **G3 Nutzen gegen die faire Referenz:** ein konditionales σ (zwei Faktoren,
  **nur IS** gefittet) muss OOS besser kalibrieren als
  (a) das heutige σ **und** (b) ein **globaler** Skalierungsfaktor.
  (b) ist die eigentliche Hürde: σ ist ohnehin zu groß, ein flacher Faktor holt
  einen Teil des Gewinns umsonst. Maß: |sd(z) − 1| je Gruppe + Abdeckung des
  80-%-Intervalls. Schlägt die Konditionierung den flachen Faktor nicht, ist sie
  Zierat und die These fällt.
- **G4 Handelsnutzen (der eigentliche Punkt):** OOS mindestens **+10 %**
  zusätzlich handelbare Zellen an klaren Tagen unter `MAX_PMODEL = 0.10`,
  **ohne** dass die realisierte Trefferquote unter Break-even 22,6 % fällt.
- **G5 Robustheit:** Leave-one-city-out — Vorzeichen hält in ≥ 80 % der Städte;
  **und** gepaart innerhalb **Stadt-Monat**.

## Zwei Fallen, die den Test sonst wertlos machen

1. **Saison-Konfundierung.** Konvektionstage sind überwiegend Sommertage. Wer
   Konvektions- gegen klare Tage über das ganze Jahr vergleicht, misst den
   Jahresgang von σ, nicht die Konvektion. Deshalb ist die Paarung **innerhalb
   Stadt-Monat** in G5 kein Zusatz, sondern die Bedingung dafür, dass G1
   überhaupt etwas bedeutet.
2. **Zirkularität.** `a_city` und `b` stammen aus derselben 700d-Kalibrierung.
   Wer z mit einem σ bildet, das auf **denselben** Tagen gefittet wurde,
   unterschätzt die Streuung systematisch. Deshalb: Koeffizienten aus dem
   IS-Fenster, im OOS-Fenster **eingefroren** (`--fix-b-from`, dafür gebaut).

## Was PASS bedeutet — und was ausdrücklich nicht

PASS heißt: σ darf konditional gesetzt werden, klar kleiner / konvektiv größer.
**Es heißt nicht, dass danach an σ gedreht wird.** Der Nutzer hat am 02.08. „nicht
eigenmächtig an σ drehen" festgehalten ([[weather-sigma-zu-gross]]); vor einer
Live-Änderung läuft ein Forward-Test, wie bei den vier laufenden Linien.

FAIL heißt: die Modellspanne enthält die Konvektionsinformation bereits, das
System ist an diesen Tagen richtig gebaut, und der 16,7-%-Nebenbefund war ein
Kleinserien-Artefakt. Auch das ist ein verwertbares Ergebnis — es schließt eine
Baustelle, statt eine zu eröffnen.

## Bekannte Schwächen (vorab benannt)

- Der auslösende Nebenbefund hat **n = 30**; bei sd = 1,0 ist das ein
  Zufallsbereich von ±0,36. Diese Pre-Reg prüft die **Mechanik**, nicht die Zahl.
- Mehrere Stationen melden in **ganzen Grad** (Panama MPMG, Seoul) → z ist dort
  quantisiert. Betrifft beide Gruppen gleich, verwässert r aber nach unten.
- 700d ist durch die Archivtiefe der Vortagesprognose hart begrenzt, nicht durch
  eine Wahl. Ein längeres Fenster ginge nur mit einer anderen Quelle.
- `--dump-residuals` über 28 Städte × 5 Modelle × 700 Tage ist ein Messlauf von
  einigen Minuten mit vielen API-Calls; die Netzabbruch-Toleranz des Skripts
  (Stadt überspringen) darf **nicht** dazu führen, dass IS und OOS
  unterschiedliche Städtemengen haben — vor der Auswertung abgleichen.

---

# ERGEBNIS (gefahren 2026-08-04, `weather_konvektiv_sigma_eval.py`)

**G1 ROT · G2 ROT · G3 GRÜN · G4 ROT · G5 ROT — FAIL.**

Datenlage besser als erhofft: die Previous-Runs-API lieferte **alle** 700 Tage
für **alle 29** Städte, 20.300 Stadttage, davon 19,3 % konvektiv. Die vorab
benannte Sorge, IS und OOS könnten unterschiedliche Städtemengen haben, ist
gegenstandslos — es fehlt keine Stadt und kein Tag. Nach dem Join mit dem
Residuen-Dump bleiben 16.272 Stadttage (10.963 IS / 5.309 OOS), der
Konvektionsanteil ist in beiden Fenstern praktisch gleich (19,1 % / 19,5 %).

## G1 — ROT. Der Effekt existiert, ist aber ein Drittel der geforderten Größe

r = **1,053** gegen geforderte 1,15.

| Spannen-Terzil | n konv | n klar | sd konv | sd klar | r |
|---|---|---|---|---|---|
| < 1,90 K | 658 | 2.940 | 1,014 | 0,980 | 1,035 |
| < 3,20 K | 734 | 2.908 | 1,059 | 0,971 | 1,090 |
| darüber | 703 | 3.020 | 1,047 | 1,015 | 1,031 |

Die Richtung stimmt in allen drei Terzilen — konvektive Tage streuen wirklich
etwas breiter. Aber die Größe liegt bei 3–9 %, und die Pre-Reg hatte 15 %
verlangt mit der Begründung „darunter lohnt keine zweite Stellschraube". Das
Urteil steht damit unabhängig von jeder Signifikanzfrage.

## G2 — ROT. Im OOS bleibt fast nichts, und das dritte Terzil dreht

r = **1,029** gegen geforderte 1,10. Im obersten Spannen-Terzil kippt das
Verhältnis auf **0,984** — dort streuen konvektive Tage sogar knapp *enger* als
klare. Die Datenlage trägt: 1.035 konvektive OOS-Tage, 294–387 je Terzil, also
weit über der Mindestgröße. Die vorab benannte Sorge, das OOS-Fenster liege
außerhalb der Konvektionssaison, hat sich nicht bestätigt.

## G3 — GRÜN, aber die Referenz ist kollabiert und das Gate deshalb wertlos

| Variante | sd konv | sd klar | Σ\|sd−1\| | Deckung konv | Deckung klar |
|---|---|---|---|---|---|
| heutiges σ | 1,092 | 1,060 | 0,153 | 78,1 % | 80,5 % |
| globaler Faktor | 1,092 | 1,060 | 0,153 | 78,1 % | 80,5 % |
| konditional | 1,049 | 1,071 | 0,120 | 79,6 % | 79,8 % |

Die konditionale Fassung schlägt beide Referenzen — aber **die beiden Referenzen
sind identisch**. Der globale Faktor kommt aus dem IS, und weil σ(s) dort per
MLE gefittet ist, gilt dort sd(z) = 1,000 exakt; der flache Faktor ist damit
1,000 und ändert nichts. Die Pre-Reg hatte (b) als „die eigentliche Hürde"
bezeichnet, weil „σ ohnehin zu groß ist und ein flacher Faktor einen Teil des
Gewinns umsonst holt". Diese Annahme trifft auf ein **sauber IS-gefittetes** σ
nicht zu — sie beschreibt die ausgelieferten Kalibrier-CSVs, nicht die Form
σ(s). G3 hat also nicht gemessen, was es messen sollte, und trägt nichts zum
Urteil bei.

## G4 — ROT, und hier stirbt die ökonomische Begründung

| | Wert |
|---|---|
| klare Stadttage OOS | 4.274 |
| handelbare Zellen heute | 22.073 |
| konditional | 22.146 (**+0,3 %**, verlangt +10 %) |
| davon neu geöffnet | 76 |
| davon getroffen | 7 = 9,2 % (Break-even 22,6 %) |

Der Grund ist simpel: der konditionale Faktor für klare Tage ist **0,990** — σ
schrumpft um ein Prozent. Damit kippen 76 von 22.073 Zellen über die
0,10-Schwelle. Die neu geöffneten Zellen sind *sauber* (9,2 % gegen 22,6 %
Break-even), es sind nur viel zu wenige, um irgendetwas zu bewegen. Der ganze
Zweck der Pre-Reg — „an klaren Tagen darf σ kleiner sein, dort werden zusätzliche
Zellen handelbar" — scheitert nicht an der Richtung, sondern an der Größe.

## G5 — ROT, und das ist der Befund, auf den es ankommt

Leave-one-city-out ist unauffällig: das Vorzeichen hält in **28 von 29** Städten
(97 %, verlangt 80 %). Das Ergebnis hängt an keiner Stadt.

Die **Saison-Paarung** dagegen fällt in sich zusammen:

    248 Stadt-Monat-Paare · sd(konv) − sd(klar) = +0,016 · t = +0,63
    129/248 positiv = 52 %

Ein Münzwurf. Und genau diese Paarung hatte die Pre-Reg als **Bedingung** dafür
benannt, „dass G1 überhaupt etwas bedeutet". Hält man Stadt und Monat fest,
verschwindet der Effekt vollständig — die 5 %, die G1 misst, sind Jahresgang:
konvektive Tage sind Sommertage, und der Sommer hat ein anderes σ.

## Antwort auf die Frage der Pre-Reg

**Die Modellspanne enthält die Konvektionsinformation bereits.** Das System ist
an diesen Tagen richtig gebaut, und der auslösende 16,7-%-Nebenbefund (n = 30)
war ein Kleinserien-Artefakt, wie im Schwächen-Abschnitt vorab vermutet. Das
schließt eine Baustelle, statt eine zu eröffnen — was die Pre-Reg ausdrücklich
als verwertbares Ergebnis vorgesehen hatte.

Auch die Konsequenz war vorab geregelt und gilt: an σ wird nichts gedreht.

## Ein Nebenbefund, der einen offenen Faden berührt — ungeprüft

OOS steht **sd(z) = 1,067**, also σ etwas zu **klein** — die entgegengesetzte
Richtung zu [[weather-sigma-zu-gross]] (02.08.: genaueste Zellen 0,70–0,75×,
Ränder 9,3 % modelliert gegen 5,8 % real).

Das ist **kein Widerspruch und keine Korrektur**, sondern eine offene Frage: die
beiden Messungen sind nicht dieselbe Größe. Hier steht das zweite Moment über
alle Tage, dort die Masse in bestimmten Wahrscheinlichkeitsbändern. Beides
zugleich ist möglich, wenn die Fehlerverteilung nicht gaußisch ist — mehr Masse
im Zentrum UND in den Rändern, weniger auf den Schultern. Der zweite Unterschied
ist die Quelle: hier ist σ(s) per MLE auf dem IS-Fenster gefittet, dort kamen a
und b aus den ausgelieferten Kalibrier-CSVs.

**Zu prüfen wäre also, ob „σ ist zu groß" eine Aussage über die FORM σ(s) ist
oder nur über die ausgelieferten CSV-Koeffizienten.** Das ist eine eigene
Messung und eine eigene Pre-Reg, nicht Teil dieser hier.
