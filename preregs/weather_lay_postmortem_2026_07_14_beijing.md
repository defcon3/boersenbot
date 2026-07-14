# Post-Mortem: Beijing-33°-NO-Lay (Zieltag 14.07.2026) — der Lay war falsch, nicht unglücklich

**Kurzfassung:** Das Ist-Hoch an ZBAA war am 14.07. **33,0 °C** (Wunderground und
METAR, je 37 Meldungen, identisch) — exakt im Verlustfenster 32,5–33,49. Der Lay
hätte **−5,67 $** verloren. Verkauft hat ihn der Autopilot-Take-Profit am 14.07.
02:44 UTC zu 0,88 → **+0,51 $** realisiert. Differenz zum gehaltenen Ausgang:
**6,18 $**.

Die Trade-Reg (`weather_lay_bets_2026_07_14.md`) führte den Trade als sauberen
Doppel-Kalibrierungs-Pass. Das war er nicht. Der Fehler steckte in den Eingangs-
daten, nicht im Ergebnis — er ist **ohne Kenntnis des Ausgangs** nachweisbar.

---

## 1. Der Forecast, der beim Trade live war

Rekonstruiert aus dem Previous-Runs-Archiv (`previous_day1`, echter 24-h-Lead —
also exakt das, was am 13.07. auf dem Tisch lag), gegen das Ist von 33,0 °C:

| Modell | roh | Fehler | 40d-korr. µ |
|---|---|---|---|
| ECMWF | 33,5 | **−0,5** | 33,7 |
| UKMO | 34,0 | −1,0 | 34,5 |
| ICON | 34,4 | −1,4 | 35,4 |
| GFS | 34,7 | −1,7 | 33,4 |
| **JMA** | **37,1** | **−4,1** | 36,6 |
| ENS (5 Modelle) | 34,74 | −1,74 | 34,72 |
| ENS ohne JMA | 34,15 | −1,15 | 34,13 |
| Median | 34,40 | −1,40 | 34,38 |

**Alle fünf Modelle waren zu warm.** Das schärfste (ECMWF) lag mit −0,5 ° fast
richtig; das Ensemble-Mittel lag mit −1,74 ° deutlich schlechter als sein bestes
Mitglied — weil JMA es um 0,6 ° nach oben zog.

## 2. Zwei Artefakte haben die Wahrscheinlichkeit erzeugt

**(a) Der JMA-Ausreißer.** JMA sagte 37,1 °C, die anderen vier lagen bei
33,5–34,7 °C. Modellspanne **3,6 °**. Das arithmetische Ensemble-Mittel hat
keinerlei Abwehr gegen ein einzelnes ausreißendes Mitglied — der Kandidat wurde
also von genau dem Modell getragen, das am Ende 4,1 ° danebenlag.

**(b) Der Vorzeichenwechsel der 700d-Bias-Korrektur.** Für ZBAA gilt:

| Sicht | ENS-Bias | ENS-Sigma |
|---|---|---|
| 700d (Ganzjahr), Lead 1 | **−0,884** | 1,447 |
| 40d (Sommer), Lead 1 | **+0,020** | 1,430 |

Die Ganzjahres-Kalibrierung hebt µ um 0,88 ° an; die Sommer-Kalibrierung tut das
nicht. Beijings Modelle haben im Juli **nicht** den Kaltbias, den sie übers Jahr
zeigen. Wer bei einem Bucket **unterhalb** des Forecasts die 700d-Sicht nimmt,
schiebt den Bucket künstlich weiter weg und macht den Lay sicherer, als er ist.

## 3. Die ehrliche Rechnung (BE 21,0 %)

| Aggregat | Kalibrierung | µ | P(33er) | EV |
|---|---|---|---|---|
| ENS (5) | 700d | 35,62 | 5,6 % | +15,4 pp |
| ENS (5) | 40d | 34,72 | 13,7 % | +7,3 pp |
| Median | 40d | 34,38 | 17,5 % | +3,5 pp |
| **ENS ohne JMA** | **40d** | **34,13** | **20,3 %** | **+0,7 pp** |

Die Sicht, die beide Artefakte entfernt — Ausreißer raus, Sommer-Kalibrierung —
sagt **EV ≈ 0**. Der Trade war nie +EV. Die 4,3 %/5,6 %, die ihn gerechtfertigt
haben, sind das Produkt eines kaputten Modelllaufs und einer saisonal falschen
Bias-Korrektur.

## 4. Das Kriterium existierte bereits — es wurde nur nicht angewandt

Die Trade-Reg begründet den Kauf mit:

> „Trockener, stabiler Hitzetag, **alle 5 Modelle einig** (roh ZBAA-Grid 33,9–37,5)"

3,6 ° Spanne als „einig". Im selben Dokument, in der Ablehnungsliste:

> „**Wuhan 39°:** Modellspanne 34–39° (**JMA-Ausreißer**)"
> „**Milan 33°:** GFS 38,9 = **8° Spanne**"

Wuhan wurde wegen eines JMA-Ausreißers verworfen, Beijing wegen desselben Musters
gekauft. Der Unterschied war nicht die Datenlage, sondern dass bei Beijing das
Ensemble-Mittel den Ausreißer unsichtbar machte und die abgeleitete P-Zahl
beruhigend aussah. **Ein Aggregat, das den Ausreißer versteckt, hat den
Ausreißer-Filter ausgehebelt.**

Der Kernbefund derselben Trade-Reg — „Lay-Value entsteht nur, wo die Modelle
einig sind" — war richtig. Er wurde auf den eigenen Kandidaten nicht angewandt.

## 5. Die Rolle des Autopiloten (nüchtern)

Der Take-Profit (`autopilot.py --profit 0.10`, **kein** Stop-Loss) hat den
Verlust abgefangen. Das ist Glück, kein Schutz, und der Mechanismus ist kein
Risikomanagement:

- **Ist das Modell richtig, kostet der TP Edge.** Er verkauft zu 0,88, was das
  Modell mit ~0,95 bewertet — nach eigener Rechnung ein bewusst schlechter
  Trade. Er deckelt die realisierte Edge bei +10 %, wo das Modell +26,6 %
  behauptet.
- **Ist das Modell falsch, rettet der TP.** Genau das ist hier passiert.

Der TP kann strukturell **nur bei billigen, hochrentierlichen Lays** feuern (ein
Lay zu 0,99 kann nie +10 % gewinnen). Er ist also ausgerechnet in der neuen,
am wenigsten validierten Zerklüftungs-Klasse aktiv — und hat dort still gegen
einen Modelldefekt versichert, von dem wir nichts wussten. Das ist kein Argument
für den TP, sondern eines dafür, das Modell zu reparieren und den TP danach
ehrlich gegen Hold-to-Settlement zu messen (offener Punkt, N ≈ 14 Lays).

## 6. Konsequenzen (umgesetzt in `weather_outlier_screen.py`)

1. **Ausreißer-robustes Ensemble.** Modelle, die > 2,0 ° vom Median der übrigen
   abweichen, fließen nicht mehr ins Handels-µ ein. Gerechnet und ausgewiesen
   werden beide Sichten; für die Kandidatenprüfung zählt die **pessimistischere**.
2. **Harter Spannen-Veto.** Modellspanne (roh, max−min) > 3,0 ° → kein Kandidat.
   Das allein hätte diesen Trade (3,6 °) verhindert.
3. **Doppel-Kalibrierung im Screen statt im Kopf.** Der Screen lädt jetzt
   zusätzlich die 40d-Sommer-Kalibrierung
   (`preregs/weather_source_calib40d_*.csv`) und verlangt, dass **beide** Sichten
   bestehen. Bisher war das eine Regel, an die man sich erinnern musste.
4. **EV-Marge statt „P < BE".** Ein Pass mit +0,7 pp ist kein Trade. Verlangt
   werden ≥ 5 pp EV auf der pessimistischsten Sicht. (Auch die dokumentierte
   Filter-Überstimmung vom 13.07. — „GFS 17 % < BE 21 %, also +EV" — wäre daran
   mit +4 pp gescheitert.)
5. **Lead-Warnung.** Bei Zieltag > 24 h weist der Screen explizit darauf hin,
   dass mit `--lead 2` nachzurechnen ist (Madrid-Lehre vom 13.07.).

## 7. Direkte Bestätigung (out of sample)

Der Screen für den **16.07.** lieferte Beijing 32 °C @ NO 0,850 (17,6 % Rendite)
als einzigen formalen Kandidaten — mit **derselben Signatur**: JMA roh 38,4 °
gegen 33,2–34,5 ° der anderen vier (Spanne 5,2 °), ECMWF wieder das kälteste und
zugleich schärfste Modell, und nach Entfernen des Ausreißers unter der
Sommer-Kalibrierung **P 14,2 % gegen BE 15,0 % → EV +0,8 pp**. Praktisch
dieselbe Zahl wie beim Verlierer.

Die oben beschriebenen Regeln lehnen diesen Kandidaten ab — sie wurden formuliert,
**bevor** das Settlement des 14.07. bekannt war, und fangen den bekannten
Verlierer rückwirkend. Der Trade wurde nicht gesetzt.

## 8. Nachtrag: der Spannen-Veto ist empirisch gedeckt (nicht nur plausibel)

Der Veto (Modellspanne > 3 ° → kein Kandidat) war zunächst nur eine Intuition und
kostet Kandidaten — am 16.07. fielen 10 von 15 Städten darunter. Faire Gegenfrage:
filtert er echtes Risiko weg oder nur Rendite? Gemessen mit
`weather_spread_conditional.py` (700 d, 10 Städte, n = 5.587 Stadt-Tage, echter
24-h-Lead, bias-bereinigt je Stadt):

| Modellspanne | n | MAE | Sigma | P(\|Fehler\| > 1,5 °) | P(\|Fehler\| > 2,5 °) |
|---|---|---|---|---|---|
| 0–1,5 ° | 1164 | 0,76 ° | 1,06 ° | 11,9 % | **2,5 %** |
| 1,5–3,0 ° | 2335 | 0,98 ° | 1,32 ° | 20,6 % | **5,1 %** |
| 3,0–5,0 ° | 1347 | 1,25 ° | 1,71 ° | 32,6 % | **11,4 %** |
| > 5,0 ° | 741 | 1,43 ° | 1,77 ° | 39,1 % | **16,9 %** |

Monoton in beide Richtungen. An der Veto-Schwelle 3,0 °: MAE 0,90 → 1,32 °
(**1,46×**), Tail-Risiko P(\|err\| > 2,5 °) 4,2 % → 13,3 % (**3,15×**). Betroffen
sind 37 % aller Tage.

**Das ist der Beijing-Mechanismus in einer Zahl.** Die Kalibrierung benutzt ein
FESTES Sigma je Stadt (ZBAA-ENS: 1,447). Real ist Sigma an ruhigen Tagen ~1,06
und an zerklüfteten Tagen ~1,77. Die Normal-Annahme unterschätzt das Risiko also
systematisch genau dort, wo die hohen Renditen liegen — denn ein Bucket ist ja
gerade deshalb teuer gepreist, weil die Modelle streiten. Der Beijing-33-Lay lief
an einem 3,6-°-Tag: dort ist P(Fehler > 2,5 °) empirisch ~13 %, nicht die ~4 %,
die das Durchschnitts-Sigma impliziert.

**Konsequenz für die „48h-Zerklüftungs"-These:** Bei 24 h Vorlauf liegen 10 von
15 Städten unter der Schwelle, bei 48 h nur noch 5 von 15. Die 48h-Leitern sind
nicht zerklüftet, weil der Markt schläft, sondern weil die Modelle sich uneinig
sind — und diese Uneinigkeit ist mit 3× Tail-Risiko korrekt bepreist. Die These
„zweistellige Renditen 2 Tage voraus = Edge" ist damit falsifiziert.

**Offener Ausbaupfad (nicht gebaut):** Statt eines harten Vetos ein
**spannen-konditioniertes Sigma** (Sigma als Funktion der Tagesspanne statt
Konstante). Das würde die 37 % vetoierten Tage wieder handelbar machen — aber mit
ehrlichen, breiteren Wahrscheinlichkeiten statt mit einer zu engen Glocke. Erst
dann wäre entscheidbar, ob dort nach korrekter Bepreisung noch Edge übrig ist.

---

**Datenquellen:** Ist = Wunderground (`api.weather.com`, Polymarket-Settlement-
Quelle) + IEM-METAR (`report_type` 3+4), lokaler Kalendertag Asia/Shanghai.
Forecast = Open-Meteo Previous-Runs (`temperature_2m_previous_day1`), also der
archivierte Originallauf mit echtem 24-h-Lead — vollständig reproduzierbar.
