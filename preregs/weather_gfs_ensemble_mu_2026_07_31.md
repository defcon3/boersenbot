# Pre-Registrierung: 31-Member-GFS-Ensemble als robusteres µ — 2026-07-31

**Status:** Vorregistrierung. Geschrieben, **bevor** irgendeine Fehler-, Treffer-
oder Ertragskennzahl gerechnet wurde. Die Sondierung vorab hat ausschließlich
**Mengen** angesehen: welche Endpunkte es gibt, wie weit die Historie reicht,
wie viele Member geliefert werden. Keine Prognosegüte, kein Vergleich.

Auswertung folgt in `weather_gfs_ensemble_mu_eval.py`.

---

## Anlass — und warum die naheliegende Fassung *nicht* getestet wird

Auslöser war ein fremdes Repo (`suislanchez/polymarket-kalshi-weather-bot`, 566
Sterne), das Bucket-Wahrscheinlichkeiten aus dem 31-Member-GFS-Ensemble von
Open-Meteo bildet. Das Repo selbst ist **keine Quelle**: es zählt rohe Member
(`count / len(members)`), korrigiert keinen Stations-Bias, kalibriert nichts,
zieht nirgends Gebühren ab und enthält in 65 Dateien keinen einzigen Backtest.
Genau der fehlende Stations-Bias hat bei uns für NYC bereits das **Vorzeichen
des Edges gedreht**. Übernommen wird also nur eine **Datenquelle**, keine Methode.

Die naheliegende These wäre: „das Ensemble liefert ein besseres σ als unsere
Spannen-Formel σ(s) = a + b·s". **Diese wird hier nicht getestet.** Die Pre-Reg
vom 14.07. (`weather_spread_sigma_2026_07_14.md`) hat die Fehlerquellen des
Beijing-Verlusts bereits nach Gewicht sortiert:

| Rang | Quelle | Wirkung |
|---|---|---|
| 1 | Bias-Vorzeichenwechsel 700d → 40d | 0,9 K in µ |
| 2 | JMA-Ausreißer im Modellmittel | 0,6 K in µ |
| 3 | Sigma | 1,32 → 1,45, **ändert am Trade nichts** |

Verdikt dort wörtlich: *„Kein Sigma der Welt repariert einen verzogenen
Mittelwert."* Ein σ-Test wäre also die Wiederholung einer beantworteten Frage.

## Die Fassung, die den Nachweis lohnt

Fehlerquelle 2 ist ein **Ausreißer eines einzelnen Modells** im Mittel aus fünf
(GFS, ICON, UKMO, JMA, ECMWF). Ein 31-Member-Ensemble hat dieses Problem
strukturell nicht: alle Member sind derselbe Modellkern mit gestörten
Anfangsbedingungen, es gibt kein „JMA, das um 4 K danebenliegt". Die Frage ist
damit nicht die **Breite** der Verteilung, sondern die **Robustheit ihrer Mitte**.

Das zielt direkt auf den teuersten Posten im laufenden Betrieb: Der harte
Spannen-Veto sperrt **37 % aller Tage**. Er bleibt seit dem 14.07. bewusst
bestehen, weil eine große Modellspanne nicht Breite anzeigt, sondern ein
**korrumpiertes µ**. Wenn das Ensemble-µ an genau diesen Tagen trägt, wird ein
Drittel des Kalenders wieder handelbar — mit ehrlichen Zahlen statt mit einer
Sperre.

---

## Hypothese

**H1 (primär):** An Tagen mit großer Modellspanne (s > 3 K, also den heute
gesperrten) ist das biaskorrigierte **Median der 31 GFS-Member** ein genauerer
Schätzer des tatsächlichen Tagesmaximums als das biaskorrigierte Mittel der
fünf Punktmodelle.

**H2 (sekundär, Bonferroni t > 2,5):** Die Bucket-Wahrscheinlichkeiten aus dem
Ensemble sind in der **Lay-Zone** besser kalibriert als die aus µ_5 + σ(s).

**H0:** Das Ensemble-µ ist nicht besser. Der Spannen-Veto bleibt, seine
Begründung wird von „gemessen an fünf Modellen" auf „auch gegen ein echtes
Ensemble gemessen" gehoben.

### Verfahren (beide Seiten identisch behandelt)

Fairness-Bedingung: **Jede** Vorverarbeitung, die µ_5 bekommt, bekommt µ_ens auch.
- **Bias:** 40-Tage-Sommer-Bias je Stadt, gleitend, ausschließlich aus Tagen
  **vor** dem Zieltag (`[[weather-lay-bucket-preference]]`-Doktrin, kein Lookahead).
- **Aggregation:** µ_5 = Mittel der fünf Modelle wie im Live-Screen.
  µ_ens = **Median** der Member (nicht Mittel — der Median ist der Punkt, an dem
  sich der Ausreißerschutz überhaupt zeigen kann; das Mittel wird als
  Sensitivität mitberichtet, ist aber nicht das vorregistrierte Signal).
- **Zielgröße:** Tagesmaximum in lokaler Zeit, aus Stundenwerten aggregiert —
  identisch zur bestehenden Pipeline (`weather_source_compare.py`).
- **Ist-Werte:** dieselbe settelnde Quelle wie im Livebetrieb (WU-**Tabelle**,
  nicht die Kachel — s. `[[weather-settlement-wu-vs-metar]]`).

### Datenquelle und der Lookahead-Fallstrick

`https://ensemble-api.open-meteo.com/v1/ensemble`, Modell `gfs025`, Feld
`temperature_2m_previous_day1`.

**Zwingend `previous_dayN`, niemals `start_date` allein.** `previous_day1`
liefert die Prognose, die vor 24 h für diesen Zeitpunkt gemacht wurde. Ein
schlichter `start_date`-Abruf in die Vergangenheit kann den *aktuellen*, rückwärts
gerechneten Lauf liefern — das wäre Lookahead und würde das Ensemble künstlich
gewinnen lassen. Die bestehende Pipeline nutzt aus demselben Grund bereits
`previous_dayN`.

## Daten — und die Grenze, die diese Studie einschneidet

Sondiert: der Ensemble-Endpoint erlaubt `start_date` **erst ab 2026-04-29**.
Das sind rund **93 Tage, ausschließlich Sommer**. Die Vorgänger-Pre-Reg forderte
≥ 300 Tage *je Stadt* — hier unerreichbar.

Konsequenzen, vorab festgehalten:
1. Ausgewertet wird **gepoolt** über die Städte der bestehenden `STATIONS`-Liste,
   nicht je Stadt. Größenordnung ~30 Städte × 93 Tage ≈ 2.800 Stadt-Tage.
2. **IS = Mai–Juni 2026, OOS = Juli 2026.** Zeit-Split, kein Städte-Split.
3. **Ein Bestehen ist ein Sommer-Befund und sonst nichts.** Der Bias dreht
   zwischen den Jahreszeiten das Vorzeichen (bekannt aus der 40d/700d-Arbeit);
   auf Herbst oder Winter darf das Ergebnis **nicht** übertragen werden. Ein
   Re-Test im Oktober ist Pflicht, bevor irgendetwas ganzjährig scharf geschaltet
   wird — dieser Punkt wandert unabhängig vom Ausgang in den Backlog.
4. Mit 93 Tagen ist die Studie **unterbesetzt für seltene Ereignisse**. Deshalb
   ist H1 auf den *Fehler* formuliert (viele Beobachtungen) und nicht auf
   Trefferquoten in einzelnen Buckets (wenige).

---

## Gates

| Gate | Bedingung |
|---|---|
| **G1** In-Sample (Mai–Jun) | Auf zerklüfteten Tagen (s > 3 K): MAE(µ_ens) < MAE(µ_5), gepaart je Stadt-Tag, **t > 2,0** |
| **G2** Out-of-Sample (Juli) | Identisches Verfahren, gleiches Vorzeichen, **t > 1,5** |
| **G3** Kalibrierung **bin-weise** (OOS) | In den Bins 2–5 %, 5–10 %, 10–20 % je einzeln: realisierte Rate im Band [0,75×; 1,25×] der vorhergesagten. **Es wird ausdrücklich NICHT über die Lay-Zone gemittelt.** |
| **G4** Praxis (OOS) | Auf den vom Spannen-Veto gesperrten Tagen: die mit µ_ens neu geöffneten Lays haben in Summe **positiven** Ertrag nach Gebühr (5 % von min(p, 1−p)) |
| **G5** Robustheit | Median des Stadt-Effekts > 0 **und** nach Streichen der besten Stadt bleibt G2 bestehen **und** kein einzelner Tag trägt > 30 % des Gesamteffekts |

**Warum G3 so und nicht wie am 14.07.:** Das damalige G3 mittelte über die
gesamte Lay-Zone und wurde von den vielen Buckets mit P ≈ 0 dominiert — es
bestand sogar für das Modell, das den Beijing-Verlust produziert hatte. Das war
ein **Design-Fehler**, dort auch als solcher protokolliert. Bin-weise
Auswertung, getrennt nach Regime, war die Konsequenz; sie wird hier zur
Vorbedingung gemacht statt nachträglich exploriert.

**Bonferroni:** G1/G2 sind je *ein* Test auf dem vorregistrierten Signal
(Median, s > 3 K). Die Sensitivitätsläufe — Mittel statt Median, Schwellen
s ∈ {2, 4} K, Lead 2 — werden berichtet, aber **kein** Gate darf über sie
erfüllt werden.

## Härtetest (deskriptiv, KEIN Gate)

Auf die bekannten Verlierer angewandt: **Beijing 32 °/33 °** (JMA roh 38,4 gegen
33,2–34,5 der übrigen vier) und die fünf am 16.07. vom Veto blockierten Buckets.
Ordnet µ_ens sie richtig ein? Eine einzelne Rettung beweist nichts — aber wenn
µ_ens ausgerechnet dort *daneben* liegt, ist H1 praktisch wertlos, egal was die
Gates sagen. Genau dieser Test hat am 14.07. die σ(s)-These gekippt, nachdem
alle Gates bestanden waren.

## Vorab-Erwartung (damit sie nicht zurechtgebogen wird)

**G1 und G2 bestehen vermutlich** — der Ausreißerschutz des Medians über 31
Member ist ein realer, mechanischer Effekt, und die Spanne der fünf Modelle ist
nachweislich oft von einem einzelnen Ausreißer getrieben.

**G4 halte ich für offen bis unwahrscheinlich.** Grund: Der Spannen-Veto sperrt
Tage, an denen die Atmosphäre selbst unentschieden ist. Ein robusteres µ macht
die Prognose *stabiler*, aber nicht notwendig *richtiger* — die
Anfangsbedingungen eines einzigen Modellkerns können gemeinsam falsch liegen,
und dann ist ein enges Ensemble schlimmer als fünf uneinige Modelle, weil es
Sicherheit vortäuscht. Das ist die eigentliche Gefahr dieser These.

Zusätzlich steht dagegen: Der Markt trifft unsere Punktprognose ohnehin besser
als wir selbst (28.07., p < 0,01). H1 verbessert einen Schätzer, dessen
*Punktgebrauch* bereits falsifiziert ist — der Nutzen kann nur über die
Bucket-Wahrscheinlichkeiten in der Lay-Zone kommen, nie über den Favoriten.

---

# NACHTRAG 31.07.2026 (noch am Tag der Registrierung): rückwirkend NICHT ausführbar

Beim Implementieren von `weather_gfs_ensemble_mu_eval.py` stellte sich heraus:
**Für Ensemble-Läufe existiert kein Archiv.** Damit ist H1 rückwirkend nicht
prüfbar — die Pre-Reg bleibt gültig, ihr Datenpfad nicht.

| Weg | Ergebnis |
|---|---|
| `ensemble-api` + `previous_day1` (+ `start_date` oder `past_days`) | 30 korrekt benannte Member-Spalten, **alle Werte `None`** |
| `historical-forecast-api/v1/ensemble` | **Not Found** |
| `previous-runs-api/v1/ensemble` | **Not Found** |
| `ensemble-api` + `temperature_2m` + `start_date` | nur die letzten ~3 Tage |
| `historical-forecast-api/v1/forecast` (Punktmodelle) | funktioniert — das ist die Quelle unserer fünf Modelle |

**Warum die Sondierung das nicht gefangen hat:** Die API akzeptiert
`previous_day1` auf dem Ensemble-Endpoint anstandslos, antwortet mit HTTP 200
und liefert die vollständige Spaltenstruktur — nur eben ohne Werte. Geprüft
worden war die *Erreichbarkeit* der Felder, nicht ihre *Befüllung*. Das ist die
Lehre für die nächste Machbarkeitsprüfung: **Feldnamen zählen ist keine
Datenprüfung. Immer einen konkreten Wert anfassen.**

## Konsequenz: Forward-Test statt Backtest

Die Gates G1–G5 bleiben **unverändert gültig** — sie werden nur später erfüllt.
Nötig ist ein täglicher Logger, der je Stadt die Lead-1-Ensemble-Prognose
festhält (~31 Aufrufe/Tag, weit innerhalb der freien Stufe). Nach ~60–90 Tagen
ist die Auswertung rechenbar; `weather_gfs_ensemble_mu_eval.py` ist fertig und
braucht dann nur den anderen Datenpfad.

Zwei Punkte, die dabei anders liegen als geplant:
1. **Der Forward-Test läuft in den Herbst.** Das ist kein Sommer-Befund mehr —
   inhaltlich sogar besser (breitere Regimeabdeckung), verlangt aber, dass der
   Bias mitwandert und die Sommer-Steigung `SIGMA_B = 0,107` **nicht** blind
   weiterbenutzt wird.
2. **Der Test wird dadurch die sauberste Vorregistrierung des Repos:** Gates und
   Auswertungscode stehen fest und committet, **bevor** die erste Beobachtung
   überhaupt existiert. Kein Backtest kann das bieten.

## Abbruchregel

Reißt **G1**, ist die These falsifiziert und wird **nicht** umparametrisiert —
insbesondere wird dann nicht auf „Mittel statt Median" oder eine andere
Spannenschwelle ausgewichen. Bestehen G1–G3, scheitert aber **G4**, gilt
dasselbe wie am 14.07.: der Veto bleibt, und der Befund lautet „besseres µ,
trotzdem nicht handelbar". Das wäre ein sauberes Ergebnis und wird als solches
committet.
