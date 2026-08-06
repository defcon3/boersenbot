# Pre-Reg: Stadt-Zellen mit positivem Ertrag — Paper-Forward-Test

**Registriert:** 06.08.2026, vor dem ersten Zieltag des Fensters
**Fenster:** Zieltage **07.08.2026 – 03.09.2026** (28 Zieltage)
**Auswertung:** ab **04.09.2026**, zusammen mit den drei laufenden Fenstern
**Modus:** **PAPER.** Kein Echtgeld, kein Eingriff in den Autobuy. Jupiters
Minimaleinsatz von 5 $ steht in keinem Verhältnis zu n = 11–19.

---

## Woher die Frage kommt

Wörtlich vom Betreiber, 06.08.2026:

> „Ich bin damals angetreten, um eine konsistente Prognosequelle zu finden.
> Wenn dort die Vorhersage für Tokio immer einen Tag darunter liegt, zu 100 %,
> habe ich den Gelddrucker gefunden. Ziel war es, die Quelle zu finden, nicht
> das, was der Markt sagt. Ist mir egal, ob die Quote 95 oder 75 ist, solange
> sie sicher gewinnt."

Der Faden war nie falsifiziert — er war **nie zu Ende gerechnet**. Am 03.08.
wurde gemessen, wie oft ein Bucket **trifft** (`weather_bucket_abstand_eval.py`:
Tel Aviv trifft die −1-Klasse in 14/18, p < 10⁻⁶). Nie gemessen wurde, was er
dabei **kostet**. Genau darin liegt der Unterschied zwischen einer
Modellschwäche, die man korrigiert, und einem Edge, den man handelt.

Am 06.08. nachgeholt (`weather_stadt_ev.py`, `weather_stadt_ev_holm.py`).

## Was die Vormessung ergeben hat — vollständig, auch das Unbequeme

In-Sample-Fenster: Zieltage **12.07.–04.08.2026**, 24 Zieltage, Lead-1-Snapshot,
3.375 Buckets, 30 Städte. Settlement `wu_settle_k` vor `settle_k`.

**Aggregiert ist der Markt fair.** Anker Markt-Favorit, positionsweiser ROI:

| Offset | n | trifft | Preis | ROI | t |
|---|---|---|---|---|---|
| ±0 | 375 | 45,3 % | 0,432 | **−0,1 %** | −0,02 |
| −1 | 374 | 20,3 % | 0,239 | −22,3 % | −2,60 |
| +1 | 362 | 21,8 % | 0,230 | −8,2 % | −0,83 |
| −2 | 366 | 3,8 % | 0,072 | −52,9 % | −3,60 |

**Und die Zellensuche findet weniger als der Zufall.** 273 Zellen
(Stadt × Offset, beide Anker, n ≥ 8), exakter Monte-Carlo-Test unter H0 „Markt
ist fair" (200.000 Simulationen je Zelle, jede Position mit ihrem **eigenen**
Preis gewürfelt — Poisson-Binomial):

    unkorrigiert p < 0,05 :  10 Zellen      Zufallserwartung: 13,7
    Bonferroni            :   0
    Holm-Bonferroni       :   0

Kleinster p-Wert 0,0112 gegen eine Holm-Rang-1-Schwelle von 1,83·10⁻⁴ —
**Faktor 61 zu groß**. Holm ist gleichmäßig mächtiger als Bonferroni, teilt mit
ihm aber die erste Hürde α/m; reißt Rang 1, stoppt das Verfahren.

**Daraus folgt: die Suche selbst hat nichts gefunden.** Was hier registriert
wird, ist deshalb ausdrücklich keine Behauptung „es gibt einen Edge", sondern
die einzige Konstruktion, unter der die drei stärksten Zellen überhaupt
entscheidbar werden: sie **vorher** zu benennen. Damit fällt die Korrektur von
α/273 auf Holm über **m = 3**.

## Hypothese

**H1:** In den drei unten benannten Zellen trifft der Bucket häufiger, als sein
Lead-1-Preis verlangt. Ein YES-Kauf zum Snapshot-Preis liefert dort einen
positionsweise gerechneten ROI > 0 nach Gebühr.

**H0:** Der Markt ist auch dort fair — Bucket i trifft mit genau seiner
Wahrscheinlichkeit p_i.

### Die drei registrierten Zellen — abschließend, keine Nachträge

| # | Zelle | Anker | IS-n | trifft | Mittelpreis | IS-ROI | IS-p (MC) |
|---|---|---|---|---|---|---|---|
| Z1 | **Beijing +1** | Markt-Favorit | 11 | 54,5 % | 0,231 | +145,6 % | 0,0112 |
| Z2 | **Taipei ±0** | Markt-Favorit | 13 | 69,2 % | 0,427 | +62,1 % | 0,0176 |
| Z3 | **Tel Aviv ±0** | Markt-Favorit | 19 | 78,9 % | 0,611 | +24,4 % | ~0,06 |

**Anker Markt-Favorit** heißt: Offset 0 ist der Bucket mit dem höchsten
`buy_yes` im Lead-1-Snapshot (`market_fav_k`), +1 der Bucket ein Grad darüber.

**Warum diese drei:** Z1 und Z2 haben die beiden kleinsten p-Werte der ganzen
Suche. Z3 hat das größte n und steht für die Ausgangsfrage des Betreibers —
Tel Aviv ist die Stadt mit der belegten Achsenverschiebung
(`weather-stadt-verschiebung-telaviv`). Z3 ist damit bewusst die schwächste der
drei; sie wird mitgeführt, weil ihr Scheitern genauso aussagekräftig ist.

**Warum nicht mehr:** Jede weitere Zelle verschärft Holm für alle. Bei m = 3
lauten die Schwellen 0,0167 / 0,025 / 0,05; bei m = 7 schon 0,0071 / 0,0083 /
0,010 — Z1 würde dann an der eigenen IS-Leistung scheitern.

## Gates

Alle Tests **einseitig** (nur „besser als fair" ist interessant), Testfamilie
**m = 3**, Holm-Bonferroni, α = 0,05.

- **G1 — Signifikanz.** Mindestens **eine** der drei Zellen überlebt Holm über
  m = 3 im Forward-Fenster. p-Wert aus demselben Monte-Carlo-Test wie oben
  (200.000 Sims, Seed dokumentiert), **nicht** aus einem t-Test: die
  ROI-Verteilung ist stark schief (viele −1, wenige große Gewinne), die
  Normalapproximation trägt bei n ≈ 15 nicht.
- **G2 — Datenmenge.** Je Zelle mindestens **10** auswertbare Zieltage im
  Fenster. Wird das verfehlt, gilt die Zelle als **nicht auswertbar**, nicht
  als gerissen — und die Schwelle wird nicht gesenkt.
- **G3 — Kosten.** Der ROI bleibt positiv, wenn auf jeden Kaufpreis **1 Cent
  Slippage** aufgeschlagen wird (Tick-Größe). Die Gebühr 0,07·min(p, 1−p) ist
  in der ROI-Formel bereits enthalten.
- **G4 — kein Einzeltag.** Nach Entfernen des **besten einzelnen Zieltags**
  (Jackknife) bleibt der ROI der Zelle positiv. Bei Preisen um 0,23 bringt ein
  einziger Treffer über +3 Einheiten — ohne dieses Gate wäre jede Zelle mit
  einem Glückstag ein „Fund".
- **G5 — Kontrolle gegen globalen Drift.** In der **unregistrierten Restmenge**
  (alle übrigen Zellen, n ≥ 8) darf der Anteil nominal signifikanter Zellen die
  Zufallserwartung von 5 % **nicht** übersteigen. Liegt er darüber, ist im
  Fenster etwas Marktweites passiert (Regimewechsel, Bepreisungsfehler über
  alle Bretter) und ein Treffer in Z1–Z3 wäre nicht als Zell-Effekt
  interpretierbar. Zusätzlich muss der Aggregat-ROI am Markt-Favoriten (Offset
  ±0, alle Städte) betragsmäßig unter 5 % bleiben.

**PASS** = G1 **und** G3 **und** G4 **und** G5, bei erfüllter G2.

## Pflichtübung: jedes Gate gegen die offengelegten Zahlen gegengerechnet

Vorschrift aus der Review vom 02.08. („jedes Gate vor dem Festschreiben gegen
die bereits offengelegten Einzelzahlen gegenrechnen — und die Gegenrechnung in
die Pre-Reg schreiben"). Durchgeführt am 06.08. **vor** dem Festschreiben:

| Gate | Gegenrechnung auf den IS-Daten | Urteil |
|---|---|---|
| G1 | Holm über m = 3: Z1 p 0,0112 < 0,0167 ✓, Z2 p 0,0176 < 0,025 ✓, Z3 p ~0,06 > 0,05 ✗ → **2 von 3 hätten bestanden.** Das Gate ist also weder unerreichbar noch geschenkt. | tragfähig |
| G2 | Beobachtungsrate je Zieltag: Z1 0,46 · Z2 0,54 · Z3 0,79 → erwartet **13 / 15 / 22** in 28 Zieltagen. Die Schwelle 10 ist für alle drei erreichbar, für Z1 mit dem geringsten Puffer. | tragfähig |
| G3 | Mit 1 ct Aufschlag: Z1 +145,6 → **+134,0 %**, Z2 +62,1 → **+58,1 %**, Z3 +24,4 → **+22,5 %**. Alle bleiben deutlich positiv; das Gate greift erst bei dünnen Effekten — genau dort soll es greifen. | tragfähig |
| G4 | Ohne besten Zieltag: Z1 +145,6 → **+108,3 %**, Z2 +62,1 → **+53,0 %**, Z3 +24,4 → **+20,4 %**. Keine Zelle hängt an einem Tag. | tragfähig |
| G5 | IS-Fenster: 10 nominal signifikante von 273 = **3,7 %**, unter der 5-%-Erwartung. Aggregat am Markt-Favoriten **−0,1 %**, weit unter 5 %. Die Kontrolle wäre im IS-Fenster erfüllt gewesen. | tragfähig |

**Kein Gate wurde nach Sicht der Zahlen geändert.** G3 und G4 wurden bewusst so
gewählt, dass sie in-sample bestehen — sie sollen Zufallsfunde im Forward
abfangen, nicht die registrierten Zellen vorab erledigen.

## Erwartung des Registrierenden

**Eher Fehlschlag.** Drei Gründe, alle vor der Auswertung notiert:

1. Die Suche insgesamt fand **10 signifikante Zellen bei 13,7 erwarteten** —
   das ist das Profil von reinem Rauschen, nicht von einem versteckten Muster.
2. Das Aggregat ist mit −0,1 % bei t = −0,02 so nah an fair, wie eine Messung
   nur sein kann. Ein echter Zell-Effekt müsste sich dort andeuten.
3. Beijing hat innerhalb einer Stunde je nach Anker und Rechenweise **+133 %,
   −57 % und +146 %** gezeigt. Solche Sprünge sind ein Fingerabdruck von zu
   kleinem n, nicht von Struktur.

Wenn Z1–Z3 das Fenster trotzdem überstehen, ist das **deshalb** etwas wert:
sie wurden vorher benannt, gegen einen exakten Test, mit Kontrollgruppe.

## Datenerhebung — es wird nichts Neues gebaut

Der Ladder-Logger (`boersenbot_weather_ladder.timer`, täglich 14:30 CEST)
schreibt bereits alle nötigen Felder nach `bb_WeatherLadders`: `buy_yes`,
`market_fav_k`, `offset_fav`, `settle_k`, `wu_settle_k`, `snapshot_utc`.
**Kein neuer Dienst, kein neuer Cron, kein Eingriff.** Das Fenster entsteht von
selbst; die Auswertung liest dieselbe Tabelle mit Datumsfilter.

Auswertung ab 04.09.2026:

```bash
python weather_stadt_ev_holm.py --von 2026-08-07 --bis 2026-09-03
python weather_stadt_ev.py      --von 2026-08-07 --bis 2026-09-03
```

## Fallen, die beim Auswerten gelten

1. **Positionsweise rechnen, nie aus dem Mittelpreis.** Beijing kam über
   EV = q/p̄ auf +133 %, positionsweise auf −57 %. Lehre vom 02.08.:
   „Break-even gehört positionsweise aus dem echten Preis."
2. **Lead 1, nicht Lead 0.** Der Snapshot vom Vortag. Lead 0 liegt für Asien
   nach dem Tagesmaximum und rechnet den Fehler klein.
3. **`wu_settle_k` vor `settle_k`.** Und für gehandelte Märkte ist Jupiters
   `result` die letzte Instanz — METAR gab am 05.08. für Shenzhen 31,0 Grad
   aus, während der Markt gegen 32 abrechnete.
4. **Favorit nur über `market_fav_k` bzw. `offset_fav` aus der DB**, nie selbst
   runden (Hong Kong ist `BUCKET_FLOOR`).
5. **Das Fenster nicht verlängern, wenn es knapp aussieht.** Grenze ist der
   03.09., festgeschrieben am 06.08.

## Was diese Pre-Reg NICHT behauptet

- Nicht, dass der −1-Lay-Autobuy falsch liegt. Andere Frage, andere Seite des
  Buchs.
- Nicht, dass eine Fremdquelle (CNN, metar.ws) helfen würde — die Schranke
  dafür ist am 04.08. gemessen (~0,3 K, `weather_icon_source_bound.py`).
- Nicht, dass Tel Avivs Achsenverschiebung unecht wäre. Sie ist mit p < 10⁻⁶
  belegt. Offen ist allein, ob der **Markt sie zu niedrig bepreist** — und
  genau danach fragt Z3.
