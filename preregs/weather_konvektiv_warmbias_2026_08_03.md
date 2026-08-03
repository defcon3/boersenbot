# Pre-Reg: Preist der Markt an konvektiv gedeckelten Tropentagen zu warm?

**Angelegt:** 2026-08-03, **vor** dem ersten Blick auf die Zielgröße.
**Anlass:** Panama City, 03.08.2026. Der Markt-Favorit lag bei 33 °C+, real wurden
es 31,0 (MPMG/Albrook). Das Modellmittel lag mit 31,6 fast richtig — über 33 kam
nur aus UKMO (34,0). Die Stunden 09–13 lokal klebten unter einem `SCT@1800`-Deck
auf 29, der Taupunkt lag ganztags bei 25–27 °C. Verdacht: an solchen Tagen preist
der Markt die Deckelung nicht ein.

## Hypothese

An Tagen, die schon **am Vortag** als konvektiv-gedeckelt erkennbar sind, liegt
der **Markt-Favorit systematisch über** dem realisierten Bucket — und zwar in den
Tropen stärker als anderswo.

Zielgröße je Stadttag: **d = market_fav_k − settle_k** (in Buckets, 1 Bucket = 1 K).
d > 0 heißt: der Markt war zu warm.

## Datenbasis (vor der Pre-Reg geprüft, ohne Blick auf d)

- `bb_WeatherLadders`, `var='max'`, 10.07.–02.08.2026.
- Nur **Lead-1-Snapshots** (Logger-Lauf am Vortag, 12:30 UTC). Lead 0 ist
  unbrauchbar: für Asien liegt er **nach** dem Tagesmaximum, der Markt kennt das
  Ergebnis dann schon. → 355 Stadttage, davon 351 mit Bedingungsdaten.
- Bedingung aus der **Vortagesprognose** (Open-Meteo previous-runs, `previous_day1`),
  also handelbar: `regen_mm` = Σ Niederschlag 06–18 h lokal,
  `wolken_tag` = ⌀ Bewölkung 09–18 h lokal.
- Tropen objektiv über die Breite: **|lat| ≤ 23,5°** → Hong Kong, Jeddah,
  Kuala Lumpur, Mexico City, Panama City, São Paulo, Shenzhen (78 Stadttage).
  Bewusst ohne Handauswahl — Jeddah (Wüste, keine Konvektion) bleibt drin und
  fällt durch die Regenbedingung von selbst heraus.

## Primärtest (genau EINER, keine Grid-Suche)

**Gruppe K:** Tropen ∧ `regen_mm ≥ 1` ∧ `wolken_tag ≥ 60`  → **n = 39, 6 Städte**
(vorab ausgezählt, ohne d anzusehen).

## Gates

- **G1 Signal:** mean(d) in K > 0 mit **t > 2,0** (einseitig).
- **G2 Spezifität:** mean(d) in K > mean(d) der **trockenen Tropentage**,
  Differenz mit **t > 1,5**. Sonst ist es ein Stadt-Effekt, kein Konvektions-Effekt.
- **G3 Kontrolle außerhalb der Tropen:** derselbe Filter auf |lat| > 23,5 (n = 37).
  Der Tropen-Effekt muss **größer** sein; ist er es nicht, ist es ein globaler
  Marktbias und die Tropen-These ist falsch.
- **G4 Menge:** n ≥ 30 Stadttage und ≥ 4 Städte in K (per Konstruktion erfüllt).
- **G5 Nicht von einer Stadt getrieben:** Leave-one-city-out über alle 6 Städte —
  Vorzeichen hält und **t > 1,0** in jedem Durchgang.

**Kein Gate, nur beschreibend** (damit daraus kein zweiter Primärtest wird):
die sechs weiteren Schwellenschnitte (Regen ≥ 3/5 mm, Wolken ≥ 60/75 % usw.) als
Sensitivität, sowie der Vergleich `round(mu_ens) − settle_k` gegen d — also ob
unsere eigene Modellseite an genau diesen Tagen besser liegt als der Markt.

## Was ein PASS bedeuten würde — und was nicht

PASS hieße: an vortags erkennbaren Konvektionstagen in den Tropen sind die
**oberen** Buckets zu teuer, Lays dort haben Rückenwind. Das wäre eine
Ausnahme von [[weather-markt-schlaegt-eigenen-favoriten]] (Markt-Fav trifft
47,4 % gegen unsere 33,2 %) — **eine konditionale Ausnahme, keine Aufhebung**.
Vor einem Einsatz müsste eine Forward-Probe laufen, nicht nur diese 24 Tage.

FAIL heißt: Panama am 03.08. war ein Einzelfall, und die Erklärung dafür ist
Erzählung, nicht Edge.

---

# ERGEBNIS (03.08.2026): **ROT** — These falsifiziert

Gefahren mit `weather_konvektiv_warmbias_eval.py`, Daten in
`preregs/weather_konvektiv_warmbias_data_2026_08_03.csv` (351 Stadttage).

| Gruppe | n | Städte | mean(d) | t |
|---|---|---|---|---|
| **K Tropen ∧ konvektiv** | 39 | 6 | **+0,128** | **+0,80** |
| Tropen trocken | 39 | 7 | −0,026 | −0,12 |
| Rest ∧ konvektiv | 37 | 14 | −0,324 | −1,64 |
| Rest trocken | 236 | 22 | −0,034 | −0,50 |

| Gate | Ergebnis |
|---|---|
| G1 Signal (t > 2,0) | **FAIL** — t = +0,80 |
| G2 Spezifität (t > 1,5) | **FAIL** — t = +0,56 |
| G3 Tropen > Rest | PASS (+0,128 vs −0,324) |
| G4 Menge | PASS (39 / 6) |
| G5 Leave-one-city-out | **FAIL** — in allen 6 Durchgängen t < 1,0 |

**Das Vorzeichen stimmt, die Größe nicht.** Die Richtung ist durchgehend
konsistent — Tropen-Konvektionstage liegen über den trockenen und deutlich über
den außertropischen Konvektionstagen — aber +0,13 Bucket ist ein Zehntel dessen,
was der Panama-Tag suggerierte, und statistisch nichts.

**Die Stadt, die die These ausgelöst hat, widerlegt sie selbst:** Panama City
kommt in K auf **−0,12 (n = 8)**, hat also an konvektiven Tagen eher zu *kalte*
Marktfavoriten. Der 03.08. (d = +2) liegt noch nicht im Log.

**Panama war ein normales Tail-Ereignis, keine Klasse.** Über alle 351 Stadttage
ist der Markt in **5,1 %** der Fälle ≥ 2 Bucket zu warm; sd(d) = 1,10 Bucket. Ein
d = +2 ist damit Alltag, kein Signal.

**Der Markt bleibt auch hier besser als wir.** In genau der Gruppe, in der wir
seine Schwäche vermutet haben, trifft der Markt-Favorit den Bucket zu **43,3 %**
exakt, unser `round(mu_ens)` zu **16,7 %** (n = 30). Das bestätigt
[[weather-markt-schlaegt-eigenen-favoriten]] konditional statt es aufzuweichen.

**Ehrliche Grenze des Befunds:** Die Nachweisgrenze in K liegt bei **0,32 Bucket**
(t = 2,0, sd = 1,00, n = 39). Ausgeschlossen ist damit ein *großer* Warmbias —
ein kleiner von 0,1–0,2 Bucket bleibt möglich, wäre aber bei Break-even 22,6 %
ohnehin nicht handelbar. Ein Nachtest lohnt erst bei deutlich längerer Reihe;
die Sensitivitätsschnitte (Regen ≥ 3 mm: +0,194, t = 1,06) zeigen nichts, was
auf eine bessere Schwelle wartet.

**Nebenbefund ohne Gate, aber notierenswert:** außerhalb der Tropen ist der Markt
an Konvektionstagen mit −0,324 (t = −1,64) tendenziell zu **kalt** — also
gegenläufig zur These. Nicht signifikant, aber der größere der beiden Effekte.
Wer die Richtung „Regen ⇒ Markt zu warm" für allgemeingültig hielte, läge in der
Mehrheit der Städte falsch herum.

## Bekannte Schwächen (vorab benannt)

- **24 Tage Log** — die Reihe beginnt am 10.07. Selbst ein PASS ist damit nur ein
  Anfangsverdacht.
- **n = 39** trägt einen Effekt ab ~0,22 Bucket; ein feinerer Bias bliebe unsichtbar.
- Hong Kong rechnet in FLOOR-Buckets (`weather_stations.BUCKET_FLOOR`); d bleibt
  in K vergleichbar, die Rundungslage ist aber eine andere. G5 deckt den Fall ab.
- Moskau fehlt (keine Koordinate im Log, settelt ohnehin über NOAA).
