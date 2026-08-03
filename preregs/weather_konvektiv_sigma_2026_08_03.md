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
