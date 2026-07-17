# Kalibrier-Reparatur 17.07.2026 — Backlog Prio 0 (Shenzhen-WU) + Prio 1 (Debias/σ(s)/Lead)

**Anlass:** Betreiber-Entscheid 17.07. („mach 1+2") auf den Optimierungs-Backlog
vom 16.07. Beides sind **Konsistenz-Reparaturen des Verfahrens, kein
Parameter-Tuning** — es wird nichts auf Marktdaten oder OOS-Ausgänge gefittet,
sondern gemessene Verfahrensfehler behoben. Die Schutz-Gates (Spannen-Veto ≤3°,
dist ≥2°, P_max ≤10 %, Doppel-Kalibrierung, Wetterfrosch-Doktrin) bleiben
unverändert.

## Prio 0 — Shenzhen gegen die Settlement-Reihe kalibrieren

**Problem (Scan 16.07.):** Polymarket settlet auf wunderground.com; für ZGSZ
speist WU die Seite nicht (nur) aus den METAR — 4/15 Tage identisch, Diffs bis
±2°. 27/28 andere Städte: WU = gerundetes METAR, dort bleibt alles beim Alten.

**Umsetzung:** `weather_source_compare.py --actuals wu` (WU-Historical-API,
31-Tage-Chunks, Tagesgrenzen in Stations-Zeitzone; Werte ganze °C = exakt die
Settlement-Größe) + `--fix-b-from` (übernimmt die σ(s)-Steigung b je Quelle aus
der Voll-CSV; der b-Fit braucht ≥3 Städte, Shenzhen läuft allein). Ergebnis als
überschreibende CSVs `weather_source_calib*_2026_07_17_shenzhen_wu.csv`
(load_calib: spätere Dateien gewinnen je (city, model)).

**Gemessen (Lead 24h):**

| Shenzhen ENS | METAR-Basis (alt) | WU-Settlement (neu) |
|---|---|---|
| 700d Bias / σ / a | −0,25 / 1,08 / 0,66 | −0,37 / **1,49** / **1,05** |
| 40d Bias / σ / a | +0,03 / 1,05 / 0,75 | −0,13 / **1,45** / **1,19** |

σ zum Settlement-Ziel ist ~**40 % breiter** als die alte Screen-Sicht (die
Scan-Schätzung √(1,0²+1,4²)≈1,7 war konservativ; die WU-METAR-Diffs sind mit
dem Forecast-Fehler teilkorreliert). Einzelmodelle gegen WU: alle σ 1,7–1,8;
**UKMO fast bias-frei (+0,01) und bestes Einzelmodell (MAE 1,33)**, GFS +1,20
zu warm. **Wirkung auf die laufenden 17.07.-Lays (Info, Näherung):** Shenzhen
33° NO wäre beim Setzen mit Settlement-Kalibrierung bei grob 4–5 % statt 0,5 %
gelegen — hätte die Gates noch bestanden, aber die Marge ist real ~8–10× dünner
als der Screen behauptete. (Zwischenstand 17.07. mittags: METAR-Max 29 / WU 27,
die Lays sind praktisch durch.)

## Prio 1 — drei Verfahrens-Fixes in den Screens

1. **Debias-vor-Mittelung** (`build_views`, von High-/Low-Screen + Selftest
   gemeinsam genutzt): jedes Modell wird erst um seinen EIGENEN Kalibrier-Bias
   korrigiert, dann gemittelt — statt rohes Mittel minus ENS-Bias. Behebt:
   (a) Modell-Ausfall — UKMO fehlt an ~150/700 Kalibriertagen, gemessene
   Verschiebung bis +0,98° (Jeddah ohne JMA); (b) robuste Sicht bekam den
   vollen ENS-Bias, obwohl der Ausreißer (samt Riesen-Bias) rausgeflogen war.
   Bei voller Modellmenge nur Konstanten-Verschiebung (erwartungsgemäß klein).
   Drop-Erkennung bleibt auf ROHWERTEN (Spanne/Ausreißer = µ-Korruptions-Signal,
   Beijing-Lehre); Modelle ohne Kalibrier-Zeile fliegen aus dem Mittel statt
   unkorrigiert mitzulaufen.
2. **σ(s) auch für Einzelmodell-Vetos** (`model_sigma`): die Kalibrier-CSVs
   tragen a/b seit heute je QUELLE (b-Fits 700d: GFS 0,238 / ICON 0,183 /
   UKMO 0,276 / JMA 0,238 / ECMWF 0,178 / ENS 0,140 — ENS identisch zum
   14.07.-Fit, stabiler Schätzer). Ältere CSVs ohne Modell-a/b fallen
   automatisch aufs feste σ zurück.
3. **Lead-Autowahl:** bei Zieltag >24h lädt der Screen die Lead-N-Familie
   (`weather_source_calib_leadN_*` / `..._calib40d_leadN_*`) oder **bricht mit
   Erzeugungs-Anleitung ab** (`--force-lead1` = markierter Notbehelf). Vorher
   nur Warnung — die 17.07.-Lays wurden am 15.07. faktisch mit Lead-24h-P
   gescreent (Madrid-Lehre 13.07.: 3/4 formale Passes waren nach Lead-2 −EV).

**Nebenfixes:** Low-Screen importierte noch das am 16.07. entfernte `MIN_EV`
(war seitdem ImportError-kaputt — niemandem aufgefallen, weil die Min-
Kalibrierung nur 4 Städte deckt); Ranking dort auf Doktrin (P_pess aufsteigend)
nachgezogen. `weather_ladder_logger.load_calib` schließt `calib40`/`_lead`
jetzt explizit aus (vorher schützte nur die zufällige Glob-Sortierung; die
künftigen `_leadN_`-Dateien hätten die 700d-Basis ÜBERSCHRIEBEN).

## Verifikation

- **Selftest erweitert + grün** (`weather_screen_selftest.py`): beide
  Beijing-Regressionsfälle werden mit neuer Logik + neuen CSVs weiter
  ABGELEHNT; neuer Fake-Kalibrier-Fall beweist Debias-Konsistenz exakt
  (voll / Modell-Ausfall / Drop → korrigiertes Mittel bleibt 30,00; alte
  Logik läge bei Ausfall/Drop 0,25° daneben).
- **Frische Voll-Kalibrierungen** 700d + 40d vom 17.07. (28 Städte, METAR)
  als neue überschreibende CSVs; Shenzhen-Basis über die 3 Tage stabil
  (Bias-Drift ≤0,01°) — der Refresh selbst ändert nichts Substanzielles.
- **Screen-Vorher/Nachher (Zieltag 18.07., gleiche Marktlage):** weiterhin
  NULL Kandidaten (8/15 Städte Spannen-Veto — kein stilles Lockern durch den
  Umbau). Sichtbare Ehrlichkeits-Gewinne: Jeddah 700d/rob 41,6 statt 42,8
  (JMA-Bias verschmierte die robuste Sicht um +1,2°), Beijing-rob = Voll-Sicht
  (Drops heben sich debiast auf), BA-40d-σ 1,0→1,2 (der 24er-Miss vom 16.07.
  fließt automatisch ein).

## Bewusst NICHT gemacht (Backlog-Rest, Betreiber-Entscheid steht aus)

Prio 2 (Saison-Harmonik, braucht Residuen-Zeitreihen), Prio 3 (+GEM/MF/CMA),
Prio 4 (EPS-Member), Prio 5 (Modell-Gewichtung) — geparkt bis mindestens zur
Klassen-Forward-Auswertung (~Ende Juli). Min-Kalibrierung weiterhin nur 4
Städte (separater Backlog-Punkt).
