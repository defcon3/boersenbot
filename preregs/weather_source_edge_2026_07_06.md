# Pre-Reg: Wetter-Quellen-Edge (ICON-Modell vs. Jupiter-Marktpreis) — 2026-07-06

**Status:** ⛔ **ZURÜCKGEZOGEN am 2026-08-04 — überholt, NICHT falsifiziert.**
Das Forward-Fenster wurde nie gestartet, der Logger `weather_edge_paper_logger.py`
nie geschrieben; kein Gate unten ist gefahren worden. Diese Pre-Reg ist damit
**kein RED** und darf auch nicht als solches zitiert werden — sie ist ein
Vorhaben, das die Entwicklung überholt hat. Begründung unten unter „ERGEBNIS".

*Ursprünglicher Status:* vorregistriert VOR dem Forward-Fenster. Wiedereinstieg
in das Wetter-Kapitel nach der falsifizierten statischen Open-Meteo-Bias-These
(Session 2026-07-01, `weather-forecast-bias-study`) — diesmal Quellenvergleich
statt Einzelquelle-Korrektur.

## Ausgangslage (Backtest, KEIN bestandenes Gate)

`weather_source_compare.py` (Commit dieser Session) vergleicht 5 unabhängige
NWP-Modelle (GFS/NOAA, ICON/DWD, UKMO, JMA, ECMWF) über Open-Meteos kostenlose
**Previous-Runs-API** (archiviert Original-Forecasts mit fixer Lead-Time,
`temperature_2m_previous_day1` = exakt 24h Vorlauf — NICHT die Historical-Forecast-API,
die nachträglich rekonstruiert) gegen echte METAR-Tageshöchstwerte (IEM ASOS-Archiv).

**Ergebnis (21 Städte, ~700 Tage, n≈15.000/Modell, Archiv reicht bis Anfang 2024
zurück — mehr ist technisch nicht verfügbar):**

| Modell | Bias | MAE |
|---|---|---|
| **ICON (DWD)** | −0,60 °C | **1,20 °C** ← beste |
| ECMWF | −0,56 °C | 1,26 °C |
| UKMO | −0,44 °C | 1,30 °C |
| GFS (NOAA) | −0,45 °C | 1,47 °C |
| JMA (Japan) | −1,59 °C | 2,01 °C ← schlechteste |

ICON schlägt die anderen 4 Modelle konsistent über das gesamte Städte-Set.
Alle Modelle unterschätzen leicht (negativer Bias, konsistent mit der alten
Open-Meteo-Bias-These) — aber der Markt könnte diesen Versatz (wie schon bei
der Open-Meteo-These gezeigt) bereits selbst einpreisen. **Die eigentlich offene
Frage ist nicht "ist ICON gut kalibriert", sondern "ist ICON besser kalibriert
als das, was der Markt einpreist" — nur dieser Unterschied ist handelbar.**

**Bekannte Auffälligkeiten (nicht als Bugs behandelt, aber Vorsicht bei
Einzelstädten):** Seoul (Incheon/RKSI liegt auf aufgeschüttetem Land — GFS/UKMO/JMA
je −2,2…−2,7 °C, ICON/ECMWF deutlich besser) und Jeddah (JMA −5,9 °C, Wüstenklima/
Extremhitze). Städte-Set: 21 von 22 Jupiter-Wetter-Markt-Städten (Hong Kong bleibt
unauflösbar, wie schon beim Latenz-Logger).

**Zusatzbefund — Ensemble-Mittel (H2, PARALLEL zu H1, ersetzt NICHT die
Einzelquellen-Strategie):** Der Nutzer will bewusst bei einer Einzelquelle
(ICON) bleiben — das Ensemble-Mittel aller 5 Modelle läuft nur als **zusätzlicher,
paralleler Track**, nicht als Ersatz. Ergebnis (nur Tage, an denen alle 5 Modelle
vorliegen, n≈12.100): Ensemble-Mittel σ=1,21 °C / MAE=1,15 °C — konsistenter als
jedes Einzelmodell inkl. ICON (σ=1,39 °C). Am deutlichsten bei **Mexico City**
(07.07.-Snapshot): Modelle wichen an diesem Tag 2,8 °C auseinander (ICON/UKMO
warm ~25,7 °C, GFS/JMA kühl ~23–24 °C, Markt lag näher an GFS/ECMWF) — auf
Höhenlagen-Kesselstädten wie Mexico City ist Einzelmodell-Vertrauen fragiler
als sonst. **H2 läuft separat mit identischer Methodik (eigene Kalibrierungs-
spalte `model=ensemble_mean` in der CSV), eigenes Gate, KEINE Vermischung mit
H1 in der Entscheidung.**

**Bereits informell geprüft (Session 2026-07-06, kein Pre-Reg-Gate, nur Disziplin-
Notiz):** "Lay die 2 äußersten Buckets mit Marktpreis ≥10 %"-Variante an einem
Tages-Schnappschuss (10 Städte) zeigte KEIN konsistentes Vorzeichen (Diff
Markt−Modell schwankte −19 bis +19 Punkte, Mittel nur +2,3 Punkte bei n=20) —
sieht nach gut kalibriertem Markt aus, aber Einzeltag/zu klein für ein Urteil.
Die absoluten Rand-Buckets (Marktpreis <1 %) liegen zudem an einer
Mindestpreis-/Tick-Grenze (mehrere unterschiedlich weit entfernte Buckets
kleben auf demselben Preis, z. B. Madrid 44/45/46°C+ alle exakt bei 0,002) —
dort ist vermutlich strukturell nichts abschöpfbar. Beides fließt NICHT in die
Gates ein, ist nur Kontext für die Forward-Phase.

## Hypothese

**H1 (zu konfirmieren):** Eine Handelsregel, die ICONs Day-Ahead-Forecast
(bias-korrigiert mit der stadtspezifischen Kalibrierung aus obigem Backtest,
`preregs/weather_source_calib_2026_07_06.csv`) in eine Normalverteilung über
die 1-°C-Marktbuckets übersetzt, hat auf Jupiters "Highest Temperature"-Märkten
einen positiven Netto-EV (nach Fee), wenn sie systematisch gegen den
Marktpreis gestellt wird.

**Modell-Wahrscheinlichkeit je Bucket:** `P(Bucket k) = Φ((k+0.5−μ)/σ) − Φ((k−0.5−μ)/σ)`
mit `μ = ICON-Forecast(Stadt, Tag) − bias(Stadt)` und `σ = sigma(Stadt)` aus der
CSV. Trade nur, wenn `P(Bucket) − Marktpreis(Bucket) − Fee > 0` (Fee-Modell wie
Momentum-Pre-Reg: `fee = 0,07 · min(p, 1−p)`, Projekt-Standard aus `autopilot.py`).

**Erwartung des Registrierenden: eher H1 falsch / kleiner Edge, wenn überhaupt.**
Grund: Die alte Bias-These zeigte, dass der Markt einen bekannten, groben
Bias bereits selbst korrigiert (Wellington-Fall). Ein Modell-vs-Modell-Unterschied
von ~0,1–0,3 °C MAE (ICON vs. ECMWF) ist klein gegen σ≈1,1–1,5 °C — der EV-Hebel
pro Bucket dürfte gering sein. Realistischer Kandidat für einen echten Edge:
Städte mit besonders großer ICON-Überlegenheit (Seoul, ggf. weitere Ausreißer-
Städte), NICHT das gepoolte Mittel.

**H2 (parallel, sekundär, ICON bleibt primär):** Dieselbe Handelsregel, aber
`μ/σ` aus dem Ensemble-Mittel (`model=ensemble_mean` in der Kalibrierungs-CSV)
statt ICON. Erwartung: H2 evtl. leicht überlegen an Tagen/Städten mit hoher
Modell-Uneinigkeit (Mexico-City-Typ), aber NICHT als Ersatz für H1 gedacht —
der Nutzer bleibt bewusst bei der Einzelquellen-Strategie, H2 ist ein
zusätzlicher Beobachtungs-Track.

## Daten & Methodik für das Forward-Fenster

**Neu zu bauendes Logger-Skript** (noch nicht geschrieben, Teil der Forward-Phase):
`weather_edge_paper_logger.py` — täglich, einmal pro Stadt kurz nach Verfügbarkeit
des ICON-00Z-Laufs (~06:00 UTC):
1. ICON-Day-Ahead-Forecast (`temperature_2m_previous_day1`-Äquivalent für "morgen",
   d. h. reguläre Forecast-API mit `models=icon_seamless`, `forecast_days=2`) ziehen.
2. Modell-Wahrscheinlichkeit je Bucket berechnen (Formel oben, Kalibrierung aus
   der eingefrorenen CSV — **kein Nachjustieren der Kalibrierung während des
   Fensters**).
3. Jupiters aktuellen Marktpreis je Bucket für den morgigen Tag snapshotten
   (`/events?category=weather`).
4. Alles nach Centron loggen (neue Tabelle `bb_WeatherSourceEdge`), inkl. dem
   Bucket mit größtem `P − Preis − Fee` (Paper-Order, KEIN Echtgeld).
5. Nach Tagesende: METAR-Ist-Hoch nachtragen, Paper-PnL berechnen.

## Forward-Fenster (einzufrieren bei Logger-Start)

Ab Logger-Start **14 volle Kalendertage** (Ziel: 21 Städte × 14 Tage ≈ 294
Stadt-Tage brutto, abzüglich Städte ohne aktiven Jupiter-Markt an dem Tag).
Fensterlänge bewusst länger als bei Crypto (15-Min-Takt vs. 1 Wert/Tag/Stadt
→ viel geringere Datendichte).

## Gates

Gates gelten **getrennt** für H1 (ICON, primär/entscheidend) und H2
(Ensemble-Mittel, parallel/informativ) — H2 kann H1 NICHT ersetzen oder dessen
Urteil verändern, auch wenn H2 besser abschneidet.

- **G-N (Power):** ≥ 150 Stadt-Tage MIT tatsächlich existierendem Trade
  (`P − Preis − Fee > 0` an mindestens einem Bucket). Sonst **UNDERPOWERED**.
- **G-Primär (GREEN):** `mean(paper_pnl) > 0` UND Cluster-t (auf Stadt-Ebene,
  nicht Tag-Ebene — Tage sind pro Stadt korreliert durch Wetterpersistenz)
  **≥ +2,0** netto. Einzelhypothese, keine Bonferroni-Korrektur.
- **G-Sekundär (Substanz):** GREEN nur relevant, wenn nicht von 1–2 Extremstädten
  (Seoul, Jeddah) getragen — Sensitivitätscheck "GREEN auch ohne Top-2-Städte"
  muss ebenfalls gerechnet und berichtet werden (deskriptiv, ändert das Urteil
  nicht, verhindert aber Fehlinterpretation).
- Alles andere → **RED**.

## Entscheidungsregel

- **RED:** Wetter-Kapitel (Quellenvergleich) geschlossen, als FAIL committen.
  Logger `weather_latency_logger.py` (These C, separates Kapitel) bleibt davon
  unberührt.
- **GREEN:** KEIN Livegang. Nächste Stufe wäre echte Order-Fill-Simulation
  gegen das tatsächliche Orderbuch (Spread/Slippage bislang nur über die
  Fee-Pauschale approximiert) + längeres Fenster.
- **UNDERPOWERED:** berichten, Logger optional verlängern (Entscheidung dann,
  nicht jetzt).

## Bekannte Limitationen (bewusst in Kauf genommen)

- σ ist mit ~700 Tagen gut geschätzt, aber die Kalibrierung stammt aus der
  *previous_day1*-Reihe (previous-runs-api), das Forward-Fenster nutzt die
  reguläre Forecast-API mit `forecast_days` — beide sollten denselben Modelllauf-
  Mechanismus abbilden, aber das ist NICHT bit-identisch geprüft. Falls im Forward-
  Fenster eine Regressionsabweichung auffällt (z. B. systematisch verschobener
  Bias ggü. der Kalibrierung), wird das gemeldet, ändert aber die Gates nicht rückwirkend.
- Keine Bonferroni-Korrektur trotz 21 Städte, weil die Gate-Entscheidung auf dem
  GEPOOLTEN Cluster-t fällt (eine Hypothese), nicht auf 21 Einzeltests.
- Normalverteilungsannahme für die Bucket-Wahrscheinlichkeit ist eine Vereinfachung
  (echte Prognosefehler sind oft leicht schief/dicker-tailed) — akzeptiert für
  diese erste Konfirmationsrunde, wie schon bei der alten Bias-These.

## Auswertung

Nach Fensterende: `python eval_weather_source_edge.py` (noch zu schreiben,
analog zu `eval_crypto_momentum.py`) → Ergebnis-Abschnitt hier nachtragen,
committen.

---

## ERGEBNIS — kein Ergebnis: ZURÜCKGEZOGEN 2026-08-04

**Das Fenster wurde nie eröffnet.** `weather_edge_paper_logger.py` und
`eval_weather_source_edge.py` sind nie entstanden, es existiert kein Stadt-Tag
mit Paper-PnL. G-N, G-Primär und G-Sekundär sind **ungefahren**. H1 und H2 sind
damit weder bestätigt noch widerlegt — sie sind unbeantwortet.

**Warum trotzdem zurückgezogen und nicht nachgeholt — zwei Gründe:**

**1. Die Doktrin der Pre-Reg gilt nicht mehr.** Ihr Kern ist die bewusste
Einzelquellen-Strategie („der Nutzer bleibt bewusst bei ICON", H2 nur als
paralleler Beobachtungs-Track). Das Live-System ist längst das Gegenteil:
`weather_outlier_screen.py` fährt fünf Modelle mit Ausreißer-Verwerfung,
Doppel-Kalibrierung (700d + 40d), Einigkeits-Gate `MAX_DIVERGENZ` und seit
04.08. stadt-konditionaler Gewichtung. Ein Test von „ICON allein gegen den
Markt" würde eine Fassung prüfen, die wir nicht mehr handeln.

**2. Die Anker-Hälfte der Frage ist inzwischen beantwortet — negativ.**
`weather_icon_source_bound.py` (04.08.) misst die **Schranke** der Quellen- und
Extraktionswahl: ICON-Varianten am selben Stationspunkt liegen im Median 0,35 K
auseinander (und `icon_seamless` *ist* ICON-D2, wo D2 existiert); die
Extraktionsstelle bewegt zwar bis 2,1 K, ist aber fast reine Konstante je Stadt
(60-Tage-sd im Median 0,32 K) — und Konstanten entfernt die Bias-Kalibrierung
per Konstruktion. Nach Abzug bleiben ~0,3 K gegen einen Anker-Restfehler von
0,79 Bucket. Die Prämisse dieser Pre-Reg — „ICON ist die beste Quelle, also
lohnt es, ICON gegen den Markt zu stellen" — trägt keinen Edge in der
Größenordnung, die sie unterstellt.

**Was ausdrücklich offen bleibt:** die *Handels*frage „schlägt unsere
kalibrierte Bucket-Verteilung den Marktpreis netto" ist damit **nicht**
entschieden — sie wird heute nur anders gestellt (markt-blinde
Wetterfrosch-Doktrin, Lay statt Back, `weather_minus1_autobuy.py`) und anders
gemessen. Der Backtest-Teil oben (5 Modelle, 21 Städte, ~700 Tage,
`preregs/weather_source_calib_2026_07_06.csv`) bleibt gültig und ist weiter im
Einsatz — zurückgezogen ist das Forward-Vorhaben, nicht die Kalibrierung.

**Verweise:** `weather_icon_source_bound.py`, `weather_outlier_screen.py`,
`BACKLOG.md` (Eintrag „Fremdquelle für Modell-Tageswerte").
