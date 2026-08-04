# BACKLOG — aufgeschobene Ideen / nächste Forschungs-Kandidaten

Zentrale Liste bewusst zurückgestellter Aufgaben (nicht „jetzt", aber nicht vergessen).
Neueste oben. Erledigtes nach unten in „## Erledigt / verworfen" oder löschen.

---

## Offen

### Latenz-Test: sitzt der Markt auf den METAR-Spiegeln oder ist er schneller?
**Hinzugefügt:** 2026-08-01 · **Status:** offen — **der Test selbst kostet nichts
und entscheidet alles Weitere**

**Anlass:** metar.ws (`app.metar.ws/docs`) verkauft METAR- und D-ATIS-Push über
WebSocket: Starter 49 €/Mon (2 Verbindungen, 10 Kanäle, 7-Tage-Test), Pro
199 €/Mon. Die Latenz-Sonde vom 31.07. wurde genau dafür gebaut — die
kostenlose Basislinie liefern, gegen die sich ein bezahlter Dienst beweisen muss.

**⚠️ Abgrenzung (04.08.2026):** Der Anbieter hat einen **zweiten** Kanal
gestartet — Tages-Max/Min aus dem Modelllauf per WebSocket („weather model
alerts", Sandbox frei: GFS 0.25°/ICON Global/ICON-EU). Dieser **Modell**-Zweig
ist geprüft und geschlossen, siehe „Fremdquelle für Modell-Tageswerte" unter
Erledigt. Der hier beschriebene **METAR-Beobachtungs**-Zweig ist davon
unberührt und bleibt offen: die Sandbox liefert weiterhin keine Live-Obs
(`metar.obs.*` → `permission_denied`), und der Test unten kostet nichts.

**Was am 01.08. bereits geprüft ist:**
- Der **freie Sandbox-Tarif liefert keine Live-Daten.** Einziger abonnierbarer
  Kanal ist `sandbox.demo`, `metar.obs.*` antwortet `permission_denied`.
  Gratis ist das Protokoll, nicht die Daten.
- **D-ATIS gibt es kostenlos ohne Auth** über `atis.info/api/<ICAO>` — aber der
  Spiegel hängt **8–11 min** hinter der Beobachtung (gemessen: KMIA 8,2 /
  KSEA 8,3 / KLGA 10,8 / KBOS 7,7 min, alle Datensätze unter 1 min alt). Damit
  ist er **nicht** schneller als NOAA (KLGA rund 7 min laut Sonde). Die Daten
  sind frei, die Geschwindigkeit ist es nicht — genau die verkauft der Dienst.
- **Stationsabdeckung passt exakt:** die sechs in der Doku genannten Kanäle sind
  unsere Settlement-Stationen (eglc, lfpb, eddm, rjtt, zbaa, klga), inklusive
  der unüblichen Wahlen London City statt Heathrow und Le Bourget statt CDG.
- **Lücken:** Hongkong hat gar kein METAR (Pseudo-Station HKO), Moskau settelt
  über NOAA, und 10 Kanäle bei Starter stehen gegen 28 Städte.
- Die `T`-Gruppe mit 0,1 °C fehlte bei KSEA und KBOS im Stichprobentext. Genau
  diese Feinstufe ist unser gemessener Bucket-Kipp-Prädiktor — vor jedem
  Verlass darauf prüfen, ob das systematisch ist oder Zufall der Meldung.

**DER TEST:** Polymarket-Trade-Tape gegen die Beobachtungszeit der
Settlement-Station. Häufen sich die Preissprünge bei **obs+7 min**, sitzt der
Markt auf den METAR-Spiegeln und eine Ein-Minuten-Quelle läge echt vorne.
Häufen sie sich bei **obs+1 min**, hat jemand den schnellen Feed längst — dann
kauft man sich den letzten Platz und die Sache ist erledigt. Vor dem Lauf Gates
fixieren; Maschinerie steht in `weather_foreknowledge_eval.py` (Beobachtungszeit
gegen Preisbewegung ist dort schon gebaut) plus die freie Polymarket-Data-API.

**Reihenfolge danach, nur bei positivem Ergebnis:** Client gegen die Sandbox
bauen (die Doku sagt zu, dass die Nutzlast formgleich ist, der Parser also
unverändert in Produktion läuft) → die 7 Trial-Tage als **reine Messzeit** in
die laufende Sonde hängen, nicht als Bauzeit → dann die 49-€-Entscheidung mit
einer echten Minutenzahl statt einer Werbezahl.

**Größenordnung im Blick behalten:** 49 €/Monat gegen ein Buch mit +10,16 $
realisiertem Gewinn in drei Wochen, dessen Ausführung ab 250 $/Wette wegbricht.
Der D-ATIS-Zweig deckt zudem nur 12 US-Drehkreuze ab, während unser US-Strang
überwiegend Gate-ROT ist — übrig sind Miami und Seattle.

**Verweise:** `weather_source_latency_probe.py` (läuft seit 31.07. auf dem VPS),
`weather_foreknowledge_eval.py`, `POLYMARKET_DATA_API.md`.

---

### Wächter für −1-Lays: vorregistrierter Ausstieg ab 16:20/17:20 Ortszeit
**Hinzugefügt:** 2026-08-01 · **Status:** gemessen, nicht gebaut — **Vorbedingung
für Breite, kein Zusatz zum heutigen Bot**

**Idee:** Der Autobuy (`weather_minus1_autobuy.py`) layt die Klasse eine Stufe
unter dem Markt-Favoriten und hält bis zum Settlement — kein Ausstieg, kein
Stop. Verloren wird genau dann, wenn das Tageshoch **exakt auf dem gelayten
Bucket stehenbleibt**. Ein Wächter würde das Signal „gerundetes laufendes
Tagesmaximum sitzt zur Ortsstunde T auf dem gelayten Bucket" auswerten und
aussteigen, statt die Position auf null laufen zu lassen.

**Zwei Gründe, warum das eine Maschine machen muss — der zweite ist der
wichtigere:**
1. Der Betreiber ist unter der Woche nicht da. Ohne Wächter läuft jede
   verlierende Position ungebremst auf null.
2. **Wenn er da ist, ist die Handentscheidung messbar negativ.** Alle drei
   Verlustausstiege der Kampagne (Tabelle unten, −4,06 $ gegenüber Halten)
   sind Handentscheidungen vor dem Bildschirm, keine folgt einer Regel; der
   Betreiber hat den Auslöser am 01.08. selbst benannt („vor dem Rechner die
   Nerven verloren", dasselbe Muster wie beim Helsinki-Aufstocken am 24.07.,
   nur mit umgekehrtem Vorzeichen). Der Wächter ist deshalb nicht nur ein
   Ersatz für Abwesenheit, sondern **der Schutz vor der Anwesenheit** — die
   Entscheidung gehört an eine feste Regel, nicht an den Nachmittag.

**✅ SCHRITT 0 ERLEDIGT (02.08.2026, `weather_daily_max_timing_isd.py`) — die
Basisrate ist breit gemessen, und sie kippt die Prüfstunde.**

4.375 Stadt-Tage, 26 Städte, zwei Sommer (2024/2025), NCEI ISD `global-hourly`,
tokenfrei. „Das Tageshoch kommt um Ortsstunde T noch":

| Ortszeit | global neu | bisher (5 EU-Städte) |
|---|---|---|
| 13:20 | **62,0 %** | 91 % |
| 14:20 | **44,5 %** | 87 % |
| 15:20 | **26,8 %** | 76 % |
| 16:20 | **13,1 %** | 41 % |
| 17:20 | **4,5 %** | 12 % |

**Die alte Messung war nicht falsch, sondern nicht übertragbar.** Auf
London/Paris/Madrid reproduziert sie sich fast punktgenau (92/83/70/45 gegen
91/87/76/41). Der Fehler war, eine europäische Kurve global anzuwenden.

Die Spreizung bei 16:20 ist der eigentliche Befund:

    Madrid 71,3 | Paris 41,8 | Ankara 26,5 | Milan 26,2 | Munich 22,7
    Chengdu 20,5 | Amsterdam 19,1 | London 15,7 | Helsinki 13,9 | Toronto 13,8
    Moscow 13,4 | Buenos Aires 12,3 | Warsaw 12,1 | Wuhan 5,4 | Beijing 5,2
    Kuala Lumpur 4,7 | Wellington 4,2 | Panama City 2,0 | Mexico City 1,9
    Seoul 1,8 | Shanghai 1,1 | Cape Town / Sao Paulo / Tel Aviv / Tokyo /
    Taipei je 0,6

Europa steht geschlossen oben, Asien und Amerika unten. In Taipei ist das Hoch
um 16:20 in 99,4 % der Fälle durch — dort feuert eine 16:20-Prüfstunde rund vier
Stunden zu spät. **Eine globale Prüfstunde ist damit nicht haltbar.** Für den
Wächter ist das kein Rückschlag, sondern eine Verbesserung: in den meisten
Städten kann er deutlich früher zuschlagen, als die alte Zahl nahelegte.

*Lücke:* Jeddah hat in ISD keine Tage mit durchgehender Nachmittagsabdeckung —
26 von 27 Stationen verwertbar.

**Nächster Schritt für den Wächter:** Prüfstunde **je Stadt** aus dieser Reihe
ableiten (z. B. die Stunde, ab der die Rate unter eine feste Schwelle fällt),
das Ganze **vorregistriert** — nicht die Schwelle solange verschieben, bis der
Backtest gefällt. Die Trennschärfe-Messung vom 24.07. (67 % gegen 2 % Basisrate)
ist damit ebenfalls neu zu rechnen: sie stand auf derselben europäischen Basis.

**Stand der Messung (24.07.2026, `weather_lay_guardrail_eval.py`,
`preregs/weather_lay_guardrail_2026_07_24.md`, Lead 1, 13 Zieltage 11.–23.07.,
in-sample):**
- Signal trennt scharf: 16:20 Ortszeit **67 % Treffer gegen 2 % Basisrate**,
  17:20 → 88 % gegen 1 %.
- Markt preist es nicht ein: Ausstiegspreis 0,545 gegen echte Gewinnquote 0,293
  (16:20); tages-geclustert **13/13 Tage positiv, Mittel +0,294, t = 5,46**,
  ohne die zwei besten Tage +0,234.
- **Früh am Tag ist der Markt dagegen exakt kalibriert** (13:20 → Kante +0,006).
  Vor ~16:20 gibt es beim Ausstieg nichts zu holen, nur Reibung zu zahlen.
- Nutzen hängt an der Breite der Auswahl:

  | Menge | Pos. | Verliererquote | Halten | Wächter 17:20 |
  |---|---|---|---|---|
  | alle Kandidaten | 144 | 21,5 % | +30,60 $ | **+79,75 $** |
  | Live-Auswahl des Bots | 38 | **7,5 %** | +10,52 $ | **+6,38 $** |

  Im engen Buch **schadet** der Wächter zu jeder Prüfstunde: bei ~92 %
  Trefferquote kappt er mehr Gewinner, als er Verlierer rettet.

**Neue Evidenz (01.08.2026, Jupiter `/v2/history`, 79 abgerechnete
Wetter-Positionen der Hot-Wallet):** Die Vorhersage ist live eingetreten.

| Menge | n | Einsatz | PnL | ROI |
|---|---|---|---|---|
| gehalten bis Settlement | 70 | 359,45 $ | **+13,73 $** | **+3,82 %** |
| vorzeitig verkauft | 9 | 44,66 $ | **−3,57 $** | **−8,00 %** |

Die 9 Verkäufe zerfallen in 6 Gewinnmitnahmen (alle grün, +0,19 bis +0,86 $)
und 3 Verlustausstiege, die das gesamte Minus tragen — davon war einer richtig:

| Ausstieg | Preis | Ist | Lay | realisiert | bei Halten | Diff |
|---|---|---|---|---|---|---|
| Seoul 27 °C 22.07. | 0,94→0,33 | 27 | verloren | −3,25 $ | −4,83 $ | **+1,58** ✓ |
| Kapstadt 15 °C 22.07. | 0,81→0,64 | 16 | gewonnen | −1,19 $ | +1,07 $ | −2,25 ✗ |
| Mexiko-Stadt 25 °C 31.07. | 0,81→0,45 | 26 | gewonnen | −2,32 $ | +1,06 $ | −3,38 ✗ |
| **Summe** | | | | **−6,76 $** | **−2,70 $** | **−4,06 $** |

Alle drei waren Handentscheidungen, keine Bot-Regel. Der Mexiko-Ausstieg fiel
auf **13:57 Ortszeit** — verkauft wurde in einen Rückgang 25→24, 89 Minuten
später kam die 26. Basisrate zu diesem Zeitpunkt: das Tageshoch kommt um 14:20
Ortszeit noch in **87 %** der Fälle (13:20 91 %, 15:20 76 %, 16:20 41 %,
17:20 12 %; 125 Stadt-Tage). ⚠️ **Diese Zahlen sind europäisch — siehe den
ISD-Befund unten. Für Mexico City gilt eine völlig andere Kurve.** Der Ausstieg
war trotzdem falsch, aber die zitierte Rechtfertigung war geborgt.
Dazu Reibung: Fill 0,451 gegen Screen 0,50
(~10 % Slippage) plus 0,13 $ Verkaufsgebühr = **8,8 % des Einsatzes allein
fürs Rausgehen**.

**Konkrete nächste Schritte:**
0. **Basisrate breit neu rechnen — Vorarbeit, ohne die die Pre-Reg auf Sand
   steht.** Die 16:20-Regel hängt heute an 125 Stadt-Tagen aus fünf
   europäischen Städten in einem Sommer. Kostenloser Ersatz gefunden und am
   01.08. live geprüft: **NCEI ISD global-hourly, ganz ohne Token**

   ```
   https://www.ncei.noaa.gov/access/services/data/v1?dataset=global-hourly
       &stations=<ISD-ID>&startDate=…&endDate=…&format=json&dataTypes=TMP
   ```

   Liefert einzelne METAR-Meldungen statt Tagesaggregate (`FM-15` = METAR,
   `FM-12` = SYNOP; `TMP:"+0300,5"` = 30,0 °C mit Qualitätsflag). Verifiziert
   an KMIA und EGLL. Damit lässt sich dieselbe Basisrate über Jahre, alle
   28 Städte, alle Jahreszeiten und beide Halbkugeln rechnen. Archivverzug
   Wochen bis Monate (Juli 2024 da, Juli 2026 noch nicht) — für Basisraten
   irrelevant, für Live/Settlement unbrauchbar. Einzige Vorarbeit: Zuordnung
   unserer 28 ICAO-Codes auf ISD-IDs (USAF+WBAN aneinandergehängt, z. B.
   Heathrow `03772099999`); NCEI veröffentlicht dafür `isd-history.csv`.
1. Pre-Reg schreiben, Parameter **vor** dem Lauf fixieren: Prüfstunde (16:20
   oder 17:20 Ortszeit), Signal-Definition, Kandidatenmenge, Mindestpreis für
   den Ausstieg (unter dem sich der Verkauf wegen Reibung nicht mehr lohnt).
2. Forward-Test schattenweise mitloggen — nicht scharf schalten, solange die
   Auswahl eng ist. Im heutigen Buch würde der Wächter Geld kosten.
3. Erst zusammen mit der Verbreiterung der Auswahl scharf schalten
   (Wächter und Breite sind ein Paket, siehe Tabelle oben).
4. Verkaufsausführung über Jupiter testen — bislang ungemessen außer den
   9 Handverkäufen (Ausführung 4–45 s, Slippage gegen Screen ~10 %).

**Vorbehalte:** 13 Tage, nur Hochsommer, in-sample. Die 16:20-Basisrate stammt
aus 5 nordhemisphärischen Sommerstädten (Helsinki, München, Paris, Madrid,
London) — auf Südhalbkugel-Winter oder Städte, deren Maximum nachts liegt, ist
sie nicht übertragbar. **Das ist ab Schritt 0 kein Vorbehalt mehr, sondern eine
Messaufgabe:** der Buenos-Aires-Fall vom 01.08. (19 °C-Lay, Ortszeit 02:42,
laufendes Hoch bereits 19) ist genau die Konstellation, für die uns die Zahl
fehlt.

**Geprüft und verworfen (01.08.):** NOAA **CDO v2**
(`ncei.noaa.gov/cdo-web/api/v2`, freier Token, 5 Anfragen/s und 10.000/Tag) ist
für diese Aufgabe das falsche Werkzeug. GHCND liefert nur **Tages**-Aggregate,
also keine Intraday-Kurve und damit keine Timing-Basisrate; als
Kalibrierungstiefe nützt es nichts, weil der bindende Engpass das
Prognose-Archiv ist (`previous_day1` reicht bis ~08/2024), nicht die Ist-Reihe;
und als Settlement-Quelle ist es riskant, weil GHCNDs Tages-TMAX über den
klimatologischen Tag der Station läuft und nicht zwingend der Marktdefinition
„all times on this day" entspricht. Der brauchbare Endpunkt ist der
Schwesterdienst ISD aus Schritt 0.

**Verweise:** `weather_lay_guardrail_eval.py` (Stufe 1 wetterseitig,
`--stage2` mit Polymarket-Preisen), `preregs/weather_lay_guardrail_2026_07_24.md`,
`weather_minus1_autobuy.py` (die zu schützende Strategie).

---

## Erledigt / verworfen

### Fremdquelle für Modell-Tageswerte (metar.ws „model alerts") + ICON-Quellen-Edge als Anker-Frage
**Hinzugefügt:** 2026-08-04 · **Status:** ✅ GESCHLOSSEN 2026-08-04 — **kein
Qualitätsgewinn, kein Abo, kein Sandbox-Client**. Messung:
`weather_icon_source_bound.py` (10 Städte, Lead d+1, 60 Tage, Open-Meteo).

**Anlass:** Werbemail des Anbieters — Tages-Max/Min direkt aus dem Modelllauf
per WebSocket auf die Flughafenstation, „kein GRIB parsen mehr". Sandbox frei
(GFS 0.25°, ICON Global, ICON-EU), Starter/Pro 49/199 € mit ICON-D2, HRRR, AROME.

**Warum das für uns nichts ändert — vier Befunde:**
1. **Die Sandbox-Modelle sind eine echte Teilmenge unserer fünf.**
   `weather_outlier_screen.py:136` fährt GFS/ICON/**UKMO/JMA/ECMWF**; die drei
   angebotenen sind drin, die drei fehlenden sind genau die, die das Ensemble
   breit machen. Wir parsen zudem kein GRIB, sondern holen fertige Tageswerte.
2. **Auch die Bezahl-Modelle sind frei.** Gegen Open-Meteo geprüft (EDDM):
   `icon_d2` → 200, `meteofrance_arome_france_hd` → 200, `gfs_hrrr` → korrekt
   „No data" (US-only). Stärker noch: **`icon_seamless` IST ICON-D2**, wo D2
   existiert (Munich/Paris/London/Milan bis auf die Nachkommastelle identisch)
   — das hochauflösende Modell der 49-€-Stufe fahren wir bei Lead 24 h längst.
3. **Stationsbezug ist kein Zugewinn** — wir extrahieren bereits auf die
   Stationskoordinate (`station_info(icao)`, Zeile 402–406), nicht auf den
   Stadtmittelpunkt.
4. **Der Spielraum der ganzen Frage ist zu klein.** Median-Spanne der
   ICON-Varianten am selben Punkt **0,35 K**. Der größere Hebel wäre die
   Extraktions*stelle* (Punkt um 5–28 km verschoben: Median max|Diff| **1,0 K**,
   Madrid 2,1 K) — **aber der ist fast reine Konstante je Stadt**
   (60-Tage-sd im Median **0,32 K** bei mittleren Versätzen bis 1,28 K/Milan),
   und Konstanten entfernt `bias_700d`/`bias_40d` per Konstruktion.

**Schluss:** Nach Abzug dessen, was die Kalibrierung ohnehin holt, bleiben aus
Quellen- **und** Extraktionswahl zusammen ~0,3 K zustandsabhängige
Variabilität — gegen einen Anker-Restfehler von 0,79 Bucket. Selbst die
perfekte Wahl erklärt den Restfehler nicht. Damit ist die alte offene These
„ICON-Quellen-Edge" **als Anker-/Qualitätsfrage** miterledigt: sie braucht den
Anbieter nicht, weil der volle Spielraum unter der Kalibrierung liegt.

**Ehrlich zur Reichweite:** Modell gegen Modell, nicht gegen Settlement — eine
**Schranke, kein Gate**. Sie sagt „hier ist nicht genug Spielraum, als dass es
sich zu messen lohnte", nicht „Quelle X ist schlechter als Y". Umgekehrt gilt:
öffnete jemand denselben Faden erneut, müsste er zuerst diese Schranke kippen.

**NICHT mitgeschlossen:** (a) der METAR-**Beobachtungs**-Latenztest oben (anderes
Produkt, weiter offen); (b) `preregs/weather_source_edge_2026_07_06.md` — das ist
eine *Handels*hypothese (ICON gegen **Marktpreis**), deren Gates nie gefahren
wurden; diese Messung entscheidet sie nicht.

**Verweise:** `weather_icon_source_bound.py`, `weather_outlier_screen.py`.

### VRP-Sleeve mit Hybrid-Risk-System kombinieren
**Status:** ✅ ERLEDIGT 2026-06-23 — **GREEN (qualifiziert)**. `vrp_hybrid_combo.py`:
VRP-gemanagt + Hybrid, monatlich, gemeinsames Fenster 2006–2026, Gewichte auf IS
gewählt / OOS evaluiert, lookahead-freier Hybrid. corr(Hybrid, VRP) OOS −0,10.
**OOS-Gewichts-Sweep:** mit steigendem VRP-Anteil sinkt MaxDD monoton
(−18,5 % → −12,0 % @50 %), Sharpe steigt (0,75 → 0,86), aber Rendite fällt
(9,8 % → 6,2 %). Diversifikation real & robust (nicht knife-edge).
**Caveats:** (a) De-Risking, kein Rendite-Boost; (b) schlägt OOS NICHT die nackte
SPY-Sharpe (0,91 — nur deren MaxDD −24 %); (c) IS-optimales Gewicht 60 % ist
overfit-anfällig (50/50 war OOS besser) → **modestes VRP-Gewicht ~25–40 % empfohlen**;
(d) Skew verschlechtert sich (Short-Vol-Charakter bleibt). Fazit: VRP **validiert als
diversifizierender Drawdown-Dämpfer** für den Hybrid, nicht als Rendite-Maschine.
Vor Live: Straddle-Approximation gegen echte Options-Daten prüfen. Test `vrp_hybrid_combo.py`.

### Overnight-Edge auf SIP-30-Min-Datensatz neu testen
**Hinzugefügt:** 2026-06-23 · **Status:** ✅ ERLEDIGT 2026-06-23 — **RED** (per Pre-Reg,
SPY primär): G2 OOS + G3 handelbar gescheitert. Phänomen deskriptiv real (QQQ/IWM),
aber Overnight-only schlägt Buy&Hold nie netto → keine Gelddruckmaschine. Details:
`preregs/overnight_intraday_2026_06_23.md`, Test `overnight_edge_test.py`.

**Idee:** Die Zerlegung der SPY-Tagesrendite in **Overnight** (Close→Open) vs.
**Intraday** (Open→Close) erneut auf sauberen Vollmarkt-Daten prüfen. Bekannter
Literatur-Befund: der Großteil der Aktienprämie fällt overnight an, intraday ist
historisch ~flach/negativ.

**Warum jetzt möglich (was sich geändert hat):** In der Session 2026-06-23 hat sich
herausgestellt, dass **Alpacas SIP-Feed gratis für Historie >15 Min** ist und bis
**2016** zurückreicht (Vollmarkt, sauber). Tool dafür existiert bereits:
`fetch_spy_30min.py` (läuft auf VPS, Keys in `config.py`). Damit entfällt die
yfinance-30-Tage-Grenze, die den bisherigen Intraday-Strang limitierte.

**Vorheriger Stand (nicht bei Null anfangen):** Der Overnight/Intraday-Strang lief
schon, endete aber **YELLOW / G0-limitiert**:
- `overnight_intraday_g3_pit.py`, `overnight_intraday_rolling_bootstrap.py`,
  `overnight_es_futures_g0check.py`, Cache `spy_overnight_ohlc.pkl`.
- Commits: `97b3b371` (Rolling-Stabilität + Bootstrap-KI + Strukturbruch),
  `0c6b0487` (Pre-Reg ES-Futures), `c21ce7ce` (ES-Futures **G0-FAIL** + YELLOW-Schluss).
- Knackpunkt war Datenqualität/Granularität — genau das adressiert der SIP-Datensatz.

**Vorbefund (2026-06-23, `cross_session_test.py`):** Erste Hinweise schon da.
Auf SPY 30-Min SIP (2016–2026, ~2630 Nächte): Nachmittag[t]→Vormittag[t+1] ist
**null** (corr −0,015, Hit 50,5 %). ABER **PM[t]→Overnight-Gap[t+1] = −0,156**
(t≈8) und PM[t]→PM[t+1] = −0,108 → **milde Overnight-/Tages-Reversion** der
Spätbewegung (R² nur ~2,4 %). Statistisch real, aber vermutlich NICHT handelbar:
zum Ausnutzen müsste man über Nacht short gehen (gegen positiven Overnight-Drift
+ ~72 bps Gap-Varianz + Spread/Finanzierung). Genau hier ansetzen beim echten Test.

**Konkrete nächste Schritte:**
1. `fetch_spy_30min.py` verallgemeinern → ggf. mehr Symbole / nur Open- & Close-Bars
   ziehen (Overnight braucht Open + vorigen Close, nicht zwingend volle 30-Min-Bars).
2. Pre-Reg G1–G5 schreiben (Overnight-Rendite vs. Intraday, netto nach Kosten;
   Achtung: Overnight ist NICHT handelbar ohne Übernacht-Halten → Strategie sauber
   definieren, z. B. „nur overnight halten" via MOC-Kauf / MOO-Verkauf).
3. OOS-Split + COVID-Kontrolle wie gehabt. Kosten realistisch (Übernacht-Spread,
   Finanzierung). Ehrlicher FAIL einkalkuliert (Projekt-Gesamtlage: Edges sterben OOS).

**Verweise:** `intraday_volume_profile.py` (U-Form-Motivation),
`preregs/intraday_momentum_spy_2026_06_23.md` (Methodik-Vorlage, HAC-Implementierung
in `intraday_momentum_test.py` wiederverwendbar).
