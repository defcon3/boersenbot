# BACKLOG — aufgeschobene Ideen / nächste Forschungs-Kandidaten

Zentrale Liste bewusst zurückgestellter Aufgaben (nicht „jetzt", aber nicht vergessen).
Neueste oben. Erledigtes nach unten in „## Erledigt / verworfen" oder löschen.

---

## Offen

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
17:20 12 %; 125 Stadt-Tage). Dazu Reibung: Fill 0,451 gegen Screen 0,50
(~10 % Slippage) plus 0,13 $ Verkaufsgebühr = **8,8 % des Einsatzes allein
fürs Rausgehen**.

**Konkrete nächste Schritte:**
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
sie nicht übertragbar; dort separat messen.

**Verweise:** `weather_lay_guardrail_eval.py` (Stufe 1 wetterseitig,
`--stage2` mit Polymarket-Preisen), `preregs/weather_lay_guardrail_2026_07_24.md`,
`weather_minus1_autobuy.py` (die zu schützende Strategie).

---

## Erledigt / verworfen

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
