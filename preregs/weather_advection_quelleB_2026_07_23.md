# Pre-Reg: Advektions-Prognose (Quelle B) — Nowcast-Scalp Wetter

**Datum:** 2026-07-23
**Autor:** veit + Claude
**Kontext:** Quelle A (Running-Max-Ratchet) über 6 Städte falsifiziert (Commits
44e6e486, 1d6e500c) — Markt ankert am Forecast-Peak und trackt live. Quelle B ist
separat und noch nicht getestet: der letzte verbliebene Prio-1-Faden des
Intraday-Nowcast-Scalps.

## Hypothese

Am Settlement-Tag führt die **Temperatur-Änderung (ΔT) einer luvseitigen
Nachbar-Station** die Temperatur der Settlement-Station (Temperaturadvektion
−V·∇T). In den letzten 1–2 h vor dem Tagesmax trägt dieses Luv-Signal Information
über das noch kommende Tagesmax, die der **Markt noch nicht gepreist** hat (Markt
verankert am nächtlichen Modelllauf + eigener Station, sieht das Luv-Frischesignal
nicht).

Nutzer-Kern-Insight: nicht Absolutwert der Nachbar-Station (Sensor-Bias,
Höhenoffset), sondern **ΔT** → systematischer Offset fällt raus.

## Daten

- **Wetter:** IEM ASOS 30-Min-METAR (`tmpc, drct, sknt`), retrospektiv für die
  bereits geloggten Markttage. Settlement-Station + Nachbar-Flughäfen ≤90 km
  (via IEM-Network-GeoJSON, Distanz + Peilung berechnet).
- **Markt:** `bb_WeatherLatency` (Centron) — `all_prices` (Bucket-Quotes als JSON,
  2-min) für exakt dieselben Städte/Tage. 22 Städte, ~9–18 Tage.
- **Start-Set:** EU4 (München EDDM, London EGLC, Paris LFPB, Madrid LEMD) — beste
  Nachbar-Dichte + Quelle A dort schon gelaufen (Vergleichbarkeit).

## Gates (vorab fixiert)

- **G0 — Physik / Lead-Lag (Killer):** Kreuzkorrelation der entrendeten ΔT-Reihe
  Settlement vs. **luvseitiger** Nachbar (Diurnal-Trend entfernt). Der Luv-Nachbar
  muss mit τ>0 FÜHREN: Peak-Korrelation bei positivem Lag, spürbar > Lag-0 UND
  > der **downwind-Kontrolle** (leeseitiger Nachbar darf nicht führen). Richtungs-
  test: Führung nur luv, nicht lee. **Kein Lead → Quelle B physikalisch tot, STOPP
  vor jedem Markt-Layer.** Schwelle: mittlere Lead-Korr (luv) − Lead-Korr (lee)
  > 0 mit t > 2 über die Tage; Peak-Lag physikalisch plausibel (Distanz/Windweg).
- **G1 — Prädiktiv fürs Max:** Luv-ΔT im Vor-Peak-Fenster sagt den Rest-Anstieg
  der Settlement-Station bis zum Tagesmax voraus (Regression, OOS-Split Städte/Tage),
  schlägt Naiv-Baseline „kein weiterer Anstieg".
- **G2 — Markt nicht gepreist:** Zum Signalzeitpunkt unterpreist der Markt den
  Bucket, auf den das Advektionssignal zeigt (Fehlbepreisung > 0 aus `all_prices`).
- **G3 — Netto:** Fehlbepreisung > Spread + 2×Fee (~5 % je Seite).
- **G4:** ≥ genug handelbare Instanzen/Zeitraum (Zahl bei G2 fixieren).
- **G5:** Nicht durch eine einzelne Stadt/einen Tag getrieben (Jackknife).

**Reihenfolge strikt G0→G3.** Jedes Gate rot = STOPP + Commit des FAIL.
OOS ist heilig, keine Post-hoc-Schnitte.

## Confounder (bekannt, im Test zu adressieren)

- Sommer-Max ist **einstrahlungsdominiert**; Advektion stark nur an Frontentagen →
  paralleles diurnales Heizen erzeugt Lag-0-Korr ohne echten Lead. Deshalb G0 auf
  **entrendeter** ΔT + Lead-vs-Lag-0 + Luv-vs-Lee-Kontrast, nicht Rohkorrelation.
- Wolken-Advektion dreht das Vorzeichen; PWS/METAR-Windrichtung verrauscht (drct
  fehlt oft bei Schwachwind → 'M').
- NWP löst Advektion explizit → Edge nur im Frische-Fenster der letzten 1–2 h.

## ERGEBNIS G0 (2026-07-23) — FAIL, STOPP

`weather_advection_g0.py` über EU4 (IEM ASOS 30-Min, entrendete ΔT-Residuen,
Luv/Lee per Windrichtung+Peilung, 15–18 auswertbare Tage/Stadt):

| Stadt  | Luv-Lead | Lee-Lead | T1 Luv(Lead−Lag)   | T2 (Luv−Lee)Lead |
|--------|----------|----------|--------------------|------------------|
| Munich | +0.078   | +0.078   | +0.097 (t 1.98)    | +0.000 (t 0.00)  |
| London | +0.125   | +0.066   | +0.058 (t 1.70)    | +0.059 (t 1.89)  |
| Paris  | +0.149   | +0.141   | −0.053 (t −1.31)   | +0.008 (t 0.19)  |
| Madrid | +0.061   | +0.057   | −0.129 (t −4.26)   | +0.003 (t 0.09)  |

**Kein Advektions-Lead in keiner Stadt.** Kernbefund: der Luv/Lee-Kontrast (T2)
ist überall ~0 → die schwache Lead-Korrelation ist **richtungsunabhängig** =
synchrones regionales Diurnal-Residuum, NICHT gerichtete Advektion. Wäre Advektion
real, müsste Luv führen und Lee nicht. Paris/Madrid zeigen sogar negatives T1.
Bestätigt den Seoul-Befund (Nachbar-Lead-Lag ≈ 0) jetzt mit dem ΔT-Ansatz über 4
Städte. Physik-erwartbar: Sommer-Max einstrahlungsdominiert, Erwärmung synchron.

**G0 rot → kein Markt-Layer (G1–G5 entfallen). Quelle B falsifiziert.**
Damit sind beide Edge-Quellen des Intraday-Nowcast-Scalps tot: Quelle A
(Ratchet, 6 Städte, 44e6e486/1d6e500c) + Quelle B (Advektion, EU4). **Prio-1-These
Intraday-Nowcast-Scalp begraben.**

## NACHTRAG G0′ (2026-07-23) — dichter PWS-Ring Madrid, bestes Instrument — FAIL

Nutzer-Beobachtung: um LEMD liegen viele WU-PWS mit großem Temperatur-Spread in
verschiedenen Richtungen. Idee: die 4-Airport-Sparsity war zu grob; ein dichter,
feiner (5-min) Ring erlaubt den ECHTEN 2D-Advektionsoperator statt „eine Luv-Station".
`weather_advection_g0b.py`: 7 PWS (IGETAF33/IMAJAD13/IMADRI651/IALCOB23/IDAGAN15/
IALGET16/IRIVAS74) via WU-PWS-History-API (Web-Key), ~5-min, LEMD im Ring-Zentrum
(Peilungen 60–356°, gute Richtungsabdeckung). Jede Station **trailing** entrendet
(kausal, killt Wärmeinsel+Sensor-Bias = 2D-ΔT), Ebenen-Fit ∇r über den Ring,
**A′(t) = −V·∇r [°C/h]** mit LEMD-METAR-Wind. Kausaler Prognosetest:
`T_L(t+τ)−T_L(t) ~ 1 + [T_L(t)−T_L(t−τ)] + A′(t)`, Tag-Jackknife-Cluster-t.

| τ [min] | n | Tage | β(A′) | Jack-t | r(A′,künftig) | r(A′,vergangen) |
|---|---|---|---|---|---|---|
| 30 | 412 | 20 | −0.127 | −0.85 | −0.011 | +0.283 |
| **60** | 412 | 20 | **−0.500** | **−1.47** | **+0.030** | **+0.341** |
| 90 | 412 | 20 | −0.534 | −0.98 | +0.051 | +0.277 |

**KEIN LEAD, robust über alle Horizonte.** β durchweg negativ (falsches Vorzeichen)
+ insignifikant. **Placebo ist der Killer:** A′ korreliert mit der KÜNFTIGEN
LEMD-Änderung ≈0 (+0.03), aber mit der VERGANGENEN +0.28..+0.34 — das Signal blickt
zurück, nicht vor (koinzident-bis-nachlaufend). A′ ist dabei nicht null (±0.5 °C/h,
Median |A′| 0.17): es gibt echte Advektions-Tendenz, sie **führt LEMD nur nicht**.
Auf 15–25 km im Sommer ist die Erwärmung regional synchron — bis eine Anomalie im
Ring-Gradienten sichtbar ist, sitzt sie schon auf LEMD.

**Quelle B jetzt mit dem BESTMÖGLICHEN Instrument endgültig falsifiziert** (dichter
5-min-Ring, echter −V·∇T-Operator, gute Geometrie, 412 Obs, 3 Horizonte). Kein
offener Advektions-Faden mehr.
