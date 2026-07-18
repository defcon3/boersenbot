# Pre-Reg 18.07.2026 — Backlog Prio 2 (Saison-Harmonik), Prio 3 (8 Modelle), Prio 5 (Gewichtung)

**Anlass:** Betreiber-Entscheid 18.07. („weiter mit der Prio-Liste") nach der
Klasse-B-Auswertung (e8866139). Die Punkte waren seit 17.07. geparkt; Prio 0+1
sind live (5236e404). **Diese Gates sind VOR der Auswertung fixiert** (Session
18.07., Auswertung startet erst nach Commit-fähigem Stand dieser Datei).

**Charakter:** Prio 2 ist wie Prio 0+1 eine Verfahrens-Frage (wie schätzt man
den Bias einer Quelle konsistent?), KEIN Markt-Edge-Test — es wird nichts auf
Marktpreise oder OOS-Ausgänge gefittet. Trotzdem gilt walk-forward-Disziplin,
weil die Harmonik mehr Parameter hat als die Konstante und sich auf 700 Tagen
leicht in-sample schönrechnet.

## Datenbasis (ein Lauf für alles)

`weather_source_compare.py --days 700 --models ext --dump-residuals
preregs/weather_residuals_lead1_2026_07_18.csv.gz` — Lead 24h
(previous_day1), Ist = METAR (IEM, report_type 3+4), 28 Städte, 8 Modelle
(Basis GFS/ICON/UKMO/JMA/ECMWF + neu GEM/MeteoFrance/CMA). Dump je Zeile:
`city,model,date,forecast,actual` (Einzelmodelle; Ensembles werden in der
Auswertung aus den Einzelzeilen gebildet). Die Live-Kalibrier-CSVs werden von
diesem Lauf NICHT berührt (kein `--calib-csv`).

**Shenzhen** bleibt in allen drei Auswertungen AUSSEN VOR (Settlement-Ziel ist
die WU-Reihe, nicht METAR — Prio 0; eine METAR-Harmonik wäre dort wieder das
falsche Ziel). Es bleiben 27 Städte.

## Gemeinsame Methodik

- **Walk-forward:** Schätzung am Tag t nutzt ausschließlich Tage < t.
  Anlauf fix 180 Tage; bewertet werden alle Tage ab Tag 181 der jeweiligen
  Stadt-Serie. Keine Parameter werden auf den Bewertungs-Tagen gewählt.
- **Bewertungs-Ebene:** ENS5-Residuen (Tageshoch-Forecast − Ist des
  arithmetischen Mittels der 5 Basismodelle, nur Tage an denen alle 5 liefern)
  für H2; Modell-Ebene für H3/H5 wie unten.
- **Statistik:** paired Differenzen der |Fehler| je Tag; primäre t-Statistik
  über die 27 Stadt-Mittel (Städte als unabhängige Einheiten — konservativ
  gegenüber Tages-Pooling mit Autokorrelation); gepoolter Effekt in % als
  Größenmaß.

## H2 (Prio 2) — bias(doy)-Harmonik schlägt beide Fenster-Konstanten

Schätzer für den ENS5-Bias am Tag t, alle walk-forward:
1. `const` — expandierendes Mittel aller bisherigen Residuen (= 700d-Logik),
2. `roll40` — Mittel der Residuen im Kalenderfenster [t−40, t); unter 15
   Werten Fallback auf `const` (= 40d-Logik),
3. `harm1` — OLS resid ~ 1 + sin(2πd/365,25) + cos(2πd/365,25) (**primär**),
4. `harm2` — + 2. Harmonische (nur Sensitivitäts-Bericht, keine Auswahl).

Zielgröße: OOS-MAE von (resid − bias_hat).

**Gate G-H2 (Screen-Einbau als Zusatz-Sicht):** harm1 hat kleineren gepoolten
OOS-MAE als `const` UND als `roll40`, und die paired t-Statistik über die
Stadt-Mittel ist > 2 gegen BEIDE. Ein Test, kein Grid — harm1 ist vorab die
einzige Einbau-Kandidatin.

**Konsequenz bei GRÜN:** Harmonik-Koeffizienten (Full-Sample-Fit je
Stadt × Quelle) als DRITTE Kalibrier-Familie in den High-Screen — sie geht nur
in die P_pess-max-Bildung ein und kann Kandidaten damit ausschließlich
verlieren, nie freischalten; σ bleibt das σ(s) der 700d-Familie; 700d- und
40d-Familie bleiben unverändert Pflicht (Backlog-Vorgabe „Doppel-Kalibrierung
behalten"). Ob die Harmonik später eine der beiden Fenster-Sichten ERSETZT
(erst das würde z. B. Beijings 700d-Vorzeichendreher heilen), ist ein
separater Betreiber-Entscheid auf Basis dieser Zahlen.
**Konsequenz bei ROT:** kein Einbau, Befund committen, Zwei-Fenster bleibt.

## H3 (Prio 3) — bringt die Erweiterung auf 8 Modelle etwas?

Deskriptiv (kein Gate nötig): Archiv-Tiefe n je neuem Modell und Stadt;
Bias/σ/MAE je Modell auf der Alle-8-Schnittmenge; Heimmodell-These (GEM in
Toronto, MeteoFrance in Paris, CMA in Beijing/Shanghai/Chengdu/Wuhan — Rang
dort vs. Median-Rang in Fremdstädten).

**Gate G-H3 (Empfehlung „Screens auf 8 umstellen" wird ausgesprochen):**
walk-forward-debiastes ens8 schlägt walk-forward-debiastes ens5 auf identischer
Tagesmenge (alle 8 liefern) im OOS-MAE gepoolt um ≥ 3 %, in ≥ 60 % der
auswertbaren Städte, UND alle 3 neuen Modelle haben n ≥ 350 in ≥ 20 Städten
(sonst trägt die Archiv-Tiefe keine 700d-Kalibrierung).

**Konsequenz:** Auch bei GRÜN in dieser Session KEIN Auto-Einbau — die
Modellmenge verschiebt µ und verändert die Spannen-Veto-Semantik (8er-Spanne
≥ 5er-Spanne). Es gibt dann eine Empfehlung mit Zahlen, Einbau ist
Betreiber-Entscheid.

## H5 (Prio 5) — Inverse-Kovarianz-Gewichtung schlägt Gleichgewichtung

Auf den 5 Basismodellen (live-relevant; 8er nur Sensitivität), je Stadt,
walk-forward mit monatlichem Refit: Modell-Varianzen + EIN gemeinsames ρ
(Gleichkorrelations-Annahme wie in der Backlog-Messung) expanding geschätzt,
w ∝ Σ⁻¹·1, negativ geklippt und renormiert; jedes Modell vorher um seinen
expanding-const-Bias korrigiert (gleiches Debias für beide Arme).

**Gate G-H5 (Empfehlung wird ausgesprochen):** OOS-MAE-Reduktion des
gewichteten vs. gleichgewichteten debiasten Mittels gepoolt ≥ 5 % UND
Stadt-t > 2. Seoul wird zusätzlich einzeln berichtet (Backlog-Messung
versprach dort −53 % σ — in-sample; hier zeigt sich, was OOS übrig bleibt).
Zusatzarm „JMA+UKMO raus, Rest gleichgewichtet": wird mitberichtet; falls er
statt der Gewichtung zur Empfehlung werden soll, gilt Bonferroni ×2 (t > 2,24).

**Konsequenz:** wie H3 — Befund + ggf. Empfehlung, kein Auto-Einbau
(µ-verschiebend).

## Prio 4 (EPS) — nicht Teil dieser Gates

Eigener Machbarkeits-Check (liefert `past_days` der Ensemble-API archivierte
Original-Forecasts oder Analysis-nahe Werte?) + eigene Pre-Reg, separat
dokumentiert.

## Ergebnis (18.07.2026, `weather_calib_prio235_eval.py` auf
`weather_residuals_lead1_2026_07_18.csv.gz` — 145.084 Zeilen, 27 Städte,
2024-08-16 bis 2026-07-16)

**Alle drei Gates ROT.** Es wird nichts eingebaut; Screens bleiben unverändert.

### G-H2 ROT — die Harmonik schlägt die Konstante, aber nicht das 40d-Fenster

Gepoolter walk-forward-MAE des debiasten ENS5: const **0,967** / roll40
**0,912** / harm1 **0,907** / harm2 0,909. harm1 vs const: **+6,22 %**
(Stadt-t 2,58; 19/27 besser) — die Saisonalität ist also REAL und die reine
700d-Konstante ist messbar der schlechteste Schätzer (Spitzen: Seoul +35,6 %,
Tel Aviv +28,1 %, Beijing +14,3 %, München +10,0 % — exakt die bekannten
Saisonspringer). Aber harm1 vs roll40: nur **+0,56 %** (Stadt-t 0,69; 17/27)
→ das rollierende 40d-Fenster fängt die Saisonalität fast vollständig, ohne
Funktionsform. **Die Zwei-Fenster-Heuristik ist empirisch quasi-optimal;
der formale Makel (max über Sichten statt einer Verteilung) kostet keine
messbare µ-Güte.** Die 2. Harmonische bringt nichts (0,909). Neben-Befund:
roll40 < const validiert nachträglich, dass die 40d-Sicht der µ-Träger der
Doppel-Kalibrierung ist und die 700d-Sicht ihr konservatives Netz.

### G-H3 ROT — 8 statt 5 Modelle: +1,52 % gepoolt, 59 % der Städte

Archiv-Tiefe wäre kein Hindernis (GEM/MF/CMA je 27/27 Städte n≥350). Aber
walk-forward-debiast ens8 vs ens5: **+1,52 %** (Gate ≥3 %), 16/27 = 59 %
(Gate ≥60 %). Gewinner sind die Problemstädte (Tel Aviv +9,8 %, Jeddah
+9,1 %, Seoul +8,0 %), Verlierer die ruhigen (Milan −8,5 %, Warsaw −7,8 %,
Toronto −4,5 %). MAE-Rang (1=best): **ICON 2,04**, UKMO 3,70, ECMWF 3,85,
MF 3,85, GFS 4,15, GEM 4,52, **CMA 6,89, JMA 7,00**. Heimmodell-These
differenziert: **GEM ist in Toronto Rang 1** (Fremd-Median 5), **MF in Paris
Rang 2** (Fremd-Median 4) — real; CMA ist auch zuhause Rang 7,5–8 (Beijing/
Chengdu/Wuhan 8, Shanghai 6) — tot.

### G-H5 ROT (knapp) — Gewichtung +5,63 % gepoolt, aber Stadt-t 1,91

Inverse-Kovarianz (Gleichkorrelation, monatlicher Refit, walk-forward)
vs Gleichgewichtung: gepoolt **+5,63 %** (Gate ≥5 % ✓), Stadt-t **1,91**
(Gate >2 ✗), 17/27 besser. **Seoul OOS +43,1 %** (1,772→1,008 — vom
in-sample versprochenen −53 % σ bleibt fast alles!), München +19,5 %,
Tel Aviv +18,4 %, Wuhan +12,7 %, Jeddah +10,3 %; Kehrseite Toronto −10,3 %,
Mexico City −7,8 %. Der Quick-Win-Arm „JMA+UKMO raus" ist klar tot:
**−1,27 %** (t −0,76) — pauschales Modell-Streichen verschlechtert.

### Muster über alle drei + mögliche Folge-Pre-Reg (Betreiber-Entscheid)

Der Nutzen konzentriert sich konsistent in Städten mit großem
Modell-Güte-Gefälle bzw. starker Saisondynamik (Seoul, Tel Aviv, München,
Jeddah, Wuhan, Beijing); in ausgeglichenen Städten kosten alle drei
Verfahren leicht. Eine stadt-konditionale Anwendung („Gewichtung nur wo das
im Anlauf geschätzte σ-Gefälle > X") wäre JETZT post hoc — falls gewünscht,
als NEUE Pre-Reg mit a-priori-Kriterium und Forward-Gate registrieren.
Zweite Option: Re-Test von H5 in ~3 Monaten (das Previous-Runs-Archiv
wächst täglich, mehr Daten → schärferes t) — beides nicht in dieser Session
entschieden.

### Prio 4 (EPS): Machbarkeit geklärt, Forward-Logger gestartet

`past_days` der Ensemble-API ist rückwirkend Lead ~0–6 h (Member-SD
0,32–0,50 vs 0,64–0,90 für Zukunft), die Previous-Runs-API führt
`ecmwf_ifs025_ensemble` nur als leeres Schema (6 Stichfenster, alle null) —
es gibt KEINE historische Lead-24h-Member-Reihe. Deshalb sammelt ab 18.07.
`weather_eps_logger.py` täglich 28 Städte × 3 EPS-Modelle (51+40+31 Member,
Tagesmaxima des morgigen lokalen Tages) nach `preregs/weather_eps_log.csv`.
Gates + Auswertung (~Anfang September): eigene Pre-Reg
`preregs/weather_eps_sigma_prereg_2026_07_18.md`.
