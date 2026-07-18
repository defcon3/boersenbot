# Pre-Reg 18.07.2026 — Backlog Prio 4: echte EPS-Ensembles statt σ-Klimatologie

**These:** Die Member-Verteilung echter Ensemble-Systeme (ECMWF-EPS 51,
GEFS 31, ICON-EPS 40 Member) ist ein flow-dependentes σ des KONKRETEN Tages
und bepreist die Grad-Buckets besser als unsere σ(s)=a+b·Spanne-Klimatologie
(die genau dafür der Arme-Leute-Proxy ist, Pre-Reg
weather_spread_sigma_2026_07_14.md).

## Machbarkeits-Befund 18.07. (warum ein Forward-Logger nötig ist)

Beide historischen Abkürzungen sind GESCHLOSSEN (gemessen, London EGLC):

1. **`past_days` der Ensemble-API ist KEIN Forecast-Archiv:** Member-SD der
   Tagesmaxima für vergangene Tage 0,32–0,50 °C vs 0,64–0,90 °C für
   Zukunftstage — die API liefert rückwirkend den jeweils jüngsten Lauf
   (Lead ~0–6 h). Eine Kalibrierung darauf würde die EPS-Schärfe auf
   Lead 24 h drastisch überschätzen.
2. **Die Previous-Runs-API kennt `ecmwf_ifs025_ensemble` nur dem Schema
   nach:** HTTP 200 mit `..._previous_day1_memberXX`-Reihen, aber ALLE Werte
   null — 6 Stichfenster 2024-06 bis 2026-07, je 3 Retries (bekannte
   Leere-Antwort-Macke ausgeschlossen).

→ Lead-24h-Member-Historie existiert nirgends rückwirkend. Sammlung ab
**18.07.2026** via `weather_eps_logger.py` (täglich, 28 Städte × 3
EPS-Modelle, Member-Tagesmaxima des morgigen lokalen Tages →
`preregs/weather_eps_log.csv`; erster Lauf 18.07. 12:51Z, 84 Zeilen,
Zieltag 19.07.).

## Design

- **Logging:** 1×/Tag zur Screen-Zeit (~06–12 UTC). Verglichen wird die
  EPS-Verteilung mit der Normal-P aus µ_korr + σ(s) DESSELBEN Zeitpunkts —
  gleiche Informationslage, kein Lead-Vorteil für eine Seite.
- **Ist:** METAR (IEM, report_type 3+4) in Stations-Zeitzone; Shenzhen: WU.
- **EPS-Bucket-P:** empirischer Anteil der Member im Bucket [k−0,5, k+0,5)
  je Modell; zusätzlich gepoolt über die 3 Modelle (gleichgewichtet).
  Bekannte Unterdispersion: sekundär auch eine walk-forward
  spread-kalibrierte Variante (Member-σ × Faktor, Faktor aus allen Tagen
  < t) — beide Varianten sind hiermit VOR der Auswertung registriert.
- **Benchmark:** Normal-P aus dem Live-Verfahren (Debias-vor-Mittelung,
  σ(s) aus der jeweils gültigen Kalibrier-CSV, pessimistischste Sicht).

## Gates (fixiert 18.07., vor jeder Auswertung)

- **G1 (Datenbasis):** ≥ 45 Logtage und ≥ 800 auswertbare Stadt-Tage
  (Auswertung frühestens ~Anfang September 2026).
- **G2 (Verteilungsgüte):** mittlerer CRPS der EPS-Verteilung (roh ODER
  kalibriert — beide werden berichtet, Bonferroni ×2: t > 2,24) schlägt die
  Benchmark-Normal auf denselben Stadt-Tagen.
- **G3 (Bucket-Ebene, entscheidend):** mittlerer Brier-Score der
  EPS-Bucket-P über alle Grad-Buckets mit Markt-Preis schlägt die
  Benchmark-P, paired t > 2 über Stadt-Tage.
- **G4 (Tail-Ehrlichkeit):** in der Lay-Zone (Benchmark-P_pess < 10 %) darf
  die EPS-P die realisierte Trefferquote nicht UNTERSCHÄTZEN (kein
  Sicherheits-Verlust durch zu scharfe Ensembles; einseitige Prüfung).

**Konsequenz bei GRÜN:** EPS-σ/P als ZUSÄTZLICHE pessimistischste Sicht in
den Screen (max-P-Logik wie Doppel-Kalibrierung — kann Kandidaten nur
kosten, nie freischalten); Ersetzungs-Fragen separat.
**Konsequenz bei ROT:** σ(s) bleibt, Logger wird abgeschaltet, Befund
committet.

**✅ Entschieden 18.07. (Betreiber): VPS-Timer.** `boersenbot_eps_logger.timer`
(täglich 07:00 UTC, Persistent=true) + oneshot-Service, deployt + verifiziert
(Probelauf idempotent „0 neue Zeilen" auf dem Tag-1-Bestand; nächster Lauf
19.07. 07:00 UTC). **Die führende Log-Reihe liegt damit auf dem VPS** —
lokal NICHT mehr laufen lassen (zwei divergierende Reihen); vor der
Auswertung `preregs/weather_eps_log.csv` per scp holen und committen.
Tag 1 (18.07., 12:51Z) wurde als Startbestand auf den VPS kopiert; ab Tag 2
ist der Lauf-Zeitpunkt konstant 07:00 UTC.
