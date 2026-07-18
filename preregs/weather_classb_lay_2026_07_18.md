# Pre-Reg: „Klasse B" — Lay der ±1/±2-Offset-Fenster (18.07.2026)

**Anlass:** Nutzer-Entscheid 18.07. („wenn es nicht mehr Bretter gibt, dann mach die
klasse-b-auswertung") nach der Shenzhen-Diskussion (29er/33er). Frage: Gibt es
zwischen den sicheren Screen-Kandidaten (dist ≥ 2°, P ≤ 10 %) und dem Zentrum eine
**handelbare Rendite-Klasse mit bewusst mehr Risiko** — oder ist das Band
Beijing-33-Land?

**Registriert VOR dem Lauf:** 18.07.2026 ~09:15 UTC. Teilweise Vorwissen liegt offen:
Die 700d-Klassenraten-Methodik existiert seit 11.07. (weather_error_quantiles.py,
Kernbefund auf 6 Default-Städten: P(±2) ≈ 6–7,5 %, P(±3) ≈ 1–2,3 %, Zentrum schärfer
als Normal-Annahme). **Neu und unbekannt** ist: volles handelbares Städte-Universum,
die echte Preisseite je Klasse (bb_WeatherLadders seit 11.07.) und der daraus
folgende Netto-ROI inkl. Fee — sowie der kleine realisierte Forward-PnL.

## Definition Klasse B

Exakte Grad-Fenster (kind='eq') mit **Offset ±1 oder ±2** zum bias-korrigierten
Modell-Favoriten k0 = round_half_up(mu_ens), var = **max** (Min-Bretter existieren
kaum). Klassen getrennt: −2, −1, +1, +2 (kalte/warme Flanke). Der Shenzhen-29er von
heute ist der Prototyp der −2-Klasse, der 33er der +2-Klasse.

## Daten & Methodik

- **A (Klassenraten, ~700d):** Methodik weather_error_quantiles.py — Open-Meteo
  previous_day1 (5-Modell-ENS, LOO-Bias), IEM-METAR-Ist, alle Städte des handelbaren
  Universums (distinct city aus bb_WeatherLadders, var='max', mu_ens NOT NULL) mit
  n ≥ 50. Gepoolt + je Stadt. (Shenzhen: METAR-Ist statt WU — bekannte, kleine
  Optimismus-Verzerrung, wird ausgewiesen.)
- **B (Preisseite, echt):** bb_WeatherLadders seit 11.07., **nur Vortags-Snapshots**
  (snapshot_utc-Datum < target_date; Zieltag-Snapshots sind intraday und halb
  entschieden), status='open', 0 < buy_no < 1. Je Klasse: n, Ø/Median buy_no.
  Kostenmodell wie im Bestand: cost = NO + 0,07·min(NO, 1−NO).
- **C (Forward-Kontrolle, echt):** dieselben Zeilen mit settle_result NOT NULL →
  realisierter Lay-PnL je Klasse (je Fenster ein Vortags-Snapshot, keine
  Doppelzählung).
- **ROI-Formel:** ROI = (1 − P_emp)/Ø-cost − 1. Signifikanz: t = ((1 − Ø-cost) −
  P_emp)/SE_P mit Binomial-SE der Klassen-P (Abstand zur Netto-Break-even-P in SEs).

## Gates (vorab fixiert)

- **G-B1:** Netto-ROI der Klasse > +3 %
- **G-B2:** t > 1,5
- **G-B3:** Forward-PnL (Teil C) widerspricht nicht: bei N ≥ 30 gesettelten Fenstern
  der Klasse kein ROI < −5 %; darunter nur nachrichtlich
- **G-B4:** Robustheit: ROI bleibt > 0 ohne die beste Stadt; Seoul-Ausschluss
  (bekannter Fat-Tail-Ausreißer) ändert das Vorzeichen nicht

## Vorab festgelegte Konsequenz

- **Grün (alle Gates):** Klasse B wird live gehandelt als GETRENNTE Serie: eigener
  Mini-Stake 2–3 $/Lay, max. 2 Lays/Tag, getrenntes Tracking, Review nach 30 Lays.
  Die A-Klasse (Screen-Gates) bleibt unverändert — kein stilles Lockern.
- **Rot/teilgrün:** kein Live-Geld; bb_WeatherLadders-Forward läuft weiter,
  Re-Auswertung ~Ende Juli zusammen mit dem Klassen-Forward.

Erwartung des Betreiber-Modells (ehrlich): ±2 grenzwertig (die 22-%-Prämie des
heutigen 29ers sah formal +EV aus), ±1 rot. Erwartung des Autors: beide Flanken
±1 klar rot, ±2 nur auf einer Flanke oder gar nicht grün.

---

## ERGEBNIS (18.07.2026, Läufe 09:15–10:05 UTC — weather_classb_eval.py)

Basis: **15.371 Stadt-Tage, 28/28 Städte** (700d, LOO-Bias) × **1.126 echte
Ladder-Fenster** seit 11.07. (Vortags-Snapshots). Rohdaten eingefroren in
classb_700d_offsets.json / classb_market_side.json.

| Klasse | P_emp 700d | Ø NO (n) | BE_net | ROI netto | t | Forward (n / hits / ROI) |
|---|---|---|---|---|---|---|
| −2 | 7,14 % ± 0,21 | 0,915 (130) | 8,0 % | **+0,88 %** | 3,9 | 79 / 5 / +0,9 % |
| −1 | 20,55 % ± 0,33 | 0,753 (134) | 23,1 % | **+3,35 %** | 7,9 | 82 / 18 / +1,0 % |
| +1 | 23,42 % ± 0,34 | 0,788 (128) | 19,7 % | **−4,60 %** | −10,8 | 79 / 18 / −3,9 % |
| +2 | 7,65 % ± 0,21 | 0,917 (121) | 7,7 % | **+0,11 %** | 0,5 | 73 / 5 / +1,6 % |

**Gates:** −2 ROT (G-B1), **−1 GRÜN (alle vier: ROI 3,35 > 3 %; t 7,9; Forward
+1,0 % widerspricht nicht; worst-drop Taipei +2,91 %, ohne Seoul +3,30 %)**,
+1 ROT (G-B1/2/4), +2 ROT (G-B1/2/4).

**Mechanismus (kohärent, beide Flanken dieselbe Ursache):** Die Fehlerverteilung
ist **warm-schief** — das Ist landet öfter im warmen Nachbarfenster (23,4 %) als im
kalten (20,6 %), der Markt bepreist aber näherungsweise symmetrisch (BE 19,7 % vs
23,1 %). Lay der kalten Nachbarklasse kassiert die Schiefe, Lay der warmen zahlt
sie. Konsistent mit der BA-Lehre 16.07. (Modelle präfrontal zu kalt) und ein
direktes Argument für Skew-fähige Verteilungen im Modell-Backlog (EPS).

**Offene Risiken (für das Review, nicht Teil der Gates):** (1) Preisseite ist nur
1 Juli-Woche (134 Fenster) — Klassenraten sind ganzjährig, Saisonalität der Schiefe
ungeprüft; (2) Snapshot-Preise 12:30 UTC ≠ garantierte Ausführbarkeit (Spread/
Liquidität); (3) Shenzhen im 700d-Teil METAR-basiert.

**Konsequenz laut Pre-Reg:** −1-Klasse qualifiziert für die Mini-Serie (2–3 $/Lay,
max. 2/Tag, getrennt, Review nach 30 Lays). **Live-Start ist Betreiber-Entscheid**
(Stand 18.07.: auf dem 11-Städte-Zwischenstand — damals ROT — wurde „Ende Juli
nachhalten" entschieden; das finale GRÜN wurde danach erreicht und vorgelegt).
Übrige Klassen: kein Live-Geld. Re-Auswertung mit gewachsenem Forward ~Ende Juli.
