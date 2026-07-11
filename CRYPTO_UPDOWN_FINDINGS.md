# Crypto-Up/Down 15m: Underdog-Sinus-These — FALSIFIZIERT

**Datum:** 2026-07-02 · **Skript:** `crypto_updown_backtest.py` · **Daten:** `bb_CryptoUpDown15m`
(Logger `crypto_updown_logger.py`, VPS-Service `boersenbot_crypto_updown`)

## These (vorregistriert 2026-06-30, VOR der Datensammlung)

Beobachtung: Bei Jupiters 15-Min-„Up or Down"-Märkten (btc/eth/sol/doge/bnb) pendelt
der Underdog-Kontrakt sichtbar im Band ~0,20–0,40 („Sinuskurve"). Frage: Ist das als
Mean-Reversion handelbar — kaufe niedrig (~0,20–0,33), verkaufe hoch (~0,35–0,45),
nach Spread + Fee positiv? Vorab notierte Einwände: (1) binäre Klippe am Range-Ende,
(2) Quote = bloße Abbildung des Spot-Random-Walks, (3) Spread + Fee fressen die Range.

## Daten

- 77.213 valide In-Range-Ticks, **1.043 gesettlete Ranges** (~207 je Asset),
  30.06. 13:00 – 02.07. 17:44 UTC, lückenlos, ~11 s Tick-Abstand (Ø 79 Ticks/Range).
- Fee-Modell aus dem Autopilot-Audit: `fee = 0.07 · min(p, 1−p)` je Kontrakt und
  Handelsseite; Settlement-Claim gebührenfrei. Kauf zum Ask, Verkauf zum Bid.
- **Caveat:** 2 Tage = EIN Marktregime (mild aufwärts, Up-Rate 53,2 %).

## Ergebnis 1 — Trade-Sim (die registrierte These): signifikant NEGATIV

Grid entry ∈ {0.20, 0.25, 0.30, 0.33} × target ∈ {0.35, 0.40, 0.45}, ein Trade je
Range, IS = erste Hälfte der 15-Min-Fenster, OOS = zweite; Signifikanz als Cluster-t
auf Fenster-Ebene (die 5 Assets lösen synchron auf, s. u.).

- **Alle 12 Settings negativ, in IS UND OOS** (IS-t −2,8 … −5,0; OOS-t −3,5 … −6,5;
  Bonferroni-Schwelle 2,87 → die Strategie ist überwiegend *signifikant verlustbringend*,
  nicht bloß edge-los).
- Bestes IS-Setting (0,33/0,45): Ø **−0,047 $/Kontrakt** IS, **−0,089** OOS
  (≈ −15 %…−30 % ROI pro Trade bei ~0,30 Einsatz).
- **Klippe dominiert:** 45–55 % aller Trades enden im Totalverlust (Einwand 1 bestätigt).
- **Alle 5 Assets einzeln negativ** (t −2,8 … −4,7) — kein „gilt nur bei manchen".

## Ergebnis 2 — Kalibrierung: Underdogs leicht ÜBERteuert (Longshot-Bias)

Realisierte Gewinnrate vs. Quoten-Mid (je Event×Bucket×Restzeit-Terzil dedupliziert):
fast durchgehend real < Mid, gesamt **won − mid = −1,5 Pkt** (Cluster-t −1,94, knapp
nicht signifikant), am stärksten in den letzten 5 Minuten (−3…−4,5 Pkt bei Mid
0,10–0,30). Der Markt ist also nicht nur effizient gegen die Sinus-These — Underdog-
Kaufen trägt zusätzlich einen kleinen systematischen Malus.

## Ergebnis 3 — Spiegeltest Favorit (EXPLORATIV, nicht vorregistriert): kein Edge

Favorit in den letzten 5 Min kaufen & bis Settlement halten, netto nach Fee:
IS je Ask-Band +1…+3 % ROI (t < 1, Rauschen), **OOS dreht jedes Band negativ**
(−0,2…−1,2 %). Der Mid-Bias ist kleiner als Ask-Spread + Fee → **nicht handelbar**.

## Ergebnis 4 — Mechanismus (warum es keinen Sinus-Edge geben kann)

- **Spot-Kopplung:** Tick-Korrelation ΔQuote ~ Δ(Spot − price_to_beat) ≈ **+0,52…+0,58**
  bei allen 5 Assets — auf 10-s-Ebene mit Mikrostruktur-Rauschen ist das sehr eng.
  Die „Sinuskurve" ist die Abbildung des Spot-Random-Walks auf [0,1] (Einwand 2 bestätigt).
- **Crypto-Beta statt Diversifikation:** In **59,8 %** der 15-Min-Fenster lösen alle
  5 Assets gleich auf (unabhängig wären 6,25 %); paarweise Übereinstimmung 80,6 %.
  Effektives N ≈ 204 Fenster, nicht 1.043 Events — „5 Assets" ist ~1 Wette ×5.
- **Choppiness:** 27 % der Ranges ohne einzigen Führungswechsel, Median 2 Wechsel.
  Die Sinus-Optik existiert (46 % mit ≥3 Wechseln), ist aber ex ante wertlos, weil
  der Preis dabei kalibriert bleibt.

## Ergebnis 5 — Streak-/Persistenz-Addendum (02.07. abends, `crypto_updown_streaks.py`)

Nutzerfrage: „Wie oft in Folge kam dieselbe Richtung?" — plus die Edge-Frage dahinter
(Serien-Abhängigkeit über Ranges hinweg, und preist der Markt sie ein?). Explorativ,
1.060 Events / 1.050 lückenlose 15-Min-Nachbar-Paare.

- **Streak-Verteilung (gepoolt ×5 Assets, ~80 % synchron):** 237× L1, 122× L2, 61× L3,
  41× L4, 15× L5, 10× L6, 4× L7, 1× L8, 2× L9, 1× L10, 3× L11. **Längste Serie: 11**
  (btc/eth/sol — dieselbe synchrone Episode; Shuffle-Erwartung für das Maximum ~8).
  Je Asset einzeln kein Permutations-p < 0.05 (min. 0.061) → **im Zufallsrahmen**.
- **Scheinbare Anomalie:** Nach Up-Ranges startete die Folge-Range mit Up-Mid **0.475**
  (Markt erwartet Reversal), real kam Up in **56.9 %** (+9.4 Pkt). Reversal-Regel
  verlor entsprechend signifikant (netto t=−4.5), Momentum-Regel nominal +2…+8 % ROI
  (Cluster-t ≤ 1.1, Hälften instabil).
- **Auflösung durch den 60-Tage-Binance-Anker (5.760 15m-Kerzen/Asset):** langfristige
  Fortsetzungsrate **0.474–0.490** (z bis −4.0: leichte ANTI-Persistenz), AC(1) −0.01…−0.02.
  Die Jupiter-Start-Quoten preisen die Fortsetzungs-Seite mit Mid **0.475–0.491** —
  **der Markt ist auf der Serien-Ebene fast exakt auf die 60-Tage-Realität kalibriert.**
  Nur unser 2-Tage-Fenster war eine Momentum-Welle (Fortsetzung 0.53–0.57, daher die
  11er-Streaks und die Scheinkante). Klassische Nichtstationaritäts-Falle.
- **Konsequenz:** Kein handelbarer Streak-Edge. Der langfristigen Reversal-Tendenz zu
  folgen ist eingepreist (Brutto-Edge ~0) und stirbt an Fee (~3.5 Pkt); auf Momentum-
  Fortsetzung zu wetten widerspricht dem 60-Tage-Anker. Einzige saubere Resttür:
  Momentum-Regel („nach Up → Up kaufen") als Pre-Reg einfrieren und im Forward-Fenster
  03.–10.07. konfirmieren (Logger läuft bis dahin) — Erwartung ehrlicherweise negativ.

## Ergebnis 6 — Momentum-Forward-Konfirmation (Pre-Reg 03.07., ausgewertet 11.07.): RED

Die „saubere Resttür" aus Ergebnis 5 ist zu. Eingefrorenes Fenster 03.–10.07.
(`preregs/crypto_momentum_forward_2026_07_03.md`, Commit 50890386),
`python eval_crypto_momentum.py`: **N=2.403 Trades / 642 Fenster-Cluster (G-N PASS),
win 0,472, netto −161,71 $/K, ROI −12,5 %, Cluster-t −4,80 → G-Primär FAIL → RED.**
Selbst brutto (Fee 0) negativ (−7,1 %, t −2,58) — die Explorations-Welle war im
Forward vollständig verschwunden: Fortsetzungsrate 0,472 [Wilson 0,452–0,492],
deckungsgleich mit dem 60-Tage-Binance-Anker (0,474–0,490). Nur der 08.07. als
einziger Tag positiv (+4,81 $/K, t +0,39); alle Assets negativ außer btc (+2,2 %,
t +0,24, N=105 — Discovery-Lag-Restgröße, Rauschen). Die Negativ-Erwartung des
Registrierenden (−5…−10 %) traf ein. Keine Post-hoc-Schnitte mehr auf diesen Daten.

## Fazit & Konsequenz

**These falsifiziert — keine Paper-/Live-Stufe.** Alle drei vorab notierten Einwände
bestätigt; zusätzlich kleiner Longshot-Bias, der die Underdog-Seite weiter verschlechtert
und auf der Favoriten-Seite den Spread nicht überlebt. Auch die vorregistrierte
Momentum-Restthese ist forward-falsifiziert (Ergebnis 6). VPS-Logger
`boersenbot_crypto_updown` **abgeschaltet am 11.07.2026** (Beschluss 02.07., Daten
bleiben in `bb_CryptoUpDown15m` reproduzierbar). Nicht weiter verfolgt (mangels
handelbarer Basis): Vorab-Klassifikation choppy vs. früh entschieden.

Reproduktion: `python crypto_updown_backtest.py` (Voll-Report),
`--fee-rate 0.0` für Brutto-Sicht. Illustration Synchronität/Verlauf: `crypto_verlauf.png`
(First-Peek 30.06., 6 h Daten).
