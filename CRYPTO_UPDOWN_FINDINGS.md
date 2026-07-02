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

## Fazit & Konsequenz

**These falsifiziert — keine Paper-/Live-Stufe.** Alle drei vorab notierten Einwände
bestätigt; zusätzlich kleiner Longshot-Bias, der die Underdog-Seite weiter verschlechtert
und auf der Favoriten-Seite den Spread nicht überlebt. Empfehlung: VPS-Logger
`boersenbot_crypto_updown` kann abgeschaltet werden (Entscheidung Betreiber); ohne
neue, konkrete These ist Weitersammeln Datenhortung. Nicht weiter verfolgt (mangels
handelbarer Basis): Vorab-Klassifikation choppy vs. früh entschieden.

Reproduktion: `python crypto_updown_backtest.py` (Voll-Report),
`--fee-rate 0.0` für Brutto-Sicht. Illustration Synchronität/Verlauf: `crypto_verlauf.png`
(First-Peek 30.06., 6 h Daten).
