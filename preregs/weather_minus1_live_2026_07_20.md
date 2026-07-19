# Pre-Reg: −1-Klasse LIVE — autonomer Auto-Lay (VPS-Timer), 20.–26.07.2026

**Registriert:** 2026-07-19 ~15:45 UTC, VOR dem ersten Kauf-Lauf (20.07. 12:45 UTC).
**Anlass:** Nutzer-Auftrag 19.07. („du sollst da allein setzen … die Woche
durchlaufen lassen"). Infra- und Auswahl-Entscheid des Nutzers am 19.07.:
VPS-Timer + „die 3 konservativsten".

## Basis (vermessen)

Klasse-B-Auswertung 18.07. (`preregs/weather_classb_lay_2026_07_18.md`,
`weather_classb_eval.py`): NO-Lay auf den Bucket **k = half_up(µ_ens) − 1**
(Highest-°C-Bretter) über 15.371 Stadt-Tage × 1.126 echte Ladder-Fenster
(12:30-UTC-Vortags-Snapshots): **+3,35 % netto, t 7,9** — einzige Klasse, die
alle Gates besteht. Mechanismus Warm-Schiefe: P(Ist = µ+1) 23,4 % >
P(Ist = µ−1) 20,6 %, Markt preist symmetrisch.

## Live-Regel (exakt, implementiert in `weather_minus1_autobuy.py`)

1. Täglich **12:45 UTC** (systemd-Timer `boersenbot_weather_minus1.timer`,
   Persistent=false), direkt nach dem 12:30-Ladder-Snapshot → identische
   Datenbasis und µ-Definition wie die Messung.
2. Kandidaten: heutiger `bb_WeatherLadders`-Snapshot, `var='max'`,
   `kind='eq'`, `offset_fav=−1`, `status='open'`, `target_date=morgen` (Lead 1).
   Kein Snapshot von heute ≥12:00 UTC → harter Abbruch (kein Alt-Daten-Kauf).
3. Live-Preis-Recheck je Markt; handelbar wenn open und **buyNo ≤ 0,97**
   (Tick = 1 Cent; Mindestrendite ~3 %). Märkte mit bestehender Position
   werden übersprungen (Idempotenz, keine Kollision mit manuellen Wetten).
4. **Auswahl: die 3 mit dem höchsten Live-NO-Preis** („konservativste 3",
   Nutzer-Entscheid). DEKLARIERTE ABWEICHUNG vom klassenreinen Test: die
   +3,35 % wurden ohne Selektion gemessen; dieser Live-Test prüft also
   Klasse × Konservativ-Selektion, nicht die reine Klasse.
5. Kauf **5 $ NO** je Markt (Jupiter-Minimum; Messbasis-Vorschlag 2–3 $ ist
   nicht handelbar), Limit = Ask + 1 Cent (Cap 0,97), max. 2 Sendeversuche,
   **kein Nachrücker** bei Sendefehler. Fill-Verifikation per Positions-API.
6. **Halten bis Settlement, kein TP** (TP-Lehre 14.07.), Claims macht der
   bestehende VPS-Autopilot (~03–04 UTC).
7. Log: `preregs/weather_minus1_live_log.csv` — **führende Datei auf dem VPS**
   (lokal nie echt laufen lassen); eine Zeile je Kandidat mit decision
   (bought / skip_price / skip_position / skip_cap / skip_closed / fail_send /
   dry_run), Kontrakten, avg_price, Signature.
8. Guards: Zeitfenster 12:30–14:30 UTC (kein Nachhol-Lauf = kein anderes
   Lead), Tages-Idempotenz übers Log (kein Doppellauf).

## Laufzeit & Review

- Kauf-Läufe **20.07.–26.07.2026** (7 Läufe, Zieltage 21.–27.07.), danach
  **Review mit dem Nutzer ~27.07.**: Netto-PnL, Trefferquote(Ist=k) vs
  implizite P (1−avg_price), Fill-Slippage vs Snapshot-Preis, je Klasse-Bucket.
- Erwartetes N ≈ 21 → **nur Indikation, kein Gate-Anspruch**; Fortsetzung /
  Abbruch / Regeländerung = Nutzer-Entscheid. Während der Woche KEINE
  Regeländerung, kein manuelles Eingreifen in die Auto-Positionen.
- Exposure: max 15 $/Tag neu, rollierend ≤ ~30 $ gebunden; Wallet-Deckung
  19.07. geprüft (70,9 $ Jupiter-USD + 0,14 SOL).

## Risiken (bekannt, akzeptiert)

- Konservativ-Selektion unvermessen (siehe oben, bewusster Nutzer-Entscheid).
- Preisseite der Messung: nur eine Juli-Woche; Saisonalität der Warm-Schiefe
  ungeprüft.
- Snapshot ≠ Ausführbarkeit: Live-Ask kann schlechter sein (wird geloggt).
- Jupiter-API-Transienten (500er killten 19.07. den Ladder-Lauf → 5xx-Retry
  in Logger + Screen nachgerüstet; Autobuy hat eigene Retries + Guards).
