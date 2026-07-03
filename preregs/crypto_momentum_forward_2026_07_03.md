# Pre-Reg: Crypto-15m-Momentum Forward-Konfirmation — 2026-07-03

**Status:** vorregistriert VOR dem Forward-Fenster (Freeze = Commit dieses Dokuments
+ `eval_crypto_momentum.py`, 2026-07-03). Motiviert durch das Streak-Addendum
(`crypto_updown_streaks.py`, Commit `86233af3`, `CRYPTO_UPDOWN_FINDINGS.md` Ergebnis 5).

## Ausgangslage (explorativer Anlass — KEIN bestandenes Gate)

Auf den 2 Beobachtungstagen 30.06.–02.07. (1.050 lückenlose 15-Min-Nachbar-Paare):

- Nach Up-Ranges startete die Folge-Range mit Up-Mid **0,475** (Markt erwartet
  Reversal), real kam Up in **56,9 %** (+9,4 Pkt). Momentum-Regel („kaufe die
  Vorgänger-Richtung am Range-Start") nominal **+2…+8 % ROI netto** auf den
  Schnitten (k≥2, k≥3, nach Up), Cluster-t ≤ 1,1.
- **Update beim Freeze (03.07., Regressionslauf, nur Daten < 03.07.):** Mit den
  ~6 h Zusatzdaten vom 02.07.-Abend (nach der gestrigen Analyse entstanden) ist
  die k≥1-Gesamtkante **netto bereits negativ** (−1,3 %, t −0,31; brutto +4,7 %,
  t +1,01) — die Momentum-Welle ebbte am 02.07. ab. Stützt die Negativ-Erwartung.
- **Selektionsverdacht (ehrlich):** Die Regel ist das direkte Komplement der auf
  denselben Daten falsifizierten Reversal-Regel (netto t=−4,5) — sie wurde gewählt,
  *weil* sie im Sample funktionierte. Genau deshalb Forward-Konfirmation statt Deploy.
- **Gegenanker:** 60-Tage-Binance (5.760 15m-Kerzen/Asset) zeigt Fortsetzungsrate
  **0,474–0,490** (leichte ANTI-Persistenz, z bis −4,0), und Jupiters Start-Quoten
  preisen die Fortsetzungs-Seite mit Mid 0,475–0,491 — der Markt ist auf der
  Serien-Ebene auf die 60-Tage-Realität kalibriert. Die 2 Tage waren mutmaßlich
  eine Momentum-Welle (Nichtstationarität).

## Hypothese

**H1 (zu konfirmieren):** Kauf der Vorgänger-Richtung am Range-Start, gehalten bis
Settlement, ist im Forward-Fenster netto (nach Spread + Fee) profitabel.

**Erwartung des Registrierenden: H1 ist FALSCH.** Bei kalibriertem Markt
(Fortsetzung real ~0,48, Ask ~0,50 + Fee ~3,5 Pkt) erwarteter Netto-ROI ≈ **−5…−10 %**.
Zweck der Registrierung: Disziplin-Übung („der 2-Tage-Kante nicht glauben, dem
60-Tage-Anker schon") und saubere Beerdigung des Up/Down-Kapitels — bzw. im
Überraschungsfall ein ehrlich erworbener Kandidat.

## Daten

`bb_CryptoUpDown15m` (Logger `crypto_updown_logger.py`, VPS-Service
`boersenbot_crypto_updown`, läuft bis 10.07.), Assets btc/eth/sol/doge/bnb.
Lade-Filter wie im Backtest (`load_ticks`): `settled=1`, `result ∈ {Up, Down}`,
`secs_to_close ∈ [0, 900]`, alle 4 Preisfelder in (0,0005, 0,9995).

**Bekannte Erhebungs-Limitation (identisch in Exploration und Forward):** Die
Event-Discovery (Pagination Tiefe 8) findet frische btc-Ranges im Mittel erst
~3 min nach Range-Start → nur ~14 % der btc-Ranges liefern einen gültigen Entry
(eth ~75 %, bnb/doge/sol ~94 %; Trichter-Check 03.07., ergebnisblind). Der
gepoolte Befund trägt damit v. a. eth/sol/doge/bnb. Bewusst NICHT gefixt: gleiche
Erhebungscharakteristik in Explorations- und Konfirmationsdaten, kein
Deploy-Risiko ins laufende Fenster.

## Definitionen (exakt = `crypto_updown_streaks.py` Block D, momentum k≥1)

- Events je Asset chronologisch; **Segmente** brechen bei Logging-Lücke
  (Nachbar-Abstand ≠ 900 s). Signal nur innerhalb eines Segments.
- **Signal:** `prev_result` = Ergebnis der unmittelbar vorhergehenden lückenlosen
  Range desselben Assets. Der Vorgänger darf vor dem Fensterbeginn liegen —
  der Trade nicht.
- **Entry:** erster Tick des Events mit `secs_to_close ≤ 895`, gültig nur wenn
  `secs_to_close ≥ 870` (echter Range-Anfang, sonst kein Trade). Kauf der Seite
  `prev_result` zum **Ask**; Filter `0,02 < ask < 0,98`.
- **Kosten:** `cost = ask + 0,07·min(ask, 1−ask)` (Audit-Fee-Modell autopilot.py).
- **Exit:** Settlement 1/0, Claim gebührenfrei. `pnl = won − cost`,
  1 Kontrakt je Event, Einheit $/Kontrakt. ROI = Σpnl / Σcost.
- **Signifikanz:** Cluster-t auf 15-Min-Fenster-Summen (`range_start_utc`) —
  die 5 Assets lösen zu ~80 % synchron auf, effektives N = Fenster.

## Forward-Fenster (eingefroren)

`range_start_utc ∈ [2026-07-03 00:00, 2026-07-10 00:00) UTC` — 7 volle UTC-Tage,
erwartet ~2.500–3.000 Trades / ~600 Fenster-Cluster bei stabilem Logger.

- Fenster + Regel wurden bereits am 02.07. abends benannt (FINDINGS Ergebnis 5,
  Commit `86233af3` — vor Existenz der Daten); dieses Dokument friert die Details ein.
  Daten vom 03.07. bis zum Freeze-Commit wurden nicht angesehen. Logger-Alive-Check
  ergebnisblind 03.07. ~18:10 UTC (nur COUNT/MIN/MAX(ts_utc), keine Ergebnisse/
  Quoten): alle 5 Assets aktiv, je 73 Events seit 00:00 UTC.
- Festes Fensterende → kein optional stopping. Zwischenstands-Peek per Skript ist
  zulässig (ändert keine Entscheidung), der Gate-Entscheid fällt ausschließlich
  auf dem vollen Fenster nach dem 10.07.

## Gates

- **G-N (Power):** ≥ 300 verwertbare Trades UND ≥ 60 Fenster-Cluster.
  Sonst **UNDERPOWERED** (kein PASS/FAIL-Urteil, nur Bericht).
- **G-Primär (GREEN):** `mean(pnl) > 0` UND **Cluster-t ≥ +2,0** netto (Fee 0,07),
  alle 5 Assets gepoolt, k≥1. Einzelhypothese → keine Bonferroni-Korrektur.
- Alles andere → **RED**.

## Entscheidungsregel

- **RED:** Kapitel Crypto-Up/Down endgültig geschlossen. Logger am 10.07. abschalten
  (bestehender Beschluss vom 02.07.). Keine weiteren Post-hoc-Schnitte auf diesen
  Daten als „neue These" ohne frische Pre-Reg + frische Daten.
- **GREEN:** KEIN Livegang. Nächste Stufe wäre eine Paper-Trade-Pre-Reg
  (echte Orderbuch-Fills statt Tick-Sim) + Logger-Verlängerung.
- **UNDERPOWERED:** berichten; Logger trotzdem abschalten (Beschluss steht).

## Sekundäranalysen (deskriptiv, KEINE Entscheidungsgrundlage)

Brutto-Sicht (Fee 0), per Asset, per Vorgänger-Richtung (nach Up / nach Down),
bedingt auf `prev_streak ≥ 2 / ≥ 3`, Kalibrierung (won − Start-Mid der gekauften
Seite, Cluster-t), Fortsetzungsrate mit Wilson-CI, PnL je UTC-Tag.

## Auswertung

`python eval_crypto_momentum.py` (Default = eingefrorenes Fenster; Gate-Block im
Output). **Skript-Regressionstest** (gelaufen 03.07. vor dem Freeze):
`python eval_crypto_momentum.py --start 2026-06-30 --end 2026-07-03` →
N=873 Trades / 222 Fenster-Cluster, win 0,526, netto −6,23 $/K (ROI −1,3 %,
t −0,31), brutto +4,7 % (t +1,01); Schnitte: k≥2 +2,9 %, k≥3 +5,9 %,
nach Up→Up +3,3 %, nach Down→Down −6,1 %. Konsistent mit der Streak-Analyse
(FINDINGS Ergebnis 5) plus den 6 h Zusatzdaten vom 02.07.-Abend.

---

## ERGEBNIS (auszufüllen nach 2026-07-10)

*offen*
