# Befund: „Lay das ganze Brett + Autopilot-TP" ist keine 50 % — es ist eine Nullsumme, die der TP verschlechtert

**Gerechnet:** 2026-07-15, `weather_layall_tp.py`.

**Anlass:** Aus dem Peking-Chart (14.07., 33 °C → 100 %, alle anderen Buckets → 0 %)
kam die These: *„Jeder Bucket-Graph fällt im Verlauf um mindestens 10 %, dort hätte
der Autopilot-TP verkauft. Wenn man am Open ALLE Buckets layt, valutiert der
Autopilot sie nach und nach alle — das wären feine 50 %."* Der Peking-33-Lay
(14.07.) war genau der Fall, in dem der TP einen Verlierer intraday rettete
(`weather_lay_postmortem_2026_07_14_beijing.md`) — die Frage war, ob das die Regel
ist oder der Ausnahmefall.

## Hypothese (vor dem Lauf festgelegt, H0)

„Lay alle Buckets + TP" ist **kein Geldautomat**:

1. **Lay-all ist per No-Arbitrage eine Nullsumme.** Bei *n* sich ausschließenden
   Buckets, von denen genau einer gewinnt: Kosten = Σ NO-Ask ≈ *n* − ΣYES ≈ *n* − 1;
   Auszahlung = die *n* − 1 NO-Gewinner. Netto = ΣYES − 1 ≈ 0, abzüglich Vig also
   leicht **negativ**.
2. **Der +10-%-TP kann strukturell nur Gewinner kappen.** Der eine Verlierer (NO
   auf den YES-Gewinner) läuft auf 0, nicht 10 % nach oben → der TP rettet ihn
   praktisch nie. Er macht Lay-all damit **noch schlechter**.

Peking war eine intraday-Ausnahme (~1 von 64), kein Beleg für die Regel.

**Entscheidungsregel (vorab):** Rendite je 1 $ Einsatz, kapitalgewichtet
(= Wallet-Verhalten). Schlägt Halten den TP-Arm, ist der TP auch auf Brett-Ebene
falsch. Kein Parameter-Grid.

---

## Universum

Alle **53 gesettleten eq-Bretter** aus `bb_WeatherLadders` (Zieltage 11.–13.07.,
je ≥ 2 Preispunkte, Settlement via Wunderground/METAR). Ein Brett = `(city, var,
target_date)` mit seinen 9 eq-Buckets; genau ein Bucket gewinnt YES. Einstieg =
NO-Ask im ersten Snapshot je Bucket, 1 Kontrakt/Bucket. Verkaufsgebühr 1,2 %.
Neun dünne/Geister-Bretter (< 3 Buckets bzw. < 1 $ Kapital, u. a. zwei
1-Bucket-Bretter mit ~0 $ Notierung) werden fürs Board-Mittel verworfen; das
kapitalgewichtete Pooled-Ergebnis ist davon ohnehin unberührt.

## Ergebnis: H0 bestätigt, sauberer als verlangt

| | **A) alle 9 Buckets** (44 Bretter) | **B) nur Einstieg < 0,909** (29 Bretter) |
|---|---|---|
| **Halten bis Settlement** | **−0,92 %** | **−1,05 %** |
| **TP +10 %** | **−2,18 %** | **−4,08 %** |
| **Differenz (TP − Halten)** | **−1,27 pp** | **−3,03 pp** |

*(Pooled = kapitalgewichtet; das Board-Mittel liegt nach dem Geister-Filter mit
−0,93 % / −1,18 % praktisch gleichauf → das Ergebnis ist robust, keine Ausreißer-Story.)*

### Der Kern der These — empirisch tot

- **Verlierer-Rettungen durch den TP: 0 von 42** (Variante B: 0 von 25).
- Der TP feuerte über alle Bretter hinweg **60×** (Variante A) — **jedes einzelne
  Mal auf einem Gewinner**, den er bei +10 % deckelte.
- Summe der so **entgangenen** Gewinne: **−3,54 $**. Summe geretteter Verluste:
  **0,00 $**.

Das ist strukturell: Der NO auf den späteren YES-Gewinner **fällt monoton auf 0**
(sein YES läuft auf 1) — er springt nicht +10 % nach oben. Die Beobachtung „jeder
Graph fällt >10 %" gilt für die *Gewinner* (deren NO läuft ohnehin auf 1,0), aber
nicht für den einen Verlierer, auf den es ankommt.

### Beispiel-Bretter (Variante A)

```
London/max/12.07.   Halten −0,1 %   →  TP −4,6 %   (perfekte Nullsumme, vom TP ruiniert)
Seoul/max/12.07.    Halten +12,8 %  →  TP +12,3 %  (Gewinner-Brett — TP trotzdem schlechter)
Beijing/max/13.07.  Halten −1,7 %   →  TP −3,0 %   (keine Rettung)
```

Auf **keinem** Brett half der TP — auch nicht auf denen, die insgesamt gewannen.

## Einordnung der Rechnung „5 × 10 % = 50 %"

- Fünf parallele Lays mit je +10 % ergeben **+10 % auf eingesetztes Kapital**, nicht
  50 % — Prozente auf parallelen Positionen mitteln sich, sie addieren sich nicht.
- Real ist Lay-all **−0,9 %** (Vig), und der TP zieht auf **−2,2 bis −4,1 %**.
- Gleiches Vorzeichen wie der 138-Einzel-Lay-Befund vom 14.07.
  (`weather_tp_vs_hold_2026_07_14.md`, dort −6,60 pp). Kleinerer Betrag hier, weil
  „das ganze Brett" viele unbewegliche 0,99er-NOs enthält, die den Effekt
  verdünnen; Variante B (nur bewegliche Buckets) liegt mit −3,03 pp näher am
  Einzel-Befund.

## Ehrliche Grenzen

1. **N = 53 Bretter, drei Zieltage.** Klein.
2. **Snapshots ~täglich** (Timer 12:30 UTC), der echte Autopilot pollt in
   Sekunden → die Simulation unterzählt TP-Auslösungen. Wichtig für die Deutung:
   Die **Gewinner-Deckelung** wird trotzdem fast vollständig erfasst (Gewinner-NO
   driftet stetig auf 1,0 → im letzten Snapshot über +10 %), während eine
   **Verlierer-Rettung** einen flüchtigen Ausreißer braucht, den der Tageslogger
   verpasst. Die 0-Rettungen-Zahl ist also eine Sicht-Obergrenze — aber die
   Asymmetrie (Deckelung sicher & mehrfach/Brett, Rettung selten & max. 1×/Brett)
   macht das **Vorzeichen robust**. Selbst mit Peking-Rate (~1/64) holt eine
   Rettung die −3,54 $ garantierte Deckelung nicht auf.
3. Beide Arme sind absolut negativ, weil hier *jeder* Bucket gelayt wird (kein
   Screen). Aussage ist der **Vergleich** der Arme, nicht die absolute Zahl.

## Konsequenz

Bestätigt den Beschluss vom 14.07. (Commit b08788ed): **Wetter-Take-Profit im
Autopiloten aus, Auto-Claim an.** „Das ganze Brett layen" ist zusätzlich als
eigenständige Strategie widerlegt — es ist die Vig-Nullsumme ohne jeden Edge; der
Edge steckt allein im **selektiven** Layen überteuerter Buckets (Screens), nicht
im Brett-Kauf.
