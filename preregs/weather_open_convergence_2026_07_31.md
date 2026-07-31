# Pre-Registrierung: Eröffnungs-Fehlpreis neuer Wetter-Bretter — 2026-07-31

**Status:** ECHTE Vorregistrierung. Geschrieben und committet, **bevor** eine
einzige Ertrags- oder Preiskennzahl gerechnet wurde. Der Datensammler
(`collect_open.py`, reines Herunterladen ohne Auswertung) lief parallel.

Auswertung folgt in `weather_open_convergence_eval.py`.

---

## Anlass

Nutzer-Idee 31.07.2026: neue Bretter im Moment ihrer Eröffnung erkennen und als
Erster zum Fehlpreis kaufen. Vier rückblickend gegriffene Anekdoten legten nahe,
dass die ersten Preise grob daneben liegen können (Mexico City 25 °C stand in
Minute 12 bei 0,68 YES und settelte auf nein — NO war dort für 0,32 zu haben).

**Diese vier Zeilen sind gesehen worden.** Sie sind Anlass, nicht Evidenz; das
Repo hat genug Beispiele dafür, dass solche Rückblick-Muster an G2 sterben. Die
Sondierung vor dieser Pre-Reg hat ausschließlich **Mengen** angesehen
(Feldnamen, Bretterzahl, Handelsdichte je Zeitfenster) — keine Preise, keine
Erträge, keine Trefferquoten.

## Zwei Befunde der Sondierung, die das Design ändern

1. **„Erster Trade" ≠ „Eröffnung".** Die Bretter schalten fahrplanmäßig ~2 Tage
   vorher frei (`acceptingOrdersTimestamp`), der erste Trade eines Buckets folgt
   aber mit 27 min bis **2812 min** Verzug — Randbuckets handeln teils erst nach
   Beginn des Zieltags. Eine Zeitachse „seit erstem Trade" würde Buckets
   unterschiedlicher Reife vermengen. **Alle Zeiten sind daher absolut, in
   Minuten seit `acceptingOrdersTimestamp`.**
2. **Nur ein Teil des Bretts ist früh sichtbar.** Handelsdichte über eine
   Stichprobe von 12 Brettern / 132 Buckets: 47 % handeln binnen 15 min, 60 %
   binnen 30 min, 72 % binnen 60 min, 100 % binnen 24 h. Ein Signal, das das
   volle Brett braucht, ist in Minute 0 nicht berechenbar.

## Was das Design ausschließt

Die naheliegende Fassung „ich kenne das Wetter besser als der Eröffnungspreis"
ist **nicht** testbar und wird hier nicht getestet. Bretter öffnen bei Lead ≈ 2,
unser Modell ist auf Lead 1 kalibriert (Madrid-Lehre 13.07.: 3 von 4 formalen
Screen-Passes waren nach Lead 2 −EV), und der Markt trifft ohnehin besser als
unsere Punktprognose (28.07., p < 0,01). **Das Signal darf deshalb keinerlei
Wetterinformation enthalten** — es wird ausschließlich aus den Preisen des
Bretts selbst gebildet.

---

## Hypothese

**H1 (primär):** Innerhalb der ersten Minuten nach Eröffnung ist die
Preisverteilung über die Buckets eines Bretts noch nicht in sich konsistent.
Einzelne Buckets weichen von der Form ab, die die übrigen Buckets desselben
Bretts implizieren, und diese Abweichung ist ein Fehlpreis: sie prognostiziert
das Settlement mit umgekehrtem Vorzeichen und ist nach Kosten handelbar.

### Signal (in Minute T verfügbar, ohne jede Wetterinformation)

Zum Bewertungszeitpunkt T hat jeder Bucket b des Bretts einen letzten
YES-normalisierten Trade-Preis p_b (Buckets ohne Trade bis T entfallen).

1. **Normieren:** q_b = p_b / Σ p — entfernt die Niveau-Komponente, damit H1
   und H3 sich nicht überlappen.
2. **Formreferenz:** Die eingeschwungene Preisverteilung über Temperatur-Buckets
   ist unimodal und näherungsweise gaußförmig. Für Bucket b wird eine
   Normalverteilung **an alle Buckets außer b** angepasst (Leave-one-out,
   kleinste Quadrate über μ und σ) und an b ausgewertet: ĝ_b.
   *Leave-one-out ist zwingend* — ein gemeinsamer Fit würde vom mutmaßlichen
   Ausreißer selbst mitgezogen, und ein naiver Nachbarschaftsmittelwert würde den
   Modus der Glocke systematisch als „zu teuer" markieren.
3. **Residuum:** r_b = q_b − ĝ_b.
   - r_b > +θ → Bucket zu teuer → **NO kaufen**
   - r_b < −θ → Bucket zu billig → **YES kaufen**

**Vorregistrierte Parameter:** T = **30 min**, θ = **0,05**, Mindestbelegung
**≥ 6 gehandelte Buckets** im Brett. T ∈ {15, 60} und θ ∈ {0,03; 0,10} sind
ausschließlich Sensitivität, nicht Ersatz.

### Ausführung und Kosten (fest, vor der Messung)

- **Einstiegspreis** aus dem Tape rekonstruiert, nicht als Mid geschätzt: für
  einen YES-Kauf der zuletzt beobachtete Preis auf der **Ask-Seite** (Taker-Kauf
  YES oder Taker-Verkauf NO), für einen NO-Kauf entsprechend die Gegenseite.
  Liegt für die benötigte Seite bis T kein Trade vor, entfällt das Signal.
- **Gebühr:** 5 % von min(p, 1−p), also `feeSchedule.rate` **ohne** den
  `rebateRate`-Nachlass — konservativ; empirisch gemessen wurden 3,6–4,5 %.
- **Exit E1 (primär):** halten bis Settlement, Auszahlung 1 oder 0.
- **Exit E2 (sekundär):** Verkauf in Minute 60 zum dann beobachteten
  Gegenseiten-Preis, erneut abzüglich Gebühr. E2 misst die reine
  Mikrostruktur-Konvergenz, E1 den Geldwert.
- Gleichgewichtet, 1 Einheit Einsatz je Signal; ROI = Gewinn / Einsatz.

## Daten

Polymarkets öffentliche API (Gamma + Data), kostenlos und historisch — kein
Risiko, kein Vorlauf, s. `POLYMARKET_DATA_API.md`. 32 Stadt-Serien
`{stadt}-daily-weather`, Zeitraum **April–Juli 2026** (ab April durchgehend
11 Buckets je Brett; davor 7, also andere Marktstruktur). Nur gesettelte Bretter.
Erwartete Größenordnung: ~2.500 Bretter, ~27.000 Buckets, davon rund 6.500 mit
Trade binnen 30 min.

**IS/OOS-Split ist ein Zeit-Split:** IS = **April–Juni 2026**,
OOS = **Juli 2026**. Kein Städte-Split — Städte-Splits lassen einen
gemeinsamen Zeit-Regimewechsel durchrutschen.

---

## Gates

| Gate | Bedingung |
|---|---|
| **G1** In-Sample (Apr–Jun) | Primärsignal, Exit E1: mittlerer Netto-ROI je Trade > 0 mit **t > 2,0** |
| **G2** Out-of-Sample (Jul) | Identische Parameter (T = 30, θ = 0,05), gleiches Vorzeichen, **t > 1,5** |
| **G3** Netto nach Kosten | G2 gilt bereits nach Ask/Bid-Kreuzen und 5 % Gebühr; zusätzlich **ROI > 0** im OOS |
| **G4** Frequenz | Im OOS im Mittel **≥ 1 Signal pro Tag** über alle Städte — darunter ist der Betrieb den Aufwand nicht wert |
| **G5** Robustheit | Median-ROI über die Städte > 0 **und** nach Streichen der besten Stadt bleibt G3 bestehen **und** kein einzelner Tag trägt > 30 % des Gesamtergebnisses |

**Bonferroni:** G1/G2 sind je *ein* Test auf den vorregistrierten Parametern.
Die Sensitivitätsläufe (3 × T mal 3 × θ = 9 Zellen) werden berichtet, aber
**kein** Gate darf über sie erfüllt werden. Reißt G1 auf den Primärparametern
und hält nur eine Sensitivitätszelle, gilt die Hypothese als **falsifiziert**.

## Vorregistrierte Sekundärhypothesen (Bonferroni, t > 2,5)

- **H2 Longshot-Bias:** Buckets, die in Minute T unter 0,10 notieren, setteln
  seltener als ihr Preis impliziert. Handel: NO auf alle Buckets < 0,10.
- **H3 Niveau-Arbitrage:** Σ_b p_b weicht in Minute T von 1,00 ab. Gemessen wird
  die Verteilung von Σ und der Anteil der Bretter, bei denen Σ nach Gebühren
  eine risikolose Position trägt (Σ_Ask < 1 − Gebühr bzw. Σ_Bid > 1 + Gebühr).
  H3 ist der einzige Zweig, der bei Erfolg *risikolos* wäre, und wird deshalb
  auch dann berichtet, wenn H1 fällt.

## Was diese Studie nicht beantworten kann

- **Buchtiefe.** Der Tape zeigt, was gehandelt wurde, nicht, was ausführbar
  gewesen wäre. Gerade bei Eröffnung sind die Bücher dünn. Als Näherung wird das
  Notional je Signal-Fenster mitberichtet; ein Signal, dessen Fenster nur wenige
  Dollar Umsatz trägt, ist praktisch nicht handelbar. Historische Buch-Snapshots
  existieren nicht.
- **Selektion über die Zeit.** Gemessen werden nur Buckets, die bis T gehandelt
  haben. Ob früh handelnde Buckets anders sind als spät handelnde, bleibt offen;
  die Handelsdichte wird deshalb je Zelle mitberichtet.
- **Latenz.** Ob der eigene Ausführungspfad (Jupiter, gemessen 4–45 s von Order
  bis Fill) schnell genug ist, ist eine getrennte Frage. Bei einem 30-Minuten-
  Fenster ist sie unkritisch — im Gegensatz zum begrabenen Latenz-Rennen.

## Abbruchregel

Reißt **G1**, wird die Hypothese als falsifiziert dokumentiert und **nicht**
durch Umparametrisierung gerettet. H2 und H3 werden unabhängig davon zu Ende
gerechnet, weil sie eigenständige Fragen sind.
