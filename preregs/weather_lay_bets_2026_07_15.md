# Trade-Reg: 1 Echtgeld-NO-Lay, Zieltag 2026-07-15 (48h-Lead-Test)

**Gesetzt:** 2026-07-13 ~13:20 UTC, 6 $ Budget, Jupiter (jupiter_buy.py, Limit-Taker).
**Anlass:** Nutzer-These: „Leitern 2 Tage voraus sind zerklüftet, viele Buckets
mit zweistelliger Rendite — prüfen." Erster Echtgeld-Trade mit **Lead-48h-
Kalibrierung** (`weather_source_compare.py --lead 2`, 40d + 700d) statt der
Lead-24h-Kalibrierung des Screens.

## Methodik-Befund 48h (Kern der Prüfung)

- **ENS verliert 24h→48h fast nichts:** MAE 1,02→1,02 °C (40d, n=160,
  Madrid/Jeddah/Helsinki/Cape Town gepoolt); Madrid-ENS-Sigma sogar 0,55→0,53.
  Der Markt preist bei 48h aber deutlich mehr Unsicherheit → strukturelle
  Edge-Quelle real.
- **Einzelmodelle degradieren teils** (ICON 1,00→1,19; JMA auf Lead 2 für
  manche Städte kaputt, Bias −2,5) → Modell-Vetos häufiger.
- **3 der 4 formalen Screen-Passes des Tages waren nach Lead-2-Kalibrierung
  −EV** (Jeddah 34°, Helsinki 24°, Cape Town 15° — Renditen 1,4–2,0 % decken
  die echte 48h-Rest-P nicht). Lehre: Bei Zieltag >24h nie mit Screen-P
  (Lead-1) handeln, immer Lead-2 nachrechnen.

## Gesetzte Wette

| Markt | NO-Fill | Kontrakte | Kosten | Rendite | BE | Tx |
|---|---|---|---|---|---|---|
| Madrid High 35 °C, 15.07. (POLY-2902148) | 0,8047 | 7,18 | 5,78 $ | 24,3 % | 19,5 % | 25emFa2L… (finalized) |

Gewinn bei Verfehlen **+1,40 $**; Verlust −5,78 $. Limit 0,81 (Ask 0,79–0,80,
dünnes Buch füllte bis 0,81; erster Sende-Versuch 429 rate-limit, Retry ok).

**P(35er-Fenster) nach allen vier Kalibrier-Sichten** (Roh-ENS 36,5 → korr. ~37,2):
700d-Lead1 5,5 % | **700d-Lead2 5,0 %** (Bias −0,85, σ 1,16) | 40d-Lead1 0,1 %
| **40d-Lead2 0,3 %** (Bias −0,66, σ 0,53). Alle klar unter BE. Schärfstes
Einzelmodell ECMWF 9–15 % < BE. **Regen 0,0 mm in allen 5 Modellen.**

**Lage:** Madrid-Dip heute (Ist 38→34→~32), Rückerwärmung laut ALLEN Modellen
schon am 14.07. (36,2–37,3), 15.07. = Plateau-Tag 2 (35,7–37,3) — kein
Sprungtag. Modelle treffen den heutigen Ist-Tag exakt (~32).

**Transparent benannte Restrisiken:**
1. Übergangslage (Rückerwärmung) — Normal-Kalibrierung erfasst Timing-Fehler
   solcher Lagen strukturell schlecht (Plateau-Struktur mildert).
2. **Zentrumsstreit light:** Markt-Fav 36° (YES 0,43) vs Kalibrier-Zentrum 37,2°;
   das gelayte 35er ist Markt-Fav−1. Hat der Markt recht, ist P(35er) eher
   15–20 % → EV ≈ 0. Der Trade wettet „Kalibrierung schlägt Markt-Zentrum" —
   die YES-Paper-Serie hat diese Wette bisher verloren (0/5), allerdings auf
   der YES-Seite mit −EV-Preisen; hier trägt die Prämie das Risiko.

## Geprüft und NICHT gesetzt (15.07.)

- **Jeddah 36° @0,77 (30 %):** GFS-L2 29 % > BE 23 %; JMA-Chaos (roh 31 vs 39).
- **Helsinki 26° @0,78 (28 %):** ICON-L2 26 % > BE 22 % (ICON/ECMWF ~26).
- **London 27° @0,87 (15 %):** UKMO 25 % (sieht 27,8); dist 1,7 < 2.
- **Tokyo 30° @0,89 (12 %):** ICON 28 % am Fenster; ECMWF-Ausreißer 35,3.
- **Paris 34° @0,90 (11 %):** GFS 19 % (35,2 vs JMA 29,7 = 5,5° Spanne).
- **Madrid 34° @0,974:** einziger überlebender formaler Pass, Rendite 2,7 % zu dünn.
- **Lows:** London-Min 14/15/16 Rendite ≤2,1 % (Spread frisst alles), Paris-Min
  ECMWF/UKMO-Vetos.

## Ausgang: vom Autopiloten geschlossen (nachgetragen 14.07.)

**Nicht gehalten.** Der Autopilot-Take-Profit verkaufte schon am 13.07. 16:40 UTC
— gut 3 h nach dem Einstieg — zu **0,91** (Einstieg 0,8047), Netto-Erlös 6,47 $
→ **+0,64 $ realisiert** (+11 % netto). Die Position existiert nicht mehr; das
Settlement am 15./16.07. entscheidet nur noch über die Methodik-Frage, nicht über
Geld.

**Der Verkauf war mark-to-market richtig:** der 35er-NO steht am 14.07. bei 0,83,
also unter dem Verkaufspreis. Der Madrid-Forecast ist seit dem Einstieg um 0,8 °
abgekühlt (korr. ENS 37,2 → 36,4), und **ICON gibt dem 35er-Bucket jetzt 32 %**
(roh 34,7 — kalter Ausreißer; die anderen vier liegen bei 35,6–36,2). Beim
Einstieg lagen *alle* Modelle unter BE, das gilt nicht mehr. Der Trade wäre heute
kein Kandidat.

**Trotzdem am 16.07. das Ist nachtragen** — als zweiter Datenpunkt zur Frage, ob
die Kalibrierung oder das Markt-Zentrum recht hat, und als Gegenprobe zum
Beijing-Post-Mortem (`weather_lay_postmortem_2026_07_14_beijing.md`): dort war
das Ist-Hoch 1,7 ° unter dem korrigierten ENS, und *alle* Modelle waren zu warm.
Verlustfenster hier: Ist-Hoch 34,5–35,49 an LEMD, lokaler Kalendertag 15.07.

**Einordnung des TP:** Er kappt die Edge bei +10 %, wo das Modell +24,3 %
behauptet — bei richtigem Modell kostet er also Rendite. Hier hat er, wie beim
Beijing-Trade, gegen ein Modell versichert, das sich als schwächer erwies als
gedacht. Beide Male dieselbe Mechanik, beide Male vor dem Settlement raus.

## Settlement-Nachtrag (17.07., LEMD-Ist 15.07.)

**Ist-Hoch LEMD 15.07.: 36 °C** — WU-Settlement-Reihe (`LEMD:9:ES`) Max 36 °C
(Peak 16:30–19:30 lokal), METAR (IEM) 36,0 °C → identisch, Madrid bleibt
settlement-konsistent. Verlustfenster 34,5–35,49 verfehlt → **der Lay hätte
GEWONNEN** (+1,40 $). Der TP-Verkauf @0,91 realisierte +0,64 $ → **TP kostete
0,76 $** — konsistent mit der TP-vs-Hold-Rechnung (e1d8d5f2), die zur
Abschaltung führte (b08788ed).

**Methodik-Befund (der eigentliche Zweck des Nachtrags):**
1. **Zentrumsstreit: der Markt hat gewonnen.** Markt-Fav 36° = Ist exakt;
   unser Kalibrier-Zentrum sagte beim Einstieg 37,2 (korr. ENS), am 14.07.
   noch 36,4. Das Ist lag 1,2° UNTER dem korrigierten ENS des Einstiegs.
2. **Warm-Bias-Muster gegen Beijing gehalten:** Beijing 14.07. Ist 1,7° unter
   korr. ENS (alle Modelle zu warm), Madrid 15.07. 1,2° unter korr. ENS —
   gleiche Richtung. ABER Buenos Aires 16.07. brach das Muster: Ist 24 °C lag
   ~1,6–1,8° ÜBER den korrigierten Sichten (alle Modelle zu kalt, präfrontale
   Warmlage). Also kein einseitig systematischer Warm-Bias, sondern **großer
   µ-Fehler in Übergangslagen, Vorzeichen lagebedingt** — genau das, was
   Spannen-Veto/dist-Gate abfangen sollen (σ repariert es nicht).
3. Der Trade selbst (Lay des Fav−1-Buckets) war trotz falschen Zentrums
   richtig — die Prämie trug den Zentrumsfehler. Lehre unverändert: Lays
   brauchen Abstand, keine Zentrums-Präzision.
