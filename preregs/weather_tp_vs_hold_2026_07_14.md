# Befund: Der Autopilot-Take-Profit kostet Edge — er rettet nicht, er kappt

**Gerechnet:** 2026-07-14, `weather_tp_vs_hold.py`.
**Anlass:** Der Beijing-33-Lay hätte −5,67 $ verloren; nur der TP
(`autopilot.py --profit 0.10`, **kein** Stop-Loss) rettete ihn mit +0,51 $
(`weather_lay_postmortem_2026_07_14_beijing.md`). Damit stand die Frage im Raum,
ob die Regel systematisch hilft — oder ob Beijing ein Glücksfall war.

**Hypothese (vor dem Lauf festgelegt):** Der TP ist EV-negativ. Er kappt die
Rendite bei +10 %, wo die volle Lay-Rendite +20–27 % beträgt, und greift **nie**
bei Verlierern — deren Preis läuft nie 10 % zu unseren Gunsten. Gewinner werden
gedeckelt, Verlierer bleiben in voller Größe.

**Entscheidungsregel (vorab):** Verglichen wird der mittlere PnL je 1 $ Einsatz.
Schlägt der Hold-Arm den TP-Arm, kostet der TP Edge. Kein Parameter-Grid, keine
Nachjustierung der 10-%-Schwelle.

---

## Ergebnis: eindeutig

**Universum:** 138 NO-Lays aus `bb_WeatherLadders` (Zieltage 11.–13.07.), jeweils
mit ≥ 2 Preispunkten, bekanntem Settlement (Wunderground) und Einstieg < 0,909 —
nur dort *kann* ein +10-%-TP überhaupt feuern (bei NO 0,95 ist der maximale
Gewinn 5,3 %). Verkaufsgebühr 1,2 %, gemessen am echten Beijing-Verkauf.

| Strategie | Rendite je 1 $ Einsatz |
|---|---|
| **Halten bis Settlement** | **−6,26 %** |
| **TP +10 %** | **−12,87 %** |
| **Differenz** | **−6,60 pp** |

## Der Mechanismus — sauberer, als die Hypothese verlangte

- Der TP hätte bei **62 von 138** Lays gefeuert (45 %).
- **Alle 62 wären beim Halten Gewinner geworden.** Der TP kappte sie um im
  Schnitt **14,7 pp**.
- **Null** Rettungen. Kein einziger der 62 TP-Verkäufe betraf einen Lay, der
  am Ende verloren hätte.
- Von den 76 Lays, bei denen der TP **nicht** feuerte, verloren **47 (62 %)** —
  ungeschützt und in voller Größe.

Das ist kein Zufall, sondern strukturell: **Ein steigender NO-Preis IST das
Signal, dass der Lay gewinnt.** Der TP kann per Konstruktion nur auf Gewinnern
auslösen. Er ist ein Gewinner-Kappen-Mechanismus ohne jede Verlust-Bremse.

**Beijing war ein Glücksfall, kein Beleg.** In 138 Lays hat der TP nie einen
Verlierer abgefangen. Dort lief der Preis hoch (der Markt erwartete einen heißeren
Nachmittag) und kippte danach zurück — 1 von ~64.

## Robust über alle Preisklassen

| Einstieg (NO) | n | Verlustquote | Halten | TP | Differenz |
|---|---|---|---|---|---|
| 0,00–0,75 | 77 | 52 % | −16,4 % | −26,9 % | **−10,5 pp** |
| 0,75–0,83 | 28 | 21 % | −0,1 % | −2,1 % | **−2,0 pp** |
| 0,83–0,909 | 33 | 3 % | **+12,0 %** | +10,7 % | **−1,4 pp** |

In **jeder** Klasse ist der TP schlechter. Je billiger der Lay (= je höher die
Rendite), desto teurer die Kappung — es gibt schlicht mehr Oberseite zu verlieren.

## Nebenbefund, der direkt die Renditefrage betrifft

Die einzige **profitable** Preisklasse ist **NO 0,83–0,909** — also 10–20 %
Rendite, 3 % Verlustquote, **+12,0 %** beim Halten. Die Klasse mit den Renditen,
die zuletzt gesucht wurden (**NO 0,75–0,83 = 20–33 %**), ist bei **−0,1 %**, also
bestenfalls Nullsumme. Darunter wird es ruinös (−16,4 %).

**Die gewünschten Renditen liegen in der Klasse, die nicht zahlt.** Beide
Echtgeld-Trades vom 13.07. (Beijing 0,79, Madrid 0,80) saßen genau dort.

*Einschränkung:* Das ist das **ungefilterte** Leiter-Universum, nicht die vom
Screen durchgelassenen Buckets. Es ist ein starkes Indiz, kein Urteil über die
gefilterte Strategie — aber es passt exakt zum Beijing-Ausgang.

## Ehrliche Grenzen

1. **N = 138, drei Zieltage, eine Woche.** Klein.
2. **Die Snapshots sind ~täglich** (Timer 12:30 UTC), der echte Autopilot pollt
   alle 20–90 s. Die Simulation sieht also nur einen Bruchteil der Momente, in
   denen der TP hätte feuern können → sie **unterzählt** Auslösungen. Der reale
   TP-Schaden ist damit **größer** als die gemessenen −6,60 pp; die Zahl ist eine
   betragsmäßige **Untergrenze**.
3. Die absoluten Renditen beider Arme sind negativ, weil hier *jeder* billige
   Bucket gelayt wird. Nur der **Vergleich** der beiden Arme ist die Aussage.

## Konsequenz

**Für Wetter-Lays ist der TP falsch.** Unsere Edge realisiert sich beim
Settlement; ein vorzeitiger Verkauf verschenkt sie und schützt gegen nichts.

**Aber:** `autopilot.py` hat **keinen Markt-Filter** — `--profit 0.10` gilt für
*alle* Positionen im Wallet, auch für In-Play-Scalps (Wimbledon), wo ein
Momentum-Exit legitim sein kann. Ein pauschales Abschalten wäre also zu grob.

**Vorschlag (nicht umgesetzt — Live-Geld, gehört freigegeben):**
Kategorie-Filter im Autopiloten: Take-Profit für `category == "weather"`
überspringen, **Auto-Claim unangetastet lassen** (der holt die Gewinne ab).
Alternativ die Wetter-Märkte auf eine Skip-Liste setzen.
