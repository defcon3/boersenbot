# Pre-Registrierung: Der Markt-Favorit als Anker — 2026-08-02

**Status:** Vorregistrierung. Fenster R (12.07.–01.08.) trägt **kein Gate** — die
Trefferquoten sind aus der Ursachen-Messung desselben Tages bekannt. Der Beleg
ist **Fenster F ab Zieltag 03.08.2026**.

Auswertung folgt in `weather_markt_anker_eval.py`.

---

## Anlass

Die Ursachen-Messung vom 02.08. hat gezeigt, dass die −1-Klasse fair bepreist ist
und der Bot nur aus seiner Auswahl verdient. Ein einziger Befund darin zeigte in
eine andere Richtung — H6, ausdrücklich ohne Gate:

| Anker | Anker trifft | Bucket **darunter** trifft |
|---|---|---|
| eigener Ensemble-Favorit `k0` | 32,6 % | **23,7 %** |
| **Markt-Favorit** `market_fav_k` | **46,8 %** | **19,7 %** |

Der Break-even des eigenen −1-Buckets liegt bei 22,6 %. **23,7 % liegen darüber,
19,7 % darunter.** Das ist der einzige Befund des Tages, der auf eine echte
Verbesserung zeigt statt auf eine Grenze.

Er reproduziert außerdem den Favoriten-Backtest vom 28.07. auf einer anderen
Stichprobe fast punktgenau (dort 33,2 % gegen 47,4 %, bei Uneinigkeit 21,8 %
gegen 46,8 %, McNemar p < 0,01).

## Was auf dem Spiel steht — die Doktrinfrage, offen benannt

Die Lay-Doktrin lautet bisher: **die eigene Prognose ist die einzige Referenz**
([[weather-lay-bucket-preference]]). Der Betreiber hat am 02.08. dazu gesagt:
*„ich wollte immer ein eigenes modell haben, aber wenn es der markt so hergibt,
meinetwegen."*

**Was bliebe vom eigenen Modell, wenn diese Pre-Reg besteht:** Das Spannen-Veto,
der P_pess-Filter, die Doppel-Kalibrierung und die Städteauswahl — alles unser.
**Was ginge:** die Wahl des Buckets. Sie käme dann aus dem Preis statt aus `mu_ens`.

Das ist keine Kleinigkeit, und es ist auch nicht nur eine Frage des Stolzes: Ein
Anker aus dem Preis ist **nicht diversifizierend**. Läuft der Markt einmal
kollektiv falsch, laufen wir mit — während ein eigenes Modell in genau diesem
Moment recht behalten könnte. Was wir gegen dieses Risiko eintauschen, ist eine
gemessene Verbesserung. Der Tausch gehört bewusst gemacht, nicht nebenbei.

## Was hier NICHT behauptet wird

**Nicht: „unser Modell ist wertlos."** Es trägt das Spannen-Veto, P_pess und die
Divergenzprüfung. Gemessen ist nur, dass es als **Punktprognose** dem Markt
unterlegen ist — das war schon am 28.07. der Befund.

**Nicht: „der Markt-Anker ist automatisch besser."** Der Bucket unter dem
Markt-Favoriten trifft seltener — aber er ist deshalb auch **teurer**. Ob nach
Preis und Gebühr etwas übrig bleibt, ist die eigentliche Frage und in H6 gar
nicht gemessen worden.

**Nicht: „das ersetzt das Preisband."** Beide könnten dasselbe messen — siehe G3.

---

## Was schon gesehen wurde — vollständige Offenlegung

1. Die Tabelle oben (325 Stadt-Tage, 21 Zieltage, 30 Städte, Lead 1).
2. Der Favoriten-Backtest vom 28.07.: eigener Favorit 33,2 % Treffer, Markt-Favorit
   45,7 %; **gleicher Bucket in nur 43,1 % der Fälle**.
3. Break-even des eigenen −1-Buckets: 22,6 % bei Ø NO 0,758.
4. **Nicht bekannt:** jeder ROI des Markt-Anker-Buchs, die Preise der
   `market_fav_k − 1`-Buckets, jede Aufteilung nach Stadt, Tag oder Preisband.
   **Die Ökonomie ist vollständig ungemessen** — H6 zählte nur Treffer.

---

## Die Gegenrechnung der Gates

*(Pflichtübung seit dem Prüfstunden-Fehler desselben Tages: jedes Gate wird vor
dem Festschreiben gegen die offengelegten Zahlen gegengerechnet.)*

**Gegenrechnung 1 — trägt der Trefferquoten-Vorteil überhaupt Geld?** Bei einem
Lay zu NO 0,77 bringt ein Gewinner rund +1,39 $ und ein Verlierer −5,11 $ (5 $
Einsatz, Gebühr 0,07). Bei 23,7 % Treffern ergibt das −3,1 % ROI, bei 19,7 %
**+2,1 %**. Die Differenz von 4 pp Trefferquote ist also rund **5,2 pp ROI** wert
— vorausgesetzt, der Preis bleibt gleich. **Genau das wird er nicht:** ein
Bucket, der seltener trifft, kostet mehr. Deshalb misst G1 den ROI und nicht die
Trefferquote.

**Gegenrechnung 2 — Teststärke.** Die Streuung je Position liegt bei rund 2,7 $
gegen eine erwartete Differenz von 0,26 $. Für t = 2 wären das grob **430
Positionen**; da beide Bücher in 43 % der Fälle denselben Bucket wählen und die
Differenz dort exakt null ist, zählen effektiv nur die Abweichungsfälle.
Bei rund zehn handelbaren Kandidaten je Zieltag heißt das **40 bis 70 Zieltage**.
**Auswertung frühestens Mitte September**, mit einer Zwischenschau, die nur nach
unten entscheiden darf. Ein Gate, das früher entscheiden will, wäre Selbstbetrug.

**Gegenrechnung 3 — was G3 ausschließen muss.** Das Preisband 0,70–0,90 wählt
Buckets bereits nach dem Preis, also nach der Marktmeinung. Der Markt-Anker
könnte deshalb **dasselbe** messen. Das ist keine theoretische Sorge: Wenn der
Markt-Favorit bei YES ≈ 0,45 liegt, hat der Bucket darunter typischerweise
NO ≈ 0,80 — mitten im Band. **G3 verlangt deshalb den Vorteil innerhalb des
Bandes**, wo beide Regeln dieselbe Preisklasse handeln.

---

## Universum, Daten, Fenster

- **Zwei Bücher, identisches Regelwerk außer dem Anker.** Buch A (heute):
  Lay auf `k0 − 1`. Buch B (neu): Lay auf `market_fav_k − 1`.
- **Alles andere bleibt gleich:** Preisband 0,70–0,90, Spannen-Veto, Cap 8, 5 $,
  Lead 1, neuester Snapshot, Gebühr `0,07 · n · min(NO, 1−NO)`.
- **`market_fav_k` ist zum Kaufzeitpunkt bekannt** — der Ladder-Logger schreibt
  ihn im 12:30-Snapshot, der Autobuy kauft 12:45. Kein Blick in die Zukunft.
- **Wahrheit:** `settle_k`.
- **Signifikanz:** gepaarter t-Test über **Tages**-Mittel, beide Bücher auf
  denselben Zieltagen.
- **Fenster R:** 12.07.–01.08., Bezifferung. **Fenster F:** ab 03.08., Beleg.

---

## Hypothesen

**H1 (Geld):** Buch B liefert einen höheren ROI als Buch A.

**H2 (Mechanismus):** Der Vorteil kommt aus der **Trefferquote**, nicht aus dem
Preis — B trifft seltener und zahlt dafür nicht den vollen Aufpreis.

**H3 (Eigenständigkeit):** Der Vorteil überlebt die Beschränkung auf das
Preisband, ist also nicht bloß eine Umschreibung des Preisfilters.

**H4 (Uneinigkeit, diagnostisch):** In den Fällen, in denen beide Anker
denselben Bucket wählen, ist die Differenz definitionsgemäß null. Der ganze
Effekt muss aus den ~57 % Abweichungsfällen kommen — dort wird er getrennt
ausgewiesen.

**H0:** Der Trefferquoten-Vorteil ist vollständig eingepreist. Buch B trifft
seltener und kostet genau so viel mehr, dass nichts übrig bleibt — dieselbe
Auflösung wie bei der −1-Klasse selbst.

---

## Gates

| Gate | Bedingung |
|---|---|
| **G0** Basis (Fenster F) | ≥ 40 Zieltage, ≥ 250 Positionen je Buch, ≥ 20 Städte, ≥ 100 Abweichungsfälle |
| **G1** Geld | ROI(B) − ROI(A) ≥ **4 pp** bei **t > 2,0** über Tagesmittel |
| **G2** Mechanismus | Trefferquote(B) < Trefferquote(A) **und** B liegt unter seinem eigenen positionsweisen Break-even |
| **G3** Eigenständigkeit | G1 hält auch, wenn **beide** Bücher auf das Preisband 0,70–0,90 beschränkt werden |
| **G4** Robustheit | Beide Hälften von Fenster F gleiches Vorzeichen; der Effekt überlebt das Streichen der stärksten Einzelstadt und des besten Zieltags |

**Bonferroni:** Vier Hypothesen, ein einziger Parameter (der Anker), **keine**
Variante davon — es wird nicht `market_fav_k − 2` geprüft, kein gewichteter
Mischanker aus beiden, kein Umschalten je nach Divergenz.

**Sequenzregeln:** Zwischenschau nach 20 Zieltagen, **nur nach unten** (Abbruch
bei ROI(B) − ROI(A) < −5 pp). Ein gutes Zwischenergebnis führt zu nichts.
Verlängerung auf 70 Zieltage, falls t nach 40 zwischen 1,0 und 2,0 liegt —
einmalig und hier festgelegt.

---

## Designfallen

**1. Der Markt-Favorit ist der teuerste Bucket, per Definition.** Sein Nachbar
darunter ist deshalb systematisch anders bepreist als unser `k0 − 1`. Jeder
Vergleich, der die Preise nicht positionsweise mitführt, misst Unsinn.

**2. Beide Bücher wählen in 43 % der Fälle denselben Bucket.** Die Gesamtzahlen
sehen dadurch ähnlicher aus, als der Effekt ist — H4 weist die Abweichungsfälle
getrennt aus, ohne dass daraus ein eigenes Gate wird.

**3. `market_fav_k` kann fehlen oder wackeln.** Er ist der Bucket mit dem
höchsten `buy_yes` unter den offenen Märkten; bei dünnem Handel kann er zwischen
Snapshots springen. Fenster F verwendet **ausschließlich** den Lead-1-Snapshot,
kein Nachziehen.

**4. Ein Markt-Anker ist nicht diversifizierend.** Siehe oben. Das ist keine
Messfalle, sondern ein Risiko, das die Messung nicht sichtbar macht — es taucht
erst auf, wenn der Markt kollektiv falsch liegt, und solche Tage sind in drei
Sommerwochen kaum enthalten.

**5. Hong Kong hat floor-Buckets.** `market_fav_k − 1` ist dort derselbe
arithmetische Schritt, aber die Bucketgrenzen liegen anders. `bucket_grenzen`
bleibt stadtabhängig.

---

## Vorab-Erwartung

**Ich erwarte H0, also G1 gerissen — und zwar aus dem Befund desselben Tages.**
Die −1-Klasse ist fair bepreist; es wäre erstaunlich, wenn ausgerechnet der
Nachbar des Markt-Favoriten es nicht wäre. Der Markt weiß, wo sein Favorit liegt,
und wird den Bucket darunter entsprechend bepreisen. Meine Schätzung: B trifft
tatsächlich seltener (das ist gemessen), zahlt aber 3–5 pp mehr dafür, und übrig
bleibt ein ROI-Unterschied nahe null.

**Was mich umstimmen würde:** ein Vorteil, der im Preisband **stärker** ist als
außerhalb. Das wäre schwer als Preiseffekt zu erklären, weil dort beide Bücher
dieselbe Preisklasse handeln.

**Zur Einordnung, damit die Erwartung stimmt:** Selbst bei vollem Erfolg wären
das nach Gegenrechnung 1 rund 5 pp ROI auf ein Buch, das derzeit +0,66 % macht.
Bei ~450 $ Umsatz im Monat sind das etwa **20 $ statt 3 $**. Das ist eine echte
Verbesserung und trotzdem kein Geschäft — die Größenordnung des Buchs ändert
sich dadurch nicht.

## Abbruchregel

Reißt **G1**, bleibt der Anker `mu_ens`, und die Doktrin bleibt, wie sie ist. Es
wird **nicht** auf einen Mischanker, eine Divergenz-Umschaltung oder einen
anderen Offset ausgewichen — jedes davon wäre eine neue These.

Reißt nur **G3** (Vorteil verschwindet im Preisband), lautet der Befund: der
Markt-Anker ist eine Umschreibung des Preisfilters und bringt nichts Eigenes.
Auch das ist ein Ergebnis und wird so committet.

Bestehen G1–G4, geht **nichts sofort live**. Es folgt ein Schattenbuch über ein
zweites Fenster, und erst danach die Entscheidung — die dann ausdrücklich auch
eine Doktrin-Entscheidung ist und dem Betreiber gehört, nicht der Messung.
