# Pre-Reg: Auf welcher Uhr sitzt der Markt? — Handelsmasse gegen die Beobachtungszeit

**Angelegt:** 2026-08-04, vor dem ersten Blick auf die Zielgröße.
**Status:** registriert, noch nicht gefahren. **Leg 1 blockiert** (s. Vorbedingung).
**Auswertung durch:** `weather_market_clock_eval.py` (noch zu schreiben).

## Anlass

`BACKLOG.md` führt seit 01.08. den Test „sitzt der Markt auf den METAR-Spiegeln
oder ist er schneller?" — angestoßen von metar.ws (METAR-Push, 49/199 €). Der
**Modell**-Zweig des Anbieters ist am 04.08. geschlossen
(`weather_icon_source_bound.py`); der **Beobachtungs**-Zweig ist offen und der
Test kostet nichts.

**Neu gegenüber dem Backlog-Eintrag — die Frage ist schärfer geworden.** Aus
`weather_source_latency.csv` (199 Erstsichtungen, Beobachtungszeit →
Erstsichtung je Quelle):

| Quelle | n | Median | p25 | p75 |
|---|---|---|---|---|
| NOAA | 97 | **3,8 min** | 3,0 | 4,9 |
| IEM | 51 | 4,7 min | 3,9 | 5,4 |
| **WU-Tabelle** (settelt) | 51 | **9,0 min** | 4,3 | 34,8 |

**Die settelnde Quelle ist die langsamste.** Wir haben über NOAA bereits rund
5 min Vorlauf gegenüber der Tabelle, auf die unsere Bretter settlen — gratis,
und wir nutzen ihn nicht. Ein 1-Minuten-Dienst könnte gegenüber NOAA damit
höchstens ~2,8 min holen, nicht die 6, die der Backlog unterstellte. Die
Kaufentscheidung hängt also nicht mehr an diesem Test, sondern an der Frage,
ob der **freie** Vorlauf überhaupt etwas wert ist.

## ⚠️ Vorbedingung für Leg 1 (Stand 04.08. nicht erfüllt)

Der Backlog behauptet, die Sonde laufe seit 31.07. auf dem VPS. **Sie läuft
nicht:** keine systemd-Unit für `weather_source_latency_probe.py`, `ps` findet
keinen Prozess, und die VPS-CSV ist byte-identisch zur lokalen (md5
`1d079914abe0fb04cfd642c88b22f29c`, letzte Zeile `2026-08-01T07:58:40Z`). Sie
wurde am 31.07. einmal mit `--hours` gestartet und ist ausgelaufen — dieselbe
Falle wie bei `jupiter-prediction-bot`: *`systemctl active` ≠ Loop lebt.*

Die Tabelle oben ruht damit auf **11 Stunden** eines einzigen Nachmittags.
**Leg 1 gilt erst als gemessen, wenn die Sonde ≥ 7 Tage durchgelaufen ist**
(Ziel ≥ 300 Erstsichtungen je Quelle). Bis dahin sind die drei Mediane oben
Indikation, kein Befund — und dürfen in keiner Entscheidung als Zahl auftreten.

## Die zwei Beine

* **Leg 1 — wann sieht *eine Quelle* die Beobachtung?** Läuft über die Sonde,
  rein deskriptiv, kein Gate. Liefert die Referenzzeiten für Leg 2.
* **Leg 2 — wann sieht *der Markt* sie?** Der eigentliche Test, unten.

## Zielgröße (Leg 2)

Ereignis = Kipp-Ereignis wie in `weather_foreknowledge_eval.py`: die settelnde
WU-History-Tabelle steht ≥ 60 min auf `k` (Plateau), dann springt das laufende
Tagesmaximum auf `k+1`.

Zeitanker ist **nicht** der Tabellensprung, sondern die `valid_time_gmt`
derjenigen Beobachtung, die das neue Maximum trägt — also `obs`.

Gemessen wird der Netto-Taker-Fluss je Minute im Fenster `[obs−30, obs+30]`,
getrennt für den sterbenden Bucket `k` und den kommenden `k+1`, normalisiert
wie gehabt auf das Tagesvolumen des Buckets. Kennzahl je Ereignis:

    m = die Minute relativ zu obs, bei der die kumulierte |Flussmasse|
        des Fensters 50 % überschreitet   (Median-Reaktionszeit)

Aggregiert über Ereignisse: `m̂ = Median(m)` mit Bootstrap-KI (10.000 Ziehungen,
auf **Stadt**-Ebene geclustert — Tage einer Stadt sind korreliert).

## Hypothese

Drei Kandidaten-Uhren, an denen der Markt hängen kann:

| Anker | Bedeutung |
|---|---|
| **obs+1** | jemand hat einen 1-Minuten-Feed — wir kaufen den letzten Platz |
| **obs+4** | Markt sitzt auf NOAA/IEM, also auf unseren eigenen freien Quellen |
| **obs+9** | Markt sitzt auf der WU-Tabelle, dem langsamsten Glied |

**H (zu konfirmieren): `m̂ ≥ 4 min`** — der Markt ist nicht schneller als die
freien Spiegel.

**Erwartung des Registrierenden: H hält.** Grund: am 29.07. war der Kauf-Vorlauf
im Gewinner-Bucket bereits als reine Latenz identifiziert (t = +6,31 bei 0 min
Ausschluss → −0,53 bei 45 min), und der Effekt saß fast vollständig im
**Verkaufs**druck auf die sterbende Stufe. Das ist Reaktion, nicht Vorwissen.
Wäre ein 1-Minuten-Feed breit im Markt, müsste die Masse enger an obs kleben,
als die Spiegel es erlauben.

Dies ist **eine** Lageschätzung, verglichen gegen drei Referenzpunkte — nicht
drei Tests. Keine Bonferroni-Korrektur.

## Gates

- **G1 (Power):** ≥ 25 Kipp-Ereignisse mit auflösbarer `valid_time_gmt` **und**
  ≥ 20 Trades im Fenster. Sonst **UNDERPOWERED** — berichten, nicht deuten.
- **G2 (Lage):** Bootstrap-KI (95 %) von `m̂` schließt **2,5 min** aus. Liegt
  `m̂` darüber, ist die 1-Minuten-These verworfen.
- **G3 (Leckfreiheit):** der Fluss in `[obs−30, obs)` trägt nicht signifikant
  (|t| < 2,0). Sonst kontaminiert Trend-Extrapolation das Fenster und der
  Ankerbezug ist wertlos — die Falle, die den 29.07.-Test definiert hat.
- **G4 (Konzentration):** Befund überlebt den Ausschluss der zwei
  ereignisreichsten Städte. Grund: beim Latenz-Zwischenstand am 06.07. kamen
  5 von 6 Episoden aus Shenzhen allein.

## Entscheidungsregel — vorregistriert, weil ein Test ohne Folge keiner ist

**Kein Ausgang dieses Tests führt zu einem Kauf.** Das ist bewusst so:

- **`m̂ ≤ 2,5 min` (G2 gerissen):** der schnelle Feed ist längst im Markt.
  Anbieterfrage **endgültig geschlossen**, Backlog-Eintrag nach „Erledigt".
- **`m̂ ≥ 4 min` (H hält):** dann existiert ein freier Vorlauf — **über NOAA,
  nicht über einen Anbieter.** Folge ist *nicht* der Kauf, sondern Handlung X:
  > **X:** Paper-Vergleich, ob der Autobuy-**Einstieg** besser wird, wenn er
  > NOAA liest statt auf die Tabelle zu warten. Er kauft im Preisband
  > 0,70–0,90 (`weather_minus1_autobuy.py`); sieht er den METAR ~5 min vor der
  > Tabelle, kauft er zu Preisen, die die Tabelle noch nicht kennt.
  > Rein historisch/Paper, **keine** Live-Änderung, eigene Pre-Reg mit eigenen
  > Gates. Erst ein positives X macht die Anbieterfrage überhaupt wieder
  > sinnvoll — und selbst dann wäre der Kaufhebel nur NOAA→1 min ≈ 2,8 min.
- **UNDERPOWERED:** Sonde weiterlaufen lassen, in vier Wochen erneut. Keine
  Deutung der Richtung.

**Ausdrücklich ausgeschlossen als Folge — dreimal falsifiziert, nicht wieder
aufmachen:** ein Ausstiegs-/Stop-Signal aus dem Vorlauf
(`weather-conditional-exit-falsified`, G1 in 7 Fassungen gerissen;
`weather-no-manual-exits`: Halten +3,82 % vs. verkauft −8,00 %) und jede Form
von Intraday-Scalp (`weather-intraday-nowcast-scalp`, endgültig begraben).
Der Vorlauf darf den **Einstieg** verbessern, nicht den Ausstieg auslösen.

## Bekannte Limitationen (bewusst in Kauf genommen)

- **Der Nullpunkt ist ein Zeitstempel, keine Messung.** `valid_time_gmt` ist die
  nominelle Beobachtungszeit; zwischen Messung und Absetzen des METAR liegen
  weitere Minuten, die wir nicht sehen. Alle drei Kandidatenanker teilen diesen
  Versatz, der **Vergleich** bleibt also gültig — die **absoluten** Minuten sind
  nach oben verzerrt und dürfen nicht als „so alt ist die Luft" gelesen werden.
- **n ist dünn.** Der 29.07.-Lauf fand 35 Kipper mit sauberem Plateau bei
  292 Stadt-Tagen. G1 kann scheitern; dann ist das Ergebnis „noch nicht
  messbar", nicht „kein Effekt".
- **Nur die Taker-Seite ist sichtbar.** Wer ruhende Orders stehen hat und
  bedient wird, erscheint mit dem Zeitstempel des Gegenübers.
- **Städte ohne METAR fallen raus** — Hong Kong (HKO-Pseudostation) und alles,
  was über die WU-Kachel statt über eine Station settelt.
- **SPECI vs. routinemäßiges METAR** wird nicht getrennt. Ein Kipp bei
  Hitze löst oft ein SPECI aus, das früher kommt als die halbstündige Meldung —
  das würde `m̂` nach unten ziehen und H *konservativ* prüfen.

## Auswertung

Nach Leg-1-Vorbedingung: `python weather_market_clock_eval.py --plateau 60
--window 30` → Ergebnis-Abschnitt hier nachtragen, committen. Maschinerie
weitgehend vorhanden: `weather_foreknowledge_eval.py` liest WU-History
(`valid_time_gmt`) und Trade-Tape bereits zusammen; umzustellen sind Zeitanker
(t0 → obs) und Kennzahl (Gruppenvergleich → Lageschätzung).

**Verweise:** `weather_source_latency_probe.py`, `weather_foreknowledge_eval.py`,
`weather_minus1_autobuy.py`, `POLYMARKET_DATA_API.md`, `BACKLOG.md`.

---

## ERGEBNIS (auszufüllen nach dem Lauf)

*offen — Leg 1 blockiert, Sonde läuft nicht*
