# Pre-Registrierung: Trägt die Prüfstunde? (korrigierte Fassung) — 2026-08-02

**Status:** Vorregistrierung. Nachfolger von `weather_pruefstunde_2026_08_02.md`,
dessen G3 an einem Fehler **in G3** gerissen ist. Auswertung folgt in
`weather_pruefstunde_v2_eval.py`.

**Was sich ändert und was nicht — vorweg, damit hier kein Nachkarten entsteht:**

| | |
|---|---|
| **Unverändert** | q = 0,12 · Raster 10:20–20:20 · Monotonie-Bedingung · Signaldefinition · Datenquellen |
| **Geändert** | nur die **Kontrolle**, die nachweislich eine Aggregatgröße gegen Einzelstädte gestellt hat |
| **Neu** | der eigentliche Test — die Trennschärfe wurde nie gerechnet |

**Kein einziger Parameter der Ableitungsregel wird angefasst.** Wäre q verändert
worden, wäre das Schwellensuche und diese Pre-Reg unzulässig.

---

## Was schon gesehen wurde — vollständige Offenlegung

Alles aus dem Lauf vom 02.08. (ISD, Sommer 2024/2025, 26 von 27 Städten):

**Die Prüfstunden** (Ortszeit): 14:20 Tokyo, Taipei, Shanghai, Tel Aviv,
Wellington, Panama City · 15:20 Seoul, São Paulo, Cape Town · 16:20 Beijing,
Kuala Lumpur, Wuhan, Mexico City · 17:20 Amsterdam, Ankara, Chengdu, Helsinki,
London, Milan, Moscow, München, Toronto, Warschau, Buenos Aires · 18:20 Paris ·
19:20 Madrid. **Jeddah:** keine verwertbaren ISD-Tage, keine Prüfstunde.

**Die beiden bestandenen Gates, die hier NICHT erneut gezählt werden:**
G1 Relevanz 22/26 Städte (85 %), G2 Stabilität **26/26 (100 %)** zwischen den
Sommern. Sie sind gemessen; sie erneut als Gate zu führen wäre Doppelzählung.

**Der Fehler:** G3 verlangte für London, Paris und Madrid je 17:20–18:20. Die
12 %, aus denen q stammt, sind aber ein **Durchschnitt über fünf Städte** der
Erstmessung, nicht die Restwahrscheinlichkeit einer Einzelstadt. Madrid liegt um
16:20 bei 71,4 %, Paris bei 45,5 %, London bei 17,9 % — drei Städte, drei
Kurven. Das Gate war mit den Zahlen unvereinbar, die in derselben Datei
offengelegt waren.

---

## Die Gegenrechnung der Gates — neue Pflichtübung

Die Lehre des Tages lautet: **jedes Gate wird vor dem Festschreiben gegen die
bereits offengelegten Zahlen gegengerechnet, und die Gegenrechnung steht in der
Pre-Reg.** Zweimal ist heute eine Aggregatzahl zur Schwelle für Einzelfälle
geworden — bei G5 der Ursachen-Pre-Reg ein Median statt des gehandelten
Mittelpreises, bei G3 hier ein Fünf-Städte-Mittel statt der Einzelstadt. Beide
Male hätte eine Minute Nachrechnen genügt.

**Gegenrechnung 1 — was die Prüfstunden-Korrektur überhaupt berühren kann.**
Sie ändert nur dort etwas, wo die neue Stunde von 16:20 abweicht. Bei rund elf
Kandidaten je Stadt und 323 Kandidaten insgesamt entfallen auf die fünf am
stärksten korrigierten Städte (Tokyo, Taipei, Shanghai, Madrid, Paris) etwa
**17 %** der Menge. **Ein global gerechneter Trennschärfe-Vergleich misst
deshalb überwiegend Verdünnung, nicht Wirkung.** Konsequenz, vorab gezogen: G1
rechnet **ausschließlich auf den Städten mit geänderter Prüfstunde**. Die
unveränderten Städte liefern definitionsgemäß identische Zahlen und gehören
nicht in den Vergleich.

**Gegenrechnung 2 — warum die Reproduktionsprobe kein Gate sein darf.** Die
gepoolte Restwahrscheinlichkeit der fünf Erstmess-Städte bei 17:20 müsste nahe
12 % liegen. Aus den bekannten Prüfstunden folgt aber: London, München und
Helsinki liegen dort bereits unter 12 %, Paris (Prüfstunde 18:20) darüber, Madrid
(19:20) deutlich darüber. Der gepoolte Wert kann damit ohne Weiteres bei 15–25 %
landen — **ohne dass irgendetwas falsch wäre**, denn die Erstmessung stammt aus
Juli 2026 über WU-METAR, diese aus 2024/2025 über ISD. Eine Kontrolle, die
Jahre und Quellen mischt, kann eine Regel nicht erledigen. **Sie läuft deshalb
diagnostisch, ohne Gate.** Genau diese Gegenrechnung hat beim ersten Mal
gefehlt.

**Gegenrechnung 3 — Teststärke.** Erwartet werden rund 100 Signal-Fälle
insgesamt, auf den korrigierten Städten entsprechend weniger. Ein t trüge dort
nichts. G1 ist deshalb als **Richtungsentscheidung** formuliert und nicht als
Signifikanztest; ein Gate, das nichts belegen kann, soll auch nicht so tun.

---

## Hypothese und Gates

**H1 (Haupttest):** Auf den Städten mit geänderter Prüfstunde ist die
**Trennschärfe** des Wächter-Signals mit der stadtspezifischen Stunde größer als
mit der starren 16:20-Prüfung. Trennschärfe = **Lift**, also
P(Lay verliert | Signal) − P(Lay verliert | kein Signal), paarweise auf
**denselben** Kandidaten gerechnet.

*Warum Lift und nicht Trefferquote:* Eine zu frühe Prüfung (Madrid) erzeugt
Fehlalarme, eine zu späte (Taipei) verpasst nichts, kommt aber zu spät zum
Handeln. Beide Fehler senken den Lift, während die reine Trefferquote je nach
Mischung in beide Richtungen wandern kann.

| Gate | Bedingung |
|---|---|
| **G0** Basis | ≥ 8 Städte mit geänderter Prüfstunde und ≥ 60 Kandidaten auf diesen Städten; IEM-Reihe für ≥ 80 % der Stadt-Tage verfügbar |
| **G1** Trennschärfe | Lift(stadtspezifisch) > Lift(starr 16:20) auf den geänderten Städten, **und** die Zahl der Signal-Fälle sinkt nicht unter die Hälfte |
| **G2** Richtungsprobe | Der Vorteil kommt **nicht** aus einer einzigen Stadt: Streichen der stärksten Stadt dreht das Vorzeichen nicht |

**Bonferroni:** Eine Hypothese, zwei Gates, **kein** freier Parameter — q, Raster
und Signaldefinition sind aus der Vorgänger-Pre-Reg übernommen und stehen fest.

**Diagnostisch, ohne Gate:** die Reproduktionsprobe (gepoolte Kurve der fünf
Erstmess-Städte gegen 91/87/76/41/12 %), die Prüfstunden in Berliner Zeit, und
die Städte ohne Prüfstunde.

---

## Vorab-Erwartung

**G1 erwarte ich als knapp bestanden oder gerissen — mit leichter Neigung zu
gerissen.** Der Grund steht in Gegenrechnung 1: Selbst auf den korrigierten
Städten ist die Menge klein, und die beiden Fehlerarten heben sich teilweise auf.
Madrid gewinnt durch die spätere Stunde an Präzision, Taipei verliert durch die
frühere möglicherweise welche — bei 14:20 ist auch dort noch nicht jeder Tag
entschieden.

**Die Reproduktionsprobe erwarte ich bei 15–25 % statt 12 %**, aus dem in
Gegenrechnung 2 genannten Grund. Das ist kein Mangel und wird nicht als solcher
berichtet.

**Was das Ergebnis wert ist:** Besteht G1, ist die Prüfstunden-Tabelle belegt und
für **manuelle** Entscheidungen verwendbar. Reißt G1, bleibt sie eine gemessene,
stabile Basisraten-Tabelle ohne nachgewiesenen Nutzen für das Signal — was für
die Handlungsregel „wann ist der Tag entschieden" **immer noch brauchbar** ist,
denn dafür genügt die Basisrate selbst. **Kein Euro Ertrag in beiden Fällen.**

## Abbruchregel

Reißt **G1**, wird **nicht** auf eine andere Signaldefinition, ein anderes q oder
eine andere Städtemenge ausgewichen. Der Befund lautet dann: die Stunde ist
stabil ableitbar, aber sie verbessert das Wächter-Signal nicht — und der Wächter
bleibt ohnehin auf Eis.

Reißt **G0** (zu wenig Daten), wird nicht gerechnet und der Test auf ein
größeres Kandidatenfenster vertagt, ohne die Schwellen anzufassen.

**Der Wächter wird in keinem Ausgang eingeschaltet.** Breite ist am 02.08.
abgelehnt worden; ohne sie schadet er dem engen Buch.

**Und diese Pre-Reg bekommt keinen dritten Anlauf.** Reißt sie, ist das Thema
Prüfstunde abgeschlossen — mit der Tabelle als Ergebnis und ohne Anspruch auf
mehr.

---

# ERGEBNIS — G1 und G2 belegt (02.08.2026)

Gerechnet mit `weather_pruefstunde_v2_eval.py` unmittelbar nach der
Vorregistrierung (Commit `2578575`). 22 der 26 Städte haben eine geänderte
Prüfstunde; darauf entfallen **246 der 322 Kandidaten**, IEM-Abdeckung 94 %.

## G1 — Trennschärfe

| Prüfung | Signale | Verlierer bei Signal | **Lift** |
|---|---|---|---|
| starr 16:20 | 69 | 78,3 % | **+75,2 pp** |
| **stadtspezifisch** | 56 | **92,9 %** | **+88,9 pp** |

**Differenz +13,7 pp, BELEGT.** Die Zahl der Signale sinkt von 69 auf 56 und
bleibt damit weit über der geforderten Hälfte. Ein Lift von 88,9 pp bei einer
Trefferquote von 92,9 % heißt: ohne Signal verlieren nur rund 4 % der Lays.

**G2 bestanden:** ohne Madrid — die stärkste Einzelstadt — bleibt die Differenz
bei **+8,4 pp**, das Vorzeichen dreht nicht.

## Die Reproduktionsprobe (diagnostisch)

| Stunde | ISD 2024/25 | WU Juli 2026 |
|---|---|---|
| 13:20 | 84,2 % | 91 % |
| 14:20 | 71,1 % | 87 % |
| 15:20 | 54,1 % | 76 % |
| 16:20 | 33,1 % | 41 % |
| **17:20** | **14,4 %** | **12 %** |

Bei 17:20 — der Stunde, aus der q stammt — stimmen beide Quellen fast überein.
Die früheren Stunden weichen stärker ab, ISD zeigt durchweg frühere Maxima.
**Ein Teil davon ist kein Jahreseffekt, sondern ein Stationsunterschied:** das
ISD-Skript nutzt für London **EGLC** und für Paris **LFPB**, die Erstmessung
EGLL und LFPG. Für London ist EGLC die settlement-relevante Station — die neue
Messung steht also auf der richtigen, die alte auf der falschen.

## Was an der Vorab-Erwartung falsch war

Beide Erwartungen waren zu pessimistisch. G1 wurde nicht „knapp bestanden oder
gerissen", sondern klar belegt; die Reproduktionsprobe landete bei 14,4 % statt
der erwarteten 15–25 %. **Wäre sie als Gate mit dem in Gegenrechnung 2
erwogenen Intervall geführt worden, hätte sie bestanden.** Die Vorsicht war
richtig, aber sie war nicht nötig — was nichts daran ändert, dass sie richtig
war: eine Kontrolle, die Jahre, Quellen *und* Stationen mischt, darf eine Regel
nicht erledigen können.

## Was das NICHT bedeutet — die Verwertung bleibt tot

Ein Signal mit 92,9 % Trefferquote sieht nach einem Ausstiegssignal aus. **Es ist
keines.** Am 01.08. wurde der konditionale Ausstieg vorregistriert gemessen und
in **allen sieben Fassungen** falsifiziert — darunter Variante B, die exakt eine
Prüfung zur Ortsstunde 16:20 ist: eingepreist 0,396, realisiert 0,421, t = −0,51.
Der Markt preist die Erholungschance korrekt bis leicht zu pessimistisch, und ein
gekappter Gewinner kostet mehr (~0,60 je Kontrakt), als eine gerettete Position
bringt (~0,40). → [[weather-conditional-exit-falsified]], [[weather-no-manual-exits]]

**Trennschärfe ist nicht Verwertbarkeit.** Das ist derselbe Befund wie bei der
−1-Klasse am selben Tag: ein Signal kann die Wahrheit gut vorhersagen und
trotzdem nichts wert sein, wenn der Markt sie ebenso gut vorhersagt.

## Status

Die Prüfstundentabelle ist damit **belegt** — als das, was sie ist: eine
stabile, stadtspezifische Antwort auf die Frage „ab wann ist das Tageshoch
entschieden". Verwendbar für die manuelle Regel „nicht vor der Prüfstunde
aufstocken", die den Helsinki-Fall vom 24.07. ausgelöst hat.

**Nicht verwendbar als Ausstiegssignal**, und der Wächter bleibt auf Eis.

## Was nicht geprüft und bewusst offen ist

- **Der 01.08.-Ausstiegstest lief mit starrer 16:20-Stunde.** Ob er mit
  stadtspezifischer Stunde anders ausginge, ist nicht gemessen. Das ist die
  einzige Tür, die dieser Befund öffnen würde — und sie ist doppelt zu:
  [[weather-conditional-exit-falsified]] sagt „nicht wieder aufmachen", und die
  Abbruchregel dieser Pre-Reg sagt „kein dritter Anlauf". **Sie wird hier nicht
  aufgestoßen.**
- **56 Signal-Fälle tragen kein t.** G1 war als Richtungsentscheidung
  formuliert, und mehr ist es auch geworden.
- **Drei Prüfstunden fallen in die Berliner Nacht** (Buenos Aires 22:20, Toronto
  23:20, Wellington 04:20) und sind manuell unbrauchbar. Tokyo liegt mit 07:20
  am Rand.
- **Jeddah hat keine Prüfstunde** (keine verwertbaren ISD-Tage) und ist zugleich
  eine der divergenten Städte aus [[weather-anker-divergenz-forwardtest]].
- **Der Sommer bleibt der Sommer.** Zwei Jahre, Juni bis August.
