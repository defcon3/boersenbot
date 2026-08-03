# Messung: Bucket-Abstandsverteilung — wie weit daneben landet der Ist?

**Gefahren:** 2026-08-03, `weather_bucket_abstand_eval.py`.
**Art:** beschreibende Messung auf Betreiber-Auftrag, **keine** vorregistrierte
Hypothese. Die einzige inferenzielle Aussage (Tel Aviv) ist unten explizit
gegen Break-even getestet und Bonferroni-korrigiert.

**Auftrag (03.08., wörtlich):** „wie oft und wo die Buckets getroffen haben?
±1, ±2, ±3 — ich glaube heute waren wir in China sogar 4 Buckets entfernt."

**Datenbasis:** `bb_WeatherLadders`, Zieltage ab 01.07.2026, nur
**Lead-1-Snapshots** (Stand vom Vortag = der Stand, auf dem der Autobuy
entscheidet). Lead 0 läge für die asiatischen Städte nach dem Tagesmaximum und
würde den Fehler kleinrechnen. 341 Stadttage mit `mu_ens`, 355 mit `market_fav_k`.

**Rundung:** Favorit ausschließlich über `weather_stations.favorit_k(mu, city)`.
Selbst gerechnetes `int(mu+0.5)` lässt Hong Kong (`BUCKET_FLOOR`, `[k, k+1)`)
fälschlich mit null Treffern erscheinen — die Falle wurde am 03.08. vormittags
schon einmal getreten.

## Gesamtbild (max-Bretter)

```
                       -4  -3  -2  -1   0  +1  +2  +3  +4
unser Favorit  n=341    3   7  20  78 113  83  25   9   2
Markt-Favorit  n=355    2   3  12  71 165  77  16   5   1
```

| | Treffer | MAE | Mittel | Schwanz |≥3| |
|---|---|---|---|---|
| `favorit_k(mu_ens)` | 33,1 % | 0,95 | **+0,04** | 6,5 % |
| `market_fav_k` | 46,5 % | 0,72 | **+0,06** | 3,9 % |

**Der globale Bias ist null — bei beiden.** Es gibt keinen Gesamtfehler zu
korrigieren; die gesamte verwertbare Verschiebung sitzt **je Stadt**. Das ist der
Grund, warum Verbesserung hier nur stadtweise möglich ist.

## Die Verschiebung je Stadt (Auszug, |Mittel| ≥ 0,8 bei n ≥ 8)

| Stadt | n | Mittel | MAE | Treffer | Markt |
|---|---|---|---|---|---|
| Tel Aviv | 18 | **−1,11** | 1,11 | 6 % | **83 %** |
| Taipei | 11 | **+1,45** | 1,45 | 0 % | 73 % |
| München | 13 | **+1,31** | 1,31 | 15 % | 23 % |
| Wuhan | 13 | **+1,08** | 1,23 | 8 % | 46 % |
| Beijing | 11 | **−1,09** | 1,09 | 45 % | **18 %** |
| Seoul | 16 | **+1,00** | 1,25 | 19 % | 35 % |
| Shanghai | 11 | +0,82 | 0,82 | 36 % | 55 % |

In **Beijing (45 % gegen 18 %)** und **Jeddah (50 % gegen 25 %)** schlagen wir den
Markt deutlich — der Rückstand ist also kein Naturgesetz, sondern stadtabhängig.

`min`-Bretter (6 Städte, 51 Stadttage): **Seoul +1,58** ist die stärkste
Verschiebung im ganzen Bestand, Tokyo liegt 5 von 5 Tagen bei −1.

## Der China-Fall ist verifiziert

**Chengdu, Zieltag 02.08.:** µ 26,10 → unser Favorit 26, Ist **30** → **+4 Buckets**.
Der Markt lag mit Favorit 25 sogar **+5** daneben. Chengdu ist die unruhigste
Stadt (MAE 1,62) und stellt 4 der 22 Schwanzfälle, am 17.07. mit −4 in die
Gegenrichtung. Zieltag 03.08. war zum Messzeitpunkt noch nicht gesettelt.

## Die einzige belegte Einzelaussage: Tel Aviv

Die −1-Klasse ist der Kern des Buchs. Je Stadt gegen Break-even 22,6 % getestet
(Binomial, Bonferroni über 30 Städte → Schwelle p < 0,0017):

| Stadt | −1-Rate | p | Urteil |
|---|---|---|---|
| **Tel Aviv** | **14/18 = 78 %** | **< 10⁻⁶** | **Lay verliert systematisch** |
| Madrid | 7/11 = 64 % | 0,0041 | verfehlt Bonferroni |
| München | 0/13 = 0 % | 0,036 | günstig, unbelegt |
| Taipei / Shanghai | 0/11 = 0 % | 0,060 | günstig, unbelegt |

Tel Aviv: 14× exakt −1, 3× −2, 1× Treffer. Unser µ ist dort praktisch nie richtig
und **nie zu kalt** — eine enge Verteilung, um genau ein Bucket verschoben. Ein
NO auf den −1-Bucket verliert dort in vier von fünf Fällen, während dieselbe
Klasse global fair bepreist ist (23,9 % gegen 22,6 %).

**Von 30 Städten überlebt eine einzige die Mehrfachtestkorrektur.** Wer aus 11–18
Tagen je Stadt ein Regelwerk baut, baut 29 Zufallsmuster. Alles außer Tel Aviv ist
Beobachtungsliste.

## Einordnung — und was daraus folgt

Der Befund ist ein **Ankerfehler**, kein Streuungsproblem: Die Verteilung in Tel
Aviv ist eng, sie sitzt nur verschoben. Das ist derselbe Mechanismus, den der
laufende Anker-Divergenz-Test misst (`d̄ ≈ D`) — dessen Auswertung am 02.09. hat
mit diesen Zahlen jetzt einen scharfen Prüfstein.

Nach der Betreiber-Entscheidung vom 02.08. wird an solchen Städten über die
**Ankerkorrektur** gearbeitet, **nicht über Sperren**. Der Befund ändert daran
nichts — er macht den Betrag messbar.

Die Folgefrage („kann man je Stadt besser werden?") ist als eigene Pre-Reg
registriert: `weather_stadt_konditional_2026_08_03.md`.
