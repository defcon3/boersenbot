# Polymarket Public Data API

Jupiters Wetter-/Sport-Märkte **sind** Polymarket-Märkte; die IDs sind
durchgereicht: `POLY-3025209` → Market-ID `3025209`, `POLY-731565` → Event-ID.
Alles öffentlich, **kein Auth, keine Kosten** (verifiziert 2026-07-24).

Damit lassen sich Mikrostruktur-Fragen (Spread, Fluss, Buchtiefe) **rückwirkend**
beantworten — kein Logger, keine Vorlaufzeit, kein Risiko.

## Endpoints

| Zweck | Aufruf |
|---|---|
| Markt-Metadaten | `GET gamma-api.polymarket.com/markets/{id}` |
| Alle Buckets eines Tages | `GET gamma-api.polymarket.com/events/{id}` |
| Event per Slug | `GET gamma-api.polymarket.com/events?slug=...` |
| **Buchtiefe** | `GET clob.polymarket.com/book?token_id=...` |
| **Trade-Tape (historisch)** | `GET data-api.polymarket.com/trades?market={conditionId}&limit=500&offset=` |
| Preishistorie (1 min) | `GET clob.polymarket.com/prices-history?market={tokenId}&interval=1d&fidelity=1` |

Slug-Muster der Wetter-Events: `highest-temperature-in-{stadt}-on-{monat}-{tag}-{jahr}`
(z. B. `highest-temperature-in-helsinki-on-july-24-2026`).

Nützliche Felder aus `/markets/{id}`: `conditionId` (für den Tape), `clobTokenIds`
(für Buch und Preishistorie), `bestBid`/`bestAsk`/`spread`, `lastTradePrice`,
`volumeNum`, `enableOrderBook`.

429-Backoff einplanen (~3–5 s). Der Tape paginiert über `offset`.

## Effektiven Spread aus dem Tape rekonstruieren

Das Frontend zeigt nur zehn Stufen um die Mitte; der Tape trägt dagegen die
**Seite** jedes Trades, und daraus lässt sich der real bezahlte Spread
zurückrechnen. Erst auf Yes-Äquivalent normalisieren:

```
BUY  Yes @ p  ->  Taker hebt den Yes-Brief bei p       ("ask")
SELL No  @ r  ->  dasselbe, bei 1-r                    ("ask")
SELL Yes @ p  ->  Taker trifft das Yes-Geld bei p      ("bid")
BUY  No  @ r  ->  dasselbe, bei 1-r                    ("bid")
```

Dann je Zeitfenster: `eff. Spread = mean(ask-Preise) - mean(bid-Preise)`.
Das misst den Spread genau dann, wenn gehandelt wird. Preisdrift innerhalb des
Fensters verzerrt nach oben — die Schätzung ist also konservativ.
Referenz-Implementierung: `weather_mm_spread_test.py`.

## Warnung: nicht `bb_WeatherLatency.all_prices` verwenden

Für Mikrostruktur ist diese Spalte unbrauchbar. Die Bucket-Preise eines
Zeitpunkts summieren sich auf Median **1,08** und bis **1,54** statt auf 1,00 —
es sind veraltete Einzelnotierungen aus verschiedenen Momenten. Eine
Autokorrelations-Auswertung darauf zeigt scheinbare Mean Reversion
(r = −0,11, t = −54), die reines Artefakt ist. Der Tape ist die richtige Quelle.

## Verwandte Quelle: Wunderground (Settlement)

Die Settlement-Tabelle liegt bei `api.weather.com` (öffentlicher Web-Key, s.
`weather_latency_logger.py` bzw. Projektnotizen):

```
https://api.weather.com/v1/location/{ICAO}:9:{LAND}/observations/historical.json
    ?apiKey=...&units=m&startDate=YYYYMMDD
```

**Zeitzonen-Falle:** `valid_time_gmt` ist UTC. Die WU-*Website* rendert die
Tabelle dagegen in der Zeitzone des Betrachters — eine Beobachtung um 14:50 UTC
erscheint einem deutschen Nutzer als 16:50 (MESZ), einem Helsinki-Nutzer als
17:50 (EEST). Beim Vergleich von Messzeit und Marktbewegung immer über
`valid_time_gmt` gehen, nie über die angezeigte Uhrzeit.
