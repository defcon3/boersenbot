# Jupiter Tennis-Wett-Automat — Stand & Plan

**Datum:** 2026-06-24
**Wallet (Hot):** `4XxStoKPzoiEJ6hUGEESfE54dCRo97LcCGk2UFieKjSi`
**Ziel:** Ein Automat, der auf Jupiter Prediction Markets (Polymarket-gespiegelt)
Tennis-Matches handelt — Limit-Orders *unter* dem fairen Wert, überwachen,
stornieren, bei Gewinn claimen. **Edge-Quelle muss erst bewiesen werden** (s. u.).

---

## 1. Heutige Wett-Post-mortems (Lehren, nicht Ergebnisse)

| Match | Was passiert | Lehre |
|---|---|---|
| Passaro (Târgu Mureș, In-Play) | Auf momentane Satzführung 2:1 gesetzt, **ohne Live-Bild**, verloren | Momentane Score-Führung ist **Rauschen**, kein Signal. Ohne Live-Zugang kein In-Play-Edge → nicht wetten. War der am schlechtesten informierte Marktteilnehmer (-EV). |
| Quinn vs. Cassone (Mallorca) | Quinn-Kauf zu Limit 0.64, Markt war bei 0.67 → **nicht gefüllt** (um 3 ¢ verpasst), Quinn gewann | Bei 0.67 lag echter Value (fair ~0.79). Auf 0.64 zu pokern, um 3 ¢ zu sparen, kostete eine +EV-Wette. **Wenn der Preis schon unter fair liegt: Füllung > letzte Cents.** Trotzdem: am Prozess bewerten, nicht am Ausgang. |
| Onclin vs. Mochizuki (Wimbledon-Quali) | Markt 0.58/0.42, faire Schätzung ~0.60/0.40 → **kein Edge** | Markt ≈ fair (Summe 1.00, keine Marge erkennbar). „Kein Edge → kein Einsatz." Richtig erkannt: Pass. |

**Kernprinzip (bestätigt):** Faire Wahrscheinlichkeit fundamental herleiten
(Ranking/Form/Belag/Elo), gegen den Marktpreis halten — **nicht** Fremdquoten
nachlaufen. Favoriten zu kurzen Quoten (z. B. -450) sind selten Value.

---

## 2. ✅ Ausführungspfad VOLLSTÄNDIG bewiesen (echtes Geld, sauber rückabgewickelt)

Die zentrale „Betfair-Sorge" (Edge finden, aber nicht ausführbar) ist **erledigt**.
Jeder Baustein wurde live mit echtem Geld getestet — Nettokosten: nur Gas-Staub.

API-Base: `https://prediction-market-api.jup.ag` (Gateway-Alias `api.jup.ag/prediction/v1`)

| Schritt | Mechanismus | Status |
|---|---|---|
| Märkte lesen | `GET /api/v1/markets/{marketId}` (Felder `clobTokenIds`, `pricing`, `outcomes`) | ✅ |
| Orderbook | `GET /api/v1/orderbook/{marketId}` → Tiefen-Ladder `{yes:[[preisCent,size]], no:[…]}` | ✅ |
| **Limit-Order platzieren** | `POST /api/v1/orders` → signieren → `POST /api/v1/execute {signedTransaction}` | ✅ live (Tx `5P1vYTzq…`) |
| **Stornieren (async!)** | `POST /api/v1/orders/cancel` → signieren → `POST /api/v1/execute {signedTransaction, context}` | ✅ live (Konto geschlossen, Einsatz zurück) |
| Verkaufen / Claimen | `DELETE /positions/{pk}` / `POST /positions/{pk}/claim` | ✅ (früher bewiesen) |

### Order-Schema (`POST /api/v1/orders`)
- `ownerPubkey`, `marketId`, `isYes` (bool), `isBuy` (bool)
- **`maxBuyPriceUsd`** = Limit-Preis Kauf (10000–999999 = $0.01–$0.99); `minSellPriceUsd` = Limit Verkauf
- `orderType: "limit"`, Menge via `depositAmount` (1e6=$1) oder `contractsMicro`
- `depositMint`: USDC `EPjFW…` oder **JupUSD `JuprjznT…USD`** (App nutzt JupUSD; auch Claims kommen in JupUSD)
- **Mindest-Order: $5**
- On-chain: 2 Signer — Owner Slot 0 (wir), Jupiter-Relayer `5uFXJogU…` Slot 1 vorsigniert → `sign_close_tx` reicht
- Platzierung braucht **kein eigenes Solana-RPC** (Jupiter relayed via `/execute`)

### Cancel-Mechanik (der knifflige Teil — async)
1. `POST /api/v1/orders/cancel {ownerPubkey, orderPubkey}` → Antwort enthält:
   - `transaction` (nur Memo+System, **kein** on-chain CloseOrder)
   - `execution: { endpoint: "/api/v1/execute", context: { type:"limit_order_cancel", ownerPubkey, orderPubkey, cancelExpiresAtMs } }`
2. `transaction` signieren
3. `POST /api/v1/execute { signedTransaction, context: <das INNERE execution.context> }`
   → `status:"cancel_requested", mode:"async"`; ein Jupiter-Keeper schließt das
   Order-Konto Sekunden später on-chain, Einsatz zurück.
- **Fallen:** Pfad ist `/api/v1/execute` (NICHT `/orders/execute`); `context` ist
  das **innere** Objekt (ganzes `execution` → `invalid_union`); Cancel-Tx roh aufs
  RPC = wirkungsloser Memo-No-op.
- **Verifikation:** `getAccountInfo(orderPubkey) == null` = geschlossen.
  Offene Limits: `GET /api/v1/orders?ownerPubkey=` (`status:pending` = ruhend/unfilled).

(Voll dokumentiert auch in der Projekt-Memory `jupiter-prediction-bot.md`.)

---

## 3. Edge-Hypothese (NOCH UNBEWIESEN — kein Live-Geld auf Verdacht)

**Beobachtung heute:** Bei allen drei Matches lag meine faire Schätzung ~auf der
Marktlinie. Das beweist nur, dass wir *schlechte* Wetten vermeiden — **kein**
Beleg für einen *positiven* Edge.

**Hypothese:** Dünne **Quali-/Challenger-Märkte** auf Jupiter/Polymarket sind ggf.
weniger effizient (wenig Volumen, langsame Preise) → ein faires Modell könnte den
Schlusskurs schlagen.

**Kritisches Risiko — Adverse Selection:** Eine ruhende Limit-Order *unter* fair
füllt bevorzugt dann, wenn neue Info den fairen Wert *unter* das Limit drückt
(man wird von Besserinformierten abgepickt). „Wenn gefüllt → Value" ist eine
Falle. Ob auf dünnen Märkten genug *nicht-adversative* Fills (Liquiditäts-Rauschen)
übrig bleiben, lässt sich **nur messen**, nicht herleiten.

---

## 4. Nächste Schritte (in dieser Reihenfolge)

1. ✅ **Faires-Wahrscheinlichkeits-Modell** gebaut → `tennis_elo_model.py`.
   - **Datenquellen-Pivot (2026-06-24):** Jeff Sackmanns `tennis_atp`/`tennis_wta`-
     Repos sind **nicht mehr öffentlich** (Account exponiert nur noch
     `tennis_MatchChartingProject`). Eigenes Elo-Rechnen daraus ist damit hinfällig.
   - Stattdessen: Live-Elo-Reports auf **tennisabstract.com** (`atp_elo_ratings.html`,
     `wta_elo_ratings.html`) — pro Spieler fertiges Overall- + Hard/Clay/Grass-Elo.
     Lokaler CSV-Cache (`tennis_elo_{atp,wta}.csv`), regeneriert aus der URL (12 h TTL).
   - Modell: Belags-Elo = `blend·Overall + (1−blend)·Belag` (blend 0.5);
     `P(A) = 1/(1+10^((EloB−EloA)/400))`. Akzentinsensitives Namens-Matching.
   - **Abdeckungs-Caveat (wichtig für die Edge-These):** Report reicht runter bis
     ATP-Rang **~531** (531 ATP / 533 WTA Spieler). Das deckt **Grand-Slam-Quali
     und Haupt-Challenger-Felder** (Ränge ~100–400) ab — also die Zielmärkte — aber
     **NICHT ITF-Futures**/Spieler jenseits ~531. Für die liefert das Modell bewusst
     `None` (kein Raten) → solche Märkte sind für diesen Bot schlicht out-of-scope.
   - Sanity: Sinner–Alcaraz Clay 0.69/0.31, Gras 0.65/0.35; Onclin–Mochizuki Gras
     0.545/0.455 (deckt sich grob mit der Hand-Schätzung ~0.60 aus Abschnitt 1).
2. ✅ **Forward-Paper-Test-Logger** gebaut → `tennis_paper_logger.py` (KEIN Echtgeld).
   - **Discovery:** Jupiter `GET /api/v1/events?subcategory={atp,wta}` → 2-Wege-Singles
     (Doppel/Futures rausgefiltert). Live-Inventory gesund: ~46 Singles, darunter echte
     **Wimbledon-Quali (Gras)** + Challenger (Piracicaba/Clay). **Polymarkets eigener
     `tennis`-Tag ist dagegen tot** (2 veraltete ITF/Doppel-Events) → Jupiter ist die Quelle.
   - **Belag** wird aus dem Turniernamen geschätzt (Keyword-Map, Default hard); Turnier
     wird mitgespeichert → Belag jederzeit in SQL korrigierbar.
   - Pro Match: faire P (Elo) vs. Markt-Ask (`buyYesPriceUsd`); nur wenn **beide** Spieler
     im Report → sonst skip (out-of-coverage, ehrlich geloggt). **Roh** gespeichert
     (`fair_a/fair_b`, `price_a/price_b`, `edge`, `value_side`) → Schwelle/Blend später
     frei variierbar (Philosophie wie der Football-Collector). Nur **PRE-MATCH** (Upsert
     bis Anpfiff = Closing-Linie; In-Play wird NICHT geloggt, da score-getrieben).
   - **Storage:** Centron `bb_TennisPaperBets` (Nutzer-Wahl; per SQL auswertbar). End-to-end
     verifiziert: Tabelle angelegt, 12 Zeilen geschrieben + zurückgelesen.
   - **Settle:** `--settle` trägt Sieger aus Jupiters `market.result` ('yes'/'no') nach
     (an echtem beendetem Match `wta-kraus-akugue` verifiziert). **Caveat:** beendete
     Events bleiben nur begrenzt gelistet → Settle innerhalb ~1–2 Tagen nach dem Match
     laufen lassen, sonst ist das Ergebnis nicht mehr abrufbar.
   - **Fehl-Match-Schutz:** Namens-Matching nachnamen-verankert + Schwelle 0.85 — fängt
     z. B. „Aziz Dougaz"→„Zizou Bergs" ab (würde sonst eine selbstbewusst-falsche faire
     Wahrscheinlichkeit erzeugen).
   - **Noch NICHT deployt** (kein systemd). Für echten Durchsatz: `--once`-Log periodisch
     (z. B. 30 min) + täglich `--settle` als Cron/systemd-Timer auf dem VPS.
3. Nach ~50–100 Matches auswerten (per SQL über `bb_TennisPaperBets`): **hatten
   unterbepreiste Seiten real positiven EV, oder frisst Adverse Selection sie auf?**
   Kalibrierung (faire P vs. realer Trefferanteil) + EV bei `edge ≥ Schwelle`.
   (G1–G5-Disziplin, OOS schlägt IS.)
4. **Erst bei grünem Paper-Test** denselben Code mit `--send` scharf schalten —
   Ausführung ist bereits voll verifiziert (Abschnitt 2).

---

## 5. Offene Punkte / Notizen
- Öffentliche Solana-RPCs (`publicnode`, `mainnet-beta`) timten zeitweise aus dem
  lokalen Netz aus → für Sends robuste Rotation/Retry nutzen; Platzieren+Cancel
  laufen ohnehin über Jupiters `/execute`-Relay (RPC nur für Verifikation/Claim nötig).
- API ist aggressiv rate-limited (429 nach ~4 schnellen Requests) → Backoff einplanen.
- `autopilot.py` (systemd `boersenbot_autopilot` auf VPS) überwacht Positionen &
  Auto-Claim; ein künftiger Tennis-Bot wäre ein separater Order-Platzier-/Storno-Loop.
