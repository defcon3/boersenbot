# Trade-Registrierung: Wimbledon-Herrenfinale In-Play-Scalp (Sinner)

**Registriert:** 2026-07-11 ~21:25 UTC — VOR dem Match (Finale 12.07. 15:00 UTC).
**Gesetzt** (Nutzer-Auftrag „setz sinner mit 10 geld"):
**POLY-2874512-0 „Jannik Sinner" [YES] @ 0,81 — 11,69 Kontrakte, Kosten 9,47 $**,
Sig `3EE97U56…`, Fill exakt am Ask (Limit 0,83, kein Slippage).

## Charakter des Trades (ehrlich): Mechanik-Scalp, KEIN Value-Claim

Es gibt keine belastbaren 2026-Formdaten in dieser Session und die
Wimbledon-Lektion vom 02.07. steht („Markt > stale Elo", Elo-Picks 1/5,
ROI −42 %). Der Markt-Preis 0,81/0,20 (Overround ~1 %, liquidester
Wimbledon-Markt des Scans) wird als fair AKZEPTIERT. Die These ist rein
mechanisch: Tennis-Märkte handeln auf Jupiter IN-PLAY (closeTime 19.07.,
verifiziert — im Gegensatz zu Fußball, wo closeTime=Anpfiff den
Norwegen–England-LTD am 11.07. killte). Der Autopilot (`boersenbot_autopilot`,
+10 % NETTO-Trigger mit Orderbuch-Simulation, Commit fe7225ec) verkauft
autonom bei Nettoerlös ≥ ~10,42 $ ≈ Preis ~0,90 — das entspricht ungefähr
„Sinner gewinnt Satz 1".

## Ex-ante-Szenarien

| Szenario | Pfad | PnL |
|---|---|---|
| Sinner gewinnt Satz 1 → Preis ≥0,90 | Autopilot-Exit +10 % netto | **≈ +0,95 $** |
| Satz 1 an Zverev, Sinner dreht | Hold (kein SL) → Resolution | +2,22 $ |
| Zverev gewinnt | Hold → Totalverlust | **−9,47 $** |
| Spread bläht sich im Kipp-Moment (~14 ¢, bekannte In-Play-Falle) | Netto-Check verweigert Verkauf → Hold | ±(s. o.) |

Bekannte Risiken der Automatik (aus früheren Sessions, bewusst akzeptiert):
(1) Polling nur alle 180 s (closeTime fern → `is_imminent` greift nie);
Mensik/Shelton-TPs feuerten trotzdem. (2) Kein Stop-Loss by design.

## Auswertung (nach dem Finale nachtragen)

Exit-Preis + Netto-PnL aus `/history` (nicht autopilot.log-Brutto!);
prüfen, ob der 180s-Poll den Satz-1-Spike erwischt hat. Einzeltrade,
N=1 — zählt in keine Edge-Bilanz, reine Ausführungs-Doku.

Kontext-Scan 11.07. (`subcategory=wimbledon`, 3 Events): Herren-Finale
(einzig liquide), Damen-Doppel 0,33/0,70, Junioren Lee/Hewitt 0,64+0,63 =
27 % Overround → als Taker unhandelbar.
