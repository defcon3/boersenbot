# -*- coding: utf-8 -*-
"""weather_layall_tp.py — "Lay das GANZE Brett am Open": TP +10 % vs. Halten.

ANLASS (15.07.): Aus dem Peking-Chart (14.07., 33 °C -> 100 %) kam die These:
"Jeder Bucket-Graph faellt im Verlauf um >=10 %, dort haette der Autopilot-TP
verkauft — wenn man am Open ALLE Buckets layt, valutiert der Autopilot sie nach
und nach alle, das waeren feine 50 %." Diese Session testet das auf ALLEN
geloggten, gesettleten eq-Brettern.

Erweitert `weather_tp_vs_hold.py` (das einzelne Lays betrachtet) auf die
Brett-Ebene: Ein "Brett" = (city, var, target_date) mit seinen 9 eq-Buckets, von
denen per Konstruktion GENAU EINER YES gewinnt (sein NO-Lay verliert) und die
anderen acht NO-Lays gewinnen.

HYPOTHESE (vorab, H0): "Lay alle Buckets + TP" ist KEIN Geldautomat.
  1. Lay-all ist per No-Arbitrage eine Nullsumme (Kosten = Sum NO-Ask ~ n-1,
     Auszahlung = die n-1 NO-Gewinner) — abzueglich Vig also leicht negativ.
  2. Der +10-%-TP kann strukturell nur GEWINNER kappen; den einen Verlierer
     (NO auf den YES-Gewinner) rettet er praktisch nie, weil dessen Preis auf 0
     laeuft, nicht 10 % nach oben. -> TP macht Lay-all NOCH schlechter.
Der Peking-Fall (14.07.) war eine intraday-Ausnahme (1 von ~64), kein Beleg.

ENTSCHEIDUNGSREGEL (vorab): Verglichen wird die Rendite je 1 $ Einsatz
(kapitalgewichtet = Wallet-Verhalten). Schlaegt Halten den TP-Arm, ist der TP
auch auf Brett-Ebene falsch. Kein Parameter-Grid.

MODELL pro Brett (kind='eq', 9 Buckets):
  - Einstieg (Lay): NO-ASK (buy_no) im ERSTEN Snapshot je Bucket.
  - Settlement: Bucket k == settle_k gewinnt YES -> dessen NO-Lay verliert (0),
    die anderen NO-Lays gewinnen (1). settle_k ausserhalb der 9 -> alle gewinnen.
  - TP: verkaufe NO, sobald NO-BID (1 - buy_yes) >= Einstieg*1.10 in einem
    spaeteren Snapshot; Erloes = bid*(1-FEE). Sonst halten bis Settlement.
  - Kapital je Brett = Summe der Einstiegs-NO-Preise (1 Kontrakt je Bucket).

Zwei Varianten:
  A) LAY ALLE 9 Buckets (woertlich "ganzes Brett").
  B) LAY nur Buckets mit Einstieg < 0.909 (nur da KANN der +10-%-TP feuern).

EHRLICHE GRENZE (vorab): Snapshots sind ~taeglich (Timer 12:30 UTC), der echte
Autopilot pollt in Sekunden. Die Simulation SIEHT nur einen Bruchteil der
TP-Momente -> sie UNTERZAEHLT Ausloesungen. Wichtig fuer die Deutung: Die
Gewinner-Deckelung wird trotzdem fast vollstaendig erfasst (Gewinner-NO driftet
STETIG auf 1,0 -> im letzten Snapshot ueber +10 %), waehrend eine
Verlierer-Rettung einen FLUECHTIGEN Ausreisser braucht, den der Tageslogger
verpasst. Die 0-Rettungen-Zahl ist also eine Sicht-Obergrenze — aber die
Asymmetrie (Deckelung sicher, Rettung selten und max. 1x/Brett) macht das
Vorzeichen robust.

Aufruf: python weather_layall_tp.py
"""
import sys
from collections import defaultdict

import pymssql

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB = {"server": "158.181.48.77", "database": "dbdata", "user": "326773", "password": "Extaler11!"}

TP = 0.10        # Take-Profit-Schwelle des Autopiloten (--profit 0.10)
FEE = 0.012      # Verkaufsgebuehr, gemessen am echten Beijing-Verkauf
CEIL = 1.0 / (1.0 + TP)   # 0,909 — darueber kann der TP rechnerisch nie feuern


def sim_board(buckets, settle_k, only_tradeable):
    """buckets: dict k -> sortierte Liste[(ts, buy_yes, buy_no)]. Kennzahlen je Brett."""
    entries = {}
    for k, snaps in buckets.items():
        p0 = snaps[0][2]                      # Einstieg = NO-Ask im ersten Snapshot
        if p0 is None or not (0 < p0 < 1.0):
            continue
        if only_tradeable and p0 >= CEIL:
            continue
        entries[k] = p0
    if not entries:
        return None

    cap = sum(entries.values())
    pnl_hold = pnl_tp = 0.0
    loser_in = settle_k in entries
    loser_rescued = False
    winners_capped = 0.0        # entgangener Gewinn durch TP-Deckelung (nur Gewinner)
    rescue_gain = 0.0           # zusaetzlicher Gewinn durch geretteten Verlierer
    n_fired = 0

    for k, p0 in entries.items():
        lost = (k == settle_k)              # dieser Bucket gewinnt YES -> NO verliert
        h = (0.0 if lost else 1.0) - p0     # Halten-PnL je Kontrakt
        t, fired = h, False
        for ts, byes, bno in buckets[k][1:]:
            if byes is None:
                continue
            bid = 1.0 - byes
            if bid >= p0 * (1.0 + TP):
                t = bid * (1.0 - FEE) - p0
                fired = True
                break
        pnl_hold += h
        pnl_tp += t
        if fired:
            n_fired += 1
            if lost:
                loser_rescued = True
                rescue_gain += (t - h)
            else:
                winners_capped += (h - t)
    return {"cap": cap, "n": len(entries), "pnl_hold": pnl_hold, "pnl_tp": pnl_tp,
            "ret_hold": pnl_hold / cap, "ret_tp": pnl_tp / cap,
            "has_loser": loser_in, "loser_rescued": loser_rescued,
            "winners_capped": winners_capped, "rescue_gain": rescue_gain, "n_fired": n_fired}


def run(boards, only_tradeable, label, min_n=3, min_cap=1.0):
    res, dropped = [], 0
    for key, (buckets, settle_k) in boards.items():
        r = sim_board(buckets, settle_k, only_tradeable)
        if not r:
            continue
        if r["n"] < min_n or r["cap"] < min_cap:     # duenne/Geister-Bretter raus
            dropped += 1
            continue
        r["key"] = key
        res.append(r)
    if not res:
        print(f"\n{label}: keine auswertbaren Bretter.")
        return []

    n = len(res)
    pooled_hold = sum(r["pnl_hold"] for r in res) / sum(r["cap"] for r in res)
    pooled_tp = sum(r["pnl_tp"] for r in res) / sum(r["cap"] for r in res)
    mean_hold = sum(r["ret_hold"] for r in res) / n
    mean_tp = sum(r["ret_tp"] for r in res) / n
    with_loser = [r for r in res if r["has_loser"]]
    rescued = [r for r in with_loser if r["loser_rescued"]]

    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    print(f"  Bretter: {n} (duenne <{min_n} Buckets / <{min_cap:.0f}$ Kap. verworfen: {dropped}) "
          f"| Buckets/Brett: {sum(r['n'] for r in res)/n:.1f}")
    print(f"  Bretter mit Verlierer-Bucket im Universum: {len(with_loser)} von {n}")
    print(f"  Kapital/Brett im Schnitt: {sum(r['cap'] for r in res)/n:.2f} $ (1 Kontrakt/Bucket)\n")
    print(f"  {'Strategie':26} {'Pooled (Wallet)':>16} {'Board-Mittel':>14}")
    print("  " + "-" * 58)
    print(f"  {'HALTEN bis Settlement':26} {pooled_hold*100:+15.2f} % {mean_hold*100:+13.2f} %")
    print(f"  {'TP +10 %':26} {pooled_tp*100:+15.2f} % {mean_tp*100:+13.2f} %")
    print(f"  {'Differenz (TP - Halten)':26} {(pooled_tp-pooled_hold)*100:+15.2f} pp {(mean_tp-mean_hold)*100:+13.2f} pp")

    print("\n  --- Verlierer-Rettung durch TP (der Kern der These) ---")
    print(f"  Bretter, auf denen der TP den Verlierer-NO rechtzeitig verkaufte: "
          f"{len(rescued)} von {len(with_loser)} ({len(rescued)/max(len(with_loser),1)*100:.0f} %)")
    print(f"  Summe geretteter Gewinn:  {sum(r['rescue_gain'] for r in res):+.2f} $")
    print(f"  Summe gedeckelter Gewinn: {-sum(r['winners_capped'] for r in res):+.2f} $ "
          f"(entgangen, weil Gewinner bei +10 % verkauft)")
    print(f"  TP feuerte insgesamt {sum(r['n_fired'] for r in res)}x | "
          f"Netto TP - Halten: {sum(r['pnl_tp'] for r in res) - sum(r['pnl_hold'] for r in res):+.2f} $")
    return res


def main():
    conn = pymssql.connect(**DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT city, var, target_date, k, snapshot_utc, buy_yes, buy_no,
               COALESCE(wu_settle_k, settle_k) AS settle
        FROM bb_WeatherLadders
        WHERE kind='eq' AND buy_no IS NOT NULL AND buy_yes IS NOT NULL
          AND COALESCE(wu_settle_k, settle_k) IS NOT NULL
        ORDER BY city, var, target_date, k, snapshot_utc""")
    boards = defaultdict(lambda: [defaultdict(list), None])
    for city, var, td, k, ts, byes, bno, settle in cur.fetchall():
        key = (city, var, str(td))
        boards[key][0][k].append((ts, byes, bno))
        boards[key][1] = settle
    conn.close()

    good, snap_hist = {}, defaultdict(int)
    for key, (buckets, settle) in boards.items():
        maxsnaps = max(len(v) for v in buckets.values())
        snap_hist[maxsnaps] += 1
        if maxsnaps >= 2:                       # >=2 Snapshots noetig fuer TP-Pfad
            good[key] = (buckets, settle)

    print(f"Gesettlete eq-Bretter gesamt: {len(boards)}")
    print("Snapshot-Tiefe (max Snaps je Brett): " +
          ", ".join(f"{k}Snap:{v}" for k, v in sorted(snap_hist.items())))
    print(f"Bretter mit >=2 Snapshots (TP-Pfad vorhanden): {len(good)}")
    print(f"Zieltage: {', '.join(sorted({td for (_, _, td) in good}))}")

    run(good, False, "VARIANTE A — LAY ALLE 9 BUCKETS (woertlich 'ganzes Brett')")
    run(good, True,  "VARIANTE B — LAY nur Buckets mit Einstieg < 0.909 (TP-faehig)")


if __name__ == "__main__":
    main()
