# -*- coding: utf-8 -*-
"""weather_tp_vs_hold.py — Kostet der Autopilot-Take-Profit Rendite oder rettet er?

ANLASS: Der Beijing-33-Lay (14.07.) haette −5,67 $ verloren; nur der TP
(autopilot.py --profit 0.10, KEIN Stop-Loss) rettete ihn mit +0,51 $. Damit steht
die Frage im Raum, ob die Regel systematisch hilft oder schadet.

HYPOTHESE (vor dem Lauf festgelegt): Der TP ist gegenueber Halten-bis-Settlement
EV-NEGATIV. Begruendung: Er verkauft zu einem Preis, den unser Modell noch fuer zu
niedrig haelt, kappt die Rendite bei +10 %, waehrend die volle Lay-Rendite +20-27 %
betraegt — und er greift NIE bei Verlierern, weil deren Preis nie 10 % zu unseren
Gunsten laeuft. Gewinner werden also gedeckelt, Verlierer bleiben in voller Groesse.

ENTSCHEIDUNGSREGEL (vorab): Verglichen wird der mittlere PnL je 1 $ Einsatz.
  TP-Arm   schlaegt Hold-Arm  -> Regel behalten
  Hold-Arm schlaegt TP-Arm    -> TP kostet Edge, Abschaltung pruefen
Kein Parameter-Grid, keine Nachjustierung der 10-%-Schwelle.

DATEN: bb_WeatherLadders (Preisleitern-Snapshots seit 11.07., taeglicher Timer
12:30 UTC) + wu_settle_k (Wunderground = Polymarket-Settlement-Quelle).
Universum: alle NO-Lays mit >=2 Preispunkten, bekanntem Ausgang und Einstiegspreis
< 0,909 — nur dort KANN ein +10-%-TP ueberhaupt feuern (bei NO 0,95 ist der
maximale Gewinn 5,3 %).

EHRLICHE GRENZE (vorab benannt, nicht hinterher): Die Snapshots sind ~taeglich,
der echte Autopilot pollt alle 20-90 s. Die Simulation SIEHT also nur einen
Bruchteil der Momente, in denen der TP haette feuern koennen -> sie UNTERZAEHLT
TP-Ausloesungen. Der gemessene TP-Effekt ist damit betragsmaessig eine Untergrenze.
Was hier NICHT behauptet wird: eine praezise EV-Zahl. Was gezeigt wird: das
VORZEICHEN und die Groessenordnung.

Aufruf: python weather_tp_vs_hold.py
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
FEE = 0.012      # Verkaufsgebuehr, gemessen am echten Beijing-Verkauf (0,0758 $ auf 6,32 $ brutto)
MAX_ENTRY = 1.0 / (1.0 + TP)   # 0,909 — darueber kann der TP rechnerisch nie feuern


def hit(kind, k, settle):
    """Ist der Bucket getroffen worden? (dann verliert der NO-Lay)"""
    if kind == "le":
        return settle <= k
    if kind == "ge":
        return settle >= k
    return settle == k


def main():
    conn = pymssql.connect(**DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT market_id, target_date, city, var, kind, k, snapshot_utc,
               buy_yes, buy_no, wu_settle_k, settle_k
        FROM bb_WeatherLadders
        WHERE buy_no IS NOT NULL AND buy_yes IS NOT NULL
          AND COALESCE(wu_settle_k, settle_k) IS NOT NULL
        ORDER BY market_id, target_date, snapshot_utc""")
    paths = defaultdict(list)
    meta = {}
    for mid, td, city, var, kind, k, ts, byes, bno, wu, me in cur.fetchall():
        key = (mid, td)
        paths[key].append((ts, byes, bno))
        meta[key] = (city, var, kind, k, wu if wu is not None else me)
    conn.close()

    trades = []
    for key, snaps in paths.items():
        if len(snaps) < 2:
            continue
        city, var, kind, k, settle = meta[key]
        p0 = snaps[0][2]                       # Einstieg: NO-Ask im ersten Snapshot
        if not (0 < p0 < MAX_ENTRY):
            continue
        getroffen = hit(kind, k, settle)

        # Hold: 1,0 wenn der Bucket verfehlt wird, sonst 0
        pnl_hold = (0.0 if getroffen else 1.0) - p0

        # TP: verkaufen, sobald der NO-BID (= 1 - buy_yes, Spread beruecksichtigt)
        # mindestens 10 % ueber dem Einstieg liegt. Sonst halten.
        pnl_tp, gefeuert, exit_px = pnl_hold, False, None
        for ts, byes, bno in snaps[1:]:
            bid = 1.0 - byes
            if bid >= p0 * (1.0 + TP):
                exit_px = bid
                pnl_tp = bid * (1.0 - FEE) - p0
                gefeuert = True
                break
        trades.append({"city": city, "var": var, "k": k, "kind": kind, "p0": p0,
                       "getroffen": getroffen, "hold": pnl_hold, "tp": pnl_tp,
                       "gefeuert": gefeuert, "exit": exit_px, "td": key[1]})

    if not trades:
        sys.exit("Keine auswertbaren Lays.")

    n = len(trades)
    fired = [t for t in trades if t["gefeuert"]]
    lost = [t for t in trades if t["getroffen"]]
    # PnL je 1 $ EINSATZ (nicht je Kontrakt) -> vergleichbar ueber Preise hinweg
    r_hold = sum(t["hold"] / t["p0"] for t in trades) / n
    r_tp = sum(t["tp"] / t["p0"] for t in trades) / n

    print("=" * 88)
    print("TP (+10 %, kein Stop-Loss) vs. HALTEN BIS SETTLEMENT")
    print("=" * 88)
    print(f"  Universum: {n} NO-Lays (Einstieg < {MAX_ENTRY:.3f}, >=2 Preispunkte, Ausgang bekannt)")
    print(f"  Zieltage: {', '.join(sorted({str(t['td']) for t in trades}))}")
    print(f"  Bucket getroffen (= Lay verloren): {len(lost)} von {n} ({len(lost)/n*100:.0f} %)")
    print(f"  TP haette gefeuert bei: {len(fired)} von {n} ({len(fired)/n*100:.0f} %)")
    print(f"  Verkaufsgebuehr angesetzt: {FEE*100:.1f} % (gemessen am echten Beijing-Verkauf)\n")

    print(f"  {'Strategie':28} {'Rendite je 1 $ Einsatz':>24}")
    print("  " + "-" * 54)
    print(f"  {'HALTEN bis Settlement':28} {r_hold*100:+23.2f} %")
    print(f"  {'TP +10 %':28} {r_tp*100:+23.2f} %")
    print(f"  {'Differenz (TP - Halten)':28} {(r_tp-r_hold)*100:+23.2f} pp")

    print("\n  --- Was der TP genau tut ---")
    if fired:
        gew = [t for t in fired if not t["getroffen"]]
        ver = [t for t in fired if t["getroffen"]]
        print(f"  Von den {len(fired)} TP-Ausloesungen waeren beim Halten")
        print(f"    {len(gew)} zu GEWINNERN geworden -> TP kappte im Schnitt "
              f"{sum((t['hold']-t['tp'])/t['p0'] for t in gew)/max(len(gew),1)*100:+.1f} pp")
        print(f"    {len(ver)} zu VERLIERERN geworden -> TP rettete im Schnitt "
              f"{sum((t['tp']-t['hold'])/t['p0'] for t in ver)/max(len(ver),1)*100:+.1f} pp")
    nf = [t for t in trades if not t["gefeuert"]]
    nfl = [t for t in nf if t["getroffen"]]
    print(f"  Von den {len(nf)} NICHT ausgeloesten Lays verloren {len(nfl)} "
          f"({len(nfl)/max(len(nf),1)*100:.0f} %) — der TP hat sie NICHT geschuetzt.")

    print("\n  --- Nach Einstiegspreis (billigere Lays = hoehere Rendite = mehr Risiko) ---")
    print(f"  {'Einstieg':>12} {'n':>4} {'Verlust%':>9} {'Halten':>9} {'TP':>9} {'Diff':>9}")
    print("  " + "-" * 56)
    for lo, hi in ((0.0, 0.75), (0.75, 0.83), (0.83, 0.909)):
        sub = [t for t in trades if lo <= t["p0"] < hi]
        if not sub:
            continue
        h = sum(t["hold"] / t["p0"] for t in sub) / len(sub)
        p = sum(t["tp"] / t["p0"] for t in sub) / len(sub)
        lo_ = sum(1 for t in sub if t["getroffen"]) / len(sub)
        print(f"  {lo:.2f}-{hi:.3f} {len(sub):4d} {lo_*100:8.0f}% {h*100:+8.1f}% {p*100:+8.1f}% {(p-h)*100:+8.1f}pp")


if __name__ == "__main__":
    main()
