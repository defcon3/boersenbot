#!/usr/bin/env python3
"""
eval_crypto_momentum.py — Forward-Konfirmation der 15m-Momentum-Regel
("kaufe am Range-Start die Richtung der Vorgaenger-Range, halte bis Settlement")
auf bb_CryptoUpDown15m. Pre-Reg: preregs/crypto_momentum_forward_2026_07_03.md
(eingefroren 2026-07-03, VOR den Forward-Daten).

EINGEFRORENES FENSTER: range_start_utc in [2026-07-03 00:00, 2026-07-10 00:00) UTC.
Erwartung des Registrierenden: NEGATIV (60-Tage-Binance-Anker: Fortsetzung
0.474-0.490, Markt kalibriert; die nominalen +2..+8 % ROI vom 30.06.-02.07. waren
mutmasslich eine Momentum-Welle). Gates siehe Pre-Reg: GREEN nur bei mean>0 UND
Cluster-t >= +2.0 netto UND N >= 300 Trades / >= 60 Fenster-Cluster.

PEEKING-REGEL: Zwischenstands-Laeufe vor dem 10.07. sind zulaessig (festes
Fensterende, kein optional stopping), der Gate-Entscheid faellt ausschliesslich
auf dem vollen Fenster. --start/--end dienen dem Regressionstest auf dem bereits
bekannten Explorationsfenster (30.06.-02.07.), nicht dem Verschieben der Pre-Reg.

Regel-Definition EXAKT wie crypto_updown_streaks.py Block D (momentum k>=1):
Entry = erster Tick mit secs_to_close <= 895, gueltig nur bei >= 870 (start_ok);
Kauf prev_result-Seite zum Ask (Filter 0.02 < ask < 0.98); cost = ask +
0.07*min(ask, 1-ask); Settlement-Claim gebuehrenfrei. Cluster-t auf 15-Min-Fenster
(Assets zu ~80 % synchron -> effektives N = Fenster, nicht Events).

Aufruf:
  python eval_crypto_momentum.py                                   # Pre-Reg-Fenster
  python eval_crypto_momentum.py --start 2026-06-30 --end 2026-07-03   # Regressionstest
"""

import argparse
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from crypto_updown_backtest import FEE_RATE, cluster_t, fee, load_ticks, wilson_ci
from crypto_updown_streaks import build_events

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

WINDOW_START = pd.Timestamp("2026-07-03 00:00:00")   # eingefroren (Pre-Reg)
WINDOW_END = pd.Timestamp("2026-07-10 00:00:00")     # exklusiv
MIN_TRADES = 300
MIN_WINDOWS = 60
T_GREEN = 2.0


def momentum_trades(ev, rate):
    """Ein Trade je Event: kaufe prev_result-Seite am Range-Start, halte bis
    Settlement. Identischer Codepfad wie crypto_updown_streaks.sim_report."""
    d = ev[ev.same_seg & ev.prev_result.notna() & ev.start_ok &
           ev.up_buy.notna() & ev.down_buy.notna()].copy()
    d["side"] = d.prev_result
    d["ask"] = np.where(d.side == "Up", d.up_buy, d.down_buy)
    d = d[(d.ask > 0.02) & (d.ask < 0.98)]
    d["won"] = (d.result == d.side).astype(int)
    d["mid0"] = np.where(d.side == "Up", (d.up_buy + d.up_sell) / 2,
                         (d.down_buy + d.down_sell) / 2)
    d["cost"] = d.ask + rate * np.minimum(d.ask, 1 - d.ask)
    d["pnl"] = d.won - d.cost
    d["pnl_b"] = d.won - d.ask
    return d


def line(label, d, pnl_col="pnl", cost_col="cost"):
    if len(d) == 0:
        return f"{label:>26}: keine Trades"
    t, nw = cluster_t(d[["window", pnl_col]].rename(columns={pnl_col: "pnl"}))
    roi = d[pnl_col].sum() / d[cost_col].sum()
    return (f"{label:>26}: N={len(d):>4} win {d.won.mean():.3f} O-Ask {d.ask.mean():.3f} | "
            f"PnL {d[pnl_col].sum():>+8.2f} $/K | ROI {roi * 100:>+6.1f}% | "
            f"Cluster-t {t:>+5.2f} (n={nw} Fenster)")


def main():
    ap = argparse.ArgumentParser(description="Forward-Eval Momentum-Regel (Pre-Reg 2026-07-03)")
    ap.add_argument("--fee-rate", type=float, default=FEE_RATE)
    ap.add_argument("--start", default=None, help="Override Fensterstart (nur Repro/Test)")
    ap.add_argument("--end", default=None, help="Override Fensterende exkl. (nur Repro/Test)")
    args = ap.parse_args()

    start = pd.Timestamp(args.start) if args.start else WINDOW_START
    end = pd.Timestamp(args.end) if args.end else WINDOW_END
    prereg = start == WINDOW_START and end == WINDOW_END

    print("=" * 78)
    print("FORWARD-EVAL: 15m-Momentum (Pre-Reg preregs/crypto_momentum_forward_2026_07_03.md)")
    print(f"Fenster: [{start} .. {end}) UTC | Fee-Rate {args.fee_rate}")
    if not prereg:
        print(">>> REPRO-/TESTMODUS: abweichendes Fenster, KEIN Pre-Reg-Urteil <<<")
    print("=" * 78)

    df = load_ticks()
    ev = build_events(df)                       # Voll-Historie: prev_result an der
    ev = ev[(ev.window >= start) & (ev.window < end)]   # Fenstergrenze braucht Vorgaenger
    if ev.empty:
        print("Keine Events im Fenster."); return
    print(f"Events im Fenster: {len(ev)} | {ev.window.min()} -> {ev.window.max()}")

    tr = momentum_trades(ev, args.fee_rate)
    nw = tr.window.nunique()

    print("\nPRIMAER (alle Assets gepoolt, k>=1, netto):")
    print(line("momentum", tr))

    print("\nSEKUNDAER (deskriptiv, keine Entscheidungsgrundlage):")
    print(line("brutto (Fee 0)", tr, pnl_col="pnl_b", cost_col="ask"))
    for k in (2, 3):
        print(line(f"prev_streak >= {k}", tr[tr.prev_streak >= k]))
    for prev in ("Up", "Down"):
        print(line(f"nach {prev} -> kaufe {prev}", tr[tr.prev_result == prev]))
    for asset, g in tr.groupby("asset"):
        print(line(f"{asset}", g))

    cont_k, cont_n = int(tr.won.sum()), len(tr)
    lo, hi = wilson_ci(cont_k, cont_n)
    e = tr.assign(pnl=tr.won - tr.mid0)[["window", "pnl"]]
    t_cal, _ = cluster_t(e)
    print(f"\nFortsetzungsrate: {cont_k}/{cont_n} = {cont_k / cont_n:.3f} "
          f"[Wilson {lo:.3f}, {hi:.3f}] (CI ignoriert Asset-Synchronitaet)")
    print(f"Kalibrierung won - Start-Mid: {(tr.won - tr.mid0).mean():+.4f} | Cluster-t {t_cal:+.2f}")

    print("\nPnL je UTC-Tag (netto):")
    for day, g in tr.groupby(tr.window.dt.date):
        t_d, nw_d = cluster_t(g[["window", "pnl"]])
        print(f"  {day}: N={len(g):>4} win {g.won.mean():.3f} "
              f"PnL {g.pnl.sum():>+7.2f} t {t_d:>+5.2f} ({nw_d} Fenster)")

    # ------------------------------------------------------------ Gate-Block
    print("\n" + "=" * 78)
    print("GATE-AUSWERTUNG (Pre-Reg 2026-07-03)")
    print("=" * 78)
    mean_pnl = tr.pnl.mean() if len(tr) else 0.0
    t_prim, _ = cluster_t(tr[["window", "pnl"]])
    g_n = len(tr) >= MIN_TRADES and nw >= MIN_WINDOWS
    g_prim = mean_pnl > 0 and t_prim >= T_GREEN
    print(f"  G-N     : N={len(tr)} (>= {MIN_TRADES}) & Fenster={nw} (>= {MIN_WINDOWS}) "
          f"-> {'PASS' if g_n else 'FAIL'}")
    print(f"  G-Primaer: mean {mean_pnl:+.4f} $/K (> 0) & Cluster-t {t_prim:+.2f} "
          f"(>= +{T_GREEN}) -> {'PASS' if g_prim else 'FAIL'}")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if not prereg:
        print("  ERGEBNIS: — (Testmodus, kein Pre-Reg-Fenster)")
    elif now < WINDOW_END:
        print(f"  ERGEBNIS: ZWISCHENSTAND — Fenster laeuft bis {WINDOW_END}, "
              "kein Gate-Entscheid")
    elif not g_n:
        print("  ERGEBNIS: UNDERPOWERED — kein PASS/FAIL-Urteil (siehe Pre-Reg)")
    elif g_prim:
        print("  ERGEBNIS: GREEN — naechste Stufe waere Paper-Trade-Pre-Reg, KEIN Livegang")
    else:
        print("  ERGEBNIS: RED — Momentum-Regel nicht konfirmiert, Kapitel zu")


if __name__ == "__main__":
    main()
