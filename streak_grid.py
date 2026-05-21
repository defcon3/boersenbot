#!/usr/bin/env python3
"""Streak-Grid: Hit-Quote + Mean(next-day return) je Streak-Laenge N=1..10+.

Datenfundament wie crossover_grid.py: crossover_universe.pkl, 502 S&P-500-
Ticker, Tages-OHLC ab 2014. Erzeugt streak_grid.json fuer die Web-Seite
/analysis/streak.

Streak: Anzahl aufeinanderfolgender Tage mit Close-to-Close < 0, gemessen
BIS einschliesslich Tag t. Frage: Was passiert am Folgetag (t+1, Close-to-Close)?
Vergleich gegen unbedingte Basis P(next up) ueber alle Aktien-Tage.

Histogramm der next-day-returns: 80 Bins ueber [-15%, +15%].
"""
import warnings; warnings.filterwarnings("ignore")
import json, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd

CACHE_FILE = Path("/home/veit/boersenbot/crossover_universe.pkl")
OUT        = Path("/home/veit/boersenbot/streak_grid.json")
N_MAX      = 10
HIST_BINS  = 80
HIST_RANGE = (-15.0, 15.0)


def stats_for(arr_pct):
    if len(arr_pct) == 0:
        return {"n": 0}
    return {
        "n":       int(len(arr_pct)),
        "mean":    round(float(np.mean(arr_pct)), 4),
        "median":  round(float(np.median(arr_pct)), 4),
        "q10":     round(float(np.quantile(arr_pct, 0.1)), 4),
        "q90":     round(float(np.quantile(arr_pct, 0.9)), 4),
        "hit_pct": round(float(np.mean(arr_pct > 0) * 100), 2),
    }


def hist_for(arr_pct):
    counts, edges = np.histogram(arr_pct, bins=HIST_BINS, range=HIST_RANGE)
    return {
        "counts": counts.tolist(),
        "edges":  [round(float(x), 3) for x in edges.tolist()],
    }


def main():
    print("Lade Cache ...", flush=True)
    with open(CACHE_FILE, "rb") as f:
        ohlc = pickle.load(f)
    print(f"  {len(ohlc)} Ticker", flush=True)

    base_n = 0
    base_up = 0
    base_sum = 0.0
    base_arr = []          # next-day-returns aller Aktien-Tage (in %)
    buckets = {N: [] for N in range(1, N_MAX + 1)}  # next-day-ret (%) je N

    t0 = time.time()
    for i, (sym, df) in enumerate(ohlc.items()):
        c = df["close"].astype(float).values.ravel()
        n = len(c)
        if n < 30:
            continue
        ret = np.empty(n); ret[0] = np.nan
        ret[1:] = c[1:] / c[:-1] - 1.0
        down = ret < 0
        streak = np.zeros(n, int)
        for k in range(1, n):
            streak[k] = streak[k-1] + 1 if down[k] else 0
        next_ret = np.append(ret[1:], np.nan)    # next-day Close-to-Close
        valid = np.isfinite(next_ret)
        base_n  += int(valid.sum())
        base_up += int(np.sum(next_ret[valid] > 0))
        base_sum += float(np.nansum(next_ret[valid]))
        base_arr.extend((next_ret[valid] * 100.0).tolist())
        for N in range(1, N_MAX + 1):
            if N < N_MAX:
                mask = (streak == N) & valid
            else:
                mask = (streak >= N) & valid
            if not mask.any():
                continue
            buckets[N].extend((next_ret[mask] * 100.0).tolist())
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ohlc)} ({time.time()-t0:.0f}s)", flush=True)

    base_arr = np.asarray(base_arr)
    print(f"\nBasis: n={base_n:,}  P(next up)={base_up/base_n:.4f}  "
          f"mean={base_sum/base_n*100:+.3f}%", flush=True)

    out = {
        "params": {
            "stk_start": "2014-01-01",
            "n_max":     N_MAX,
            "hist_bins": HIST_BINS,
            "hist_range_pct": list(HIST_RANGE),
        },
        "baseline": {
            "stats": stats_for(base_arr),
            "hist":  hist_for(base_arr),
        },
        "grid": {},
    }
    for N in range(1, N_MAX + 1):
        a = np.asarray(buckets[N])
        if len(a) == 0:
            out["grid"][str(N)] = {"n": 0}
            continue
        out["grid"][str(N)] = {
            "N": N,
            "label": (f">={N}" if N == N_MAX else str(N)),
            "stats": stats_for(a),
            "hist":  hist_for(a),
            "edge_hit_pp":  round(stats_for(a)["hit_pct"]
                                  - out["baseline"]["stats"]["hit_pct"], 3),
            "edge_mean_pp": round(stats_for(a)["mean"]
                                  - out["baseline"]["stats"]["mean"], 4),
        }

    print("\n--- Roh-Befund je N ---")
    print(f"{'N':>3} {'n':>9} {'Hit %':>7} {'Mean %':>8} {'Δ hit pp':>10}")
    base_hit = out["baseline"]["stats"]["hit_pct"]
    for N in range(1, N_MAX + 1):
        e = out["grid"][str(N)]
        if e.get("n", 1) == 0 or "stats" not in e: continue
        print(f"{e['label']:>3} {e['stats']['n']:>9,} "
              f"{e['stats']['hit_pct']:>7.2f} "
              f"{e['stats']['mean']:>+8.3f} "
              f"{e['edge_hit_pp']:>+10.2f}")

    with open(OUT, "w") as f:
        json.dump(out, f)
    print(f"\nGeschrieben: {OUT}  ({OUT.stat().st_size/1e3:.1f} KB)")


if __name__ == "__main__":
    main()
