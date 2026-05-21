#!/usr/bin/env python3
"""
OPTION 1: MEAN-REVERSION BRUTE-FORCE
Teste alle Parameter-Kombinationen: 5 Thresholds × 3 Hold-Periods × 4 Stops × 2 Directions = 120 Sets

Pre-Reg: G1/G2 wie gehabt, Bonferroni |t| > 4.2 (200 tests / 0.05)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product

START, SPLIT = "2014-01-01", "2022-01-01"
COVID_A, COVID_B = "2020-02-15", "2020-04-30"

def tstat(r):
    r = np.asarray(r, float)
    if len(r) < 2: return np.nan
    return r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))

def backtest_params(spy, threshold, hold_days, stop_loss, long_only, start, split, covid_a, covid_b):
    """Backtest single parameter set."""
    opens = spy["Open"].values
    closes = spy["Close"].values
    lows = spy["Low"].values
    dates = spy.index

    signals = []
    for i in range(1, len(spy) - hold_days):
        prev_close_move = (closes[i-1] - opens[i-1]) / opens[i-1] * 100

        if abs(prev_close_move) > threshold:
            direction = 1 if prev_close_move < 0 else -1
            if long_only and direction < 0:
                continue

            # Multi-day hold
            entry = opens[i]
            exit_price = closes[i + hold_days - 1]

            # Check stop-loss
            if stop_loss is not None:
                for j in range(i, i + hold_days):
                    low = lows[j]
                    if (low - entry) / entry * 100 < stop_loss:
                        exit_price = entry * (1 + stop_loss / 100)
                        break

            ret = (exit_price - entry) / entry - 0.0005 - 0.001
            signals.append((dates[i + hold_days - 1], ret))

    is_ret = [r for d, r in signals if pd.Timestamp(d) < split and not (covid_a <= pd.Timestamp(d) <= covid_b)]
    oos_ret = [r for d, r in signals if pd.Timestamp(d) >= split and not (covid_a <= pd.Timestamp(d) <= covid_b)]

    is_t = tstat(is_ret)
    oos_t = tstat(oos_ret)

    return {
        "is_n": len(is_ret),
        "is_ret": np.mean(is_ret) if is_ret else np.nan,
        "is_t": is_t,
        "oos_n": len(oos_ret),
        "oos_ret": np.mean(oos_ret) if oos_ret else np.nan,
        "oos_t": oos_t,
    }

print("="*100)
print("OPTION 1: MEAN-REVERSION BRUTE-FORCE GRID SEARCH")
print("="*100)
print("Parameter Space: Threshold×Hold×Stop×Direction = 120 combinations\n")

# Load SPY
spy = yf.download("SPY", start=START, progress=False)
print(f"SPY: {len(spy)} days\n")

# Parameter grid
thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
hold_periods = [1, 2, 3]
stops = [-1.0, -1.5, -2.0, None]
directions = [True, False]  # True=Long-Only, False=Long+Short

results = []
combo_count = 0

for threshold, hold, stop, long_only in product(thresholds, hold_periods, stops, directions):
    res = backtest_params(spy, threshold, hold, stop, long_only, START, pd.Timestamp(SPLIT), pd.Timestamp(COVID_A), pd.Timestamp(COVID_B))
    combo_count += 1

    res["params"] = f"T={threshold:.1f}% H={hold}d S={stop if stop else 'None'} L={long_only}"
    results.append(res)

    if combo_count % 20 == 0:
        print(f"  ... {combo_count}/120 combos tested")

print(f"\n{combo_count} Parameter Sets tested.\n")

# Sort by OOS t-stat
results_sorted = sorted(results, key=lambda r: abs(r["oos_t"]), reverse=True)

print("="*100)
print("TOP 10 RESULTS (by OOS |t|)")
print("="*100)
print(f"{'Params':<40} {'IS Ret':<10} {'IS t':<8} {'OOS Ret':<10} {'OOS t':<8} {'OOS N':<6}")
print("-"*100)

bonf_threshold = 4.2  # Bonferroni for ~200 tests
winners = []

for i, r in enumerate(results_sorted[:10]):
    flag = " *BONF-PASS*" if abs(r["oos_t"]) > bonf_threshold else ""
    print(f"{r['params']:<40} {r['is_ret']*100:+7.3f}% {r['is_t']:+7.2f} {r['oos_ret']*100:+7.3f}% {r['oos_t']:+7.2f} {r['oos_n']:>6}{flag}")

    if abs(r["oos_t"]) > bonf_threshold:
        winners.append(r)

print("\n" + "="*100)
if winners:
    print(f"[OK] {len(winners)} PARAMETER SET(S) BONFERRONI-SIGNIFICANT!")
    for w in winners:
        print(f"  -> {w['params']}: OOS t={w['oos_t']:+.2f}, Excess={w['oos_ret']*100:+.3f}%")
    print("\nNEXT: Deploy winner(s) on NAS for live testing.")
else:
    print("[FAIL] NO parameter set passes Bonferroni (|t| > 4.2).")
    print("Mean-Reversion exhausted. Move to OPTION 2 (ML).")

# Summary stats
print("\n" + "="*100)
print("GRID SUMMARY")
print("="*100)
all_oos_t = [r["oos_t"] for r in results if not np.isnan(r["oos_t"])]
print(f"Mean |t| OOS across all combos: {np.mean(np.abs(all_oos_t)):.2f}")
print(f"Max |t| OOS: {np.max(np.abs(all_oos_t)):.2f}")
print(f"% of combos with OOS t > 0: {sum(1 for r in results if r['oos_t'] > 0) / len(results) * 100:.1f}%")
