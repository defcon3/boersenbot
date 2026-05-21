#!/usr/bin/env python3
"""
ALTERNATIVE STRESS-INDICES TEST

Hypothesen:
  S1: NFCI (Chicago Fed National Financial Conditions Index)
      > Train-Q75 -> SPY/QQQ/IWM Forward-Returns niedriger
  S2: STLFSI4 (St. Louis Fed Financial Stress Index)
      > Train-Q75 -> SPY/QQQ/IWM Forward-Returns niedriger

Beide haben volle Historie auf FRED:
  NFCI: 1971+ (weekly)
  STLFSI4: 1993+ (weekly)

Pre-Reg:
  Train: 2000-2018
  Test: 2019-2025 (ohne COVID 2020-02-15 bis 2020-04-30)
  Bonferroni: 2 x 4 x 3 = 24 Tests -> |t| > 3.0
  Sign-Match: Train und Test gleiches Vorzeichen erforderlich
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from fred_helper import get_series, get_series_meta

# CONFIG
TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")
COVID_A = pd.Timestamp("2020-02-15")
COVID_B = pd.Timestamp("2020-04-30")

BONFERRONI_T = 3.0
NAIVE_T = 1.5

# HELPERS
def fetch_returns(ticker, start="2000-01-01"):
    df = yf.download(ticker, start=start, progress=False)
    close = np.asarray(df['Close']).flatten()
    return pd.Series(close, index=df.index).pct_change()

def welch_t(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return np.nan, np.nan
    se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    if se == 0: return np.nan, np.nan
    return (a.mean() - b.mean()) / se, a.mean() - b.mean()

def run_test(signal, returns, fwd_days, train_end=TRAIN_END):
    """Generic Test mit Train-Q75-Threshold (kein Snooping)."""
    signal_daily = signal.reindex(returns.index, method='ffill')

    # Q75 NUR aus Train-Period
    train_signal = signal_daily[signal_daily.index <= train_end]
    q75 = train_signal.dropna().quantile(0.75)
    condition = signal_daily > q75

    fwd_ret = returns.rolling(fwd_days).sum().shift(-fwd_days)

    train_mask = (fwd_ret.index <= train_end) & ~((fwd_ret.index >= COVID_A) & (fwd_ret.index <= COVID_B))
    test_mask = (fwd_ret.index >= TEST_START) & ~((fwd_ret.index >= COVID_A) & (fwd_ret.index <= COVID_B))

    train_sig = fwd_ret[train_mask & condition].dropna()
    train_no = fwd_ret[train_mask & ~condition].dropna()
    test_sig = fwd_ret[test_mask & condition].dropna()
    test_no = fwd_ret[test_mask & ~condition].dropna()

    t_train, diff_train = welch_t(train_sig, train_no)
    t_test, diff_test = welch_t(test_sig, test_no)

    sign_match = (np.sign(diff_train) == np.sign(diff_test)) if not np.isnan(diff_train) else False

    return {
        "q75": q75,
        "t_train": t_train, "diff_train": diff_train, "n_train": len(train_sig),
        "t_test": t_test, "diff_test": diff_test, "n_test": len(test_sig),
        "sign_match": sign_match
    }

# DATA LOAD
print("="*80)
print("ALTERNATIVE STRESS-INDICES TEST")
print("="*80)

print("\n[1/4] Lade Stress-Indices...", flush=True)

# NFCI
nfci = get_series("NFCI", start="2000-01-01")
nfci_meta = get_series_meta("NFCI")
print(f"  NFCI:    {len(nfci):5d} Werte ({nfci.index.min().date()} - {nfci.index.max().date()})")
print(f"    Title: {nfci_meta['title']}")
print(f"    Freq: {nfci_meta['frequency']}")

# STLFSI4
stlfsi = get_series("STLFSI4", start="2000-01-01")
stlfsi_meta = get_series_meta("STLFSI4")
print(f"\n  STLFSI4: {len(stlfsi):5d} Werte ({stlfsi.index.min().date()} - {stlfsi.index.max().date()})")
print(f"    Title: {stlfsi_meta['title']}")
print(f"    Freq: {stlfsi_meta['frequency']}")

# Correlation
nfci_aligned = nfci.reindex(stlfsi.index, method='ffill').dropna()
common = stlfsi.dropna().index.intersection(nfci_aligned.index)
if len(common) > 0:
    corr = np.corrcoef(stlfsi.loc[common], nfci_aligned.loc[common])[0,1]
    print(f"\n  NFCI vs STLFSI4 Korrelation: {corr:.3f}")

print("\n[2/4] Lade ETF-Returns...", flush=True)
assets = {}
for ticker in ["SPY", "QQQ", "IWM"]:
    assets[ticker] = fetch_returns(ticker)
print(f"  {len(assets)} Assets geladen")

# TESTS
print("\n[3/4] Lasse Tests laufen (2 indices x 4 windows x 3 assets = 24 tests)...", flush=True)

windows = [5, 10, 20, 60]
results = []

for idx_name, idx_series in [("NFCI", nfci), ("STLFSI4", stlfsi)]:
    for asset_name, ret in assets.items():
        for fwd in windows:
            r = run_test(idx_series, ret, fwd)
            results.append({
                "index": idx_name, "asset": asset_name, "fwd": fwd, **r
            })

# OUTPUT
print("\n[4/4] Auswertung...\n")
print("="*108)
print(f"{'Index':<8} {'Asset':<5} {'Fwd':>4}  {'Q75':>6}  {'TrDiff%':>8} {'tTrain':>7}  {'TeDiff%':>8} {'tTest':>7}  {'Sign?':<5}  {'Naive':<6}  {'Bonf':<6}  {'nTest':>6}")
print("="*108)

for r in results:
    naive = abs(r['t_test']) > NAIVE_T and r['n_test'] >= 100 and r['sign_match']
    bonf  = abs(r['t_test']) > BONFERRONI_T and r['n_test'] >= 100 and r['sign_match']
    print(f"{r['index']:<8} {r['asset']:<5} {r['fwd']:>4}  "
          f"{r['q75']:>6.3f}  "
          f"{r['diff_train']*100:>+8.3f} {r['t_train']:>+7.2f}  "
          f"{r['diff_test']*100:>+8.3f} {r['t_test']:>+7.2f}  "
          f"{'YES' if r['sign_match'] else 'NO':<5}  "
          f"{'PASS' if naive else 'FAIL':<6}  "
          f"{'PASS' if bonf else 'FAIL':<6}  "
          f"{r['n_test']:>6}")

# AGGREGATE
naive_pass = sum(1 for r in results if abs(r['t_test']) > NAIVE_T and r['n_test'] >= 100 and r['sign_match'])
bonf_pass = sum(1 for r in results if abs(r['t_test']) > BONFERRONI_T and r['n_test'] >= 100 and r['sign_match'])

print("\n" + "="*108)
print(f"  Naive (|t|>{NAIVE_T}, n>=100, sign-match): {naive_pass}/{len(results)} PASS")
print(f"  Bonferroni (|t|>{BONFERRONI_T}, n>=100, sign-match): {bonf_pass}/{len(results)} PASS")

# Cross-Asset Konsistenz
print("\n" + "="*108)
print("CROSS-ASSET KONSISTENZ pro Index-Window-Kombo:")
print("="*108)
df = pd.DataFrame(results)
for idx in ["NFCI", "STLFSI4"]:
    for fwd in windows:
        sub = df[(df['index']==idx) & (df['fwd']==fwd)]
        signs = np.sign(sub['diff_test'].values)
        same = abs(signs.sum()) == len(signs)
        ts = sub['t_test'].values
        diffs = sub['diff_test'].values * 100
        print(f"  {idx:<8} fwd={fwd:3d}d  "
              f"SPY: diff={diffs[0]:+.3f}%/t={ts[0]:+.2f}  "
              f"QQQ: diff={diffs[1]:+.3f}%/t={ts[1]:+.2f}  "
              f"IWM: diff={diffs[2]:+.3f}%/t={ts[2]:+.2f}  "
              f"Sign-Cons: {'YES' if same else 'NO'}")

# VERDICT
print("\n" + "="*108)
print("VERDICT")
print("="*108)
if bonf_pass == 0:
    print("  Alle Tests Bonferroni-FAIL -> Stress-Indices liefern keinen robusten Edge.")
elif bonf_pass <= 2:
    print(f"  {bonf_pass}/24 Bonferroni-PASS - isolierte Hits, kein universelles Pattern.")
else:
    print(f"  {bonf_pass}/24 Bonferroni-PASS - potentieller Edge, weiter analysieren!")

# Probability of random PASS
import math
n_tests = len(results)
expected_random = n_tests * (1 - 0.99865)  # |t|>3 ~ p=0.00135 two-tailed
print(f"\n  Bei {n_tests} Tests und |t|>3.0 erwartet zufaellig: ~{expected_random:.2f} PASSes")
print(f"  Beobachtet: {bonf_pass} -> {'im Zufallsbereich' if bonf_pass <= expected_random + 1 else 'ueber Zufall'}")
