#!/usr/bin/env python3
"""
HOLDOUT-TEST: Stress-Index auf Risk-Off-Assets

PRE-REG (vor dem Test fixiert):
  Hypothese: STLFSI4 > 0.459 (Train-Q75) -> Risk-Off-Asset Forward-Returns abnormal

Holdout-Assets (nie zuvor in unseren Tests):
  - TLT (20+Y Treasuries) - Flight-to-Quality
  - IEF (7-10Y Treasuries) - safe haven
  - HYG (High-Yield Corp Bonds) - junk
  - GLD (Gold) - safe haven

Train: 2003-2018 (TLT-Inception 2002-07)
Test:  2019-2025 (ohne COVID 2020-02-15 bis 2020-04-30)

Bonferroni: 4 Assets x 4 Windows = 16 Tests -> |t| > 3.0
Sign-Match: Train + Test gleiches Vorzeichen

Klassische Hypothese:
  TLT/IEF/GLD: Stress -> POSITIVE Forward-Returns (Flight-to-Quality)
  HYG:         Stress -> NEGATIVE Forward-Returns (Risk-Off Junk)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from fred_helper import get_series

TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")
COVID_A = pd.Timestamp("2020-02-15")
COVID_B = pd.Timestamp("2020-04-30")

BONFERRONI_T = 3.0
NAIVE_T = 1.5

def fetch_returns(ticker, start="2003-01-01"):
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

def run_test(signal, returns, fwd_days, q75):
    signal_daily = signal.reindex(returns.index, method='ffill')
    condition = signal_daily > q75
    fwd_ret = returns.rolling(fwd_days).sum().shift(-fwd_days)

    train_mask = (fwd_ret.index <= TRAIN_END) & ~((fwd_ret.index >= COVID_A) & (fwd_ret.index <= COVID_B))
    test_mask = (fwd_ret.index >= TEST_START) & ~((fwd_ret.index >= COVID_A) & (fwd_ret.index <= COVID_B))

    train_sig = fwd_ret[train_mask & condition].dropna()
    train_no = fwd_ret[train_mask & ~condition].dropna()
    test_sig = fwd_ret[test_mask & condition].dropna()
    test_no = fwd_ret[test_mask & ~condition].dropna()

    t_train, diff_train = welch_t(train_sig, train_no)
    t_test, diff_test = welch_t(test_sig, test_no)
    sign_match = (np.sign(diff_train) == np.sign(diff_test)) if not np.isnan(diff_train) else False

    return {
        "t_train": t_train, "diff_train": diff_train, "n_train": len(train_sig),
        "t_test": t_test, "diff_test": diff_test, "n_test": len(test_sig),
        "sign_match": sign_match
    }

print("="*80)
print("HOLDOUT TEST: Stress-Index auf Risk-Off-Assets")
print("="*80)

print("\n[1/3] Lade STLFSI4 + Holdout-Assets...", flush=True)

stlfsi = get_series("STLFSI4", start="2003-01-01")
print(f"  STLFSI4: {len(stlfsi)} Werte")

# Q75 aus Train-Period
train_signal = stlfsi[stlfsi.index <= TRAIN_END]
q75 = train_signal.dropna().quantile(0.75)
print(f"  Train-Q75 (Schwelle): {q75:.4f}")

assets = {}
for ticker in ["TLT", "IEF", "HYG", "GLD"]:
    assets[ticker] = fetch_returns(ticker)
    print(f"  {ticker}: {len(assets[ticker])} Tage ({assets[ticker].index.min().date()} - {assets[ticker].index.max().date()})")

print("\n[2/3] Lasse Tests laufen (4 assets x 4 windows = 16 tests)...", flush=True)
windows = [5, 10, 20, 60]
results = []
for ticker, ret in assets.items():
    for fwd in windows:
        r = run_test(stlfsi, ret, fwd, q75)
        results.append({"asset": ticker, "fwd": fwd, **r})

print("\n[3/3] Auswertung...\n")
print("="*108)
print(f"{'Asset':<5} {'Fwd':>4}  {'TrDiff%':>8} {'tTrain':>7}  {'TeDiff%':>8} {'tTest':>7}  {'Sign?':<5}  {'Naive':<6}  {'Bonf':<6}  {'nTest':>6}")
print("="*108)

for r in results:
    naive = abs(r['t_test']) > NAIVE_T and r['n_test'] >= 30 and r['sign_match']
    bonf  = abs(r['t_test']) > BONFERRONI_T and r['n_test'] >= 30 and r['sign_match']
    print(f"{r['asset']:<5} {r['fwd']:>4}  "
          f"{r['diff_train']*100:>+8.3f} {r['t_train']:>+7.2f}  "
          f"{r['diff_test']*100:>+8.3f} {r['t_test']:>+7.2f}  "
          f"{'YES' if r['sign_match'] else 'NO':<5}  "
          f"{'PASS' if naive else 'FAIL':<6}  "
          f"{'PASS' if bonf else 'FAIL':<6}  "
          f"{r['n_test']:>6}")

naive_pass = sum(1 for r in results if abs(r['t_test']) > NAIVE_T and r['n_test'] >= 30 and r['sign_match'])
bonf_pass = sum(1 for r in results if abs(r['t_test']) > BONFERRONI_T and r['n_test'] >= 30 and r['sign_match'])

print("\n" + "="*108)
print(f"  Naive (|t|>{NAIVE_T}, n>=30, sign-match): {naive_pass}/{len(results)} PASS")
print(f"  Bonferroni (|t|>{BONFERRONI_T}, n>=30, sign-match): {bonf_pass}/{len(results)} PASS")

# Cross-Window Konsistenz per Asset
print("\n" + "="*108)
print("KONSISTENZ pro Asset ueber alle Forward-Windows:")
print("="*108)
df = pd.DataFrame(results)
for asset in ["TLT", "IEF", "HYG", "GLD"]:
    sub = df[df['asset']==asset]
    train_signs = np.sign(sub['diff_train'].values)
    test_signs = np.sign(sub['diff_test'].values)
    train_consistent = abs(train_signs.sum()) == len(train_signs)
    test_consistent = abs(test_signs.sum()) == len(test_signs)
    print(f"  {asset}: Train-Sign-Cons: {'YES' if train_consistent else 'NO'} ({train_signs}) | "
          f"Test-Sign-Cons: {'YES' if test_consistent else 'NO'} ({test_signs})")

print("\n" + "="*108)
print("INTERPRETATION")
print("="*108)
print("Klassische Theorie: Stress -> TLT/IEF/GLD positiv, HYG negativ")
print("Contrarian-These (aus 2019-2025): Stress -> alle positiv (mean-reversion)")

# Direction-Test
tlt_train_sign = np.sign(np.mean([r['diff_train'] for r in results if r['asset']=='TLT' and not np.isnan(r['diff_train'])]))
tlt_test_sign = np.sign(np.mean([r['diff_test'] for r in results if r['asset']=='TLT' and not np.isnan(r['diff_test'])]))
print(f"\nTLT (Treasuries): Train Mean-Direction={'+' if tlt_train_sign>0 else '-'} | Test Mean-Direction={'+' if tlt_test_sign>0 else '-'}")

hyg_train_sign = np.sign(np.mean([r['diff_train'] for r in results if r['asset']=='HYG' and not np.isnan(r['diff_train'])]))
hyg_test_sign = np.sign(np.mean([r['diff_test'] for r in results if r['asset']=='HYG' and not np.isnan(r['diff_test'])]))
print(f"HYG (Junk):       Train Mean-Direction={'+' if hyg_train_sign>0 else '-'} | Test Mean-Direction={'+' if hyg_test_sign>0 else '-'}")

if bonf_pass == 0:
    print("\nVERDICT: KEIN robuster Edge nach Bonferroni.")
elif bonf_pass <= 3:
    print(f"\nVERDICT: {bonf_pass}/16 Bonferroni-PASS - isolierte Hits.")
else:
    print(f"\nVERDICT: {bonf_pass}/16 Bonferroni-PASS - potentieller Edge!")
