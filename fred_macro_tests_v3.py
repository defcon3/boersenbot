#!/usr/bin/env python3
"""
FRED MACRO TESTS V3 - Mit API-Key, volle Historie

Was sich gegenüber V1/V2 ändert:
  - Volle FRED-Historie (2000+ statt nur ~2022+)
  - Echtes Train (2000-2018) vs Test (2019-2025) Split
  - Caching für schnelles Re-Run
  - Bonferroni-Korrektur (N Tests x 2 Hypothesen)

Tests:
  H1: Yield-Inversion → SPY Forward-Return abnormal (zwei-seitig)
  H2: HY-Credit-Spread Q75 → SPY Forward-Return abnormal
  H3: Jobless-Claims-4W-MA-Change Q75 → SPY Forward-Return abnormal
  + Cross-Asset (QQQ, IWM) Validierung
  + Multiple Forward-Windows (5, 10, 20, 60d)

Strict Pre-Reg:
  G1: Train-Effekt in einer Richtung (sign-consistency mit Test)
  G2: Test |t| > 2.58 (Bonferroni: ~12 Tests pro Hypothese)
  G3: Min 100 Test-Signale
  G4: Cross-Asset Konsistenz (mindestens 2/3 gleiche Richtung)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from fred_helper import get_series

# ===========================================================================
# CONFIG
# ===========================================================================

TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")
COVID_A = pd.Timestamp("2020-02-15")
COVID_B = pd.Timestamp("2020-04-30")

# Bonferroni: 3 Hypothesen x 4 Windows x 3 Assets = 36 -> |t|>3.0 strict
BONFERRONI_T = 3.0
NAIVE_T = 1.5

# ===========================================================================
# HELPERS
# ===========================================================================

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

def run_split_test(signal_series, returns, condition_fn, fwd_days):
    """Generic Train/Test mit Bedingungs-Funktion."""
    signal_daily = signal_series.reindex(returns.index, method='ffill')
    condition = condition_fn(signal_daily)

    fwd_ret = returns.rolling(fwd_days).sum().shift(-fwd_days)

    train_mask = (fwd_ret.index <= TRAIN_END) & ~((fwd_ret.index >= COVID_A) & (fwd_ret.index <= COVID_B))
    test_mask = (fwd_ret.index >= TEST_START) & ~((fwd_ret.index >= COVID_A) & (fwd_ret.index <= COVID_B))

    train_signal = fwd_ret[train_mask & condition].dropna()
    train_no = fwd_ret[train_mask & ~condition].dropna()
    test_signal = fwd_ret[test_mask & condition].dropna()
    test_no = fwd_ret[test_mask & ~condition].dropna()

    t_train, diff_train = welch_t(train_signal, train_no)
    t_test, diff_test = welch_t(test_signal, test_no)

    return {
        "t_train": t_train, "diff_train": diff_train, "n_train_sig": len(train_signal),
        "t_test": t_test, "diff_test": diff_test, "n_test_sig": len(test_signal),
        "sign_match": (np.sign(diff_train) == np.sign(diff_test)) if not np.isnan(diff_train) else False
    }

# ===========================================================================
# DATA LOADING
# ===========================================================================

print("="*80)
print("FRED MACRO TESTS V3 - Volle Historie via API")
print("="*80)

print("\n[1/5] Lade FRED-Daten (mit Cache)...", flush=True)
try:
    yield_curve = get_series("T10Y2Y", start="2000-01-01")
    hy_spread = get_series("BAMLH0A0HYM2", start="2000-01-01")
    claims = get_series("ICSA", start="2000-01-01")
except Exception as e:
    print(f"\n[FAIL] FRED-API-Fehler: {e}")
    exit(1)

print(f"  T10Y2Y:        {len(yield_curve):5d} Werte ({yield_curve.index.min().date()} - {yield_curve.index.max().date()})")
print(f"  HY-Spread:     {len(hy_spread):5d} Werte ({hy_spread.index.min().date()} - {hy_spread.index.max().date()})")
print(f"  Claims:        {len(claims):5d} Werte ({claims.index.min().date()} - {claims.index.max().date()})")

print("\n[2/5] Lade ETF-Returns...", flush=True)
spy_ret = fetch_returns("SPY")
qqq_ret = fetch_returns("QQQ")
iwm_ret = fetch_returns("IWM")
print(f"  SPY/QQQ/IWM: {len(spy_ret)} Tage")

# ===========================================================================
# DEFINIERE CONDITIONS (Train-only Quantile, kein Snooping)
# ===========================================================================

def cond_yield_inversion(signal):
    return signal < 0

def make_cond_q75_train(signal_series, train_end=TRAIN_END):
    """Q75-Threshold nur aus Train-Period."""
    train_data = signal_series[signal_series.index <= train_end]
    q75 = train_data.dropna().quantile(0.75)
    return lambda s: s > q75, q75

def make_cond_claims_change_q75(claims_series, train_end=TRAIN_END):
    """Claims-4W-MA-Change Q75 nur aus Train."""
    def transform(s):
        ma = s.rolling(20).mean()
        return ma.pct_change(20)

    def get_cond(s):
        chg = transform(s)
        train_chg = chg[chg.index <= train_end]
        q75 = train_chg.dropna().quantile(0.75)
        return chg > q75, q75

    return get_cond

# ===========================================================================
# RUN TESTS
# ===========================================================================

print("\n[3/5] Definiere Conditions (Train-only Quantile)...", flush=True)

# H2 Condition (HY-Spread > Train-Q75)
hy_cond, hy_q75 = make_cond_q75_train(hy_spread)
print(f"  HY-Spread Train-Q75: {hy_q75:.3f}")

# H3 Condition (Claims-Change > Train-Q75)
def cond_claims(returns_index):
    claims_daily = claims.reindex(returns_index, method='ffill')
    ma = claims_daily.rolling(20).mean()
    chg = ma.pct_change(20)
    train_chg = chg[chg.index <= TRAIN_END]
    q75 = train_chg.dropna().quantile(0.75)
    return chg > q75, q75

# ===========================================================================
# RESULT MATRIX
# ===========================================================================

print("\n[4/5] Lasse Tests laufen...", flush=True)

results = []
assets = [("SPY", spy_ret), ("QQQ", qqq_ret), ("IWM", iwm_ret)]
windows = [5, 10, 20, 60]

# H1: Yield-Inversion
for asset_name, ret in assets:
    for fwd in windows:
        r = run_split_test(yield_curve, ret, cond_yield_inversion, fwd)
        results.append({"hypothesis": "H1-YieldInv", "asset": asset_name, "fwd": fwd, **r})

# H2: HY-Spread > Train-Q75
for asset_name, ret in assets:
    for fwd in windows:
        r = run_split_test(hy_spread, ret, hy_cond, fwd)
        results.append({"hypothesis": "H2-HYSpread", "asset": asset_name, "fwd": fwd, **r})

# H3: Claims-Change > Train-Q75
for asset_name, ret in assets:
    claims_aligned = claims.reindex(ret.index, method='ffill')
    ma = claims_aligned.rolling(20).mean()
    chg = ma.pct_change(20)
    train_chg = chg[chg.index <= TRAIN_END]
    q75 = train_chg.dropna().quantile(0.75)

    # Mache Series für chg (so dass run_split_test funktioniert)
    chg_series = chg.dropna()
    cond_fn = lambda s: s > q75

    for fwd in windows:
        r = run_split_test(chg_series, ret, cond_fn, fwd)
        results.append({"hypothesis": "H3-Claims", "asset": asset_name, "fwd": fwd, **r})

# ===========================================================================
# ANALYSE
# ===========================================================================

print("\n[5/5] Auswertung...", flush=True)
print("\n" + "="*100)
print(f"{'Hypothesis':<14} {'Asset':<5} {'Fwd':>4}  {'TrDiff%':>8} {'tTrain':>7}  {'TeDiff%':>8} {'tTest':>7}  {'Sign?':<5}  {'Naive':<6}  {'Bonf':<6}  {'nTest':>6}")
print("="*100)

for r in results:
    naive = abs(r['t_test']) > NAIVE_T and r['n_test_sig'] >= 100 and r['sign_match']
    bonf  = abs(r['t_test']) > BONFERRONI_T and r['n_test_sig'] >= 100 and r['sign_match']
    print(f"{r['hypothesis']:<14} {r['asset']:<5} {r['fwd']:>4}  "
          f"{r['diff_train']*100:>+8.3f} {r['t_train']:>+7.2f}  "
          f"{r['diff_test']*100:>+8.3f} {r['t_test']:>+7.2f}  "
          f"{'YES' if r['sign_match'] else 'NO':<5}  "
          f"{'PASS' if naive else 'FAIL':<6}  "
          f"{'PASS' if bonf else 'FAIL':<6}  "
          f"{r['n_test_sig']:>6}")

# Aggregate
naive_passes = sum(1 for r in results if abs(r['t_test']) > NAIVE_T and r['n_test_sig'] >= 100 and r['sign_match'])
bonf_passes = sum(1 for r in results if abs(r['t_test']) > BONFERRONI_T and r['n_test_sig'] >= 100 and r['sign_match'])

print(f"\n{'='*100}")
print(f"  Naive (|t|>{NAIVE_T}, n>=100, sign-match): {naive_passes}/{len(results)} PASS")
print(f"  Bonferroni (|t|>{BONFERRONI_T}, n>=100, sign-match): {bonf_passes}/{len(results)} PASS")

# Cross-asset Konsistenz pro Hypothesen-Window-Combo
print(f"\n{'='*100}")
print("CROSS-ASSET KONSISTENZ (gleiche Richtung in SPY/QQQ/IWM):")
print("="*100)
df = pd.DataFrame(results)
for hyp in ["H1-YieldInv", "H2-HYSpread", "H3-Claims"]:
    for fwd in windows:
        sub = df[(df['hypothesis']==hyp) & (df['fwd']==fwd)]
        signs = np.sign(sub['diff_test'].values)
        same = abs(signs.sum()) == len(signs)  # alle gleich
        ts = sub['t_test'].values
        print(f"  {hyp:<14} fwd={fwd:3d}d  Test-t: {ts[0]:+.2f} / {ts[1]:+.2f} / {ts[2]:+.2f}  Sign-Consistent: {'YES' if same else 'NO'}")

print(f"\n{'='*100}")
print("VERDICT")
print("="*100)
if bonf_passes == 0:
    print("  Alle Hypothesen unter Bonferroni FALSIFIZIERT.")
elif bonf_passes >= 3:
    print(f"  {bonf_passes} Tests Bonferroni-robust - schaue dir die einzelnen Faelle an!")
else:
    print(f"  {bonf_passes} isolierte Tests Bonferroni-PASS - schwacher Hinweis, aber kein Edge.")
