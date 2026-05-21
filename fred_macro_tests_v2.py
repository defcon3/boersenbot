#!/usr/bin/env python3
"""
FRED MACRO TESTS V2 - Option B + C

B) CONTRARIAN-Hypothesen (umgekehrte Richtung von V1):
   B1: Yield-Inversion -> SPY BUY (Forward-Return > Baseline)
   B2: HY-Spread > Q75 -> SPY BUY

C) H3 TIEFER:
   C1-C5: Verschiedene Forward-Windows (5, 10, 20, 60, 120 Tage)
   C6: Validierung auf NDX (QQQ)
   C7: Validierung auf RUT (IWM)

Strict Pre-Reg:
   - Neuer Split: Train 2010-2018, Test 2019-2025 (incl. COVID + 2022-25)
   - One-tailed Test (gerichtete Hypothese)
   - Bonferroni: 5 Windows -> Threshold t > 2.58 statt 1.5
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO
import time

# ===========================================================================
# FRED LOADER
# ===========================================================================

def fetch_fred(series_id, start="2010-01-01", retries=3):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; FRED-Research/1.0)'}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            df.columns = ['DATE', 'VALUE']
            df['DATE'] = pd.to_datetime(df['DATE'])
            df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
            df = df.dropna().set_index('DATE')
            return df['VALUE']
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                return None

def fetch_ticker_returns(ticker, start="2010-01-01"):
    df = yf.download(ticker, start=start, progress=False)
    close = np.asarray(df['Close']).flatten()
    return pd.Series(close, index=df.index).pct_change()

def welch_t(a, b):
    """Welch's t-test (2-sample, unequal variance)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return np.nan, np.nan
    se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    if se == 0: return np.nan, np.nan
    return (a.mean() - b.mean()) / se, a.mean() - b.mean()

# ===========================================================================
# DATA LOAD
# ===========================================================================

print("="*80)
print("FRED MACRO TESTS V2 - Option B + C")
print("="*80)

print("\n[1/4] Lade Daten...", flush=True)
spy_ret = fetch_ticker_returns("SPY")
qqq_ret = fetch_ticker_returns("QQQ")
iwm_ret = fetch_ticker_returns("IWM")
print(f"  SPY: {len(spy_ret)}, QQQ: {len(qqq_ret)}, IWM: {len(iwm_ret)}")

yield_curve = fetch_fred("T10Y2Y")
hy_spread = fetch_fred("BAMLH0A0HYM2")
claims = fetch_fred("ICSA")
print(f"  T10Y2Y: {len(yield_curve)}, HY-Spread: {len(hy_spread)}, Claims: {len(claims)}")

# NEUER Split (anders als V1!)
TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")
COVID_A = pd.Timestamp("2020-02-15")
COVID_B = pd.Timestamp("2020-04-30")

# ===========================================================================
# OPTION B: CONTRARIAN H1+H2
# ===========================================================================

print("\n" + "="*80)
print("OPTION B: CONTRARIAN-HYPOTHESEN")
print("="*80)

def test_contrarian(name, signal, returns, threshold_logic, fwd_days=20):
    signal_daily = signal.reindex(returns.index, method='ffill')

    if threshold_logic == "lt_zero":
        condition = signal_daily < 0
        desc = "Signal < 0"
    elif threshold_logic == "gt_q75_train":
        # Q75 NUR aus Training-Period berechnen (kein Snooping)
        train_signal = signal_daily[signal_daily.index <= TRAIN_END]
        q75 = train_signal.dropna().quantile(0.75)
        condition = signal_daily > q75
        desc = f"Signal > {q75:.3f} (Train-Q75)"

    fwd_ret = returns.rolling(fwd_days).sum().shift(-fwd_days)

    train_mask = (fwd_ret.index <= TRAIN_END) & ~((fwd_ret.index >= COVID_A) & (fwd_ret.index <= COVID_B))
    test_mask = (fwd_ret.index >= TEST_START) & ~((fwd_ret.index >= COVID_A) & (fwd_ret.index <= COVID_B))

    train_signal_ret = fwd_ret[train_mask & condition].dropna()
    train_no_ret = fwd_ret[train_mask & ~condition].dropna()
    test_signal_ret = fwd_ret[test_mask & condition].dropna()
    test_no_ret = fwd_ret[test_mask & ~condition].dropna()

    t_train, diff_train = welch_t(train_signal_ret, train_no_ret)
    t_test, diff_test = welch_t(test_signal_ret, test_no_ret)

    # CONTRARIAN-Hypothese: Signal-Tage haben HÖHERE Forward-Returns
    g1 = (diff_train > 0)
    g2 = (diff_test > 0) and (t_test > 1.5)
    g3 = (diff_test > 0.0005)
    g4 = len(test_signal_ret) >= 50

    all_pass = all([g1, g2, g3, g4])

    print(f"\n  {name} ({desc})")
    print(f"    Train: Signal={len(train_signal_ret):4d} Diff={diff_train*100:+.3f}% t={t_train:+.2f}")
    print(f"    Test:  Signal={len(test_signal_ret):4d} Diff={diff_test*100:+.3f}% t={t_test:+.2f}")
    print(f"    Gates: G1={g1} G2={g2} G3={g3} G4={g4} -> {'PASS' if all_pass else 'FAIL'}")

    return {"name": name, "diff_test": diff_test, "t_test": t_test, "pass": all_pass, "n": len(test_signal_ret)}

print("\n[2/4] Teste B1: Contrarian Yield-Inversion...", flush=True)
b1 = test_contrarian("B1: Yield-Inversion -> BUY", yield_curve, spy_ret, "lt_zero")

print("\n[3/4] Teste B2: Contrarian HY-Spread...", flush=True)
b2 = test_contrarian("B2: HY-Spread > Train-Q75 -> BUY", hy_spread, spy_ret, "gt_q75_train")

# ===========================================================================
# OPTION C: H3 TIEFER (Multi-Window + Multi-Market)
# ===========================================================================

print("\n" + "="*80)
print("OPTION C: H3 TIEFER (Multi-Window + NDX/RUT)")
print("="*80)

def test_h3_window(name, returns, fwd_days):
    """H3: Claims-4W-MA-Change > Train-Q75 -> Forward-Returns NIEDRIGER."""
    claims_daily = claims.reindex(returns.index, method='ffill')
    ma4 = claims_daily.rolling(20).mean()
    pct_chg = ma4.pct_change(20)

    # Q75 NUR aus Training (kein Snooping)
    train_chg = pct_chg[pct_chg.index <= TRAIN_END]
    q75 = train_chg.dropna().quantile(0.75)
    condition = pct_chg > q75

    fwd_ret = returns.rolling(fwd_days).sum().shift(-fwd_days)

    train_mask = (fwd_ret.index <= TRAIN_END) & ~((fwd_ret.index >= COVID_A) & (fwd_ret.index <= COVID_B))
    test_mask = (fwd_ret.index >= TEST_START) & ~((fwd_ret.index >= COVID_A) & (fwd_ret.index <= COVID_B))

    train_signal_ret = fwd_ret[train_mask & condition].dropna()
    train_no_ret = fwd_ret[train_mask & ~condition].dropna()
    test_signal_ret = fwd_ret[test_mask & condition].dropna()
    test_no_ret = fwd_ret[test_mask & ~condition].dropna()

    t_train, diff_train = welch_t(train_signal_ret, train_no_ret)
    t_test, diff_test = welch_t(test_signal_ret, test_no_ret)

    # H3-Hypothese: Signal-Tage haben NIEDRIGERE Returns -> diff < 0, t < 0
    # Bonferroni: 5 Windows -> Threshold ~|t| > 2.58
    g1 = (diff_train < 0)
    g2_naive = (diff_test < 0) and (abs(t_test) > 1.5)
    g2_bonferroni = (diff_test < 0) and (abs(t_test) > 2.58)
    g3 = (diff_test < -0.001)
    g4 = len(test_signal_ret) >= 50

    all_pass_naive = all([g1, g2_naive, g3, g4])
    all_pass_strict = all([g1, g2_bonferroni, g3, g4])

    print(f"\n  {name} (fwd={fwd_days}d, train-Q75={q75:.4f})")
    print(f"    Train: Signal={len(train_signal_ret):4d} Diff={diff_train*100:+.3f}% t={t_train:+.2f}")
    print(f"    Test:  Signal={len(test_signal_ret):4d} Diff={diff_test*100:+.3f}% t={t_test:+.2f}")
    print(f"    Gates Naive (t>1.5): {'PASS' if all_pass_naive else 'FAIL'}")
    print(f"    Gates Bonf  (t>2.58): {'PASS' if all_pass_strict else 'FAIL'}")

    return {"name": name, "fwd": fwd_days, "diff": diff_test, "t": t_test,
            "naive": all_pass_naive, "strict": all_pass_strict, "n": len(test_signal_ret)}

print("\n[4/4] H3 mit verschiedenen Forward-Windows...", flush=True)

c_results = []
for fwd in [5, 10, 20, 60, 120]:
    c_results.append(test_h3_window(f"C-SPY-{fwd}d", spy_ret, fwd))

print("\n--- Validierung auf NDX (QQQ) ---")
c_qqq = test_h3_window("C-QQQ-20d", qqq_ret, 20)

print("\n--- Validierung auf RUT (IWM) ---")
c_iwm = test_h3_window("C-IWM-20d", iwm_ret, 20)

# ===========================================================================
# ZUSAMMENFASSUNG
# ===========================================================================

print("\n" + "="*80)
print("ENDERGEBNIS")
print("="*80)

print("\nOption B (Contrarian):")
for r in [b1, b2]:
    print(f"  {r['name']:<50} {'PASS' if r['pass'] else 'FAIL'} (diff={r['diff_test']*100:+.3f}%, t={r['t_test']:+.2f}, n={r['n']})")

print("\nOption C (H3 Tiefer + Validierung):")
print("  Bonferroni-Korrektur: 5 Tests -> |t|>2.58")
for r in c_results:
    print(f"  {r['name']:<20} fwd={r['fwd']:3d}d  diff={r['diff']*100:+.3f}%  t={r['t']:+.2f}  Naive={'P' if r['naive'] else 'F'}  Bonf={'P' if r['strict'] else 'F'}  (n={r['n']})")

print(f"\n  Validation:")
print(f"  {c_qqq['name']:<20} fwd={c_qqq['fwd']:3d}d  diff={c_qqq['diff']*100:+.3f}%  t={c_qqq['t']:+.2f}  Naive={'P' if c_qqq['naive'] else 'F'}  (n={c_qqq['n']})")
print(f"  {c_iwm['name']:<20} fwd={c_iwm['fwd']:3d}d  diff={c_iwm['diff']*100:+.3f}%  t={c_iwm['t']:+.2f}  Naive={'P' if c_iwm['naive'] else 'F'}  (n={c_iwm['n']})")

n_b_pass = sum(r['pass'] for r in [b1, b2])
n_c_naive_pass = sum(r['naive'] for r in c_results) + (1 if c_qqq['naive'] else 0) + (1 if c_iwm['naive'] else 0)
n_c_strict_pass = sum(r['strict'] for r in c_results)

print(f"\n  Option B: {n_b_pass}/2 bestaetigt")
print(f"  Option C (Naive): {n_c_naive_pass}/7 bestaetigt")
print(f"  Option C (Bonferroni-strict): {n_c_strict_pass}/5 bestaetigt")
