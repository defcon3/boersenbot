#!/usr/bin/env python3
"""
HYG SENSITIVITY ANALYSE

Grid-Test: 4 Thresholds x 5 Holdings = 20 Setups auf HYG+JNK = 40 Tests

Frage: Ist der HYG-Stress-Buy-Edge robust ueber das Grid,
       oder nur an dem einen Q75/20d-Punkt?

Robust = mehrheitlich PASS, sign-konsistent, Mean-Sharpe-Lift > 0
Overfit = nur Q75/20d PASS, andere FAIL
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
SLIPPAGE = 0.0005

THRESHOLDS = [0.70, 0.75, 0.80, 0.85]  # Quantile aus Train
HOLDINGS = [5, 10, 20, 30, 60]
NORMAL_EXP = 0.5
STRESS_EXP = 1.0

print("="*80)
print("HYG SENSITIVITY GRID: 4 Thresholds x 5 Holdings = 20 Setups")
print("="*80)

# Data
print("\n[1/4] Lade Daten...", flush=True)
stlfsi = get_series("STLFSI4", start="2003-01-01")
train_signal = stlfsi[stlfsi.index <= TRAIN_END].dropna()
print(f"  STLFSI4 Train-Quantile:")
for q in THRESHOLDS:
    print(f"    Q{int(q*100)}: {train_signal.quantile(q):.4f}")

def fetch_returns(ticker, start="2007-04-11"):
    df = yf.download(ticker, start=start, progress=False)
    close = np.asarray(df['Close']).flatten()
    return pd.Series(close, index=df.index).pct_change()

hyg_ret = fetch_returns("HYG")
jnk_ret = fetch_returns("JNK", start="2007-12-01")

def metrics(returns):
    r = returns.dropna()
    if len(r) < 10: return {"sharpe": 0, "ann_ret": 0, "maxdd": 0}
    cum = (1 + r).cumprod()
    ann_ret = (cum.iloc[-1] ** (252/len(r))) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    maxdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"sharpe": sharpe, "ann_ret": ann_ret, "maxdd": maxdd}

def backtest(asset_ret, threshold_val, holding_days):
    """S1 Sizing 50/100 Strategy."""
    signal_daily = stlfsi.reindex(asset_ret.index, method='ffill')
    entry = (signal_daily.shift(1) > threshold_val).astype(int)
    in_stress = entry.rolling(holding_days).sum().clip(0, 1) > 0
    exposure = pd.Series(np.where(in_stress, STRESS_EXP, NORMAL_EXP), index=asset_ret.index)

    pnl_gross = asset_ret * exposure.shift(1).fillna(0)
    exp_change = exposure.diff().abs().fillna(0)
    pnl_net = pnl_gross - exp_change * SLIPPAGE

    bh_oos = asset_ret[(asset_ret.index >= TEST_START) & ~((asset_ret.index >= COVID_A) & (asset_ret.index <= COVID_B))]
    strat_oos = pnl_net[(pnl_net.index >= TEST_START) & ~((pnl_net.index >= COVID_A) & (pnl_net.index <= COVID_B))]

    bh_m = metrics(bh_oos)
    st_m = metrics(strat_oos)

    return {
        "strat_sharpe": st_m["sharpe"], "bh_sharpe": bh_m["sharpe"],
        "strat_maxdd": st_m["maxdd"], "bh_maxdd": bh_m["maxdd"],
        "strat_ann": st_m["ann_ret"], "bh_ann": bh_m["ann_ret"]
    }

print("\n[2/4] Grid auf HYG...", flush=True)
hyg_results = []
for q in THRESHOLDS:
    thresh = train_signal.quantile(q)
    for h in HOLDINGS:
        r = backtest(hyg_ret, thresh, h)
        r['q'] = q; r['h'] = h; r['thresh'] = thresh
        hyg_results.append(r)

print("\n[3/4] Grid auf JNK...", flush=True)
jnk_results = []
for q in THRESHOLDS:
    thresh = train_signal.quantile(q)
    for h in HOLDINGS:
        r = backtest(jnk_ret, thresh, h)
        r['q'] = q; r['h'] = h; r['thresh'] = thresh
        jnk_results.append(r)

# DISPLAY: Sharpe-Lift Matrix
print("\n[4/4] Auswertung:")
print("\n=== HYG Sharpe-Lift vs B&H ===")
print(f"{'':<7} " + "  ".join(f"{'h='+str(h)+'d':>7}" for h in HOLDINGS))
for q in THRESHOLDS:
    row = [f"Q{int(q*100)}:"]
    for h in HOLDINGS:
        r = next(x for x in hyg_results if x['q']==q and x['h']==h)
        lift = r['strat_sharpe'] - r['bh_sharpe']
        row.append(f"{lift:+.3f}")
    print(f"  {row[0]:<5} " + "  ".join(f"{v:>7}" for v in row[1:]))

print("\n=== JNK Sharpe-Lift vs B&H ===")
print(f"{'':<7} " + "  ".join(f"{'h='+str(h)+'d':>7}" for h in HOLDINGS))
for q in THRESHOLDS:
    row = [f"Q{int(q*100)}:"]
    for h in HOLDINGS:
        r = next(x for x in jnk_results if x['q']==q and x['h']==h)
        lift = r['strat_sharpe'] - r['bh_sharpe']
        row.append(f"{lift:+.3f}")
    print(f"  {row[0]:<5} " + "  ".join(f"{v:>7}" for v in row[1:]))

# MaxDD-Improvement
print("\n=== HYG MaxDD-Improvement (BH MaxDD - Strat MaxDD, positiv = besser) ===")
print(f"{'':<7} " + "  ".join(f"{'h='+str(h)+'d':>7}" for h in HOLDINGS))
for q in THRESHOLDS:
    row = [f"Q{int(q*100)}:"]
    for h in HOLDINGS:
        r = next(x for x in hyg_results if x['q']==q and x['h']==h)
        imp = r['bh_maxdd'] - r['strat_maxdd']  # both negative, so positive diff = better
        row.append(f"{imp*100:+.1f}%")
    print(f"  {row[0]:<5} " + "  ".join(f"{v:>7}" for v in row[1:]))

# Robustheit-Analyse
print("\n" + "="*80)
print("ROBUSTHEIT")
print("="*80)
positive_hyg = sum(1 for r in hyg_results if r['strat_sharpe'] > r['bh_sharpe'])
positive_jnk = sum(1 for r in jnk_results if r['strat_sharpe'] > r['bh_sharpe'])
mean_lift_hyg = np.mean([r['strat_sharpe'] - r['bh_sharpe'] for r in hyg_results])
mean_lift_jnk = np.mean([r['strat_sharpe'] - r['bh_sharpe'] for r in jnk_results])

print(f"\nHYG: {positive_hyg}/20 Setups schlagen B&H | Mean Sharpe-Lift: {mean_lift_hyg:+.3f}")
print(f"JNK: {positive_jnk}/20 Setups schlagen B&H | Mean Sharpe-Lift: {mean_lift_jnk:+.3f}")

# Best Setup
best_hyg = max(hyg_results, key=lambda r: r['strat_sharpe'] - r['bh_sharpe'])
best_jnk = max(jnk_results, key=lambda r: r['strat_sharpe'] - r['bh_sharpe'])
print(f"\nBest HYG-Setup: Q{int(best_hyg['q']*100)}/{best_hyg['h']}d -> Lift {best_hyg['strat_sharpe']-best_hyg['bh_sharpe']:+.3f}")
print(f"Best JNK-Setup: Q{int(best_jnk['q']*100)}/{best_jnk['h']}d -> Lift {best_jnk['strat_sharpe']-best_jnk['bh_sharpe']:+.3f}")

# Current deployed setup
deployed = next(r for r in hyg_results if r['q']==0.75 and r['h']==20)
print(f"\nDeployed (Q75/20d): Lift {deployed['strat_sharpe']-deployed['bh_sharpe']:+.3f}")

# Verdict
if positive_hyg >= 15 and positive_jnk >= 15 and mean_lift_hyg > 0 and mean_lift_jnk > 0:
    print("\nVERDICT: Edge ROBUST ueber das Grid")
elif positive_hyg >= 10:
    print("\nVERDICT: Edge teilweise robust - Best Setup nicht universell stabil")
else:
    print("\nVERDICT: Edge FRAGIL - nur an wenigen Punkten")
