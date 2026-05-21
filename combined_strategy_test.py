#!/usr/bin/env python3
"""
COMBINED STRATEGY: Hybrid SPY + HYG Stress-Buy als Portfolio

Frage: Bringt Diversifikation einen zusaetzlichen Sharpe-Lift?

Setups:
  S0: 100% SPY Buy-and-Hold
  S1: 100% HYG Buy-and-Hold
  S2: 100% Hybrid-SPY (MA50/200 + VIX-Norm)
  S3: 100% HYG-Strategy (S1 Sizing 50/100, Q75/20d)
  S4: 50/50 Hybrid-SPY + HYG-Strategy (rebalanced monthly)
  S5: 60/40 Hybrid-SPY + HYG-Strategy
  S6: 40/60 Hybrid-SPY + HYG-Strategy

Test-Period: 2019-2025 OOS (HYG-Inception 2007, also Hybrid Test-Period dominant)
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

print("="*80)
print("COMBINED STRATEGY: Hybrid SPY + HYG Stress-Buy")
print("="*80)

# DATA
print("\n[1/4] Lade Daten...", flush=True)

spy = yf.download("SPY", start="2007-04-11", progress=False)
spy_close = pd.Series(np.asarray(spy['Close']).flatten(), index=spy.index)
spy_ret = spy_close.pct_change()

vix = yf.download("^VIX", start="2007-04-11", progress=False)["Close"]
vix = pd.Series(np.asarray(vix).flatten(), index=vix.index)

hyg = yf.download("HYG", start="2007-04-11", progress=False)
hyg_close = pd.Series(np.asarray(hyg['Close']).flatten(), index=hyg.index)
hyg_ret = hyg_close.pct_change()

stlfsi = get_series("STLFSI4", start="2003-01-01")
print(f"  SPY/HYG: {len(spy_close)} Tage, VIX: {len(vix)}, STLFSI4: {len(stlfsi)}")

# HYBRID SPY STRATEGY
print("\n[2/4] Berechne Hybrid-SPY-Strategy...")
ma50 = spy_close.rolling(50).mean()
ma200 = spy_close.rolling(200).mean()
uptrend = (ma50 > ma200).astype(int)

vix_aligned = vix.reindex(spy_close.index, method='ffill')
vix_mean = vix_aligned.rolling(60).mean()
vix_std = vix_aligned.rolling(60).std()
vix_norm = ((vix_aligned - vix_mean) / (vix_std + 1e-6)) * 0.1
vix_norm_clip = vix_norm.clip(-0.5, 0.5)
size = (1 - vix_norm_clip).clip(0.2, 1.0)
hybrid_exposure = uptrend * size

# Daily PnL Hybrid (vorher 1 Tag Shift fuer realistic execution)
hybrid_pnl = spy_ret * hybrid_exposure.shift(1).fillna(0)
hybrid_slippage = hybrid_exposure.diff().abs().fillna(0) * 0.0005  # 5bps
hybrid_pnl_net = hybrid_pnl - hybrid_slippage

# HYG STRATEGY S1 (Q75/20d)
print("\n[3/4] Berechne HYG-Strategy...")
train_signal = stlfsi[stlfsi.index <= TRAIN_END].dropna()
threshold = train_signal.quantile(0.75)
print(f"  STLFSI4 Q75: {threshold:.4f}")

stlfsi_daily = stlfsi.reindex(hyg_ret.index, method='ffill')
entry = (stlfsi_daily.shift(1) > threshold).astype(int)
in_stress = entry.rolling(20).sum().clip(0, 1) > 0
hyg_exposure = pd.Series(np.where(in_stress, 1.0, 0.5), index=hyg_ret.index)

hyg_pnl_gross = hyg_ret * hyg_exposure.shift(1).fillna(0)
hyg_slippage = hyg_exposure.diff().abs().fillna(0) * 0.0005
hyg_pnl_net = hyg_pnl_gross - hyg_slippage

# METRICS
def metrics(returns, label):
    r = returns.dropna()
    if len(r) < 10: return None
    cum = (1 + r).cumprod()
    ann_ret = (cum.iloc[-1] ** (252/len(r))) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    downside = r[r < 0]
    sortino_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else vol
    sortino = ann_ret / sortino_vol if sortino_vol > 0 else 0
    maxdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"label": label, "ann_ret": ann_ret, "vol": vol,
            "sharpe": sharpe, "sortino": sortino, "maxdd": maxdd, "n": len(r)}

# OOS Filter (without COVID)
def oos(s):
    return s[(s.index >= TEST_START) & ~((s.index >= COVID_A) & (s.index <= COVID_B))]

# Strategies (OOS PnL streams)
spy_bh = oos(spy_ret)
hyg_bh = oos(hyg_ret)
hybrid_strat = oos(hybrid_pnl_net)
hyg_strat = oos(hyg_pnl_net)

# Align indices for combined strategies
aligned = pd.concat([hybrid_strat, hyg_strat], axis=1, keys=['hybrid', 'hyg']).dropna()
print(f"\nAligned OOS days: {len(aligned)}")

combo_50_50 = aligned['hybrid'] * 0.5 + aligned['hyg'] * 0.5
combo_60_40 = aligned['hybrid'] * 0.6 + aligned['hyg'] * 0.4
combo_40_60 = aligned['hybrid'] * 0.4 + aligned['hyg'] * 0.6

# CORRELATION
corr = aligned['hybrid'].corr(aligned['hyg'])
print(f"\nCorrelation Hybrid vs HYG-Strategy: {corr:.3f}")
if corr < 0.3:
    print("  -> Niedrige Korrelation -> Diversifikation moeglich!")
elif corr > 0.7:
    print("  -> Hohe Korrelation -> wenig Diversifikation")
else:
    print("  -> Moderate Korrelation")

# OUTPUT
print("\n[4/4] Ergebnisse...")
print("="*80)

results = [
    metrics(spy_bh, "S0: SPY Buy-and-Hold"),
    metrics(hyg_bh, "S1: HYG Buy-and-Hold"),
    metrics(hybrid_strat, "S2: Hybrid-SPY (live)"),
    metrics(hyg_strat, "S3: HYG-Strategy (live)"),
    metrics(combo_50_50, "S4: 50/50 Hybrid+HYG"),
    metrics(combo_60_40, "S5: 60/40 Hybrid+HYG"),
    metrics(combo_40_60, "S6: 40/60 Hybrid+HYG"),
]

print(f"{'Strategy':<28} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>9}")
print("-"*80)
for m in results:
    if m:
        print(f"{m['label']:<28} {m['ann_ret']*100:>+7.2f}% {m['vol']*100:>6.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['maxdd']*100:>+8.2f}%")

# Best Combo
best = max(results, key=lambda m: m['sharpe'] if m else 0)
print(f"\nBeste Sharpe: {best['label']} ({best['sharpe']:.3f})")

# Diversifikations-Effekt
print("\n=== DIVERSIFIKATIONS-EFFEKT ===")
print(f"Hybrid alone:  Sharpe {results[2]['sharpe']:.3f}")
print(f"HYG alone:     Sharpe {results[3]['sharpe']:.3f}")
print(f"50/50 Mix:     Sharpe {results[4]['sharpe']:.3f}")
print(f"60/40 Mix:     Sharpe {results[5]['sharpe']:.3f}")
print(f"40/60 Mix:     Sharpe {results[6]['sharpe']:.3f}")

# Naive Average
naive_avg = (results[2]['sharpe'] + results[3]['sharpe']) / 2
print(f"\nNaiver Mittelwert: {naive_avg:.3f}")
print(f"50/50 Tatsaechlich: {results[4]['sharpe']:.3f}")
if results[4]['sharpe'] > naive_avg:
    print(f"-> Diversifikations-Bonus: +{results[4]['sharpe']-naive_avg:.3f} Sharpe")
else:
    print(f"-> Kein Diversifikations-Bonus")
