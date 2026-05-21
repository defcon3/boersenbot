#!/usr/bin/env python3
"""
HYG OVERLAY-STRATEGIES: Variante A

Wenn Stress-Effekt real ist, sollte ein Overlay-Ansatz (immer long + EXTRA
bei Stress) besser sein als pure Buy-and-Hold.

Strategien zu vergleichen:
  S0: Buy-and-Hold (Baseline)
  S1: Position-Sizing (50% normal, 100% bei Stress) - immer im Markt, mehr bei Stress
  S2: Leverage (100% normal, 150% bei Stress) - Stress-Boost
  S3: Tilt (80% normal, 120% bei Stress) - sanftes Overlay
  S4: Inverse (Cash bei Stress, 100% sonst) - Sanity-Check (sollte SCHLECHTER sein)
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
HOLDING_DAYS = 20

print("="*80)
print("HYG OVERLAY-STRATEGIES (Variante A)")
print("="*80)

# DATA
print("\n[1/4] Daten laden...", flush=True)
stlfsi = get_series("STLFSI4", start="2003-01-01")
train_q75 = stlfsi[stlfsi.index <= TRAIN_END].dropna().quantile(0.75)
print(f"  STLFSI4 Train-Q75: {train_q75:.4f}")

hyg = yf.download("HYG", start="2007-04-11", progress=False)
hyg_ret = pd.Series(np.asarray(hyg['Close']).flatten(), index=hyg.index).pct_change()

jnk = yf.download("JNK", start="2007-12-01", progress=False)
jnk_ret = pd.Series(np.asarray(jnk['Close']).flatten(), index=jnk.index).pct_change()

print(f"  HYG: {len(hyg_ret)} Tage")
print(f"  JNK: {len(jnk_ret)} Tage")

# EXPOSURE-FUNKTIONEN
def make_exposure(signal, threshold, normal_size, stress_size, holding_days, returns_index):
    """
    Konstruiert Expsoure-Series:
      - default: normal_size
      - wenn signal > threshold in last holding_days: stress_size
    """
    signal_daily = signal.reindex(returns_index, method='ffill')
    entry_signal = (signal_daily.shift(1) > threshold).astype(int)
    in_stress = entry_signal.rolling(holding_days).sum().clip(0, 1) > 0
    exposure = np.where(in_stress, stress_size, normal_size)
    return pd.Series(exposure, index=returns_index)

def apply_strategy(asset_ret, exposure, slippage):
    """Wende Exposure auf Asset an, mit Slippage."""
    pnl_gross = asset_ret * exposure.shift(1).fillna(0)
    exposure_change = exposure.diff().abs().fillna(0)
    slippage_cost = exposure_change * slippage
    return pnl_gross - slippage_cost

def metrics(returns, label):
    r = returns.dropna()
    if len(r) < 10:
        return {"label": label, "ann_ret": 0, "vol": 0, "sharpe": 0, "sortino": 0, "maxdd": 0}
    cumulative = (1 + r).cumprod()
    ann_ret = (cumulative.iloc[-1] ** (252/len(r))) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    downside = r[r < 0]
    sortino_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else vol
    sortino = ann_ret / sortino_vol if sortino_vol > 0 else 0
    rolling_max = cumulative.cummax()
    maxdd = ((cumulative - rolling_max) / rolling_max).min()
    return {"label": label, "ann_ret": ann_ret, "vol": vol, "sharpe": sharpe, "sortino": sortino, "maxdd": maxdd, "n": len(r)}

# STRATEGIE-VARIANTEN
strategies = [
    # (name, normal_size, stress_size)
    ("S0: Buy-and-Hold",              1.0, 1.0),
    ("S1: Sizing 50/100",             0.5, 1.0),
    ("S2: Leverage 100/150",          1.0, 1.5),
    ("S3: Tilt 80/120",               0.8, 1.2),
    ("S4: Inverse 100/0",             1.0, 0.0),  # Sanity-Check (sollte schlecht sein)
]

print("\n[2/4] Test Strategien auf HYG...", flush=True)
print("="*100)

oos_mask_hyg = lambda r: (r.index >= TEST_START) & ~((r.index >= COVID_A) & (r.index <= COVID_B))

results_hyg = []
for name, normal, stress in strategies:
    exp = make_exposure(stlfsi, train_q75, normal, stress, HOLDING_DAYS, hyg_ret.index)
    pnl = apply_strategy(hyg_ret, exp, SLIPPAGE)
    pnl_oos = pnl[oos_mask_hyg(pnl)]
    m = metrics(pnl_oos, name)
    m['avg_exposure'] = exp[oos_mask_hyg(exp)].mean()
    results_hyg.append(m)

print(f"\n{'Strategy':<28} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'AvgExp':>7}")
print("-"*100)
for m in results_hyg:
    print(f"{m['label']:<28} {m['ann_ret']*100:>+7.2f}% {m['vol']*100:>6.2f}% "
          f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['maxdd']*100:>+7.2f}% "
          f"{m['avg_exposure']*100:>6.1f}%")

# Sharpe-Differenz gegen Baseline
baseline = results_hyg[0]
print(f"\nVergleich gegen Buy-and-Hold (Baseline Sharpe {baseline['sharpe']:.2f}):")
for m in results_hyg[1:]:
    delta_sharpe = m['sharpe'] - baseline['sharpe']
    print(f"  {m['label']:<28} Delta-Sharpe: {delta_sharpe:+.3f}")

# JNK Validation
print("\n[3/4] Validierung auf JNK...", flush=True)
oos_mask_jnk = lambda r: (r.index >= TEST_START) & ~((r.index >= COVID_A) & (r.index <= COVID_B))

results_jnk = []
for name, normal, stress in strategies:
    exp = make_exposure(stlfsi, train_q75, normal, stress, HOLDING_DAYS, jnk_ret.index)
    pnl = apply_strategy(jnk_ret, exp, SLIPPAGE)
    pnl_oos = pnl[oos_mask_jnk(pnl)]
    m = metrics(pnl_oos, name.replace("S", "J"))
    m['avg_exposure'] = exp[oos_mask_jnk(exp)].mean()
    results_jnk.append(m)

print(f"\n{'Strategy (JNK)':<28} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'AvgExp':>7}")
print("-"*100)
for m in results_jnk:
    print(f"{m['label']:<28} {m['ann_ret']*100:>+7.2f}% {m['vol']*100:>6.2f}% "
          f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['maxdd']*100:>+7.2f}% "
          f"{m['avg_exposure']*100:>6.1f}%")

baseline_jnk = results_jnk[0]
print(f"\nVergleich gegen Buy-and-Hold JNK (Baseline Sharpe {baseline_jnk['sharpe']:.2f}):")
for m in results_jnk[1:]:
    delta_sharpe = m['sharpe'] - baseline_jnk['sharpe']
    print(f"  {m['label']:<28} Delta-Sharpe: {delta_sharpe:+.3f}")

# VERDICT
print("\n[4/4] Verdict")
print("="*100)

# Best Strategy?
best_hyg = max(results_hyg, key=lambda m: m['sharpe'])
best_jnk = max(results_jnk, key=lambda m: m['sharpe'])

print(f"\nBeste HYG-Strategy: {best_hyg['label']} (Sharpe {best_hyg['sharpe']:+.2f})")
print(f"Beste JNK-Strategy: {best_jnk['label']} (Sharpe {best_jnk['sharpe']:+.2f})")

# Validation: Same Best?
if best_hyg['label'] == best_jnk['label']:
    print(f"\n[OK] HYG und JNK beide bestaetigen {best_hyg['label']} als beste")
else:
    print(f"\n[INCONSIST] HYG bevorzugt {best_hyg['label']}, JNK bevorzugt {best_jnk['label']}")

# Edge gegen Buy-and-Hold?
if best_hyg['sharpe'] > baseline['sharpe'] and best_jnk['sharpe'] > baseline_jnk['sharpe']:
    print(f"[EDGE] Beste Strategy schlaegt B&H in beiden!")
    delta_h = best_hyg['sharpe'] - baseline['sharpe']
    delta_j = best_jnk['sharpe'] - baseline_jnk['sharpe']
    print(f"  HYG: +{delta_h:.3f} Sharpe, JNK: +{delta_j:.3f} Sharpe")
elif best_hyg['sharpe'] <= baseline['sharpe']:
    print(f"[NO EDGE] Keine Variante schlaegt B&H konsistent")
