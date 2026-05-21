#!/usr/bin/env python3
"""
HYG STRESS-BUY STRATEGY: Backtest + JNK-Validation

A) Strategy:
   Wenn STLFSI4 > 0.2909 (Train-Q75) -> Buy HYG am naechsten Trading-Tag
   Halten fuer 20 Tage (best Forward-Window aus Pre-Reg)
   Slippage: 5bps round-trip
   Vergleich: Strategy-Returns vs Buy-and-Hold HYG vs Cash

B) JNK-Validation:
   Dasselbe Pattern auf JNK (anderer HY-Bond-ETF) testen
   Wenn HYG-Edge echt, sollte JNK aehnliches Verhalten zeigen
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
SLIPPAGE = 0.0005  # 5bps round-trip
HOLDING_DAYS = 20  # best fwd-window aus Pre-Reg

print("="*80)
print("HYG STRESS-BUY STRATEGY: Backtest + JNK-Validation")
print("="*80)

# DATA
print("\n[1/5] Lade Daten...", flush=True)
stlfsi = get_series("STLFSI4", start="2003-01-01")
train_q75 = stlfsi[stlfsi.index <= TRAIN_END].dropna().quantile(0.75)
print(f"  STLFSI4 Train-Q75 (Schwelle): {train_q75:.4f}")

hyg = yf.download("HYG", start="2007-04-11", progress=False)
hyg_close = pd.Series(np.asarray(hyg['Close']).flatten(), index=hyg.index)
hyg_ret = hyg_close.pct_change()
print(f"  HYG: {len(hyg_close)} Tage")

jnk = yf.download("JNK", start="2007-12-01", progress=False)
jnk_close = pd.Series(np.asarray(jnk['Close']).flatten(), index=jnk.index)
jnk_ret = jnk_close.pct_change()
print(f"  JNK: {len(jnk_close)} Tage")

# STRATEGY BACKTEST
def backtest_strategy(asset_ret, signal, threshold, holding_days, slippage):
    """
    Long-Only Strategy:
      Wenn signal > threshold (gestern) -> heute Long
      Halten fuer holding_days
      Position-Size = 1.0 (voll long)
      Bei Re-Trigger: Position laeuft weiter (max overlapping)

    Returns: pd.DataFrame mit strategy-pnl
    """
    signal_daily = signal.reindex(asset_ret.index, method='ffill')

    # Entry-Signal (gestriges Signal triggert heutigen Entry)
    entry_signal = (signal_daily.shift(1) > threshold).astype(int)

    # Track open positions
    df = pd.DataFrame({
        'asset_ret': asset_ret,
        'signal': signal_daily,
        'entry': entry_signal,
    })

    # Position: Count open positions in last holding_days window
    df['open_positions'] = df['entry'].rolling(holding_days).sum().clip(0, holding_days)

    # Cap exposure at 1.0 (no levered)
    df['exposure'] = (df['open_positions'] / holding_days).clip(0, 1.0)

    # Daily PnL (return * exposure)
    df['strategy_ret_gross'] = df['asset_ret'] * df['exposure'].shift(1).fillna(0)

    # Slippage when entering (proportional to exposure change)
    df['exposure_change'] = df['exposure'].diff().abs().fillna(0)
    df['slippage_cost'] = df['exposure_change'] * slippage
    df['strategy_ret_net'] = df['strategy_ret_gross'] - df['slippage_cost']

    return df

def metrics(returns, label=""):
    """Compute key metrics."""
    r = returns.dropna()
    if len(r) < 10:
        return {"label": label, "n": len(r), "annualized": 0, "sharpe": 0, "sortino": 0, "maxdd": 0}

    cumulative = (1 + r).cumprod()
    annual_ret = (cumulative.iloc[-1] ** (252/len(r))) - 1
    annual_vol = r.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

    downside = r[r < 0]
    sortino_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else annual_vol
    sortino = annual_ret / sortino_vol if sortino_vol > 0 else 0

    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    maxdd = drawdown.min()

    return {
        "label": label,
        "n": len(r),
        "annualized_ret": annual_ret,
        "annualized_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "maxdd": maxdd,
        "final_value": cumulative.iloc[-1]
    }

# RUN STRATEGY
print("\n[2/5] Run Strategy auf HYG...", flush=True)
hyg_strat = backtest_strategy(hyg_ret, stlfsi, train_q75, HOLDING_DAYS, SLIPPAGE)

# Split In-Sample / Out-of-Sample
hyg_is = hyg_strat[hyg_strat.index <= TRAIN_END]
hyg_oos = hyg_strat[(hyg_strat.index >= TEST_START) & ~((hyg_strat.index >= COVID_A) & (hyg_strat.index <= COVID_B))]

# Buy-and-Hold Comparison
hyg_bh_is = hyg_ret[hyg_ret.index <= TRAIN_END]
hyg_bh_oos = hyg_ret[(hyg_ret.index >= TEST_START) & ~((hyg_ret.index >= COVID_A) & (hyg_ret.index <= COVID_B))]

print("\n[3/5] HYG Backtest Ergebnisse:")
print("="*80)
results_hyg = [
    metrics(hyg_is['strategy_ret_net'], "HYG Strategy IS (2007-2018)"),
    metrics(hyg_bh_is, "HYG Buy-and-Hold IS"),
    metrics(hyg_oos['strategy_ret_net'], "HYG Strategy OOS (2019-2025)"),
    metrics(hyg_bh_oos, "HYG Buy-and-Hold OOS"),
]

print(f"{'Period':<40} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'N':>6}")
print("-"*80)
for m in results_hyg:
    print(f"{m['label']:<40} {m['annualized_ret']*100:>+7.2f}% "
          f"{m['annualized_vol']*100:>6.2f}% "
          f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
          f"{m['maxdd']*100:>+7.2f}% {m['n']:>6}")

# Strategy Activity Rate
oos_exposure = hyg_oos['exposure'].mean()
print(f"\nStrategy Exposure OOS: {oos_exposure*100:.1f}% (average)")
print(f"Entry-Tage OOS: {hyg_oos['entry'].sum()}")

# B) JNK VALIDATION
print("\n[4/5] JNK-Validation (anderer HY-Bond-ETF)...")
print("="*80)
jnk_strat = backtest_strategy(jnk_ret, stlfsi, train_q75, HOLDING_DAYS, SLIPPAGE)
jnk_is = jnk_strat[jnk_strat.index <= TRAIN_END]
jnk_oos = jnk_strat[(jnk_strat.index >= TEST_START) & ~((jnk_strat.index >= COVID_A) & (jnk_strat.index <= COVID_B))]
jnk_bh_is = jnk_ret[jnk_ret.index <= TRAIN_END]
jnk_bh_oos = jnk_ret[(jnk_ret.index >= TEST_START) & ~((jnk_ret.index >= COVID_A) & (jnk_ret.index <= COVID_B))]

results_jnk = [
    metrics(jnk_is['strategy_ret_net'], "JNK Strategy IS (2007-2018)"),
    metrics(jnk_bh_is, "JNK Buy-and-Hold IS"),
    metrics(jnk_oos['strategy_ret_net'], "JNK Strategy OOS (2019-2025)"),
    metrics(jnk_bh_oos, "JNK Buy-and-Hold OOS"),
]

print(f"{'Period':<40} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'N':>6}")
print("-"*80)
for m in results_jnk:
    print(f"{m['label']:<40} {m['annualized_ret']*100:>+7.2f}% "
          f"{m['annualized_vol']*100:>6.2f}% "
          f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
          f"{m['maxdd']*100:>+7.2f}% {m['n']:>6}")

# Direct comparison: Strategy minus Buy-Hold
print("\n[5/5] Strategy Excess Returns vs Buy-and-Hold:")
print("="*80)

def excess_metrics(strat, bh, label):
    """Excess returns."""
    aligned = pd.concat([strat, bh], axis=1, keys=['strat', 'bh']).dropna()
    excess = aligned['strat'] - aligned['bh']
    m = metrics(excess, label)
    # t-stat
    t = excess.mean() / (excess.std() / np.sqrt(len(excess))) if len(excess) > 1 else 0
    m['t_stat'] = t
    return m

excess_hyg_oos = excess_metrics(hyg_oos['strategy_ret_net'], hyg_bh_oos, "HYG Excess OOS")
excess_jnk_oos = excess_metrics(jnk_oos['strategy_ret_net'], jnk_bh_oos, "JNK Excess OOS")

for m in [excess_hyg_oos, excess_jnk_oos]:
    print(f"{m['label']:<25} AnnRet: {m['annualized_ret']*100:+.2f}% | Sharpe: {m['sharpe']:+.2f} | t-stat: {m['t_stat']:+.2f}")

# VERDICT
print("\n" + "="*80)
print("VERDICT")
print("="*80)

hyg_oos_strat = next(m for m in results_hyg if m['label'] == "HYG Strategy OOS (2019-2025)")
hyg_oos_bh = next(m for m in results_hyg if m['label'] == "HYG Buy-and-Hold OOS")
jnk_oos_strat = next(m for m in results_jnk if m['label'] == "JNK Strategy OOS (2019-2025)")
jnk_oos_bh = next(m for m in results_jnk if m['label'] == "JNK Buy-and-Hold OOS")

print(f"\nHYG OOS:")
print(f"  Strategy Sharpe {hyg_oos_strat['sharpe']:+.2f} vs Buy-Hold {hyg_oos_bh['sharpe']:+.2f}")
print(f"  Excess vs BH: t={excess_hyg_oos['t_stat']:+.2f}, AnnRet={excess_hyg_oos['annualized_ret']*100:+.2f}%")

print(f"\nJNK OOS (Validation):")
print(f"  Strategy Sharpe {jnk_oos_strat['sharpe']:+.2f} vs Buy-Hold {jnk_oos_bh['sharpe']:+.2f}")
print(f"  Excess vs BH: t={excess_jnk_oos['t_stat']:+.2f}, AnnRet={excess_jnk_oos['annualized_ret']*100:+.2f}%")

# Save artifacts
hyg_strat.to_csv("hyg_strategy_pnl.csv")
jnk_strat.to_csv("jnk_strategy_pnl.csv")
print(f"\nArtifacts: hyg_strategy_pnl.csv, jnk_strategy_pnl.csv")
