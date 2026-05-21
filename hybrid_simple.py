#!/usr/bin/env python3
"""
OPTION 3-SIMPLIFIED: HYBRID SYSTEM (Minimal)

Rules:
1. MA50 > MA200 = uptrend signal (0 or 1)
2. Position-Size = signal * (1 - VIX-normalized/10)
3. Daily return = SPY daily return * position-size
4. Metrics: Sharpe, MaxDD, Sortino
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings; warnings.filterwarnings("ignore")

START = "2014-01-01"

def metrics(returns):
    """Compute Sharpe, Sortino, MaxDD."""
    ret = np.asarray(returns, float)
    if len(ret) < 2:
        return np.nan, np.nan, np.nan

    sharpe = ret.mean() / (ret.std() + 1e-6) * np.sqrt(252)
    downside = np.sqrt(np.mean(np.minimum(ret, 0)**2))
    sortino = ret.mean() / (downside + 1e-6) * np.sqrt(252)

    cumret = np.cumprod(1 + ret)
    dd = cumret / np.maximum.accumulate(cumret) - 1
    maxdd = dd.min()

    return sharpe, sortino, maxdd

print("="*80)
print("OPTION 3: HYBRID SYSTEM (Simplified)")
print("="*80)

# Load
spy = yf.download("SPY", start=START, progress=False)
vix = yf.download("^VIX", start=START, progress=False)
print(f"Data loaded: {len(spy)} days\n")

# Extract (ensure 1D arrays)
close = np.asarray(spy["Close"].values).flatten()
o = np.asarray(spy["Open"].values).flatten()
h = np.asarray(spy["High"].values).flatten()
l = np.asarray(spy["Low"].values).flatten()
vix_close = np.asarray(vix["Close"].values).flatten()

# Returns
rets = np.diff(close) / close[:-1]
dates = spy.index[1:]

# MA50/MA200
ma50 = pd.Series(close).rolling(50).mean().values
ma200 = pd.Series(close).rolling(200).mean().values
uptrend = (ma50 > ma200).astype(float)

# VIX normalized
vix_mean = np.mean(vix_close[:len(vix_close)//2])
vix_norm = np.clip((vix_close - vix_mean) / (vix_mean + 1) * 0.1, -0.5, 0.5)

# Align to returns length (drop first day)
uptrend = uptrend[1:len(rets)+1]
vix_norm = vix_norm[1:len(rets)+1]

# Position-size
pos_size = uptrend * np.clip(1 - vix_norm, 0.2, 1.0)

# Portfolio return
port_ret = rets * pos_size

# Split IS/OOS
split_date = pd.Timestamp("2022-01-01")
is_idx = dates < split_date
oos_idx = dates >= split_date

is_ret = port_ret[is_idx]
oos_ret = port_ret[oos_idx]
spy_oos_ret = rets[oos_idx]

# Metrics
is_sh, is_so, is_dd = metrics(is_ret)
oos_sh, oos_so, oos_dd = metrics(oos_ret)
spy_sh, _, spy_dd = metrics(spy_oos_ret)

print(f"IS (2014-2021): Sharpe {is_sh:.2f}, Sortino {is_so:.2f}, MaxDD {is_dd*100:.1f}%")
print(f"OOS (2022-2026): Sharpe {oos_sh:.2f}, Sortino {oos_so:.2f}, MaxDD {oos_dd*100:.1f}%")
print(f"SPY OOS: Sharpe {spy_sh:.2f}, MaxDD {spy_dd*100:.1f}%\n")

# Gates
g1 = is_sh > 1.0
g2 = oos_sh > 0.8
g3 = oos_dd > -0.20
g4 = oos_so > 1.5

print(f"Gates: G1={g1} G2={g2} G3={g3} G4={g4}")
print(f"\nResult: {'[OK] VALID' if all([g1,g2,g3,g4]) else '[PARTIAL] Check metrics'}")
