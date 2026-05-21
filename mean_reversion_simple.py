#!/usr/bin/env python3
"""
MEAN-REVERSION DAILY — MINIMAL VERSION

Signal: SPY close_move > 2% or < -2% yesterday -> kontrarian Long today
Entry: Today's open, Exit: Today's close (1-day hold)
Costs: 5bps + 0.1% slippage per round-trip
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

START, SPLIT = "2014-01-01", "2022-01-01"
COVID_A, COVID_B = "2020-02-15", "2020-04-30"

def tstat(r):
    r = np.asarray(r, float)
    if len(r) < 2: return np.nan
    return r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))

print("="*80)
print("MEAN-REVERSION DAILY | Pre-Reg 2026-05-21")
print("="*80)

# Load SPY
spy = yf.download("SPY", start=START, progress=False)
print(f"SPY: {len(spy)} days\n")

# Extract OHLC
opens = spy["Open"].values
closes = spy["Close"].values
highs = spy["High"].values
lows = spy["Low"].values
dates = spy.index

# Identify signals: prev day close_move > +/- 2%
signals = []
for i in range(1, len(spy)):
    prev_close_move = (closes[i-1] - opens[i-1]) / opens[i-1] * 100

    if abs(prev_close_move) > 2.0:
        # Signal: kontrarian next day
        direction = 1 if prev_close_move < 0 else -1
        entry_price = opens[i]
        exit_price = closes[i]

        # Apply stop-loss
        low = lows[i]
        if (low - entry_price) / entry_price * 100 < -1.5:
            exit_price = entry_price * (1 - 0.015)

        # Return (before costs)
        ret = (exit_price - entry_price) / entry_price
        # Apply costs
        ret -= 0.0005 + 0.001  # 5bps + 0.1% slippage

        signals.append({
            "date": dates[i],
            "ret": ret,
            "prev_move": prev_close_move
        })

print(f"Signals: {len(signals)}\n")

# Split IS/OOS
split_ts = pd.Timestamp(SPLIT)
covid_a = pd.Timestamp(COVID_A)
covid_b = pd.Timestamp(COVID_B)

is_ret = [s["ret"] for s in signals if pd.Timestamp(s["date"]) < split_ts and not (covid_a <= pd.Timestamp(s["date"]) <= covid_b)]
oos_ret = [s["ret"] for s in signals if pd.Timestamp(s["date"]) >= split_ts and not (covid_a <= pd.Timestamp(s["date"]) <= covid_b)]

is_t = tstat(is_ret)
oos_t = tstat(oos_ret)

print(f"IS (2014-2021): N={len(is_ret)}, Return={np.mean(is_ret)*100:+.3f}%, t={is_t:+.2f}")
print(f"OOS (2022-2026): N={len(oos_ret)}, Return={np.mean(oos_ret)*100:+.3f}%, t={oos_t:+.2f}")
print(f"\nBonferroni (|t|>2.58): {'PASS' if abs(oos_t) > 2.58 else 'FAIL'}")
print(f"Result: {'[OK] PROFITABLE SIGNAL' if abs(oos_t) > 2.58 else '[FAIL] No edge OOS'}")
