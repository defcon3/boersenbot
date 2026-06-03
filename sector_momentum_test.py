#!/usr/bin/env python3
"""
SEKTOR-MOMENTUM-ROTATION (Pre-Reg 2026-06-03)

HYPOTHESE: Cross-sectional 12-1M-Momentum der US-Sektor-SPDRs liefert OOS
einen risk-adjusted Edge ueber SPY-Buy&Hold. Strukturell anders als alle
bisherigen Timing-Filter (echtes Querschnittssignal statt Single-Asset-Timing).

EHRLICHE PRIOR: ~30% OOS-tauglich. Sektor-Momentum ist post-2008 stark
zerfallen; landet es im Rauschen, ist das ein sauberes Negativ-Ergebnis.

DESIGN (vor Lauf festgelegt):
  - Universum: 11 SPDR-Sektor-ETFs
      XLB XLE XLF XLI XLK XLP XLU XLV XLY XLRE XLC
    Inception-Falle: XLRE ab 2015-10, XLC ab 2018-06 -> zum Zeitpunkt t
    nur die handelbaren Tickers ranken (keine NaN-Mogelei).
  - Signal: 12-1M-Momentum = P[t-1]/P[t-13]-1 (12M-Return, juengsten Monat
    geskippt -> gegen Short-Term-Reversal). Monatsendkurse, adjusted.
  - Strategie: Top-N gleichgewichtet, monatliches Rebalance, long-only.
    N in {2,3,4} (Default 3). Position fuer Monat t+1, geformt mit Info bis t-1.
  - Benchmark: SPY-Buy&Hold (Monatsreturns).
  - IS bis 2014-12-31 / OOS ab 2015-01-01.
  - Kosten: 10 bps je Rebalance-Leg (Turnover-basiert) fuer G5.

PRE-REG-GATES (OOS):
  G1 (Risk-Adj):   OOS-Sharpe(Strat) > OOS-Sharpe(SPY)
  G2 (Signifikanz): Bootstrap auf OOS-Monats-Excess (Strat-SPY) p < 0.05 (one-sided)
  G3 (Drawdown):   OOS-MaxDD(Strat) <= OOS-MaxDD(SPY) (nicht schlechter)
  G4 (Robust):     G1 haelt fuer ALLE N in {2,3,4} (kein Single-N-Treffer)
  G5 (Kosten):     net-of-cost (10bps/Leg) OOS haelt noch G1

VERDICT:
  GREEN:  G1+G2+G3 fuer Default-N=3 PASS UND G4 (alle N) UND G5
  YELLOW: G1+G2 fuer N=3 PASS, aber G4 ODER G5 FAIL (fragil)
  RED:    G1 oder G2 fuer N=3 FAIL -> Hypothese widerlegt
"""
import warnings; warnings.filterwarnings("ignore")
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

# ===========================================================================
# CONFIG
# ===========================================================================
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
           "XLRE", "XLC"]
START = "1998-12-01"
END = "2026-06-30"
IS_END = pd.Timestamp("2014-12-31")
OOS_START = pd.Timestamp("2015-01-01")

LOOKBACK = 13   # P[t-1]/P[t-13]: 12M-Return, juengsten Monat geskippt
SKIP = 1
N_VALUES = [2, 3, 4]
N_DEFAULT = 3
COST_BPS = 10   # je Rebalance-Leg (G5)
N_BOOT = 20000

CACHE = Path(__file__).parent / "sector_momentum_ohlc.pkl"

# ===========================================================================
# HELPERS
# ===========================================================================

def fetch_monthly_close(symbols):
    """Adjusted Monatsend-Close pro Symbol (Cache wenn vorhanden)."""
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    raw = yf.download(" ".join(symbols), start=START, end=END,
                      progress=False, group_by="ticker", auto_adjust=True)
    monthly = {}
    for s in symbols:
        try:
            close = raw[s]["Close"].dropna() if len(symbols) > 1 else raw["Close"].dropna()
        except Exception:
            continue
        if len(close) > 50:
            monthly[s] = close.resample("ME").last()
    with open(CACHE, "wb") as f:
        pickle.dump(monthly, f)
    return monthly

def sharpe(monthly_ret):
    r = np.asarray(monthly_ret, float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return (r.mean() / r.std(ddof=1)) * np.sqrt(12)

def max_drawdown(monthly_ret):
    r = np.asarray(monthly_ret, float)
    r = r[~np.isnan(r)]
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())

def cagr(monthly_ret):
    r = np.asarray(monthly_ret, float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return np.nan
    total = np.prod(1 + r)
    return total ** (12 / len(r)) - 1

def bootstrap_p_greater_zero(excess, n_boot=N_BOOT, seed=42):
    """One-sided: P(mean <= 0) unter Resampling -> p-Wert fuer mean>0."""
    x = np.asarray(excess, float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float((means <= 0).mean())

# ===========================================================================
# BACKTEST
# ===========================================================================

def run_strategy(prices_df, ret_df, n_top, with_cost=False):
    """
    prices_df: index=Monatsende, cols=ETFs (adjusted close)
    ret_df:    Monatsreturns derselben ETFs
    Liefert pd.Series Monats-Strategiereturns (Holding-Monat als Index).
    """
    dates = prices_df.index
    out = {}
    prev_holdings = set()
    for i in range(LOOKBACK, len(dates) - 1):
        t = dates[i]
        hold_month = dates[i + 1]
        p_now = prices_df.iloc[i - SKIP]       # P[t-1]
        p_then = prices_df.iloc[i - LOOKBACK]  # P[t-13]
        mom = (p_now / p_then - 1)
        mom = mom[p_now.notna() & p_then.notna()].dropna()
        if len(mom) < n_top:
            continue
        picks = list(mom.sort_values(ascending=False).head(n_top).index)
        # Realisierter Holding-Return (Monat t+1), gleichgewichtet
        fwd = ret_df.loc[hold_month, picks]
        fwd = fwd.dropna()
        if len(fwd) == 0:
            continue
        ret = float(fwd.mean())
        if with_cost:
            cur = set(picks)
            changed = len(cur.symmetric_difference(prev_holdings))
            # symmetric_difference zaehlt verkaufte + gekaufte Namen
            turnover_legs = changed / n_top
            ret -= turnover_legs * (COST_BPS / 10000.0)
            prev_holdings = cur
        out[hold_month] = ret
    return pd.Series(out).sort_index()

# ===========================================================================
# MAIN
# ===========================================================================

print("=" * 80)
print("SEKTOR-MOMENTUM-ROTATION")
print("=" * 80)

print("\n[1/4] Lade Monats-Close fuer 11 Sektor-SPDRs + SPY...", flush=True)
monthly = fetch_monthly_close(SECTORS + ["SPY"])
print(f"  geladen: {sorted(monthly.keys())}")
for s in SECTORS + ["SPY"]:
    if s in monthly:
        print(f"    {s}: {monthly[s].index.min().date()} .. {monthly[s].index.max().date()} ({len(monthly[s])})")

spy = monthly.pop("SPY")
spy_ret = spy.pct_change()

prices = pd.DataFrame({s: monthly[s] for s in SECTORS if s in monthly})
ret = prices.pct_change()
print(f"\n  Sektor-Panel: {prices.shape[0]} Monate x {prices.shape[1]} ETFs")

# ---------------------------------------------------------------------------
print("\n[2/4] Backtest pro N (brutto + netto)...", flush=True)
rows = []
strat_series = {}
for n in N_VALUES:
    s_gross = run_strategy(prices, ret, n, with_cost=False)
    s_net = run_strategy(prices, ret, n, with_cost=True)
    strat_series[n] = (s_gross, s_net)

# SPY auf gleichen Holding-Index bringen
def align_spy(strat_idx):
    return spy_ret.reindex(strat_idx)

# ---------------------------------------------------------------------------
print("\n[3/4] Kennzahlen IS / OOS...\n", flush=True)
hdr = f"{'N':>2} {'phase':5} {'CAGR':>8} {'Sharpe':>7} {'MaxDD':>8} | {'SPY-CAGR':>9} {'SPY-Shrp':>8} {'SPY-DD':>8}  net-Shrp"
print(hdr)
print("-" * len(hdr))

metrics = {}
for n in N_VALUES:
    s_gross, s_net = strat_series[n]
    for phase, mask_fn in [("IS", lambda idx: idx <= IS_END),
                           ("OOS", lambda idx: idx >= OOS_START)]:
        mg = s_gross[mask_fn(s_gross.index)]
        mn = s_net[mask_fn(s_net.index)]
        sp = align_spy(mg.index)
        m = dict(
            cagr=cagr(mg), sharpe=sharpe(mg), maxdd=max_drawdown(mg),
            net_sharpe=sharpe(mn),
            spy_cagr=cagr(sp), spy_sharpe=sharpe(sp), spy_maxdd=max_drawdown(sp),
            excess=(mg.values - sp.values),
        )
        metrics[(n, phase)] = m
        print(f"{n:>2} {phase:5} {m['cagr']*100:7.2f}% {m['sharpe']:7.2f} "
              f"{m['maxdd']*100:7.2f}% | {m['spy_cagr']*100:8.2f}% {m['spy_sharpe']:8.2f} "
              f"{m['spy_maxdd']*100:7.2f}%  {m['net_sharpe']:7.2f}")

# ---------------------------------------------------------------------------
print("\n[4/4] Gates + Verdict\n", flush=True)

# Default-N Gates
md = metrics[(N_DEFAULT, "OOS")]
g1 = md["sharpe"] > md["spy_sharpe"]
p_boot = bootstrap_p_greater_zero(md["excess"])
g2 = (not np.isnan(p_boot)) and p_boot < 0.05
g3 = md["maxdd"] >= md["spy_maxdd"]   # weniger negativ = besser/gleich
# G4: G1 fuer alle N
g4 = all(metrics[(n, "OOS")]["sharpe"] > metrics[(n, "OOS")]["spy_sharpe"] for n in N_VALUES)
# G5: net OOS-Sharpe (N_DEFAULT) noch > SPY
g5 = md["net_sharpe"] > md["spy_sharpe"]

print(f"  Default N={N_DEFAULT} (OOS):")
print(f"    G1 Sharpe>SPY     : {'PASS' if g1 else 'FAIL'} ({md['sharpe']:.2f} vs {md['spy_sharpe']:.2f})")
print(f"    G2 Bootstrap p<.05: {'PASS' if g2 else 'FAIL'} (p={p_boot:.3f}, mean_excess={np.nanmean(md['excess'])*100:+.3f}%/Mt)")
print(f"    G3 MaxDD<=SPY     : {'PASS' if g3 else 'FAIL'} ({md['maxdd']*100:.1f}% vs {md['spy_maxdd']*100:.1f}%)")
print(f"    G4 G1 alle N      : {'PASS' if g4 else 'FAIL'} (" +
      ", ".join(f"N{n}:{metrics[(n,'OOS')]['sharpe']:.2f}>{metrics[(n,'OOS')]['spy_sharpe']:.2f}" for n in N_VALUES) + ")")
print(f"    G5 net-of-cost G1 : {'PASS' if g5 else 'FAIL'} (net {md['net_sharpe']:.2f} vs SPY {md['spy_sharpe']:.2f})")

print("\n--- FINAL VERDICT ---")
if g1 and g2 and g3 and g4 and g5:
    print("GREEN: Sektor-Momentum schlaegt SPY OOS risk-adjusted, robust + kostenfest.")
elif g1 and g2:
    print("YELLOW: N=3 schlaegt SPY (G1+G2), aber fragil (G4/G5 nicht beide PASS).")
else:
    print("RED: kein robuster OOS-Edge ueber SPY -> Hypothese widerlegt.")

# Speichern
out_rows = []
for (n, phase), m in metrics.items():
    out_rows.append(dict(N=n, phase=phase, cagr=m["cagr"], sharpe=m["sharpe"],
                         maxdd=m["maxdd"], net_sharpe=m["net_sharpe"],
                         spy_cagr=m["spy_cagr"], spy_sharpe=m["spy_sharpe"],
                         spy_maxdd=m["spy_maxdd"],
                         mean_excess_mo=float(np.nanmean(m["excess"]))))
pd.DataFrame(out_rows).to_csv("sector_momentum_results.csv", index=False)
print("\nGespeichert: sector_momentum_results.csv")
