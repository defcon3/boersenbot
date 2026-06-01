#!/usr/bin/env python3
"""
CREDIT-STRESS-BUY GENERALISIERUNG (Backlog-Idee 2, 2026-06-01)

Frage: Funktioniert der HYG-Stress-Buy-Mechanismus (Credit-Spread-Mean-Reversion,
der einzige bisherige Gewinner -> hyg_stress_buy_edge) auch auf VERWANDTEN,
UNABHAENGIGEN Credit-Instrumenten?

  LQD  - Investment-Grade Corporate Bonds (hoehere Qualitaet, mehr Duration)
  EMB  - Emerging-Market Sovereign Bonds (riskanter, anderes Spread-Regime)
  BKLN - Senior/Leveraged Loans (floating-rate, fast nur Credit, kaum Duration)

Disziplin (feedback_rigor_over_speed):
  - NICHTS getunt. Exakt die HYG-Edge-Variante S1 (Sizing 50/100), 20d-Holding,
    5bps Slippage, STLFSI4-Schwelle = Train-Q75 (bis 2018-12-31), instrument-unabh.
  - HYG+JNK laufen als Reproduktions-Anker mit (muessen den alten Befund zeigen).
  - OOS = 2019-2025, COVID-Akutphase (15.02.-30.04.2020) exkludiert wie im Original.
  - Kriterien: (a) Delta-Sharpe S1 vs B&H positiv, (b) Excess-AnnRet positiv,
    (c) t-stat des taeglichen Excess MIT Newey-West (lag=20, wg. overlapping
    Holding-Fenster) -> sonst ist der naive t-stat aufgeblasen.
  - Bonferroni ueber die 3 NEUEN Instrumente: alpha 0.05 / 3 = 0.0167 (einseitig,
    da gerichtete Edge-Hypothese) -> t-Schwelle ~2.13.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from fred_helper import get_series

TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")
COVID_A = pd.Timestamp("2020-02-15")
COVID_B = pd.Timestamp("2020-04-30")
SLIPPAGE = 0.0005      # 5bps round-trip
HOLDING_DAYS = 20
NORMAL, STRESS = 0.5, 1.0   # S1 Sizing 50/100

# (Ticker, yfinance-Start, Rolle)
INSTRUMENTS = [
    ("HYG",  "2007-04-11", "anchor"),   # Original-Edge
    ("JNK",  "2007-12-01", "anchor"),   # Original-Validation
    ("LQD",  "2002-07-30", "test"),     # IG Corp
    ("EMB",  "2007-12-19", "test"),     # EM Sovereign
    ("BKLN", "2011-03-03", "test"),     # Senior Loans
]

print("=" * 92)
print("CREDIT-STRESS-BUY GENERALISIERUNG: LQD / EMB / BKLN (Anker HYG/JNK)")
print("=" * 92)

# --- Signal -------------------------------------------------------------
stlfsi = get_series("STLFSI4", start="2003-01-01")
train_q75 = stlfsi[stlfsi.index <= TRAIN_END].dropna().quantile(0.75)
print(f"\nSTLFSI4 Schwelle (Train-Q75 bis 2018): {train_q75:.4f}")
print(f"STLFSI4 Datenreichweite: {stlfsi.index.min().date()} .. {stlfsi.index.max().date()}")


def make_exposure(signal, threshold, index):
    sig = signal.reindex(index, method="ffill")
    entry = (sig.shift(1) > threshold).astype(int)
    in_stress = entry.rolling(HOLDING_DAYS).sum().clip(0, 1) > 0
    return pd.Series(np.where(in_stress, STRESS, NORMAL), index=index)


def apply_strategy(asset_ret, exposure):
    gross = asset_ret * exposure.shift(1).fillna(0)
    cost = exposure.diff().abs().fillna(0) * SLIPPAGE
    return gross - cost


def metrics(r):
    r = r.dropna()
    if len(r) < 30:
        return None
    cum = (1 + r).cumprod()
    ann = cum.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    dn = r[r < 0]
    svol = dn.std() * np.sqrt(252) if len(dn) > 0 else vol
    sortino = ann / svol if svol > 0 else 0
    maxdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return dict(ann=ann, vol=vol, sharpe=sharpe, sortino=sortino, maxdd=maxdd, n=len(r))


def nw_tstat(x, lag=HOLDING_DAYS):
    """t-stat des Mittelwerts mit Newey-West HAC-Standardfehler."""
    x = np.asarray(x.dropna())
    n = len(x)
    if n < 30:
        return 0.0, 1.0
    mean = x.mean()
    dem = x - mean
    gamma0 = (dem @ dem) / n
    var = gamma0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)               # Bartlett-Kernel
        cov = (dem[k:] @ dem[:-k]) / n
        var += 2 * w * cov
    se = np.sqrt(var / n)
    t = mean / se if se > 0 else 0.0
    p_one = 1 - stats.norm.cdf(t)           # einseitig (Edge > 0)
    return t, p_one


def oos_mask(idx):
    return (idx >= TEST_START) & ~((idx >= COVID_A) & (idx <= COVID_B))


# --- Lauf ---------------------------------------------------------------
rows = []
for ticker, start, role in INSTRUMENTS:
    px = yf.download(ticker, start=start, progress=False)
    if px.empty:
        print(f"  [WARN] {ticker}: keine Daten")
        continue
    close = pd.Series(np.asarray(px["Close"]).flatten(), index=px.index)
    ret = close.pct_change()

    exp = make_exposure(stlfsi, train_q75, ret.index)
    s1 = apply_strategy(ret, exp)

    m_oos = oos_mask(s1.index)
    s1_oos, bh_oos = s1[m_oos], ret[m_oos]
    excess = (s1_oos - bh_oos).dropna()

    ms, mb = metrics(s1_oos), metrics(bh_oos)
    if ms is None or mb is None:
        print(f"  [WARN] {ticker}: zu wenig OOS-Daten")
        continue
    t, p = nw_tstat(excess)
    rows.append(dict(
        ticker=ticker, role=role,
        bh_sharpe=mb["sharpe"], s1_sharpe=ms["sharpe"],
        d_sharpe=ms["sharpe"] - mb["sharpe"],
        excess_ann=(1 + excess.mean()) ** 252 - 1,
        bh_dd=mb["maxdd"], s1_dd=ms["maxdd"],
        t=t, p=p, n=ms["n"], avg_exp=exp[m_oos].mean(),
        first=close.index.min().date(),
    ))

# --- Tabelle ------------------------------------------------------------
print("\nOOS 2019-2025 (COVID-Akut exkl.), S1 Sizing 50/100 vs Buy-and-Hold")
print("-" * 92)
print(f"{'Tkr':<5}{'Rolle':<8}{'BH-Shp':>7}{'S1-Shp':>7}{'dShp':>7}"
      f"{'ExAnn':>8}{'BH-DD':>8}{'S1-DD':>8}{'NW-t':>7}{'p(1s)':>8}{'AvgExp':>8}")
print("-" * 92)
for r in rows:
    print(f"{r['ticker']:<5}{r['role']:<8}{r['bh_sharpe']:>7.2f}{r['s1_sharpe']:>7.2f}"
          f"{r['d_sharpe']:>+7.2f}{r['excess_ann']*100:>+7.2f}%"
          f"{r['bh_dd']*100:>+7.1f}%{r['s1_dd']*100:>+7.1f}%"
          f"{r['t']:>+7.2f}{r['p']:>8.3f}{r['avg_exp']*100:>7.1f}%")

# --- Verdikt ------------------------------------------------------------
BONF_ALPHA = 0.05 / 3   # 3 neue Instrumente
print("\n" + "=" * 92)
print("VERDIKT")
print("=" * 92)
print(f"Bonferroni-Schwelle (3 Tests, einseitig): p < {BONF_ALPHA:.4f}  (NW-t > ~2.13)")

anchors = [r for r in rows if r["role"] == "anchor"]
tests = [r for r in rows if r["role"] == "test"]

print("\nAnker (Reproduktion des alten Befunds):")
for r in anchors:
    ok = "edge" if r["d_sharpe"] > 0 and r["excess_ann"] > 0 else "kein-edge"
    print(f"  {r['ticker']}: dSharpe {r['d_sharpe']:+.3f}, ExAnn {r['excess_ann']*100:+.2f}%  -> {ok}")

print("\nTest-Instrumente (Generalisierung):")
passed = 0
for r in tests:
    crit_a = r["d_sharpe"] > 0
    crit_b = r["excess_ann"] > 0
    crit_c = r["p"] < BONF_ALPHA
    full = crit_a and crit_b and crit_c
    passed += full
    print(f"  {r['ticker']:<5} dSharpe {r['d_sharpe']:+.3f} [{ 'OK' if crit_a else 'x'}] | "
          f"ExAnn {r['excess_ann']*100:+.2f}% [{'OK' if crit_b else 'x'}] | "
          f"NW-p {r['p']:.3f} [{'OK' if crit_c else 'x'}]  => "
          f"{'GENERALISIERT' if full else 'nicht robust'}")

print(f"\n{passed}/{len(tests)} Test-Instrumente bestehen alle 3 Gates inkl. Bonferroni.")
if passed == len(tests):
    print("=> Mechanismus generalisiert sauber -> echter Credit-Spread-MR-Edge.")
elif passed > 0:
    print("=> Teilweise -> instrumentspezifisch, kein universeller Mechanismus.")
else:
    print("=> Generalisiert NICHT -> HYG-Edge war wahrscheinlich Overfit/Glueck.")
