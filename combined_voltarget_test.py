#!/usr/bin/env python3
"""
VOL-TARGETING auf der Combined-Strategie (Hybrid-SPY + HYG) — allerletzter Check.

FRAGE (Grok): Schlaegt eine VOL-GETARGETETE Leverage (Exposure invers zur
realisierten Vol, gedeckelt) die KONSTANTE Leverage aus combined_leverage_test.py?
Rettet dynamisches Sizing den Sharpe, ohne den Tail aufzublaehen?

MECHANIK:
  L_t = clip(target_vol / realized_vol_{t-1}, 0, L_MAX)     # kein Lookahead
  realized_vol = trailing 20d-Std der Combined-Tagesreturns, annualisiert
  -> in ruhigen Phasen mehr Hebel, im Stress automatisch runter.

EHRLICH (sonst schmeichelt sich Vol-Targeting selbst):
  - Financing auf geliehenen Teil max(L-1,0) x (DFF + 1.5% Spread), wie zuvor.
  - TURNOVER-Kosten 5 bps auf |dL_t| — Vol-Targeting handelt staendig die
    Exposure nach; das zu ignorieren waere geschoent.
  - Fairer Kernvergleich: Vol-Target vs KONSTANTE Leverage bei GLEICHER
    realisierter Vol (nur dann zeigt sich, ob dynamisches Sizing wirklich hilft).
  - Weiterhin KEINE Edge-Pre-Reg, nur Sizing-Exploration. Live => eigene Pre-Reg.
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
BROKER_SPREAD = 0.015
VOL_WINDOW = 20
L_MAX = 3.0
TURN_BPS = 0.0005
TARGET_VOLS = [0.08, 0.10, 0.12, 0.15]

print("=" * 88)
print("VOL-TARGETING — Combined 50/50 (Hybrid-SPY + HYG)")
print("=" * 88)

# --- Daten + Combined-Stream (spiegelt combined_leverage_test.py) -----------
print("\n[1/3] Lade Daten + baue Combined-50/50 (net)...", flush=True)
spy = yf.download("SPY", start="2007-04-11", progress=False)
spy_close = pd.Series(np.asarray(spy["Close"]).flatten(), index=spy.index)
spy_ret = spy_close.pct_change()
vix = yf.download("^VIX", start="2007-04-11", progress=False)["Close"]
vix = pd.Series(np.asarray(vix).flatten(), index=vix.index)
hyg = yf.download("HYG", start="2007-04-11", progress=False)
hyg_close = pd.Series(np.asarray(hyg["Close"]).flatten(), index=hyg.index)
hyg_ret = hyg_close.pct_change()
stlfsi = get_series("STLFSI4", start="2003-01-01")
dff = get_series("DFF", start="2007-01-01") / 100.0

ma50, ma200 = spy_close.rolling(50).mean(), spy_close.rolling(200).mean()
uptrend = (ma50 > ma200).astype(int)
vix_aligned = vix.reindex(spy_close.index, method="ffill")
vix_norm = ((vix_aligned - vix_aligned.rolling(60).mean()) /
            (vix_aligned.rolling(60).std() + 1e-6)) * 0.1
size = (1 - vix_norm.clip(-0.5, 0.5)).clip(0.2, 1.0)
hybrid_exp = uptrend * size
hybrid_net = spy_ret * hybrid_exp.shift(1).fillna(0) - hybrid_exp.diff().abs().fillna(0) * 0.0005

threshold = stlfsi[stlfsi.index <= TRAIN_END].dropna().quantile(0.75)
stlfsi_d = stlfsi.reindex(hyg_ret.index, method="ffill")
in_stress = (stlfsi_d.shift(1) > threshold).astype(int).rolling(20).sum().clip(0, 1) > 0
hyg_exp = pd.Series(np.where(in_stress, 1.0, 0.5), index=hyg_ret.index)
hyg_net = hyg_ret * hyg_exp.shift(1).fillna(0) - hyg_exp.diff().abs().fillna(0) * 0.0005

aligned = pd.concat([hybrid_net, hyg_net], axis=1, keys=["hybrid", "hyg"]).dropna()
combo = aligned["hybrid"] * 0.5 + aligned["hyg"] * 0.5
rf_daily = (dff.reindex(combo.index, method="ffill").fillna(0)) / 252.0

# realisierte Vol (annualisiert), 1 Tag verzoegert -> kein Lookahead
realized_vol = combo.rolling(VOL_WINDOW).std().shift(1) * np.sqrt(252)


def metrics(r):
    r = r.dropna()
    cum = (1 + r).cumprod()
    ann = cum.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    dn = r[r < 0].std() * np.sqrt(252)
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"ann": ann, "vol": vol, "sharpe": ann / vol if vol else 0,
            "sortino": ann / dn if dn else 0, "mdd": mdd}


def const_lever(L):
    fin = (L - 1.0) * (rf_daily + BROKER_SPREAD / 252.0)
    return L * combo - fin


def vol_target(target):
    L = (target / realized_vol).clip(0, L_MAX).fillna(0.0)
    fin = np.maximum(L - 1.0, 0.0) * (rf_daily + BROKER_SPREAD / 252.0)
    turn = L.diff().abs().fillna(0.0) * TURN_BPS
    ret = L * combo - fin - turn
    return ret, L


def windowed(s, include_covid):
    s = s[s.index >= TEST_START]
    if not include_covid:
        s = s[~((s.index >= COVID_A) & (s.index <= COVID_B))]
    return s


# --- Auswertung ------------------------------------------------------------
for include_covid in (False, True):
    tag = "OOS MIT COVID (Tail-Stress)" if include_covid else "OOS OHNE COVID (Haupt)"
    print(f"\n[2/3] {tag}", flush=True)
    spy_m = metrics(windowed(spy_ret, include_covid))
    print(f"  {'Variante':<26} {'avgL':>5} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} "
          f"{'Sortino':>8} {'MaxDD':>9}")
    print("  " + "-" * 78)
    print(f"  {'SPY Buy&Hold (Bench)':<26} {'-':>5} {spy_m['ann']*100:>+7.2f}% "
          f"{spy_m['vol']*100:>6.2f}% {spy_m['sharpe']:>7.2f} {spy_m['sortino']:>8.2f} "
          f"{spy_m['mdd']*100:>+8.2f}%")
    print(f"  {'Combined x1.0 (unlev.)':<26} {1.0:>5.1f} ", end="")
    m = metrics(windowed(combo, include_covid))
    print(f"{m['ann']*100:>+7.2f}% {m['vol']*100:>6.2f}% {m['sharpe']:>7.2f} "
          f"{m['sortino']:>8.2f} {m['mdd']*100:>+8.2f}%")
    for tv in TARGET_VOLS:
        ret, L = vol_target(tv)
        m = metrics(windowed(ret, include_covid))
        avgL = windowed(L, include_covid).mean()
        dd_flag = "  (!) DD > SPY" if m["mdd"] < spy_m["mdd"] else ""
        print(f"  VolTarget {tv*100:>4.0f}% (Lmax{L_MAX:.0f})    {avgL:>5.2f} "
              f"{m['ann']*100:>+7.2f}% {m['vol']*100:>6.2f}% {m['sharpe']:>7.2f} "
              f"{m['sortino']:>8.2f} {m['mdd']*100:>+8.2f}%{dd_flag}")

# --- Fairer Kernvergleich: gleiche realisierte Vol -------------------------
print("\n[3/3] Fairer Vergleich bei GLEICHER realisierter Vol (OOS o.COVID)", flush=True)
print("  Frage: bringt dynamisches Sizing bei gleicher Vol mehr Sharpe als fixer Hebel?\n")
print(f"  {'Variante':<28} {'Vol':>7} {'Sharpe':>7} {'MaxDD':>9}")
print("  " + "-" * 56)
# Vol-Target 15% (hoechste, kommt SPY-Vol am naechsten) vs konstante Leverage
# mit GLEICHER realisierter OOS-Vol
ret_vt, L_vt = vol_target(0.15)
m_vt = metrics(windowed(ret_vt, False))
target_vol = m_vt["vol"]
# konstantes L finden, das dieselbe realisierte OOS-Vol erzeugt
base_vol = metrics(windowed(combo, False))["vol"]
L_match = target_vol / base_vol
m_cl = metrics(windowed(const_lever(L_match), False))
print(f"  {'VolTarget 15% (dynamisch)':<28} {m_vt['vol']*100:>6.2f}% "
      f"{m_vt['sharpe']:>7.2f} {m_vt['mdd']*100:>+8.2f}%")
print(f"  {f'Konstant x{L_match:.2f} (statisch)':<28} {m_cl['vol']*100:>6.2f}% "
      f"{m_cl['sharpe']:>7.2f} {m_cl['mdd']*100:>+8.2f}%")
better = m_vt["sharpe"] > m_cl["sharpe"]
print(f"\n  => Dynamisches Sizing {'BESSER' if better else 'NICHT besser'} bei gleicher Vol "
      f"(Sharpe {m_vt['sharpe']:.2f} vs {m_cl['sharpe']:.2f})")
print("\n  Einordnung: Vol-Targeting senkt Exposure im Stress (besserer Tail), aber")
print("  erzeugt keinen neuen Edge. Ob es SPY absolut schlaegt, haengt allein am Lmax")
print("  — und hoher Lmax holt das Tail-Risiko zurueck. Kein Deploy-Freibrief.")
