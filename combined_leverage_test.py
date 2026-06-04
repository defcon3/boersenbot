#!/usr/bin/env python3
"""
LEVERAGE-VARIANTE auf der Combined-Strategie (Hybrid-SPY + HYG), Grok-Vorschlag.

FRAGE: Die Combined-50/50 hat niedrige Vol + kleinen Drawdown. Schlaegt eine
gehebelte Version (1.25x..2x) den SPY ABSOLUT, ohne dessen Drawdown zu
ueberschreiten? (Grok: "1.5-2x bei niedriger Vol koennte SPY absolut UND
risk-adjusted schlagen.")

EHRLICHE EINORDNUNG (zuerst lesen):
  - Das ist KEINE neue Edge-Pre-Reg. Die Combined-Strategie ist bereits
    getestet/deployed. Leverage ERZEUGT keinen Edge, sie SKALIERT ihn:
    Sharpe bleibt ~gleich (minus Financing-Drag), MaxDD skaliert ~linear mit L.
  - Konstante Tages-Leverage hat Volatility-Drag (Pfadabhaengigkeit) und
    Tail-Risiko: im Stress (COVID) wird der Drawdown ueberproportional.
  - Financing wird explizit abgezogen (geliehener Teil (L-1) kostet
    Risk-free + Broker-Spread). Vergleich daher ehrlich, nicht geschoent.
  - Ein Live-Einsatz braeuchte eine EIGENE Forward-Test-/Risk-Pre-Reg
    (Margin-Calls, Gap-Risiko, Rebalancing-Frequenz). Dies hier ist eine
    Sizing-EXPLORATION, kein Deploy-Freibrief.
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
BROKER_SPREAD = 0.015          # 1.5% ueber Risk-free fuer geliehenes Kapital
LEVERAGES = [1.0, 1.25, 1.5, 1.75, 2.0]

print("=" * 84)
print("LEVERAGE-VARIANTE — Combined 50/50 (Hybrid-SPY + HYG)")
print("=" * 84)

# --- Daten + Strategie-Streams (spiegelt combined_strategy_test.py) ---------
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

# Risk-free (Fed Funds effektiv) fuer Financing
dff = get_series("DFF", start="2007-01-01") / 100.0    # in Dezimal p.a.

# Hybrid-SPY
ma50, ma200 = spy_close.rolling(50).mean(), spy_close.rolling(200).mean()
uptrend = (ma50 > ma200).astype(int)
vix_aligned = vix.reindex(spy_close.index, method="ffill")
vix_norm = ((vix_aligned - vix_aligned.rolling(60).mean()) /
            (vix_aligned.rolling(60).std() + 1e-6)) * 0.1
size = (1 - vix_norm.clip(-0.5, 0.5)).clip(0.2, 1.0)
hybrid_exp = uptrend * size
hybrid_net = spy_ret * hybrid_exp.shift(1).fillna(0) - hybrid_exp.diff().abs().fillna(0) * 0.0005

# HYG-Strategy
threshold = stlfsi[stlfsi.index <= TRAIN_END].dropna().quantile(0.75)
stlfsi_d = stlfsi.reindex(hyg_ret.index, method="ffill")
in_stress = (stlfsi_d.shift(1) > threshold).astype(int).rolling(20).sum().clip(0, 1) > 0
hyg_exp = pd.Series(np.where(in_stress, 1.0, 0.5), index=hyg_ret.index)
hyg_net = hyg_ret * hyg_exp.shift(1).fillna(0) - hyg_exp.diff().abs().fillna(0) * 0.0005

aligned = pd.concat([hybrid_net, hyg_net], axis=1, keys=["hybrid", "hyg"]).dropna()
combo = aligned["hybrid"] * 0.5 + aligned["hyg"] * 0.5
rf_daily = (dff.reindex(combo.index, method="ffill").fillna(0)) / 252.0


def metrics(r):
    r = r.dropna()
    cum = (1 + r).cumprod()
    ann = cum.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    dn = r[r < 0].std() * np.sqrt(252)
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"ann": ann, "vol": vol, "sharpe": ann / vol if vol else 0,
            "sortino": ann / dn if dn else 0, "mdd": mdd, "n": len(r)}


def lever(combo_r, L):
    """Konstante Tages-Leverage L mit Financing auf den geliehenen Teil (L-1)."""
    fin = (L - 1.0) * (rf_daily + BROKER_SPREAD / 252.0)
    return L * combo_r - fin


def windowed(s, include_covid):
    s = s[s.index >= TEST_START]
    if not include_covid:
        s = s[~((s.index >= COVID_A) & (s.index <= COVID_B))]
    return s


# --- Auswertung: OOS ohne COVID (Haupt) + mit COVID (Tail-Stress) ----------
for include_covid in (False, True):
    tag = "OOS MIT COVID (Tail-Stress)" if include_covid else "OOS OHNE COVID (Haupt)"
    print(f"\n[2/3] {tag}", flush=True)
    spy_m = metrics(windowed(spy_ret, include_covid))
    print(f"  {'Strategie':<24} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>9}")
    print("  " + "-" * 70)
    print(f"  {'SPY Buy&Hold (Bench)':<24} {spy_m['ann']*100:>+7.2f}% {spy_m['vol']*100:>6.2f}% "
          f"{spy_m['sharpe']:>7.2f} {spy_m['sortino']:>8.2f} {spy_m['mdd']*100:>+8.2f}%")
    for L in LEVERAGES:
        m = metrics(windowed(lever(combo, L), include_covid))
        beats_ret = m["ann"] > spy_m["ann"]
        dd_ok = m["mdd"] > spy_m["mdd"]      # weniger negativ = besser
        flag = ""
        if beats_ret and dd_ok:
            flag = "  <- schlaegt SPY-Return UND haelt DD"
        elif not dd_ok:
            flag = "  (!) DD schlechter als SPY"
        print(f"  Combined x{L:<22.2f} {m['ann']*100:>+7.2f}% {m['vol']*100:>6.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['mdd']*100:>+8.2f}%{flag}")

# --- Fazit -----------------------------------------------------------------
print("\n[3/3] Einordnung", flush=True)
spy_noc = metrics(windowed(spy_ret, False))
# hoechstes L, das OHNE COVID noch besseren MaxDD als SPY hat
safe_L = None
beat_L = None
for L in LEVERAGES:
    m = metrics(windowed(lever(combo, L), False))
    if m["mdd"] > spy_noc["mdd"]:
        safe_L = L
    if m["ann"] > spy_noc["ann"] and beat_L is None:
        beat_L = L
print("=" * 84)
print(f"  SPY (OOS o.COVID): AnnRet {spy_noc['ann']*100:+.2f}%  Sharpe {spy_noc['sharpe']:.2f}  "
      f"MaxDD {spy_noc['mdd']*100:+.2f}%")
print(f"  Hoechstes L mit besserem MaxDD als SPY: x{safe_L}")
print(f"  Kleinstes L, das SPY-Return schlaegt:   {'x'+str(beat_L) if beat_L else 'keines im Raster'}")
print("\n  WICHTIG: Sharpe bleibt ueber alle L praktisch konstant (Leverage skaliert,")
print("  erzeugt keinen Edge). Im COVID-Tail verschlechtert sich der Drawdown")
print("  ueberproportional. Ein Live-Einsatz braucht eigene Forward-/Risk-Pre-Reg.")
