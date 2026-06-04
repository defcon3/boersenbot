#!/usr/bin/env python3
"""
GOLD-SLEEVE als dritter Combined-Baustein
Pre-Reg: preregs/gold_sleeve_combined_2026_06_04.md (Commit-Anchor f3692f51)

P_g = (1-g) * Combined-50/50 + g * GLD_net   (g in {10,20,30}%)
Combined = Hybrid-SPY (MA50/200+VIX) + HYG-Stress (STLFSI4-Q75), net,
identisch zu combined_strategy_test.py.

GATES (alle 5 noetig):
  G1 g=20% OOS o.COVID: Sharpe(P20)>Sharpe(Combined) UND Sortino>=
  G2 MaxDD(P20) >= MaxDD(Combined)  (kein DD-Schaden)
  G3 Sharpe-Lift haelt bei g=10% UND g=30%  (Gewichts-Robustheit)
  G4 Korr(GLD,Combined)<0.40 UND COVID-DD nicht schlechter
  G5 HELD-OUT Gold-Baer 2013-2018: Sharpe(P20) >= Sharpe(Combined) - 0.15
     (Gold darf auch im eigenen Baerenmarkt die Combined nicht schaedigen,
      sonst war der OOS-Nutzen nur Bull-Beta = Period-Mining)
"""
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd
import yfinance as yf
from fred_helper import get_series

TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2025-12-31")
CTRL_START = pd.Timestamp("2013-01-01")     # Gold-Baerenmarkt, held-out
CTRL_END = pd.Timestamp("2018-12-31")
COVID_A = pd.Timestamp("2020-02-15")
COVID_B = pd.Timestamp("2020-04-30")
WEIGHTS = [0.10, 0.20, 0.30]
PRIMARY_G = 0.20
SLIP = 0.0005
N_BOOT = 2000
EXP_BLOCK = 21
SEED = 20260604
RNG = np.random.default_rng(SEED)
TD = 252

print("=" * 86)
print("GOLD-SLEEVE — dritter Combined-Baustein (Hybrid-SPY + HYG + GLD)")
print("=" * 86)

# --- Daten + Combined (spiegelt combined_strategy_test.py) ------------------
print("\n[1/4] Lade Daten + baue Combined-50/50 (net) + GLD...", flush=True)
spy = yf.download("SPY", start="2007-04-11", progress=False)
spy_close = pd.Series(np.asarray(spy["Close"]).flatten(), index=spy.index)
spy_ret = spy_close.pct_change()
vix = yf.download("^VIX", start="2007-04-11", progress=False)["Close"]
vix = pd.Series(np.asarray(vix).flatten(), index=vix.index)
hyg = yf.download("HYG", start="2007-04-11", progress=False)
hyg_close = pd.Series(np.asarray(hyg["Close"]).flatten(), index=hyg.index)
hyg_ret = hyg_close.pct_change()
gld = yf.download("GLD", start="2007-04-11", progress=False)
gld_close = pd.Series(np.asarray(gld["Close"]).flatten(), index=gld.index)
gld_ret = gld_close.pct_change()
stlfsi = get_series("STLFSI4", start="2003-01-01")

ma50, ma200 = spy_close.rolling(50).mean(), spy_close.rolling(200).mean()
uptrend = (ma50 > ma200).astype(int)
vix_aligned = vix.reindex(spy_close.index, method="ffill")
vix_norm = ((vix_aligned - vix_aligned.rolling(60).mean()) /
            (vix_aligned.rolling(60).std() + 1e-6)) * 0.1
size = (1 - vix_norm.clip(-0.5, 0.5)).clip(0.2, 1.0)
hybrid_exp = uptrend * size
hybrid_net = spy_ret * hybrid_exp.shift(1).fillna(0) - hybrid_exp.diff().abs().fillna(0) * SLIP

threshold = stlfsi[stlfsi.index <= TRAIN_END].dropna().quantile(0.75)
stlfsi_d = stlfsi.reindex(hyg_ret.index, method="ffill")
in_stress = (stlfsi_d.shift(1) > threshold).astype(int).rolling(20).sum().clip(0, 1) > 0
hyg_exp = pd.Series(np.where(in_stress, 1.0, 0.5), index=hyg_ret.index)
hyg_net = hyg_ret * hyg_exp.shift(1).fillna(0) - hyg_exp.diff().abs().fillna(0) * SLIP

aligned = pd.concat([hybrid_net, hyg_net, gld_ret], axis=1,
                    keys=["hybrid", "hyg", "gld"]).dropna()
combo = aligned["hybrid"] * 0.5 + aligned["hyg"] * 0.5
gld_r = aligned["gld"]
print(f"  Combined-Tage: {len(combo)}  {combo.index.min().date()}..{combo.index.max().date()}")


def blend(g):
    """P_g daily fixed-weight; gold-Leg-Rebalancing-Slippage konservativ."""
    p_gross = (1 - g) * combo + g * gld_r
    gold_turn = g * (gld_r - p_gross).abs()      # taegl. Rebal-Trade im Gold-Leg
    return p_gross - gold_turn * SLIP


def metrics(r):
    r = r.dropna()
    cum = (1 + r).cumprod()
    ann = cum.iloc[-1] ** (TD / len(r)) - 1
    vol = r.std() * np.sqrt(TD)
    dn = r[r < 0].std() * np.sqrt(TD)
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {"ann": ann, "vol": vol, "sharpe": ann / vol if vol else 0,
            "sortino": ann / dn if dn else 0, "mdd": mdd, "n": len(r)}


def win(s, start, end, drop_covid):
    s = s[(s.index >= start) & (s.index <= end)]
    if drop_covid:
        s = s[~((s.index >= COVID_A) & (s.index <= COVID_B))]
    return s


def sb_idx(n, blk, rng):
    p = 1.0 / blk
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    coin, jumps = rng.random(n) < p, rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = jumps[t] if coin[t] else (idx[t - 1] + 1) % n
    return idx


def sharpe_arr(a):
    return a.mean() / a.std(ddof=1) * np.sqrt(TD) if a.std(ddof=1) else 0.0


def boot_sharpe_diff(p, c):
    """gepaarter Stationary-Bootstrap auf Sharpe(P)-Sharpe(Combined)."""
    p, c = p.dropna().to_numpy(), c.dropna().to_numpy()
    n = len(p)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ix = sb_idx(n, EXP_BLOCK, RNG)
        diffs[b] = sharpe_arr(p[ix]) - sharpe_arr(c[ix])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    frac = np.mean(diffs <= 0)
    return lo, hi, 2 * min(frac, 1 - frac)


def table(label, start, end, drop_covid):
    print(f"\n  --- {label} ---")
    c = metrics(win(combo, start, end, drop_covid))
    print(f"  {'Variante':<22} {'AnnRet':>8} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>9}")
    print("  " + "-" * 66)
    print(f"  {'Combined (g=0)':<22} {c['ann']*100:>+7.2f}% {c['vol']*100:>6.2f}% "
          f"{c['sharpe']:>7.2f} {c['sortino']:>8.2f} {c['mdd']*100:>+8.2f}%")
    out = {"combo": c, "pg": {}}
    for g in WEIGHTS:
        m = metrics(win(blend(g), start, end, drop_covid))
        out["pg"][g] = m
        d = "  +DD-ok" if m["mdd"] >= c["mdd"] else "  (!) DD schlechter"
        print(f"  Combined+Gold {g*100:>2.0f}%   {m['ann']*100:>+7.2f}% {m['vol']*100:>6.2f}% "
              f"{m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['mdd']*100:>+8.2f}%{d}")
    return out


# --- Drei Fenster ----------------------------------------------------------
print("\n[2/4] Performance ueber die drei vorregistrierten Fenster", flush=True)
oos = table("OOS 2019-2025 OHNE COVID (Haupt)", TEST_START, TEST_END, True)
oos_c = table("OOS 2019-2025 MIT COVID (Stress)", TEST_START, TEST_END, False)
ctrl = table("KONTROLLE 2013-2018 (Gold-Baerenmarkt, held-out)", CTRL_START, CTRL_END, False)

# --- Bootstrap auf den primaeren Sharpe-Lift -------------------------------
print("\n[3/4] Bootstrap-KI Sharpe-Lift (g=20%, OOS o.COVID)", flush=True)
lo, hi, p = boot_sharpe_diff(win(blend(PRIMARY_G), TEST_START, TEST_END, True),
                             win(combo, TEST_START, TEST_END, True))
print(f"  Sharpe(P20)-Sharpe(Combined) = {oos['pg'][0.20]['sharpe']-oos['combo']['sharpe']:+.3f}"
      f"  95%-KI [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}")

# --- Korrelation -----------------------------------------------------------
corr_oos = float(win(gld_r, TEST_START, TEST_END, True).corr(win(combo, TEST_START, TEST_END, True)))
print(f"  Korr(GLD, Combined) OOS o.COVID = {corr_oos:+.3f}")

# --- GATES -----------------------------------------------------------------
print("\n[4/4] GATE-CHECK", flush=True)
c0, p20 = oos["combo"], oos["pg"][0.20]
g1 = (p20["sharpe"] > c0["sharpe"]) and (p20["sortino"] >= c0["sortino"])
g2 = p20["mdd"] >= c0["mdd"]
g3 = all(oos["pg"][g]["sharpe"] > oos["combo"]["sharpe"] for g in (0.10, 0.30))
g4 = (corr_oos < 0.40) and (oos_c["pg"][0.20]["mdd"] >= oos_c["combo"]["mdd"])
g5 = ctrl["pg"][0.20]["sharpe"] >= ctrl["combo"]["sharpe"] - 0.15

print("=" * 86)
print(f"  G1 Risk-adj Lift (g20, OOS): Sharpe {p20['sharpe']:.2f} vs {c0['sharpe']:.2f}, "
      f"Sortino {p20['sortino']:.2f} vs {c0['sortino']:.2f}  -> {'PASS' if g1 else 'FAIL'}")
print(f"  G2 Kein DD-Schaden:          MaxDD {p20['mdd']*100:+.1f}% vs {c0['mdd']*100:+.1f}%  "
      f"-> {'PASS' if g2 else 'FAIL'}")
print(f"  G3 Gewichts-Robustheit:      Lift bei 10% & 30%  -> {'PASS' if g3 else 'FAIL'}")
print(f"  G4 Echte Diversifikation:    Korr {corr_oos:+.2f}<0.40 & COVID-DD ok  "
      f"-> {'PASS' if g4 else 'FAIL'}")
print(f"  G5 Gold-Baer-Kontrolle:      Sharpe {ctrl['pg'][0.20]['sharpe']:.2f} vs "
      f"{ctrl['combo']['sharpe']:.2f} (Tol -0.15)  -> {'PASS' if g5 else 'FAIL'}")
all_pass = g1 and g2 and g3 and g4 and g5
print(f"\n  GESAMT: {'GRUEN — alle 5 Gates' if all_pass else 'RED — mindestens 1 Gate FAIL'}")

# --- Export ----------------------------------------------------------------
def pack(m):
    return {k: round(v * 100, 2) if k in ("ann", "vol", "mdd") else round(v, 3)
            for k, v in m.items() if k != "n"}

res = {
    "primary_g": PRIMARY_G, "weights": WEIGHTS,
    "oos_combo": pack(oos["combo"]), "oos_p20": pack(oos["pg"][0.20]),
    "oos_p10": pack(oos["pg"][0.10]), "oos_p30": pack(oos["pg"][0.30]),
    "covid_combo": pack(oos_c["combo"]), "covid_p20": pack(oos_c["pg"][0.20]),
    "ctrl_combo": pack(ctrl["combo"]), "ctrl_p20": pack(ctrl["pg"][0.20]),
    "sharpe_lift": round(p20["sharpe"] - c0["sharpe"], 3),
    "sharpe_lift_ci": [round(lo, 3), round(hi, 3)], "sharpe_lift_p": round(p, 3),
    "corr_gld_combo": round(corr_oos, 3),
    "gates": {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4), "G5": bool(g5)},
    "verdict": "GREEN" if all_pass else "RED",
}
with open("gold_sleeve_results.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)
print("\nGespeichert: gold_sleeve_results.json")
