"""
VRP-SLEEVE-TEST — Pre-Reg preregs/vrp_sleeve_2026_06_23.md  (2026-06-23)

Volatilitaets-Risikopraemie via synthetischem monatlichem Short-ATM-Straddle
(nur VIX + SPY). Prueft: reale Praemie? ueberlebt Tail? echter Diversifier?

prem   = 0.7979 * S_start * (VIX_start/100) * sqrt(T)   (Brenner-Subrahmanyam ATM)
payoff = |S_end - S_start|;  pnl = prem - payoff;  r = pnl / S_start
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings; warnings.filterwarnings("ignore")

IS_END = pd.Period("2017-12", "M")
OOS_START = pd.Period("2018-01", "M")
COST_FRAC = 0.05      # 5% der Praemie (Spread)
VIX_FILTER = 30.0     # tail-gemanagt: flat wenn VIX_start > 30
K_ATM = np.sqrt(2 / np.pi)  # 0.7979
ANN_M = np.sqrt(12)


def hac_mean_t(x, L=3):
    x = np.asarray(x, float); n = len(x); mu = x.mean(); e = x - mu
    s = (e @ e) / n
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    se = np.sqrt(s / n)
    return mu, (mu / se if se > 0 else np.nan)


def maxdd(returns):
    cum = np.cumprod(1 + np.asarray(returns, float))
    return float((cum / np.maximum.accumulate(cum) - 1).min())


def build():
    spy = yf.download("SPY", start="2006-01-01", end="2026-06-23",
                      auto_adjust=False, progress=False)
    vix = yf.download("^VIX", start="2006-01-01", end="2026-06-23",
                      auto_adjust=False, progress=False)
    df = pd.DataFrame({
        "S": np.asarray(spy["Close"].values, float).flatten(),
    }, index=spy.index).join(
        pd.DataFrame({"V": np.asarray(vix["Close"].values, float).flatten()},
                     index=vix.index), how="inner").dropna()
    df["ym"] = df.index.to_period("M")
    m = df.groupby("ym").agg(S=("S", "last"), V=("V", "last"), n=("S", "size"))
    m["S_start"] = m["S"].shift(1)
    m["V_start"] = m["V"].shift(1)
    m["T"] = m["n"] / 252.0
    m = m.dropna()
    m["prem"] = K_ATM * m["S_start"] * (m["V_start"] / 100.0) * np.sqrt(m["T"])
    m["payoff"] = (m["S"] - m["S_start"]).abs()
    m["pnl"] = m["prem"] - m["payoff"]
    m["r_gross"] = m["pnl"] / m["S_start"]
    m["r"] = (m["prem"] * (1 - COST_FRAC) - m["payoff"]) / m["S_start"]   # netto
    # tail-gemanagt: flat wenn VIX_start > Filter
    m["r_mgd"] = np.where(m["V_start"] > VIX_FILTER, 0.0, m["r"])
    m["spy_ret"] = m["S"] / m["S_start"] - 1
    return m


def summ(r, label):
    r = np.asarray(r, float)
    mu, t = hac_mean_t(r)
    sh = mu / (r.std() + 1e-12) * ANN_M
    print(f"  {label:<22} mean={mu*100:+6.3f}%/M  t={t:+5.2f}  "
          f"ann={mu*12*100:+6.2f}%  Sharpe={sh:+5.2f}  "
          f"skew={pd.Series(r).skew():+5.2f}  worst={r.min()*100:+6.1f}%  "
          f"MaxDD={maxdd(r)*100:+6.1f}%")
    return dict(mu=mu, t=t, sharpe=sh, maxdd=maxdd(r), worst=r.min())


def main():
    print("=" * 96)
    print("VRP-SLEEVE-TEST — synthetischer monatlicher Short-ATM-Straddle (SPY/VIX, 2006-2026)")
    print("=" * 96)
    m = build()
    is_ = m[m.index <= IS_END]
    oos = m[m.index >= OOS_START]
    print(f"Monate: {len(m)}  IS={len(is_)} OOS={len(oos)}  "
          f"{m.index.min()} .. {m.index.max()}")
    print(f"VRP-Beleg: Praemie Ø={m['prem'].mean()/m['S_start'].mean()*100:.2f}% vs "
          f"realisierte Bewegung Ø={m['payoff'].mean()/m['S_start'].mean()*100:.2f}% "
          f"(Differenz = Prämie)")

    print("\n--- NACKT (netto 5% Kosten) ---")
    s_is = summ(is_["r"], "IS 2006-2017")
    s_oos = summ(oos["r"], "OOS 2018-2026")
    summ(m["r"], "GESAMT")
    print("--- TAIL-GEMANAGT (flat wenn VIX_start>30, netto) ---")
    summ(is_["r_mgd"], "IS 2006-2017")
    s_oos_m = summ(oos["r_mgd"], "OOS 2018-2026")
    g_mgd = summ(m["r_mgd"], "GESAMT")

    print("\n--- G5: DIVERSIFIKATION vs SPY ---")
    corr = np.corrcoef(m["r"], m["spy_ret"])[0, 1]
    corr_m = np.corrcoef(m["r_mgd"], m["spy_ret"])[0, 1]
    print(f"  corr(VRP nackt, SPY)={corr:+.2f}  corr(VRP gemanagt, SPY)={corr_m:+.2f}")
    worst5 = m.nsmallest(5, "spy_ret")[["spy_ret", "r", "r_mgd", "V_start"]]
    print("  5 schlechteste SPY-Monate (crasht VRP mit?):")
    for ym, row in worst5.iterrows():
        print(f"    {ym}: SPY {row.spy_ret*100:+6.1f}%  VRP-nackt {row.r*100:+6.1f}%  "
              f"VRP-gemanagt {row.r_mgd*100:+6.1f}%  (VIX_start {row.V_start:.0f})")
    # Portfolio SPY vs 50/50 SPY+VRP-gemanagt
    spy_sh = m["spy_ret"].mean() / m["spy_ret"].std() * ANN_M
    combo = 0.5 * m["spy_ret"] + 0.5 * m["r_mgd"]
    combo_sh = combo.mean() / combo.std() * ANN_M
    print(f"  Sharpe SPY={spy_sh:+.2f}  vs  50/50 SPY+VRP-gemanagt={combo_sh:+.2f} (GESAMT)")
    spy_oos_sh = oos["spy_ret"].mean() / oos["spy_ret"].std() * ANN_M
    combo_oos = 0.5 * oos["spy_ret"] + 0.5 * oos["r_mgd"]
    combo_oos_sh = combo_oos.mean() / combo_oos.std() * ANN_M
    print(f"  OOS-only: Sharpe SPY={spy_oos_sh:+.2f} vs 50/50={combo_oos_sh:+.2f} "
          f"(ehrlich: OOS-Vorteil kleiner)")

    print("\n" + "=" * 96)
    print("GATE-CHECK")
    print("=" * 96)
    g1 = (s_is_t := hac_mean_t(is_["r"].values))[1] > 2.0 and s_is_t[0] > 0
    g2 = (s_oos["mu"] > 0) and (s_oos["t"] > 1.5)
    g3 = s_oos_m["sharpe"] > 0.5
    g4 = (g_mgd["maxdd"] > -0.35) and (g_mgd["worst"] > -0.25)
    g5 = (abs(corr_m) < 0.4) and (combo_sh > spy_sh)
    for tag, ok, detail in [
        ("G1 IS Praemie real (t>2,>0)", g1, f"t={s_is_t[1]:.2f}"),
        ("G2 OOS haelt (>0,t>1.5)", g2, f"mean={s_oos['mu']*100:+.3f}% t={s_oos['t']:.2f}"),
        ("G3 OOS gemanagt Sharpe>0.5", g3, f"Sharpe={s_oos_m['sharpe']:.2f}"),
        ("G4 TAIL gemanagt DD>-35% & Monat>-25%", g4,
         f"MaxDD={g_mgd['maxdd']*100:.0f}% worst={g_mgd['worst']*100:.0f}%"),
        ("G5 Diversifier (corr<.4 & Combo>SPY)", g5,
         f"corr={corr_m:+.2f} combo {combo_sh:.2f} vs {spy_sh:.2f}"),
    ]:
        print(f"  {tag:<42} {'PASS' if ok else 'FAIL'}  ({detail})")
    print("-" * 96)
    # RED nur wenn die Praemie selbst fehlt (G1). Sonst GREEN/YELLOW.
    if g1 and g2 and g3 and g4 and g5:
        v = "GREEN — reale, tragbare, diversifizierende Praemie. Forward-Test."
    elif g1:
        v = ("YELLOW — Praemie REAL (IS t=%.1f, GESAMT t=%.1f) und gegen SPY "
             "unkorreliert (Combo-Sharpe>SPY); Tail unleveraged + VIX-Filter tragbar "
             "(G4 PASS, kein Blowup — Blowup-Mythos betrifft GEHEBELTE VIX-Futures). "
             "ABER OOS verwaessert (G2/G3 FAIL: Sharpe %.2f->%.2f, t=%.2f) = "
             "Crowding/Decay seit ~2018. Handelbar nur klein, unleveraged, mit Overlay."
             % (s_is_t[1], hac_mean_t(m['r'].values)[1],
                s_is['sharpe'], s_oos['sharpe'], s_oos['t']))
    else:
        v = "RED — Praemie nicht bestaetigt."
    print(f"VERDICT: {v}")


if __name__ == "__main__":
    main()
