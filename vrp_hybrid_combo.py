"""
VRP + HYBRID — 2-Sleeve-Portfolio-Test (2026-06-23)
Backlog-Item: hebt die Kombination Hybrid-Risk-System + VRP-Sleeve den
Portfolio-Sharpe real — auch OOS (wo VRP verwaessert ist)?

Disziplin:
  - Gewichte werden auf IS gewaehlt, OOS NUR evaluiert (kein OOS-Tuning).
  - Hybrid lookahead-frei (Position aus Signal von t-1).
  - VRP-Sleeve = gemanagt (VIX>30-Filter), Definition aus vrp_sleeve_test.py importiert.
  - Gemeinsames Fenster, monatliches Rebalancing auf Zielgewichte.

Entscheidungsregel: Kombination lohnt, wenn OOS Sharpe(Combo @ IS-Gewicht) >
Sharpe(Hybrid allein) UND MaxDD nicht wesentlich schlechter.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import warnings; warnings.filterwarnings("ignore")

from vrp_sleeve_test import build as vrp_build, maxdd, hac_mean_t, ANN_M

IS_END = pd.Period("2017-12", "M")
OOS_START = pd.Period("2018-01", "M")


def hybrid_monthly():
    """Lookahead-freie Hybrid-Monatsrenditen (MA50/200 + VIX-Sizing) ueber 2006-2026."""
    spy = yf.download("SPY", start="2006-01-01", end="2026-06-23",
                      auto_adjust=True, progress=False)
    vix = yf.download("^VIX", start="2006-01-01", end="2026-06-23",
                      auto_adjust=True, progress=False)
    df = pd.DataFrame({"c": np.asarray(spy["Close"].values, float).flatten()},
                      index=spy.index).join(
        pd.DataFrame({"v": np.asarray(vix["Close"].values, float).flatten()},
                     index=vix.index), how="inner").dropna()
    c = df["c"].values
    v = df["v"].values
    ret = np.empty(len(c)); ret[0] = 0.0
    ret[1:] = c[1:] / c[:-1] - 1
    ma50 = pd.Series(c).rolling(50).mean().values
    ma200 = pd.Series(c).rolling(200).mean().values
    uptrend = (ma50 > ma200).astype(float)
    vix_mean = np.nanmean(v[:len(v) // 2])
    vix_norm = np.clip((v - vix_mean) / (vix_mean + 1) * 0.1, -0.5, 0.5)
    pos = uptrend * np.clip(1 - vix_norm, 0.2, 1.0)
    # lookahead-frei: Rendite an Tag t nutzt Position aus t-1
    hyb = np.empty(len(c)); hyb[:] = np.nan
    hyb[1:] = ret[1:] * pos[:-1]
    df["hyb"] = hyb
    df["spy_ret"] = ret
    df["ym"] = df.index.to_period("M")
    g = df.dropna().groupby("ym")
    monthly = pd.DataFrame({
        "hybrid": g["hyb"].apply(lambda x: (1 + x).prod() - 1),
        "spy": g["spy_ret"].apply(lambda x: (1 + x).prod() - 1),
    })
    return monthly


def stats(r):
    r = np.asarray(r, float)
    mu = r.mean()
    return dict(ann=mu * 12 * 100, sharpe=mu / (r.std() + 1e-12) * ANN_M,
                maxdd=maxdd(r) * 100, worst=r.min() * 100,
                skew=float(pd.Series(r).skew()))


def show(label, r):
    s = stats(r)
    print(f"  {label:<28} ann={s['ann']:+6.2f}%  Sharpe={s['sharpe']:+5.2f}  "
          f"MaxDD={s['maxdd']:+6.1f}%  worst={s['worst']:+6.1f}%  skew={s['skew']:+5.2f}")
    return s


def main():
    print("=" * 92)
    print("VRP + HYBRID — 2-Sleeve-Portfolio (monatlich, 2006-2026)")
    print("=" * 92)
    vrp = vrp_build()[["r_mgd"]].rename(columns={"r_mgd": "vrp"})
    hyb = hybrid_monthly()
    d = hyb.join(vrp, how="inner").dropna()
    is_ = d[d.index <= IS_END]
    oos = d[d.index >= OOS_START]
    print(f"Gemeinsame Monate: {len(d)}  IS={len(is_)} OOS={len(oos)}  "
          f"{d.index.min()} .. {d.index.max()}")
    print(f"corr(Hybrid, VRP): IS={np.corrcoef(is_.hybrid, is_.vrp)[0,1]:+.2f}  "
          f"OOS={np.corrcoef(oos.hybrid, oos.vrp)[0,1]:+.2f}")

    # --- IS: optimales VRP-Gewicht (Grid) + inverse-vol ---
    grid = np.round(np.arange(0.0, 0.61, 0.05), 2)
    is_sh = {w: stats((1 - w) * is_.hybrid + w * is_.vrp)["sharpe"] for w in grid}
    w_opt = max(is_sh, key=is_sh.get)
    inv_h, inv_v = 1 / is_.hybrid.std(), 1 / is_.vrp.std()
    w_iv = min(0.6, inv_v / (inv_h + inv_v))  # inverse-vol, gedeckelt

    print(f"\n--- IS-gewaehlte Gewichte (NICHT auf OOS getunt) ---")
    print(f"  Sharpe-optimal: VRP-Gewicht w={w_opt:.2f} (IS-Sharpe {is_sh[w_opt]:.2f})")
    print(f"  Inverse-Vol:    VRP-Gewicht w={w_iv:.2f}")

    for tag, sub in [("IN-SAMPLE 2006-2017", is_), ("OUT-OF-SAMPLE 2018-2026", oos)]:
        print(f"\n--- {tag} ---")
        show("SPY (Benchmark)", sub.spy)
        s_h = show("Hybrid allein", sub.hybrid)
        show("VRP-gemanagt allein", sub.vrp)
        show("50/50 Hybrid+VRP", 0.5 * sub.hybrid + 0.5 * sub.vrp)
        s_opt = show(f"Combo w_VRP={w_opt:.2f} (IS-opt)", (1 - w_opt) * sub.hybrid + w_opt * sub.vrp)
        show(f"Combo w_VRP={w_iv:.2f} (inv-vol)", (1 - w_iv) * sub.hybrid + w_iv * sub.vrp)
        if tag.startswith("OUT"):
            oos_h, oos_opt = s_h, s_opt

    print("\n--- OOS-ROBUSTHEIT ueber VRP-Gewicht (deskriptiv, NICHT zur Wahl) ---")
    print(f"  {'w_VRP':>6}{'ann%':>9}{'Sharpe':>9}{'MaxDD%':>9}")
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        s = stats((1 - w) * oos.hybrid + w * oos.vrp)
        print(f"  {w:>6.2f}{s['ann']:>8.2f}%{s['sharpe']:>9.2f}{s['maxdd']:>8.1f}%")

    print("\n" + "=" * 92)
    print("VERDICT (Entscheidungsregel: OOS Combo@IS-Gewicht > Hybrid allein, Tail ok?)")
    print("=" * 92)
    better_sharpe = oos_opt["sharpe"] > oos_h["sharpe"]
    tail_ok = oos_opt["maxdd"] >= oos_h["maxdd"] - 3  # max 3pp schlechter
    print(f"  OOS Hybrid allein : Sharpe {oos_h['sharpe']:+.2f}  MaxDD {oos_h['maxdd']:+.1f}%")
    print(f"  OOS Combo (w={w_opt:.2f}): Sharpe {oos_opt['sharpe']:+.2f}  MaxDD {oos_opt['maxdd']:+.1f}%")
    delta = oos_opt["sharpe"] - oos_h["sharpe"]
    if better_sharpe and tail_ok:
        print(f"  -> GREEN: VRP-Sleeve hebt OOS-Sharpe um {delta:+.2f} ohne Tail-Verschlechterung.")
    elif better_sharpe and not tail_ok:
        print(f"  -> YELLOW: Sharpe +{delta:.2f}, aber Tail schlechter -> nur klein/gehedged.")
    else:
        print(f"  -> NEGATIV: VRP hebt OOS-Sharpe NICHT ({delta:+.2f}). "
              f"OOS-Decay frisst den Diversifikationsvorteil.")


if __name__ == "__main__":
    main()
