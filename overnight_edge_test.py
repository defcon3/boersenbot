"""
OVERNIGHT-EDGE-TEST — Pre-Reg preregs/overnight_intraday_2026_06_23.md  (2026-06-23)

Deskriptiv: faellt die Aktienpraemie ueber Nacht (Close->Open) an, Intraday flach?
Handelbar: schlaegt Overnight-only (MOC-Kauf/MOO-Verkauf) netto Buy&Hold?

Daten: yfinance Tages-OHLC auto_adjust=True (Dividenden -> Overnight), 2010-2026.
Primaer SPY, Robustheit QQQ/IWM. Cross-Check Open/Close-Qualitaet vs SIP-30-Min.
"""
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import warnings; warnings.filterwarnings("ignore")

IS_END = pd.Timestamp("2021-12-31")
OOS_START = pd.Timestamp("2022-01-01")
COVID = (pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-30"))
ANN = np.sqrt(252)
COST_RT = 2.0  # bps Round-Trip/Tag (Overnight-only: 1 MOC-Kauf + 1 MOO-Verkauf)


def hac_mean_t(x, L=10):
    """t-Stat des Mittelwerts mit Newey-West (HAC)."""
    x = np.asarray(x, float)
    n = len(x)
    mu = x.mean()
    e = x - mu
    s = (e @ e) / n
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        s += 2 * w * (e[l:] @ e[:-l]) / n
    se = np.sqrt(s / n)
    return mu, mu / se if se > 0 else np.nan


def decompose(sym):
    df = yf.download(sym, start="2010-01-01", end="2026-06-23",
                     auto_adjust=True, progress=False)
    o = np.asarray(df["Open"].values, float).flatten()
    c = np.asarray(df["Close"].values, float).flatten()
    idx = df.index
    on = o[1:] / c[:-1] - 1          # Overnight
    idd = c[1:] / o[1:] - 1          # Intraday
    full = c[1:] / c[:-1] - 1        # Buy&Hold
    return pd.DataFrame({"ON": on, "ID": idd, "FULL": full}, index=idx[1:])


def stats(r):
    mu, t = hac_mean_t(r)
    return dict(ann=mu * 252 * 100, mean_bps=mu * 1e4, t=t,
                sharpe=mu / (r.std() + 1e-12) * ANN, n=len(r))


def block(name, d):
    print(f"\n  {name}")
    for lbl, col in [("Overnight (ON)", "ON"), ("Intraday  (ID)", "ID"),
                     ("Buy&Hold  (FU)", "FULL")]:
        s = stats(d[col].values)
        print(f"    {lbl}: ann={s['ann']:+6.2f}%  mean={s['mean_bps']:+6.2f}bps  "
              f"t={s['t']:+5.2f}  Sharpe={s['sharpe']:+5.2f}  n={s['n']}")


def run_symbol(sym, sip_check=None):
    print("=" * 84)
    print(f"{sym}")
    print("=" * 84)
    d = decompose(sym)
    is_ = d[d.index <= IS_END]
    oos = d[d.index >= OOS_START]
    cum_on = (1 + d["ON"]).prod()
    cum_id = (1 + d["ID"]).prod()
    print(f"  Kumuliert 2010-2026: Overnight x{cum_on:.2f}  |  Intraday x{cum_id:.2f}  "
          f"|  Buy&Hold x{(1+d['FULL']).prod():.2f}")
    block("IS 2010-2021:", is_)
    block("OOS 2022-2026:", oos)

    # --- handelbar: Overnight-only netto vs Buy&Hold (OOS) ---
    on_oos = oos["ON"].values
    net = on_oos - COST_RT / 1e4
    net1 = on_oos - 1.0 / 1e4
    bh = stats(oos["FULL"].values)
    print(f"\n  HANDELBAR (OOS): Overnight-only netto vs Buy&Hold")
    print(f"    Overnight brutto : ann={stats(on_oos)['ann']:+6.2f}%  Sharpe={stats(on_oos)['sharpe']:+5.2f}")
    sn1 = dict(ann=net1.mean()*252*100, sharpe=net1.mean()/(net1.std()+1e-12)*ANN)
    sn = dict(ann=net.mean()*252*100, sharpe=net.mean()/(net.std()+1e-12)*ANN)
    print(f"    netto 1bp RT     : ann={sn1['ann']:+6.2f}%  Sharpe={sn1['sharpe']:+5.2f}")
    print(f"    netto 2bp RT     : ann={sn['ann']:+6.2f}%  Sharpe={sn['sharpe']:+5.2f}")
    print(f"    Buy&Hold         : ann={bh['ann']:+6.2f}%  Sharpe={bh['sharpe']:+5.2f}")

    if sip_check is not None:
        # Cross-Check: Intraday-Rendite yfinance vs SIP (Open/Close-Qualitaet)
        m = pd.concat([d["ID"], sip_check["ID_sip"]], axis=1, join="inner").dropna()
        if len(m) > 50:
            corr = np.corrcoef(m.iloc[:, 0], m.iloc[:, 1])[0, 1]
            print(f"\n  [Cross-Check vs SIP] Intraday-Korr yfinance~SIP = {corr:.3f} "
                  f"(n={len(m)}) -> Open/Close-Qualitaet {'OK' if corr>0.97 else 'PRUEFEN'}")
    return d, is_, oos


def sip_intraday():
    with open("spy_30min_sip_2016_2026.pkl", "rb") as f:
        bars = pickle.load(f)
    df = pd.DataFrame(bars)
    df["et"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
    df["date"] = df["et"].dt.normalize().dt.tz_localize(None)
    df["tod"] = df["et"].dt.strftime("%H:%M")
    rows = []
    for d, g in df.groupby("date"):
        gi = g.set_index("tod")
        if not {"09:30", "15:30"}.issubset(gi.index):
            continue
        rows.append((d, float(gi.loc["15:30", "c"]) / float(gi.loc["09:30", "o"]) - 1))
    s = pd.DataFrame(rows, columns=["date", "ID_sip"]).set_index("date")
    return s


def gates(is_, oos):
    on_is = stats(is_["ON"].values)
    on_oos = stats(oos["ON"].values)
    id_is_ann = stats(is_["ID"].values)["ann"]
    id_oos_ann = stats(oos["ID"].values)["ann"]
    net = oos["ON"].values - COST_RT / 1e4
    net_ann = net.mean() * 252 * 100
    net_sharpe = net.mean() / (net.std() + 1e-12) * ANN
    bh_sharpe = stats(oos["FULL"].values)["sharpe"]

    g1 = (on_is["t"] > 2.0) and (on_is["mean_bps"] > 0) and (on_is["ann"] > id_is_ann)
    g2 = (on_oos["mean_bps"] > 0) and (on_oos["t"] > 1.5) and (on_oos["ann"] > id_oos_ann)
    g3 = (net_ann > 0) and (net_sharpe > bh_sharpe)
    print("\n" + "=" * 84)
    print("GATE-CHECK (SPY)")
    print("=" * 84)
    print(f"  G1 IS  (ON t>2 & >0 & ON>ID): {'PASS' if g1 else 'FAIL'}  "
          f"(t={on_is['t']:.2f}, ON {on_is['ann']:+.1f}% vs ID {id_is_ann:+.1f}%)")
    print(f"  G2 OOS (ON >0 & t>1.5 & ON>ID): {'PASS' if g2 else 'FAIL'}  "
          f"(t={on_oos['t']:.2f}, ON {on_oos['ann']:+.1f}% vs ID {id_oos_ann:+.1f}%)")
    print(f"  G3 HANDELBAR (netto 2bp: ann>0 & Sharpe>B&H): {'PASS' if g3 else 'FAIL'}  "
          f"(netto ann={net_ann:+.1f}%, Sharpe {net_sharpe:+.2f} vs B&H {bh_sharpe:+.2f})")
    print("-" * 84)
    if g1 and g2 and g3:
        print("VERDICT: GREEN — deskriptiv UND handelbar. Forward-Test.")
    elif g1 and g2 and not g3:
        print("VERDICT: YELLOW — Overnight-Praemie REAL, aber durch taegliche")
        print("         Round-Trip-Kosten NICHT erntbar (Overnight-only < Buy&Hold netto).")
    else:
        print("VERDICT: RED — deskriptive Hypothese nicht bestaetigt.")


def main():
    print("OVERNIGHT-EDGE-TEST — Pre-Reg 2026-06-23\n")
    try:
        sipc = sip_intraday()
    except Exception as e:
        print(f"(SIP-Cross-Check uebersprungen: {e})")
        sipc = None
    d, is_, oos = run_symbol("SPY", sip_check=sipc)
    gates(is_, oos)

    print("\n" + "=" * 84)
    print("ROBUSTHEIT (QQQ, IWM)")
    print("=" * 84)
    for sym in ["QQQ", "IWM"]:
        run_symbol(sym)

    # G5: per-Jahr ON vs ID (SPY)
    print("\n" + "=" * 84)
    print("G5: PER-JAHR ON vs ID (SPY, ann.%) + COVID-Kontrolle")
    print("=" * 84)
    by_year = d.groupby(d.index.year)
    won = 0
    for yr, g in by_year:
        a_on, a_id = stats(g["ON"].values)["ann"], stats(g["ID"].values)["ann"]
        won += a_on > a_id
        print(f"  {yr}: ON {a_on:+7.1f}%  ID {a_id:+7.1f}%  {'ON>ID' if a_on>a_id else 'ID>ON'}")
    print(f"  -> ON>ID in {won}/{by_year.ngroups} Jahren")
    d_nc = d[~((d.index >= COVID[0]) & (d.index <= COVID[1]))]
    s_on = stats(d_nc["ON"].values); s_id = stats(d_nc["ID"].values)
    print(f"  ohne COVID: ON ann={s_on['ann']:+.1f}% (t={s_on['t']:.2f})  "
          f"ID ann={s_id['ann']:+.1f}%")


if __name__ == "__main__":
    main()
