#!/usr/bin/env python3
"""
HYBRID-FILTER-AUDIT  (Backlog-Idee 1, 2026-06-01)

Frage: Verdient der live laufende Hybrid-Filter
       (MA50/200-Trendfilter + VIX-Sizing) seine Komplexitaet,
       oder kostet er gegenueber Buy-and-Hold SPY nur Rendite?

Signal exakt wie nas_hybrid_calculator.py / hybrid_signal_calculator.py:
    uptrend       = 1 if MA50 > MA200 else 0            (SMA auf Raw-Close)
    vix_norm      = clip( (vix - vix_mean60)/(vix_std60+1e-6) * 0.1, -0.5, 0.5)
    size          = clip(1 - vix_norm, 0.2, 1.0)
    position_size = uptrend * size                      (0  oder  0.2..1.0)

Position wird am Close von Tag t entschieden und auf die Rendite von
Tag t+1 angewendet (kein Lookahead). Nicht-investiertes Kapital = 0 % (Cash).

Benchmark: Buy-and-Hold SPY (position = 1.0).
Renditen   : SPY Total-Return (Adj Close).
MAs/VIX-z  : exakt die im Live-Code verwendeten Roh-Serien.

Gate (aus Backlog): Sharpe UND MaxDD vs B&H + OOS-Split.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

START = "2007-01-01"           # ^VIX/SPY weit genug zurueck; MA200 frisst ~1 Jahr
ANN = 252


def fetch():
    spy = yf.download("SPY", start=START, auto_adjust=False, progress=False)
    vix = yf.download("^VIX", start=START, auto_adjust=False, progress=False)
    # MultiIndex-Spalten von neueren yfinance-Versionen flachklopfen
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    df = pd.DataFrame({
        "spy_close": spy["Close"],        # Raw-Close fuer MA (wie Live)
        "spy_adj":   spy["Adj Close"],    # Total-Return fuer P&L
        "vix":       vix["Close"],
    }).dropna()
    return df


def build_signal(df):
    c = df["spy_close"]
    ma50 = c.rolling(50).mean()
    ma200 = c.rolling(200).mean()
    uptrend = (ma50 > ma200).astype(float)

    v = df["vix"]
    vmean = v.rolling(60).mean()
    vstd = v.rolling(60).std(ddof=1)          # statistics.stdev = Stichprobe (ddof=1)
    vnorm_raw = (v - vmean) / (vstd + 1e-6) * 0.1
    vnorm = vnorm_raw.clip(-0.5, 0.5)
    size = (1.0 - vnorm).clip(0.2, 1.0)

    pos = uptrend * size
    df = df.copy()
    df["ma50"], df["ma200"] = ma50, ma200
    df["uptrend"] = uptrend
    df["position"] = pos
    df["spy_ret"] = df["spy_adj"].pct_change()
    # Position von t auf Rendite t+1 -> position shiften
    df["strat_ret"] = df["position"].shift(1) * df["spy_ret"]
    df["bh_ret"] = df["spy_ret"]               # Buy & Hold = immer 1.0
    return df.dropna(subset=["ma200", "strat_ret"])


def metrics(rets, name):
    rets = rets.dropna()
    n = len(rets)
    eq = (1 + rets).cumprod()
    total = eq.iloc[-1] - 1
    years = n / ANN
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = rets.std(ddof=1) * np.sqrt(ANN)
    sharpe = (rets.mean() * ANN) / vol if vol > 0 else np.nan
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    # Sortino
    downside = rets[rets < 0].std(ddof=1) * np.sqrt(ANN)
    sortino = (rets.mean() * ANN) / downside if downside > 0 else np.nan
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    return {
        "name": name, "n": n, "years": years,
        "total": total, "cagr": cagr, "vol": vol,
        "sharpe": sharpe, "sortino": sortino,
        "maxdd": maxdd, "calmar": calmar,
    }


def print_block(title, df):
    s = metrics(df["strat_ret"], "Hybrid-Filter")
    b = metrics(df["bh_ret"], "Buy&Hold SPY")
    tim = (df["position"].shift(1) > 0).mean()
    avg_pos = df["position"].shift(1).mean()
    print(f"\n{'='*72}\n{title}")
    print(f"  Zeitraum: {df.index[0].date()} .. {df.index[-1].date()}  "
          f"({s['years']:.1f} J, {s['n']} Tage)")
    print(f"  Zeit-im-Markt Hybrid: {tim*100:5.1f} %   "
          f"Ø-Position: {avg_pos:.3f}")
    hdr = f"  {'Metrik':<14}{'Hybrid':>14}{'Buy&Hold':>14}{'Diff':>14}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    def row(lbl, key, pct=True, hib=True):
        hv, bv = s[key], b[key]
        if pct:
            hs, bs = f"{hv*100:.2f}%", f"{bv*100:.2f}%"
            d = (hv - bv) * 100
            ds = f"{d:+.2f}pp"
        else:
            hs, bs = f"{hv:.3f}", f"{bv:.3f}"
            d = hv - bv
            ds = f"{d:+.3f}"
        flag = ""
        better = (hv > bv) if hib else (hv < bv)
        flag = "  <-- Hybrid" if better else ""
        print(f"  {lbl:<14}{hs:>14}{bs:>14}{ds:>14}{flag}")
    row("Total-Return", "total", pct=True)
    row("CAGR", "cagr", pct=True)
    row("Vol p.a.", "vol", pct=True, hib=False)
    row("Sharpe", "sharpe", pct=False, hib=True)
    row("Sortino", "sortino", pct=False, hib=True)
    row("MaxDrawdown", "maxdd", pct=True, hib=True)   # weniger negativ = besser
    row("Calmar", "calmar", pct=False, hib=True)
    return s, b


def main():
    print("Lade SPY + ^VIX ...", flush=True)
    df = fetch()
    df = build_signal(df)

    # Voller Zeitraum
    print_block("GESAMT (In- + Out-of-Sample)", df)

    # OOS-Split: Haelfte/Haelfte am Median-Datum
    split = df.index[len(df) // 2]
    is_df = df.loc[:split]
    oos_df = df.loc[split:]
    print_block(f"IN-SAMPLE  (bis {split.date()})", is_df)
    print_block(f"OUT-OF-SAMPLE (ab {split.date()})", oos_df)

    # Zusaetzlicher fester Cut 2018-01-01 (klar getrennte Markt-Regime)
    cut = "2018-01-01"
    print_block(f"OOS-CHECK ab {cut}", df.loc[cut:])

    # Jahres-Breakdown: wo verdient der Filter, wo kostet er?
    print(f"\n{'='*72}\nKALENDERJAHR-BREAKDOWN (Total-Return)")
    print(f"  {'Jahr':<6}{'Hybrid':>10}{'Buy&Hold':>10}{'Diff':>10}"
          f"{'Ø-Pos':>8}  Kommentar")
    print("  " + "-" * 60)
    for yr, g in df.groupby(df.index.year):
        h = (1 + g["strat_ret"]).prod() - 1
        bh = (1 + g["bh_ret"]).prod() - 1
        avgp = g["position"].shift(1).mean()
        diff = (h - bh) * 100
        tag = "Filter rettet" if diff > 3 else ("Filter kostet" if diff < -3 else "")
        print(f"  {yr:<6}{h*100:>9.1f}%{bh*100:>9.1f}%{diff:>+9.1f}pp"
              f"{avgp:>8.2f}  {tag}")

    print("\n" + "=" * 72)
    print("Hinweis: Cash-Rendite = 0 % (konservativ; echte Geldmarkt-Verzinsung "
          "wuerde Hybrid leicht beguenstigen).")


if __name__ == "__main__":
    main()
