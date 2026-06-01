#!/usr/bin/env python3
"""
VIX-TERMINSTRUKTUR-FILTER  (Backlog-Idee 3, 2026-06-01)

Frage: Liefert die VIX-Terminstruktur (VIX vs VIX3M, Contango/Backwardation)
       ein SCHAERFERES / SCHNELLERES "raus aus SPY"-Signal als der langsame
       MA50/200-Cross aus dem Hybrid-Filter (Idee 1, der sich als 2008-Artefakt
       mit OOS-Rendite-Drag entpuppte)?

Mechanismus-These: Normal ist Contango (VIX < VIX3M, ratio<1). Bei akutem
Stress invertiert die Kurve -> Backwardation (VIX > VIX3M, ratio>1). Das
passiert TAGE bevor ein MA50/200-Cross reagiert (z.B. COVID-Crash 02/2020).

Regeln (alle deterministisch 0/1, Cash=0 % wenn draussen, kein Lookahead:
Signal von Tag t -> Rendite t+1):
  B&H        : immer 1.0
  MA-Trend   : long wenn MA50>MA200, sonst Cash      (der langsame Amtsinhaber)
  VIX-TS     : long wenn VIX/VIX3M <= 1.0, sonst Cash (parameterfreie Schwelle)
  Kombi      : long wenn (MA50>MA200) UND (ratio<=1.0)

Disziplin (feedback_rigor_over_speed): Hauptschwelle = 1.0 ist theoriegeleitet
(invertierte Kurve), NICHT in-sample optimiert. 0.95/1.05 nur als Sensitivitaet.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

START = "2007-01-01"
ANN = 252


def fetch():
    spy = yf.download("SPY",   start=START, auto_adjust=False, progress=False)
    vix = yf.download("^VIX",  start=START, auto_adjust=False, progress=False)
    v3m = yf.download("^VIX3M", start=START, auto_adjust=False, progress=False)
    for d in (spy, vix, v3m):
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
    df = pd.DataFrame({
        "spy_close": spy["Close"],
        "spy_adj":   spy["Adj Close"],
        "vix":       vix["Close"],
        "vix3m":     v3m["Close"],
    }).dropna()
    return df


def build(df, ts_threshold=1.0):
    c = df["spy_close"]
    uptrend = (c.rolling(50).mean() > c.rolling(200).mean()).astype(float)
    ratio = df["vix"] / df["vix3m"]
    contango = (ratio <= ts_threshold).astype(float)   # 1 = ruhig (long), 0 = Backwardation

    df = df.copy()
    df["ratio"] = ratio
    df["uptrend"] = uptrend
    df["contango"] = contango
    df["spy_ret"] = df["spy_adj"].pct_change()

    df["pos_bh"]    = 1.0
    df["pos_trend"] = uptrend
    df["pos_vixts"] = contango
    df["pos_combo"] = uptrend * contango

    for k in ("bh", "trend", "vixts", "combo"):
        df[f"ret_{k}"] = df[f"pos_{k}"].shift(1) * df["spy_ret"]
    # MA200 frisst ~1 Jahr -> erst ab da gueltig
    return df.dropna(subset=["spy_ret"]).iloc[200:]


def metrics(rets):
    rets = rets.dropna()
    n = len(rets); eq = (1 + rets).cumprod()
    years = n / ANN
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = rets.std(ddof=1) * np.sqrt(ANN)
    sharpe = (rets.mean() * ANN) / vol if vol > 0 else np.nan
    maxdd = (eq / eq.cummax() - 1).min()
    dn = rets[rets < 0].std(ddof=1) * np.sqrt(ANN)
    sortino = (rets.mean() * ANN) / dn if dn > 0 else np.nan
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    return dict(n=n, years=years, cagr=cagr, vol=vol, sharpe=sharpe,
                sortino=sortino, maxdd=maxdd, calmar=calmar)


STRATS = [("bh", "Buy&Hold"), ("trend", "MA-Trend"),
          ("vixts", "VIX-TS"), ("combo", "Kombi")]


def block(title, df):
    print(f"\n{'='*84}\n{title}")
    print(f"  {df.index[0].date()} .. {df.index[-1].date()}  "
          f"({len(df)/ANN:.1f} J, {len(df)} Tage)")
    print(f"\n  {'Strategie':<12}{'CAGR':>9}{'Vol':>8}{'Sharpe':>8}"
          f"{'Sortino':>9}{'MaxDD':>9}{'Calmar':>8}{'ZiM':>7}")
    print("  " + "-" * 80)
    out = {}
    for key, lbl in STRATS:
        m = metrics(df[f"ret_{key}"])
        zim = (df[f"pos_{key}"].shift(1) > 0).mean()
        out[key] = m
        print(f"  {lbl:<12}{m['cagr']*100:>8.2f}%{m['vol']*100:>7.1f}%"
              f"{m['sharpe']:>8.2f}{m['sortino']:>9.2f}{m['maxdd']*100:>8.1f}%"
              f"{m['calmar']:>8.2f}{zim*100:>6.0f}%")
    return out


def main():
    print("Lade SPY + ^VIX + ^VIX3M ...", flush=True)
    df = build(fetch(), ts_threshold=1.0)

    block("GESAMT (In- + Out-of-Sample), Schwelle ratio<=1.0", df)
    split = df.index[len(df) // 2]
    block(f"IN-SAMPLE  (bis {split.date()})", df.loc[:split])
    block(f"OUT-OF-SAMPLE (ab {split.date()})", df.loc[split:])
    block("OOS-CHECK ab 2018-01-01", df.loc["2018-01-01":])

    # Jahres-Breakdown: schlaegt VIX-TS den MA-Trend in den Crash-Jahren?
    print(f"\n{'='*84}\nKALENDERJAHR (Total-Return) - schlaegt VIX-TS den traegen MA-Trend?")
    print(f"  {'Jahr':<6}{'B&H':>8}{'MA-Trend':>10}{'VIX-TS':>9}{'Kombi':>9}"
          f"{'TS-ZiM':>8}  Krisenjahr?")
    print("  " + "-" * 70)
    for yr, g in df.groupby(df.index.year):
        r = {k: (1 + g[f"ret_{k}"]).prod() - 1 for k, _ in STRATS}
        zim = (g["pos_vixts"].shift(1) > 0).mean()
        tag = "<-- Stress" if r["bh"] < 0 or yr in (2018, 2020, 2022) else ""
        print(f"  {yr:<6}{r['bh']*100:>7.1f}%{r['trend']*100:>9.1f}%"
              f"{r['vixts']*100:>8.1f}%{r['combo']*100:>8.1f}%{zim*100:>7.0f}%  {tag}")

    # COVID-Reaktionszeit: wann ging wer raus?
    print(f"\n{'='*84}\nCOVID-REAKTION (Feb-Mrz 2020): wer ist wann raus?")
    covid = df.loc["2020-02-14":"2020-03-31"]
    last_in = {"trend": None, "vixts": None}
    for k in last_in:
        out_days = covid.index[covid[f"pos_{k}"] == 0]
        last_in[k] = out_days[0].date() if len(out_days) else "blieb drin"
    print(f"  SPY-Hoch ~19.02.2020. Erster Risk-OFF-Tag:")
    print(f"    MA-Trend (MA50/200): {last_in['trend']}")
    print(f"    VIX-TS (Backwardation): {last_in['vixts']}")

    # Schwellen-Sensitivitaet (NUR Robustheit, nicht zum Cherry-Picken)
    print(f"\n{'='*84}\nSCHWELLEN-SENSITIVITAET VIX-TS (gesamt, Sharpe/MaxDD)")
    print(f"  {'Schwelle':<10}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'ZiM':>7}")
    print("  " + "-" * 44)
    raw = build(fetch(), ts_threshold=1.0)  # ratio ist schwellenunabhaengig, rebuild billig
    for thr in (0.95, 1.0, 1.05):
        contango = (raw["ratio"] <= thr).astype(float)
        ret = contango.shift(1) * raw["spy_ret"]
        m = metrics(ret); zim = (contango.shift(1) > 0).mean()
        mark = "  <- Hauptregel" if thr == 1.0 else ""
        print(f"  <={thr:<7}{m['cagr']*100:>8.2f}%{m['sharpe']:>9.2f}"
              f"{m['maxdd']*100:>8.1f}%{zim*100:>6.0f}%{mark}")

    print("\n" + "=" * 84)
    print("Lesart: VIX-TS gewinnt nur, wenn es B&H-Sharpe schlaegt UND in 2020/2018/2022")
    print("frueher/besser schuetzt als MA-Trend. Cash=0 % (konservativ).")


if __name__ == "__main__":
    main()
