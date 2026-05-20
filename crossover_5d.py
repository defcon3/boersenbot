#!/usr/bin/env python3
"""MA50 x MA200 Crossover -> 5-Handelstage Forward-Return je Event.

Aggregat-Statistik + JSON-Output fuer Web-Seite /analysis/crossover.

Datenbasis:
  S&P-500 (Wikipedia-Liste), Tages-OHLC ab 2014-01-01 via yfinance
  (geben MA200-Vorlauf + Crossover-Events ab ~Mitte 2014).

Definitionen:
  MA50/MA200 = SMA(Close, 50/200)
  Golden Cross: MA50[t-1] <= MA200[t-1] AND MA50[t] > MA200[t]
  Death  Cross: MA50[t-1] >= MA200[t-1] AND MA50[t] < MA200[t]
  5d-Return:  Close[t+5_TT] / Close[t] - 1   (5 Handelstage)
  "wie erwartet": Golden r5>0, Death r5<0

Unbedingte Basis (Vergleich): alle (Aktie, Tag) mit gueltigem Close[t+5]
ergibt die Drift-Verteilung gegen die wir messen.
"""
import warnings; warnings.filterwarnings("ignore")
import io, json, pickle, os, time
from pathlib import Path
import numpy as np, pandas as pd, requests, yfinance as yf

STK_START = "2014-01-01"
HOLD_DAYS = 5                                 # Handelstage
CACHE_FILE = Path("/home/veit/boersenbot/crossover_universe.pkl")
OUT_JSON   = Path("/home/veit/boersenbot/crossover_events.json")
OUT_AGG    = Path("/home/veit/boersenbot/crossover_aggregate.json")


def load_universe():
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    html = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=hdr, timeout=30).text
    sp = pd.read_html(io.StringIO(html))[0]
    sp["Symbol"] = sp["Symbol"].astype(str).str.replace(".", "-", regex=False)
    sector_of = dict(zip(sp["Symbol"], sp["GICS Sector"]))
    return sp["Symbol"].tolist(), sector_of


def load_ohlc(tickers):
    """Tages-OHLC Cache (~80 MB). Erster Lauf 5-10 Min, danach Sekunden."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            print(f"  Cache: {len(cache)} Ticker aus {CACHE_FILE.name}",
                  flush=True)
            return cache
        except Exception as ex:
            print(f"  Cache defekt ({ex}), neu laden", flush=True)
    print(f"  Lade {len(tickers)} Ticker ab {STK_START} ...", flush=True)
    cache = {}
    BATCH = 60
    for i in range(0, len(tickers), BATCH):
        ch = tickers[i:i+BATCH]
        try:
            data = yf.download(ch, start=STK_START, interval="1d",
                               progress=False, group_by="ticker",
                               threads=True, auto_adjust=True)
        except Exception as ex:
            print(f"  Batch {i}: FEHLER {ex}", flush=True); continue
        for t in ch:
            try:
                d = data[t][["Close"]].dropna()
                if len(d) < 220:               # min MA200+20 fuer >=1 Event
                    continue
                c = d["Close"].astype(float).values.ravel()
                dts = pd.to_datetime(d.index).normalize()
                cache[t] = pd.DataFrame({"date": dts, "close": c})
            except Exception:
                pass
        print(f"    {min(i+BATCH, len(tickers))}/{len(tickers)} "
              f"({len(cache)} ok)", flush=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Cache geschrieben: {CACHE_FILE}", flush=True)
    return cache


def find_events(df_one, ticker, sector):
    """Pro Ticker: Liste Crossover-Events mit 5d-Forward-Return."""
    c = df_one["close"].values
    dts = df_one["date"].values
    ma50  = pd.Series(c).rolling(50).mean().values
    ma200 = pd.Series(c).rolling(200).mean().values
    diff  = ma50 - ma200
    # An t passiert Crossover, wenn Vorzeichen von diff[t-1]->diff[t] wechselt
    events = []
    for t in range(200, len(c) - HOLD_DAYS):
        if not (np.isfinite(diff[t-1]) and np.isfinite(diff[t])):
            continue
        kind = None
        if diff[t-1] <= 0 and diff[t] > 0:
            kind = "golden"
        elif diff[t-1] >= 0 and diff[t] < 0:
            kind = "death"
        if kind is None:
            continue
        c0, c5 = c[t], c[t + HOLD_DAYS]
        if not (c0 > 0 and np.isfinite(c5)):
            continue
        r5 = c5 / c0 - 1.0
        expected = (kind == "golden" and r5 > 0) or \
                   (kind == "death"  and r5 < 0)
        events.append({
            "ticker": ticker,
            "sector": sector,
            "date": pd.Timestamp(dts[t]).strftime("%Y-%m-%d"),
            "kind": kind,
            "close_t": round(float(c0), 4),
            "close_t5": round(float(c5), 4),
            "r5_pct": round(float(r5) * 100, 4),
            "ma50_t": round(float(ma50[t]), 4),
            "ma200_t": round(float(ma200[t]), 4),
            "expected": bool(expected),
        })
    return events


def unconditional_baseline(ohlc):
    """Verteilung aller (Aktie, Tag) 5-Handelstage-Forward-Returns 2015-2026."""
    rets = []
    for t, df in ohlc.items():
        c = df["close"].values
        if len(c) < HOLD_DAYS + 1:
            continue
        r = c[HOLD_DAYS:] / c[:-HOLD_DAYS] - 1.0
        rets.extend(r[np.isfinite(r)].tolist())
    return np.asarray(rets) * 100  # in Prozent


def aggregate(events, baseline):
    g = [e["r5_pct"] for e in events if e["kind"] == "golden"]
    d = [e["r5_pct"] for e in events if e["kind"] == "death"]
    def s(arr, hit_rule):
        a = np.asarray(arr)
        return {
            "n": int(len(a)),
            "mean":   float(np.mean(a))   if len(a) else None,
            "median": float(np.median(a)) if len(a) else None,
            "q10":    float(np.quantile(a, 0.1)) if len(a) else None,
            "q90":    float(np.quantile(a, 0.9)) if len(a) else None,
            "hit_pct": float(np.mean(hit_rule(a))*100) if len(a) else None,
        }
    return {
        "golden": s(g, lambda a: a > 0),
        "death":  s(d, lambda a: a < 0),
        "baseline": {
            "n": int(len(baseline)),
            "mean":   float(np.mean(baseline)),
            "median": float(np.median(baseline)),
            "q10":    float(np.quantile(baseline, 0.1)),
            "q90":    float(np.quantile(baseline, 0.9)),
            "pct_positive": float(np.mean(baseline > 0) * 100),
        },
    }


def main():
    print("="*78)
    print(f"MA50 x MA200 CROSSOVER -> {HOLD_DAYS}-Handelstage Forward-Return")
    print(f"  S&P-500, ab {STK_START}, yfinance Tages-OHLC")
    print("="*78, flush=True)

    print("\n[1/4] Universum + Sektoren laden...", flush=True)
    tickers, sector_of = load_universe()
    print(f"      {len(tickers)} Ticker, "
          f"{len(set(sector_of.values()))} GICS-Sektoren", flush=True)

    print("\n[2/4] OHLC + Cache ...", flush=True)
    ohlc = load_ohlc(tickers)
    print(f"      {len(ohlc)} Ticker mit ausreichend Historie", flush=True)

    print("\n[3/4] Crossover-Events extrahieren ...", flush=True)
    events = []
    for t, df in ohlc.items():
        sec = sector_of.get(t, "Unknown")
        events.extend(find_events(df, t, sec))
    g_n = sum(1 for e in events if e["kind"] == "golden")
    d_n = sum(1 for e in events if e["kind"] == "death")
    print(f"      {len(events)} Events total  "
          f"(Golden {g_n}, Death {d_n})", flush=True)

    print("\n[4/4] Unbedingte Basis + Aggregate ...", flush=True)
    baseline = unconditional_baseline(ohlc)
    print(f"      Basis: {len(baseline):,} (Aktie,Tag)-Paare, "
          f"Median {np.median(baseline):+.3f}%, "
          f"%>0 {(baseline>0).mean()*100:.1f}%", flush=True)
    agg = aggregate(events, baseline)

    # --- Konsolen-Ausgabe ----------------------------------------------------
    print("\n" + "="*78)
    print("AGGREGAT")
    print("="*78)
    b = agg["baseline"]
    print(f"\nUnbedingte Basis (jeder Tag, jede Aktie, +{HOLD_DAYS} Handelstage):")
    print(f"  n={b['n']:>8,}  Ø {b['mean']:+6.3f}%  "
          f"Median {b['median']:+6.3f}%  "
          f"Q10 {b['q10']:+6.2f}%  Q90 {b['q90']:+6.2f}%  "
          f"%>0 {b['pct_positive']:.1f}%")

    for k in ("golden", "death"):
        s = agg[k]
        exp = "+ (erwartet steigend)" if k == "golden" else "- (erwartet fallend)"
        print(f"\n{k.upper()} CROSS   {exp}")
        if s["n"] == 0:
            print("  -- keine Events --")
            continue
        print(f"  n={s['n']:>6}   Ø {s['mean']:+6.3f}%   "
              f"Median {s['median']:+6.3f}%   "
              f"Q10 {s['q10']:+6.2f}%   Q90 {s['q90']:+6.2f}%")
        diff_mean = s["mean"] - b["mean"]
        diff_hit  = s["hit_pct"] - (b["pct_positive"] if k == "golden"
                                    else 100 - b["pct_positive"])
        print(f"  Hit-Quote (erwartete Richtung):  {s['hit_pct']:5.1f}%  "
              f"(Basis: {b['pct_positive'] if k=='golden' else 100-b['pct_positive']:.1f}%, "
              f"Diff {diff_hit:+.1f} pp)")
        print(f"  Mean-Diff vs Basis:              {diff_mean:+.3f}%")

    # --- JSON-Output --------------------------------------------------------
    with open(OUT_JSON, "w") as f:
        json.dump(events, f)
    with open(OUT_AGG, "w") as f:
        json.dump({
            "params": {"hold_days": HOLD_DAYS, "stk_start": STK_START},
            "aggregate": agg,
            "baseline_returns_pct": baseline.round(4).tolist(),
        }, f)
    print(f"\nEvents:    {OUT_JSON}  ({len(events)} rows)")
    print(f"Aggregat:  {OUT_AGG}")


if __name__ == "__main__":
    main()
