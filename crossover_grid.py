#!/usr/bin/env python3
"""MA-Crossover-Grid: (fast, slow) x horizon -> Aggregate je Zelle.

Pro (fast, slow, horizon)-Kombi:
  - Golden Cross Events + Forward-Return-Verteilung
  - Death  Cross Events + Forward-Return-Verteilung
  - Hit-Quote (erwartete Richtung) vs unbedingte Basis bei diesem Horizont

JSON-Output dient der Web-Seite /analysis/crossover (Plotly-Histogramm
+ Heatmap fast x slow bei gewaehltem Horizont).
"""
import warnings; warnings.filterwarnings("ignore")
import io, json, pickle, time
from pathlib import Path
import numpy as np, pandas as pd, requests, yfinance as yf

STK_START = "2014-01-01"
CACHE_FILE = Path("/home/veit/boersenbot/crossover_universe.pkl")
OUT_GRID   = Path("/home/veit/boersenbot/crossover_grid.json")

FASTS      = [5, 10, 20, 50, 100]
SLOWS      = [20, 50, 100, 150, 200, 250]
HORIZONS   = [1, 3, 5, 10, 20, 60]
HIST_BINS  = 60
HIST_RANGE = (-25.0, 25.0)            # % Renditen, fuer Histogramm-Binning


def load_universe():
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    html = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=hdr, timeout=30).text
    sp = pd.read_html(io.StringIO(html))[0]
    sp["Symbol"] = sp["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return sp["Symbol"].tolist()


def load_ohlc(tickers):
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        print(f"  Cache: {len(cache)} Ticker", flush=True)
        return cache
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
                if len(d) < 280:
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
    return cache


def baseline_returns_pct(ohlc, h):
    rets = []
    for t, df in ohlc.items():
        c = df["close"].values
        if len(c) < h + 1:
            continue
        r = c[h:] / c[:-h] - 1.0
        rets.extend(r[np.isfinite(r)].tolist())
    return np.asarray(rets) * 100.0


def find_events_for_pair(ohlc, fast, slow, h):
    """Sammelt alle Crossover-Events + Forward-Return (%) fuer (fast, slow, h).
    Rueckgabe: (golden_rets_pct, death_rets_pct) als np-Arrays."""
    g, d = [], []
    for t, df in ohlc.items():
        c = df["close"].values
        if len(c) < slow + h + 1:
            continue
        ma_f = pd.Series(c).rolling(fast).mean().values
        ma_s = pd.Series(c).rolling(slow).mean().values
        diff = ma_f - ma_s
        # vektor: Vorzeichenwechsel an Position t
        prev = diff[:-1]
        cur  = diff[1:]
        gold_mask  = np.isfinite(prev) & np.isfinite(cur) & \
                     (prev <= 0) & (cur > 0)
        death_mask = np.isfinite(prev) & np.isfinite(cur) & \
                     (prev >= 0) & (cur < 0)
        # Index in c: ts (0-basiert) entspricht prev/cur[ts-1]
        # also Crossover an t = ts+1
        gold_idx  = np.flatnonzero(gold_mask)  + 1
        death_idx = np.flatnonzero(death_mask) + 1
        # Forward-Return brauchen c[t+h]
        for idx_arr, sink in ((gold_idx, g), (death_idx, d)):
            valid = idx_arr[(idx_arr + h) < len(c)]
            if valid.size == 0:
                continue
            r = c[valid + h] / c[valid] - 1.0
            sink.extend((r[np.isfinite(r)] * 100.0).tolist())
    return np.asarray(g), np.asarray(d)


def stats_block(arr_pct, hit_rule):
    if len(arr_pct) == 0:
        return None
    a = arr_pct
    return {
        "n":      int(len(a)),
        "mean":   round(float(np.mean(a)), 4),
        "median": round(float(np.median(a)), 4),
        "q10":    round(float(np.quantile(a, 0.1)), 4),
        "q90":    round(float(np.quantile(a, 0.9)), 4),
        "hit_pct": round(float(np.mean(hit_rule(a)) * 100), 2),
    }


def hist_block(arr_pct):
    counts, edges = np.histogram(arr_pct, bins=HIST_BINS, range=HIST_RANGE)
    return {
        "counts": counts.tolist(),
        "edges":  [round(float(x), 3) for x in edges.tolist()],
    }


def main():
    print("="*78)
    print("MA-CROSSOVER GRID  (fast x slow) x horizon")
    print(f"  fast in {FASTS}")
    print(f"  slow in {SLOWS}")
    print(f"  horizon in {HORIZONS}  Handelstage")
    print("="*78, flush=True)

    print("\n[1/3] Universum + OHLC ...", flush=True)
    tickers = load_universe()
    ohlc = load_ohlc(tickers)
    print(f"      {len(ohlc)} Ticker mit ausreichend Historie", flush=True)

    print("\n[2/3] Unbedingte Basis je Horizont ...", flush=True)
    baseline = {}
    for h in HORIZONS:
        b = baseline_returns_pct(ohlc, h)
        baseline[h] = {
            "stats": {
                "n":      int(len(b)),
                "mean":   round(float(np.mean(b)), 4),
                "median": round(float(np.median(b)), 4),
                "q10":    round(float(np.quantile(b, 0.1)), 4),
                "q90":    round(float(np.quantile(b, 0.9)), 4),
                "pct_positive": round(float(np.mean(b > 0) * 100), 2),
            },
            "hist": hist_block(b),
        }
        print(f"  h={h:>2}TT  n={baseline[h]['stats']['n']:>8,}  "
              f"Med {baseline[h]['stats']['median']:+6.3f}%  "
              f"%>0 {baseline[h]['stats']['pct_positive']:.1f}%", flush=True)

    print("\n[3/3] Grid-Backtest ...", flush=True)
    t0 = time.time()
    pairs = [(f, s) for f in FASTS for s in SLOWS if s > f]
    total = len(pairs) * len(HORIZONS)
    grid = {}
    done = 0
    for (f, s) in pairs:
        # Einmal Events sammeln je (f, s) fuer den groessten Horizont gibt
        # nicht so viel — wir muessen sowieso je horizont neu c[t+h] holen.
        # Da der Crossover-Set fix ist, koennte man optimieren; bei 27 Paaren
        # x 6 Horizonten ist es aber schnell genug ohne.
        for h in HORIZONS:
            g, d = find_events_for_pair(ohlc, f, s, h)
            key = f"{f}_{s}_{h}"
            grid[key] = {
                "fast": f, "slow": s, "horizon": h,
                "golden": {
                    **(stats_block(g, lambda a: a > 0) or {"n": 0}),
                    "hist": hist_block(g) if len(g) else None,
                },
                "death":  {
                    **(stats_block(d, lambda a: a < 0) or {"n": 0}),
                    "hist": hist_block(d) if len(d) else None,
                },
            }
            done += 1
            if done % 10 == 0 or done == total:
                el = time.time() - t0
                print(f"  {done:>3}/{total}  ({el:.0f}s)  "
                      f"f{f}/s{s}/h{h}  "
                      f"G n={grid[key]['golden']['n']}  "
                      f"D n={grid[key]['death']['n']}", flush=True)

    out = {
        "params": {
            "fasts":     FASTS,
            "slows":     SLOWS,
            "horizons":  HORIZONS,
            "stk_start": STK_START,
            "hist_bins": HIST_BINS,
            "hist_range_pct": HIST_RANGE,
        },
        "baseline": baseline,
        "grid":     grid,
    }
    with open(OUT_GRID, "w") as f:
        json.dump(out, f)
    print(f"\nGeschrieben: {OUT_GRID}  ({OUT_GRID.stat().st_size/1e6:.1f} MB)")
    print("\n--- Quick-Look: bestes und schlechtestes (fast, slow, h) "
          "nach Mean-Diff Golden vs Basis ---")
    rows = []
    for key, cell in grid.items():
        h = cell["horizon"]
        base_mean = baseline[h]["stats"]["mean"]
        if cell["golden"]["n"] >= 100:
            rows.append((
                key,
                cell["fast"], cell["slow"], h,
                cell["golden"]["n"], cell["golden"]["mean"],
                cell["golden"]["mean"] - base_mean,
                cell["golden"]["hit_pct"],
                base_mean,
            ))
    rows.sort(key=lambda r: r[6], reverse=True)
    print(f"{'cell':<14}{'f':>4}{'s':>5}{'h':>4}"
          f"{'n':>6}{'meanG':>8}{'-base':>8}{'hit%':>7}{'base':>7}")
    for r in rows[:5]:
        print(f"{r[0]:<14}{r[1]:>4}{r[2]:>5}{r[3]:>4}"
              f"{r[4]:>6}{r[5]:>+8.3f}{r[6]:>+8.3f}"
              f"{r[7]:>7.1f}{r[8]:>+7.3f}")
    print("...")
    for r in rows[-5:]:
        print(f"{r[0]:<14}{r[1]:>4}{r[2]:>5}{r[3]:>4}"
              f"{r[4]:>6}{r[5]:>+8.3f}{r[6]:>+8.3f}"
              f"{r[7]:>7.1f}{r[8]:>+7.3f}")


if __name__ == "__main__":
    main()
