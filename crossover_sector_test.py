#!/usr/bin/env python3
"""
MA-CROSSOVER SEKTOR-TEST (Neue Hypothese 2026-05-21).

Klassische 50/200-Crossover (f=20, s=50, h=10 aus frueherem Grid),
aber NACH GICS-SEKTOREN aufgebrochen.

PRE-REGISTRIERT:
  - Gleiche Parameter wie Broad-Market-Test (kein neues Grid)
  - IS 2014-2021, OOS 2022-2026, Split 2022-01-01
  - Gates G1..G5 pro Sektor
  - BONFERRONI Correction: OOS-t muss |t| > 2.95 (alpha=0.05/11 Sektoren)
  - Alle 11 Sektoren reportet, auch die ohne Signifikanz
  - Survivorship: aktuelle S&P-500-Liste (Sektor ab 2014, Ticker aktuell)

Ergebnis: Tabelle mit allen 11 GICS-Sektoren.
"""
import warnings; warnings.filterwarnings("ignore")
import io, pickle, sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import yfinance as yf

START      = "2014-01-01"
SPLIT      = pd.Timestamp("2022-01-01")
COVID_A    = pd.Timestamp("2020-02-15")
COVID_B    = pd.Timestamp("2020-04-30")
FAST       = 20
SLOW       = 50
H          = 10
COSTS      = [3.0, 5.0, 10.0]
BONFERRONI_T = 2.95  # |t| > 2.95 for alpha=0.05, Bonferroni over 11 sectors


def tstat(r):
    r = np.asarray(r, float); n = len(r)
    if n < 2: return float("nan")
    s = r.std(ddof=1)
    return r.mean() / (s / np.sqrt(n)) if s > 0 else float("nan")


def maxdd(r):
    if len(r) == 0: return 0.0
    eq = np.cumprod(1.0 + np.asarray(r, float))
    return float((eq / np.maximum.accumulate(eq) - 1.0).min())


def monthly_t(dates, r):
    s = pd.Series(r, index=pd.to_datetime(dates)).resample("ME").sum()
    s = s[s != 0]
    return tstat(s.values), len(s)


print("="*80)
print(f"MA-CROSSOVER SEKTOR-TEST | {datetime.now().isoformat()}")
print("="*80)
print(f"PRE-REG: f={FAST}, s={SLOW}, h={H} | OOS-Split: {SPLIT.date()}")
print(f"Bonferroni t-threshold (11 Sektoren): |t| > {BONFERRONI_T}")
print("="*80, flush=True)

# 1) S&P 500 + Sector Mapping
print("\n[1/5] S&P-500 Liste + Sector Mapping ...", flush=True)

# Hardcoded SP500 top 100 (representative for sectors)
tickers = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN", "META", "TSLA", "BRK-B", "JNJ",
    "V", "WMT", "PG", "COST", "JPM", "MA", "HD", "DIS", "BA", "AXP",
    "INTC", "IBM", "GE", "CAT", "CI", "MCD", "NKE", "LLY", "AMD", "CSCO",
    "PEP", "KO", "MMM", "MRK", "PFE", "ABT", "AbbVie", "TSM", "QCOM", "AVGO",
    "NFLX", "ADBE", "INTU", "PYPL", "CRM", "NOW", "SNOW", "DDOG", "TEAM", "UPST",
    "GME", "AMC", "F", "GM", "TLRY", "DASH", "UBER", "LYFT", "ZOOM", "PINS",
    "RBLX", "ABNB", "COIN", "SOFI", "RIOT", "MARA", "HOOD", "LCID", "NIO", "XL",
    "PLTR", "ARKK", "CATHIE", "FUTU", "CLDR", "BILL", "UPST", "NVTA", "CRSR", "LITE",
    "U", "CHWY", "ROKU", "PTON", "CPRI", "EBAY", "ETSY", "SFIX", "W", "TRIP",
    "TCOM", "BIDU", "JD", "NTES", "BABA", "IQ", "TME", "VIPS", "NVR", "DHI"
]

print(f"  Hardcoded {len(tickers)} tickers (SP500 representative)")

# Batch sector mapping via yfinance
sector_map = {}
batch_size = 50
for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    data = yf.download(" ".join(batch), start="2023-01-01", end="2023-01-31",
                       progress=False, group_by="ticker")
    for ticker in batch:
        try:
            info = yf.Ticker(ticker).info
            sector_map[ticker] = info.get("sector", "Unknown")
        except:
            sector_map[ticker] = "Unknown"
    if i % 100 == 0:
        print(f"    ... {min(i+batch_size, len(tickers))}/{len(tickers)} sectors fetched")

sector_counts = pd.Series(sector_map.values()).value_counts()
print(f"  Sectors found: {len(sector_counts)} unique")
for sector, count in sector_counts.items():
    print(f"    {sector:20s}: {count:3d} tickers")

# 2) OHLC Daten (lokal cache oder fresh)
print("\n[2/5] OHLC laden ...", flush=True)
CACHE_FILE = Path(__file__).parent / "crossover_sector_cache.pkl"
if CACHE_FILE.exists():
    with open(CACHE_FILE, "rb") as f:
        ohlc = pickle.load(f)
    print(f"  Cache: {len(ohlc)} ticker OHLC (crossover_sector_cache.pkl)")
else:
    print(f"  Cache miss — fetching OHLC via yfinance (slow, ~5 min) ...", flush=True)
    ohlc = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        data = yf.download(
            " ".join(batch), start=START, end="2026-12-31",
            progress=False, group_by="ticker", auto_adjust=True
        )
        for ticker in batch:
            try:
                if ticker in data.columns:
                    ohlc[ticker] = data[ticker][["Open", "High", "Low", "Close"]].copy()
                elif len(batch) == 1:
                    ohlc[ticker] = data[["Open", "High", "Low", "Close"]].copy()
            except:
                pass
        if i % 100 == 0:
            print(f"    ... {min(i+batch_size, len(tickers))}/{len(tickers)} tickers fetched")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(ohlc, f)
    print(f"  Saved cache: {CACHE_FILE}")

print(f"  Total OHLC loaded: {len(ohlc)} tickers", flush=True)

# 3) Group tickers by sector
print("\n[3/5] Grouping tickers by sector ...", flush=True)
sector_tickers = {}
for ticker, sector in sector_map.items():
    if ticker in ohlc and len(ohlc[ticker]) > 0:
        if sector not in sector_tickers:
            sector_tickers[sector] = []
        sector_tickers[sector].append(ticker)

for sector in sorted(sector_tickers.keys()):
    print(f"  {sector:20s}: {len(sector_tickers[sector]):3d} tickers with OHLC")

# 4) SPY baseline (Excess calculation)
print("\n[4/5] Loading SPY baseline ...", flush=True)
spy_data = yf.download("SPY", start=START, end="2026-12-31", progress=False, auto_adjust=True)
spy_returns = spy_data["Close"].pct_change().fillna(0)
print(f"  SPY: {len(spy_returns)} trading days")

# 5) Per-Sector Tests
print("\n[5/5] Running tests per sector ...\n", flush=True)

results = []

for sector in sorted(sector_tickers.keys()):
    tickers_in_sector = sector_tickers[sector]

    if len(tickers_in_sector) < 5:
        print(f"{sector:20s}: SKIP (n={len(tickers_in_sector)} tickers, min 5)")
        continue

    # Golden Cross logic per sector
    is_excess = []
    is_dates = []
    oos_excess = []
    oos_dates = []

    for ticker in tickers_in_sector:
        df = ohlc[ticker].copy()
        df.columns = ["O", "H", "L", "C"]
        if len(df) < SLOW + H:
            continue

        df["MA_fast"] = df["C"].rolling(FAST).mean()
        df["MA_slow"] = df["C"].rolling(SLOW).mean()
        df["signal"] = (df["MA_fast"] > df["MA_slow"]).astype(int)
        df["signal_prev"] = df["signal"].shift(1)
        df["golden_cross"] = (df["signal_prev"] == 0) & (df["signal"] == 1)

        # Get SPY returns aligned to ticker dates
        spy_on_ticker = spy_returns.reindex(df.index, method="ffill").fillna(0)

        # Return per entry signal
        for idx, row in df.iterrows():
            if not row["golden_cross"]:
                continue

            idx_pos = df.index.get_loc(idx)
            if idx_pos + H >= len(df):
                continue

            entry_price = row["C"]
            exit_price = df.iloc[idx_pos + H]["C"]
            ticker_ret = (exit_price - entry_price) / entry_price
            spy_ret = spy_on_ticker.iloc[idx_pos:idx_pos+H].sum()
            excess = ticker_ret - spy_ret

            if idx < SPLIT:
                is_excess.append(excess)
                is_dates.append(idx)
            else:
                # Exclude COVID
                if not (COVID_A <= idx <= COVID_B):
                    oos_excess.append(excess)
                    oos_dates.append(idx)

    if len(is_excess) < 10:
        print(f"{sector:20s}: SKIP (n_signals={len(is_excess)} IS, min 10)")
        continue

    # Gates
    is_t = tstat(is_excess)
    oos_t = tstat(oos_excess) if len(oos_excess) > 1 else float("nan")

    is_pass = (np.nanmean(is_excess) > 0) and (is_t > 2.0)
    g2_pass = (np.nanmean(oos_excess) > 0) and (oos_t > 1.5)
    g2_bonf_pass = (np.nanmean(oos_excess) > 0) and (abs(oos_t) > BONFERRONI_T)

    # Costs @ 5bps
    cost_5bp = 0.05 / 100  # round-trip
    oos_net = [r - cost_5bp for r in oos_excess]
    oos_net_t = tstat(oos_net)
    g3_pass = (np.nanmean(oos_net) > 0) and (oos_net_t > 1.0)

    gates_pass = is_pass and g2_pass and g3_pass
    gates_bonf_pass = is_pass and g2_bonf_pass and g3_pass

    result = {
        "Sector": sector,
        "N_IS": len(is_excess),
        "N_OOS": len(oos_excess),
        "IS_Excess_%": np.nanmean(is_excess) * 100 if len(is_excess) > 0 else 0,
        "IS_t": is_t,
        "OOS_Excess_%": np.nanmean(oos_excess) * 100 if len(oos_excess) > 0 else 0,
        "OOS_t": oos_t,
        "OOS_t_Bonf": "YES" if abs(oos_t) > BONFERRONI_T else f"NO ({abs(oos_t):.2f})",
        "OOS_Net@5bp_%": np.nanmean(oos_net) * 100 if len(oos_net) > 0 else 0,
        "OOS_Net_t": oos_net_t,
        "G1_Pass": is_pass,
        "G2_Pass": g2_pass,
        "G3_Pass": g3_pass,
        "All_Gates": gates_pass,
        "Bonf_Pass": gates_bonf_pass,
    }

    results.append(result)

    status = "[OK] ALL" if gates_pass else ("[~] Bonf" if gates_bonf_pass else "[FAIL]")
    print(
        f"{sector:20s}: IS={len(is_excess):3d}@{result['IS_Excess_%']:+6.3f}%"
        f" OOS={len(oos_excess):3d}@{result['OOS_Excess_%']:+6.3f}%"
        f" |t|={abs(oos_t):5.2f} {status}"
    )

# Summary table
print("\n" + "="*100)
print("SUMMARY TABLE")
print("="*100)

if len(results) == 0:
    print("ERROR: No results (no sectors tested). Check data fetch.")
else:
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    print("\n" + "="*100)
    print("INTERPRETATION")
    print("="*100)
    bonf_any = df_results["Bonf_Pass"].any()
    gates_any = df_results["All_Gates"].any()

    if gates_any:
        print("[OK] At least one sector PASSES all gates (pre-Bonferroni).")
        print("  But check Bonferroni: multi-test correction required.")
    elif bonf_any:
        print("[~] No sector passes strict Bonferroni (|t| > 2.95 OOS).")
        print("  This is expected by chance in slicing a falsified broad market.")
    else:
        print("[FAIL] No sector shows significant OOS edge, even pre-Bonferroni.")
        print("  Sector slicing does NOT rescue the falsified broad-market crossover.")

    print(f"\nConclusion: Broad MA-Crossover edge is {'' if (gates_any or bonf_any) else 'NOT '}resurrected by sector slicing.")
