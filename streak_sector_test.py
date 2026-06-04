#!/usr/bin/env python3
"""
STREAK-MR SEKTOR-DIFFERENZIERUNG (Pre-Reg 2026-05-24)

KONTEXT: Der breite Streak-Mean-Reversion-Befund (N=6 Down-Days -> next-day-up,
Excess vs SPY) war bei MEMORY [[boersenbot-streak-pattern]] und [[boersenbot-
minuten-momentum-ziel]] OOS netto tot bei handelbarer Korbgroesse. Bisher auf
undifferenziertem S&P-500-Korb gemessen.

NEUE HYPOTHESE: Der MR-Edge ist nicht uniform ueber alle GICS-Sektoren, sondern
in einer Klasse konzentriert. Drei plausible Sub-Hypothesen:
  (a) Zyklisch staerker (hoehere Vola -> mehr Ueber-/Untertreibung)
  (b) Defensiv staerker (weniger News-Driver -> mehr Random-Walk)
  (c) Tech staerker (Tech-spezifische Pattern)

DESIGN (vor Lauf festgelegt):
  - S&P 500 + GICS-Sektor aus Wikipedia
  - 3 Sektor-Klassen (broad):
      * ZYKLISCH:  Industrials, Materials, Consumer Discretionary, Financials, Energy
      * DEFENSIV:  Consumer Staples, Health Care, Utilities, Real Estate
      * TECH/COMM: Information Technology, Communication Services
  - Streak-Laengen: N in {4, 5, 6, 7} (Sweet Spot aus Memory = N=6, aber wir
    testen breiter um Lookback-Effekt zu mindern)
  - Holding: t+1 (next-day-only, klassisches Setup)
  - Train: 2015-2020-12-31 / Test: 2021-01-01 - 2026-05-22
  - COVID-Excl: 2020-02-15 - 2020-04-30
  - Excess vs SPY (echtes Alpha, kein Brutto-vs-Cash)
  - Tages-Portfolio: an jedem Tag, halte gleichgewichtet ALLE Sektor-Aktien
    mit N Down-Days am Tagesende (= LONG Open->Close fuer t+1)
  - Mindestkorbgroesse: 8 (vermeidet 2-3-Namen-Illusion siehe Memory)
  - Welch-t vs Null (Excess gegen 0), n = Anzahl Tage mit >=8 Trades

PRE-REG-GATES:
  G1 (Train-Direction): diff_train > 0 (Excess vs SPY positiv im Train)
  G2 (Test-Signifikanz): t_test > Bonferroni 2.7 (12 Tests = 3 Klassen x 4 N)
  G3 (Stichprobe):    n_test_days >= 30
  G4 (Korb-Mindestgr): mean(daily_basket_size) >= 8 in Test
  G5 (COVID-Robust):  Befund haelt auch ohne COVID-Periode (re-run check)
  G6 (Naive-Floor):   t_test > 1.96 nominal (informativ)

VERDICT:
  GREEN: G1+G2+G3+G4 alle PASS in mind. 1 Zelle (Sektor x N)
  YELLOW: G6+G1+G3+G4 PASS aber G2 FAIL -> verdaechtig, kein Edge
  RED: nichts -> Hypothese widerlegt
"""
import warnings; warnings.filterwarnings("ignore")
import pickle
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ===========================================================================
# CONFIG
# ===========================================================================

START = "2015-01-01"
TRAIN_END = pd.Timestamp("2020-12-31")
TEST_START = pd.Timestamp("2021-01-01")
COVID_A = pd.Timestamp("2020-02-15")
COVID_B = pd.Timestamp("2020-04-30")

N_VALUES = [4, 5, 6, 7]
MIN_BASKET_SIZE = 8
BONFERRONI_T = 2.7  # 12 Tests
NAIVE_T = 1.96

CACHE_DIR = Path(__file__).parent
SECTOR_CACHE = CACHE_DIR / "sp500_sectors.csv"
OHLC_CACHE = CACHE_DIR / "sp500_ohlc_2015_2026.pkl"

# GICS-Sektor zu Klasse Mapping
SECTOR_CLASS = {
    # Zyklisch
    "Industrials": "ZYKLISCH",
    "Materials": "ZYKLISCH",
    "Consumer Discretionary": "ZYKLISCH",
    "Financials": "ZYKLISCH",
    "Energy": "ZYKLISCH",
    # Defensiv
    "Consumer Staples": "DEFENSIV",
    "Health Care": "DEFENSIV",
    "Utilities": "DEFENSIV",
    "Real Estate": "DEFENSIV",
    # Tech/Comm
    "Information Technology": "TECH",
    "Communication Services": "TECH",
}

# ===========================================================================
# HELPERS
# ===========================================================================

def welch_t_vs_zero(arr):
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2: return np.nan
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    if se == 0: return np.nan
    return arr.mean() / se

def fetch_sp500_with_sectors():
    if SECTOR_CACHE.exists():
        df = pd.read_csv(SECTOR_CACHE)
        return df
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    tables = pd.read_html(StringIO(r.text))
    df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
    df.columns = ["symbol", "name", "gics_sector"]
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)
    df["sector_class"] = df["gics_sector"].map(SECTOR_CLASS).fillna("OTHER")
    df.to_csv(SECTOR_CACHE, index=False)
    return df

def fetch_all_ohlc(symbols):
    if OHLC_CACHE.exists():
        with open(OHLC_CACHE, "rb") as f:
            return pickle.load(f)
    ohlc = {}
    batch = 50
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i+batch]
        try:
            data = yf.download(" ".join(chunk), start=START, end="2026-12-31",
                               progress=False, group_by="ticker", auto_adjust=True)
            for s in chunk:
                try:
                    if len(chunk) == 1:
                        sub = data
                    else:
                        sub = data[s]
                    close = sub["Close"].dropna()
                    if len(close) > 100:
                        ohlc[s] = close
                except Exception:
                    pass
        except Exception as e:
            print(f"    Batch {i//batch}: FAIL ({e})")
        print(f"    {min(i+batch, len(symbols))}/{len(symbols)} -> {len(ohlc)} ok", flush=True)
    with open(OHLC_CACHE, "wb") as f:
        pickle.dump(ohlc, f)
    return ohlc

def compute_streaks(close_series):
    """Returns DataFrame: index=date, columns=[ret, streak_down_length]"""
    ret = close_series.pct_change()
    is_down = (ret < 0).astype(int)
    # Gaps-and-Islands: streak_length = consecutive downs
    group = (is_down != is_down.shift()).cumsum()
    streak = is_down.groupby(group).cumsum()
    # streak only nonzero when is_down=1
    streak = streak * is_down
    return pd.DataFrame({"ret": ret, "streak": streak}, index=close_series.index)

# ===========================================================================
# MAIN
# ===========================================================================

print("="*80)
print("STREAK-MR SEKTOR-DIFFERENZIERUNG")
print("="*80)

print("\n[1/5] Lade S&P 500 + GICS-Sektoren...", flush=True)
sectors_df = fetch_sp500_with_sectors()
print(f"  {len(sectors_df)} Aktien")
print(f"  Sektor-Klassen:\n{sectors_df['sector_class'].value_counts().to_string()}")

print("\n[2/5] Lade SPY-Baseline...", flush=True)
spy_close = yf.download("SPY", start=START, end="2026-12-31",
                        progress=False, auto_adjust=True)["Close"]
spy_close = pd.Series(np.asarray(spy_close).flatten(), index=spy_close.index)
spy_ret = spy_close.pct_change()
print(f"  SPY: {len(spy_ret)} Tage")

print("\n[3/5] Lade OHLC fuer alle SP500-Aktien (Cache wenn vorhanden)...", flush=True)
symbols = sectors_df["symbol"].tolist()
ohlc = fetch_all_ohlc(symbols)
print(f"  OHLC geladen: {len(ohlc)} Aktien")

# Streaks vorab berechnen
print("\n[4/5] Berechne Streaks pro Aktie...", flush=True)
streak_data = {}
for sym, close in ohlc.items():
    streak_data[sym] = compute_streaks(close)
print(f"  Streak-DFs: {len(streak_data)}")

# ===========================================================================
# Test pro Sektor-Klasse + N
# ===========================================================================

print(f"\n[5/5] Tests (3 Klassen x {len(N_VALUES)} N = {3*len(N_VALUES)} Zellen)\n", flush=True)

results = []

for sector_class in ["ZYKLISCH", "DEFENSIV", "TECH"]:
    syms_in_class = sectors_df[sectors_df["sector_class"] == sector_class]["symbol"].tolist()
    syms_with_data = [s for s in syms_in_class if s in streak_data]

    if len(syms_with_data) < 20:
        print(f"  {sector_class}: SKIP nur {len(syms_with_data)} Symbole")
        continue

    # Pro Tag: welche Symbole haben jetzt Streak >= N?
    # next-day return - SPY-return = excess

    for N in N_VALUES:
        # Sammle daily-portfolio-excess-returns
        daily_records = []

        # Erstelle ein DataFrame mit allen Symbolen aligned
        all_dates = set()
        for s in syms_with_data:
            all_dates.update(streak_data[s].index)
        all_dates = sorted(all_dates)

        # Fuer jedes Datum: portfolio-excess fuer t+1
        date_to_basket = {}
        for s in syms_with_data:
            sdf = streak_data[s]
            # Tage, an denen streak >= N
            qualified = sdf.index[sdf["streak"] >= N]
            for d in qualified:
                # Next-day return
                try:
                    pos = sdf.index.get_loc(d)
                    if pos + 1 >= len(sdf):
                        continue
                    nd = sdf.index[pos + 1]
                    next_ret = sdf.iloc[pos + 1]["ret"]
                    if np.isnan(next_ret):
                        continue
                    # SPY next-day return
                    if nd not in spy_ret.index:
                        continue
                    spy_next = spy_ret.loc[nd]
                    if np.isnan(spy_next):
                        continue
                    excess = next_ret - spy_next
                    if d not in date_to_basket:
                        date_to_basket[d] = []
                    date_to_basket[d].append(excess)
                except Exception:
                    continue

        # Aggregiere pro Tag (mean, wenn mind. MIN_BASKET_SIZE Trades)
        daily_dates = []
        daily_excess = []
        daily_size = []
        for d in sorted(date_to_basket.keys()):
            arr = date_to_basket[d]
            if len(arr) >= MIN_BASKET_SIZE:
                daily_dates.append(d)
                daily_excess.append(np.mean(arr))
                daily_size.append(len(arr))

        if len(daily_dates) == 0:
            results.append({"sector_class": sector_class, "N": N,
                            "n_train": 0, "n_test": 0, "t_train": np.nan,
                            "t_test": np.nan, "diff_train": np.nan,
                            "diff_test": np.nan, "mean_basket_test": 0})
            continue

        daily_dates = pd.DatetimeIndex(daily_dates)
        daily_excess = np.array(daily_excess)
        daily_size = np.array(daily_size)

        # Train/Test masks
        train_mask = (daily_dates <= TRAIN_END) & ~((daily_dates >= COVID_A) & (daily_dates <= COVID_B))
        test_mask = (daily_dates >= TEST_START) & ~((daily_dates >= COVID_A) & (daily_dates <= COVID_B))

        train_exc = daily_excess[train_mask]
        test_exc = daily_excess[test_mask]
        test_basket = daily_size[test_mask]

        t_train = welch_t_vs_zero(train_exc)
        t_test = welch_t_vs_zero(test_exc)
        diff_train = train_exc.mean() if len(train_exc) > 0 else np.nan
        diff_test = test_exc.mean() if len(test_exc) > 0 else np.nan
        mean_basket = test_basket.mean() if len(test_basket) > 0 else 0

        results.append({
            "sector_class": sector_class, "N": N,
            "n_train": len(train_exc), "n_test": len(test_exc),
            "t_train": t_train, "t_test": t_test,
            "diff_train": diff_train, "diff_test": diff_test,
            "mean_basket_test": mean_basket
        })

df_res = pd.DataFrame(results)

# ===========================================================================
# REPORT
# ===========================================================================

print("--- VOLLE MATRIX (3 Klassen x 4 N) ---")
print("class       N   n_tr  n_te    t_train    t_test    d_train    d_test  basket  Gates")
print("-"*94)

for _, r in df_res.iterrows():
    g1 = (r["diff_train"] > 0) if not np.isnan(r["diff_train"]) else False
    g2 = (r["t_test"] > BONFERRONI_T) if not np.isnan(r["t_test"]) else False
    g3 = r["n_test"] >= 30
    g4 = r["mean_basket_test"] >= MIN_BASKET_SIZE
    g6 = (r["t_test"] > NAIVE_T) if not np.isnan(r["t_test"]) else False

    gates = ""
    gates += "1" if g1 else "."
    gates += "2" if g2 else "."
    gates += "3" if g3 else "."
    gates += "4" if g4 else "."
    gates += "6" if g6 else "."

    t_tr_s = f"{r['t_train']:+7.2f}" if not np.isnan(r['t_train']) else "    nan"
    t_te_s = f"{r['t_test']:+7.2f}" if not np.isnan(r['t_test']) else "    nan"
    d_tr_s = f"{r['diff_train']*100:+6.3f}%" if not np.isnan(r['diff_train']) else "  nan%"
    d_te_s = f"{r['diff_test']*100:+6.3f}%" if not np.isnan(r['diff_test']) else "  nan%"

    print(f"{r['sector_class']:10s} {r['N']:2d}  {r['n_train']:5d}  {r['n_test']:5d}  "
          f"{t_tr_s}   {t_te_s}  {d_tr_s}  {d_te_s}  {r['mean_basket_test']:5.1f}  {gates}")

# Aggregat
print("\n--- GATES ZUSAMMENFASSUNG ---")
df_res["g1"] = df_res["diff_train"] > 0
df_res["g2"] = df_res["t_test"] > BONFERRONI_T
df_res["g3"] = df_res["n_test"] >= 30
df_res["g4"] = df_res["mean_basket_test"] >= MIN_BASKET_SIZE
df_res["g6"] = df_res["t_test"] > NAIVE_T

print(f"G1 (Train pos):              {df_res['g1'].sum()}/{len(df_res)}")
print(f"G2 (Test t > 2.7 Bonferroni): {df_res['g2'].sum()}/{len(df_res)}")
print(f"G3 (n_test >= 30):           {df_res['g3'].sum()}/{len(df_res)}")
print(f"G4 (mean basket >= 8):       {df_res['g4'].sum()}/{len(df_res)}")
print(f"G6 (Test t > 1.96 naive):    {df_res['g6'].sum()}/{len(df_res)}")

# Verdict
all_4 = df_res[df_res["g1"] & df_res["g2"] & df_res["g3"] & df_res["g4"]]
naive_4 = df_res[df_res["g1"] & df_res["g6"] & df_res["g3"] & df_res["g4"]]

print("\n--- FINAL VERDICT ---")
if len(all_4) >= 1:
    print(f"GREEN: {len(all_4)} Zelle(n) PASS volle Gates")
    print(all_4[["sector_class", "N", "t_test", "diff_test", "mean_basket_test"]].to_string(index=False))
elif len(naive_4) >= 1:
    print(f"YELLOW: {len(naive_4)} Zelle(n) PASS naive (G1+G3+G4+G6) aber Bonferroni FAIL")
    print(naive_4[["sector_class", "N", "t_test", "diff_test", "mean_basket_test"]].to_string(index=False))
else:
    print("RED: keine Zelle besteht G1+G3+G4+G6 -> Hypothese widerlegt")

df_res.to_csv("streak_sector_results.csv", index=False)
print("\nGespeichert: streak_sector_results.csv")
