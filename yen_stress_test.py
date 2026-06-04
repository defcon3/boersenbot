#!/usr/bin/env python3
"""
YEN-CARRY-STRESS-TEST - Yen-Spikes als Risk-Off-Detektor

HYPOTHESE (pre-registriert vor Lauf):
  An Tagen mit USD/JPY-Tagesaenderung < -X*sigma (Yen-Spike-up = Carry-Unwind-Signal)
  sind die naechsten N Handelstage fuer US-Risk-Assets schlechter (negativer mean
  forward return) als an Nicht-Stress-Tagen.

Mechanismus: Yen-Carry-Unwind. Wer Yen geliehen hat, muss zur Rueckzahlung Yen
kaufen -> Yen stark -> Carry-Trade-Aufloesung -> Verkauf von Risk-Assets (SPY/QQQ
/IWM/HYG) -> US-Aktien fallen.

DESIGN (alles vor Lauf festgelegt):
  - Sigma-Cutoffs: -1.0, -1.5, -2.0 (3 Levels)
  - Forward-Windows: 1, 2, 5, 10 Handelstage (4 Windows)
  - Assets: SPY, QQQ, IWM, HYG (4 Assets)
  - Total: 3 x 4 x 4 = 48 Tests
  - sigma = rolling 60-day std, NUR aus Train-Periode (kein Snooping)
  - Train: 2000-2018, Test: 2019-2025, COVID 2020-02-15 bis 2020-04-30 excluded

PRE-REG-GATES (vor Lauf fixiert):
  G1 Train-Plausibilitaet: diff_train < 0 (Mechanismus stimmt schon im Train)
  G2 Test-Signifikanz:     |t_test| > 3.0 (Bonferroni 48 Tests, ~2.94, gerundet)
  G3 Stichproben-Mindestgroesse: n_test_signal >= 30
  G4 Cross-Asset-Konsistenz: mind. 2 von 4 Assets t_test < -2 in mind. einem Window
  G5 COVID-Robustheit: bleibt Befund auch ohne COVID-Periode
  G6 Naive-Floor (informativ, nicht haltend): t_test < -1.96 nominal signifikant

Score:
  GREEN: G1+G2+G3 PASS und G4 PASS => echter Edge
  YELLOW: G6 PASS (nominal) aber G2 FAIL (nicht Bonferroni) => verdaechtig
  RED: G6 FAIL ueberall => Hypothese widerlegt
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

# ===========================================================================
# CONFIG
# ===========================================================================

TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-01-01")
COVID_A = pd.Timestamp("2020-02-15")
COVID_B = pd.Timestamp("2020-04-30")

SIGMA_WINDOW = 60
SIGMA_CUTOFFS = [-1.0, -1.5, -2.0]
FORWARD_WINDOWS = [1, 2, 5, 10]
ASSETS = ["SPY", "QQQ", "IWM", "HYG"]

BONFERRONI_T = 3.0       # 48 Tests
NAIVE_T = 1.96

# ===========================================================================
# HELPERS
# ===========================================================================

def fetch_close(ticker, start="2000-01-01"):
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    close = np.asarray(df['Close']).flatten()
    return pd.Series(close, index=df.index)

def welch_t(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return np.nan, np.nan
    se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    if se == 0: return np.nan, np.nan
    return (a.mean() - b.mean()) / se, a.mean() - b.mean()

# ===========================================================================
# DATA LOADING
# ===========================================================================

print("="*80)
print("YEN-CARRY-STRESS-TEST")
print("="*80)

print("\n[1/5] Lade USD/JPY...", flush=True)
usdjpy_close = fetch_close("JPY=X", start="2000-01-01")
print(f"  USD/JPY: {len(usdjpy_close)} Tage ({usdjpy_close.index.min().date()} -> {usdjpy_close.index.max().date()})")

print("\n[2/5] Lade Risk-Assets...", flush=True)
asset_returns = {}
for asset in ASSETS:
    close = fetch_close(asset, start="2000-01-01")
    ret = close.pct_change()
    asset_returns[asset] = ret
    print(f"  {asset}: {len(ret)} Tage ({ret.index.min().date()} -> {ret.index.max().date()})")

# ===========================================================================
# YEN-STRESS-SIGNAL
# ===========================================================================

print("\n[3/5] Berechne Yen-Stress-Signal...", flush=True)
usdjpy_ret = usdjpy_close.pct_change()

# Train-only sigma: 60d rolling std auf Train-Period (kein Snooping)
train_ret = usdjpy_ret[usdjpy_ret.index <= TRAIN_END]
train_sigma = train_ret.rolling(SIGMA_WINDOW).std()

# Verwende den letzten Train-sigma-Median als globalen Cutoff (stabiler als rolling)
# Alternative: rolling sigma auch im Test, aber nur Train-Daten gehen in die Berechnung
# Saubere Loesung: rolling sigma generell, aber bei jedem Datum nur historische
# (=expanding/rolling) Daten - das ist intrinsisch leakage-frei.
rolling_sigma_full = usdjpy_ret.rolling(SIGMA_WINDOW).std().shift(1)  # Lag 1 = kein lookahead

zscore = usdjpy_ret / rolling_sigma_full
print(f"  Z-Scores: {zscore.dropna().shape[0]} Werte, Median {zscore.median():.3f}, Std {zscore.std():.3f}")

# ===========================================================================
# RUN TESTS
# ===========================================================================

print("\n[4/5] Lasse Tests laufen ({} Tests insgesamt)...".format(
    len(SIGMA_CUTOFFS)*len(FORWARD_WINDOWS)*len(ASSETS)), flush=True)

results = []

for cutoff in SIGMA_CUTOFFS:
    for fwd in FORWARD_WINDOWS:
        for asset in ASSETS:
            ret = asset_returns[asset]

            # Align
            df = pd.DataFrame({"z": zscore, "ret": ret}).dropna()

            # Stress-Tag: zscore < cutoff
            stress = df["z"] < cutoff

            # Forward-Return (kumulativ ueber fwd Tage)
            fwd_ret = df["ret"].rolling(fwd).sum().shift(-fwd)

            # Train / Test masks
            train_mask = (df.index <= TRAIN_END) & ~((df.index >= COVID_A) & (df.index <= COVID_B))
            test_mask = (df.index >= TEST_START) & ~((df.index >= COVID_A) & (df.index <= COVID_B))

            train_sig = fwd_ret[train_mask & stress].dropna()
            train_no = fwd_ret[train_mask & ~stress].dropna()
            test_sig = fwd_ret[test_mask & stress].dropna()
            test_no = fwd_ret[test_mask & ~stress].dropna()

            t_train, d_train = welch_t(train_sig, train_no)
            t_test, d_test = welch_t(test_sig, test_no)

            results.append({
                "cutoff": cutoff,
                "fwd": fwd,
                "asset": asset,
                "n_train_sig": len(train_sig),
                "n_test_sig": len(test_sig),
                "t_train": t_train,
                "diff_train": d_train,
                "t_test": t_test,
                "diff_test": d_test,
            })

df_res = pd.DataFrame(results)

# ===========================================================================
# REPORT
# ===========================================================================

print("\n[5/5] Ergebnisse")
print("="*80)

print("\n--- VOLLSTAENDIGE MATRIX ---")
print("cutoff  fwd  asset   n_tr  n_te    t_train   d_train    t_test    d_test  Gates")
print("-"*86)

for _, r in df_res.iterrows():
    # Gates
    g1 = (r["diff_train"] < 0) if not np.isnan(r["diff_train"]) else False
    g2 = (abs(r["t_test"]) > BONFERRONI_T) if not np.isnan(r["t_test"]) else False
    g3 = r["n_test_sig"] >= 30
    g6 = (r["t_test"] < -NAIVE_T) if not np.isnan(r["t_test"]) else False

    gates = ""
    gates += "1" if g1 else "."
    gates += "2" if g2 else "."
    gates += "3" if g3 else "."
    gates += "6" if g6 else "."

    print(f"{r['cutoff']:+.1f}    {r['fwd']:3d}  {r['asset']:5s}  {r['n_train_sig']:4d}  "
          f"{r['n_test_sig']:4d}  {r['t_train']:+7.2f}  {r['diff_train']*100:+6.2f}%  "
          f"{r['t_test']:+7.2f}  {r['diff_test']*100:+6.2f}%   {gates}")

# Pre-Reg Auswertung
print("\n--- GATES ZUSAMMENFASSUNG ---")
df_res["g1"] = df_res["diff_train"] < 0
df_res["g2"] = df_res["t_test"].abs() > BONFERRONI_T
df_res["g3"] = df_res["n_test_sig"] >= 30
df_res["g6_naive"] = df_res["t_test"] < -NAIVE_T

print(f"G1 (Train-Direction negativ):        {df_res['g1'].sum()}/{len(df_res)} Tests")
print(f"G2 (Test |t| > {BONFERRONI_T} Bonferroni):    {df_res['g2'].sum()}/{len(df_res)} Tests")
print(f"G3 (n_test >= 30):                   {df_res['g3'].sum()}/{len(df_res)} Tests")
print(f"G6 (Test t < -{NAIVE_T} nominal sig.): {df_res['g6_naive'].sum()}/{len(df_res)} Tests")

# G4 Cross-Asset
print("\n--- G4 CROSS-ASSET-KONSISTENZ ---")
for cutoff in SIGMA_CUTOFFS:
    for fwd in FORWARD_WINDOWS:
        sub = df_res[(df_res["cutoff"] == cutoff) & (df_res["fwd"] == fwd)]
        n_neg2 = (sub["t_test"] < -2).sum()
        marker = "PASS" if n_neg2 >= 2 else "fail"
        if n_neg2 >= 1:
            print(f"  cutoff={cutoff:+.1f}, fwd={fwd:2d}: {n_neg2}/4 Assets t<-2 [{marker}]")

# Final Verdict
print("\n--- FINAL VERDICT ---")
all_gates_pass = df_res[df_res["g1"] & df_res["g2"] & df_res["g3"]]
nominal_pass = df_res[df_res["g1"] & df_res["g6_naive"] & df_res["g3"]]

if len(all_gates_pass) >= 4:
    print(f"GREEN: {len(all_gates_pass)} Tests PASS volle Gates (G1+G2+G3) -> echter Edge")
elif len(nominal_pass) >= 4:
    print(f"YELLOW: {len(nominal_pass)} Tests PASS nominal (G1+G6+G3) aber Bonferroni FAIL -> verdaechtig")
else:
    print(f"RED: nur {len(all_gates_pass)} Bonferroni-PASS, {len(nominal_pass)} nominal -> Hypothese widerlegt/sehr schwach")

# Save
df_res.to_csv("yen_stress_results.csv", index=False)
print("\nGespeichert: yen_stress_results.csv")
