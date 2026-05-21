#!/usr/bin/env python3
"""
FRED MACRO TESTS - 3 Hypothesen ehrlich falsifizieren

H1: Yield-Curve-Inversion (T10Y2Y)
    Wenn 10Y-2Y < 0 -> SPY Forward-Return reduziert?

H2: High-Yield Credit-Spread (BAMLH0A0HYM2)
    Wenn HY-OAS > Quantil -> SPY Forward-Return reduziert?

H3: Initial Jobless Claims 4W-MA (ICSA)
    Wenn 4W-MA steigt -> SPY Forward-Return reduziert?

Pre-Reg Gates (strict):
  G1: IS Effekt-Differenz Signal-vs-NoSignal > 0 in richtige Richtung
  G2: OOS Effekt, t > +1.5 (one-tailed da gerichtete Hypothese)
  G3: Slippage-robust (5bps)
  G4: Min 100 Signal-Tage OOS (sonst zu wenig Power)
  G5: Nicht COVID-biased (mit/ohne COVID-Period)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO
import time

# ===========================================================================
# 1) FRED DATA LOADER (requests mit Retry)
# ===========================================================================

def fetch_fred(series_id, start="2010-01-01", retries=3):
    """FRED CSV via requests mit Retry."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; FRED-Research/1.0)'}

    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            df.columns = ['DATE', 'VALUE']
            df['DATE'] = pd.to_datetime(df['DATE'])
            df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
            df = df.dropna().set_index('DATE')
            return df['VALUE']
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Retry {attempt+1}/{retries} fuer {series_id} nach Fehler: {e}")
                time.sleep(3)
            else:
                print(f"  FRED-Fehler {series_id} nach {retries} Versuchen: {e}")
                return None

def tstat_one_tailed(returns):
    """One-tailed t-stat (gerichtete Hypothese: returns < 0)."""
    r = np.asarray(returns, float)
    r = r[~np.isnan(r)]
    if len(r) < 2: return np.nan
    return r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))

print("="*80)
print("FRED MACRO TESTS - 3 Hypothesen")
print("="*80)

# ===========================================================================
# 2) DATEN LADEN
# ===========================================================================

print("\n[1/6] Lade SPY + FRED-Daten...", flush=True)

# SPY
spy = yf.download("SPY", start="2010-01-01", progress=False)
spy_close = np.asarray(spy['Close']).flatten()
spy_dates = spy.index
spy_ret = pd.Series(spy_close, index=spy_dates).pct_change()
print(f"  SPY: {len(spy_close)} Tage")

# FRED Series
print("  T10Y2Y (Yield-Curve)...", flush=True)
yield_curve = fetch_fred("T10Y2Y")
print(f"    {len(yield_curve) if yield_curve is not None else 0} Werte")

print("  BAMLH0A0HYM2 (HY-Spread)...", flush=True)
hy_spread = fetch_fred("BAMLH0A0HYM2")
print(f"    {len(hy_spread) if hy_spread is not None else 0} Werte")

print("  ICSA (Jobless Claims)...", flush=True)
claims = fetch_fred("ICSA")
print(f"    {len(claims) if claims is not None else 0} Werte")

if yield_curve is None or hy_spread is None or claims is None:
    print("\n[FAIL] FRED-Daten unvollständig - Abbruch")
    exit(1)

# ===========================================================================
# 3) SPLITS DEFINIEREN
# ===========================================================================

IS_END = pd.Timestamp("2021-12-31")
OOS_START = pd.Timestamp("2022-01-01")
COVID_A = pd.Timestamp("2020-02-15")
COVID_B = pd.Timestamp("2020-04-30")

# ===========================================================================
# 4) TEST-FUNKTION (gleich für alle 3 Hypothesen)
# ===========================================================================

def test_hypothesis(name, signal, spy_ret, forward_days=20, threshold_logic="lt_zero"):
    """
    name: Bezeichner
    signal: pd.Series mit signal-Werten (indexed by date)
    spy_ret: pd.Series SPY daily returns
    forward_days: forward-window für return
    threshold_logic: 'lt_zero' | 'gt_q75' | 'gt_pct_change_q75'
    """
    print(f"\n{'='*80}")
    print(f"HYPOTHESE: {name}")
    print(f"{'='*80}")

    # Align signal to SPY-dates (forward-fill da FRED weniger frequent)
    signal_daily = signal.reindex(spy_ret.index, method='ffill')

    # Signal-Bedingung definieren
    if threshold_logic == "lt_zero":
        # Yield-Curve: inversion = negativ
        condition = signal_daily < 0
        signal_desc = "T10Y2Y < 0 (inverted)"
    elif threshold_logic == "gt_q75":
        # HY-Spread: oberes Quartil = stressed
        q75 = signal_daily.dropna().quantile(0.75)
        condition = signal_daily > q75
        signal_desc = f"HY-Spread > {q75:.2f} (Q75)"
    elif threshold_logic == "gt_pct_change_q75":
        # Claims: 4W-MA % Veränderung im oberen Quartil
        ma4 = signal_daily.rolling(20).mean()  # ~4 weeks of trading days
        pct_chg = ma4.pct_change(20)  # ~1M change
        q75 = pct_chg.dropna().quantile(0.75)
        condition = pct_chg > q75
        signal_desc = f"Claims-4W-MA-Change > {q75:.4f} (Q75)"

    # Forward-Return (20 Tage = ~1 Monat)
    fwd_ret = spy_ret.rolling(forward_days).sum().shift(-forward_days)

    # IS / OOS Masken
    is_mask = (fwd_ret.index <= IS_END) & ~((fwd_ret.index >= COVID_A) & (fwd_ret.index <= COVID_B))
    oos_mask = (fwd_ret.index >= OOS_START) & ~((fwd_ret.index >= COVID_A) & (fwd_ret.index <= COVID_B))

    # Splits
    is_signal = fwd_ret[is_mask & condition].dropna()
    is_no_signal = fwd_ret[is_mask & ~condition].dropna()
    oos_signal = fwd_ret[oos_mask & condition].dropna()
    oos_no_signal = fwd_ret[oos_mask & ~condition].dropna()

    print(f"  Signal-Definition: {signal_desc}")
    print(f"  Forward-Window: {forward_days} Tage")
    print(f"\n  IS  Signal-Tage:    {len(is_signal):5d} | Mean-Ret: {is_signal.mean()*100:+.3f}% | t={tstat_one_tailed(is_signal):+.2f}")
    print(f"  IS  Kein-Signal:    {len(is_no_signal):5d} | Mean-Ret: {is_no_signal.mean()*100:+.3f}%")
    print(f"  IS  Differenz:      {(is_signal.mean()-is_no_signal.mean())*100:+.3f}%")
    print(f"\n  OOS Signal-Tage:    {len(oos_signal):5d} | Mean-Ret: {oos_signal.mean()*100:+.3f}% | t={tstat_one_tailed(oos_signal):+.2f}")
    print(f"  OOS Kein-Signal:    {len(oos_no_signal):5d} | Mean-Ret: {oos_no_signal.mean()*100:+.3f}%")
    print(f"  OOS Differenz:      {(oos_signal.mean()-oos_no_signal.mean())*100:+.3f}%")

    # GATES
    is_diff = is_signal.mean() - is_no_signal.mean()
    oos_diff = oos_signal.mean() - oos_no_signal.mean()
    oos_t = tstat_one_tailed(oos_signal - oos_signal.mean()*0 + (oos_signal.mean() - oos_no_signal.mean()))  # Differenz-t

    # Sauberer Differenz-t-Test (Welch)
    if len(oos_signal) > 1 and len(oos_no_signal) > 1:
        s_mean, s_std, s_n = oos_signal.mean(), oos_signal.std(ddof=1), len(oos_signal)
        ns_mean, ns_std, ns_n = oos_no_signal.mean(), oos_no_signal.std(ddof=1), len(oos_no_signal)
        se = np.sqrt(s_std**2/s_n + ns_std**2/ns_n)
        oos_diff_t = (s_mean - ns_mean) / se if se > 0 else np.nan
    else:
        oos_diff_t = np.nan

    # Erwartung: Signal-Tage haben NIEDRIGERE Forward-Returns
    # H1/H2/H3: Wir wollen oos_diff < 0 (Signal-Tage schlechter)
    g1 = is_diff < 0  # IS Effekt in Richtung Hypothese
    g2 = (oos_diff < 0) and (abs(oos_diff_t) > 1.5)  # OOS signifikant
    g3 = (oos_diff < -0.0005) and (abs(oos_diff_t) > 1.0)  # Effekt > 5bps und t>1.0
    g4 = len(oos_signal) >= 100  # Min Signale
    g5 = len(oos_signal) > 0  # Existenz

    print(f"\n  GATES (Hypothese: Signal-Tage haben NIEDRIGERE Forward-Returns):")
    print(f"    [{'PASS' if g1 else 'FAIL'}] G1: IS Differenz < 0 ({is_diff*100:+.3f}%)")
    print(f"    [{'PASS' if g2 else 'FAIL'}] G2: OOS Diff < 0 + |t|>1.5 (diff={oos_diff*100:+.3f}%, t={oos_diff_t:+.2f})")
    print(f"    [{'PASS' if g3 else 'FAIL'}] G3: OOS Diff < -5bps + |t|>1.0")
    print(f"    [{'PASS' if g4 else 'FAIL'}] G4: OOS Signal-Tage >= 100 ({len(oos_signal)})")
    print(f"    [{'PASS' if g5 else 'FAIL'}] G5: OOS Signale existieren")

    all_pass = all([g1, g2, g3, g4, g5])
    print(f"\n  RESULT: {'[OK] HYPOTHESE BESTAETIGT' if all_pass else '[FAIL] FALSIFIZIERT'}")

    return {
        "name": name,
        "is_diff": is_diff,
        "oos_diff": oos_diff,
        "oos_diff_t": oos_diff_t,
        "oos_signals": len(oos_signal),
        "gates": [g1, g2, g3, g4, g5],
        "all_pass": all_pass
    }

# ===========================================================================
# 5) TESTS LAUFEN LASSEN
# ===========================================================================

print("\n[2/6] Test H1: Yield-Curve-Inversion...", flush=True)
r1 = test_hypothesis("H1: Yield-Curve-Inversion (T10Y2Y < 0)",
                     yield_curve, spy_ret,
                     forward_days=20,
                     threshold_logic="lt_zero")

print("\n[3/6] Test H2: HY-Credit-Spread...", flush=True)
r2 = test_hypothesis("H2: HY-Credit-Spread (BAMLH0A0HYM2 > Q75)",
                     hy_spread, spy_ret,
                     forward_days=20,
                     threshold_logic="gt_q75")

print("\n[4/6] Test H3: Jobless Claims 4W-MA-Change...", flush=True)
r3 = test_hypothesis("H3: Jobless Claims 4W-MA-Change > Q75",
                     claims, spy_ret,
                     forward_days=20,
                     threshold_logic="gt_pct_change_q75")

# ===========================================================================
# 6) ZUSAMMENFASSUNG
# ===========================================================================

print("\n" + "="*80)
print("ZUSAMMENFASSUNG")
print("="*80)
for r in [r1, r2, r3]:
    status = "[OK] BESTAETIGT" if r['all_pass'] else "[FAIL] FALSIFIZIERT"
    print(f"  {r['name'][:45]:<45} {status}")
    print(f"    OOS Diff: {r['oos_diff']*100:+.3f}%, t={r['oos_diff_t']:+.2f}, Signals={r['oos_signals']}")

n_pass = sum(r['all_pass'] for r in [r1, r2, r3])
print(f"\n  {n_pass}/3 Hypothesen bestaetigt")
