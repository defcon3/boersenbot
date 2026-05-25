"""
DIVIDEND CAPTURE DE — Pre-Reg-Test 2026-05-25
Pre-Reg: preregs/dividend_capture_de_2026_05_25.md (commit 871bac1b)

Hypothese: Kauf T-1 (Cum-Day Close), Verkauf T+N (Close), DAX-40 + MDAX-50,
Netto-Excess vs ^GDAXI nach 26.375 % Steuer auf Brutto-Dividende.

Hold-Windows: N in {1, 3, 5, 10, 20} Trading-Tage zwischen Buy und Sell.
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

# ----------------------------- KONSTANTEN ------------------------------------
DAX40 = [
    'ADS.DE', 'AIR.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE', 'BMW.DE',
    'BNR.DE', 'CBK.DE', 'CON.DE', '1COV.DE', 'DBK.DE', 'DB1.DE', 'DHL.DE',
    'DTE.DE', 'DTG.DE', 'ENR.DE', 'EOAN.DE', 'FRE.DE', 'HEI.DE', 'HEN3.DE',
    'HNR1.DE', 'IFX.DE', 'MBG.DE', 'MRK.DE', 'MTX.DE', 'MUV2.DE', 'P911.DE',
    'PAH3.DE', 'QIA.DE', 'RHM.DE', 'RWE.DE', 'SAP.DE', 'SHL.DE', 'SIE.DE',
    'SRT3.DE', 'SY1.DE', 'VOW3.DE', 'VNA.DE', 'ZAL.DE',
]
MDAX50 = [
    'AIXA.DE', 'AOX.DE', 'ARL.DE', 'BC8.DE', 'BOSS.DE', 'COK.DE', 'DEZ.DE',
    'DUE.DE', 'EVD.DE', 'EVK.DE', 'EVT.DE', 'FIE.DE', 'FNTN.DE', 'FPE3.DE',
    'FRA.DE', 'G1A.DE', 'GBF.DE', 'GFT.DE', 'GLJ.DE', 'GXI.DE', 'HAB.DE',
    'HEN.DE', 'HFG.DE', 'HLE.DE', 'HOT.DE', 'JUN3.DE', 'KGX.DE', 'KRN.DE',
    'LEG.DE', 'LHA.DE', 'LXS.DE', 'NDA.DE', 'NEM.DE', 'PNE3.DE', 'PSM.DE',
    'RAA.DE', 'RHK.DE', 'SAX.DE', 'SDF.DE', 'SHA.DE', 'SOW.DE', 'STO3.DE',
    'SZG.DE', 'SZU.DE', 'TKA.DE', 'TLX.DE', 'TUI1.DE', 'UN01.DE', 'WAF.DE',
    'WCH.DE',
]
TICKERS = DAX40 + MDAX50
BENCHMARK = '^GDAXI'

START = '2010-01-01'
END = '2025-12-31'
COVID_START = pd.Timestamp('2020-02-15')
COVID_END = pd.Timestamp('2020-04-30')

TAX_NET = 1 - 0.26375  # 0.73625
HOLD_WINDOWS = [1, 3, 5, 10, 20]
TRAIN_END = pd.Timestamp('2018-12-31')
TEST_START = pd.Timestamp('2019-01-01')

CACHE = Path('dax_mdax_divs_2010_2025.pkl')
BENCH_CACHE = Path('gdaxi_2010_2025.pkl')


# ------------------------------ DATEN FETCH -----------------------------------
def fetch_data():
    if CACHE.exists():
        print(f'[Cache] {CACHE} existiert -> lade')
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    print(f'[Fetch] Lade {len(TICKERS)} Ticker von yfinance...')
    data = {}
    n_ok, n_fail = 0, 0
    for i, t in enumerate(TICKERS, 1):
        try:
            tk = yf.Ticker(t)
            hist = tk.history(start=START, end=END, auto_adjust=False)
            if len(hist) < 100:
                print(f'  [{i}/{len(TICKERS)}] {t}: nur {len(hist)} Tage -> SKIP')
                n_fail += 1
                continue
            divs = hist['Dividends'][hist['Dividends'] > 0]
            data[t] = {
                'prices': hist[['Open', 'High', 'Low', 'Close']].copy(),
                'dividends': divs.copy(),
            }
            print(f'  [{i}/{len(TICKERS)}] {t}: {len(hist)} Tage, {len(divs)} Div-Events')
            n_ok += 1
        except Exception as e:
            print(f'  [{i}/{len(TICKERS)}] {t}: FAIL ({e})')
            n_fail += 1
        time.sleep(0.1)
    print(f'[Fetch] OK={n_ok} FAIL={n_fail}')
    with open(CACHE, 'wb') as f:
        pickle.dump(data, f)
    return data


def fetch_benchmark():
    if BENCH_CACHE.exists():
        with open(BENCH_CACHE, 'rb') as f:
            return pickle.load(f)
    print(f'[Fetch] {BENCHMARK}...')
    bench = yf.Ticker(BENCHMARK).history(start=START, end=END, auto_adjust=False)['Close']
    with open(BENCH_CACHE, 'wb') as f:
        pickle.dump(bench, f)
    return bench


# ------------------------------ EVENT BUILD ----------------------------------
def _normalize(idx_or_ts):
    ts = pd.Timestamp(idx_or_ts)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def build_events(data, benchmark):
    """
    Pro Dividenden-Event:
    - buy_idx = letzter Trading-Tag VOR ex_date (cum-day close)
    - sell_idx = buy_idx + N
    """
    benchmark = benchmark.copy()
    benchmark.index = benchmark.index.tz_localize(None) if benchmark.index.tz is not None else benchmark.index
    benchmark.index = pd.DatetimeIndex([_normalize(x) for x in benchmark.index])

    events = []
    for ticker, d in data.items():
        prices = d['prices'].copy()
        prices.index = pd.DatetimeIndex([_normalize(x) for x in prices.index])
        divs = d['dividends'].copy()
        divs.index = pd.DatetimeIndex([_normalize(x) for x in divs.index])

        for ex_date, div_amount in divs.items():
            if div_amount <= 0:
                continue
            if COVID_START <= ex_date <= COVID_END:
                continue
            # Find buy_idx: letzter Trading-Tag strikt VOR ex_date
            mask_before = prices.index < ex_date
            if not mask_before.any():
                continue
            buy_idx_label = prices.index[mask_before][-1]
            buy_pos = prices.index.get_loc(buy_idx_label)
            buy_price = float(prices['Close'].iloc[buy_pos])

            for N in HOLD_WINDOWS:
                sell_pos = buy_pos + N
                if sell_pos >= len(prices):
                    continue
                sell_idx_label = prices.index[sell_pos]
                sell_price = float(prices['Close'].iloc[sell_pos])
                net_div = float(div_amount) * TAX_NET
                strat_return = (sell_price - buy_price + net_div) / buy_price

                # Benchmark same window
                b_buy_idx = benchmark.index[benchmark.index <= buy_idx_label]
                b_sell_idx = benchmark.index[benchmark.index <= sell_idx_label]
                if len(b_buy_idx) == 0 or len(b_sell_idx) == 0:
                    continue
                b_buy = float(benchmark.loc[b_buy_idx[-1]])
                b_sell = float(benchmark.loc[b_sell_idx[-1]])
                bench_return = (b_sell - b_buy) / b_buy
                excess = strat_return - bench_return

                events.append({
                    'ticker': ticker,
                    'ex_date': ex_date,
                    'buy_date': buy_idx_label,
                    'sell_date': sell_idx_label,
                    'N': N,
                    'div_gross_pct': float(div_amount) / buy_price * 100,
                    'strat_return_pct': strat_return * 100,
                    'bench_return_pct': bench_return * 100,
                    'excess_pct': excess * 100,
                })
    return pd.DataFrame(events)


# ------------------------------ STATS ----------------------------------------
def one_sided_p(t, p_two):
    """Convert two-sided p to one-sided (right tail)."""
    return p_two / 2 if t > 0 else 1 - p_two / 2


def analyze(events):
    train = events[events['ex_date'] <= TRAIN_END]
    test = events[events['ex_date'] >= TEST_START]
    rows = []
    for N in HOLD_WINDOWS:
        tr = train[train['N'] == N]['excess_pct'].values
        te = test[test['N'] == N]['excess_pct'].values
        if len(tr) < 10 or len(te) < 10:
            rows.append({'N': N, 'n_train': len(tr), 'n_test': len(te), 'note': 'too few samples'})
            continue

        tr_mean = float(np.mean(tr))
        te_mean = float(np.mean(te))
        tr_t, tr_p2 = stats.ttest_1samp(tr, 0)
        te_t, te_p2 = stats.ttest_1samp(te, 0)
        tr_p1 = one_sided_p(tr_t, tr_p2)
        te_p1 = one_sided_p(te_t, te_p2)
        te_median = float(np.median(te))
        try:
            wilcox = stats.wilcoxon(te, alternative='greater')
            wilcox_p = float(wilcox.pvalue)
        except Exception:
            wilcox_p = 1.0

        rows.append({
            'N': N,
            'n_train': len(tr), 'mean_train_%': tr_mean, 't_train': float(tr_t), 'p1_train': float(tr_p1),
            'n_test': len(te), 'mean_test_%': te_mean, 't_test': float(te_t), 'p1_test': float(te_p1),
            'median_test_%': te_median, 'wilcox_p_test': wilcox_p,
        })
    return pd.DataFrame(rows)


def check_gates(stats_df):
    """
    G1: t_train > 2.0 AND mean_train > 0
    G2: t_test > 2.58 (Bonferroni K=5, einseitig) AND mean_test > 0
    G3: median_test > 0 AND wilcox_p < 0.05
    G4: mean_test > 0.3 (%) — practical significance
    """
    results = []
    for _, row in stats_df.iterrows():
        if 'note' in row and isinstance(row.get('note'), str):
            results.append({'N': row['N'], 'G1': False, 'G2': False, 'G3': False, 'G4': False, 'all_pass': False})
            continue
        g1 = (row['t_train'] > 2.0) and (row['mean_train_%'] > 0)
        g2 = (row['t_test'] > 2.58) and (row['mean_test_%'] > 0)
        g3 = (row['median_test_%'] > 0) and (row['wilcox_p_test'] < 0.05)
        g4 = row['mean_test_%'] > 0.3
        results.append({
            'N': int(row['N']),
            'G1': bool(g1), 'G2': bool(g2), 'G3': bool(g3), 'G4': bool(g4),
            'all_pass': bool(g1 and g2 and g3 and g4),
        })
    return pd.DataFrame(results)


# ------------------------------ MAIN -----------------------------------------
def main():
    print('=' * 78)
    print('DIVIDEND CAPTURE DE — Pre-Reg-Test 2026-05-25')
    print('=' * 78)

    data = fetch_data()
    benchmark = fetch_benchmark()
    print(f'\n[Build] {len(data)} Aktien mit Daten, Benchmark {len(benchmark)} Datenpunkte')

    events = build_events(data, benchmark)
    print(f'[Build] {len(events)} Event-Reihen (alle N kombiniert)')
    print(f'[Build] Davon Train: {(events.ex_date <= TRAIN_END).sum()}, Test: {(events.ex_date >= TEST_START).sum()}')

    events.to_csv('dividend_capture_events.csv', index=False)
    print(f'[Save] dividend_capture_events.csv ({len(events)} Zeilen)')

    print('\n' + '=' * 78)
    print('STATISTIK PRO HOLD-WINDOW N')
    print('=' * 78)
    stats_df = analyze(events)
    print(stats_df.to_string(index=False))
    stats_df.to_csv('dividend_capture_stats.csv', index=False)

    print('\n' + '=' * 78)
    print('GATE-CHECK')
    print('=' * 78)
    gates = check_gates(stats_df)
    print(gates.to_string(index=False))
    gates.to_csv('dividend_capture_gates.csv', index=False)

    any_pass = gates['all_pass'].any()
    print('\n' + '=' * 78)
    if any_pass:
        winners = gates[gates['all_pass']]['N'].tolist()
        print(f'VERDICT: GREEN — Hold-Window(s) {winners} bestehen ALLE 4 Gates')
        print('Edge-Kandidat. Naechster Schritt: Forward-Test mit neuer Pre-Reg.')
    else:
        print('VERDICT: RED — kein Hold-Window besteht alle 4 Gates')
        # Schau nach Type
        any_g1 = gates['G1'].any()
        if not any_g1:
            print('  Strukturell: KEIN Train-Window hatte direction-pass (G1) -> Hypothese stark widerlegt')
        else:
            g1_winners = gates[gates['G1']]['N'].tolist()
            print(f'  G1 passed fuer N={g1_winners}, aber spaeter gescheitert (G2/G3/G4)')
    print('=' * 78)


if __name__ == '__main__':
    main()
