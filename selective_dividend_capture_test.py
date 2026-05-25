"""
SELECTIVE DIVIDEND CAPTURE DE — Pre-Reg #2-Test 2026-05-25
Pre-Reg: preregs/selective_dividend_capture_de_2026_05_25.md (commit ad95df6a)

Hypothese: Top-10-Aktien nach Train-Excess (2010-2018) liefern auch in
Test (2019-2025) positives Mean-Excess. Hold-Window N=5 FIX.
"""
import pickle
import pandas as pd
import numpy as np
from scipy import stats

# ----------------- KONSTANTEN aus Pre-Reg #2 -----------------
HOLD_N = 5
TRAIN_END = pd.Timestamp('2018-12-31')
TEST_START = pd.Timestamp('2019-01-01')
TEST_END = pd.Timestamp('2025-12-31')
COVID_START = pd.Timestamp('2020-02-15')
COVID_END = pd.Timestamp('2020-04-30')
MIN_TRAIN_EVENTS = 5
TOP_K = 10
TAX_NET = 1 - 0.26375

EVENTS_CSV = 'dividend_capture_events.csv'

def main():
    print('=' * 78)
    print('SELECTIVE DIVIDEND CAPTURE — Pre-Reg #2 Test')
    print('=' * 78)

    df = pd.read_csv(EVENTS_CSV)
    df['ex_date'] = pd.to_datetime(df['ex_date'])
    df = df[df['N'] == HOLD_N].copy()
    print(f'\nGesamt-Events bei N={HOLD_N}: {len(df)}')

    train = df[df['ex_date'] <= TRAIN_END]
    test = df[(df['ex_date'] >= TEST_START) & (df['ex_date'] <= TEST_END)]
    print(f'  Train: {len(train)} | Test: {len(test)}')

    # ----- Top-10 aus Train selektieren -----
    train_grp = train.groupby('ticker')['excess_pct'].agg(['mean', 'count']).reset_index()
    train_grp = train_grp[train_grp['count'] >= MIN_TRAIN_EVENTS]
    train_grp = train_grp.sort_values('mean', ascending=False)
    top10 = train_grp.head(TOP_K).copy()

    print('\n' + '=' * 78)
    print(f'TOP {TOP_K} aus Train (gerankt nach mean(Train-Excess) bei N={HOLD_N})')
    print('=' * 78)
    print(top10.to_string(index=False, float_format=lambda x: f'{x:+.3f}'))

    top10_tickers = top10['ticker'].tolist()

    # ----- Test auf den Top-10 -----
    test_sel = test[test['ticker'].isin(top10_tickers)].copy()
    print(f'\nTest-Events fuer Top-10: {len(test_sel)} (erwartet ~60-70)')

    arr = test_sel['excess_pct'].values
    n_test = len(arr)
    mean_test = float(np.mean(arr))
    t_test, p2 = stats.ttest_1samp(arr, 0)
    p1 = p2 / 2 if t_test > 0 else 1 - p2 / 2
    median_test = float(np.median(arr))
    try:
        wilcox = stats.wilcoxon(arr, alternative='greater')
        wilcox_p = float(wilcox.pvalue)
    except Exception:
        wilcox_p = 1.0

    print('\n' + '=' * 78)
    print('TEST-STATISTIK (aggregat)')
    print('=' * 78)
    print(f'  n              = {n_test}')
    print(f'  mean           = {mean_test:+.4f} %')
    print(f'  t-test         = {t_test:+.3f}')
    print(f'  p1 (one-sided) = {p1:.5f}')
    print(f'  median         = {median_test:+.4f} %')
    print(f'  wilcox p1      = {wilcox_p:.5f}')

    # ----- Pro Aktie -----
    print('\n' + '=' * 78)
    print('TEST-STATISTIK pro Aktie (fuer G5)')
    print('=' * 78)
    per_ticker = test_sel.groupby('ticker')['excess_pct'].agg(['mean', 'count']).reset_index()
    per_ticker = per_ticker.sort_values('mean', ascending=False)
    print(per_ticker.to_string(index=False, float_format=lambda x: f'{x:+.3f}'))
    positive_tickers = (per_ticker['mean'] > 0).sum()
    print(f'\nAktien mit Test-Mean > 0: {positive_tickers} von {len(per_ticker)}')

    # ----- Gates -----
    g1 = mean_test > 0
    g2 = t_test > 2.0
    g3 = (median_test > 0) and (wilcox_p < 0.05)
    g4 = mean_test > 0.5
    g5 = positive_tickers >= 6

    print('\n' + '=' * 78)
    print('GATE-CHECK')
    print('=' * 78)
    print(f'  G1 (Mean > 0):            {"PASS" if g1 else "FAIL":4s}  mean = {mean_test:+.3f}')
    print(f'  G2 (t > 2.0):             {"PASS" if g2 else "FAIL":4s}  t    = {t_test:+.3f}')
    print(f'  G3 (Median>0, Wilcox<.05): {"PASS" if g3 else "FAIL":4s}  med={median_test:+.3f}  wilcox_p={wilcox_p:.4f}')
    print(f'  G4 (Mean > 0.5%):         {"PASS" if g4 else "FAIL":4s}  mean = {mean_test:+.3f}')
    print(f'  G5 (>= 6/10 positiv):     {"PASS" if g5 else "FAIL":4s}  {positive_tickers}/10')
    all_pass = g1 and g2 and g3 and g4 and g5

    print('\n' + '=' * 78)
    if all_pass:
        print('VERDICT: GREEN — selektive Strategie besteht ALLE 5 Gates.')
        print('Edge-Kandidat. Naechster Schritt: Forward-Test als neue Pre-Reg.')
    else:
        print('VERDICT: RED — selektive Strategie FAILED Gate(s).')
        if not g1:
            print('  G1 failed -> keine selektive Outperformance in Test')
        elif not g2:
            print('  G1 PASS, G2 FAIL -> Effekt positiv aber nicht signifikant (Rauschen-Verdacht)')
        elif not g3:
            print('  G2 PASS, G3 FAIL -> Outlier-getrieben, kein konsistentes Pattern')
        elif not g4:
            print('  G3 PASS, G4 FAIL -> statistisch sig. aber praktisch irrelevant')
        elif not g5:
            print('  G4 PASS, G5 FAIL -> von 1-2 Ausreissern getragen, nicht broad pattern')
    print('=' * 78)

    # Export
    top10.to_csv('selective_div_top10_train.csv', index=False)
    test_sel.to_csv('selective_div_test_events.csv', index=False)
    per_ticker.to_csv('selective_div_test_per_ticker.csv', index=False)

    return {
        'top10': top10_tickers,
        'test_n': n_test,
        'mean_test': mean_test,
        't_test': float(t_test),
        'p1_test': float(p1),
        'median_test': median_test,
        'wilcox_p': wilcox_p,
        'positive_tickers': int(positive_tickers),
        'gates': dict(G1=g1, G2=g2, G3=g3, G4=g4, G5=g5, all_pass=all_pass),
    }


if __name__ == '__main__':
    main()
