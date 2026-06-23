"""
EX-TAG-REBOUND-TEST — 2026-06-23
Hypothese (Nutzer, visuell von veitluther.de-Charts):
  "2-3 Tage NACH dem Ex-Tag ist der Kurs genauso tief oder tiefer,
   dann 2-5 Tage spaeter ist er wieder DEUTLICH erhoeht."

Das ist NICHT der klassische Dividend-Capture (Kauf Cum-Tag T-1) — sondern
ein DELAYED-DIP-Setup: Kauf erst in die Delle (Ex+2 / Ex+3), Verkauf 2-5 Tage
spaeter. KEINE Dividende eingesammelt (Kauf liegt nach Ex-Tag) -> reine
Kurs-Mean-Reversion-These.

Daten: DAX40 + MDAX50, 2010-2025 (Cache aus dividend_capture_test.py).
Excess IMMER vs ^GDAXI (kritisch: DE-Dividenden clustern im Fruehjahr ->
roher Anstieg waere sonst nur Bullenmarkt-Drift, nicht Div-Effekt).
COVID (15.02.-30.04.2020) ausgeschlossen. OOS-Split: Train <=2018, Test >=2019.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CACHE = Path('dax_mdax_divs_2010_2025.pkl')
BENCH_CACHE = Path('gdaxi_2010_2025.pkl')

COVID_START = pd.Timestamp('2020-02-15')
COVID_END = pd.Timestamp('2020-04-30')
TRAIN_END = pd.Timestamp('2018-12-31')
TEST_START = pd.Timestamp('2019-01-01')

MAX_K = 12  # Pfad bis Ex+12 Trading-Tage


def _norm(x):
    ts = pd.Timestamp(x)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def load():
    with open(CACHE, 'rb') as f:
        data = pickle.load(f)
    with open(BENCH_CACHE, 'rb') as f:
        bench = pickle.load(f)
    bench.index = pd.DatetimeIndex([_norm(x) for x in bench.index])
    return data, bench


def build_paths(data, bench):
    """Pro Event: kumul. Rendite Aktie & Benchmark von Ex-Tag-Close zu Ex+k."""
    rows = []
    for ticker, d in data.items():
        prices = d['prices'].copy()
        prices.index = pd.DatetimeIndex([_norm(x) for x in prices.index])
        closes = prices['Close']
        divs = d['dividends'].copy()
        divs.index = pd.DatetimeIndex([_norm(x) for x in divs.index])

        for ex_date, amt in divs.items():
            if amt <= 0 or COVID_START <= ex_date <= COVID_END:
                continue
            # ex_pos = Zeile des Ex-Tags (Dividende steht auf dem Ex-Tag)
            if ex_date not in prices.index:
                pos_arr = prices.index[prices.index >= ex_date]
                if len(pos_arr) == 0:
                    continue
                ex_label = pos_arr[0]
            else:
                ex_label = ex_date
            ex_pos = prices.index.get_loc(ex_label)
            if ex_pos + MAX_K >= len(prices):
                continue
            base = float(closes.iloc[ex_pos])
            div_pct = float(amt) / base * 100

            # Benchmark-Close am Ex-Tag (oder letzter davor)
            b_at = bench.index[bench.index <= ex_label]
            if len(b_at) == 0:
                continue
            b_base = float(bench.loc[b_at[-1]])

            rec = {'ticker': ticker, 'ex_date': ex_date, 'div_pct': div_pct}
            for k in range(0, MAX_K + 1):
                px = float(closes.iloc[ex_pos + k])
                stock_ret = (px - base) / base * 100
                lbl = prices.index[ex_pos + k]
                b_idx = bench.index[bench.index <= lbl]
                b_px = float(bench.loc[b_idx[-1]]) if len(b_idx) else b_base
                bench_ret = (b_px - b_base) / b_base * 100
                rec[f'stock_k{k}'] = stock_ret
                rec[f'excess_k{k}'] = stock_ret - bench_ret
            rows.append(rec)
    return pd.DataFrame(rows)


def describe_path(df, label):
    print(f'\n--- KURSPFAD ab Ex-Tag-Close ({label}, n={len(df)}) ---')
    print(f"{'Tag':>5} {'Stock med%':>11} {'Excess med%':>12} {'Excess mean%':>13} "
          f"{'%>0':>6}")
    for k in range(0, MAX_K + 1):
        ex = df[f'excess_k{k}']
        st = df[f'stock_k{k}']
        print(f"Ex+{k:<3} {st.median():>10.3f} {ex.median():>12.3f} "
              f"{ex.mean():>13.3f} {(ex > 0).mean() * 100:>5.1f}")


def one_sided_p(t, p2):
    return p2 / 2 if t > 0 else 1 - p2 / 2


def strategy_grid(df):
    """Kauf Ex+kb-Close, Verkauf Ex+ks-Close. Excess vs Bench, Train/Test."""
    print('\n' + '=' * 86)
    print('STRATEGIE: Kauf in die Delle (Ex+kb), Verkauf Ex+ks  —  Excess vs ^GDAXI')
    print('=' * 86)
    train = df[df['ex_date'] <= TRAIN_END]
    test = df[df['ex_date'] >= TEST_START]
    combos = []
    for kb in (2, 3):
        for ks in range(kb + 2, kb + 6):  # 2..5 Tage Haltedauer
            if ks > MAX_K:
                continue
            combos.append((kb, ks))
    K = len(combos)
    # Bonferroni: einseitig, alpha=0.05/K -> t-Schwelle
    from scipy.stats import t as tdist
    print(f'(Bonferroni K={K} Kombos -> OOS-t-Schwelle einseitig: '
          f'{tdist.ppf(1 - 0.05 / K, 400):.2f})')
    print(f"\n{'kb':>3} {'ks':>3} {'hold':>5} | {'n_tr':>5} {'tr_mean%':>9} {'tr_t':>6} "
          f"| {'n_te':>5} {'te_mean%':>9} {'te_med%':>8} {'te_t':>6} {'te_p1':>8} {'%>0':>6}")
    rows = []
    for kb, ks in combos:
        tr = (train[f'excess_k{ks}'] - train[f'excess_k{kb}']).values
        te = (test[f'excess_k{ks}'] - test[f'excess_k{kb}']).values
        tr_t, tr_p2 = stats.ttest_1samp(tr, 0)
        te_t, te_p2 = stats.ttest_1samp(te, 0)
        te_p1 = one_sided_p(te_t, te_p2)
        print(f"{kb:>3} {ks:>3} {ks-kb:>5} | {len(tr):>5} {tr.mean():>9.3f} {tr_t:>6.2f} "
              f"| {len(te):>5} {te.mean():>9.3f} {np.median(te):>8.3f} {te_t:>6.2f} "
              f"{te_p1:>8.4f} {(te>0).mean()*100:>5.1f}")
        rows.append({'kb': kb, 'ks': ks, 'hold': ks - kb,
                     'tr_mean': tr.mean(), 'tr_t': tr_t,
                     'te_mean': te.mean(), 'te_t': te_t, 'te_p1': te_p1,
                     'te_winrate': (te > 0).mean()})
    return pd.DataFrame(rows), K


def main():
    print('=' * 86)
    print('EX-TAG-REBOUND-TEST — Delayed-Dip-Hypothese (Nutzer, 2026-06-23)')
    print('=' * 86)
    data, bench = load()
    df = build_paths(data, bench)
    print(f'[Build] {len(df)} Dividenden-Events (DAX40+MDAX50, ohne COVID)')
    df.to_csv('exday_rebound_paths.csv', index=False)

    describe_path(df, 'ALLE')
    describe_path(df[df['ex_date'] <= TRAIN_END], 'TRAIN <=2018')
    describe_path(df[df['ex_date'] >= TEST_START], 'TEST >=2019')

    grid, K = strategy_grid(df)

    print('\n' + '=' * 86)
    print('VERDICT')
    print('=' * 86)
    # Hypothese-Check Teil 1: ist Ex+2/Ex+3 ~tief (Excess-Median <= ~0)?
    d2 = df['excess_k2'].median()
    d3 = df['excess_k3'].median()
    print(f'Teil 1 "Ex+2/Ex+3 noch tief": Excess-Median Ex+2={d2:.3f}%, Ex+3={d3:.3f}%')
    # Teil 2: signifikanter OOS-Rebound aus der Delle?
    from scipy.stats import t as tdist
    thr = tdist.ppf(1 - 0.05 / K, 400)
    winners = grid[(grid['te_t'] > thr) & (grid['te_mean'] > 0)]
    if len(winners):
        print(f'Teil 2 "Rebound": {len(winners)} Kombo(s) OOS-signifikant '
              f'(t>{thr:.2f} & mean>0):')
        print(winners.to_string(index=False))
        print('-> GREEN-Kandidat, Forward-Test rechtfertigt sich.')
    else:
        best = grid.loc[grid['te_t'].idxmax()]
        print(f'Teil 2 "Rebound": KEINE Kombo OOS-signifikant (Bonferroni t>{thr:.2f}).')
        print(f'   Bestes OOS: kb={int(best.kb)} ks={int(best.ks)} '
              f'mean={best.te_mean:.3f}% t={best.te_t:.2f} -> RED/insignifikant.')


if __name__ == '__main__':
    main()
