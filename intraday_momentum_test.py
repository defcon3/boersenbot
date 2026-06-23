"""
MARKET INTRADAY MOMENTUM (SPY) — Test zur Pre-Reg
  preregs/intraday_momentum_spy_2026_06_23.md

Hypothese: ret1 (09:30->10:00) prognostiziert ret_last (15:30->16:00).
Strategie: 15:30 sign(ret1) long/short, 16:00 schliessen.
Daten: spy_30min_sip_2016_2026.pkl (Alpaca SIP, via fetch_spy_30min.py auf VPS).
"""
import pickle
import numpy as np
import pandas as pd

PKL = 'spy_30min_sip_2016_2026.pkl'
IS_END = pd.Timestamp('2021-12-31').date()
OOS_START = pd.Timestamp('2022-01-01').date()
COVID = (pd.Timestamp('2020-02-15').date(), pd.Timestamp('2020-04-30').date())
ANN = np.sqrt(252)


# ---------- Newey-West (HAC) fuer OLS y = X b + e ----------
def hac_ols(y, X, L=5):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        Xe = X * e[:, None]
        G = Xe[l:].T @ Xe[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    t = beta / se
    return beta, t, V


def hac_mean_t(x, L=5):
    """t-Stat des Mittelwerts mit HAC (Regression auf Konstante)."""
    x = np.asarray(x, float)
    X = np.ones((len(x), 1))
    b, t, _ = hac_ols(x, X, L)
    return float(b[0]), float(t[0])


def load():
    with open(PKL, 'rb') as f:
        bars = pickle.load(f)
    df = pd.DataFrame(bars)
    df['t'] = pd.to_datetime(df['t'], utc=True)
    df['et'] = df['t'].dt.tz_convert('America/New_York')
    df['date'] = df['et'].dt.date
    df['tod'] = df['et'].dt.strftime('%H:%M')
    return df


def build(df):
    """Pro Handelstag ret1, ret12, ret_last aus den 30-Min-Bars."""
    need = {'09:30', '15:00', '15:30'}
    rows = []
    for d, g in df.groupby('date'):
        g = g.set_index('tod')
        if not need.issubset(set(g.index)):
            continue  # halbe Tage / unvollstaendig -> raus
        b1, b12, bl = g.loc['09:30'], g.loc['15:00'], g.loc['15:30']
        # falls Duplikate -> erstes nehmen
        if isinstance(b1, pd.DataFrame):
            b1, b12, bl = b1.iloc[0], b12.iloc[0], bl.iloc[0]
        rows.append({
            'date': d,
            'ret1': b1['c'] / b1['o'] - 1,
            'ret12': b12['c'] / b12['o'] - 1,
            'ret_last': bl['c'] / bl['o'] - 1,
        })
    out = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    return out


def split(e):
    is_ = e[e['date'] <= IS_END]
    oos = e[e['date'] >= OOS_START]
    return is_, oos


def predictive(e, label):
    X = np.column_stack([np.ones(len(e)), e['ret1'].values])
    beta, t, _ = hac_ols(e['ret_last'].values, X)
    # R2
    yhat = X @ beta
    ss_res = np.sum((e['ret_last'].values - yhat) ** 2)
    ss_tot = np.sum((e['ret_last'].values - e['ret_last'].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f'  [{label}] ret_last = a + b*ret1 : b={beta[1]:+.4f} '
          f'HAC-t={t[1]:.2f}  R2={r2*100:.3f}%  n={len(e)}')
    return beta[1], t[1]


def strat_stats(e, cost_bps=0.0, signal='ret1'):
    sign = np.sign(e[signal].values)
    gross = sign * e['ret_last'].values
    net = gross - cost_bps / 1e4  # 1 Round-Trip/Tag
    mean, tt = hac_mean_t(net)
    sharpe = mean / net.std() * ANN if net.std() > 0 else float('nan')
    hit = (net > 0).mean() * 100
    ann_ret = mean * 252 * 100
    return {'mean_bps': mean * 1e4, 't': tt, 'sharpe': sharpe,
            'hit%': hit, 'ann%': ann_ret, 'n': len(e)}


def show(tag, s):
    print(f'  {tag:<26} mean={s["mean_bps"]:+.3f}bps  t={s["t"]:+.2f}  '
          f'Sharpe(ann)={s["sharpe"]:+.2f}  Hit={s["hit%"]:.1f}%  '
          f'AnnRet={s["ann%"]:+.2f}%  n={s["n"]}')


def main():
    print('=' * 86)
    print('MARKET INTRADAY MOMENTUM (SPY) — Pre-Reg-Test 2026-06-23')
    print('=' * 86)
    df = load()
    e = build(df)
    e_nc = e[~((e['date'] >= COVID[0]) & (e['date'] <= COVID[1]))]
    is_, oos = split(e)
    print(f'Events: {len(e)} Handelstage  {e.date.min()} .. {e.date.max()}'
          f'  | IS={len(is_)} OOS={len(oos)}')

    print('\n--- PRAEDIKTIVE REGRESSION (ret1 -> ret_last) ---')
    b_is, t_is = predictive(is_, 'IS 2016-2021')
    b_oos, t_oos = predictive(oos, 'OOS 2022-2026')
    predictive(e, 'GESAMT')

    print('\n--- STRATEGIE sign(ret1)*ret_last ---')
    print('IS 2016-2021:')
    s_is = strat_stats(is_)
    show('brutto', s_is)
    print('OOS 2022-2026:')
    s_oos = strat_stats(oos)
    show('brutto', s_oos)
    s_oos1 = strat_stats(oos, 1.0)
    s_oos2 = strat_stats(oos, 2.0)
    show('netto 1bp RT', s_oos1)
    show('netto 2bp RT', s_oos2)

    print('\n--- ALT-SIGNAL sign(ret12) & kombiniert (OOS, brutto) ---')
    show('sign(ret12)', strat_stats(oos, 0.0, 'ret12'))
    e2 = oos.copy()
    e2['combo'] = e2['ret1'] + e2['ret12']
    show('sign(ret1+ret12)', strat_stats(e2, 0.0, 'combo'))

    print('\n--- G5: PER-JAHR (Strategie brutto, sign(ret1)) ---')
    for yr, g in e.assign(y=[d.year for d in e['date']]).groupby('y'):
        s = strat_stats(g)
        print(f'  {yr}: mean={s["mean_bps"]:+.3f}bps t={s["t"]:+.2f} '
              f'Sharpe={s["sharpe"]:+.2f} Hit={s["hit%"]:.1f}% n={s["n"]}')
    print('  --- ohne COVID-Fenster (IS-Regression) ---')
    predictive(e_nc[e_nc['date'] <= IS_END], 'IS o.COVID')

    print('\n' + '=' * 86)
    print('GATE-CHECK')
    print('=' * 86)
    g1 = (t_is > 2.0) and (b_is > 0) and (s_is['mean_bps'] > 0)
    g2 = (s_oos['mean_bps'] > 0) and (s_oos['t'] > 1.5)
    g3 = (s_oos2['mean_bps'] > 0) and (s_oos2['t'] > 1.0)
    g4 = all(strat_stats(g)['n'] >= 200
             for _, g in e.assign(y=[d.year for d in e['date']]).groupby('y')
             if g.iloc[0]['date'].year < 2026)  # 2026 unvollstaendig
    print(f'G1 (IS t>2 & b>0 & mean>0):        {"PASS" if g1 else "FAIL"} '
          f'(t={t_is:.2f}, b={b_is:+.4f})')
    print(f'G2 (OOS mean>0 & t>1.5):           {"PASS" if g2 else "FAIL"} '
          f'(mean={s_oos["mean_bps"]:+.3f}bps, t={s_oos["t"]:.2f})')
    print(f'G3 (OOS netto 2bp mean>0 & t>1):   {"PASS" if g3 else "FAIL"} '
          f'(mean={s_oos2["mean_bps"]:+.3f}bps, t={s_oos2["t"]:.2f})')
    print(f'G4 (>=200 Signale/Jahr):           {"PASS" if g4 else "FAIL"}')
    allp = g1 and g2 and g3 and g4
    print('-' * 86)
    if allp:
        print('VERDICT: GREEN (vorbehaltlich G5) — Forward-Test rechtfertigt sich.')
    elif g1 and not g2:
        print('VERDICT: RED — IS-Edge vorhanden, aber OOS zusammengebrochen (Decay).')
    elif g1 and g2 and not g3:
        print('VERDICT: YELLOW — Edge da, aber von Kosten aufgefressen.')
    else:
        print('VERDICT: RED — Hypothese nicht bestaetigt.')


if __name__ == '__main__':
    main()
